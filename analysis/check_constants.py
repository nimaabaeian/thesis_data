#!/usr/bin/env python3
"""Verify every analysis constant against the pinned controller source. FAILS the build.

The notebook hard-codes controller constants (thresholds, meal sizes, energy costs, the
affinity EMA). They are load-bearing: B2's thresholds, B9's EMA re-threading and V1/V5's
energy checks are all wrong if any of them drifted. Re-typing them from memory is exactly how
that happens.

This checker is **key-aware**. An earlier version searched for the expected *number* anywhere
under the right symbol, so `SS_THRESHOLDS = {"ss1": 0.80, ...}` would "verify" the value 0.80
without ever checking it belonged to `ss1` — the dict could have been permuted and the check
would still pass. Dictionaries are now parsed and compared key by key.

It also verifies EVERY commit the deployment ran (the controller changed mid-study) and asserts
those commits agree with each other, so the analysis rests on a constant that was stable for the
whole deployment rather than on one snapshot of it.

Exit codes:
    0  PASS      every constant found and matching, at every deployment commit
    1  FAIL      mismatch, missing constant, unpinned, or source unavailable

There is deliberately no "skip and pass" path. Unverified constants are a build failure.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PIN = json.loads((ANALYSIS_DIR / "controller_source.json").read_text())
TOL = 1e-9


# ---------------------------------------------------------------------------------------
# Parsing: resolve a symbol to a scalar or a {key: value} mapping, keys included.
# ---------------------------------------------------------------------------------------
def _const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    return None


def symbol_table(path: Path) -> dict[str, object]:
    """Map every assigned name -> scalar float, or dict[str, float] for dict literals.

    Handles module-level, class-level and `self.X = ...` assignments alike, because the
    controller uses all three.
    """
    out: dict[str, object] = {}
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, OSError):
        return out

    def record(name: str, value_node):
        if isinstance(value_node, ast.Dict):
            d: dict[str, float] = {}
            for k, v in zip(value_node.keys, value_node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    val = _const(v)
                    if val is not None:
                        d[k.value] = val
            if d:
                out.setdefault(name, d)
            return
        val = _const(value_node)
        if val is not None:
            out.setdefault(name, val)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    record(t.id, node.value)
                elif isinstance(t, ast.Attribute):
                    record(t.attr, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                record(node.target.id, node.value)
            elif isinstance(node.target, ast.Attribute):
                record(node.target.attr, node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The drive's core constants live as KEYWORD DEFAULTS in HungerModel.__init__:
            #     def __init__(self, drain_hours: float = 4.0, hungry_threshold: float = 60.0, ...)
            # A walker that only reads assignments cannot see them, and would report the
            # thresholds this entire analysis rests on as NOT_FOUND.
            a = node.args
            pos = list(a.posonlyargs) + list(a.args)
            for arg, default in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                record(arg.arg, default)
            for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                if default is not None:
                    record(arg.arg, default)
    return out


def check_commit(root: Path) -> list[dict]:
    rows: list[dict] = []
    for mod, spec in PIN["modules"].items():
        path = root / spec["path"]
        if not path.exists():
            for name in spec["constants"]:
                rows.append(dict(module=mod, constant=name, key=None, expected=None,
                                 found=None, status="MODULE_NOT_FOUND"))
            continue
        table = symbol_table(path)
        for name, c in spec["constants"].items():
            sym = c["symbol"]
            found = table.get(sym)
            if c.get("kind") == "dict":
                if not isinstance(found, dict):
                    for k, want in c["keys"].items():
                        rows.append(dict(module=mod, constant=name, key=k, expected=want,
                                         found=None, status="NOT_FOUND"))
                    continue
                for k, want in c["keys"].items():          # KEY-AWARE: each key checked by name
                    got = found.get(k)
                    if got is None:
                        st = "KEY_MISSING"
                    elif abs(got - float(want)) < TOL:
                        st = "PASS"
                    else:
                        st = "MISMATCH"
                    rows.append(dict(module=mod, constant=name, key=k, expected=float(want),
                                     found=got, status=st))
            else:
                want = float(c["value"])
                if found is None:
                    st, got = "NOT_FOUND", None
                elif isinstance(found, dict):
                    st, got = "TYPE_MISMATCH", found
                elif abs(found - want) < TOL:
                    st, got = "PASS", found
                else:
                    st, got = "MISMATCH", found
                rows.append(dict(module=mod, constant=name, key=None, expected=want,
                                 found=got, status=st))
    return rows


def clone(commit: str) -> Path | None:
    tmp = tempfile.mkdtemp(prefix=f"controller-{commit[:8]}-")
    try:
        subprocess.run(["git", "clone", "--quiet", PIN["repo"], tmp], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", tmp, "checkout", "--quiet", commit], check=True,
                       capture_output=True)
        return Path(tmp)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None,
                    help="local checkout of the controller repo (any commit; will be checked out)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    commits = list(PIN.get("deployment_commits") or {PIN.get("commit"): "pinned"})
    result = {"repo": PIN["repo"], "commits": commits, "status": None, "per_commit": {}}

    if not commits or any(c in (None, "", "UNPINNED") for c in commits):
        print("constants_check=FAIL")
        print("  controller_source.json is UNPINNED. The constants this analysis depends on")
        print("  (thresholds, meal sizes, energy costs, the affinity EMA) are therefore")
        print("  unverified, and the analysis cannot be trusted. Pin the deployed commit(s).")
        result["status"] = "FAIL_UNPINNED"
        if args.json:
            args.json.write_text(json.dumps(result, indent=2))
        return 1

    roots: dict[str, Path] = {}
    for c in commits:
        if args.source:
            # Check the requested commit out of the provided checkout, in a detached worktree.
            tmp = tempfile.mkdtemp(prefix=f"wt-{c[:8]}-")
            try:
                subprocess.run(["git", "-C", str(args.source), "worktree", "add", "--detach",
                                "--quiet", tmp, c], check=True, capture_output=True)
                roots[c] = Path(tmp)
            except subprocess.CalledProcessError as e:
                print(f"constants_check=FAIL (cannot check out {c[:12]} from {args.source}: "
                      f"{e.stderr.decode(errors='replace').strip()})")
                result["status"] = "FAIL_CHECKOUT"
                if args.json:
                    args.json.write_text(json.dumps(result, indent=2))
                return 1
        else:
            r = clone(c)
            if r is None:
                print("constants_check=FAIL")
                print(f"  Could not obtain {PIN['repo']} @ {c[:12]}.")
                print("  The controller source is private; pass --source <path-to-checkout>.")
                print("  Constants remain UNVERIFIED, which is a build failure, not a warning.")
                result["status"] = "FAIL_SOURCE_UNAVAILABLE"
                if args.json:
                    args.json.write_text(json.dumps(result, indent=2))
                return 1
            roots[c] = r

    overall = "PASS"
    tables: dict[str, list[dict]] = {}
    for c, root in roots.items():
        rows = check_commit(root)
        tables[c] = rows
        result["per_commit"][c] = rows
        bad = [r for r in rows if r["status"] != "PASS"]
        print(f"--- {c[:12]}  ({PIN.get('deployment_commits', {}).get(c, '')})")
        if bad:
            overall = "FAIL"
            for r in bad:
                key = f"[{r['key']}]" if r["key"] else ""
                print(f"    {r['status']:16s} {r['module']}::{r['constant']}{key} "
                      f"expected={r['expected']} found={r['found']}")
        else:
            print(f"    all {len(rows)} constants match")

    # The deployment spanned more than one commit: they must agree with each other, or the
    # analysis is using one set of constants to interpret data generated under another.
    if len(tables) > 1:
        ref = commits[0]
        for c in commits[1:]:
            a = {(r["module"], r["constant"], r["key"]): r["found"] for r in tables[ref]}
            b = {(r["module"], r["constant"], r["key"]): r["found"] for r in tables[c]}
            drift = [k for k in a if a[k] != b.get(k)]
            if drift:
                overall = "FAIL"
                print(f"\n  CONSTANT DRIFT between {ref[:8]} and {c[:8]}:")
                for k in drift:
                    print(f"    {k}: {a[k]} -> {b.get(k)}")
            else:
                print(f"\n  No constant drift between {ref[:8]} and {c[:8]} — the values were "
                      f"stable across the whole deployment.")

    result["status"] = overall
    print(f"\nconstants_check={overall}")
    if overall != "PASS":
        print("  The analysis depends on these values. It is INVALID until they agree.")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
