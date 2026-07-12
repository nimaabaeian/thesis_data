#!/usr/bin/env python3
"""Verify the constants the analysis relies on against the pinned controller source.

The notebook hard-codes controller constants (thresholds, meal sizes, energy costs, the affinity
EMA parameters). Those constants are load-bearing: B2's thresholds, B9's EMA re-threading, and
V1/V5's energy checks are all wrong if any of them drifted. Re-typing them from memory is exactly
how that happens, so this script checks them programmatically.

It reports one of three states, and never confuses them:

  PASS      every constant in controller_source.json was found in the checked-out source
            and matches the value the notebook uses
  MISMATCH  a constant was found but the value differs -> the analysis is invalid, exit 1
  SKIPPED   the controller source is not available locally, or the commit is UNPINNED
            -> exit 0, but the reproducibility report records the constants as UNVERIFIED

Usage:
    python analysis/check_constants.py [--source /path/to/alwaysOn-embodiedBehaviour]
    python analysis/check_constants.py --clone      # clone the pinned commit into a temp dir
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PIN = json.loads((ANALYSIS_DIR / "controller_source.json").read_text())


def numeric_literals(path: Path) -> dict[str, list[float]]:
    """Every numeric literal assigned to a NAME or stored in a dict, keyed by the name.

    Deliberately permissive: we are looking for `HUNGRY_THRESHOLD = 60.0` and
    `SS_THRESHOLDS = {"ss1": 0.80, ...}` alike, so both assignment forms are harvested and the
    caller matches on the value.
    """
    out: dict[str, list[float]] = {}
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, OSError):
        return out

    def numbers_in(node) -> list[float]:
        vals: list[float] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) \
                    and not isinstance(sub.value, bool):
                vals.append(float(sub.value))
            elif isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.USub) \
                    and isinstance(sub.operand, ast.Constant) \
                    and isinstance(sub.operand.value, (int, float)):
                vals.append(-float(sub.operand.value))
        return vals

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                name = None
                if isinstance(t, ast.Name):
                    name = t.id
                elif isinstance(t, ast.Attribute):
                    name = t.attr
                if name and node.value is not None:
                    out.setdefault(name, []).extend(numbers_in(node.value))
    return out


def check(source_root: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    for mod, spec in PIN["modules"].items():
        path = source_root / spec["path"]
        if not path.exists():
            # try a flat layout too
            alt = source_root / mod
            path = alt if alt.exists() else path
        if not path.exists():
            for name, c in spec["constants"].items():
                rows.append(dict(module=mod, constant=name, expected=c["value"],
                                 found=None, status="MODULE_NOT_FOUND"))
            continue
        lits = numeric_literals(path)
        text = path.read_text(errors="replace")
        for name, c in spec["constants"].items():
            sym = c.get("symbol", name)
            expected = float(c["value"])
            found = lits.get(sym, [])
            if any(abs(v - expected) < 1e-9 for v in found):
                status = "PASS"
                got = expected
            elif found:
                status = "MISMATCH"
                got = found[0] if len(found) == 1 else found
            else:
                # Fall back to a textual search: the constant may live in a config dict or a
                # default argument rather than a bare assignment.
                pat = re.compile(rf"{re.escape(sym)}\s*[=:]\s*(-?\d+(?:\.\d+)?)")
                m = pat.search(text)
                if m and abs(float(m.group(1)) - expected) < 1e-9:
                    status, got = "PASS", expected
                elif m:
                    status, got = "MISMATCH", float(m.group(1))
                else:
                    status, got = "NOT_FOUND", None
            rows.append(dict(module=mod, constant=name, expected=expected,
                             found=got, status=status))
    if any(r["status"] == "MISMATCH" for r in rows):
        overall = "MISMATCH"
    elif all(r["status"] == "PASS" for r in rows):
        overall = "PASS"
    else:
        overall = "INCOMPLETE"
    return rows, overall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None,
                    help="path to a checkout of the controller repo")
    ap.add_argument("--clone", action="store_true",
                    help="clone the pinned commit into a temp dir and check that")
    ap.add_argument("--json", type=Path, default=None, help="write the result table here")
    args = ap.parse_args()

    commit = str(PIN.get("commit", "UNPINNED"))
    result = {"repo": PIN["repo"], "commit": commit, "status": None, "rows": []}

    if commit == "UNPINNED" and not args.source:
        print("constants_check=SKIPPED")
        print("  controller_source.json has commit=UNPINNED.")
        print("  Set it to the 40-char SHA the deployment ran, or pass --source <path>.")
        print("  The constants in the notebook are therefore UNVERIFIED against source.")
        result["status"] = "SKIPPED_UNPINNED"
        if args.json:
            args.json.write_text(json.dumps(result, indent=2))
        return 0

    root: Path | None = args.source
    tmp = None
    if args.clone:
        if commit == "UNPINNED":
            print("constants_check=SKIPPED (cannot clone: commit is UNPINNED)")
            result["status"] = "SKIPPED_UNPINNED"
            if args.json:
                args.json.write_text(json.dumps(result, indent=2))
            return 0
        tmp = tempfile.mkdtemp(prefix="controller-src-")
        try:
            subprocess.run(["git", "clone", "--quiet", PIN["repo"], tmp], check=True)
            subprocess.run(["git", "-C", tmp, "checkout", "--quiet", commit], check=True)
            root = Path(tmp)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"constants_check=SKIPPED (clone failed: {e})")
            result["status"] = "SKIPPED_CLONE_FAILED"
            if args.json:
                args.json.write_text(json.dumps(result, indent=2))
            return 0

    if root is None or not root.exists():
        print("constants_check=SKIPPED (no controller source available)")
        print("  Pass --source <path> or --clone. Constants remain UNVERIFIED against source.")
        result["status"] = "SKIPPED_NO_SOURCE"
        if args.json:
            args.json.write_text(json.dumps(result, indent=2))
        return 0

    rows, overall = check(root)
    result["rows"] = rows
    result["status"] = overall
    width = max(len(r["constant"]) for r in rows)
    for r in rows:
        flag = {"PASS": "ok", "MISMATCH": "MISMATCH", "NOT_FOUND": "not found",
                "MODULE_NOT_FOUND": "module missing"}[r["status"]]
        print(f"  [{flag:>14s}] {r['module']:<22s} {r['constant']:<{width}s} "
              f"expected={r['expected']!s:<8s} found={r['found']}")
    print(f"\nconstants_check={overall}")
    if overall == "MISMATCH":
        print("  A constant in the notebook does not match the pinned source. The analysis")
        print("  depends on these values (B2 thresholds, B9 EMA, V1/V5 energy checks) and is")
        print("  INVALID until they agree.")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 1 if overall == "MISMATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
