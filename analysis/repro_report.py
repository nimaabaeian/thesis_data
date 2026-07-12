#!/usr/bin/env python3
"""Emit outputs/reproducibility_report.md: what went in, what ran, what came out.

Records input hashes, software versions, notebook execution status, constants-check status, and
output hashes. The point is that a reader can tell, without rerunning anything, whether the numbers
in the report came from the data the manifest describes, and whether the constants were ever
actually checked against the controller source or merely asserted.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
OUT = ANALYSIS / "outputs"
FIGS = ANALYSIS / "figures"
NB = ANALYSIS / "orexigenic_analysis.ipynb"

PACKAGES = ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "lifelines",
            "matplotlib", "pyarrow", "nbformat", "nbconvert"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blob := f.read(1 << 20):
            h.update(blob)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "(unavailable)"


def nb_status() -> tuple[str, int, int, int]:
    if not NB.exists():
        return "NOT BUILT", 0, 0, 0
    nb = json.loads(NB.read_text())
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    unexec = sum(c.get("execution_count") is None for c in code)
    errors = sum(1 for c in code for o in c.get("outputs", [])
                 if o.get("output_type") == "error")
    if not code:
        return "EMPTY", 0, 0, 0
    if unexec == 0 and errors == 0:
        return "EXECUTED CLEAN", len(code), unexec, errors
    if errors:
        return "EXECUTED WITH ERRORS", len(code), unexec, errors
    return "PARTIALLY EXECUTED", len(code), unexec, errors


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status, n_cells, unexec, errors = nb_status()

    # Constants check — run it and capture its real state rather than assuming it passed.
    cc_json = OUT / "constants_check.json"
    try:
        subprocess.run([sys.executable, str(ANALYSIS / "check_constants.py"),
                        "--json", str(cc_json)], capture_output=True, text=True, check=False)
        cc = json.loads(cc_json.read_text()) if cc_json.exists() else {"status": "NOT RUN"}
    except Exception as e:
        cc = {"status": f"ERROR: {e}"}

    man = json.loads((ANALYSIS / "data_manifest.json").read_text()) \
        if (ANALYSIS / "data_manifest.json").exists() else None

    L = ["# Reproducibility report", "",
         f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_", "",
         "## What can and cannot be reproduced", "",
         "**The raw data are not in this repository and cannot be.** The SQLite databases hold",
         "participant faces, names and chat transcripts, and `analysis/private/` holds the identity",
         "canonicalisation and role-assignment maps. Both are git-ignored.",
         "",
         "Consequently:",
         "",
         "- **Full independent reproduction requires controlled access to the excluded data and the",
         "  role mappings.** There is no way around this and no claim to the contrary is made here.",
         "- **Without that data, a third party can still verify**: that the build is deterministic",
         "  (fixed seed, pinned dependencies), that the analysis constants match the pinned controller",
         "  source (`check_constants.py`), and — if they hold the data — that their copy hashes to the",
         "  values in `data_manifest.json`.",
         ""]

    L += ["## Git state", "",
          f"- commit: `{git('rev-parse', 'HEAD')}`",
          f"- branch: `{git('rev-parse', '--abbrev-ref', 'HEAD')}`",
          f"- dirty: `{'yes' if git('status', '--porcelain') else 'no'}`", ""]

    L += ["## Software", "",
          f"- python: `{platform.python_version()}` ({platform.system()} {platform.machine()})", ""]
    L += ["| package | version |", "|---|---|"]
    for p in PACKAGES:
        try:
            from importlib.metadata import version
            L.append(f"| {p} | {version(p)} |")
        except Exception:
            L.append(f"| {p} | (not installed) |")
    L.append("")
    lock = ANALYSIS / "requirements.lock"
    L += [f"- environment lock: `{'analysis/requirements.lock' if lock.exists() else 'MISSING'}`",
          f"- pinned deps: `analysis/requirements.txt`", ""]

    L += ["## Controller constants", ""]
    st = cc.get("status", "UNKNOWN")
    if st == "PASS":
        L += [f"- **PASS** — every constant matches `{cc.get('repo')}` @ `{cc.get('commit','')[:12]}`.", ""]
    elif str(st).startswith("SKIPPED"):
        L += [f"- **UNVERIFIED** (`{st}`). The controller source was not available, or",
              "  `controller_source.json` still has `commit: UNPINNED`. The constants used by the",
              "  analysis (thresholds, meal sizes, energy costs, affinity EMA) are therefore taken on",
              "  trust. Pin the commit and re-run `make check-constants` to close this.", ""]
    elif st == "MISMATCH":
        L += ["- **MISMATCH — THE ANALYSIS IS INVALID.** A constant does not match the pinned source.", ""]
    else:
        L += [f"- status: `{st}`", ""]

    L += ["## Inputs", ""]
    if man:
        files = [e for e in man["expected_files"] if e.get("sha256")]
        rows = sum(e.get("total_rows", 0) or 0 for e in man["expected_files"])
        L += [f"- manifest: `analysis/data_manifest.json` ({len(files)} files, {rows:,} rows)",
              f"- collection dates: {', '.join(man['collection_dates'])}",
              f"- private maps hashed (contents never published): {len(man.get('private_files', []))}",
              "", "| file | rows | sha256 (first 16) |", "|---|---:|---|"]
        for e in man["expected_files"]:
            if e.get("kind") in ("executive_control", "salience_network", "chat_bot", "vision"):
                L.append(f"| `{e['path']}` | {e.get('total_rows', '?')} | `{e.get('sha256','')[:16]}` |")
        L.append("")
    else:
        L += ["- **manifest MISSING** — run `make manifest`.", ""]

    L += ["## Notebook execution", "",
          f"- status: **{status}**",
          f"- code cells: {n_cells}, unexecuted: {unexec}, errored: {errors}", ""]

    L += ["## Outputs", "", "| file | bytes | sha256 (first 16) |", "|---|---:|---|"]
    for f in sorted(OUT.glob("*")):
        if f.name == "reproducibility_report.md" or f.is_dir():
            continue
        L.append(f"| `outputs/{f.name}` | {f.stat().st_size} | `{sha256(f)[:16]}` |")
    for f in sorted(FIGS.glob("*.png")):
        L.append(f"| `figures/{f.name}` | {f.stat().st_size} | `{sha256(f)[:16]}` |")
    L.append("")

    L += ["## Determinism", "",
          "- global seed `SEED=42` set once in the notebook setup cell and used by every bootstrap,",
          "  permutation and model fit.",
          "- databases opened read-only (`mode=ro`).",
          "- `make execute` regenerates every artifact under `analysis/outputs/` and",
          "  `analysis/figures/` from a clean state (`make clean-outputs` first).", ""]

    (OUT / "reproducibility_report.md").write_text("\n".join(L))
    print(f"wrote outputs/reproducibility_report.md  (notebook: {status}, constants: {st})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
