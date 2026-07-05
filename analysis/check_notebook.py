#!/usr/bin/env python3
"""Validate that the generated notebook is fully executed, error-free, and leaks no
real participant identity into publishable surfaces (notebook JSON, outputs/, figures/).
Real names are read from the private maps in analysis/private/ (git-ignored)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def real_names(private_dir: Path) -> set[str]:
    """All real identity strings from the private canon + pseudonym maps."""
    names: set[str] = set()
    canon = private_dir / "identity_canon.json"
    if canon.exists():
        for k, v in json.loads(canon.read_text()).items():
            names.update((k, v))
    pseud = private_dir / "pseudonym_map.json"
    if pseud.exists():
        names.update(json.loads(pseud.read_text()).keys())
    names.discard("unknown")
    return {n for n in names if n.strip()}


def scan_text(text: str, pattern: re.Pattern) -> set[str]:
    return {m.group(0).lower() for m in pattern.finditer(text)}


def find_leaks(nb_path: Path) -> list[tuple[str, str]]:
    analysis_dir = nb_path.resolve().parent
    names = real_names(analysis_dir / "private")
    if not names:
        print("leak_check=SKIPPED (no private identity maps found)")
        return []
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    leaks: list[tuple[str, str]] = []
    # 1. The notebook itself: cell sources AND outputs.
    for hit in scan_text(nb_path.read_text(), pattern):
        leaks.append((str(nb_path), hit))
    # 2. Published side files: outputs/ (text + parquet) and figures/ (svg text).
    for f in sorted((analysis_dir / "outputs").glob("*")):
        if f.suffix in (".csv", ".md", ".txt", ".json"):
            for hit in scan_text(f.read_text(errors="replace"), pattern):
                leaks.append((str(f), hit))
        elif f.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(f)
            for c in df.columns:
                if df[c].dtype == object:
                    blob = "\n".join(df[c].dropna().astype(str).unique())
                    for hit in scan_text(blob, pattern):
                        leaks.append((f"{f}:{c}", hit))
    for f in sorted((analysis_dir / "figures").glob("*.svg")):
        for hit in scan_text(f.read_text(errors="replace"), pattern):
            leaks.append((str(f), hit))
    return leaks


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "analysis/orexigenic_analysis.ipynb")
    nb = json.loads(path.read_text())
    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    errors = []
    for i, cell in enumerate(code_cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append((i, output.get("ename"), output.get("evalue")))
    unexecuted = sum(cell.get("execution_count") is None for cell in code_cells)
    print(f"code_cells={len(code_cells)} unexecuted={unexecuted} errors={errors}")

    leaks = find_leaks(path)
    if leaks:
        print(f"leak_check=FAIL ({len(leaks)} real-name leak(s) in publishable files):")
        for where, name in leaks:
            print(f"  {where}: '{name}'")
    else:
        print("leak_check=PASS (no real names in notebook, outputs/, or figures/)")
    return 1 if unexecuted or errors or leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
