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


# Raw Telegram user IDs are 9-10 digit integers. They are stable, real-world personal
# identifiers — as identifying as a name, and trivially reversible by anyone who has messaged
# that account — but a name-based scan cannot see them. A per-subscriber table exported straight
# from `chat_id` would have published them. Subscribers are pseudonymised to S## codes in the
# notebook; this catches any regression of that.
#
# The match must be a WHOLE FIELD VALUE, never a substring. Hex run_ids and 12-digit float
# timestamps are full of incidental 9-digit runs, and matching those would drown the real signal
# in false positives (it did, on the first attempt).
TELEGRAM_ID = re.compile(r"[1-9]\d{8,9}(?:\.0)?")

# Epoch timestamps are also 10 digits and are legitimate analysis data. Anything a plausible
# unix time (2001-2033) is a timestamp, not a user ID.
_EPOCH_LO, _EPOCH_HI = 1_000_000_000, 2_000_000_000


def scan_id_values(values) -> set[str]:
    """Flag whole values that ARE a raw numeric user ID."""
    hits = set()
    for v in values:
        s = str(v).strip()
        if not TELEGRAM_ID.fullmatch(s):
            continue
        n = int(s.removesuffix(".0"))
        if _EPOCH_LO <= n <= _EPOCH_HI:
            continue          # an epoch timestamp, not a user ID
        hits.add(s)
    return hits


def csv_fields(text: str):
    """Every comma/newline-delimited field in a CSV, as whole values."""
    for line in text.splitlines():
        for field in line.split(","):
            yield field.strip()


def find_leaks(nb_path: Path) -> list[tuple[str, str]]:
    """Scan every publishable surface for BOTH real names and raw numeric user IDs.

    The name scan alone is not sufficient: `chat_id` is a raw Telegram user ID, is not a name,
    and was not covered by IDENTITY_COLS. It reached an exported per-subscriber table. Both
    classes of identifier are checked here.
    """
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

    def scan_names(where: str, text: str) -> None:
        for hit in scan_text(text, pattern):
            leaks.append((where, f"name:{hit}"))

    def scan_ids(where: str, values) -> None:
        for hit in scan_id_values(values):
            leaks.append((where, f"raw-user-id:{hit}"))

    # 1. The notebook itself: cell sources AND outputs. Names only — the JSON is full of base64
    #    image payloads and coordinate arrays, and there is no notion of a "field" in it.
    scan_names(str(nb_path), nb_path.read_text())

    # 2. Published side files: outputs/ (text + parquet) and figures/ (svg text).
    for f in sorted((analysis_dir / "outputs").glob("*")):
        if f.suffix in (".csv", ".md", ".txt", ".json"):
            text = f.read_text(errors="replace")
            scan_names(str(f), text)
            if f.suffix == ".csv":
                scan_ids(str(f), csv_fields(text))
        elif f.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(f)
            for c in df.columns:
                vals = df[c].dropna().unique()
                if df[c].dtype == object:
                    scan_names(f"{f}:{c}", "\n".join(map(str, vals)))
                # Float columns are durations/timestamps, never identities; only integer-like or
                # string columns can carry a raw user ID.
                if not pd.api.types.is_float_dtype(df[c]):
                    scan_ids(f"{f}:{c}", vals)
    for f in sorted((analysis_dir / "figures").glob("*.svg")):
        scan_names(str(f), f.read_text(errors="replace"))
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
