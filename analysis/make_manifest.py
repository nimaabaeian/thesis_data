#!/usr/bin/env python3
"""Build (or verify) the machine-readable data manifest.

The public repository does not, and cannot, ship the raw data: the SQLite databases contain
participant faces, names and chat text, and `analysis/private/` holds the identity and role maps.
So "reproducible" here has a precise and limited meaning, and this manifest is what makes it
checkable rather than merely asserted:

  * `make manifest`        writes data_manifest.json from the data actually present
  * `make verify-manifest` re-hashes the data and fails if anything drifted

Full independent reproduction requires controlled access to the excluded data. What a third party
CAN verify without it: that the code is deterministic, that the constants match the pinned
controller source, and that the inputs they were given hash to the values recorded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
MANIFEST = ANALYSIS / "data_manifest.json"

DB_KINDS = ["executive_control", "salience_network", "chat_bot", "vision"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blob := f.read(chunk):
            h.update(blob)
    return h.hexdigest()


def db_profile(path: Path) -> dict:
    """Row counts and schema fingerprint, so a silently-mutated DB cannot pass as identical."""
    prof: dict = {"tables": {}}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cur = con.cursor()
        names = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        schema_blob = []
        for t in names:
            try:
                n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.Error:
                n = None
            cols = [f"{r[1]}:{r[2]}" for r in cur.execute(f'PRAGMA table_info("{t}")')]
            prof["tables"][t] = {"rows": n, "columns": len(cols)}
            schema_blob.append(f"{t}({','.join(cols)})")
        prof["schema_sha256"] = hashlib.sha256(
            "\n".join(sorted(schema_blob)).encode()).hexdigest()[:16]
        prof["total_rows"] = sum(v["rows"] or 0 for v in prof["tables"].values())
        con.close()
    except sqlite3.Error as e:
        prof["error"] = str(e)
    return prof


def build() -> dict:
    data_root = REPO / "data"
    sessions = sorted(p.name for p in data_root.iterdir()
                      if p.is_dir() and (p / "data_collection").is_dir()) if data_root.exists() else []
    man: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Raw data are NOT distributed with this repository: the databases contain "
                 "participant faces, names and chat text. This manifest records exactly what the "
                 "analysis consumed, so a holder of the data can confirm they have the same bytes. "
                 "It does not make the analysis reproducible without that data."),
        "collection_dates": sessions,
        "expected_files": [],
        "private_files": [],
        "schema_version": 1,
    }
    for sess in sessions:
        for kind in DB_KINDS:
            p = data_root / sess / "data_collection" / f"{kind}.db"
            entry = {"path": str(p.relative_to(REPO)), "session": sess, "kind": kind}
            if p.exists():
                entry.update({
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                    **db_profile(p),
                })
            else:
                entry["status"] = "MISSING"
            man["expected_files"].append(entry)
        mem = data_root / sess / "memory"
        if mem.is_dir():
            for f in sorted(mem.glob("*.json")):
                man["expected_files"].append({
                    "path": str(f.relative_to(REPO)), "session": sess, "kind": "memory",
                    "sha256": sha256(f), "bytes": f.stat().st_size,
                })
    # Private maps: hashed, never contents. Their presence changes the analysis (roles, identity
    # canonicalisation), so their hashes belong in the manifest even though they are git-ignored.
    priv = ANALYSIS / "private"
    if priv.is_dir():
        for f in sorted(priv.glob("*.json")):
            man["private_files"].append({
                "path": str(f.relative_to(REPO)), "sha256": sha256(f), "bytes": f.stat().st_size,
                "note": "git-ignored; contents never published",
            })
    return man


def verify() -> int:
    if not MANIFEST.exists():
        print("manifest=MISSING (run `make manifest` first)")
        return 1
    old = json.loads(MANIFEST.read_text())
    new = build()
    old_map = {e["path"]: e for e in old["expected_files"] + old.get("private_files", [])}
    new_map = {e["path"]: e for e in new["expected_files"] + new.get("private_files", [])}
    problems = []
    for path, e in old_map.items():
        if path not in new_map:
            problems.append(f"  MISSING NOW: {path}")
            continue
        n = new_map[path]
        if e.get("sha256") and n.get("sha256") and e["sha256"] != n["sha256"]:
            problems.append(f"  CHANGED: {path}\n    was {e['sha256'][:16]}  now {n['sha256'][:16]}")
        if e.get("total_rows") is not None and n.get("total_rows") != e.get("total_rows"):
            problems.append(f"  ROW COUNT CHANGED: {path} "
                            f"{e.get('total_rows')} -> {n.get('total_rows')}")
    for path in new_map:
        if path not in old_map:
            problems.append(f"  NEW FILE (not in manifest): {path}")
    if problems:
        print("manifest=FAIL — inputs differ from the manifest:")
        print("\n".join(problems))
        return 1
    print(f"manifest=PASS ({len(old_map)} files, hashes and row counts match)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check the data against the manifest")
    args = ap.parse_args()
    if args.verify:
        return verify()
    man = build()
    MANIFEST.write_text(json.dumps(man, indent=2))
    n_db = sum(1 for e in man["expected_files"] if e.get("kind") in DB_KINDS)
    n_rows = sum(e.get("total_rows", 0) or 0 for e in man["expected_files"])
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    print(f"  {len(man['collection_dates'])} collection dates: {', '.join(man['collection_dates'])}")
    print(f"  {n_db} databases, {n_rows:,} total rows")
    print(f"  {len(man['private_files'])} private files hashed (contents never published)")
    missing = [e['path'] for e in man['expected_files'] if e.get('status') == 'MISSING']
    if missing:
        print(f"  WARNING: {len(missing)} expected file(s) missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
