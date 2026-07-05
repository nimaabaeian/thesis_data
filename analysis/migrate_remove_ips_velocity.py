#!/usr/bin/env python3
"""Remove velocity fields from copied salience-network SQLite databases.

The migration is intentionally conservative:
- checkpoint WAL data into the main DB before copying;
- copy each DB and sidecar file into an ignored data_backup_* directory;
- drop only face_ips_events.vel_score and face_ips_events.weight_vel;
- recreate v_face_ips_timeline without velocity columns;
- verify every migrated DB no longer exposes velocity fields.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


VIEW_SQL = """
CREATE VIEW v_face_ips_timeline AS
SELECT
    timestamp_utc,
    timestamp_local,
    timezone,
    timestamp_epoch,
    monotonic_sec,
    run_elapsed_sec,
    day_rome,
    run_id,
    is_test_run,
    valid_for_analysis,
    track_id,
    face_id,
    person_id,
    social_state,
    CAST(is_known AS INTEGER)         AS is_known,
    CAST(eligible AS INTEGER)         AS eligible,
    CAST(is_active_target AS INTEGER) AS is_active_target,
    bbox_area,
    ips,
    ips_before_habituation,
    CAST(habituation_applied AS INTEGER) AS habituation_applied,
    habituation_multiplier,
    habituation_elapsed_sec,
    habituation_ips_delta,
    stimulus_type,
    context_label,
    prox_score,
    cent_score,
    gaze_score,
    weight_prox,
    weight_cent,
    weight_gaze
FROM face_ips_events
"""


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def copy_backup(db_path: Path, backup_root: Path) -> None:
    rel_dir = db_path.parent.relative_to(ROOT)
    dest_dir = backup_root / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in (db_path, db_path.with_suffix(db_path.suffix + "-wal"), db_path.with_suffix(db_path.suffix + "-shm")):
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)


def migrate_one(db_path: Path, backup_root: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        copy_backup(db_path, backup_root)

        before = columns(conn, "face_ips_events")
        conn.execute("BEGIN")
        conn.execute("DROP VIEW IF EXISTS v_face_ips_timeline")
        if "vel_score" not in before and "weight_vel" not in before:
            status = "already migrated"
        else:
            if "vel_score" in before:
                conn.execute("ALTER TABLE face_ips_events DROP COLUMN vel_score")
            if "weight_vel" in before:
                conn.execute("ALTER TABLE face_ips_events DROP COLUMN weight_vel")
            status = "migrated"

        # Recompute the logged IPS values using the new prox+cent+gaze formula.
        # Preserve the old post/pre ratio so habituation/recovery effects remain encoded.
        conn.execute(
            """
            UPDATE face_ips_events
            SET
                ips = (
                    (0.5 * COALESCE(prox_score, 0.0)
                     + 0.15 * COALESCE(cent_score, 0.0)
                     + 0.5 * COALESCE(gaze_score, 0.0))
                    * CASE
                        WHEN ips_before_habituation IS NOT NULL
                             AND ABS(ips_before_habituation) > 1e-12
                        THEN COALESCE(ips, ips_before_habituation) / ips_before_habituation
                        ELSE 1.0
                      END
                ),
                ips_before_habituation = (
                    0.5 * COALESCE(prox_score, 0.0)
                    + 0.15 * COALESCE(cent_score, 0.0)
                    + 0.5 * COALESCE(gaze_score, 0.0)
                )
            """
        )
        conn.execute(
            """
            UPDATE face_ips_events
            SET habituation_ips_delta =
                CASE
                    WHEN ips IS NULL OR ips_before_habituation IS NULL THEN habituation_ips_delta
                    ELSE ips - ips_before_habituation
                END
            """
        )
        conn.execute(VIEW_SQL)
        conn.execute("COMMIT")

        after = columns(conn, "face_ips_events")
        forbidden = {"vel_score", "weight_vel"} & after
        if forbidden:
            raise RuntimeError(f"columns still present: {sorted(forbidden)}")
        view_cols = {row[1] for row in conn.execute("PRAGMA table_info(v_face_ips_timeline)")}
        forbidden_view = {"vel_score", "weight_vel"} & view_cols
        if forbidden_view:
            raise RuntimeError(f"view columns still present: {sorted(forbidden_view)}")
        return status
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def main() -> int:
    dbs = sorted(DATA_DIR.glob("*/data_collection/salience_network.db"))
    if not dbs:
        print("No salience_network.db files found.", file=sys.stderr)
        return 1

    backup_root = ROOT / f"data_backup_remove_ips_velocity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for db_path in dbs:
        status = migrate_one(db_path, backup_root)
        print(f"{status}: {db_path}")
    print(f"Backups written to: {backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
