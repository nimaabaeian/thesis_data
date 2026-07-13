#!/usr/bin/env python3
"""End-to-end determinism: execute the whole analysis twice from clean, compare every artifact.

A fixed seed in the source is a promise, not a proof. This test keeps the promise honest by
running the real pipeline twice into scratch output directories and comparing every generated
CSV, Parquet table, report and figure. Anything that differs — an unseeded shuffle, a dict
iteration order, a timestamp baked into data rather than into a header — shows up here and
nowhere else.

It is slow (two full notebook executions), so it is marked and excluded from the default run:

    make determinism        # or: pytest -m slow analysis/tests/test_determinism.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"
NB = ANALYSIS / "orexigenic_analysis.ipynb"
OUT = ANALYSIS / "outputs"
FIGS = ANALYSIS / "figures"
CACHE = ANALYSIS / "cache"
DATA = REPO / "data"

HAVE_DATA = DATA.exists() and any(DATA.glob("*/data_collection/executive_control.db"))
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAVE_DATA, reason="raw data not present (not distributed)"),
]

# Reports embed a generation timestamp in their header; that is metadata about the run, not a
# result, so those lines are excluded from the comparison. Everything else must match exactly.
TIMESTAMP_LINE = ("_Generated", "generated_utc", "Generated ")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stable_text(p: Path) -> str:
    return "\n".join(l for l in p.read_text(errors="replace").splitlines()
                     if not any(t in l for t in TIMESTAMP_LINE))


def run_once(tag: str) -> Path:
    """Execute the notebook from a genuinely clean state; snapshot the artifacts."""
    for d in (OUT, FIGS, CACHE):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="0")
    subprocess.run([sys.executable, str(ANALYSIS / "build_notebook.py"), str(NB)],
                   check=True, capture_output=True, env=env)
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", str(NB),
                        "--inplace", "--ExecutePreprocessor.timeout=1800"],
                       capture_output=True, env=env)
    assert r.returncode == 0, f"execution {tag} failed:\n{r.stderr.decode()[-2500:]}"
    snap = Path(tempfile.mkdtemp(prefix=f"det-{tag}-"))
    shutil.copytree(OUT, snap / "outputs")
    shutil.copytree(FIGS, snap / "figures")
    return snap


@pytest.fixture(scope="module")
def two_runs():
    a = run_once("a")
    b = run_once("b")
    yield a, b
    shutil.rmtree(a, ignore_errors=True)
    shutil.rmtree(b, ignore_errors=True)


def test_same_artifacts_produced(two_runs):
    a, b = two_runs
    fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    assert fa == fb, f"different artifacts produced: {fa ^ fb}"


def test_csv_outputs_are_identical(two_runs):
    a, b = two_runs
    diffs = []
    for pa in sorted((a / "outputs").glob("*.csv")):
        pb = b / "outputs" / pa.name
        da, db = pd.read_csv(pa), pd.read_csv(pb)
        try:
            pd.testing.assert_frame_equal(da, db, check_exact=False, rtol=0, atol=0)
        except AssertionError as e:
            diffs.append(f"{pa.name}: {str(e).splitlines()[0]}")
    assert not diffs, "CSV outputs differ between runs:\n" + "\n".join(diffs)


def test_parquet_outputs_are_identical(two_runs):
    a, b = two_runs
    diffs = []
    for pa in sorted((a / "outputs").glob("*.parquet")):
        pb = b / "outputs" / pa.name
        da, db = pd.read_parquet(pa), pd.read_parquet(pb)
        try:
            pd.testing.assert_frame_equal(da, db, check_exact=False, rtol=0, atol=0)
        except AssertionError as e:
            diffs.append(f"{pa.name}: {str(e).splitlines()[0]}")
    assert not diffs, "Parquet outputs differ between runs:\n" + "\n".join(diffs)


def test_reports_are_identical_modulo_timestamp(two_runs):
    a, b = two_runs
    diffs = []
    for pa in sorted((a / "outputs").glob("*.md")):
        pb = b / "outputs" / pa.name
        if stable_text(pa) != stable_text(pb):
            diffs.append(pa.name)
    assert not diffs, f"reports differ between runs (beyond their timestamp): {diffs}"


def test_figures_are_byte_identical(two_runs):
    a, b = two_runs
    diffs = [pa.name for pa in sorted((a / "figures").glob("*.png"))
             if sha(pa) != sha(b / "figures" / pa.name)]
    assert not diffs, f"figures are not reproducible byte-for-byte: {diffs}"
