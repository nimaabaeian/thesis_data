#!/usr/bin/env python3
"""Every p-value quoted in a conclusion must appear in the multiplicity ledger.

The previous version ran BH *inside B10* — before D1 had executed — so D1's numbers could never
have entered a family, while its verdict and the README quoted them anyway. It also quoted the
dose x role and dose x phase interaction terms and corrected neither.

These tests make that class of omission a build failure: they scrape every p-value out of the
published surfaces and require each one to be findable in `multiplicity_table.csv`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "outputs"
LEDGER = OUT / "multiplicity_table.csv"

needs_outputs = pytest.mark.skipif(not LEDGER.exists(),
                                   reason="outputs not built; run `make execute`")

# Matches p = 0.03, p=.016, p=1.9e-6, p<0.001 …
P_RE = re.compile(r"\bp\s*[=<]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER)


def quoted_ps(text: str) -> list[float]:
    out = []
    for m in P_RE.finditer(text):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            pass
    return out


def in_ledger(p: float, led: pd.DataFrame, rtol: float = 0.06) -> bool:
    """Is this p-value one the ledger records? Tolerant of the rounding used in prose."""
    vals = led["p"].dropna().values
    if len(vals) == 0:
        return False
    # Prose rounds (p=0.016 for 0.01578...), so compare on relative and absolute tolerance.
    return bool(((abs(vals - p) <= rtol * max(abs(p), 1e-12)) | (abs(vals - p) < 5e-4)).any())


@needs_outputs
def test_ledger_has_the_required_columns():
    led = ledger()
    for c in ["analysis", "model", "term", "p", "family", "status"]:
        assert c in led.columns, f"the ledger must record `{c}`"


@needs_outputs
def test_every_confirmatory_p_is_corrected():
    led = ledger()
    conf = led[led["status"] == "confirmatory"]
    assert len(conf) > 0
    assert conf["q_bh"].notna().all(), "every confirmatory p must carry a BH q-value"
    assert conf["sig_q05"].notna().all()


@needs_outputs
def test_exploratory_ps_are_recorded_but_not_corrected():
    """B4's separated cell, B10.1's non-randomised permutation, B10.2's residual dose and
    B10.3's eligibility stage are recorded in full and deliberately NOT corrected."""
    led = ledger()
    expl = led[led["status"] == "exploratory"]
    assert len(expl) > 0, "the exploratory p-values must still be RECORDED"
    assert expl["q_bh"].isna().all(), \
        "correcting a p-value that came out of a diverging likelihood would only dress it up"


@needs_outputs
def test_D1_is_in_the_ledger():
    """D1's permutation p-values are quoted in its verdict and in the README. The previous
    version ran BH before D1 existed, so they could not have been corrected."""
    led = ledger()
    d1 = led[led["analysis"] == "D1"]
    assert len(d1) >= 1, "D1's permutation p-values must be registered"
    assert (d1["status"] == "confirmatory").all()
    assert d1["q_bh"].notna().all()


@needs_outputs
def test_interaction_terms_are_corrected():
    """dose x role and dose x phase were quoted in the old B10 verdict and corrected in neither
    family."""
    led = ledger()
    conf = led[led["status"] == "confirmatory"]
    assert conf["term"].str.contains(":").any(), \
        "interaction terms that are quoted must enter a BH family"


@needs_outputs
def test_b3_adjusted_ps_are_registered():
    """The 'survives adjustment' claim rests on these, so they are quoted in a conclusion."""
    led = ledger()
    b3 = led[led["analysis"] == "B3"]
    assert len(b3) >= 2, "the unadjusted AND adjusted B3 p-values must both be registered"


@needs_outputs
def test_every_p_quoted_in_results_summary_is_in_the_ledger():
    led = ledger()
    txt = (OUT / "results_summary.md").read_text()
    missing = [p for p in quoted_ps(txt) if not in_ledger(p, led)]
    assert not missing, (
        f"results_summary.md quotes p-value(s) absent from multiplicity_table.csv: {missing}")


@needs_outputs
def test_every_p_quoted_in_the_readme_is_in_the_ledger():
    led = ledger()
    txt = (REPO / "README.md").read_text()
    missing = [p for p in quoted_ps(txt) if not in_ledger(p, led)]
    assert not missing, (
        f"README.md quotes p-value(s) absent from multiplicity_table.csv: {missing}")


@needs_outputs
def test_every_p_quoted_in_success_criteria_is_in_the_ledger():
    led = ledger()
    txt = (OUT / "success_criteria.csv").read_text()
    missing = [p for p in quoted_ps(txt) if not in_ledger(p, led)]
    assert not missing, (
        f"success_criteria.csv quotes p-value(s) absent from multiplicity_table.csv: {missing}")
