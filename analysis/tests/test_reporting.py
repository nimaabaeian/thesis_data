#!/usr/bin/env python3
"""Regression checks for published counts, evidence labels, and attribution scope.

The inferential tests already protect p-values. These checks protect the non-p-value headline
facts that previously drifted: 33 raw sheet absences were described as 33 genuine no-shows even
though log reconciliation leaves 23.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "outputs"
README = REPO / "README.md"
PANEL = OUT / "b10_scheduled_day_panel_reconciliation.csv"
ATTRIBUTION = OUT / "b10_feeder_attribution_sensitivity.csv"

needs_outputs = pytest.mark.skipif(not PANEL.exists(),
                                   reason="outputs not built; run `make execute`")


@needs_outputs
def test_published_attendance_reconciliation_counts_are_consistent():
    panel = pd.read_csv(PANEL)
    n_scheduled = len(panel)
    n_sheet_absent = int((panel["attended_sheet"] == 0).sum())
    n_overridden = int(((panel["attended_sheet"] == 0) & (panel["attended"] == 1)).sum())
    n_no_show = int((panel["attended"] == 0).sum())

    assert (n_scheduled, n_sheet_absent, n_overridden, n_no_show) == (96, 33, 10, 23)

    readme = README.read_text()
    assert "sign-in sheet marks **33 absences**" in readme
    assert "logs override 10 of them" in readme
    assert "**23 genuine no-show zeros**" in readme
    assert "33 are genuine" not in readme


@needs_outputs
def test_scorecard_claims_match_their_evidence_scope():
    criteria = pd.read_csv(OUT / "success_criteria.csv").set_index("id")
    expected = {
        "RQ1-4": "Observed Starving interactions show a pattern consistent with priority reallocation",
        "RQ2-a": "Deficit severity is associated with recovery-related behaviour",
        "RQ2-b": "13 of 17 observed Starving episodes recovered by feeding",
        "RQ2-c": "Long-run Starving occupancy is not identified; observed occupancy was 1.67%",
        "RQ3-a": "Role groups differed descriptively in attendance and interaction exposure",
        "RQ3-b": ("Affinity updating and eligibility operate as programmed; downstream "
                   "behavioural effects remain inconclusive"),
    }
    for claim_id, claim in expected.items():
        assert criteria.loc[claim_id, "claim"] == claim


@needs_outputs
def test_public_surfaces_do_not_reintroduce_disallowed_causal_claims():
    surfaces = "\n".join([
        README.read_text(),
        (OUT / "results_summary.md").read_text(),
        (OUT / "success_criteria.csv").read_text(),
    ])
    disallowed = [
        "Being told to feed the robot made people",
        "The role manipulation changed what people did",
        "Deficit expression elicits recovery behaviour",
        "Long-run Starving occupancy is low",
        "The remote channel is a real but WEAK recovery pathway",
    ]
    for phrase in disallowed:
        assert phrase not in surfaces


@needs_outputs
def test_feeder_attribution_sensitivity_is_explicit_and_aggregate_only():
    assert ATTRIBUTION.exists(), "the three attribution definitions must be exported"
    table = pd.read_csv(ATTRIBUTION)
    assert table["method"].tolist() == [
        "direct interaction log",
        "active-interaction timestamp attribution",
        "verified feeder_face_id only",
    ]
    assert list(table["coverage"].between(0, 1)) == [True, True, True]
    assert float(table.loc[table["method"] == "direct interaction log",
                           "phase1_no_feed_meal_count"].iloc[0]) == 0
    assert float(table.loc[table["method"] == "active-interaction timestamp attribution",
                           "phase1_no_feed_energy"].iloc[0]) == 115
    assert not any(c in table.columns for c in ["person_id", "face_id", "feeder_face_id"])

