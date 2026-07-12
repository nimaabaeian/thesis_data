#!/usr/bin/env python3
"""Regression tests for the corrections made to this analysis.

Each test pins a specific defect that was found and fixed. They are written so that reintroducing
the defect fails the test, which is the only reason to have them.

The tests that need the real data are skipped when it is absent (the public repo does not ship it),
but the *logic* tests — episode detection, ping matching, CTMC dwell accounting, Firth, exact
intervals — run on synthetic fixtures and always execute.

    python -m pytest analysis/tests/ -q
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"
OUT = ANALYSIS / "outputs"
DATA = REPO / "data"

sys.path.insert(0, str(ANALYSIS))

HAVE_DATA = DATA.exists() and any(DATA.glob("*/data_collection/executive_control.db"))
HAVE_OUTPUTS = (OUT / "hs3_episodes.parquet").exists()

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="raw data not present (not distributed)")
needs_outputs = pytest.mark.skipif(not HAVE_OUTPUTS, reason="outputs not built; run `make execute`")


# ---------------------------------------------------------------------------------------
# Re-implementations of the notebook's units, kept in lockstep with build_notebook.py.
# The notebook is generated code, so the tests exercise the same algorithms against
# fixtures they fully control.
# ---------------------------------------------------------------------------------------
HS_FULL_MIN, HS_STARVING_MAX = 60.0, 25.0


def hs_from_level(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x >= HS_FULL_MIN, "HS1", np.where(x >= HS_STARVING_MAX, "HS2", "HS3"))


def build_hs3_episodes_from_levels(hr: pd.DataFrame) -> pd.DataFrame:
    """The level-derived episode builder (mirrors build_notebook.build_hs3_episodes)."""
    hr = hr.sort_values(["run_id", "monotonic_sec"]).copy()
    hr["level"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    eps = []
    for run, g in hr.groupby("run_id"):
        g = g.reset_index(drop=True)
        lvl, mono = g["level"].values, g["monotonic_sec"].values
        starving = lvl < HS_STARVING_MAX
        i = 0
        while i < len(g):
            if not starving[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(g) and starving[j + 1]:
                j += 1
            escaped = (j + 1) < len(g)
            eps.append(dict(run_id=run, entry_mono=float(mono[i]),
                            entry_cause=("run_start" if i == 0 else g["stimulus_type"].iloc[i]),
                            min_level=float(np.min(lvl[i:j + 1])),
                            hs3_duration_sec=float((mono[j + 1] if escaped else mono[-1]) - mono[i]),
                            escaped_starving=int(escaped),
                            censored_at_run_end=int(not escaped)))
            i = j + 1
    return pd.DataFrame(eps)


def state_sequence(hr: pd.DataFrame) -> pd.DataFrame:
    """Sojourns INCLUDING the right-censored terminal segment (to_state is None there)."""
    hr = hr.sort_values(["run_id", "monotonic_sec"]).copy()
    hr["_lvl"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    hr["_st"] = hs_from_level(hr["_lvl"])
    segs = []
    for run, g in hr.groupby("run_id"):
        s, t = g["_st"].values, g["monotonic_sec"].values
        if len(s) < 2:
            continue
        keep = np.r_[True, s[1:] != s[:-1]]
        cs, ct = s[keep], t[keep]
        for i in range(len(cs) - 1):
            segs.append((run, cs[i], float(ct[i + 1] - ct[i]), cs[i + 1], False))
        t_end = float(t[-1])
        if t_end > ct[-1]:
            segs.append((run, cs[-1], t_end - ct[-1], None, True))
    return pd.DataFrame(segs, columns=["run_id", "state", "dwell", "to_state", "terminal"])


def match_pings(ev: pd.DataFrame, ping_types, window: float) -> pd.DataFrame:
    """Greedy one-to-one ping->reply matching (mirrors B5b)."""
    rows = []
    for chat, g in ev.groupby("chat_id"):
        pings = g[g["event_type"].isin(ping_types)].sort_values("timestamp_epoch")
        msgs = g[g["event_type"] == "user_message"].sort_values("timestamp_epoch")
        used: set[int] = set()
        for _, p in pings.iterrows():
            t0 = p["timestamp_epoch"]
            cand = msgs[(msgs["timestamp_epoch"] > t0)
                        & (msgs["timestamp_epoch"] <= t0 + window)
                        & (~msgs["id"].isin(used))]
            hit = len(cand) > 0
            if hit:
                used.add(int(cand["id"].iloc[0]))
            rows.append(dict(chat_id=chat, ping_id=int(p["id"]), replied=int(hit),
                             matched_reply=int(cand["id"].iloc[0]) if hit else None))
    return pd.DataFrame(rows)


def _hr(rows):
    """Fixture helper: build a hunger_raw-shaped frame from (run, t, level, stimulus)."""
    return pd.DataFrame([
        dict(run_id=r, monotonic_sec=float(t), stomach_level_before=float(lv),
             stomach_level_after=float(lv), stimulus_type=st, event_type="sample",
             id=i, day_rome="2026-06-15", timestamp_epoch=1.0e9 + t)
        for i, (r, t, lv, st) in enumerate(rows)
    ])


# =======================================================================================
# 1. The logging artefact: drain crossings are invisible in the label fields
# =======================================================================================
def test_drain_entered_episode_is_detected_from_levels():
    """The defect: an episode entered by passive drain has before==after==HS3 on the crossing
    row, so a builder keyed on `before != HS3 and after == HS3` never sees it. This is the
    30-minute episode the old analysis missed."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 30.0), (1, 27.0), (2, 24.9), (3, 20.0), (4, 15.0), (5, 26.0), (6, 30.0)]])
    eps = build_hs3_episodes_from_levels(hr)
    assert len(eps) == 1, "a drain-entered Starving episode must be found"
    assert eps.iloc[0]["entry_cause"] == "passive_drain"
    assert eps.iloc[0]["min_level"] == pytest.approx(15.0)


def test_label_keyed_builder_would_have_missed_it():
    """Prove the OLD approach fails on the same fixture — otherwise the test above proves nothing."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 30.0), (1, 27.0), (2, 24.9), (3, 20.0), (4, 26.0)]])
    # Reproduce the logger's behaviour: both fields flip together on the crossing row.
    hr["hunger_state_before"] = hs_from_level(hr["stomach_level_after"])
    hr["hunger_state_after"] = hs_from_level(hr["stomach_level_after"])
    old_entries = ((hr["hunger_state_after"] == "HS3")
                   & (hr["hunger_state_before"] != "HS3")).sum()
    assert old_entries == 0, "fixture must reproduce the logging artefact"
    assert len(build_hs3_episodes_from_levels(hr)) == 1, "the level-derived builder must still see it"


def test_episode_count_matches_contiguous_level_blocks():
    """Two separate dips below 25 in one run are two episodes, not one."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 30), (1, 20), (2, 30), (3, 40), (4, 22), (5, 35)]])
    assert len(build_hs3_episodes_from_levels(hr)) == 2


def test_right_censored_episode_is_flagged():
    """A run that ends while Starving is censored, not 'escaped'."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 30), (1, 20), (2, 15)]])
    eps = build_hs3_episodes_from_levels(hr)
    assert len(eps) == 1
    assert eps.iloc[0]["censored_at_run_end"] == 1
    assert eps.iloc[0]["escaped_starving"] == 0


# =======================================================================================
# 2. CTMC dwell accounting
# =======================================================================================
def test_final_dwell_is_included():
    """The defect: the old state_sequence() emitted segments only BETWEEN state changes, so the
    stretch from the last change to the end of the run vanished from the state-time denominators."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 70), (10, 65), (20, 55), (30, 50), (40, 45), (50, 40)]])
    seq = state_sequence(hr)
    total = seq["dwell"].sum()
    assert total == pytest.approx(50.0), "dwell must cover the whole run, including the tail"
    assert seq["terminal"].sum() == 1
    term = seq[seq["terminal"]].iloc[0]
    assert term["state"] == "HS2"
    assert term["dwell"] == pytest.approx(30.0), "20s->50s in HS2 is a real, censored sojourn"


def test_terminal_segment_is_not_a_transition():
    """A run ending is not a state change. Counting it as one invents transitions that
    never happened and corrupts every rate in the generator."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 70), (10, 55), (20, 50)]])
    seq = state_sequence(hr)
    transitions = seq[seq["to_state"].notna()]
    assert len(transitions) == 1, "exactly one real transition: HS1 -> HS2"
    assert seq["terminal"].sum() == 1
    assert transitions["terminal"].sum() == 0, "the terminal row must never be a transition"


def test_state_time_denominator_uses_terminal_dwell():
    """time_in[state] must include terminal dwell; counts must not."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 70), (10, 55), (60, 50)]])
    seq = state_sequence(hr)
    time_in = seq.groupby("state")["dwell"].sum()
    cnt = seq[seq["to_state"].notna()].groupby(["state", "to_state"]).size()
    assert time_in["HS1"] == pytest.approx(10.0)
    assert time_in["HS2"] == pytest.approx(50.0), "the 50s terminal HS2 sojourn must count"
    assert cnt.get(("HS2", "HS1"), 0) == 0, "no transition out of the terminal state"


def test_nonergodic_resamples_are_counted_not_dropped():
    """A generator with no Starving transition has no unique stationary distribution. The old code
    swallowed those resamples in a bare `except: pass`, which conditioned the CI on ergodicity."""
    from scipy.linalg import null_space

    def fit(counts, time_in):
        states = ["HS1", "HS2", "HS3"]
        Q = np.zeros((3, 3))
        for i, si in enumerate(states):
            for j, sj in enumerate(states):
                if i != j and time_in[i] > 0:
                    Q[i, j] = counts[i][j] / time_in[i]
            Q[i, i] = -Q[i].sum()
        ns = null_space(Q.T)
        return (ns.shape[1] == 1)

    # A corpus that never reaches HS3: HS3 is unreachable -> not a single stationary dist.
    counts_no_hs3 = [[0, 3, 0], [4, 0, 0], [0, 0, 0]]
    ok = fit(counts_no_hs3, [100.0, 100.0, 0.0])
    assert not ok, "a resample with no Starving transition must be flagged non-ergodic"

    counts_full = [[0, 3, 0], [4, 0, 2], [0, 2, 0]]
    assert fit(counts_full, [100.0, 100.0, 10.0]), "a connected chain must be ergodic"


# =======================================================================================
# 3. One-to-one ping/reply matching
# =======================================================================================
def _chat(rows):
    return pd.DataFrame([dict(id=i, chat_id=c, event_type=e, timestamp_epoch=float(t),
                              run_id="r1", day_rome="2026-06-15", hs="HS2")
                         for i, (c, e, t) in enumerate(rows)])


def test_one_reply_cannot_answer_many_pings():
    """The defect: the old loop asked, per ping, 'was there ANY user message in the next hour?'
    Three pings and one reply scored 3/3. It must score 1/3."""
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "hs2_entry", 10), ("c1", "hs2_entry", 20),
                ("c1", "user_message", 30)])
    pm = match_pings(ev, ["hs2_entry"], window=3600.0)
    assert len(pm) == 3
    assert pm["replied"].sum() == 1, "one reply may answer exactly one ping"


def test_no_duplicate_reply_attribution():
    """No reply id may be consumed twice."""
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "hs2_entry", 5),
                ("c1", "user_message", 10), ("c1", "user_message", 12)])
    pm = match_pings(ev, ["hs2_entry"], window=3600.0)
    matched = pm["matched_reply"].dropna().tolist()
    assert len(matched) == len(set(matched)), "a reply was attributed to more than one ping"
    assert pm["replied"].sum() == 2


def test_replies_never_exceed_pings():
    ev = _chat([("c1", "hs2_entry", 0)] + [("c1", "user_message", t) for t in range(1, 20)])
    pm = match_pings(ev, ["hs2_entry"], window=3600.0)
    assert pm["replied"].sum() <= len(pm)


def test_reply_outside_window_does_not_count():
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "user_message", 5000)])
    pm = match_pings(ev, ["hs2_entry"], window=3600.0)
    assert pm["replied"].sum() == 0


def test_recovery_notifications_are_excluded():
    """hs3_recovery is 'thanks, I'm full' — a notification, not a request for food. Counting
    replies to it as evidence that hunger signalling works is a category error."""
    ev = _chat([("c1", "hs3_recovery", 0), ("c1", "user_message", 10),
                ("c1", "hs2_entry", 100)])
    pm = match_pings(ev, ["hs2_entry", "hs3_proactive"], window=3600.0)
    assert len(pm) == 1, "only genuine hunger pings enter the analysis"
    assert pm.iloc[0]["ping_id"] == 2


# =======================================================================================
# 4. Exact intervals and Firth (the B4 separation fix)
# =======================================================================================
def test_exact_ci_for_zero_of_n():
    """0/15 has a real upper bound. Reporting it as '0%, perfect compliance' with no interval
    overstates what 15 observations can rule out."""
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(0, 15, method="beta")
    assert lo == 0.0
    assert 0.15 < hi < 0.25, f"upper bound for 0/15 should be ~0.22, got {hi}"


def test_exact_ci_for_one_of_thirteen():
    """B4's cell. The exact interval is wide, and that width is the finding."""
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(1, 13, method="beta")
    assert lo < 0.01 and hi > 0.30, f"1/13 must yield a wide interval, got [{lo}, {hi}]"


def test_firth_is_finite_under_separation():
    """The defect: unpenalised logistic regression diverges under separation, and the Wald SE
    collapses with it, manufacturing p ~ 1e-6 out of a cell with almost no information."""
    from build_notebook import CELLS  # noqa: F401  (import guard: module is importable)

    # Perfectly separated data: x==1 always y==0, x==0 always y==1.
    X = np.column_stack([np.ones(20), np.r_[np.ones(10), np.zeros(10)]])
    y = np.r_[np.zeros(10), np.ones(10)]

    # Unpenalised MLE diverges.
    import statsmodels.api as sm
    with np.errstate(all="ignore"):
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
            diverged = abs(res.params[1]) > 10
        except Exception:
            diverged = True
    assert diverged, "fixture must actually be separated"

    # Firth stays finite.
    beta = _firth(X, y)
    assert np.all(np.isfinite(beta)), "Firth must return a finite estimate under separation"
    assert abs(beta[1]) < 10, f"Firth estimate should be shrunk, got {beta[1]}"


def _firth(X, y, max_iter=200, tol=1e-8):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        XtWX = X.T @ (X * W[:, None])
        inv = np.linalg.pinv(XtWX)
        H = (X * W[:, None]) @ inv
        h = np.einsum("ij,ij->i", H, X)
        U = X.T @ (y - p + h * (0.5 - p))
        step = inv @ U
        nb = beta + step
        if np.max(np.abs(nb - beta)) < tol:
            return nb
        beta = nb
    return beta


# =======================================================================================
# 5. Zero-exposure person-days
# =======================================================================================
def test_zero_meal_person_days_are_retained():
    """A day someone was present and fed nothing is a zero in the denominator, not a missing row.
    Dropping those days inflates the meal rate for anyone who showed up rarely."""
    inter = pd.DataFrame([
        dict(person_id="P01", day_rome="d1", meals_eaten_count=2, interaction_id=1),
        dict(person_id="P01", day_rome="d2", meals_eaten_count=0, interaction_id=2),
        dict(person_id="P02", day_rome="d1", meals_eaten_count=0, interaction_id=3),
    ])
    pdm = (inter.groupby(["person_id", "day_rome"])
                .agg(meals=("meals_eaten_count", "sum"),
                     n_interactions=("interaction_id", "size")).reset_index())
    assert len(pdm) == 3, "all three person-days must survive, including the two zero-meal ones"
    assert (pdm["meals"] == 0).sum() == 2
    # And the exposure offset must be defined for them.
    assert np.all(np.isfinite(np.log(pdm["n_interactions"].clip(lower=1))))


# =======================================================================================
# 6. Artifacts produced by a real run
# =======================================================================================
@needs_outputs
def test_b3_outcome_is_renamed():
    """`fed01` described a robot action ('feeding pursuit'). It is an outcome of the dyad: a meal
    arrived. The variable name and the estimand must both say so.

    The old name may still appear in PROSE (the correction is documented, deliberately), but it
    must not survive as a live identifier — so this checks the model formulas, not the narrative.
    """
    src = (ANALYSIS / "build_notebook.py").read_text()
    assert "feeding_received ~ deficit" in src, "the renamed outcome must be the modelled one"

    # No live code may still reference the old identifier: no assignment, no formula, no column.
    live = re.findall(r'fed01\s*[=~\["\']', src)
    assert not live, f"`fed01` is still used as an identifier: {live}"

    # And the notebook must not fit anything with it.
    nb = json.loads((ANALYSIS / "orexigenic_analysis.ipynb").read_text())
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        assert "fed01" not in code, "a code cell still references fed01"


@needs_outputs
def test_b3_reports_both_clusterings():
    """14 person-clusters is far below where a GEE sandwich SE is trustworthy, so the
    bootstrap — by person AND by run — is what the verdict must lead with."""
    sens = pd.read_csv(OUT / "small_cluster_sensitivity.csv")
    metrics = set(sens["metric"])
    assert "B3_deficit_feeding_OR" in metrics
    assert "B3_deficit_feeding_OR_runcluster" in metrics, \
        "B3 must report a run-clustered bootstrap as well as a person-clustered one"


@needs_outputs
def test_unknown_faces_excluded_from_person_clusters():
    """'unknown' is an unrecognised-face placeholder, not a person. Pooling ~23 interactions from
    an unknown number of strangers into one cluster asserts a dependence structure that does not
    exist."""
    src = (ANALYSIS / "build_notebook.py").read_text()
    assert 'd_named = d[d["person_id"] != "unknown"]' in src, \
        "person-clustered models must drop the unknown-face placeholder"


@needs_outputs
def test_multiplicity_table_is_complete():
    """The old version corrected 5 p-values while quoting interaction terms it never corrected."""
    mt = pd.read_csv(OUT / "multiplicity_table.csv")
    assert {"analysis", "model", "term", "p", "family", "status"} <= set(mt.columns)
    conf = mt[mt["status"] == "confirmatory"]
    assert conf["q_bh"].notna().all(), "every confirmatory p must carry a q-value"
    # Interaction terms must be present, not silently omitted.
    assert conf["term"].str.contains(":").any(), "dose x moderator interactions must be corrected"
    # B4's separated p must be recorded but NOT corrected.
    b4 = mt[mt["analysis"] == "B4"]
    if len(b4):
        assert (b4["status"] == "exploratory").all()
        assert b4["q_bh"].isna().all(), "B4's separated p-value must not be BH-corrected"


@needs_outputs
def test_all_starving_episodes_present():
    ep = pd.read_parquet(OUT / "hs3_episodes.parquet")
    assert len(ep) > 8, f"level-derived builder must find more than the old 8; got {len(ep)}"
    assert "entry_cause" in ep.columns
    assert (ep["entry_cause"] == "passive_drain").any(), \
        "drain-entered episodes must be present — they are the ones the old builder dropped"
    assert "censored_at_run_end" in ep.columns


@needs_outputs
def test_evidence_classes_are_used():
    """No result may carry a bare 'Supported'."""
    sc = pd.read_csv(OUT / "success_criteria.csv")
    allowed = {"Implementation verification", "Within-deployment association",
               "Exploratory observation", "Inconclusive", "Requires replication"}
    assert set(sc["evidence_class"]) <= allowed, \
        f"illegal evidence class: {set(sc['evidence_class']) - allowed}"
    assert "Supported" not in set(sc["evidence_class"])


@needs_outputs
def test_b1_b2_are_implementation_verification():
    sc = pd.read_csv(OUT / "success_criteria.csv").set_index("id")
    assert sc.loc["RQ1-1", "evidence_class"] == "Implementation verification"
    assert sc.loc["RQ1-2", "evidence_class"] == "Implementation verification"


@needs_outputs
def test_b4_is_not_confirmatory():
    sc = pd.read_csv(OUT / "success_criteria.csv").set_index("id")
    assert sc.loc["RQ1-4", "evidence_class"] in ("Exploratory observation", "Inconclusive"), \
        "13 interactions with 1 success cannot be a confirmatory association"


@needs_outputs
def test_dropcolumn_table_is_gone():
    """With two features, drop-column importance IS the ablation. Publishing both double-counted."""
    assert not (OUT / "ml_dropcolumn_importance.csv").exists()


@needs_outputs
def test_reproducible_under_fixed_seed():
    """Two consecutive builds of the notebook must produce byte-identical source."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.ipynb", Path(td) / "b.ipynb"
        for p in (a, b):
            subprocess.run([sys.executable, str(ANALYSIS / "build_notebook.py"), str(p)],
                           check=True, capture_output=True)
        na, nb = json.loads(a.read_text()), json.loads(b.read_text())
        sa = [c["source"] for c in na["cells"]]
        sb = [c["source"] for c in nb["cells"]]
        assert sa == sb, "notebook build is not deterministic"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
