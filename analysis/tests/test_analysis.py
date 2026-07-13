#!/usr/bin/env python3
"""Regression tests for the corrections made to this analysis.

Every test exercises the PRODUCTION functions in `analysis/statistical_helpers.py` — the same
module the notebook imports. Nothing is reimplemented here.

That is the point. The previous suite carried its own copies of the episode builder, the ping
matcher, the sojourn splitter and Firth, so the tests could pass while the notebook shipped
different code. Worse, one of those copies (the flapping detector) contained the *same* bug as
the notebook, so the test that "verified" it could never have failed on any input. Duplicated
logic does not test anything; it tests itself.

    make test
"""
from __future__ import annotations

import json
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

# THE PRODUCTION CODE. If an import here fails, the notebook is broken too.
from statistical_helpers import (  # noqa: E402
    EVIDENCE_CLASSES, HS_STARVING_MAX,
    build_hs3_episodes, build_hs_crossings, hs_from_level, flapping,
    state_sequence, fit_ctmc, is_irreducible,
    match_pings, control_windows, assign_run_by_time,
    exact_prop_ci, cluster_bootstrap, exact_permutation_p, enumerate_label_assignments,
    firth_logit, firth_profile_ci,
    fit_gee_checked, adjustment_verdict, ipw_diagnostics, spec_agreement,
)

HAVE_OUTPUTS = (OUT / "hs3_episodes.parquet").exists()
needs_outputs = pytest.mark.skipif(not HAVE_OUTPUTS, reason="outputs not built; run `make execute`")


def _hr(rows):
    """hunger_raw-shaped fixture from (run, t, level, stimulus)."""
    return pd.DataFrame([
        dict(run_id=r, monotonic_sec=float(t), stomach_level_before=float(lv),
             stomach_level_after=float(lv), stimulus_type=st, event_type="sample",
             id=i, day_rome="2026-06-15", timestamp_epoch=1.0e9 + t, session_id="s",
             active_energy_cost=0.0, exec_interaction_id=None)
        for i, (r, t, lv, st) in enumerate(rows)])


def _chat(rows):
    return pd.DataFrame([dict(id=i, chat_id=c, event_type=e, timestamp_epoch=float(t),
                              run_id="r1", day_rome="2026-06-15", hs="HS2")
                         for i, (c, e, t) in enumerate(rows)])


# =======================================================================================
# 1. The logging artefact: drain crossings are invisible in the label fields
# =======================================================================================
def test_drain_entered_episode_is_detected_from_levels():
    """The 30-minute episode the old builder could not see. It required a logged
    `before != HS3 -> after == HS3` row, which the logger never emits for a drain crossing."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 30.0), (1, 27.0), (2, 24.9), (3, 20.0), (4, 15.0), (5, 26.0), (6, 30.0)]])
    eps = build_hs3_episodes(hr)
    assert len(eps) == 1
    assert eps.iloc[0]["entry_cause"] == "passive_drain"
    assert eps.iloc[0]["min_level"] == pytest.approx(15.0)


def test_label_keyed_builder_would_have_missed_it():
    """Prove the OLD approach fails on the same fixture — otherwise the test above proves nothing."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 30.0), (1, 27.0), (2, 24.9), (3, 20.0), (4, 26.0)]])
    hr["hunger_state_before"] = hs_from_level(hr["stomach_level_after"])
    hr["hunger_state_after"] = hs_from_level(hr["stomach_level_after"])
    old_entries = ((hr["hunger_state_after"] == "HS3")
                   & (hr["hunger_state_before"] != "HS3")).sum()
    assert old_entries == 0, "fixture must reproduce the logging artefact"
    assert len(build_hs3_episodes(hr)) == 1, "the level-derived builder must still see it"


def test_two_dips_are_two_episodes():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 30), (1, 20), (2, 30), (3, 40), (4, 22), (5, 35)]])
    assert len(build_hs3_episodes(hr)) == 2


def test_right_censored_episode_is_flagged():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 30), (1, 20), (2, 15)]])
    eps = build_hs3_episodes(hr)
    assert eps.iloc[0]["censored_at_run_end"] == 1
    assert eps.iloc[0]["escaped_starving"] == 0


def test_crossings_include_drain_falls():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 62), (1, 59), (2, 58)]])
    cr = build_hs_crossings(hr)
    assert len(cr) == 1
    assert cr.iloc[0]["cause"] == "passive_drain"


# =======================================================================================
# 2. Flapping — the detector that could never fire
# =======================================================================================
def test_flapping_detects_a_genuine_reversal():
    """The old detector compared from_state to the PREVIOUS from_state. A reversal requires
    from_state == previous TO_state. The condition was unsatisfiable and returned 0 on any data."""
    m = pd.DataFrame([
        dict(run_id="r1", monotonic_sec=100.0, from_state="HS1", to_state="HS2", cause="interaction_cost"),
        dict(run_id="r1", monotonic_sec=120.0, from_state="HS2", to_state="HS1", cause="feeding"),
    ])
    fl, tot, rev = flapping(m, window=120.0)
    assert fl == 1, "a crossing undone 20 s later IS a rapid reversal"
    assert tot == 2
    assert rev.iloc[0]["gap_sec"] == pytest.approx(20.0)


def test_flapping_ignores_a_repeat_in_the_same_direction():
    m = pd.DataFrame([
        dict(run_id="r1", monotonic_sec=100.0, from_state="HS1", to_state="HS2", cause="c"),
        dict(run_id="r1", monotonic_sec=110.0, from_state="HS1", to_state="HS2", cause="c"),
    ])
    assert flapping(m, window=120.0)[0] == 0


def test_flapping_respects_the_window():
    m = pd.DataFrame([
        dict(run_id="r1", monotonic_sec=0.0, from_state="HS1", to_state="HS2", cause="c"),
        dict(run_id="r1", monotonic_sec=500.0, from_state="HS2", to_state="HS1", cause="c"),
    ])
    assert flapping(m, window=120.0)[0] == 0


# =======================================================================================
# 3. CTMC: terminal dwell, irreducibility, stationary validation
# =======================================================================================
def test_final_dwell_is_included():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 70), (10, 65), (20, 55), (30, 50), (40, 45), (50, 40)]])
    seq = state_sequence(hr)
    assert seq["dwell"].sum() == pytest.approx(50.0), "dwell must cover the whole run"
    assert seq["terminal"].sum() == 1
    term = seq[seq["terminal"]].iloc[0]
    assert term["state"] == "HS2"
    assert term["dwell"] == pytest.approx(30.0)


def test_terminal_segment_is_not_a_transition():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 70), (10, 55), (20, 50)]])
    seq = state_sequence(hr)
    tr = seq[seq["to_state"].notna()]
    assert len(tr) == 1, "exactly one real transition: HS1 -> HS2"
    assert tr["terminal"].sum() == 0, "the terminal row must never be a transition"


def test_state_time_uses_terminal_dwell_but_counts_do_not():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 70), (10, 55), (60, 50)]])
    fit = fit_ctmc(state_sequence(hr))
    assert fit["time_in"]["HS2"] == pytest.approx(50.0), "the 50 s terminal HS2 sojourn must count"
    assert fit["counts"].loc["HS2", "HS1"] == 0, "no transition out of the terminal state"


def test_irreducibility_is_strong_connectivity_not_nullspace_dim():
    """A chain that never reaches HS3 is REDUCIBLE. Inferring ergodicity from dim(null(Q'))==1
    conflates 'unique stationary distribution' with 'irreducible' — a reducible chain with one
    absorbing class also has a 1-D null space."""
    reducible = np.array([[0, 3, 0], [4, 0, 0], [0, 0, 0]])
    assert not is_irreducible(reducible)
    connected = np.array([[0, 3, 0], [4, 0, 2], [0, 2, 0]])
    assert is_irreducible(connected)


def test_stationary_vector_is_validated_not_coerced():
    """fit_ctmc must VALIDATE pi (>=0, sums to 1, pi @ Q ~ 0), not abs() an arbitrary basis
    vector and normalise it."""
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in
              [(0, 70), (10, 55), (20, 70), (30, 55), (40, 20), (50, 70), (60, 20), (70, 70)]])
    fit = fit_ctmc(state_sequence(hr))
    if fit["stationary_valid"]:
        pi = fit["pi"]
        assert np.all(pi >= -1e-9)
        assert pi.sum() == pytest.approx(1.0)
        assert np.max(np.abs(pi @ fit["Q"])) < 1e-7


def test_unidentified_chain_reports_a_reason():
    hr = _hr([("r1", t, lv, "passive_drain") for t, lv in [(0, 70), (10, 55), (20, 50)]])
    fit = fit_ctmc(state_sequence(hr))
    assert not fit["irreducible"], "a chain that never reaches Starving is reducible"
    if not fit["stationary_valid"]:
        assert fit["reason"], "a failure must say WHY"


# =======================================================================================
# 4. Ping <-> reply matching
# =======================================================================================
def test_one_reply_cannot_answer_many_pings():
    """The old loop asked, per ping, 'was there ANY message in the next hour?' Three pings and
    one reply scored 3/3. It must score 1/3."""
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "hs2_entry", 10), ("c1", "hs2_entry", 20),
                ("c1", "user_message", 30)])
    pm = match_pings(ev, ["hs2_entry"], 3600.0)
    assert len(pm) == 3
    assert pm["replied"].sum() == 1


def test_reply_matches_the_NEAREST_preceding_ping():
    """Reply-centric matching. A greedy forward walk would credit the reply to the ping an hour
    old rather than the one sent 10 seconds before it."""
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "hs2_entry", 3000), ("c1", "user_message", 3010)])
    pm = match_pings(ev, ["hs2_entry"], 3600.0)
    answered = pm[pm.replied == 1]
    assert len(answered) == 1
    assert answered.iloc[0]["t"] == 3000.0, "the NEAREST preceding ping must claim the reply"


def test_no_duplicate_reply_attribution():
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "hs2_entry", 5),
                ("c1", "user_message", 10), ("c1", "user_message", 12)])
    pm = match_pings(ev, ["hs2_entry"], 3600.0)
    matched = pm["matched_reply"].dropna().tolist()
    assert len(matched) == len(set(matched))
    assert pm["replied"].sum() == 2


def test_replies_never_exceed_pings():
    ev = _chat([("c1", "hs2_entry", 0)] + [("c1", "user_message", t) for t in range(1, 20)])
    pm = match_pings(ev, ["hs2_entry"], 3600.0)
    assert pm["replied"].sum() <= len(pm)


def test_reply_outside_window_does_not_count():
    ev = _chat([("c1", "hs2_entry", 0), ("c1", "user_message", 5000)])
    assert match_pings(ev, ["hs2_entry"], 3600.0)["replied"].sum() == 0


def test_recovery_notifications_are_excluded():
    """hs3_recovery is 'thanks, I'm full'. A reply to a thank-you is not evidence that hunger
    signalling works."""
    ev = _chat([("c1", "hs3_recovery", 0), ("c1", "user_message", 10), ("c1", "hs2_entry", 100)])
    pm = match_pings(ev, ["hs2_entry", "hs3_proactive"], 3600.0)
    assert len(pm) == 1
    assert pm.iloc[0]["ping_id"] == 2


# --- controls ---------------------------------------------------------------------------
def _spans():
    return pd.DataFrame([dict(run_id="R1", t_start=0.0, t_end=100000.0)])


def test_run_assignment_is_by_time_not_run_id():
    """The chat and executive DBs have DISJOINT run_id namespaces (zero overlap), so joining on
    run_id matches nothing. Wall-clock containment is the only correct key."""
    got = assign_run_by_time([50.0, 999999.0], _spans())
    assert got == ["R1", None]


def test_control_window_is_ping_free_and_one_reply_one_window():
    ev = _chat([("c1", "hs2_entry", 5000), ("c1", "user_message", 5100)])
    pm = match_pings(ev, ["hs2_entry"], 3600.0)
    pm["exec_run_id"] = assign_run_by_time(pm["t"], _spans())
    ctrl = control_windows(ev, pm, 3600.0, _spans(), seed=1)
    ok = ctrl[ctrl["matched"] == True]  # noqa: E712
    for t0 in ok["t"]:
        assert not ((pm["t"] >= t0 - 3600.0) & (pm["t"] <= t0 + 3600.0)).any(), \
            "a control window must contain no ping"


def test_control_reply_is_not_double_counted():
    """A single user message must not satisfy several control windows, exactly as for pings."""
    ev = _chat([("c1", "hs2_entry", 5000), ("c1", "hs2_entry", 6000),
                ("c1", "user_message", 20000)])
    pm = match_pings(ev, ["hs2_entry"], 900.0)
    pm["exec_run_id"] = assign_run_by_time(pm["t"], _spans())
    ctrl = control_windows(ev, pm, 900.0, _spans(), seed=3)
    ok = ctrl[ctrl["matched"] == True]  # noqa: E712
    assert ok["replied"].sum() <= 1, "one message cannot satisfy two control windows"


# =======================================================================================
# 5. Exact intervals, Firth, permutation
# =======================================================================================
def test_exact_ci_for_zero_of_n():
    est, lo, hi = exact_prop_ci(0, 15)
    assert lo == 0.0 and 0.15 < hi < 0.25, "0/15 has a real upper bound (~0.22)"


def test_exact_ci_for_one_of_thirteen():
    est, lo, hi = exact_prop_ci(1, 13)
    assert lo < 0.01 and hi > 0.30, "B4's cell must yield a wide interval"


def test_firth_is_finite_under_separation():
    """Plain ML diverges under separation and its Wald SE collapses with it, manufacturing a
    tiny p-value from a cell with almost no information. B4 is exactly that shape."""
    X = np.column_stack([np.ones(20), np.r_[np.ones(10), np.zeros(10)]])
    y = np.r_[np.zeros(10), np.ones(10)]
    import statsmodels.api as sm
    with np.errstate(all="ignore"):
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
            diverged = abs(res.params[1]) > 10
        except Exception:
            diverged = True
    assert diverged, "fixture must actually be separated"
    beta = firth_logit(X, y)
    assert np.all(np.isfinite(beta)) and abs(beta[1]) < 10


def test_firth_profile_ci_is_finite():
    X = np.column_stack([np.ones(20), np.r_[np.ones(10), np.zeros(10)]])
    y = np.r_[np.zeros(10), np.ones(10)]
    b, lo, hi, p = firth_profile_ci(X, y, 1)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < b < hi


def test_permutation_is_exactly_enumerated():
    """C(12,2) = 66 assignments. Approximating a 66-point discrete distribution with 5,000
    random draws adds Monte-Carlo error for nothing."""
    ids = [f"P{i:02d}" for i in range(12)]
    labs = ["feeder"] * 2 + ["normal"] * 10
    assigns = enumerate_label_assignments(ids, labs)
    assert len(assigns) == 66


def test_permutation_p_has_the_design_floor():
    df = pd.DataFrame([dict(person_id=f"P{i:02d}", role="feeder" if i < 2 else "normal",
                            y=10.0 if i < 2 else 1.0) for i in range(12)])
    r = exact_permutation_p(df, "person_id", "role",
                            lambda d: d[d.role == "feeder"]["y"].mean()
                            / max(d[d.role == "normal"]["y"].mean(), 1e-9))
    assert r["n_assignments"] == 66
    assert r["p"] >= r["floor"] - 1e-12, "p cannot fall below 1/66"


# =======================================================================================
# 6. Model status must be reported, never swallowed
# =======================================================================================
def test_failed_model_is_reported_not_caught():
    import statsmodels.api as sm
    df = pd.DataFrame(dict(y=[0, 1, 0, 1], x=[0, 1, 0, 1], g=["a", "a", "b", "b"]))
    r = fit_gee_checked("y ~ nonexistent_column", df, groups="g",
                        family=sm.families.Binomial(), focal="nonexistent_column")
    assert r["status"] == "failed"
    assert r["reason"], "a failure must carry a reason"


def test_adjustment_verdict_refuses_on_a_failed_model():
    """'Survives adjustment' requires EVERY prespecified model to produce a finite, directionally
    consistent estimate. A model that failed does not get quietly excluded from the tally."""
    ok = dict(status="ok", estimate=1.5, cluster="person", reason=None)
    bad = dict(status="failed", estimate=np.nan, cluster="run", reason="did not converge")
    assert adjustment_verdict([ok, ok], focal_sign=+1.0)["survives"]
    v = adjustment_verdict([ok, bad], focal_sign=+1.0)
    assert not v["survives"]
    assert "failed" in v["reason"]


def test_adjustment_verdict_refuses_on_sign_flip():
    a = dict(status="ok", estimate=+1.5, cluster="person", reason=None)
    b = dict(status="ok", estimate=-0.8, cluster="run", reason=None)
    assert not adjustment_verdict([a, b], focal_sign=+1.0)["survives"]


def test_spec_agreement_requires_magnitude_not_just_sign():
    """+0.041 and +0.168 agree in sign while differing four-fold. Calling them 'consistent'
    would hide the entire finding."""
    r = spec_agreement({"a": 0.041, "b": 0.168}, rel_tol=0.5)
    assert r["sign_agree"] and not r["magnitude_agree"]
    r2 = spec_agreement({"a": 0.100, "b": 0.120}, rel_tol=0.5)
    assert r2["sign_agree"] and r2["magnitude_agree"]


def test_ipw_diagnostics_flag_positivity():
    p = np.array([0.02, 0.5, 0.9])
    obs = np.array([1, 1, 1])
    w = 0.5 / p
    d = ipw_diagnostics(p, obs, w)
    assert d["positivity_violation"], "a propensity of 0.02 is a positivity violation"
    assert d["ess"] < d["n_observed"], "ESS must be below n when weights are unequal"


# =======================================================================================
# 7. Artifacts produced by a real run
# =======================================================================================
@needs_outputs
def test_b3_outcome_is_renamed():
    src = (ANALYSIS / "build_notebook.py").read_text()
    assert "feeding_received ~ deficit" in src
    assert not re.findall(r'fed01\s*[=~\["\']', src), "`fed01` is still a live identifier"
    assert "feed_pursuit=" not in src, "`feed_pursuit` is still a live column name"
    nb = json.loads((ANALYSIS / "orexigenic_analysis.ipynb").read_text())
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            code = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            assert "fed01" not in code


@needs_outputs
def test_b3_reports_status_for_every_model():
    t = pd.read_csv(OUT / "b3_adjusted_models.csv")
    assert {"model", "cluster", "status", "reason", "odds_ratio"} <= set(t.columns)
    assert t["status"].notna().all(), "every model must carry a status"
    ok = t[(t.model == "ANALYSIS-SPECIFIED adjusted")]
    assert len(ok) == 2, "the analysis-specified model must be fitted under BOTH clusterings"


@needs_outputs
def test_all_starving_episodes_present():
    ep = pd.read_parquet(OUT / "hs3_episodes.parquet")
    assert len(ep) > 8, f"level-derived builder must beat the old 8; got {len(ep)}"
    assert (ep["entry_cause"] == "passive_drain").any(), \
        "drain-entered episodes must be present — the old builder dropped exactly these"


@needs_outputs
def test_evidence_classes_are_used():
    sc = pd.read_csv(OUT / "success_criteria.csv")
    assert set(sc["evidence_class"]) <= EVIDENCE_CLASSES
    assert "Supported" not in set(sc["evidence_class"])


@needs_outputs
def test_b1_b2_b10_are_implementation_verification():
    sc = pd.read_csv(OUT / "success_criteria.csv").set_index("id")
    assert sc.loc["RQ1-1", "evidence_class"] == "Implementation verification"
    assert sc.loc["RQ1-2", "evidence_class"] == "Implementation verification"
    assert sc.loc["RQ3-b", "evidence_class"] == "Implementation verification", \
        "affinity is a deterministic EMA; RQ3-b is not a general learning result"


@needs_outputs
def test_b4_and_b10_1_are_not_confirmatory():
    sc = pd.read_csv(OUT / "success_criteria.csv").set_index("id")
    assert sc.loc["RQ1-4", "evidence_class"] in ("Exploratory observation", "Inconclusive")
    assert sc.loc["RQ3-a", "evidence_class"] == "Exploratory observation", \
        "roles were not randomised and there are 2 people per role"


@needs_outputs
def test_zero_exposure_days_are_kept_and_not_faked():
    """A scheduled day with no attendance is a genuine zero. It must stay in the panel, and it
    must NOT be given a fake exposure of 1."""
    p = pd.read_csv(OUT / "b10_scheduled_day_panel.csv")
    assert (p["attended"] == 0).sum() > 0, "no-show days must be present"
    zero = p[p["n_interactions"] == 0]
    assert len(zero) > 0
    assert (zero["meal_count"] == 0).all()
    assert (zero["delivered_energy"] == 0).all()


@needs_outputs
def test_meal_count_and_energy_are_distinct():
    """A count of meals is not stomach points: meals are SMALL 10 / MEDIUM 25 / LARGE 45."""
    p = pd.read_csv(OUT / "b10_scheduled_day_panel.csv")
    assert {"meal_count", "delivered_energy"} <= set(p.columns)
    fed = p[p["meal_count"] > 0]
    if len(fed):
        assert not np.allclose(fed["meal_count"], fed["delivered_energy"]), \
            "meal_count and delivered_energy must not be the same number"


NEGATORS = ("no ", "not ", "never", "without", "nor ", "cannot", "n't")


def _affirms(text: str, phrase: str) -> list[str]:
    """Occurrences of `phrase` that are NOT disclaimed by a nearby negation.

    The point is to forbid the CLAIM, not the word. A report that says "this is not randomisation
    inference" is doing exactly what it should; one that says "survives randomisation" is not.
    """
    hits = []
    low = text.lower()
    start = 0
    while (i := low.find(phrase, start)) != -1:
        window = low[max(0, i - 90):i]
        if not any(n in window for n in NEGATORS):
            hits.append(text[max(0, i - 90):i + len(phrase) + 40].replace("\n", " "))
        start = i + len(phrase)
    return hits


@needs_outputs
def test_no_randomisation_claims_anywhere():
    """Roles were assigned by availability, so no conclusion may claim randomisation inference.

    Mentioning the phrase in order to DISCLAIM it is correct and must not fail; asserting it is
    what is forbidden.
    """
    surfaces = {f: (OUT / f).read_text() for f in ["results_summary.md", "success_criteria.csv"]}
    surfaces["README.md"] = (REPO / "README.md").read_text()
    for name, txt in surfaces.items():
        for banned in ("survives randomisation", "survives randomization"):
            assert banned not in txt.lower(), f"{name} claims '{banned}'"
        affirmed = _affirms(txt, "randomisation inference")
        assert not affirmed, (
            f"{name} asserts randomisation inference without disclaiming it "
            f"(roles were assigned by availability):\n  " + "\n  ".join(affirmed))


@needs_outputs
def test_dropcolumn_table_is_gone():
    assert not (OUT / "ml_dropcolumn_importance.csv").exists()


@needs_outputs
def test_no_duplicate_statistical_implementations():
    """The notebook must not redefine what statistical_helpers already provides."""
    src = (ANALYSIS / "build_notebook.py").read_text()
    for fn in ["def build_hs3_episodes", "def state_sequence", "def match_pings",
               "def control_windows", "def firth_logit", "def flapping", "def fit_ctmc",
               "def exact_prop_ci"]:
        assert fn not in src, f"{fn} is duplicated in the notebook instead of imported"
