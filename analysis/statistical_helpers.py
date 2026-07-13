"""Production statistical functions for the orexigenic-drive analysis.

This module is the single implementation of every non-trivial statistical or data-unit
routine in the analysis. The notebook imports it; the tests import it. Nothing is
reimplemented in either.

That matters more than it sounds. The previous test suite carried its *own* copies of the
episode builder, the ping matcher, the CTMC sojourn splitter and Firth — so the tests could
pass while the notebook shipped different code, and a bug in the notebook was invisible to
them. Worse, one of those duplicate copies (the flapping detector) contained the *same* bug
as the notebook, and the test that "verified" it could never have failed. Duplicated logic
does not test anything; it only tests itself.

Everything here is pure: it takes frames and returns frames or numbers, touches no globals,
and does no I/O.
"""
from __future__ import annotations

import warnings
from math import comb
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.linalg import null_space
from statsmodels.stats.proportion import proportion_confint

SEED = 42

# ---------------------------------------------------------------------------------------
# Evidence classes. A result carries exactly one. The generic word "Supported" is banned:
# it flattened an implementation check, a clustered association and a 13-interaction
# descriptive into one word, and that word did work it had not earned.
# ---------------------------------------------------------------------------------------
EV_IMPL = "Implementation verification"
EV_ASSOC = "Within-deployment association"
EV_EXPL = "Exploratory observation"
EV_INCONC = "Inconclusive"
EV_REPL = "Requires replication"
EVIDENCE_CLASSES = {EV_IMPL, EV_ASSOC, EV_EXPL, EV_INCONC, EV_REPL}

HS_FULL_MIN = 60.0
HS_STARVING_MAX = 25.0


# =======================================================================================
# Hunger state: derived from the LEVEL, never from the logged label fields
# =======================================================================================
def hs_from_level(level) -> np.ndarray:
    """Label from level, using the controller's own 60/25 constants.

    The executive logger writes `hunger_state_before` and `hunger_state_after` as the SAME
    value on a passive-drain crossing, so anything keyed on those columns is blind to every
    drain-driven crossing. The level is the software integrator and is ground truth; the
    label is a derived view of it.
    """
    x = pd.to_numeric(level, errors="coerce")
    return np.where(x >= HS_FULL_MIN, "HS1", np.where(x >= HS_STARVING_MAX, "HS2", "HS3"))


def build_hs_crossings(hr: pd.DataFrame) -> pd.DataFrame:
    """Every threshold crossing, derived from the level series, with its triggering cause."""
    hr = hr.sort_values(["run_id", "monotonic_sec", "id"]).copy()
    hr["level"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    hr["lab"] = hs_from_level(hr["level"])
    rows = []
    for run, g in hr.groupby("run_id"):
        g = g.reset_index(drop=True)
        lab = g["lab"].values
        for i in np.where(lab[1:] != lab[:-1])[0]:
            r = g.iloc[i + 1]
            rows.append(dict(
                run_id=run, day_rome=r["day_rome"], session_id=r.get("session_id"),
                id=int(r["id"]), timestamp_epoch=r["timestamp_epoch"],
                monotonic_sec=float(r["monotonic_sec"]),
                from_state=lab[i], to_state=lab[i + 1],
                event_type=r["event_type"], cause=r["stimulus_type"],
                stomach_level_before=float(g["level"].iloc[i]),
                stomach_level_after=float(r["level"]),
                level_delta=float(r["level"]) - float(g["level"].iloc[i]),
                exec_interaction_id=r.get("exec_interaction_id")))
    return pd.DataFrame(rows)


def build_hs3_episodes(hr: pd.DataFrame) -> pd.DataFrame:
    """Starving episodes from LEVEL crossings (level < 25), not from the logged labels.

    `entry_cause` records what pushed the robot under: `passive_drain` (nobody there) versus
    `interaction_cost` (someone was interacting). The old builder required a logged
    `before != HS3 -> after == HS3` row, which never occurs for a drain crossing, so it saw
    only the interaction-entered episodes — i.e. exactly the ones where a human was standing
    there to feed the robot. Its "8/8 recovered by feeding" was the selection rule restating
    itself.
    """
    hr = hr.sort_values(["run_id", "monotonic_sec", "id"]).copy()
    hr["level"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    eps = []
    for run, g in hr.groupby("run_id"):
        g = g.reset_index(drop=True)
        lvl, mono = g["level"].values, g["monotonic_sec"].values
        starving = lvl < HS_STARVING_MAX
        run_end = float(mono.max())
        i = 0
        while i < len(g):
            if not starving[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(g) and starving[j + 1]:
                j += 1
            entry_mono = float(mono[i])
            escaped = (j + 1) < len(g)
            escape_mono = float(mono[j + 1]) if escaped else np.nan
            escape_event = g["event_type"].iloc[j + 1] if escaped else None
            post = g.iloc[j + 1:] if escaped else g.iloc[0:0]
            full_hit = post[post["level"] >= HS_FULL_MIN]
            full_mono = float(full_hit["monotonic_sec"].iloc[0]) if len(full_hit) else np.nan
            full_event = full_hit["event_type"].iloc[0] if len(full_hit) else None
            censor = full_mono if np.isfinite(full_mono) else (escape_mono if escaped else run_end)
            win = g[(g["monotonic_sec"] >= entry_mono) & (g["monotonic_sec"] <= censor)]
            feeds = win[win["event_type"] == "feeding"].sort_values(["monotonic_sec", "id"])
            first_feed = float(feeds["monotonic_sec"].iloc[0]) if len(feeds) else np.nan
            eps.append(dict(
                episode_id=int(g["id"].iloc[i]), run_id=run, day_rome=g["day_rome"].iloc[i],
                entry_ts_epoch=g["timestamp_epoch"].iloc[i], entry_mono=entry_mono,
                entry_cause=("run_start" if i == 0 else g["stimulus_type"].iloc[i]),
                entry_level=float(lvl[i]), min_level=float(np.min(lvl[i:j + 1])),
                first_feed_mono=first_feed,
                time_to_first_feed_sec=(first_feed - entry_mono) if np.isfinite(first_feed) else np.nan,
                escape_mono=escape_mono,
                time_to_escape_starving_sec=(escape_mono - entry_mono) if escaped else np.nan,
                full_recovery_mono=full_mono,
                time_to_full_recovery_sec=(full_mono - entry_mono) if np.isfinite(full_mono) else np.nan,
                exit_mono=censor, episode_duration_sec=float(censor - entry_mono),
                hs3_duration_sec=float((escape_mono if escaped else run_end) - entry_mono),
                n_samples=int(j - i + 1),
                received_feed=int(len(feeds) > 0),
                escaped_starving=int(escaped),
                escaped_starving_by_feeding=int(escaped and escape_event == "feeding"),
                recovered_to_full=int(np.isfinite(full_mono)),
                recovered_to_full_by_feeding=int(np.isfinite(full_mono) and full_event == "feeding"),
                resolved_by_feeding=int(np.isfinite(full_mono) and full_event == "feeding"),
                censored_at_run_end=int(not escaped),
                meals_received=int(len(feeds)),
                total_active_energy_during_episode=float(
                    win[win["event_type"] == "active_cost"]["active_energy_cost"].sum()),
                exit_cause=("recovered_full_by_feeding" if (np.isfinite(full_mono) and full_event == "feeding")
                            else "recovered_full_nonfeeding" if np.isfinite(full_mono)
                            else "escaped_starving_by_feeding" if (escaped and escape_event == "feeding")
                            else "escaped_starving_nonfeeding" if escaped
                            else "censored_end_of_run")))
            i = j + 1
    return pd.DataFrame(eps)


def flapping(m: pd.DataFrame, window: float = 120.0) -> tuple[int, int, pd.DataFrame]:
    """Count crossings that UNDO the previous one within `window` seconds.

    A reversal requires `from_state == previous to_state` — you can only undo a crossing by
    going back the way you came. The original detector tested `from_state == previous
    from_state`, which is unsatisfiable, so it returned 0 on any data whatsoever and "the
    labels do not flap" was reported on that basis. They do flap.
    """
    fl, tot, rows = 0, 0, []
    for run, g in m.sort_values(["run_id", "monotonic_sec"]).groupby("run_id"):
        prev = None
        for _, r in g.iterrows():
            if prev is not None and (r.monotonic_sec - prev[0]) < window \
                    and r.from_state == prev[2] and r.to_state == prev[1]:
                fl += 1
                rows.append(dict(run_id=run, gap_sec=float(r.monotonic_sec - prev[0]),
                                 down_cause=prev[3], up_cause=r.cause))
            prev = (r.monotonic_sec, r.from_state, r.to_state, r.cause)
            tot += 1
    return fl, tot, pd.DataFrame(rows)


# =======================================================================================
# CTMC
# =======================================================================================
def state_sequence(hr: pd.DataFrame) -> pd.DataFrame:
    """Per-run sojourns INCLUDING the right-censored terminal segment.

    `to_state` is None on the terminal row: the run ended, nothing transitioned. Transition
    counts must therefore use `to_state.notna()`, while state-time denominators use ALL rows.
    That asymmetry is the whole point — the old code emitted segments only *between* changes,
    so every run's final dwell vanished from the denominators.
    """
    hr = hr.sort_values(["run_id", "monotonic_sec", "id"]).copy()
    hr["_lvl"] = hr["stomach_level_after"].fillna(hr["stomach_level_before"])
    hr = hr.dropna(subset=["_lvl", "monotonic_sec"])
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


def is_irreducible(counts: np.ndarray) -> bool:
    """Strong connectivity of the transition graph, by transitive closure.

    Irreducibility is a graph property and must be tested as one. Inferring it from the
    dimension of the null space of Q conflates "unique stationary distribution" with
    "irreducible": a reducible chain with a single absorbing class also has a
    one-dimensional null space, so the old check would have called it ergodic.
    """
    n = counts.shape[0]
    reach = (np.asarray(counts) > 0)
    np.fill_diagonal(reach, True)
    for k in range(n):                                  # Floyd-Warshall transitive closure
        reach = reach | (reach[:, [k]] & reach[[k], :])
    return bool(reach.all())


def fit_ctmc(seq: pd.DataFrame, states: Sequence[str] = ("HS1", "HS2", "HS3"),
             tol: float = 1e-8) -> dict:
    """Fit the generator and VALIDATE the stationary vector rather than assuming it.

    Returns a dict with the generator, counts, state-time, and:
      irreducible                   - strong connectivity of the transition graph
      unique_stationary_distribution- null space of Q' is exactly one-dimensional
      stationary_valid              - pi >= 0, sums to 1, and pi @ Q ~ 0 within `tol`

    The old code took `abs()` of an arbitrary null-space basis vector and normalised it. That
    is not a validation, it is a coercion: it will happily return a "distribution" from a
    vector with mixed signs, which is exactly the signature of a chain that has no stationary
    distribution at all.
    """
    states = list(states)
    time_in = seq.groupby("state")["dwell"].sum().reindex(states).fillna(0.0)
    tr = seq[seq["to_state"].notna()]
    cnt = (tr.groupby(["state", "to_state"]).size().unstack(fill_value=0)
             .reindex(index=states, columns=states, fill_value=0))
    Q = np.zeros((len(states), len(states)))
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i != j and time_in[si] > 0:
                Q[i, j] = cnt.loc[si, sj] / time_in[si]
        Q[i, i] = -Q[i].sum()

    out = dict(Q=Q, counts=cnt, time_in=time_in, pi=None,
               irreducible=is_irreducible(cnt.values),
               unique_stationary_distribution=False, stationary_valid=False,
               reason=None)
    try:
        ns = null_space(Q.T)
    except Exception as e:                                        # pragma: no cover
        out["reason"] = f"null_space failed: {e}"
        return out
    if ns.shape[1] != 1:
        out["reason"] = f"null space of Q' has dimension {ns.shape[1]}, not 1"
        return out
    out["unique_stationary_distribution"] = True

    v = ns[:, 0]
    # A valid stationary vector is single-signed. Mixed signs mean there is no stationary
    # distribution here, and abs() would have hidden that.
    if not (np.all(v >= -tol) or np.all(v <= tol)):
        out["reason"] = "null-space vector has mixed signs: no stationary distribution"
        return out
    v = np.abs(v)
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        out["reason"] = "null-space vector does not normalise"
        return out
    pi = v / s
    resid = float(np.max(np.abs(pi @ Q)))
    if np.any(pi < -tol) or abs(pi.sum() - 1.0) > 1e-9 or resid > tol:
        out["reason"] = f"stationary vector failed validation (max|pi@Q| = {resid:.2e})"
        return out
    out["pi"] = pi
    out["stationary_valid"] = True
    out["residual"] = resid
    return out


# =======================================================================================
# Ping <-> reply matching
# =======================================================================================
def match_pings(ev: pd.DataFrame, ping_types: Sequence[str], window: float) -> pd.DataFrame:
    """REPLY-CENTRIC one-to-one matching: each reply answers the NEAREST eligible preceding ping.

    The distinction from ping-centric greedy matching is not cosmetic. Walking pings forward
    and giving each the first unconsumed reply assigns a reply to the *earliest* ping still
    looking for one, which can be an hour older than a ping sent moments before the reply
    arrived. Walking replies backward assigns each reply to the ping it plausibly answers.
    Both are one-to-one; only one is defensible.

    A reply is used at most once. `replied` is therefore <= 1 per ping by construction.
    """
    out = []
    for chat, g in ev.groupby("chat_id"):
        pings = (g[g["event_type"].isin(ping_types)]
                 .sort_values("timestamp_epoch").reset_index(drop=True))
        msgs = g[g["event_type"] == "user_message"].sort_values("timestamp_epoch")
        matched: dict[int, dict] = {}
        for _, m in msgs.iterrows():                  # replies, in time order
            t = m["timestamp_epoch"]
            cand = pings[(pings["timestamp_epoch"] < t)
                         & (pings["timestamp_epoch"] >= t - window)
                         & (~pings.index.isin(matched))]
            if not len(cand):
                continue
            k = int(cand["timestamp_epoch"].idxmax())  # NEAREST preceding, not earliest
            matched[k] = dict(reply_id=int(m["id"]), latency=float(t - pings.loc[k, "timestamp_epoch"]))
        for k, p in pings.iterrows():
            hit = matched.get(int(k))
            out.append(dict(
                chat_id=chat, run_id=p["run_id"], day_rome=p["day_rome"],
                ping_id=int(p["id"]), ping_type=p["event_type"], hs=p["hs"],
                t=float(p["timestamp_epoch"]),
                replied=int(hit is not None),
                matched_reply=hit["reply_id"] if hit else None,
                latency_sec=hit["latency"] if hit else np.nan))
    return pd.DataFrame(out)


def assign_run_by_time(t: Iterable[float], run_spans: pd.DataFrame) -> list:
    """Map each timestamp to the run whose [t_start, t_end] contains it.

    NECESSARY because the four SQLite databases do NOT share a run_id namespace — the chat and
    executive `run_id` sets have ZERO overlap. Joining pings to runs on `run_id` silently matches
    nothing. Wall-clock containment is the pipeline's documented cross-stream key, and it is the
    only correct one here.
    """
    spans = [(float(r.t_start), float(r.t_end), r.run_id) for r in run_spans.itertuples()]
    out = []
    for ts in t:
        hit = None
        for a, b, rid in spans:
            if a <= ts <= b:
                hit = rid
                break
        out.append(hit)
    return out


def control_windows(ev: pd.DataFrame, pm: pd.DataFrame, window: float,
                    run_spans: pd.DataFrame, *, run_col: str = "exec_run_id",
                    tod_caliper: float = 2 * 3600.0, seed: int = SEED) -> pd.DataFrame:
    """One matched control window per ping: same subscriber, same run, similar time-of-day.

    Every constraint here exists because relaxing it manufactures a result:

    * **Same run.** Controls drawn from the whole calendar land at night and between sessions,
      when the robot was off and nobody was going to message it. That put the control at
      ~1/172 and roughly doubled the apparent ping effect.
    * **Time-of-day caliper.** People message at lunchtime, not at 07:00. Without a caliper the
      control is drawn from hours the subscriber was never going to reply in.
    * **Ping-free and non-overlapping.** A control window containing a ping is not a control,
      and two overlapping controls double-count the same stretch of time.
    * **A reply satisfies at most one control window**, exactly as for pings — otherwise the
      control rate is inflated by the same double-counting the ping side was fixed for.

    `run_col` must name a column of pm holding the EXECUTIVE run id (see assign_run_by_time):
    the chat DB's own run_id lives in a different namespace and will match nothing.
    """
    rng = np.random.default_rng(seed)
    spans = {r.run_id: (float(r.t_start), float(r.t_end)) for r in run_spans.itertuples()}
    rows = []
    for chat, pg in pm.groupby("chat_id"):
        msgs = (ev[(ev["chat_id"] == chat) & (ev["event_type"] == "user_message")]
                .sort_values("timestamp_epoch"))
        ping_t = pg["t"].values
        used_replies: set[int] = set()
        placed: list[tuple[float, float]] = []       # accepted control windows, for overlap
        for p in pg.itertuples():
            rid = getattr(p, run_col, None)
            span = spans.get(rid)
            if span is None:
                # The ping fell outside every monitored run (robot off): there is no window in
                # which a control could legitimately be drawn, so it is recorded as unmatched
                # rather than silently dropped.
                rows.append(dict(chat_id=chat, run_id=rid, ping_id=p.ping_id,
                                 t=np.nan, replied=np.nan, matched=False))
                continue
            lo, hi = span[0], span[1] - window
            if hi <= lo:
                rows.append(dict(chat_id=chat, run_id=rid, ping_id=p.ping_id,
                                 t=np.nan, replied=np.nan, matched=False))
                continue
            tod = p.t % 86400.0                       # ping's time-of-day
            t0 = None
            for _ in range(400):
                cand = float(rng.uniform(lo, hi))
                if abs(((cand % 86400.0) - tod + 43200.0) % 86400.0 - 43200.0) > tod_caliper:
                    continue                          # outside the time-of-day caliper
                if np.any((ping_t >= cand - window) & (ping_t <= cand + window)):
                    continue                          # must be ping-free
                if any(not (cand + window <= a or cand >= b) for a, b in placed):
                    continue                          # must not overlap another control
                t0 = cand
                break
            if t0 is None:
                rows.append(dict(chat_id=chat, run_id=rid, ping_id=p.ping_id,
                                 t=np.nan, replied=np.nan, matched=False))
                continue
            placed.append((t0, t0 + window))
            cand_msgs = msgs[(msgs["timestamp_epoch"] > t0)
                             & (msgs["timestamp_epoch"] <= t0 + window)
                             & (~msgs["id"].isin(used_replies))]
            hit = len(cand_msgs) > 0
            if hit:
                used_replies.add(int(cand_msgs["id"].iloc[0]))   # one reply, one window
            rows.append(dict(chat_id=chat, run_id=rid, ping_id=p.ping_id,
                             t=t0, replied=int(hit), matched=True))
    return pd.DataFrame(rows, columns=["chat_id","run_id","ping_id","t","replied","matched"])


# =======================================================================================
# Intervals, bootstraps, permutation
# =======================================================================================
def exact_prop_ci(k: int, n: int) -> tuple[float, float, float]:
    """Clopper-Pearson. The right tool for 0/n and n/n cells, where Wald is meaningless."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = proportion_confint(int(k), int(n), method="beta")
    return (k / n, float(lo), float(hi))


def boot_ci(x, fn=np.mean, n=5000, alpha=0.05, seed=SEED):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = np.array([fn(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return (fn(x), np.nanpercentile(stats, 100 * alpha / 2), np.nanpercentile(stats, 100 * (1 - alpha / 2)))


def boot_diff_ci(a, b, fn=np.mean, n=5000, alpha=0.05, seed=SEED):
    a = np.asarray(pd.Series(a).dropna(), float)
    b = np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = np.array([fn(rng.choice(a, len(a), True)) - fn(rng.choice(b, len(b), True)) for _ in range(n)])
    return (fn(a) - fn(b), np.nanpercentile(d, 100 * alpha / 2), np.nanpercentile(d, 100 * (1 - alpha / 2)))


def cluster_bootstrap(df: pd.DataFrame, cluster_col: str, fit_fn: Callable,
                      *, n: int = 1000, seed: int = SEED, label: str = "effect",
                      verbose: bool = True) -> dict:
    """Resample whole clusters and refit. THIS is the interval we lead with.

    At 12-15 clusters the GEE sandwich SE is anti-conservative, so the asymptotic CI is
    reported second. The refit-failure count is returned too: a high failure rate is itself
    diagnostic of separation or a degenerate design cell, and silently dropping those refits
    conditions the interval on the effect being estimable.
    """
    rng = np.random.default_rng(seed)
    clusters = pd.Series(df[cluster_col].dropna().unique())
    vals, n_fail = [], 0
    for _ in range(n):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for i, c in enumerate(picked):
            g = df[df[cluster_col] == c].copy()
            g[cluster_col] = f"{c}__boot{i}"
            parts.append(g)
        sample = pd.concat(parts, ignore_index=True)
        try:
            v = fit_fn(sample)
            if np.isfinite(v):
                vals.append(float(v))
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 25:
        if verbose:
            print(f"{label}: cluster bootstrap UNSTABLE ({len(vals)}/{n} refits succeeded)")
        return dict(lo=np.nan, median=np.nan, hi=np.nan, n_ok=len(vals), n_fail=n_fail)
    lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
    if verbose:
        print(f"{label}: cluster bootstrap median {mid:.3g} [95% {lo:.3g}, {hi:.3g}] "
              f"({len(vals)}/{n} refits, {n_fail} failed)")
    return dict(lo=float(lo), median=float(mid), hi=float(hi), n_ok=len(vals), n_fail=n_fail)


def enumerate_label_assignments(cluster_ids: Sequence, labels: Sequence) -> list[dict]:
    """EVERY distinct assignment of the observed labels to the observed clusters.

    With 2 treated among 12 there are C(12,2) = 66 assignments. Approximating a 66-point
    discrete distribution with 5,000 random draws adds Monte-Carlo error for nothing. Enumerate.
    """
    from itertools import combinations
    ids = list(cluster_ids)
    labs = list(labels)
    uniq = sorted(set(labs))
    if len(uniq) != 2:
        raise ValueError(f"exact enumeration supports 2 labels, got {uniq}")
    treat = [l for l in labs if l == uniq[0]]
    k = len(treat)
    out = []
    for chosen in combinations(range(len(ids)), k):
        m = {ids[i]: (uniq[0] if i in chosen else uniq[1]) for i in range(len(ids))}
        out.append(m)
    return out


def exact_permutation_p(df: pd.DataFrame, cluster_col: str, label_col: str,
                        stat_fn: Callable) -> dict:
    """Two-sided permutation p over the EXACT enumeration of person-level label assignments.

    NOTE ON INTERPRETATION. This is only *randomisation inference* if the labels were actually
    randomised. In this study they were not — roles were assigned by participant availability —
    so the permutation distribution does not have a design justification and this is a
    LABEL-PERMUTATION SENSITIVITY: it asks how unusual the observed split is among all ways the
    same labels could have fallen on the same people. It does not license a causal or
    population claim, and the caller must not describe it as randomisation inference.
    """
    cl = df[[cluster_col, label_col]].drop_duplicates().reset_index(drop=True)
    obs = stat_fn(df)
    assigns = enumerate_label_assignments(cl[cluster_col].tolist(), cl[label_col].tolist())
    null = []
    for m in assigns:
        d2 = df.copy()
        d2[label_col] = d2[cluster_col].map(m)
        try:
            v = stat_fn(d2)
            if np.isfinite(v):
                null.append(float(v))
        except Exception:
            pass
    null = np.asarray(null, float)
    if not np.isfinite(obs) or len(null) < 2:
        return dict(observed=obs, p=np.nan, n_assignments=len(null), floor=np.nan)
    centre = np.median(null)
    p = float(np.mean(np.abs(null - centre) >= np.abs(obs - centre) - 1e-12))
    return dict(observed=float(obs), p=p, n_assignments=len(null),
                floor=1.0 / len(null), null=null)


# =======================================================================================
# Firth penalised logistic (for quasi-separation)
# =======================================================================================
def firth_logit(X, y, max_iter: int = 200, tol: float = 1e-8):
    """Penalised (Jeffreys-prior) logistic MLE. Finite under separation, where plain ML is not.

    Under quasi-separation the ordinary MLE runs to +/-inf and the Wald SE collapses with it,
    so their ratio manufactures a tiny p-value out of a cell containing almost no information.
    B4 (1 success in 13) is exactly that shape.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        inv = np.linalg.pinv(X.T @ (X * W[:, None]))
        H = (X * W[:, None]) @ inv
        h = np.einsum("ij,ij->i", H, X)
        step = inv @ (X.T @ (y - p + h * (0.5 - p)))
        nb = beta + step
        if not np.all(np.isfinite(nb)):
            break
        if np.max(np.abs(nb - beta)) < tol:
            beta = nb
            break
        beta = nb
    return beta


def _firth_profile_ll(X, y, idx, b_fixed):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    keep = [j for j in range(X.shape[1]) if j != idx]
    off = X[:, idx] * b_fixed
    Xr = X[:, keep]
    bb = np.zeros(Xr.shape[1])
    for _ in range(200):
        eta = Xr @ bb + off
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        inv = np.linalg.pinv(Xr.T @ (Xr * W[:, None]))
        H = (Xr * W[:, None]) @ inv
        h = np.einsum("ij,ij->i", H, Xr)
        nb = bb + inv @ (Xr.T @ (y - p + h * (0.5 - p)))
        if not np.all(np.isfinite(nb)):
            break
        if np.max(np.abs(nb - bb)) < 1e-8:
            bb = nb
            break
        bb = nb
    eta = Xr @ bb + off
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    W = p * (1 - p)
    Xf = np.column_stack([Xr, X[:, idx]])
    sign, logdet = np.linalg.slogdet(Xf.T @ (Xf * W[:, None]))
    return ll + 0.5 * logdet if sign > 0 else ll


def firth_profile_ci(X, y, idx, alpha: float = 0.05, grid: int = 241, span: float = 8.0):
    """Profile penalised-likelihood CI. Wald is meaningless under separation; the profile is not."""
    beta = firth_logit(X, y)
    b0 = float(beta[idx])
    ll_max = _firth_profile_ll(X, y, idx, b0)
    crit = sps.chi2.ppf(1 - alpha, 1) / 2.0
    lo = np.nan
    for b in np.linspace(b0 - span, b0, grid):
        if ll_max - _firth_profile_ll(X, y, idx, b) <= crit:
            lo = b
            break
    hi = np.nan
    for b in np.linspace(b0, b0 + span, grid)[::-1]:
        if ll_max - _firth_profile_ll(X, y, idx, b) <= crit:
            hi = b
            break
    lr = 2.0 * (ll_max - _firth_profile_ll(X, y, idx, 0.0))
    return b0, float(lo), float(hi), float(sps.chi2.sf(max(lr, 0.0), 1))


# =======================================================================================
# Model fitting with EXPLICIT failure reporting
# =======================================================================================
def fit_gee_checked(formula: str, data: pd.DataFrame, groups: str, family,
                    cov_struct=None, offset=None, focal: str | None = None) -> dict:
    """Fit a GEE and REPORT its status instead of swallowing failures.

    A model that failed to converge, produced an infinite estimate, or dropped the focal term
    entirely must not be silently caught and then cited as evidence that a result "survives
    adjustment". Every failure mode is named and returned.
    """
    res = dict(formula=formula, cluster=groups, n=len(data), status="ok", reason=None,
               converged=None, estimate=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
               cov_struct="exchangeable")
    if len(data) == 0:
        res.update(status="failed", reason="empty dataset")
        return res

    def _fit(cs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return smf.gee(formula, groups=groups, data=data, family=family,
                           cov_struct=cs, offset=offset).fit()

    # The exchangeable working correlation is the default, but it destabilises when a few
    # clusters carry hundreds of rows each (it must estimate one within-cluster correlation from
    # very unbalanced blocks) and can return a non-finite estimate. Independence is a valid
    # working correlation — the sandwich SE remains consistent either way — so we fall back to it
    # and RECORD that we did, rather than reporting a blank or silently swapping models.
    tried = cov_struct or sm.cov_struct.Exchangeable()
    try:
        m = _fit(tried)
    except Exception as e:
        res.update(status="failed", reason=f"fit raised: {type(e).__name__}: {e}")
        return res

    def _bad(mm):
        if focal is None or focal not in mm.params.index:
            return focal is not None
        return not np.isfinite(float(mm.params[focal]))

    if _bad(m) and cov_struct is None:
        try:
            m2 = _fit(sm.cov_struct.Independence())
            if not _bad(m2):
                m = m2
                res["cov_struct"] = "independence (exchangeable gave a non-finite estimate)"
        except Exception:
            pass

    res["converged"] = bool(getattr(m, "converged", True))
    if focal is None:
        return res
    if focal not in m.params.index:
        res.update(status="failed", reason=f"focal term '{focal}' absent from the design "
                                           f"(aliased or dropped)")
        return res
    b = float(m.params[focal])
    se = float(m.bse[focal]) if focal in m.bse.index else np.nan
    ci = m.conf_int().loc[focal].tolist()
    res.update(estimate=b, lo=float(ci[0]), hi=float(ci[1]), p=float(m.pvalues[focal]))
    if not np.isfinite(b):
        res.update(status="failed", reason="non-finite estimate")
    elif not np.isfinite(se) or se <= 0:
        res.update(status="failed", reason="non-finite or zero standard error")
    elif abs(b) > 15:
        # exp(15) ~ 3.3e6: an odds ratio this large on a binary predictor is the signature of
        # separation, not of an effect.
        res.update(status="separation", reason=f"|coef| = {abs(b):.1f} indicates separation")
    elif not res["converged"]:
        res.update(status="not_converged", reason="GEE did not converge")
    return res


def adjustment_verdict(models: Iterable[dict], focal_sign: float) -> dict:
    """Decide whether a result may be said to 'survive adjustment'. It may not, by default.

    The claim requires that EVERY prespecified adjusted model produced a finite focal estimate
    AND that all of them point the same way. A model that failed, hit separation, or dropped the
    focal term does not get to be quietly excluded from the tally.
    """
    ms = list(models)
    bad = [m for m in ms if m["status"] != "ok" or not np.isfinite(m["estimate"])]
    signs = [np.sign(m["estimate"] - 0.0) for m in ms if m["status"] == "ok" and np.isfinite(m["estimate"])]
    consistent = bool(signs) and all(s == focal_sign for s in signs)
    survives = (not bad) and consistent and len(ms) > 0
    return dict(
        survives=survives,
        n_models=len(ms), n_failed=len(bad),
        directionally_consistent=consistent,
        reason=("all prespecified adjusted models converged with finite, directionally "
                "consistent estimates" if survives
                else "; ".join(filter(None, [
                    f"{len(bad)} model(s) failed: " + "; ".join(
                        f"[{m['cluster']}] {m['status']}: {m['reason']}" for m in bad) if bad else "",
                    "" if consistent else "focal estimates are not directionally consistent",
                ]))))


# =======================================================================================
# Missingness / IPW diagnostics
# =======================================================================================
def ipw_diagnostics(p_obs: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> dict:
    """Everything a reader needs to judge whether an IPW estimate is trustworthy.

    Reporting a weighted coefficient without the predicted-probability range, the weight range,
    and the effective sample size is asking to be believed rather than checked. A propensity
    near 0 or 1 is a positivity violation: some stratum is effectively never observed, and no
    weight can recover information that was never collected.
    """
    p_obs = np.asarray(p_obs, float)
    w = np.asarray(weights, float)
    obs = np.asarray(observed).astype(bool)
    w_obs = w[obs]
    ess = (w_obs.sum() ** 2) / np.sum(w_obs ** 2) if len(w_obs) and np.sum(w_obs ** 2) > 0 else np.nan
    return dict(
        p_min=float(np.nanmin(p_obs)), p_max=float(np.nanmax(p_obs)),
        w_min=float(np.nanmin(w_obs)) if len(w_obs) else np.nan,
        w_max=float(np.nanmax(w_obs)) if len(w_obs) else np.nan,
        w_mean=float(np.nanmean(w_obs)) if len(w_obs) else np.nan,
        ess=float(ess), n_observed=int(obs.sum()), n_total=int(len(obs)),
        ess_frac=float(ess / obs.sum()) if obs.sum() else np.nan,
        positivity_violation=bool(np.nanmin(p_obs) < 0.05 or np.nanmax(p_obs) > 0.95),
    )


def spec_agreement(estimates: dict[str, float], *, rel_tol: float = 0.5) -> dict:
    """Do specifications agree in SIGN and MAGNITUDE, not just sign?

    "All three doses agree in sign" was offered as corroboration when one of the three was a
    literal additive term in the outcome's definition. Sign agreement is a very weak property:
    +0.041 and +0.168 agree in sign while differing four-fold, and reporting them as
    "consistent" would hide the entire finding.
    """
    vals = {k: v for k, v in estimates.items() if np.isfinite(v)}
    if len(vals) < 2:
        return dict(sign_agree=False, magnitude_agree=False, reason="fewer than 2 finite estimates")
    signs = {np.sign(v) for v in vals.values()}
    sign_agree = len(signs) == 1
    lo, hi = min(map(abs, vals.values())), max(map(abs, vals.values()))
    ratio = hi / lo if lo > 0 else np.inf
    magnitude_agree = bool(ratio <= 1.0 / rel_tol) if rel_tol > 0 else False
    return dict(sign_agree=bool(sign_agree), magnitude_agree=magnitude_agree,
                max_ratio=float(ratio), estimates=vals,
                reason=("sign and magnitude agree" if sign_agree and magnitude_agree
                        else "signs agree but magnitudes differ by "
                             f"{ratio:.1f}x (tolerance {1/rel_tol:.1f}x)" if sign_agree
                        else "signs disagree"))
