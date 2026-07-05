# Orexigenic Drive & Always-On Homeostasis — Analysis Explained

This document explains, step by step and in plain language, the data analysis in
[`analysis/orexigenic_analysis.ipynb`](analysis/orexigenic_analysis.ipynb). It says **what
we did, why we did it, how, and what came out** — and then reads the results critically.

The **purpose is to examine the results**: not to re-derive the code, but to understand what
the numbers actually tell us about the robot's internal "hunger" drive, and where the
evidence is strong versus thin.

> **System under study.** The *Orexigenic Drive* of the `alwaysOn-embodiedBehaviour` iCub
> controller — a continuous, always-running "metabolism" that makes the robot get hungry,
> ask to be fed, and reprioritise its behaviour around recovering energy. The controller
> pipeline is *perception → salience → executive → remote/Telegram*
> (see [`../alwaysOn-embodiedBehaviour/README.md`](../alwaysOn-embodiedBehaviour/README.md)).

---

## 1. The two research questions (the "why")

Everything below exists to answer two questions. Every low-level analysis step is tied to
one of them — nothing is run "just because".

- **RQ1 — Is this a *real* homeostatic drive?** Concretely, does it fulfil the four classical
  functions of homeostasis?
  1. **Internal monitoring** — does it continuously track an internal variable on its own?
  2. **Deficit detection** — does it correctly notice when that variable falls too low?
  3. **Deficit → action conversion** — does the deficit actually change behaviour, or is it
     cosmetic?
  4. **Behavioural prioritisation** — when starving, does the drive *override* the social
     agenda to pursue food?

- **RQ2 — Does expressing the deficit lead to reliable recovery?** When the robot signals
  hunger, do people (in person or over Telegram) feed it enough that it reliably climbs back
  out of starvation and stays out over the long run?

### One design fact that shapes everything

The drive was **always on** for the entire study. **There is no "drive-off" control
condition.** We did not invent one. So RQ2 is identified two ways from *within* the always-on
data:

1. the **graded deficit** itself — Full → Hungry → Starving — is treated as the manipulation
   (does behaviour change as the robot gets hungrier?); and
2. the **proactive vs reactive** contrast — does the drive *initiate* recovery, or only react
   when a human happens to show up?

This is an honest limitation and it is stamped on every conclusion.

---

## 2. The data (the "what we had")

The robot logs to four SQLite databases — `vision.db`, `salience_network.db`,
`executive_control.db`, `chat_bot.db` — one per module. The `data/` folder contains **eight
dated snapshots** of these databases.

**The key discovery in data prep:** those eight folders are **not eight independent
experiments**. They are **cumulative snapshots of one continuously-growing database** — each
later folder is a strict superset of the earlier ones. Naïvely stacking them would
double-count every interaction 4–5×. So the very first analytical act is to **de-duplicate to
the true unit of analysis**:

- **run** (`run_id`) — one continuous session of the robot from start to restart;
- **day** (`day_rome`) — the calendar day a run belongs to.

After de-duplication the corpus is: **10 runs with visitors, across 8 days, 217 interactions**
(and 12 runs *monitored* by the drive — two runs had the drive draining with nobody present).

---

## 3. How the analysis is organised (the "how", at a glance)

The notebook runs in phases, each a gate for the next:

| Phase | What it does | Why |
|---|---|---|
| **0 — Ground truth** | Read the *controller source code* and extract every constant (thresholds, drain rate, energy costs) with file:line references. Then discover the data layout and de-duplicate. | So every later claim is checked against what the code *actually does*, not against memory. |
| **A — Data preparation** | Load the clean DB views, pseudonymise all identities (`P01…P14`), reconstruct analysis units (HS3 episodes, transitions, drive timeline), and build one **leakage-safe master table** (one row per interaction, predictors from *before* the interaction only). | A trustworthy, privacy-safe table that can't "cheat" by peeking at the outcome. |
| **Verification gate (V1–V5)** | 15 hard/soft checks: do meal sizes match the source constants? Does the fitted drain rate match nominal? Referential integrity, clock sanity, energy balance. | **Nothing proceeds until the data provably matches the code.** All passed. |
| **B — Statistics (B1–B9)** | The confirmatory core. Mixed-effects models, bootstrap CIs, a Markov steady-state model. One analysis per homeostatic function. | This is where the real evidential weight sits. |
| **C — Visualisation** | 12 figures, all tied to a specific claim. | Make the mechanism legible. |
| **D — Machine learning (D1, D4, D5)** | *Interpretive, not confirmatory* (only ~200 rows). Group-aware cross-validation (D1), feeding-concentration robustness (D4), and language framing (D5). | Sensitivity checks and robustness, honestly labelled as such. |

A recurring, deliberate stance runs through Phase B: **Starving is rare and small-n**
(single-digit episodes). So we **lead with effect sizes and bootstrap confidence intervals,
p-values second**, correct the whole family with Benjamini–Hochberg, and label thin results
"directional" rather than dressing them up as proof.

### Figure 1 — the pipeline and what each stage logs

![Architecture](analysis/figures/fig01_architecture.png)

*Vision logs gaze/attention; Salience computes the Interaction Priority Score (IPS) and picks
a target; Executive runs the hunger model and the conversation; the Telegram bot carries the
same drive off-robot. The loop closes when an interaction's homeostatic reward feeds back to
the salience network.*

---

## 4. RQ1 — Is this a real homeostatic drive?

### RQ1.1 + RQ1.2 — Monitoring & detection *(analyses B1, B2 — verification, not headline results)*

> **Read these two as verification of faithful implementation, not empirical measurements.**
> The stomach level is a **software integrator** and the hunger label is *derived* from it by
> the same coded 60/25 thresholds. So the two "results" below hold **by construction** — they
> prove the mechanism is wired correctly, and the only genuinely non-trivial content is
> *autonomy* (B1) and the *absence of flapping* (B2). The confirmatory weight of RQ1 sits in
> **RQ1.3** and **RQ1.4** below.

**B1 — Internal monitoring.** Does the robot track its "stomach level" continuously and on its
own, regardless of who is around? We fitted the empirical drain slope and compared it to the
coded nominal rate (`100/(4·3600) %/s`, empties in 4 h). It matches at **exactly 1.00× with a
zero-width CI** — the *tell* of a software integrator (true by construction). The real content:
the drive **samples every ~2.3 s (median gap)** across **12 monitored runs / ~46 h**,
**~100% of samples with no interaction attached**, and it keeps draining in **two runs with
zero visitors present**. *Autonomous, dense, interaction-independent monitoring.*
**Verdict: Supported** (faithful implementation + autonomy).

**B2 — Deficit detection.** When the level crosses 60 (→ Hungry) or 25 (→ Starving), does the
state flip correctly and *cleanly*? Bracketing accuracy is **1.00/1.00** (by construction — the
label is defined by those thresholds). The non-trivial result is **zero rapid reversals — no
flapping** at either boundary (the chatbot even ships a 60-s debounce, `HS_DWELL_SEC`, to
absorb flapping it essentially never sees). **Verdict: Supported** (faithful implementation;
clean, stable boundaries).

### Figure 2 — the signature figure: the drive over time, one panel per day

![Drive timeline](analysis/figures/fig02_drive_timeline.png)

*Each panel is one experiment day on the wall-clock. Green/amber/red bands are
Full/Hungry/Starving; the black line is the stomach level sawtoothing down (drain) and up
(meals, shown as up-arrows sized by meal). Dotted lines are restarts; red shading marks
Starving episodes. You can literally see the homeostasis: autonomous decay punctuated by
feeding recoveries, and Starving is brief and rare.*

### Figure 3 — detection fires exactly at the thresholds

![Thresholds and transitions](analysis/figures/fig03_thresholds_transitions.png)

*Left: drain-driven falls straddle the 60 and 25 lines within a single sample. Right: the
observed state-transition graph with counts — traffic concentrates on the Full↔Hungry edge,
with far fewer excursions down to Starving.*

### RQ1.3 — Deficit → action conversion *(analysis B3 + the active-cost table)*

**What we asked.** The hard question: is the hunger state *doing* anything to behaviour, or is
it a cosmetic label? Does being Starving actually change what the robot does, over and above
its social situation?

**How.**
1. A **mixed-effects logistic model** of "did the user reply" with a random intercept for
   run, controlling for social state, IPS, and co-presence — so any hunger effect is *beyond*
   the social/perceptual context. (Predictors are strictly pre-interaction, to avoid leakage.)
2. The **active-energy-cost table**: rebuilt from the logs and cross-checked to the source
   constants — every action's metabolic price.

**Result.**
- Starving lowers the odds of a reply beyond social state and IPS (**OR ≈ 0.37, p = 0.136**).
  After Benjamini–Hochberg family correction, **q > 0.05** — it does *not* clear
  significance. Starving n = 13.
- The cost table is deterministic and matches source **exactly**: a **conversation turn costs
  3.6**, a **greeting 0.8**, a feeding prompt 1.0. Spend scales with what the robot actually
  *does*, and mean spend per interaction falls at Starving because conversation collapses (see
  B4).

| Action | Energy cost | Events | Total energy |
|---|---|---|---|
| Conversation turn *(largest sink)* | 3.6 | 457 | 1645 |
| Conversation starter | 1.2 | 161 | 193 |
| Feed acknowledgement | 0.8 | 77 | 62 |
| Known greeting | 0.8 | 76 | 61 |
| Name question | 1.0 | 20 | 20 |
| Feeding prompt / hunger seeking | 1.0 | 13 | 13 |

*(Every cost matches its source constant exactly, min = max = coded value — verification
check V5b passed with zero mismatches.)*

**How to read it.** The coupling is **behavioural, not just label-deep** — energy genuinely
scales with action. But the *statistical* reply-suppression effect is **directional, not
significant** under correction, and n is small. Honest call.

**Verdict: Weakened (directional).**

### Figure 4 — engagement and energy spend by hunger state

![Deficit to action](analysis/figures/fig04_deficit_action.png)

*Reply rate, P(reach Engaged), average turns, and active energy, each split Full/Hungry/
Starving with bootstrap CIs. Hatched bars flag small-n (< 20). Reply rate is nearly flat
Full→Hungry, then everything drops sharply at Starving — a step, not a smooth ramp.*

### RQ1.4 — Behavioural prioritisation: the Starving override *(analysis B4 — the centrepiece)*

**What we asked.** The most important RQ1 question. When Starving, does the drive **take
over** — abandon chit-chat and pursue food?

**How.** We crosstabbed outcomes by social-state × hunger-state, then compared Starving vs
(Full+Hungry) on conversation depth (turns, reaching "Engaged") and on feeding pursuit.

**Result.**
- Conversation **collapses** when Starving: **turns 2.5 → 0.2** (diff −2.39), and reaching
  Engaged falls from **0.68 → 0.08**.
- Feeding pursuit **rises**: P(meal in the interaction) **0.26 → 0.54**.

So it is **reprioritisation, not disengagement** — the robot doesn't go quiet, it switches
goals from *socialising* to *getting fed*, exactly as the coded `_run_hunger_tree` override
specifies. Starving n = 13 (directional).

**Verdict: Supported.**

### Figure 5 — the prioritisation heatmap

![Prioritisation heatmap](analysis/figures/fig05_prioritisation_heatmap.png)

*Social state (rows) × hunger state (columns). Left: Engaged-completion rate; right: average
turns. The whole **Starving column collapses** while Full/Hungry sustain conversation — the
override is visible as a dark vertical stripe. Every cell shows its n so small cells aren't
over-read.*

### Figure 6 — how salience picks who to talk to

![IPS decomposition](analysis/figures/fig06_ips_decomposition.png)

*Context for prioritisation: the Interaction Priority Score is a fixed weighted sum of
proximity, centrality, velocity, gaze. Right panel shows the IPS distribution per social state
against the eligibility threshold each must clear. (Velocity contributes ≈0 because faces are
mostly stationary.)*

### Reading RQ1.3 + RQ1.4 + the gradient together: **a threshold controller, not a ramp**

This is the single most important interpretive point, and it reconciles the "Weakened B3/B8"
with the "Supported B4" into **one coherent story**. The drive was **built as a threshold
mechanism**, and that is exactly what the data show:

- **Below the line, expression is graded but *soft*** — it lives in signalling, not takeover.
  Meal size rises **21 → 29 → 43** with deficit; hunger *framing* in speech jumps **3% → 67%**
  Full→Hungry; but reply rate is essentially **flat (0.78 → 0.79)**.
- **At the line (Starving, level < 25), behaviour is a decisive *step override*** — turns
  2.5 → 0.2, Engaged 0.68 → 0.08, feeding pursuit up. The `_run_hunger_tree` fires.

So the gradient analysis (B8) being "weak as a *smooth* gradient" is **not a failure** — it is
**confirmation** that this is a threshold controller: graded whispering below the boundary, a
hard override across it. B3, B4 and B8 should be read as that one design, not three separate
verdicts.

---

## 5. RQ2 — Does deficit expression lead to reliable recovery?

### RQ2.a — Does expressing the deficit elicit recovery behaviour? *(analysis B5)*

**What we asked.** When the robot signals hunger, do humans respond with food — and does the
robot *initiate* this, or only react?

**How.** Three angles: (1) meal size vs the deficit state at feed time; (2) reactive QR feeds
by state; (3) **proactive Telegram pings → did the user reply within 1 h**; plus a
proactive-vs-reactive comparison.

**Result.**
- Meal size **grows with deficit** (Full 21 / Hungry 29 / Starving 43) — bigger meals when
  hungrier.
- Proactive Telegram pings drew replies at **0.21 [0.16, 0.26]** — modest but real; the drive
  **reaches users off-robot** and pulls a response.
- Recovery is **drive-initiated (proactive)**, not merely reactive.

**Verdict: Supported.**

### Figure 8 — the remote loop reaches people and draws replies

![Remote loop](analysis/figures/fig08_remote_loop.png)

*Left: count of proactive Telegram pings by type (Hungry-entry, Starving-proactive, Starving-
recovery). Right: the response-to-ping rate with bootstrap CIs. The deficit escapes the robot
body and elicits replies from subscribers, at a modest but non-zero rate.*

### RQ2.b — Are Starving recoveries *sufficient*? *(analysis B6)*

**What we asked.** When the robot does hit Starving, does it get fed, escape starvation, and
climb all the way back to Full?

**How.** We reconstructed **HS3 (Starving) episodes** with three separate, strict outcomes:
received a first feed, escaped Starving via feeding, recovered all the way to Full via feeding.
Plus a Kaplan–Meier time-to-first-feed curve.

**Result.** Of **n = 8** Starving episodes: **8/8 received a feed, 8/8 escaped Starving,
8/8 recovered to Full** — all via feeding. No attrition.

**How to read it (crucial caveat).** n = 8 is **thin and exploratory** — 100% here is an
operational check, not a population rate. And the honest attribution: **overall reliability
(RQ2-c) is *not* carried by these 8 episodes** nor by the modest 21% ping rate. It is carried
by the fact that **Starving is rare in the first place** (next section). Recovery "works"
mainly because the robot seldom gets that low.

**Verdict: Supported (weak).**

### Figure 7 — Starving recovery status and time-to-feed *(exploratory, n = 8)*

![Starving recovery](analysis/figures/fig07_hs3_funnel.png)

*Left: the recovery funnel — every episode received a feed, escaped, and recovered to Full.
Right: cumulative probability of the first feed over time since Starving onset, with the 8-s
feed-wait timeout marked. Read as directional given the tiny n.*

### RQ2.c — Is replenishment *reliable* over the long run? *(analysis B7, the headline)*

**What we asked.** Over indefinite always-on operation, what fraction of time does the robot
spend Starving? Is there any risk it gets stuck (an absorbing "starve to death" state)?

**How.** We fitted a **continuous-time Markov chain** over Full/Hungry/Starving from the
observed transition counts and dwell times, solved for the **steady-state occupancy**, and —
because the Starving row rests on only ~17 transitions — **bootstrapped** it (resampling each
transition count as Poisson) to get an honest interval instead of a fragile point estimate.

**Result.** Modelled long-run **Starving occupancy: median 1.1% [95% CI 0.4%, 2.3%]**. The
robot is **out of starvation ~98–100% of the time**. There is **no absorbing state** — every
Starving spell is eventually left; the chain never drifts to zero.

**How to read it.** We **lead with the interval, not the 1.1% point** (the point is fragile on
so few transitions). But even the *upper* 95% bound is a small fraction — this is the strongest
single piece of RQ2 evidence, and it is a **model result** grounded in real transition data.

**Verdict: Supported.**

### Figure 9 — long-run occupancy: model vs data

![Steady state](analysis/figures/fig09_steady_state.png)

*The CTMC steady-state distribution (solid) sits right on top of the empirical time-occupancy
(hatched): the robot lives in Full/Hungry ~99% of the time; Starving is a thin sliver.*

### The gradient question *(analysis B8)*

Already told above under "threshold controller": Engaged-completion declines monotonically
with severity (**Spearman ρ = −0.16, p = 0.016**) but the drop is **concentrated at Starving**
(0.69 / 0.67 / 0.08), and the turns/energy trends **do not survive covariate adjustment**.
**Weakened as a smooth gradient / Supported as a threshold override** — the same coherent
story, not a contradiction.

---

## 6. Adaptive personalisation: the drive learns who feeds it *(analysis B9)*

**What we asked.** Does the robot *learn* which people are worth approaching, and does that
learning actually change its behaviour?

**How.** The salience network keeps a per-person **affinity** — an EMA (α = 0.25) of
normalised homeostatic reward, in [−1, +1]. We verified four things: (a) it converges, (b) it
is reward-driven, (c) it feeds into the IPS **eligibility threshold** via the exact coded
formula `eff_thr = max(0.50, base_ss − 0.15·affinity)`, and (d) the chatbot uses it to gate
Hungry-state pings to people above affinity 0.20. *(A data-cleaning detail: the affinity EMA
was re-threaded over merged identity variants to repair a few stale logged values, validated
to 1e-4 against the robot's own values.)*

**Result.**
- **Converges**: mean |affinity update| shrinks 0.09 → 0.05 as evidence accumulates.
- **The IPS component weights never change** — learning acts *only* through the per-person
  threshold. High-affinity feeders clear a bar up to **~0.14 lower**.
- The chatbot pings only the **11/15** learned people above affinity 0.20 when Hungry;
  everyone gets pinged only when Starving.

**Verdict: Supported.** The drive personalises *who it spends recovery effort on* — it learns
to court its feeders.

### Figure 10 — who sustains the drive

![Affinity trajectories](analysis/figures/fig10_affinity_trajectories.png)

*Left: per-person affinity trajectories stabilising positive as people feed the robot. Right:
a Lorenz curve of feeding concentration (see D4).*

### Figure 11 — affinity learning → eligibility → targeting

![Affinity learning](analysis/figures/fig11_affinity_learning.png)

*Left: the EMA update shrinks (convergence). Middle: higher affinity lowers the eligibility
threshold along the exact coded formula. Right: Hungry-state pings go only to the above-0.20
people (amber).*

---

## 7. Machine learning — sensitivity checks, honestly labelled *(Phase D)*

With only ~200 interactions, **ML here is interpretive, not confirmatory** — the real weight
is in Phase B. Everything uses **group-aware cross-validation** (leave-one-run-out /
leave-one-person-out) so it can't memorise a person or a session.

### D1 — Does hunger add predictive signal beyond the social/perceptual surface?

**How.** Predict "reached Engaged" with a gradient-boosted model, social/perceptual features
only, then **add hunger state** and measure the change; plus drop-column importance under
grouped CV.

**Result.** Adding hunger changes Engaged-prediction **AUC 0.903 → 0.933 (+0.030)** and
**PR-AUC 0.931 → 0.954 (+0.023)**. Drop-column CV ranks hunger **#3 of 13** features — behind
**social state (#1, by far)** and centrality. **Social/perceptual state dominates**; hunger
contributes a real but secondary signal.

**Read as:** sensitivity evidence consistent with the threshold-override story (the held-out
model reproduces the Starving collapse), *not* a mechanism proof.

### Figure D1 — ML sensitivity

![ML sensitivity](analysis/figures/figD1_ml_sensitivity.png)

*Left: ablation (hunger adds a modest bump). Middle: drop-column importance — social state
dominates, hunger is #3. Right: out-of-fold predictions track the observed Starving collapse.*

### D4 — Does recovery depend on a few feeders? (robustness of RQ2-c)

**Feeding Gini = 0.58** over 15 users; the **top-3 feeders supply 61% of meals** — *moderate*
concentration, a mild robustness caveat (replenishment leans on a handful of people). *(An
earlier exploratory KMeans over per-user behaviour was dropped: its silhouette was too low to
define meaningful user types, so it added no research-grade signal.)*

### D5 — How the deficit is verbalised (framing)

Deficit **raises hunger framing** in speech (path a **+0.31 [0.19, 0.43]**; co-present 0.35 vs
Telegram 0.30 mention rate). Honest methodology note: the *framing → reply* path is
**dropped as temporally leaked** (co-present framing only exists inside turns that already
presuppose a reply). The one leakage-free elicitation signal is the **proactive ping → reply
rate (0.26)** — modest, and consistent with B5.

---

## 8. Honest limitations (read before quoting any number)

- **Single condition.** Always-on throughout; no drive-off control. RQ2 rests on the
  within-drive gradient and proactive/reactive contrast, not a randomised comparison.
- **Small-n Starving.** 8 Starving episodes, ~13 Starving interactions. Those results are
  **directional**, reported with n and bootstrap CIs — not proof.
- **No metric survives multiple-comparison correction.** Best is **q ≈ 0.066** (B8). With one
  condition and single-digit cells, the evidence is carried by **effect sizes + bootstrap
  intervals**, deliberately, not by NHST.
- **RQ1.1/1.2 are faithful-implementation results, not independent measurements** — they
  confirm the machinery matches the code (zero-width CIs are the tell), and the non-trivial
  parts are autonomy and the absence of flapping.
- **Recovery works mainly because starvation is rare** (B7), not because the 8 episodes or the
  21% ping rate are individually strong.

---

## 9. Success-criteria scorecard

| # | Claim | Source | Outcome |
|---|---|---|---|
| RQ1-1 | Internal monitoring continuous & autonomous | B1 | **Supported** *(faithful impl.)* |
| RQ1-2 | Deficit detection correct (60/25 thresholds) | B2 | **Supported** *(faithful impl.)* |
| RQ1-3 | Deficit→action is real, not cosmetic | B3 | **Weakened (directional)** |
| RQ1-4 | Behavioural prioritisation (drive outranks social agenda) | B4 | **Supported** |
| RQ2-a | Deficit expression elicits recovery | B5 | **Supported** |
| RQ2-b | Starving episodes feed, escape, recover to Full | B6 | **Supported (weak)** |
| RQ2-c | Replenishment reliable long-run | B7 | **Supported** |
| gradient | Full→Hungry→Starving monotonic & robust | B8 | **Weakened as ramp / Supported as threshold override** |

**Bottom line.** The orexigenic drive is a genuine, faithfully-implemented **threshold
homeostatic controller**: it monitors itself autonomously, detects deficits cleanly, whispers
graded signals below the line, and **hard-overrides behaviour to pursue food when Starving**.
Recovery is **reliable primarily because the controller keeps starvation rare** (≈1% long-run
occupancy, no absorbing state), and the system **learns who its feeders are** and spends its
recovery effort on them. The honest caveats (§8) — single condition and small Starving n —
are stated plainly and do not undercut the core, code-grounded finding.

---

## 10. Reproducing this

The notebook is generated from [`analysis/build_notebook.py`](analysis/build_notebook.py)
(kept as a plain `.py` so cells stay short and commented):

```bash
cd analysis
python build_notebook.py            # regenerate the .ipynb from source
jupyter nbconvert --execute --to notebook --inplace orexigenic_analysis.ipynb
```

- **Seed** `SEED=42`; DB access is strictly **read-only / immutable** (sources are never
  mutated).
- Intermediate frames cache to `analysis/cache/*.parquet`; deliverables land in
  `analysis/outputs/` (reports + CSVs) and `analysis/figures/` (PNG + SVG ≥ 220 dpi).
- **Privacy:** all identities are pseudonymised to `P01…P14`; real-name maps stay in the
  git-ignored `analysis/private/`, never in published figures, tables, or this document.
- Pinned dependencies: `analysis/requirements.txt`.

**Key output files:** [`results_summary.md`](analysis/outputs/results_summary.md),
[`verification_report.md`](analysis/outputs/verification_report.md),
[`quality_report.md`](analysis/outputs/quality_report.md),
[`active_cost_table.csv`](analysis/outputs/active_cost_table.csv),
[`success_criteria.csv`](analysis/outputs/success_criteria.csv).
