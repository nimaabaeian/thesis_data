# Orexigenic Drive and Always-On Homeostatic Regulation

This report explains the analysis in
[`analysis/orexigenic_analysis.ipynb`](analysis/orexigenic_analysis.ipynb). Every number below is
read from the regenerated artifacts in [`analysis/outputs/`](analysis/outputs/); none is typed by
hand.

The system under study is the *Orexigenic Drive* in the `alwaysOn-embodiedBehaviour` iCub
controller. The drive continuously tracks internal energy, detects hunger states, changes the
robot's behaviour when energy is low, and uses social and remote interaction to seek replenishment:

`perception -> salience -> executive regulation -> remote/Telegram signalling`

---

## Read this first

**This version supersedes an earlier report whose headline claims did not survive audit.** Several
of them were artefacts of bugs, not findings. They are listed in
[§2](#2-what-changed-and-why) rather than quietly rewritten, because a reader who saw the earlier
numbers deserves to know which ones moved and by how much.

Every result now carries exactly one **evidence class**, and the classes are not interchangeable:

| Class | Meaning |
|---|---|
| `Implementation verification` | Follows from the controller source. Confirms the code is faithfully implemented and logged. **Not a discovered fact.** |
| `Within-deployment association` | A cluster-aware association in this deployment. **Not causal, not a population estimate.** |
| `Exploratory observation` | Descriptive. Too small-n or too selection-prone to support inference. |
| `Inconclusive` | The analysis ran and did not settle the question. |
| `Requires replication` | Suggestive; identification needs new data. |

The single most important structural fact about this study: **the drive was always on.** There is
no drive-off condition. Nothing here identifies a causal effect of the drive, and no amount of
modelling can change that.

---

## Contents

1. [Main findings](#1-main-findings)
2. [What changed, and why](#2-what-changed-and-why)
3. [Research questions and design](#3-research-questions-and-design)
4. [Data and pipeline](#4-data-and-pipeline)
5. [How to read the statistics](#5-how-to-read-the-statistics)
6. [RQ1 — Does a deficit change behaviour?](#6-rq1--does-a-deficit-change-behaviour)
7. [RQ2 — Does deficit expression close the recovery loop?](#7-rq2--does-deficit-expression-close-the-recovery-loop)
8. [RQ3 — What does the adaptive memory actually establish?](#8-rq3--what-does-the-adaptive-memory-actually-establish)
9. [Machine-learning sensitivity check](#9-machine-learning-sensitivity-check)
10. [Scorecard](#10-scorecard)
11. [What these data cannot establish](#11-what-these-data-cannot-establish)
12. [Reproducing this](#12-reproducing-this)

---

## 1. Main findings

**The strongest result is RQ1's behavioural coupling.** When the robot is in deficit (energy below
60), the odds that a meal arrives during an interaction are **5.3x** higher than at Full
(person-cluster bootstrap **[2.9, 9.1]**; run-cluster bootstrap [2.6, 7.1]; leave-one-person-out
4.4–6.5). It survives adjustment for social state, trigger mode, phase, and the person's prior
interaction count. This is an association within a single always-on deployment — it is not a causal
effect of the drive, because there is no drive-off condition to compare against.

**Meal size scales with deficit severity**: Full 21 → Hungry 29 → Starving 43 stomach points,
**+10.9 per deficit step** (run-cluster bootstrap [+6.2, +14.0]), and the gradient survives
excluding the two obligated feeders.

**The remote channel is a weak recovery pathway.** With one-to-one reply matching and "thanks, I'm
full" notifications excluded, **36/172** hunger pings drew a reply within an hour (0.21, exact
[0.15, 0.28]) — against **0.09** [0.05, 0.15] in matched no-ping control windows drawn from the
same running hours. The difference is **+0.12** (subscriber-cluster bootstrap [+0.05, +0.19]). Real,
but small.

**The robot starved more than the previous report showed.** There are **17** Starving episodes in
the corpus, not 8, and **13/17** recovered to Full by feeding (exact [0.50, 0.93]) — not 100%. The
longest episode ran **30 minutes** down to level 10.5 **in a run with 15 logged interactions**:
people were present, and the robot was not fed. That single episode is ~65% of all Starving time in
the corpus and was **invisible** to the previous analysis.

**RQ2's reliability claim is withdrawn.** The robot spent **1.67%** of observed seconds in Starving
(run-cluster bootstrap [0.38%, 4.04%]) — that empirical figure is assumption-free and stands. The
modelled "long-run Starving occupancy of 1.0% [0.2, 3.1]" does not: Starving occupancy differs
**~9x between Phase 1 and Phase 2** (2.81% vs 0.30%), so a time-homogeneous Markov chain fitted
across both estimates the stationary fraction of a process that never ran.

**RQ3's central claim was largely a restatement of the source code.** Affinity is a *deterministic*
EMA of delivered energy; `fed` and `affinity_before` are terms inside the update rule and alone
explain **R² = 0.41** of every update. The previous model omitted both and regressed Δaffinity on
`duration` (correlated 0.58 with `fed`) in their place. Properly controlled, the engagement slope is
**+0.041** against the **+0.168** previously reported.

**What RQ3 does establish is about the humans, and it decomposes.** Obligated feeders delivered
**2.7x** the meal energy per person-day (randomisation p = 0.048 against a design floor of 0.015) —
but **per interaction they fed only 1.2x as often** (randomisation p = 0.36). The difference is
**exposure**: they came to the robot **2.2x more often**. Being told to feed the robot made people
*visit* it. It did not make them markedly more generous once there.

**Scope.** One robot, one site, 8 session-days, 12 monitored runs, 14 named people, and **2 people
per controlled role**. Every effect here is existence-and-magnitude evidence for *this* loop.

---

## 2. What changed, and why

Eight corrections, in descending order of how much they moved a conclusion. All are reproduced in
[`analysis/outputs/results_summary.md`](analysis/outputs/results_summary.md).

### 2.1 Starving episodes: 8 → 17 *(this one inverts a headline)*

The executive logger writes `hunger_state_before` and `hunger_state_after` as **the same value** on
the row where a threshold is crossed by *passive drain*:

```
monotonic  event   stimulus       hs_before  hs_after  level_before  level_after
11529.10   sample  passive_drain  HS2        HS2       25.012734     25.005765
11530.10   sample  passive_drain  HS3        HS3       25.005765     24.998797   <- the crossing
```

Only *discrete* events (`interaction_cost`, `feeding`, `mode`) produce a genuine `before != after`
row. The old episode builder looked for `after == "HS3" and before != "HS3"`, so it was **blind to
every drain-entered episode** — that is, to exactly the episodes the robot fell into when nobody was
interacting with it, and therefore when nobody was there to feed it.

Its "8/8 received a feed, escaped, and recovered to Full; median 21 s" was the **selection rule
restating itself**.

The same gap emptied `v_hs_transitions` of all 32 real drain-driven Full→Hungry falls, leaving it
with an impossible ledger: 8 entries into Starving and 17 exits from it. Everything is now derived
from the **level series**, which is ground truth.

| | Old | Corrected |
|---|---|---|
| Starving episodes | 8 | **17** |
| Recovered to Full by feeding | 8/8 = 100% | **13/17 = 76%** (exact [0.50, 0.93]) |
| Longest episode | 108 s | **1801 s (30 min), to level 10.5** |

### 2.2 RQ3's core model regressed Δaffinity on its own update equation

The controller's rule, with every constant taken from source:

```
credit     = reward_delta + active_energy_cost · 1[meals > 0]
Δaffinity  = α · (clip(credit / 25) − affinity_before)
```

and `reward_delta` is **exactly** `stomach_level_end − stomach_level_start` (verified to machine
precision, not assumed). So:

- `active_energy_cost` — used by the old analysis as an *"independent dose agreement check"* — is a
  **literal additive term in the credit**. Agreement there was arithmetic, not evidence. It is
  dropped as a dose.
- `fed` and `affinity_before` are the two dominant terms of the update, and the old model **omitted
  both**, putting `duration` (correlated 0.58 with `fed`, and 52% missing, differentially by role)
  in their place.

| Specification | Slope |
|---|---:|
| **OLD**: duration, no controls | **+0.168** |
| Duration, complete case, controlled | +0.100 |
| Duration, IPW-weighted, controlled | +0.109 |
| **PRIMARY**: `n_turns` (fully observed), controlled | **+0.041** |

### 2.3 The role manipulation worked through attendance, not generosity

"Obligated feeders supplied meals at 2.7x the unconstrained rate" was presented as evidence that the
roles changed *feeding behaviour*. The 2.7x is real, but it decomposes:

| Quantity | Feeder vs unconstrained | Randomisation p |
|---|---:|---:|
| **Total delivery** (meals / person-day) | **2.7x** | 0.048 |
| **Per-encounter** (meals / interaction) | 1.2x | 0.36 |
| **Exposure** (interactions / day) | **2.2x** | — |

Exposure is a **mediator** of the role, not a confounder — telling someone to feed the robot makes
them go to the robot — so the per-encounter figure is a *decomposition*, not a bias correction. Both
are now reported. The old report conflated them.

### 2.4 B4 was quasi-separation reported as precision

Thirteen Starving interactions; **exactly one** reached Engaged. The old analysis fitted an ordinary
logistic GEE and reported `OR 0.03 [0.008, 0.136], p = 1.9e-6`. Under quasi-separation the MLE
diverges and the Wald SE collapses with it, manufacturing a tiny p-value from a cell containing
almost no information. The tell was in the analysis's own output: its cluster bootstrap lower bound
came back as `3.3e-11`, which is not a confidence bound, it is a divergence.

Refitted with **Firth's penalised likelihood** and an exact test. Its p-value is now **excluded from
the confirmatory families**. Evidence class: `Exploratory observation`.

### 2.5 "The labels do not flap" was backwards

B2's one non-trivial claim was **zero** rapid reversals at either threshold. Its detector tested
`r.from_state == prev.from_state`, when a reversal requires `r.from_state == prev.to_state`. The
condition was **unsatisfiable** — it would have returned zero on any data whatsoever.

Corrected: there are **57** rapid reversals (55 at the 60 boundary, 2 at 25; median gap **29 s**),
overwhelmingly *action cost pushes the level under the threshold → someone feeds the robot → back
over*. The flapping is real. It is precisely why `chatBot.py` carries a 60 s `HS_DWELL_SEC`
debounce — which the old write-up cited approvingly without noticing that it contradicted the claim
it was supporting.

### 2.6 The CTMC dropped every run's final dwell

`state_sequence()` emitted segments only *between* state changes, so the stretch from the last change
to the end of each run — a genuine right-censored sojourn — never entered the state-time
denominators. Non-ergodic bootstrap resamples were silently swallowed by a bare `except: pass`,
which conditioned the interval on ergodicity. And the generator pooled visited runs with idle runs,
and Phase 1 with Phase 2, despite Starving occupancy differing **~9x** between the phases.

### 2.7 The remote-loop analysis double-counted replies, and had no control

The old loop asked, per ping, *"was there **any** user message in the next hour?"* — so one reply
could be credited to five pings. It also pooled `hs3_recovery` ("thanks, I'm full") notifications in
with hunger pings; a reply to a thank-you is not evidence that hunger signalling works. And it had
**no control condition at all**, so "21% of pings got a reply" had nothing to be 21% *against* —
people message the bot anyway.

Now: one-to-one matching (a reply is consumed by at most one ping), recovery notifications excluded,
and matched control windows drawn **from inside the monitored run spans** — the hours the robot was
actually running and a reply was possible. (Drawing controls from the whole calendar, including
nights and gaps between sessions, put the control at 1/172 and inflated the effect roughly
two-fold. The control set *is* the analysis here.)

### 2.8 The ML section reported one CV split with no interval and no null

Two features (`ss_rank`, `hs_rank`), one grouped-CV split, ΔAUC = +0.088, no confidence interval, no
permutation test. It also printed a "drop-column importance" table as a second, corroborating
analysis — but with two features, drop-column importance **is** the ablation (`auc_without hs_rank`
= the social-only AUC, to four decimals). One result, printed twice.

Now: 25 repeated grouped-CV runs and a within-run permutation null. ΔAUC = **+0.081 [+0.072,
+0.094]**, permutation **p = 0.005**. It survives — but the redundant table is deleted.

---

## 3. Research questions and design

- **RQ1** — Does the orexigenic drive instantiate the four operational functions of homeostasis:
  internal monitoring, deficit detection, deficit-to-action coupling, and priority reallocation?
- **RQ2** — Does expressing a deficit promote recovery-oriented engagement sufficient to support
  reliable replenishment?
- **RQ3** — Does the robot's learned memory of past interactions reflect participants' behaviour,
  and does it influence how the robot engages with people later?

**Two design facts govern everything.**

**The drive was always on.** No off-switch condition exists. RQ2 is therefore identified only
*within* the always-on deployment, by comparing behaviour across energy states. No causal share is
recoverable.

**Two people per controlled role.** In Phase 1 (first 4 days) two participants were obligated
feeders, two were asked to interact but never feed, and everyone else was unconstrained; in Phase 2
all constraints lifted. Role labels were external metadata and were **never controller inputs**.
With 2 people per role, role is nearly aliased with identity — any "role effect" is also "these two
particular people". Role contrasts are therefore reported as a **descriptive manipulation check**
with person-level randomisation inference, whose p-value has a hard floor of ~0.015 set by the
`C(12,2) = 66` possible assignments.

---

## 4. Data and pipeline

Four asynchronous streams (`vision`, `salience_network`, `executive_control`, `chat_bot`) across
eight dated snapshots. A key preparation finding: **the snapshots are cumulative views of one
growing database, not eight independent experiments.** Naively stacking them would double-count by
roughly 4–5x.

After de-duplication:

- **12 monitored runs** (10 with visitors, 2 idle), **8 session-days**, ~46 h
- **217 co-present interactions** across **14 named, pseudonymised people**
- **208 learning-eligible affinity update events**
- **165,460 hunger-level events**; **17 Starving episodes**; **180 threshold crossings**

Five gated stages, each preventing a specific error: ingest and de-duplicate → harmonise identity
and time → verify logs against the controller (V1–V5, all passing) → freeze analysis tables → run
inference.

> **One identity column was previously unprotected.** `feeder_face_id` — the person who delivered
> each meal — was not in `IDENTITY_COLS`, so it was never canonicalised or pseudonymised. That was
> both a real-name leak vector and the reason meals could not be attributed to people at all
> (feeding events carry **no** `exec_interaction_id`; `feeder_face_id` is the only link). Fixed, and
> the leak checker in [`analysis/check_notebook.py`](analysis/check_notebook.py) enforces it.

---

## 5. How to read the statistics

The cluster count is small — **14 people, 12 runs, 8 days** — and two consequences drive every
modelling choice.

**Asymptotic sandwich SEs are not trustworthy at 14 clusters.** GEE robust SEs are anti-conservative
well below ~40 clusters. GEE/MixedLM is used as the point-estimate engine, but **the person-cluster
bootstrap interval is what every verdict leads with**; the asymptotic CI is reported second. Where
they disagree, the bootstrap is the honest statement.

**Some quantities in this system are true by construction.** The stomach level is a software
integrator, the hunger label is derived from it by the same thresholds, and per-person affinity is a
deterministic EMA of the logged reward. Analyses of *those* are `Implementation verification`.

| Question | Outcome | Model |
|---|---|---|
| Does deficit predict feeding received? | binary, per interaction | Logistic GEE, clustered by person **and** by run |
| Does Starving suppress social completion? | binary, per interaction | **Firth penalised** logistic + Fisher exact (quasi-separation) |
| Does meal size scale with deficit? | continuous | Cluster-robust OLS, hunger state as an **ordered** predictor |
| Do pings elicit replies? | binary, per ping | One-to-one matching vs **matched control windows** |
| Did obligated feeders feed more? | counts per person-day | Poisson GEE + **exposure offset** + person-level **randomisation** |
| Does engagement dose change affinity? | continuous | Mixed model, controlling for `fed` and `affinity_before` |
| How often is the robot Starving? | time occupancy | **Empirical** (assumption-free); CTMC only with its diagnostics |

**Unidentified faces (`person_id == "unknown"`) are excluded from person-clustered models.** They
are not one person; pooling ~23 interactions from an unknown number of strangers into a single
cluster asserts a dependence structure that does not exist.

**Multiplicity.** Every p-value entering a conclusion is registered at the point it is computed and
exported to [`multiplicity_table.csv`](analysis/outputs/multiplicity_table.csv) — **including** the
dose × role and dose × phase interaction terms and the role contrasts, which the previous version
quoted in its verdicts but never corrected. Of 15 confirmatory p-values, **4 survive** BH at q<0.05.
B4's separated p-value is recorded but deliberately **not** corrected and **not** used to support
any claim.

---

## 6. RQ1 — Does a deficit change behaviour?

### B1/B2 — Monitoring and detection · `Implementation verification`

The drain rate matches nominal at **1.00x** with a degenerate CI, because the stomach level is a
software integrator. Bracket accuracy is **1.00** because the label is computed from the level by
the same thresholds. **Neither is a measurement of anything**, and neither carries evidential
weight.

Three things here *are* informative:

- **Dense autonomous sampling**: every 2.3 s across 12 runs / 46 h, including two runs with no
  visitors.
- **Detection latency is tight**: drain-driven falls are caught within **0.0070** stomach points —
  about one sampling step. (Not 3.6 points; see [§2.5](#25-the-labels-do-not-flap-was-backwards).)
- **The labels flap**: 57 rapid reversals, median gap 29 s. Real, and the reason the debounce exists.

### B3 — Deficit and feeding received · `Within-deployment association`

**The outcome was misnamed.** It is `meals_eaten_count > 0` — *a meal arrived* — which is an outcome
of the human-robot dyad, not a robot action. It was previously called `fed01` and described as
"feeding pursuit". The robot can pursue feeding and get nothing, and can receive food without
asking. It is now `feeding_received`, and the estimand is a **deficit–feeding association**.

| | Full | Deficit |
|---|---:|---:|
| Feeding received | 0.15 | **0.43** |
| Mean meal size | 21 | **31** |

**OR 5.3**, person-cluster bootstrap **[2.9, 9.1]**, run-cluster bootstrap [2.6, 7.1], LOPO 4.4–6.5,
robust to adjustment for social state, trigger mode, phase, and prior interaction count.

**Separately, and as `Implementation verification` only**: hunger framing (3% → 66%), the 20
feed-seeking speech acts, and the 172 proactive pings (vs **0** at Full) are **state-gated in
source**. "Proactive pings: 0 → 172" is not a finding; it is an `if` statement. The previous report
listed these alongside the measured outcomes in its headline, which inflated the apparent evidence.

![Deficit to action](analysis/figures/fig04_deficit_action.png)

### B4 — Starving priority reallocation · `Exploratory observation`

**Thirteen Starving interactions. One reached Engaged.** Six of the thirteen come from a single
person.

| | n | Engaged | Exact 95% CI |
|---|---:|---:|---|
| Starving | 13 | 1 (0.08) | [0.00, 0.36] |
| Full/Hungry | 204 | 139 (0.68) | [0.61, 0.74] |

Meanwhile **feeding received rises** (0.26 → 0.54) and turns fall (−2.39). The **opposing
directions** are the substance: this looks like priority reallocation toward recovery rather than
undifferentiated disengagement. But the magnitude is not estimable from 13 interactions. The Firth
OR (0.048, profile CI [0.005, 0.216]) is a **bound, not an estimate**.

![Prioritisation heatmap](analysis/figures/fig05_prioritisation_heatmap.png)

---

## 7. RQ2 — Does deficit expression close the recovery loop?

### B5 — Recovery behaviour · `Within-deployment association`

**Meal size scales with deficit severity**: 21 / 29 / 43, **+10.9 per step** (run-cluster bootstrap
[+6.2, +14.0]), surviving exclusion of the two obligated feeders (+7.5/step).

**The remote loop, against its control:**

| | rate | exact 95% CI |
|---|---:|---|
| After a hunger ping (60 min) | 36/172 = **0.21** | [0.15, 0.28] |
| Matched no-ping control window | 16/172 = **0.09** | [0.05, 0.15] |
| **Difference** | **+0.12** | subscriber-cluster bootstrap **[+0.05, +0.19]** |

Real, and small. Response rates vary 0.00–0.50 across subscribers.

![Remote loop](analysis/figures/fig08_remote_loop.png)

### B6 — Observed Starving episodes · `Exploratory observation`

Over **all 17** episodes (not the 8 the label-keyed builder could see): **13/17** received a feed
(exact [0.50, 0.93]) and **13/17** recovered to Full by feeding. The episodes cluster in **9 runs**,
so the effective n is nearer 9 than 17.

> **The longest episode ran 30 minutes, down to level 10.5, in a run with 15 logged interactions.**
> People were present. The robot was not fed. It is ~65% of all Starving time in the corpus, and the
> previous analysis could not see it.

### B7 — Long-run occupancy · `Inconclusive`

**Empirically** — and this needs no model — the robot spent **1.67%** of observed seconds in
Starving (run-cluster bootstrap [0.38%, 4.04%]). That figure stands.

The **modelled** stationary occupancy does not. With terminal dwells restored and non-ergodic
resamples counted rather than discarded, the fitted CTMC gives 0.89% [0.15, 2.90] — but it fails its
own diagnostics:

| Stratum | runs | hours | Starving transitions | empirical Starving |
|---|---:|---:|---:|---:|
| Pooled | 12 | 46.0 | 17 | 1.67% |
| Phase 1 | 8 | 25.1 | 11 | **2.81%** |
| Phase 2 | 4 | 20.9 | 6 | **0.30%** |
| Idle runs | 2 | 0.7 | 0 | *not identifiable* |

**Starving occupancy differs ~9x between the two halves of the study.** A time-homogeneous chain
fitted across both estimates the stationary fraction of a process that never ran. This is a
structural deficiency, not a matter of tightening the fit.

![Steady state](analysis/figures/fig09_steady_state.png)

### D4 — Feeding concentration · `Exploratory observation`

Gini **0.57** over 14 named users; the **top 3 supply 62%** of all meal energy. This is the standing
caveat on every RQ2 result: the regulatory loop closed because a handful of specific people chose to
close it.

---

## 8. RQ3 — What does the adaptive memory actually establish?

### B9 — The mechanism · `Implementation verification`

The affinity EMA is reproduced **exactly** from the logged rewards using the controller's own
constants: `reward_delta` **is** the stomach-level change (max |diff| = 0.0), `eff_thr = max(0.50,
base_ss − 0.15·affinity)` matches every logged selection (max error 0.0000, n=3378), the perceptual
IPS weights never change (1 distinct combination across 216,940 events), and Hungry-state pings are
gated to the 11/14 people above affinity 0.20. The reconstruction also reproduces the robot's **own
persisted memory** for the 12 people it stored under a single identity — an artefact independent of
the event log.

**None of this is evidence that the robot "learns about people".** Affinity is a deterministic
function of delivered energy. It cannot drift, cannot be noisy, and cannot encode anything else. The
previous report framed RQ3 as showing the memory "is not consistent with purely uncontrolled drift".
Drift was never a live alternative: the EMA is four lines of arithmetic.

### B10.1 — The role manipulation · `Within-deployment association`

**This is the only genuinely empirical question in RQ3, and it is about the humans.**

| | Feeder vs unconstrained | bootstrap | randomisation p |
|---|---:|---|---:|
| **Total delivery** (meals/person-day) | **2.7x** | [1.0, 8.4] | **0.048** |
| **Per-encounter** (meals/interaction) | 1.2x | [0.8, 2.7] | 0.36 |
| **Exposure** (interactions/day) | **2.2x** | — | — |

The no-feed pair supplied **0/15** meals in Phase 1 — perfect compliance, though 15 observations only
bound their feed rate below 0.22. The feeder excess shrank **0.43x** once the obligation lifted.

**Being told to feed the robot made people go to the robot. It did not make them markedly more
generous once there.** With 2 people per role, this shows the manipulation took *for these four
people* and estimates **nothing** about a population.

![Role validation](analysis/figures/fig12_role_validation.png)

### B10.2 / B10.3 — Dose and downstream expression

**Δaffinity is a programmed learning-rule response.** Controlling for `fed` and `affinity_before` —
both terms *in the update rule* — and using the fully observed dose, +1 SD of engagement is
associated with **Δaffinity +0.041** [+0.016, +0.066], person-cluster bootstrap [+0.009, +0.155].
The uncontrolled specification reported **+0.168**.

**Downstream**, prior affinity is associated with next-day proactive approaches *per opportunity* at
**RR 1.19** [1.07, 1.33] (1.55 without the exposure offset), among the 62 person-days where the
person returned. But the entire path runs through `eff_thr = max(0.50, base − 0.15·affinity)`, which
B9 verifies to a maximum error of **0.0000**. **"Affinity predicts approaches" is substantially a
restatement of that line of source code**, not an independent behavioural finding.

![Affinity dose model](analysis/figures/fig13_affinity_dose.png)

---

## 9. Machine-learning sensitivity check

Two features (`ss_rank`, `hs_rank`), 217 interactions, grouped CV by run. Adding hunger state changes
held-out Engaged-prediction AUC by **+0.081 [+0.072, +0.094]** across 25 repeated grouped-CV runs,
**permutation p = 0.005** against a within-run shuffle null. Social state remains the dominant
predictor.

Sensitivity evidence only. It corroborates the *direction* of B3/B4 and confirms nothing — and the
out-of-fold Starving column rests on the same 13 interactions that make B4 exploratory, so it is not
independent corroboration.

---

## 10. Scorecard

| Claim | Source | Evidence class |
|---|---|---|
| Internal monitoring is continuous and autonomous | B1 | `Implementation verification` |
| Deficit detection follows the coded 60/25 thresholds | B2 | `Implementation verification` |
| **Deficit is associated with feeding received** | **B3** | **`Within-deployment association`** |
| Starving reallocates priority away from social completion | B4 | `Exploratory observation` |
| Deficit expression elicits recovery behaviour | B5 | `Within-deployment association` |
| Observed Starving episodes resolve by feeding | B6 | `Exploratory observation` |
| Long-run Starving occupancy is low | B7 | **`Inconclusive`** |
| The role manipulation changed what people did | B10.1 | `Within-deployment association` |
| Affinity encodes history and is expressed downstream | B10 | `Within-deployment association` |
| Hunger state adds held-out predictive signal | D1 | `Within-deployment association` |

**Synthesis.** The engineering verification is strong and uninteresting: the controller does what it
says. The one solid empirical result is **B3** — the deficit is associated with a five-fold increase
in the odds of a meal arriving, and it survives every robustness check applied to it. **B5** adds a
graded meal-size response and a weak-but-real remote channel. **B10.1** shows the role manipulation
took, by changing attendance. Everything else is either the source code restated, or too small to
estimate.

---

## 11. What these data cannot establish

Each of the following requires **new data**. None is a matter of better analysis.

- **Drive-on vs drive-off causal identification.** There is no off condition. The causal share of the
  drive in any observed feeding is not identified and cannot be.
- **Multi-site generalisation.** One robot, one site, one convenience sample.
- **Stable role-effect estimation.** Two people per controlled role. The randomisation p-value has a
  hard floor of ~0.015 set by the number of possible assignments. More people per role — not more
  modelling.
- **Reliable Starving-episode rates.** 17 episodes clustered in 9 runs, some right-censored.
- **A calibrated long-run occupancy.** The deployment is not a time-homogeneous process.
- **Population-level conclusions of any kind.**

---

## 12. Reproducing this

```bash
make all          # clean regeneration: manifest -> execute -> check -> test -> repro report
```

| Target | What it does |
|---|---|
| `make execute` | Rebuild and run the notebook top-to-bottom from a clean state |
| `make check` | Notebook fully executed, error-free, and **leaks no participant identity** |
| `make test` | 27 regression tests, each pinning one corrected defect |
| `make manifest` | SHA-256 + row counts + schema fingerprints for every input |
| `make check-constants` | Verify `CONST` against the pinned controller source |
| `make repro` | Input hashes, software versions, execution status, output hashes |

**Full independent reproduction requires controlled access to data this repository does not and
cannot ship.** The SQLite databases contain participant faces, names and chat transcripts;
`analysis/private/` holds the identity-canonicalisation and role-assignment maps. Both are
git-ignored.

Without that data a third party can still verify that the build is deterministic (fixed seed, pinned
dependencies), that the analysis constants match the pinned controller source, and — if they hold
the data — that their copy hashes to the values in
[`analysis/data_manifest.json`](analysis/data_manifest.json).

> **Constants are currently `UNVERIFIED` against source.** `analysis/controller_source.json` has
> `commit: UNPINNED`. Set it to the 40-char SHA the deployment ran and re-run `make check-constants`
> to close this. The reproducibility report records the unverified state rather than quietly passing.
