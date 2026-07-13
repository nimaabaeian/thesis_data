# Orexigenic Drive and Always-On Homeostatic Regulation

This report explains the analysis in
[`analysis/orexigenic_analysis.ipynb`](analysis/orexigenic_analysis.ipynb). Every number below is
read from the regenerated artifacts in [`analysis/outputs/`](analysis/outputs/); none is typed by
hand, and a test asserts that every p-value quoted here appears in the multiplicity ledger.

The system under study is the *Orexigenic Drive* in the `social-robot-embodied-behaviour-architecture`
iCub controller. The drive tracks internal energy, detects hunger states, changes the robot's
behaviour when energy is low, and uses social and remote interaction to seek replenishment:

`perception -> salience -> executive regulation -> remote/Telegram signalling`

---

## Read this first

Every result carries exactly one **evidence class**. They are not interchangeable, and the
differences between them are the point:

| Class | Meaning |
|---|---|
| `Implementation verification` | Follows from the controller source. Confirms the code is faithfully implemented and logged. **Not a discovered fact.** |
| `Within-deployment association` | A cluster-aware association in this deployment. **Not causal, not a population estimate.** |
| `Exploratory observation` | Descriptive. Too small-n or too selection-prone to support inference. |
| `Inconclusive` | The analysis ran and did not settle the question. |
| `Requires replication` | Suggestive; identification needs new data. |

Three structural facts govern everything below.

**The drive was always on.** There is no drive-off condition. Nothing here identifies a causal
effect of the drive, and no amount of modelling can change that.

**Roles were not randomised.** The Phase-1 feeder / no-feed roles were assigned **by participant
availability**. There is therefore no randomisation inference in this report, and no result is
described as surviving a randomisation test. A label-permutation *sensitivity* is reported instead,
over the exact enumeration of all possible assignments, and it is labelled as such.

**Affinity is a deterministic EMA of delivered energy.** It cannot drift and cannot encode anything
else. Analyses of it are implementation verification, not evidence that the robot "learns".

---

## 1. Main findings

**The strongest result is RQ1's behavioural coupling.** When the robot is in deficit (energy below
60), the odds that a meal arrives during an interaction are **5.3x** higher than at Full
(person-cluster bootstrap **[2.8, 9.0]**; run-cluster [2.6, 7.1]; leave-one-person-out 4.4–6.5). It
**survives** the prespecified adjustment for social state, trigger mode, day and prior interaction
count, under both clustering schemes (adjusted OR 5.4 and 5.7). An association within a single
always-on deployment — not a causal effect.

**Meal size scales with deficit severity**: Full 21 → Hungry 29 → Starving 43 stomach points,
**+10.9 per deficit step** (run-cluster bootstrap [+6.2, +14.0]), and it **survives excluding the
two obligated feeders** (+7.1/step, [+2.0, +12.2] — the interval, not just the point estimate).

**The remote channel works, and is weak.** With reply-centric one-to-one matching, "thanks, I'm
full" notifications excluded, and controls matched on **subscriber, run and time-of-day**:

| Window | after a ping | matched control | paired difference | subscriber-cluster | run-cluster |
|---:|---:|---:|---:|---|---|
| 15 min | 0.12 | 0.02 | **+0.10** | [+0.06, +0.14] | [+0.06, +0.14] |
| 30 min | 0.17 | 0.03 | **+0.17** | [+0.09, +0.26] | [+0.10, +0.26] |
| 60 min | 0.21 | 0.04 | **+0.22** | [+0.12, +0.31] | [+0.14, +0.38] |

Stable across five independent control draws.

**The robot starved more than the previous report showed.** There are **17** Starving episodes, not
8, and **13/17** recovered to Full by feeding (exact [0.50, 0.93]) — not 100%. The longest ran **30
minutes** down to level 10.5 **in a run with 15 logged interactions**: people were present, and the
robot was not fed. That episode is ~65% of all Starving time and was **invisible** to the previous
analysis.

**RQ2's reliability claim is withdrawn.** The robot spent **1.67%** of observed seconds in Starving
(run-cluster bootstrap [0.38%, 4.04%]) — assumption-free, and it stands. The modelled "1.0% [0.2,
3.1]" does not: Starving occupancy differs **~9x between Phase 1 and Phase 2** (2.81% vs 0.30%), so
a time-homogeneous chain fitted across both describes neither.

**RQ3's central claim was the source code restated.** Affinity's four programmed inputs (`credit`,
`active_energy_cost`, `fed`, `affinity_before`) explain **R² = 0.45** of every update. Controlling
for them and using the fully observed dose, the engagement slope is **+0.041** against **+0.168**
uncontrolled — same sign, **4.1x apart**, so they do **not** agree.

**What RQ3 does establish is about the humans, and it decomposes.** On the complete scheduled-day
panel (96 scheduled person-days, **33 of them no-shows kept as genuine zeros**), obligated feeders
delivered **4.3x the energy** per scheduled day (bootstrap [1.6, 20.0]) — but **per interaction they
fed only 1.1x as often** ([0.0, 2.8]). The gap is **attendance**: they came **4.3x more often**.
Being told to feed the robot made people *visit* it.

---

## 2. What changed, and why

Nine corrections. All reproduced in
[`results_summary.md`](analysis/outputs/results_summary.md).

### 2.1 Starving episodes: 8 → 17

The executive logger writes `hunger_state_before` and `hunger_state_after` as **the same value** on
a passive-drain crossing:

```
monotonic  event   stimulus       hs_before  hs_after  level_before  level_after
11530.10   sample  passive_drain  HS3        HS3       25.005765     24.998797   <- the crossing
```

The old episode builder looked for `after == "HS3" and before != "HS3"`, so it was **blind to every
drain-entered episode** — precisely the ones the robot fell into when nobody was interacting with
it, and therefore when nobody was there to feed it. Its "8/8 recovered by feeding" was the
**selection rule restating itself**. Everything now derives hunger state from the **level series**.

### 2.2 RQ3's model regressed Δaffinity on its own update equation

```
credit    = reward_delta + active_energy_cost · 1[meals > 0]
Δaffinity = α · (clip(credit / 25) − affinity_before)
```

and `reward_delta` **is** `stomach_level_end − stomach_level_start` (verified to machine precision).
So `active_energy_cost` — used as an *"independent dose agreement check"* — is a **literal additive
term in the outcome's definition**. It is dropped as a dose. `fed` and `affinity_before` are now
controlled, and the full update rule is modelled explicitly.

| Specification | Slope |
|---|---:|
| **OLD**: duration, no controls | **+0.168** |
| Duration, complete case, controlled | +0.100 |
| Duration, IPW-weighted, controlled | +0.109 |
| **PRIMARY**: `n_turns` (fully observed), controlled | **+0.041** |

Sign agreement was never the test. The controlled and uncontrolled slopes differ **4.1-fold**,
and even the two secondary duration fits sit ~2.5x above the primary.

### 2.3 The role manipulation worked through attendance, and was never randomised

Roles were assigned **by availability**. All randomisation-inference language is gone. On the
complete scheduled-day panel:

| Quantity | Feeder vs unconstrained |
|---|---:|
| **Delivered energy** per scheduled day (`sum(meal_delta)`) | **4.3x** |
| Meals per interaction | 1.1x |
| **Interactions per scheduled day** (attendance) | **4.3x** |

A **count of meals is not an energy** — meals are SMALL 10 / MEDIUM 25 / LARGE 45 — and the previous
report called counts "meal energy". Exposure is a **mediator** of the role, not a confounder.

### 2.4 B4 was quasi-separation reported as precision

Thirteen Starving interactions; **one** reached Engaged. The old GEE reported `OR 0.03, p = 1.9e-6`
from a diverging likelihood. Refitted with **Firth's penalised likelihood**; its p-value is
**excluded from the confirmatory families**.

### 2.5 "The labels do not flap" was backwards

The detector tested `from_state == prev.from_state`; a reversal requires `from_state ==
prev.to_state`. The condition was **unsatisfiable** — it would return zero on any data. There are
**57** rapid reversals (median gap 29 s). The flapping is real, and it is why `chatBot.py` carries a
60 s `HS_DWELL_SEC` debounce.

### 2.6 The CTMC dropped every run's final dwell

Terminal (right-censored) sojourns never entered the state-time denominators. Non-ergodic resamples
were swallowed by a bare `except: pass`. **Irreducibility is now tested as strong connectivity of
the transition graph**, not inferred from the null-space dimension; the stationary vector is
**validated** (π ≥ 0, Σπ = 1, πQ ≈ 0) rather than coerced by taking `abs()` of an arbitrary basis
vector; and the condition is named `unique_stationary_distribution`, not "ergodic".

### 2.7 The remote loop double-counted replies and had no control

One reply could be credited to five pings. `hs3_recovery` ("thanks, I'm full") notifications were
pooled in with hunger pings. And there was **no control condition at all**, so "21% of pings got a
reply" had nothing to be 21% *against*. Matching is now **reply-centric** (each reply claims the
*nearest* preceding ping), and controls are matched on subscriber, run and time-of-day, ping-free,
non-overlapping, with each message satisfying at most one window.

### 2.8 Meal attribution used a field populated for 20 of 108 feeds

`feeder_face_id` names 8 people when 14 received meals. Feeds are now attributed to the interaction
**active at the time** (104/108 events, 96% of the energy).

### 2.9 The ML section had no interval and no null

Two features, one CV split, ΔAUC = +0.088, and a "drop-column importance" table that — with two
features — **is** the ablation. Now 25 repeated grouped-CV runs and a within-run permutation null.

---

## 3. Results

| id | claim | source | evidence class |
|---|---|---|---|
| RQ1-1 | Internal monitoring is continuous and autonomous | B1 | `Implementation verification` |
| RQ1-2 | Deficit detection follows the coded 60/25 thresholds | B2 | `Implementation verification` |
| **RQ1-3** | **Deficit is associated with feeding received** | **B3** | **`Within-deployment association`** |
| RQ1-4 | Starving reallocates priority away from social completion | B4 | `Exploratory observation` |
| RQ2-a | Deficit expression elicits recovery behaviour | B5 | `Within-deployment association` |
| RQ2-b | Observed Starving episodes resolve by feeding | B6 | `Exploratory observation` |
| RQ2-c | Long-run Starving occupancy is low | B7 | **`Inconclusive`** |
| RQ3-a | The role manipulation changed what people did | B10.1 | `Exploratory observation` |
| RQ3-b | Affinity encodes history and is expressed downstream | B10 | `Implementation verification` |
| D1 | Hunger state adds held-out predictive signal | D1 | `Within-deployment association` |

**Multiplicity.** BH runs **once, at the very end**, after every analysis — including D1 — has
registered. The complete ledger is
[`multiplicity_table.csv`](analysis/outputs/multiplicity_table.csv): 16 confirmatory p-values, of
which 9 survive at q<0.05, plus 8 exploratory ones recorded in full and deliberately **not**
corrected (B4's separated cell, B10.1's non-randomised permutation, B10.2's residual dose on top of
the update rule, B10.3's eligibility stage). The previous version corrected 5, ran BH *inside B10*
before D1 had executed, and quoted interaction terms it never corrected.

**RQ3-b decomposed** (B10.3), using real exposure from the perception and salience logs:

| Stage | Estimate | Reading |
|---|---|---|
| 1. P(detected on d+1) | OR 3.36, p = 0.052 | affinity does not clearly predict who comes back |
| 2. eligible \| detected | RR 3.51 [0.38, 32.66], p = 0.27 | rises descriptively (3% → 9%), **not** distinguishable — and this stage **is** the coded threshold |
| 3. proactive \| eligible | RR 1.03 [0.97, 1.08], p = 0.31 | the only stage with behavioural content |

Stage 2 is `eff_thr = max(0.50, base_ss − 0.15·affinity)`, which B9 verifies to a maximum error of
**0.0000**. It is arithmetic, not a finding.

---

## 4. What these data cannot establish

New data are required for each. None is a matter of better analysis.

- **Drive-on vs drive-off causal identification.** There is no off condition.
- **Multi-site generalisation.** One robot, one site, one convenience sample.
- **Any role effect beyond these four people.** Roles were assigned by availability, 2 per role, so
  role is nearly aliased with identity. There is **no randomisation to license inference**, and the
  label-permutation p has a hard floor of 0.022 set by the 45 possible assignments.
- **Reliable Starving-episode rates.** 17 episodes clustered in a handful of runs, some censored.
- **A calibrated long-run occupancy.** The deployment is not a time-homogeneous process.
- **Any independent evidence that the robot "learns" about people.** Affinity is a deterministic EMA
  of delivered energy, and the downstream path runs through a threshold verified exactly.
- **Population-level conclusions of any kind.**

---

## 5. Reproducing this

```bash
make all
# clean-outputs -> verify-manifest -> check-constants -> execute -> check -> test -> repro
```

| Target | What it does |
|---|---|
| `make all` | The acceptance build. Fails on any gate. |
| `make check-constants` | Every constant, **key by key**, at **every pinned deployment commit**. Fails hard. |
| `make verify-manifest` | Fails if inputs differ from the manifest. Never silently re-hashes. |
| `make test` | Regression tests that import the **same** functions the notebook runs. |
| `make determinism` | Executes the analysis **twice** from clean and compares every artifact. |
| `make repro` | Input hashes, versions, execution status, output hashes — and verifies them. |

**Constants are pinned and verified.** The controller changed *during* the deployment
(`b84e2f8` ran 15–18 June; `be46774` "STABLE VERSION" ran 18–29 June). Both are pinned in
[`controller_source.json`](analysis/controller_source.json), all 33 constants are verified key-by-key
at **each**, and the checker asserts they agree with each other — so the analysis rests on values
that were stable for the whole study, not on one snapshot of it.

**Tests exercise production code.** Every statistical and data-unit routine lives in
[`statistical_helpers.py`](analysis/statistical_helpers.py); the notebook imports it and so do the
tests. The previous suite carried its own copies — and one of them (the flapping detector) contained
the *same* bug as the notebook, so the test that "verified" it could never have failed.

**Full independent reproduction requires controlled access to data this repository does not and
cannot ship.** The SQLite databases contain participant faces, names and chat transcripts;
`analysis/private/` holds the identity, role and attendance maps. Both are git-ignored. Without them
a third party can still verify that the build is deterministic, that the constants match the pinned
source, and — if they hold the data — that their copy hashes to
[`data_manifest.json`](analysis/data_manifest.json).
