# Orexigenic Drive and Always-On Homeostatic Regulation — Analysis Explained

This document explains the data analysis in
[`analysis/orexigenic_analysis.ipynb`](analysis/orexigenic_analysis.ipynb): **what we did, why,
how, and what came out** — and reads the results critically, separating what the code guarantees
from what the data show.

> **System under study.** The *Orexigenic Drive* of the `alwaysOn-embodiedBehaviour` iCub
> controller — a socially embedded, always-on embodied homeostatic regulatory loop that
> continuously integrates energy depletion, detects deficit states, biases recovery-oriented action selection,
> and reallocates behavioural priority toward replenishment. Pipeline:
> *perception → salience → executive regulation → remote/Telegram signalling*.

**Contents**

1. [Research questions, design & evidential structure](#1-research-questions-design--evidential-structure)
2. [The data & pipeline](#2-the-data--pipeline)
3. [Statistical approach](#3-statistical-approach)
4. [RQ1 — Does a deficit change behaviour?](#4-rq1--does-a-deficit-change-behaviour)
5. [RQ2 — Does deficit expression close the recovery loop?](#5-rq2--does-deficit-expression-close-the-recovery-loop)
6. [RQ3 — Does adaptive regulatory memory encode behaviour? Mechanism (B9) + validation (B10)](#6-rq3--does-adaptive-regulatory-memory-encode-behaviour-mechanism-b9--validation-b10)
7. [Machine-learning sensitivity check](#7-machine-learning-sensitivity-check-phase-d)
8. [Scorecard & synthesis](#8-scorecard--synthesis)
9. [Reproducing this](#9-reproducing-this)
- [Appendix A — Instrumentation verification (B1/B2)](#appendix-a--instrumentation-verification-b1b2)

---

## 1. Research questions, design & evidential structure

- **RQ1** — To what extent does the orexigenic drive instantiate the four operational functions
  of classical homeostasis: (1) internal monitoring, (2) deficit detection,
  (3) deficit-to-action-selection coupling, (4) behavioural priority reallocation?
- **RQ2** — Does expressing an orexigenic deficit promote recovery-oriented engagement
  sufficient to support reliable energy replenishment in an always-on social robot?
- **RQ3** — Does the robot's **adaptive regulatory memory** (the learned per-person homeostatic
  affinity) encode observed participant behaviour rather than uncontrolled drift, and is that
  learned state subsequently expressed in the robot's allocation of proactive approaches?

RQ1's first two functions (monitoring, detection) are **implementation-verification claims**:
the stomach is a software integrator and the hunger label is derived from its level by the same
coded thresholds under test. They verify instrumentation, not empirical effects, and live in
[Appendix A](#appendix-a--instrumentation-verification-b1b2). The empirical weight of RQ1 is in
functions (3) and (4): B3 and B4.

### Two design facts that shape everything

**First: the drive was always on.** RQ2 is identified from within the always-on data: the
**graded deficit** (Full → Hungry → Starving) is the manipulation, and the **proactive vs
reactive** contrast tests whether the drive *initiates* recovery.

**Second: two 4-day phases with a participant role manipulation.** In **Phase 1** (first four
experiment days) two participants were **obligated feeders**, two were told to **interact but
never feed**, and everyone else behaved normally. In **Phase 2** (last four days) all
constraints were lifted. The roles *induce* known behaviour, giving RQ3 ground truth to test
the learning against. These roles are external experimental labels, not controller inputs. The
role map is private (`analysis/private/role_phase.json`); published outputs carry pseudonyms and
role labels only. With **2 people per controlled role**, role contrasts are manipulation
validation with wide uncertainty — never population inference.

### Literature-aligned vocabulary

The terminology is aligned with the reference corpus rather than generic software wording:

- **Socially embedded / interaction-aware robot** follows Dautenhahn et al.: the robot is analysed
  as structurally coupled with a human social environment, not as an isolated controller.
- **Always-on cognitive architecture** follows Pasquali et al.: the relevant unit is continual
  perception, experience accumulation, and social-context awareness in a dynamic environment.
- **Proactive / mixed-initiative behaviour** follows Moulin-Frier et al. and Senft et al.: hunger
  signalling is treated as robot-initiated action allocation, not merely messaging.
- **Adaptive regulatory memory** follows adaptive HRI work such as Ahmad et al. and Tanevska et al.:
  per-person affinity is a user-history-conditioned state that changes later policy expression.
- **Drive competition / action selection** follows Guerrero-Rosado & Verschure, Ngo et al.,
  Hulme et al., and Yoshida et al.: the orexigenic variable is interpreted as an internal
  drive-function signal that biases recovery-oriented action selection; because this dataset
  analyses one implemented drive, the claim remains a threshold homeostatic policy rather than
  multi-drive allostatic orchestration.
- **Loop-level regulation** borrows cautiously from homeostasis/allostasis literature
  (Sterling; Bettinger & Friston): B7 is a coupled human-robot regulatory outcome, while the
  implemented controller remains a threshold homeostatic policy rather than a full predictive
  allostatic controller.

### Evidential hierarchy — what carries the claims

The analysis reduces to three inferential results (B3, B4, B10), one supporting reliability
observation (B7), and an instrumentation appendix. Ranked by identification strength — not by
effect size — the hierarchy is:

1. **RQ3 role manipulation (B10) — highest-identification evidence.** The only experimental
   handle in the study: obligated feeders supplied meals at **2.7× the
   unconstrained rate**, and the no-feed pair fed **zero** (complete separation, exact
   Clopper–Pearson). Near-perfect compliance.
2. **RQ3 dose → affinity (B10).** +0.17 per SD of engagement, and it *strengthens* to **+0.23
   (p ≈ 7×10⁻¹⁴)** when the identity-reconstructed people are excluded. The affinity itself is
   **externally verified**: the reconstruction matches the robot's own persisted memory exactly
   (max |Δ| = 0.000) for the 12 people stored under one identity, and reconciles the 2 identities
   the robot forked in memory. The repair is explicitly validated rather than assumed.
3. **RQ3 downstream expression (B10).** Prior-day affinity predicts next-day proactive approaches
   (**RR 1.55**, activity-controlled). The cleanest causal-direction claim in the study:
   predictor strictly precedes outcome, so it is leakage-free by construction.
4. **RQ1 behavioural coupling (B3 + B4).** Deficit → feeding-pursuit **OR 4.9** and Starving →
   social-completion suppression **OR 0.03** — both person-clustered GEE, bootstrap-confirmed,
   both inside the BH family. This is the behavioural readout that the regulatory state changes
   action allocation.

**What *not* to stake the defense on:**

- **B7's ~1% Starving occupancy** is a **supporting reliability observation, not a primary
  identification result** —
  a pooled-regime stationary fraction resting on ~10–17 Starving-row transitions from a
  non-stationary process, and downstream of human behaviour. It should be phrased as a
  loop-level outcome, not as a controller-only property.
- **B1/B2** are implementation verification (Appendix A), not inferential findings.

### The answers, up front

**RQ1 — the drive is a *threshold* regulatory policy.** Monitoring and detection are
implementation-verification facts (Appendix A). The load-bearing results are behavioural: at the
deficit line the robot biases action selection toward a **proactive recovery policy** absent at Full — hunger framing
3% → 67%, feed-seeking acts and proactive Telegram pings 1 → 20 and 0 → 172, feeding pursuit
0.15 → 0.43 (**OR 4.9 [2.6, 9.5]**, person-clustered GEE); at the starving line it **overrides**
social completion (turns 2.5 → 0.2, Engaged 0.68 → 0.08; **OR 0.03 [0.01, 0.14]**). *Recovery
behaviour is recruited at 60 and priority is reallocated at 25 — a two-threshold controller, not
a smooth ramp.*

**RQ2 — the HRI regulatory loop closes at the deployment level.** The deficit elicits graded
feeding (meal size 21 → 29 → 43) and
modest off-robot replies (proactive ping-reply 0.21–0.26). As a *supporting* reliability
observation, long-run Starving occupancy sits at **~1%** (run-level block bootstrap 95% CI
0.2–3.1%) with no absorbing "starve-out" state — an outcome of the closed loop (drive signals →
humans engage → energy is replenished), **not a controller-only property**, and one that leans on
a few responsive feeders (Gini 0.57, top-3 = 62%). This is not the primary identification result; the
experimental weight is in RQ3.

**RQ3 — adaptive regulatory memory tracks interaction history and is behaviourally expressed.** The
manipulation validated: obligated feeders supplied meals at **2.7× the unconstrained rate**
[1.2, 5.9], the no-feed pair complied perfectly (**0 feeds in 15 Phase-1 interactions**), and the
feeder excess shrank once roles lifted. The pre-specified core model
(`Δaffinity ~ dose×role + dose×phase + (1|person)`) shows engagement dose predicting affinity
gains (**+0.17 per SD of duration [+0.12, +0.21]**, strengthening to +0.23 without reconstructed
identities), **moderated by role** (for obligated feeders, reward delivery rather than chat length
dominates the update) and
**attenuated in Phase 2**; all three dose definitions agree and the effect survives
leave-one-person-out. The learned regulatory state is then expressed in policy: **prior affinity raises next-day proactive
approaches 1.55× per +1 affinity [1.09, 2.20]**, activity-adjusted and leakage-free.

---

## 2. The data & pipeline

The robot logs to four SQLite databases (`vision`, `salience_network`, `executive_control`,
`chat_bot`); `data/` holds eight dated snapshots. **Key data-prep discovery:** the folders are
**cumulative snapshots of one growing database**, not eight experiments — naïve stacking would
double-count 4–5×. Everything is therefore de-duplicated to the true units:

- **run** (`run_id`) — one continuous robot session (12 monitored, 10 with visitors);
- **day** (`day_rome`) — 8 days, split 4 + 4 into Phase 1 / Phase 2;
- **interaction** — 217 after de-duplication, over 14 named (pseudonymised) people;
- **affinity update event** — 205 learning-eligible events (the RQ3 unit);
- **person-day, ping, transition, episode** — for the analyses that need them.

### Five gated stages

The analysis is organised as five logical stages with **hard gates** between them — nothing
downstream runs if a gate fails:

1. **Ingest & dedup.** Resolve the cumulative snapshots to canonical `run` / `interaction` /
   `event` units (the double-count fix above). This is the highest-value step; it is first and
   immovable.
2. **Identity & clock harmonisation.** Canonicalise → pseudonymise every identity **once,
   centrally**, and stamp every row with `(run_id, person_id, monotonic_sec, timestamp_epoch,
   day_rome)` before any analysis touches it.
3. **Verification gate (V1–V5 + quality).** One gate against constants extracted from the
   controller source (with file:line references): meal deltas, per-action energy costs, drain
   rate, thresholds, referential integrity, clock sanity. Emits
   [`verification_report.md`](analysis/outputs/verification_report.md); hard checks block the
   run. All passed; nothing proceeds otherwise.
4. **Unit construction.** Freeze the primary analytic tables — `interactions` (unit =
   interaction), `person_day` (unit = person×day), `hs_segments` (unit = state sojourn). Every
   model draws from one of these, so the clustering unit is never ambiguous.
5. **Inference.** B3/B4 on `interactions`; B10 on `person_day` + affinity events; B7 on
   `hs_segments`. Figures and summary last.

### Data harmonisation (four asynchronous streams)

Aligning vision / salience / executive / chat streams sampled at very different rates:

- **Anchor = the interaction**, joined to the other streams by **identity × time-window** —
  nearest event within a pre-window bracketing the interaction start, a **single documented
  rule** (the anchor-lag distribution is reported as a data-quality diagnostic, not silently
  patched with three fallbacks).
- **Identity is the master join key:** `face_id` (vision) ↔ canonical `person_id` ↔ `user_key`
  (chat), resolved through **one** canonicalisation map applied *before* any join. The only
  exposure — the merged-affinity-EMA reconstruction — is robustness-checked in B10.
- **Two clocks, explicit roles:** `monotonic_sec` for within-run durations/dwell (immune to
  wall-clock skew); `timestamp_epoch` for cross-stream alignment and day keys. Never mixed in a
  single calculation.
- **Ordinal categoricals** (HS1 < HS2 < HS3; SS1 < SS4) kept as ordered factors; continuous dose
  predictors z-scored — no one-hot encoding that discards the ordering.
- **Missingness handled at the harmonisation layer, not the model:** duration is missing ~53% of
  the time *differentially by phase*, so the fully-observed `n_turns` and `active_energy_cost` are
  the **primary** dose and `duration_sec` is a **confirmatory secondary** — the complete-case
  duration slope is never the load-bearing number.

---

## 3. Statistical approach

The model choice is driven by two facts: **what kind of outcome is being analysed** and
**whether observations are independent**. Most outcomes here are binary events, event counts,
continuous learning updates, or time spent in hunger states; and many observations repeat
within the same people, runs and days. For that reason, **no confirmatory claim rests on
row-independence tests** such as plain t-tests, ANOVA, chi-square tests, or unclustered
correlations. Those tests would treat repeated observations from the same person/run as if
they came from new independent participants.

### Why each model was used

| Purpose in this study | Outcome type | Model used | Why this model was chosen |
|---|---|---|---|
| Deficit-to-action conversion: does Hungry/Starving increase feeding pursuit? | Binary event per interaction: pursued feeding, yes/no | **Logistic GEE, clustered on person** | The outcome is binary, so logistic regression gives odds ratios. GEE keeps the interpretation population-level while using robust sandwich SEs for repeated observations within people. |
| Starving override: does Starving suppress completed social engagement? | Binary event per interaction: Engaged vs not engaged | **Logistic GEE, clustered on person**, adjusted for social state | Same binary-outcome logic, with clustering because the same people contribute multiple interactions. Adjustment separates hunger-state override from the current social-state context. |
| Role manipulation: did obligated feeders supply more meals per person-day? | Count outcome: number of meals | **Poisson GEE, clustered on person** | Meal supply is a count/rate, so Poisson regression estimates rate ratios. GEE protects the inference from repeated person-days belonging to the same participant. |
| Downstream use of learning: does prior affinity increase next-day proactive approaches? | Count outcome: number of proactive approaches | **Poisson GEE, clustered on person** | The question is about a count of robot actions on the next day. A rate-ratio model is the natural scale, and clustering handles repeated days per person. |
| Core affinity validation: does engagement dose predict `Δaffinity`, moderated by role/phase? | Continuous outcome: change in affinity after an update | **Linear mixed model with a person random intercept** | `Δaffinity` is continuous. The random intercept lets each person have their own baseline while estimating the common dose, role and phase effects. Cluster-robust OLS is reported as a companion check. |
| No-feed role compliance | 0 feeds in the no-feed group during Phase 1 | **Exact Clopper-Pearson confidence interval** | Complete separation makes a logistic/Poisson GLM unstable or unidentifiable. Exact binomial intervals report the compliance result directly. |
| Long-run reliability (supporting): what fraction of time is the robot Starving? | Time occupancy across Full/Hungry/Starving states | **Continuous-time Markov chain (CTMC)** with **run-level block bootstrap** | Reliability is about transitions and dwell times, not a per-row mean. CTMC estimates the steady-state occupancy implied by observed state changes; run-level bootstrap gives an interval that respects run-level clustering. Reported as a supporting observation (pooled, non-stationary process — see B7). |
| Machine-learning sensitivity check | Held-out prediction of engagement | **Regularised logistic models with group-aware CV** | This is not confirmatory inference. It checks whether hunger adds out-of-sample signal beyond social state while leaving out whole runs or people to avoid leakage. |

### The models, compactly (math + variables)

Notation: person $i$, observation $j$ (interaction / person-day / learning event). $e^{\beta}$ is
an **odds ratio** (logistic) or **rate ratio** (Poisson). All GEE fits use an **exchangeable**
working correlation with **robust sandwich** standard errors, so inference stays valid even if
that correlation is misspecified.

**Logistic GEE — B3 (deficit→action), B4 (Starving override).** Binary outcome $y_{ij}\in\{0,1\}$
(fed-here / reached-Engaged):
$$\mathrm{logit}\,\Pr(y_{ij}=1)=\beta_0+\beta_1 D_{ij}+\boldsymbol\gamma^{\top}\mathbf s_{ij},
\qquad \mathrm{corr}(y_{ij},y_{ij'})=\rho\ \text{(within person }i).$$
$D_{ij}$ = deficit indicator (B3: Hungry+Starving vs Full) or Starving indicator (B4);
$\mathbf s_{ij}$ = social-state controls. Effect reported as $\text{OR}=e^{\beta_1}$.

**Poisson GEE — B10 meal rate, B10 downstream.** Count $c_{ij}$ (meals/person-day, or next-day
proactive approaches):
$$\log\mathbb E[c_{ij}]=\beta_0+\beta_1 x_{ij}+\boldsymbol\gamma^{\top}\mathbf w_{ij},
\qquad \text{RR}=e^{\beta_1}.$$
$x$ = feeder indicator (meal rate) or prior-day affinity (downstream); $\mathbf w$ = phase, and —
for the leakage-free downstream model — yesterday's activity count.

**Linear mixed model — B10 core (`Δaffinity ~ dose×role + dose×phase + (1|person)`).** Learning
event $k$ of person $i$:
$$\Delta a_{ik}=\beta_0+\beta_1 z_{ik}+\beta_2\,(z\!\cdot\!\text{role})_{ik}
+\beta_3\,(z\!\cdot\!\text{phase})_{ik}+\boldsymbol\gamma^{\top}\mathbf c_{ik}+u_i+\varepsilon_{ik},
\quad u_i\sim\mathcal N(0,\tau^2).$$
$\Delta a_{ik}$ = affinity change; $z_{ik}$ = standardised engagement dose; the person random
intercept $u_i$ absorbs stable individual baselines so $\beta_1$ is a within-person dose slope.

**CTMC — B7 (long-run occupancy).** States $\{\text{HS1,HS2,HS3}\}$, generator $Q$ from observed
transition counts $n_{ab}$ and total dwell time $T_a$ in state $a$:
$$q_{ab}=\frac{n_{ab}}{T_a}\ (a\neq b),\quad q_{aa}=-\!\sum_{b\neq a}q_{ab};\qquad
\boldsymbol\pi Q=\mathbf 0,\ \textstyle\sum_a\pi_a=1.$$
The reported quantity is the Starving occupancy $\pi_{\text{HS3}}$; mean Starving sojourn is
$-1/q_{\text{HS3,HS3}}$.

**Exact Clopper–Pearson — B10 no-feed compliance.** For $x$ feeds in $n$ interactions the $1-\alpha$
interval uses Beta quantiles; with the observed $x=0$ it collapses to
$\big[0,\ 1-(\alpha/2)^{1/n}\big]$ — chosen because complete separation makes a GLM
unidentifiable.

**Affinity EMA — B9 (the learned quantity itself).** Per person, over their reward sequence:
$$a\leftarrow a+\alpha\,(r_{\text{norm}}-a),\quad
r_{\text{norm}}=\mathrm{clip}\!\big(\text{credit}/25,\,-1,\,1\big),$$
$\alpha=0.25$ on positive updates, $0.10$ on negative — an exponentially-weighted moving average
of normalised homeostatic reward, i.e. the controller's drive-reduction signal, driving the
eligibility threshold in B9.

**Multiplicity & uncertainty.** Benjamini–Hochberg controls the FDR **within** each pre-declared
family (order $p_{(1)}\!\le\!\dots\!\le\!p_{(m)}$; reject up to the largest $k$ with
$p_{(k)}\le \tfrac{k}{m}\alpha$). Every primary effect is re-checked with a **cluster bootstrap**
that resamples **whole persons** (not rows), so the interval respects within-person dependence.

### Why not the simpler textbook tests?

- **t-tests/ANOVA** are for independent continuous outcomes. Most primary outcomes here are
  binary, counts, time occupancy, or repeated continuous updates; using t-tests/ANOVA would
  answer the wrong outcome-scale question and ignore clustering.
- **Mann-Whitney/Kruskal-Wallis/Spearman** are useful descriptive non-parametric companions,
  but they do not naturally handle the repeated person/run structure or the role/phase
  moderation needed for RQ3.
- **Plain linear regression** is appropriate for independent continuous outcomes; the affinity
  analysis instead uses a mixed model because multiple updates come from the same person.
- **Cox proportional hazards regression** would be the standard survival model for larger
  time-to-event data, but the Starving episode set has only 8 episodes, so a Cox model would be
  overfit — time-to-first-feed is reported as a single descriptive statistic only (§5), and the
  CTMC carries the reliability observation.
- **Negative binomial regression** was not the primary count model because the confirmatory
  count analyses are small, clustered rate-ratio questions handled with robust Poisson GEE and
  bootstrap checks. If strong overdispersion dominated a larger count analysis, negative
  binomial would be the natural alternative.

Other safeguards:

- **Multiplicity** → Benjamini-Hochberg **within two pre-declared families** (RQ1/2 behaviour;
  RQ3 adaptation). **Every p-value used as evidence sits inside a declared family** — including the
  small-n B4 Starving override, which is folded into the RQ1/2 family for honest book-keeping even
  though its verdict is led by effect size + CI + bootstrap, not NHST (5/5 metrics survive q<0.05).
  Implementation checks (B1/B2) get no inferential p-values at all.
- **Power** → simulation-based **minimum detectable effects** under the observed clustering
  (never post-hoc power): this design reliably detects ORs ≳ 3 and Δaffinity slopes ≳ 0.075
  per SD; role contrasts (2 people/role) detect only very large effects.
- **Small-n Starving results** → Starving is small-n (13 interactions, 8 episodes), so the
  report leads with effect sizes + CIs, uses "directional" labels, and avoids covariate models
  on single-digit episode counts.

---

## 4. RQ1 — Does a deficit change behaviour?

*(Monitoring & detection — RQ1 functions 1–2 — are implementation-verification checks and live in
[Appendix A](#appendix-a--instrumentation-verification-b1b2). The two results below carry RQ1.)*

**B3 — deficit-to-action-selection coupling.** The right contrast is **Full vs deficit (Hungry +
Starving)**. A deficit biases action selection toward a recovery policy absent at Full:

| Behaviour (Full → Deficit) | Full | Deficit | Note |
|---|---|---|---|
| Hunger framing in speech | 2.8% | 66.5% | prompt-driven (coded gate) |
| Feed-seeking speech acts | 1 | 20 | deficit-only (coded gate) |
| Proactive Telegram pings | 0 | 172 | deficit-only (coded gate) |
| Co-present feeding pursuit | 0.15 | 0.43 | **emergent human response** |
| Mean meal size | 21.2 | 31.4 | emergent |

The coded gates verify that regulatory state changes the available action repertoire; the
emergent rows measure the behavioural expression of that state change. **Inferential anchor** —
logistic GEE
`fed_here ~ deficit + C(initial_state)`, cluster = person, so $e^{\beta_{\text{deficit}}}$ is the
odds of pursuing feeding in a deficit vs at Full: **OR 4.9 [2.6, 9.5]**, p ≈ 2×10⁻⁶;
leave-one-person-out OR range 4.1–6.1. *Verdict: Supported.*

![Deficit to action](analysis/figures/fig04_deficit_action.png)

*Fig 4 — units: 367 interaction turns, 710 chat messages, 217 co-present interactions, and
193 deficit-gated action events. Recovery-action rates Full vs deficit (left, bootstrap CIs)
and the deficit-gated actions plotted on the actual stomach-level timeline (right).*

**B4 — Starving priority reallocation.** Logistic GEE
`reached_Engaged ~ starving + C(initial_state)`, cluster = person — $e^{\beta_{\text{starving}}}$
is the odds of a completed conversation when Starving, holding social state fixed. When Starving,
social-completion behaviour is **strongly suppressed** (turns 2.5 → 0.2; Engaged 0.68 → 0.08;
**OR 0.034 [0.008, 0.136]**) while feeding pursuit **rises** (0.26 → 0.54) — priority
reallocation toward recovery, not undifferentiated disengagement, as the coded `_run_hunger_tree`
override specifies. Starving n = 13: a large, directionally robust effect,
not a precise estimate. *Verdict: Supported.*

![Prioritisation heatmap](analysis/figures/fig05_prioritisation_heatmap.png)

*Fig 5 — unit: interaction, n = 217; Starving column n = 13. Completion peaks at
Greeted×Hungry (0.93) and stays high for known people until Starving shifts priority away
from social completion.*

---

## 5. RQ2 — Does deficit expression close the recovery loop?

**B5 — deficit expression elicits recovery behaviour.** Meal size grows with the deficit at feed time
(Full 21 / Hungry 29 / Starving 43); proactive Telegram pings drew replies within 1 h at
**0.21 [0.15, 0.26]** (hs3-specific: 0.26) — modest but measurable, and drive-*initiated*:
co-present interactions are 83% reply-bearing when proactive vs 42% reactive.
*Verdict: Supported.*

![Remote loop](analysis/figures/fig08_remote_loop.png)

*Fig 8 — unit: proactive Telegram ping, n = 234 across 12 subscribers; response window = 1 h.
Response-to-ping rate by ping type, bootstrap 95% CIs, ping counts as n-labels on the bars.*

**B6 — observed Starving episodes (exploratory, n = 8, one sentence).** All 8 Starving episodes
received a feed, escaped Starving, and recovered to Full via feeding (median time-to-first-feed
21 s) — an operational status check, not a population recovery rate; reliability is carried by
B7, not by these episodes.

**B7 — long-run reliability (a *supporting* loop-level observation).** A continuous-time
Markov chain over Full/Hungry/Starving, fitted from observed transitions and dwell times, gives
long-run **Starving occupancy: median 1.0% [95% CI 0.2%, 3.1%]** by run-level block bootstrap
(the cluster-honest interval; the transition-level Poisson bootstrap's 1.1% [0.4, 2.3] agrees),
with **no absorbing state**. **Reading:** the transition rates record the coupled human-robot
regulatory loop — every observed Starving spell ended because someone fed the robot — so this is
a *loop-level outcome*, not a controller-only property. **Three reasons it stays
supporting:** (i) it rests on only ~10–17 Starving-row transitions; (ii) the CTMC
is time-homogeneous but the deployment is not — one generator pools visited daytime runs, idle
no-visitor runs and both phases, so the figure is the stationary fraction of a *pooled* process,
not of any real operating regime (the pooling is conservative: idle drain-only runs push Starving
*up*, so read the interval as an order-of-magnitude ceiling, not a calibrated rate); and (iii)
replenishment leans on a few responsive feeders (Gini 0.57, top-3 = 62% of meals). Single-
condition caveat: the drive's exact causal share of the feeding cannot be isolated.
*Verdict: Supported, as an outcome of the loop.*

![Steady state](analysis/figures/fig09_steady_state.png)

*Fig 9 — unit: CTMC state sojourn/transition reconstructed from 165,460 hunger events across
12 monitored runs. Modelled occupancy lands on the empirical time-occupancy (Starving 1.1%
vs 1.8%); mean Starving sojourn 163 s.*

---

## 6. RQ3 — Does adaptive regulatory memory encode behaviour? Mechanism (B9) + validation (B10)

**This is the highest-identification result because it uses the role manipulation as an external
validation handle.**

**B9 — the mechanism, verified.** The salience network implements per-person **adaptive
regulatory memory** as homeostatic affinity — an
EMA (α = 0.25) of normalised homeostatic reward, i.e. drive reduction, in [−1, +1]. Verified against the code and
logs: the EMA **converges** (mean |update| 0.10 → 0.06); the IPS perceptual weights **never
change** — learning acts only through the per-person eligibility threshold
`eff_thr = max(0.50, base_ss − 0.15·affinity)` (reproduces every logged value to 1e-4,
giving high-affinity people up to ~0.14 lower a bar); and the chatbot gates Hungry-state pings
to the 11/14 people above affinity 0.20. *(Data repair: the EMA was re-threaded over merged
identity variants, validated to 1e-4 for all non-merged people. Because this re-threading is the one
cleaning step that touches an RQ3 outcome variable, B10 re-fits its core dose→affinity model with all
canonicalization-merged people excluded — `outputs/rq3_affinity_repair_robustness.csv` — and the
duration slope not only survives but strengthens (+0.17 → +0.23, p≈7e-14), so the effect is carried by
non-reconstructed people, not manufactured by the repair.)*

**External validation against the robot's own memory.** The re-threaded EMA is cross-checked
against the persisted `data/*/memory/homeostatic_learning.json` snapshot — the affinity the robot
*actually stored*, an artefact independent of the event log it was derived from. The check keeps two
cases strictly separate, because the split is in **memory**, not in the event log:

- **Validation — the 12 people the robot stored under a single identity:** the reconstruction
  reproduces the persisted memory **exactly (max |Δ| = 0.000)**, with the update count matching
  too. This is clean external confirmation of the reconstruction, independent of the event log.
- **Reconciliation — the 2 people the robot *forked* in memory** under case/spelling variants of one
  name (one across 3 keys, one across 2): the cleaned event log already **consolidates** each
  person's events under one key, and the update counts are **conserved exactly** (29 = 7+21+1 and
  14 = 0+14), so the re-thread yields the *correct merged* affinity that — by construction — equals
  no single memory fork. We therefore report **no per-fork residual** for them (it would be
  meaningless); we report only that leaving them unmerged fragments affinity by up to **0.93** across
  forks (one fork logs 0.00 over a live 0.9+ history). The merge does not create the signal; it
  reconciles a fragmentation the robot's own memory exhibits.

Net: the reconstruction is externally verified where memory is unambiguous, and demonstrably repairs
memory where the robot forked an identity (`outputs/rq3_memory_crosscheck.csv`). B9 shows the
mechanism works as coded; **whether the learned state tracks behaviour is B10's question.**

**B10 — the validation (all three confirmatory metrics survive BH, q < 0.05).**

1. **Manipulation check.** Feeders supplied meals at **2.7× the unconstrained rate** in
   Phase 1 (person-clustered Poisson GEE [1.2, 5.9], p = .013); the no-feed pair recorded
   **0 feeds in 15 Phase-1 interactions** (exact CI [0, 0.22] — perfect compliance, handled
   with exact intervals since separation makes a GLM unidentifiable); the feeder excess shrank
   ×0.43 [0.15, 1.22] once roles lifted.
2. **Core model** — the pre-specified form `y = role·x + phase·x + b`, one row per learning
   event, linear mixed model with person random intercept, fitted pooled and within each
   phase. For unconstrained people, +1 SD of interaction duration predicts **Δaffinity +0.17
   [+0.12, +0.21]** (p ≈ 2×10⁻¹³); the coupling is **moderated by role** (feeder × duration
   −0.15 [−0.26, −0.04], p = .006 — in the obligated-feeder group, reward delivery dominates
   the update over chat duration, and long non-feeding chats cost the robot energy) and **attenuates in Phase 2** (−0.08
   [−0.14, −0.02], p = .012). Duration is missing for ~53% of events *differentially by
   phase* (salience-link dependent), so the fully-observed doses `n_turns` and
   `active_energy_cost` serve as the pre-specified **primary** agreement checks — all three
   agree in sign, all p < 10⁻⁶ (`outputs/rq3_missingness.csv`, `outputs/rq3_model_results.csv`).
   Leave-one-person-out slope range +0.13 to +0.22; **excluding the identity-reconstructed
   people strengthens the slope to +0.23** (`outputs/rq3_affinity_repair_robustness.csv`).
3. **Downstream expression, leakage-free.** Affinity *as of yesterday* predicts today's proactive
   approaches: **rate ratio 1.55 per +1 affinity [1.09, 2.20]** (p = .016, Poisson GEE,
   controlling yesterday's interaction count and phase).

**Verdict: Supported.** The adaptive regulatory memory tracks socially embedded interaction history,
including experimentally induced feeding roles, and is subsequently expressed in the robot's
own proactive allocation. The evidence is incompatible with an interpretation of purely
uncontrolled drift in this deployment. (Limit: with 2 people per controlled role, role contrasts
are validation, not population inference.)

![Affinity trajectories](analysis/figures/fig10_affinity_trajectories.png)

*Fig 10 — unit: learning update event, n = 239 raw updates / 205 learning-eligible RQ3 events
over 14 named people plus unknown. Affinity trajectories coloured by Phase-1 experimental role
label (green = feeder, red dashed = no-feed, blue = unconstrained; dashed vertical = phase
boundary).*

![Role validation](analysis/figures/fig12_role_validation.png)

*Fig 12 — units: interaction and person-day; 217 interactions, 14 named people, 8 days;
controlled roles = 2 feeders + 2 no-feed in Phase 1. The manipulation validated: feed
probability per interaction with exact 95% CIs (left; the no-feed 0/15 and its Phase-2 release
are directly visible) and meals per person-day with the GEE rate ratios (right). Roles are
external experiment metadata, not robot software inputs.*

![Affinity dose model](analysis/figures/fig13_affinity_dose.png)

*Fig 13 — unit: learning update event; duration-linked subset n ≈ 97, fully observed dose
checks use all learning-eligible events n = 205. The core RQ3 model: raw duration-linked
learning events with per-role trends (left) and the mixed-model coefficients with 95% CIs
(right) — dose slope, role/phase moderations, and the dose-agreement slopes.*

---

## 7. Machine-learning sensitivity check *(Phase D)*

A single specificity footnote, not a modelling contribution: ~200 rows, group-aware CV
(leave-one-run/person-out), descriptive only. **D1 — does hunger add held-out signal beyond
social state?** Adding hunger state improves Engaged prediction AUC 0.669 → 0.757 (+0.088) and
PR-AUC +0.068 (leave-one-run-out GBM; the leave-one-person-out variant is weaker at 0.735, and
both are reported in `ml_model_metrics.csv`); drop-column CV ranks hunger #2/2 behind social
state. The out-of-fold predictions reproduce the Starving suppression pattern — **corroborating B4, not
proving it**. Social state dominates, which is why hunger is treated as sensitivity evidence
rather than a confirmatory mechanism test.

![ML sensitivity](analysis/figures/figD1_ml_sensitivity.png)

*Fig D1 — unit: interaction, n = 217; grouped CV leaves out runs/persons. Hunger adds held-out
signal; social state dominates; out-of-fold predictions track the Starving suppression pattern.*

---

## 8. Scorecard & synthesis

| # | Claim | Source | Outcome |
|---|---|---|---|
| RQ3 | Adaptive regulatory memory tracks interaction history and is expressed downstream | B10 | **Supported** |
| RQ1-3 | Deficit-to-action-selection coupling is behaviourally expressed | B3 | **Supported** |
| RQ1-4 | Starving reallocates priority away from social completion | B4 | **Supported** |
| RQ2-a | Deficit expression elicits recovery | B5 | **Supported** |
| RQ2-c | Replenishment reliable long-run (loop-level outcome) | B7 | **Supported** *(supporting obs.)* |
| RQ2-b | Observed Starving episodes resolve by feeding | B6 | **Supported (exploratory)** |
| RQ1-1 | Internal monitoring continuous & autonomous | B1 | **Supported** *(faithful impl. — App. A)* |
| RQ1-2 | Deficit detection correct (60/25) | B2 | **Supported** *(faithful impl. — App. A)* |

**Synthesis.** The highest-identification result is **RQ3**: the adaptive regulatory memory recovers the
experimentally induced roles, follow engagement dose in the pattern the design implies
(strengthening once identity-reconstructed people are removed), and predict the subsequent
allocation of proactive approaches — the study's experimental validation handle. **RQ1**
supplies the behavioural mechanism: a faithfully implemented **threshold homeostatic
controller** that biases action selection toward recovery actions at Hungry and reallocates priority toward
replenishment at Starving. **RQ2** shows the regulatory loop **closes in deployment**:
human engagement, with demonstrated participation from the drive, kept the robot out of
starvation ~99% of the time — a loop-level outcome of people feeding a signalling robot, not a
controller-only property.

**Scope & next steps.** Everything above is a within-deployment characterization of *one*
robot at *one* site over 8 session-days (12 runs, 14 named people, convenience sample) — existence-
and-magnitude evidence for this HRI loop, not population estimates that transfer across robots,
sites, or cohorts. Next steps: more people per controlled role (RQ3 rests on 2/role) and a
multi-site replication for generalization.

---

## 9. Reproducing this

The notebook is generated from [`analysis/build_notebook.py`](analysis/build_notebook.py):

```bash
make execute   # regenerate and execute analysis/orexigenic_analysis.ipynb
make check     # validate execution state and identity redaction
```

- Seed `SEED=42`; DB access strictly read-only; deterministic re-runs from
  `analysis/cache/*.parquet`.
- **Privacy:** identities pseudonymised to `P01…P14`; real-name and role maps live only in the
  git-ignored `analysis/private/`; `make check` scans every published surface for leaks.
- Pinned dependencies: `analysis/requirements.txt`.

**Key outputs:** [`results_summary.md`](analysis/outputs/results_summary.md),
[`verification_report.md`](analysis/outputs/verification_report.md),
[`rq3_model_results.csv`](analysis/outputs/rq3_model_results.csv),
[`rq3_missingness.csv`](analysis/outputs/rq3_missingness.csv),
[`rq3_affinity_repair_robustness.csv`](analysis/outputs/rq3_affinity_repair_robustness.csv),
[`rq3_memory_crosscheck.csv`](analysis/outputs/rq3_memory_crosscheck.csv),
[`bh_corrected_pvalues.csv`](analysis/outputs/bh_corrected_pvalues.csv),
[`success_criteria.csv`](analysis/outputs/success_criteria.csv).

---

## Appendix A — Instrumentation verification (B1/B2)

These two checks are **implementation-verification checks** and are therefore not inferential
results: they verify that the logged data behaves as the controller source specifies. They are
kept out of the results narrative because the threshold labels are derived by the same code path
being verified.

**B1/B2 — monitoring & detection.** The stomach level is a software integrator and the hunger
label is derived from it by the coded 60/25 thresholds, so "drain = 1.00× nominal" (zero-width
CI) and "transitions bracket the thresholds (1.00/1.00 accuracy)" hold by implementation. The
non-trivial content is operational rather than inferential: the
drive samples every ~2.3 s across 12 runs / 46 h, keeps draining in two runs with zero visitors
(autonomy), and shows **zero rapid reversals** at either threshold (no label flapping).
*Reading: the instrumentation behaves exactly as coded; no empirical claim is made here.*

![Drive timeline](analysis/figures/fig02_drive_timeline.png)

*Fig 2 — unit: hunger-level event, n = 165,460 across 12 monitored runs / 8 days. The
homeostatic loop made visible: autonomous sawtooth drain, discrete feeding recoveries
(arrows ∝ meal size), Starving as thin red slivers.*
