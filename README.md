# Orexigenic Drive and Always-On Homeostatic Regulation

This report explains the analysis in
[`analysis/orexigenic_analysis.ipynb`](analysis/orexigenic_analysis.ipynb). It is written for
readers who want to understand the study logic, the main results, and the strength of the
evidence without needing to inspect every model.

The system under study is the *Orexigenic Drive* in the `alwaysOn-embodiedBehaviour` iCub
controller. The drive continuously tracks internal energy, detects hunger states, changes the
robot's behaviour when energy is low, and uses social and remote interaction to seek
replenishment. The controller pipeline is:

`perception -> salience -> executive regulation -> remote/Telegram signalling`.

## Contents

**Part I — Summary**

1. [Main findings](#1-main-findings)

**Part II — Background and methods**

2. [Research questions and study design](#2-research-questions-and-study-design)
3. [Data and analysis pipeline](#3-data-and-analysis-pipeline)
4. [How to read the statistics](#4-how-to-read-the-statistics)

**Part III — Results**

5. [RQ1: Does a deficit change behaviour?](#5-rq1-does-a-deficit-change-behaviour)
6. [RQ2: Does deficit expression close the recovery loop?](#6-rq2-does-deficit-expression-close-the-recovery-loop)
7. [RQ3: Does adaptive regulatory memory encode behaviour?](#7-rq3-does-adaptive-regulatory-memory-encode-behaviour)
8. [Machine-learning sensitivity check](#8-machine-learning-sensitivity-check)

**Part IV — Synthesis**

9. [Scorecard and synthesis](#9-scorecard-and-synthesis)

**Part V — Appendices**

- [Appendix A: Instrumentation verification](#appendix-a-instrumentation-verification)
- [Corrections](#corrections)

---

## Part I — Summary

### 1. Main findings

The analysis supports all three research questions, with different levels of evidential
strength.

**RQ1: the drive acts as a two-threshold homeostatic policy.** Monitoring and hunger-state
detection are implementation checks: the software computes them directly from the coded
thresholds. The empirical result is behavioural. When the robot crosses the Hungry threshold
(energy below 60), it shifts toward recovery-oriented action: hunger framing rises from 3% to
67%, feed-seeking acts rise from 1 to 20, proactive Telegram pings rise from 0 to 172, and
co-present feeding pursuit rises from 0.15 to 0.43. In the main clustered model, deficit
increases the odds of feeding pursuit by **4.9x [2.6, 9.5]**. When the robot reaches Starving
(energy below 25), it reallocates priority away from social completion: completed engagement
falls from 0.68 to 0.08, **OR 0.03 [0.01, 0.14]**.

**RQ2: the recovery loop closes at the deployment level.** Deficits elicit stronger recovery
behaviour: meal size increases from Full to Hungry to Starving (**21 -> 29 -> 43**), and proactive
remote pings receive modest but measurable replies (**0.21 [0.15, 0.26]** within 1 hour). As a
supporting reliability result, modelled long-run Starving occupancy is about **1.0% [0.2%,
3.1%]**. This is a property of the coupled human-robot loop, not the controller alone, and it
leans on a few responsive feeders (Gini 0.57; top 3 people supplied 62% of meals).

**RQ3: adaptive regulatory memory tracks interaction history and is later expressed in robot
behaviour.** This is the strongest identification result because it uses the Phase-1 role
manipulation as an external validation handle. Obligated feeders supplied meals at **2.7x** the
unconstrained rate [1.2, 5.9], while the no-feed pair supplied **0 feeds in 15 Phase-1
interactions**. The core model shows that +1 SD of engagement duration predicts **Delta affinity
+0.17 [+0.12, +0.21]**; the result strengthens to **+0.23** when identity-reconstructed people
are excluded. Prior-day affinity then predicts next-day proactive approaches, **RR 1.55 [1.09,
2.20]**, after controlling for recent activity.

The main limitation is scope: this is one robot, one site, 8 session-days, 12 monitored runs, and
14 named people. The results are within-deployment evidence for this human-robot regulatory loop,
not population estimates across robots, sites, or user groups.

---

## Part II — Background and methods

### 2. Research questions and study design

The study asks three connected questions.

- **RQ1** - To what extent does the orexigenic drive instantiate the four operational functions
  of classical homeostasis: (1) internal monitoring, (2) deficit detection,
  (3) deficit-to-action-selection coupling, (4) behavioural priority reallocation?
- **RQ2** - Does expressing an orexigenic deficit promote recovery-oriented engagement
  sufficient to support reliable energy replenishment in an always-on social robot?
- **RQ3** - Does the robot's learned memory of past interactions reflect participants' behaviour,
  and does that memory influence how the robot engages with people later?

For RQ1, the first two functions are software verification checks. The empirical question is
whether the internal deficit state changes behaviour. For RQ2, the key issue is whether hunger
signals lead to human engagement and feeding often enough to avoid persistent starvation. For RQ3,
this is tested by asking whether the robot's **adaptive regulatory memory** (the learned
per-person homeostatic affinity) encodes observed participant behaviour rather than uncontrolled
drift, and whether that learned state is subsequently expressed in the robot's allocation of
proactive approaches.

Two design facts shape the analysis.

**The drive was always on.** There is no off-switch control condition. RQ2 is therefore identified
within the always-on deployment by comparing behaviour across energy states: Full, Hungry, and
Starving. The key contrast is whether the robot initiates more recovery-oriented behaviour when
the internal deficit becomes stronger.

**The study had two 4-day phases with assigned roles in Phase 1.** In Phase 1, two participants
were obligated feeders, two were asked to interact but never feed, and all others were
unconstrained. In Phase 2, all constraints were lifted. These role labels were external
experimental metadata, not controller inputs. Because each controlled role has only 2 people,
role contrasts validate the manipulation but should not be read as population-level estimates.

The evidence is ordered as follows:

1. **RQ3 role manipulation and learning validation.** This is the clearest external test because
   the study deliberately induced different feeding histories.
2. **RQ3 downstream expression.** Prior affinity predicts next-day proactive approaches, so the
   learned state precedes the behaviour it predicts.
3. **RQ1 behavioural coupling.** Deficit and Starving states change action selection and priority.
4. **RQ2 reliability.** Low Starving occupancy supports loop-level recovery but depends on the
   coupled human-robot setting.
5. **Appendix checks.** Monitoring and threshold detection verify the instrumentation rather than
   provide independent empirical effects.

---

### 3. Data and analysis pipeline

The robot logged four asynchronous streams: `vision`, `salience_network`, `executive_control`,
and `chat_bot`. The `data/` folders contain eight dated snapshots. A key preparation finding was
that these snapshots are cumulative views of one growing database, not eight independent
experiments. Naively stacking them would double-count observations by roughly 4-5x.

After de-duplication, the main analysis units were:

- **12 monitored runs**, including 10 runs with visitors.
- **8 session-days**, split into Phase 1 and Phase 2.
- **217 co-present interactions** involving 14 named, pseudonymised people.
- **205 learning-eligible affinity update events** for RQ3.
- Supporting units such as person-day summaries, Telegram pings, state transitions, and hunger
  episodes.

The pipeline has five gated stages. Each stage exists to prevent a specific error from entering
the inferential analysis.

1. **Ingest and de-duplicate.** Convert cumulative snapshots into canonical runs, interactions,
   and events.
2. **Harmonise identity and time.** Apply one central identity map, pseudonymise identities, and
   stamp rows with both monotonic time and wall-clock time.
3. **Verify the logs against the controller.** Check meal deltas, energy costs, drain rate,
   thresholds, referential integrity, and clock ordering. All hard checks passed; see
   [`verification_report.md`](analysis/outputs/verification_report.md).
4. **Construct analysis tables.** Freeze the interaction, person-day, and hunger-state tables so
   each model has a clear unit of analysis.
5. **Run inference and produce figures.** RQ1 uses interaction-level behaviour, RQ2 uses recovery
   and state-occupancy analyses, and RQ3 uses person-day and learning-event analyses.

Several harmonisation choices matter for interpretation. Interactions are the anchor for joining
streams by identity and time window. `monotonic_sec` is used for within-run durations, while
`timestamp_epoch` is used for cross-stream alignment and day labels. Missing duration values are
handled before modelling: because duration is missing for about 53% of learning events and varies
by phase, fully observed doses (`n_turns` and `active_energy_cost`) are used as primary agreement
checks for RQ3.

---

### 4. How to read the statistics

The models were chosen to match the outcome being analysed and the repeated-observation
structure of the data. Many observations come from the same people, runs, or days, so the
analysis avoids simple row-independent tests as primary evidence.

| Question | Outcome | Model | Plain-language interpretation |
|---|---|---|---|
| Does deficit increase feeding pursuit? | Yes/no per interaction | Logistic GEE clustered by person | Estimates how much the odds of feeding pursuit change in deficit states while accounting for repeated observations from the same person. |
| Does Starving suppress social completion? | Yes/no per interaction | Logistic GEE clustered by person | Estimates whether completed engagement becomes less likely when the robot is Starving. |
| Did obligated feeders feed more? | Meal counts per person-day | Poisson GEE clustered by person | Estimates a meal-rate ratio between role groups. |
| Does prior affinity increase proactive approaches? | Count of next-day approaches | Poisson GEE clustered by person | Estimates a next-day approach-rate ratio while controlling for recent activity. |
| Does engagement dose change affinity? | Continuous affinity update | Linear mixed model with person intercept | Estimates whether more engagement is associated with larger affinity gains while allowing each person to have a different baseline. |
| Did the no-feed group comply? | 0 feeds in Phase 1 | Exact binomial interval | Reports the result directly because complete separation makes a standard model unstable. |
| How often is the robot Starving in the long run? | Time in hunger states | Continuous-time Markov chain with run-level bootstrap | Estimates the stationary Starving fraction implied by observed transitions and dwell times. |

Uncertainty is reported with confidence intervals and bootstrap checks where appropriate. P-values
for primary claims are Benjamini-Hochberg corrected within two pre-declared families: RQ1/RQ2
behaviour and RQ3 adaptation. All five primary inferential metrics survive correction at
q < 0.05.

The report treats small samples carefully. Starving interactions are rare (13 interactions; 8
episodes), so those results are led by effect sizes, confidence intervals, and directional
interpretation rather than by complex covariate models.

---

## Part III — Results

### 5. RQ1: Does a deficit change behaviour?

Monitoring and threshold detection are described in [Appendix A](#appendix-a-instrumentation-verification)
because they verify the implementation. RQ1's empirical test is whether the internal deficit
state changes action selection.

#### B3: Deficit-to-action coupling

The main contrast is Full versus deficit, where deficit combines Hungry and Starving. The result
is a state-contingent shift toward recovery.

| Behaviour | Full | Deficit | Interpretation |
|---|---:|---:|---|
| Hunger framing in speech | 2.8% | 66.5% | Coded gate opens hunger framing in deficit. |
| Feed-seeking speech acts | 1 | 20 | Feed-seeking appears almost entirely in deficit. |
| Proactive Telegram pings | 0 | 172 | Remote recovery attempts are deficit-gated. |
| Co-present feeding pursuit | 0.15 | 0.43 | Human-facing recovery pursuit increases. |
| Mean meal size | 21.2 | 31.4 | Meals are larger when the robot is in deficit. |

The coded rows show that the controller exposes a different action repertoire when energy is low.
The co-present pursuit and meal-size rows show how that state change is expressed in behaviour.

The inferential anchor is a person-clustered logistic GEE:
`fed01 ~ deficit` (see [Corrections](#corrections)). Deficit increases the odds of feeding pursuit by
**OR 4.9 [2.6, 9.5]**, p about 2e-6; leave-one-person-out OR range 4.1-6.1.

**Verdict: supported.** The deficit state biases action selection toward goal-directed recovery.

![Deficit to action](analysis/figures/fig04_deficit_action.png)

*Fig 4 - units: 367 interaction turns, 710 chat messages, 217 co-present interactions, and 193
deficit-gated action events. Recovery-action rates Full vs deficit are shown with bootstrap CIs,
and deficit-gated actions are plotted on the stomach-level timeline.*

#### B4: Starving priority reallocation

The next question is whether the more severe Starving state merely reduces activity or actively
changes priorities. The evidence supports priority reallocation toward recovery.

When Starving, completed social engagement is strongly suppressed: turns fall from 2.5 to 0.2,
Engaged completion falls from 0.68 to 0.08, and the model estimate is **OR 0.034 [0.008,
0.136]**. At the same time, feeding pursuit rises from 0.26 to 0.54. This pattern is not simple
disengagement; it is a shift away from social completion and toward replenishment.

The model is `reached_Engaged ~ starving + C(initial_state)`, clustered by person. Starving has
only 13 interactions, so the estimate should be read as a large directional effect rather than a
precise population estimate.

**Verdict: supported.** Starving reallocates priority away from social completion and toward
recovery.

![Prioritisation heatmap](analysis/figures/fig05_prioritisation_heatmap.png)

*Fig 5 - unit: interaction, n = 217; Starving column n = 13. Completion is high for known people
until Starving shifts priority away from social completion.*

---

### 6. RQ2: Does deficit expression close the recovery loop?

RQ1 shows that the robot changes behaviour when energy is low. RQ2 asks whether those signals are
followed by human engagement and replenishment in deployment.

#### B5: Deficit expression elicits recovery behaviour

Meal size increases with deficit severity: **Full 21, Hungry 29, Starving 43**. Proactive
Telegram pings receive replies within 1 hour at **0.21 [0.15, 0.26]** overall, and 0.26 for
Starving-specific pings. Co-present interactions are reply-bearing more often when proactive
(83%) than reactive (42%).

**Verdict: supported.** Deficit expression elicits measurable recovery behaviour, although remote
reply rates are modest.

![Remote loop](analysis/figures/fig08_remote_loop.png)

*Fig 8 - unit: proactive Telegram ping, n = 234 across 12 subscribers; response window = 1 hour.
Bars show response-to-ping rates with bootstrap 95% CIs.*

#### B6: Observed Starving episodes

All 8 observed Starving episodes received a feed, escaped Starving, and recovered to Full via
feeding. Median time to first feed was 21 seconds.

This is an operational status check, not a population recovery rate. Reliability is better
summarised by the occupancy result in B7.

#### B7: Long-run loop reliability

A continuous-time Markov chain fitted to Full, Hungry, and Starving transitions gives modelled
long-run **Starving occupancy of 1.0% [0.2%, 3.1%]** using the run-level block bootstrap. A
transition-level bootstrap gives a similar estimate, 1.1% [0.4%, 2.3%]. No absorbing starvation
state appears in the observed deployment.

This result should be read carefully. The transition rates describe the coupled human-robot loop:
the robot signalled, people responded, and feeding replenished energy. The result is therefore
not a controller-only property. It is also supporting rather than primary evidence because it is
based on few Starving transitions, pools non-identical operating regimes into one model, and
depends on a small number of responsive feeders. The pooled model is conservative because idle
drain-only periods push Starving occupancy upward.

**Verdict: supported as a loop-level outcome.** In this deployment, human engagement maintained
the robot outside Starving for roughly 97-100% of the time.

![Steady state](analysis/figures/fig09_steady_state.png)

*Fig 9 - unit: CTMC state sojourn/transition reconstructed from 165,460 hunger events across 12
monitored runs. Modelled occupancy is close to empirical time-occupancy (Starving 1.1% vs 1.8%);
mean Starving sojourn is 163 s.*

---

### 7. RQ3: Does adaptive regulatory memory encode behaviour?

RQ3 links the controller mechanism to an external validation test. First, the report verifies what
the robot learns. Then it asks whether the learned state tracks experimentally induced
interaction histories and predicts later robot behaviour.

#### B9: Mechanism verification

The salience network implements per-person adaptive regulatory memory as **homeostatic affinity**:
an exponentially weighted moving average of normalised homeostatic reward, bounded in [-1, +1].
In plain terms, people who help reduce the robot's deficit can become easier for the robot to
select proactively later.

The mechanism behaves as coded:

- Affinity updates converge over time: mean absolute update falls from 0.10 to 0.06.
- Perceptual IPS weights stay fixed; learning acts through the per-person eligibility threshold:
  `eff_thr = max(0.50, base_ss - 0.15 * affinity)`.
- The threshold reconstruction matches logged values to 1e-4.
- High-affinity people can receive up to about a 0.14 lower approach threshold.
- Hungry-state pings are gated to the 11 of 14 people above affinity 0.20.

The affinity reconstruction was also checked against the robot's persisted
`homeostatic_learning.json` memory. For the 12 people stored under one identity, the
reconstruction exactly matches the robot's memory (max absolute difference 0.000) and update
counts match. For 2 people whose identity was forked in memory under spelling or case variants,
the cleaned event log conserves update counts and produces the correct merged affinity. Leaving
those forks unmerged would fragment affinity by up to 0.93.

Because this identity repair touches an RQ3 outcome, the core dose-to-affinity model was refit
excluding the reconstructed people. The result strengthens from +0.17 to **+0.23** (p about
7e-14), so the dose-affinity association is not manufactured by the repair.

#### B10: External validation

The Phase-1 role manipulation provides the main test of whether affinity tracks behaviour.

**Manipulation check.** Obligated feeders supplied meals at **2.7x** the unconstrained rate in
Phase 1 [1.2, 5.9], p = .013. The no-feed pair complied perfectly, with **0 feeds in 15 Phase-1
interactions**. Once roles were lifted in Phase 2, the feeder excess shrank.

**Core affinity model.** The pre-specified model tests whether engagement dose predicts change in
affinity while allowing role and phase to moderate that relationship. For unconstrained people,
+1 SD of interaction duration predicts **Delta affinity +0.17 [+0.12, +0.21]**, p about 2e-13.
The relationship is moderated by role: for obligated feeders, reward delivery dominates the
update more than chat duration does. It also attenuates in Phase 2. Fully observed dose checks
using `n_turns` and `active_energy_cost` agree in sign, and leave-one-person-out slopes range
from +0.13 to +0.22.

**Downstream expression.** Affinity as of yesterday predicts today's proactive approaches:
**rate ratio 1.55 per +1 affinity [1.09, 2.20]**, p = .016, controlling for yesterday's
interaction count and phase. Because predictor and outcome are on different days, this test is
leakage-free by construction.

**Verdict: supported.** Adaptive regulatory memory tracks socially embedded interaction history,
including experimentally induced feeding roles, and is later expressed in the robot's proactive
allocation. The result is not consistent with purely uncontrolled drift in this deployment.

![Affinity trajectories](analysis/figures/fig10_affinity_trajectories.png)

*Fig 10 - unit: learning update event, n = 239 raw updates / 205 learning-eligible RQ3 events
over 14 named people plus unknown. Trajectories are coloured by Phase-1 role; the dashed line
marks the phase boundary.*

![Role validation](analysis/figures/fig12_role_validation.png)

*Fig 12 - units: interaction and person-day; 217 interactions, 14 named people, 8 days. The
left panel shows feed probability per interaction with exact CIs; the right panel shows meals
per person-day with GEE rate ratios.*

![Affinity dose model](analysis/figures/fig13_affinity_dose.png)

*Fig 13 - unit: learning update event; duration-linked subset n about 97, fully observed dose
checks use all 205 learning-eligible events. The figure shows raw duration-linked events,
per-role trends, and mixed-model coefficients with 95% CIs.*

---

### 8. Machine-learning sensitivity check

The machine-learning analysis is a sensitivity check, not a confirmatory model. Its purpose is to
ask whether hunger state adds held-out predictive information beyond social state.

Using group-aware cross-validation on about 200 interaction rows, adding hunger state improves
Engaged prediction AUC from **0.669 to 0.757** (+0.088) and PR-AUC by **+0.068**. Drop-column
cross-validation ranks hunger behind social state, meaning social context remains the stronger
predictor. Out-of-fold predictions reproduce the Starving suppression pattern, which
corroborates B4 but does not prove it.

![ML sensitivity](analysis/figures/figD1_ml_sensitivity.png)

*Fig D1 - unit: interaction, n = 217; grouped CV leaves out runs or people. Hunger adds held-out
signal, social state dominates, and out-of-fold predictions track the Starving suppression
pattern.*

---

## Part IV — Synthesis

### 9. Scorecard and synthesis

| Claim | Source | Outcome |
|---|---|---|
| Adaptive regulatory memory tracks interaction history and is expressed downstream | B10 / RQ3 | **Supported** |
| Deficit-to-action-selection coupling is behaviourally expressed | B3 / RQ1 | **Supported** |
| Starving reallocates priority away from social completion | B4 / RQ1 | **Supported** |
| Deficit expression elicits recovery behaviour | B5 / RQ2 | **Supported** |
| Long-run replenishment is reliable at the loop level | B7 / RQ2 | **Supported** as a supporting observation |
| Observed Starving episodes resolve by feeding | B6 / RQ2 | **Supported** as exploratory |
| Internal monitoring is continuous and autonomous | B1 / Appendix A | **Supported** as implementation verification |
| Deficit detection follows the 60/25 thresholds | B2 / Appendix A | **Supported** as implementation verification |

Taken together, the results show a coherent homeostatic loop. RQ1 establishes the behavioural
mechanism: hunger thresholds change action selection, and Starving shifts priority away from
ordinary social completion. RQ2 shows that this signalling loop was sufficient in deployment to
maintain low Starving occupancy, while making clear that this is a human-robot loop-level result.
RQ3 provides the strongest validation: experimentally induced feeding histories are reflected in
the learned affinity state, and that state predicts later proactive allocation.

The scope remains deliberately narrow. These are within-deployment findings for one robot at one
site over 8 session-days. The next scientific step is replication with more people per controlled
role and across additional sites.

---

## Part V — Appendices

### Appendix A: Instrumentation verification

B1 and B2 are kept separate from the main empirical results because they verify that the logged
data behaves as the controller source specifies. They are not independent inferential tests.

**B1: internal monitoring.** The stomach level is a software integrator. Passive drain matches
the nominal rate exactly (1.00x), as expected for implemented software rather than a noisy
biological measurement. The non-trivial operational result is dense autonomous sampling: about
every 2.3 seconds across 12 runs / 46 hours, including two runs with no visitors.

**B2: deficit detection.** Hunger labels are derived from the coded thresholds: Full to Hungry at
60, and Hungry to Starving at 25. Transitions bracket those thresholds with 1.00/1.00 accuracy.
The useful check is that there were zero rapid reversals at either threshold, so the labels do
not flap around the boundary.

**Reading:** the instrumentation behaves exactly as coded. The empirical homeostasis claims rest
on the behavioural and learning results in RQ1-RQ3.

---

### Corrections

**B3 model formula.** An earlier version of this README stated the B3 inferential anchor as
`fed_here ~ deficit + C(initial_state)`. The code (`analysis/build_notebook.py`) actually fits
`fed01 ~ deficit` — a person-clustered logistic GEE with no `initial_state` covariate. `fed01` is
a binary feeding-pursuit indicator local to the B3 co-present-interaction table; `fed_here` is a
separate variable (`meals_eaten_count > 0`) used later, in B5. The two were conflated in the
write-up. The `C(initial_state)` covariate does appear in the notebook, but as part of the B4
model (`reached_ss4 ~ starving + C(initial_state)`), not B3.

The reported effect size, CI, and p-value (OR 4.9 [2.6, 9.5], p about 2e-6, LOPO range 4.1-6.1)
match the notebook's actual `fed01 ~ deficit` output, so the B3 verdict is unaffected. Only the
formula text and variable name were stale; they have been corrected in place above rather than
silently rewritten, per the note in this section.

![Drive timeline](analysis/figures/fig02_drive_timeline.png)

*Fig 2 - unit: hunger-level event, n = 165,460 across 12 monitored runs / 8 days. The timeline
shows autonomous drain, discrete feeding recoveries, and brief Starving periods.*
