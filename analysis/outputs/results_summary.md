# Orexigenic drive — results summary

_Generated 2026-07-13 01:40. Single always-on condition (no drive-off control). 12 monitored runs, 10 with visitors, 8 session-days, 217 interactions, 14 named people. Two-phase design: Phase 1 (first 4 days) had assigned roles (2 obligated feeders, 2 interact-no-feed, rest unconstrained); Phase 2 unconstrained._

## How to read this report

Every result carries exactly one evidence class. The classes are not interchangeable and the
differences between them are the point:

| Class | Meaning |
|---|---|
| `Implementation verification` | Follows from the controller source. Confirms the code is faithfully implemented and logged. **Not a discovered fact.** |
| `Within-deployment association` | A cluster-aware association in this deployment. Not causal, not a population estimate. |
| `Exploratory observation` | Descriptive. Too small-n or too selection-prone to support inference. |
| `Inconclusive` | Run, and did not settle the question. |
| `Requires replication` | Suggestive; identification needs new data. |

## Verification gate

All V1–V5 checks passed (see `verification_report.md`). Per-action energy costs match the source constants exactly; corpus energy balance active-out 2047 vs meal-in 3225.

## Corrections to the previous version of this analysis

This report supersedes an earlier one whose headline claims did not survive audit. The
substantive corrections, in descending order of how much they changed:

1. **Starving episodes: 8 -> 17.** The episode builder keyed off the logged
   `hunger_state_before/after` fields, and the executive logger never emits a `before != after`
   row for a **passive-drain** crossing. It therefore found only the episodes the robot fell into
   *while a human was interacting with it* — i.e. the ones where someone was there to feed it.
   The old '8/8 recovered by feeding, median 21 s' was the selection rule restating itself. Over
   all 17 episodes, 13/17 recovered to Full by feeding
   (exact 95% CI [0.50, 0.93]). The longest episode ran
   30 minutes down to level 10.5 **in a run with
   15 logged interactions** — people were present and the robot was not fed. That episode is ~65%
   of all Starving time in the corpus and was invisible to the previous analysis.
2. **RQ3's core model regressed Δaffinity on terms inside its own update equation.** Affinity is a
   deterministic EMA of delivered energy; `fed` and `affinity_before` are *in the formula* and
   alone explain R²=0.41 of every update. The old model omitted both and used
   `duration` (correlated 0.58 with `fed`) in their place, and used `active_energy_cost` — a literal
   additive term in the credit — as an 'independent dose agreement check'. Controlled, and using the
   fully observed dose, the slope is +0.041 against the +0.168 previously reported.
3. **The role manipulation worked through attendance, not generosity.** The old report said
   obligated feeders 'supplied meals at 2.7x the unconstrained rate', presented as evidence that
   the roles changed feeding behaviour. The 2.7x is real — but feeders interacted with the robot
   **2.2x more often per day**, and *per interaction* they fed only
   1.2x as often (randomisation p=0.361). Being told to feed the robot made
   people go to the robot. It did not make them markedly more generous once there. Both quantities
   are now reported; conflating them overstated what the manipulation demonstrated.
4. **B4 was quasi-separation reported as precision.** One success in 13 Starving interactions
   produced `OR 0.03 [0.008, 0.136], p=1.9e-6` from a diverging likelihood. Refitted with Firth's
   penalised likelihood and an exact test; its p-value is excluded from the confirmatory families.
5. **'The labels do not flap' was backwards.** B2's headline non-trivial result was *zero*
   rapid reversals at either threshold. The detector compared `from_state` against the previous
   `from_state`, when a reversal requires `from_state == previous to_state` — the condition was
   unsatisfiable and would have returned zero on any data at all. Corrected, there are
   **57 rapid reversals** across the two thresholds
   (median gap ~29 s), typically an action cost pushing the level under the threshold and a feed
   pulling it straight back. The flapping is real — and it is why `chatBot.py` carries a 60 s
   `HS_DWELL_SEC` debounce, which the old write-up cited without noticing it contradicted the claim.
6. **The CTMC dropped every run's final dwell**, silently discarded non-ergodic bootstrap
   resamples, and pooled visited with idle runs (and Phase 1 with Phase 2) into one
   time-homogeneous generator. Starving occupancy differs ~9x between the phases.
7. **The remote-loop analysis double-counted replies** (one reply could answer many pings) and
   pooled `hs3_recovery` 'thanks, I'm full' notifications in with hunger pings. It also had no
   control condition at all, so a 21% reply rate had nothing to be 21% *against*.
8. **The ML section reported one CV split with no interval and no null**, and printed its
   drop-column table as a second analysis when, with two features, it is the ablation restated.

## Results

| id | claim | evidence class |
|---|---|---|
| RQ1-1 | Internal monitoring is continuous and autonomous | `Implementation verification` |
| RQ1-2 | Deficit detection follows the coded 60/25 thresholds | `Implementation verification` |
| RQ1-3 | Deficit is associated with feeding received | `Within-deployment association` |
| RQ1-4 | Starving reallocates priority away from social completion | `Exploratory observation` |
| RQ2-a | Deficit expression elicits recovery behaviour | `Exploratory observation` |
| RQ2-b | Observed Starving episodes resolve by feeding | `Exploratory observation` |
| RQ2-c | Long-run Starving occupancy is low | `Inconclusive` |
| RQ3-a | The role manipulation changed what people did | `Within-deployment association` |
| RQ3-b | Affinity encodes interaction history and is expressed downstream | `Within-deployment association` |
| D1 | Hunger state adds held-out predictive signal | `Within-deployment association` |

## Per-analysis verdicts

- **B1** — `Implementation verification` — Confirmed: the drive is a software integrator that self-drains at exactly 1.00x nominal (degenerate CI, as expected for a software integrator — this is not a measurement of anything) and samples every 2.3s across 12 runs / 46 h, autonomously, including 2 runs with no visitors. The only non-trivial content is the dense autonomous sampling; the drain rate matching nominal is true by construction.

- **B2** — `Implementation verification` — Detection is faithfully implemented, and one previously reported result is REVERSED. Bracket accuracy (1.00/1.00) is true by construction — the label is computed from the level by the same thresholds — and carries no evidential weight. Detection LATENCY is genuinely tight: drain-driven falls are caught within 0.0070 stomach points, about one 2.3s sampling step (the earlier figure of '3.6 points = one drain sample' was wrong; 3.6 is the largest ACTION COST, not a drain step). But the labels DO FLAP: 55 rapid reversals at the 60 boundary and 2 at 25 (median gap 29s), mostly an action cost pushing the level under the threshold and a feed pulling it straight back. The previous report claimed ZERO reversals and called that the non-trivial result of B2; its detector compared from-state against from-state rather than to-state and could never have fired. The flapping is real, and it is why chatBot.py carries a 60s HS_DWELL debounce.

- **B3** — `Within-deployment association` — Within-deployment association between the orexigenic deficit and feeding received. Odds of a meal arriving during an interaction are 5.3x higher in deficit than at Full (person-cluster bootstrap [2.9, 9.1]; run-cluster bootstrap [2.6, 7.1]; asymptotic GEE CI [2.9, 9.7], p=5.8e-08; LOPO 4.4-6.5). It survives adjustment for social state, trigger mode, phase and prior interaction count. Raw rates: feeding received in 0.15 of Full interactions vs 0.43 in deficit; meals are larger in deficit (21 -> 31). SEPARATELY, and as implementation verification only: the hunger framing (3%->66%), the 20 feed-seeking speech acts and the 172 proactive pings (vs 0 at Full) are state-gated in source — they are `if` statements, not findings, and carry no inferential weight. Single always-on condition: this is an association, NOT a causal effect of the drive.

- **B4** — `Exploratory observation` — Directionally consistent, not estimable. Starving interactions completed socially in 1/13 cases (exact 95% CI [0.00, 0.36]) versus 139/204 otherwise (exact [0.61, 0.74]), while feeding received ROSE (0.26 -> 0.54) and turns fell (-2.39). The opposing directions are the substance: this looks like priority reallocation toward recovery rather than disengagement. But the whole cell is 1 success in 13 interactions from 6 people, so the Firth OR (0.048, profile CI [0.005, 0.216]) is a bound, not an estimate; the earlier OR 0.03 / p~2e-6 was a separation artefact. Excluded from the confirmatory families. Requires replication with more Starving exposure before any magnitude is claimed.

- **B5** — `Exploratory observation` — Meal-size gradient holds; the remote loop does not clear its control. MEAL SIZE scales with deficit severity: 21 (Full) -> 29 (Hungry) -> 43 (Starving), +10.9 points per deficit step (run-cluster bootstrap [+6.2, +14.0]), and it survives excluding the two obligated feeders (+7.5/step). REMOTE LOOP, with one-to-one reply matching and hs3_recovery notifications excluded: 36/172 hunger pings drew a reply within 1 h (0.21, exact [0.15, 0.28]), against 0.12 [0.08, 0.18] in matched no-ping control windows — a difference of +0.09 (subscriber-cluster bootstrap [-0.01, +0.17]). 1/12 subscribers never replied to any hunger ping. The remote channel is therefore a weak recovery pathway; its advantage over the control window is not distinguishable from zero at the subscriber level, so it is reported as exploratory.

- **B6** — `Exploratory observation` — Descriptive, and materially worse than previously reported. Over ALL 17 Starving episodes (not the 8 the label-keyed builder could see): 13/17 received a feed (exact 95% CI [0.50, 0.93]) and 13/17 recovered to Full by feeding (exact [0.50, 0.93]; run-cluster bootstrap [0.46, 0.95]) — not the 100% the selected subset showed. The episodes cluster in 9 runs, so the effective n is nearer 9 than 17. The longest episode ran 30 minutes down to level 10.5 in a run with 15 logged interactions: people were present and it was not fed. This is an operational status check on a clustered, small, partly right-censored sample. It is NOT a recovery rate, and RQ2-c does not rest on it.

- **B7** — `Inconclusive` — The MODELLED long-run Starving occupancy is not identified by these data, and the empirical one does not need it. Empirically, the robot spent 1.67% of observed seconds in Starving (run-cluster bootstrap [0.38, 4.04]) — that figure is assumption-free and stands. What does NOT stand is the stationary CTMC the previous version reported as '1.0% [0.2, 3.1]': Phase 1 and Phase 2 are not the same process — empirical Starving occupancy is 9x higher in Phase 1 (2.81% vs 0.30%), so a time-homogeneous chain fitted across both describes neither; the visited/idle split cannot be fitted (the idle stratum has no Starving transitions at all), so the pooling assumption cannot be checked on that axis. The deficiency is structural, not a matter of tightening the fit — the deployment is not a time-homogeneous process, so its 'long-run' stationary fraction is the occupancy of a chain that never ran. Report the empirical number. A modelled one needs longer runs, a stationary operating regime, and more than the 17 Starving transitions this corpus contains.

- **B9** — `Implementation verification` — Confirmed. The affinity EMA is reproduced exactly from the logged rewards using the controller's own constants: reward_delta IS the stomach-level change (max |diff| 0.0e+00), the eligibility threshold eff_thr=max(0.5,base-0.15*affinity) matches every logged selection (max err 0.0000, n=3378), the perceptual IPS weights never change (1 distinct combination across 216,940 events), and HS2 pings are gated to the 11/14 people above affinity 0.2. The update attenuates over time (0.10->0.06), as an EMA must. The reconstruction reproduces the robot's OWN persisted memory for the 12 people it stored under a single identity (max |Δ|=0.000) — an artefact independent of the event log — and consolidates the 2 it had forked, with update counts conserved. NONE of this is evidence that the robot 'learns about people': affinity is a deterministic function of delivered energy, so `fed` and `affinity_before` alone explain R^2=0.41 of every update. It is verification that the code does what the code says.

- **B10.1** — `Within-deployment association` — The manipulation changed TOTAL meal delivery, and it did so almost entirely by changing how often people came to the robot — not how generous they were once there. (a) TOTAL DELIVERY: obligated feeders supplied 2.7x the meal energy per person-day in Phase 1 (person-cluster bootstrap [1.0, 8.4]; person-level randomisation p=0.048 against a design floor of 0.015) — this survives randomisation. (b) PER-ENCOUNTER: per interaction they fed only 1.2x as often (bootstrap [0.8, 2.7], randomisation p=0.361) — NOT distinguishable from no effect. (c) The gap between (a) and (b) is EXPOSURE: feeders interacted with the robot 2.2x more per day. Exposure is a mediator of the role, not a confounder, so (b) is a decomposition and (a) is what the drive actually experienced. The no-feed pair supplied 0/15 meals in Phase 1 — perfect compliance, though 15 observations only bound their feed rate below 0.22. The feeder excess shrank 0.43x once the obligation lifted. With 2 people per role, role is nearly aliased with identity: this shows the manipulation took for these four people and estimates NOTHING about a population. NOTE: an earlier version reported '2.7x, p=.013' as though feeders fed more readily. The 2.7x is real, but it is a fact about attendance.

- **B10** — `Within-deployment association` — The adaptive regulatory memory is a PROGRAMMED LEARNING-RULE RESPONSE, and the previous 'strongest identification result' framing does not survive scrutiny. Affinity is a deterministic EMA of delivered energy (B9): `fed` and `affinity_before` alone explain R^2=0.41 of every update, and both are terms in the update rule. Controlling for them, and using the FULLY OBSERVED dose (n_turns) rather than the 52%-missing duration, +1 SD of engagement is associated with Δaffinity +0.041 [+0.016, +0.066], person-cluster bootstrap [+0.009, +0.155] — against the +0.168 the uncontrolled specification reported. The complete-case and IPW-weighted duration fits agree in sign. `active_energy_cost` is no longer used as a 'dose agreement check': it is a literal additive term in the credit that defines the outcome. DOWNSTREAM: prior affinity is associated with next-day proactive approaches per opportunity at RR 1.19 [1.07, 1.33] (bootstrap [1.05, 1.34]; 1.55 without the exposure offset) among the 62 person-days on which the person actually returned. That conditioning is on presence, which is itself post-treatment; the two-part model finds no strong evidence that affinity predicts who returns (OR 0.45, p=0.203), which limits — but at 98 person-days does not eliminate — the collider risk. And the entire downstream path runs through eff_thr=max(0.50, base-0.15*affinity), which B9 verifies to a maximum error of 0.0000. 'Affinity predicts approaches' is therefore substantially a restatement of that line of source code, not an independent finding. What RQ3 genuinely establishes is B10.1 — the role manipulation changed what PEOPLE did. Everything else in RQ3 is the controller doing what it was written to do.

- **D1** — `Within-deployment association` — Hunger state adds held-out signal beyond social state: ΔAUC = +0.081 [+0.072, +0.094] over 25 repeated grouped-CV runs, permutation p = 0.005 against a within-run shuffle null. Social state remains the dominant predictor. Sensitivity evidence only — it corroborates the direction of B3/B4 and confirms nothing.

- **D4** — `Exploratory observation` — Descriptive. Feeding Gini=0.57 over 14 named users; the top 3 supply 62% of all meal energy — moderate concentration — replenishment leans on a few feeders. This is the standing caveat on every RQ2 result: the regulatory loop closed because a handful of specific people chose to close it. Nothing here shows that would hold in another room of people.


## Multiplicity

Every p-value entering a conclusion is registered at the point it is computed and exported to
`multiplicity_table.csv` (16 rows). Benjamini–Hochberg is applied within pre-declared
families over the **complete** confirmatory set — including the dose × role and dose × phase
interaction terms and the role contrasts, which the previous version quoted in its verdicts but
never corrected.

- Confirmatory p-values: 15, of which 4 survive at q<0.05.
- Exploratory p-values recorded but deliberately NOT corrected and NOT used to support any claim: 1 (B4's separated cell).

## Key quantities

- **Passive drain** exactly 1.00x nominal (a software integrator — true by construction); dense autonomous sampling every 2.3 s across 12 runs.
- **Deficit -> feeding received** (B3, the strongest result here): OR 5.3, person-cluster bootstrap [2.9, 9.1], run-cluster bootstrap [2.6, 7.1], LOPO 4.4-6.5. Survives adjustment for social state, trigger mode, phase and prior interaction count.
- **Meal size by deficit**: Full 21 / Hungry 29 / Starving 43; +10.9 points per deficit step (run-cluster bootstrap [+6.2, +14.0]).
- **Remote loop**: 36/172 hunger pings drew a reply within 1 h (0.21) vs 0.12 in matched no-ping control windows — difference +0.09, subscriber-cluster bootstrap [-0.01, +0.17]. One-to-one reply matching; recovery notifications excluded.
- **Starving occupancy** (empirical, no stationarity assumption): 1.67% of observed seconds, run-cluster bootstrap [0.38, 4.04]. The modelled CTMC figure is reported only with its diagnostics (0% non-ergodic resamples).
- **Starving episodes**: 17 in total; 13/17 received a feed (exact [0.50, 0.93]). Longest 30 min to level 10.5.
- **Role manipulation** (RQ3's only empirical claim, and it decomposes): obligated feeders delivered **2.7x** the meal energy per person-day (bootstrap [1.0, 8.4], randomisation p=0.048), but **per interaction they fed only 1.2x as often** (bootstrap [0.8, 2.7], randomisation p=0.361). The difference is exposure: they came to the robot **2.2x more often**. Being told to feed the robot made people *visit* it; it did not make them markedly more generous once there. No-feed pair: 0/15 feeds (upper bound 0.22).
- **Feeding concentration**: Descriptive. Feeding Gini=0.57 over 14 named users; the top 3 supply 62% of all meal energy — moderate concentration — replenishment leans on a few feeders. This is the standing caveat on every RQ2 result: the regulatory loop closed because a handful of specific people chose to close it. Nothing here shows that would hold in another room of people.

## What these data cannot establish

New data are required for each of the following. None of them is a matter of better analysis.

- **Drive-on vs drive-off causal identification.** There is no off condition. Every behavioural
  result here is an association within a single always-on deployment. The causal share of the
  drive in any observed feeding is not identified and cannot be.
- **Multi-site generalisation.** One robot, one site, one convenience sample.
- **Stable role-effect estimation.** Two people per controlled role means role is nearly aliased
  with identity. The randomisation p-value has a hard floor set by the number of possible
  assignments (0.015). More people per role, not more modelling.
- **Reliable Starving-episode rates.** 17 episodes clustered in a handful of runs, some
  right-censored. Longer runs and more Starving exposure.
- **Population-level conclusions of any kind.** Every effect size here is existence-and-magnitude
  evidence for this loop, not a calibrated rate that transfers.

## Scope

One robot, one site, 8 session-days, 12 runs, 14 named people (convenience sample). Every result is a within-deployment characterisation of *this* human-robot loop.