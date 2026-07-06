# Verification report

_Generated 2026-07-06 12:35_

| id | check | severity | result | detail |
|---|---|---|---|---|
| V1a | HS1<->HS2 transitions bracket 60 | soft | PASS | (bracket_frac=1.00, n=121) |
| V1b | HS2<->HS3 transitions bracket 25 | soft | PASS | (bracket_frac=1.00, n=14) |
| V1c | meal_delta == SMALL/MEDIUM/LARGE const | hard | PASS | (mismatches=0) |
| V1d | fitted drain rate ~ nominal 100/(4h) | soft | PASS | (median=6.94e-03/s vs nominal=6.94e-03/s, n_seg=85) |
| V1e | stomach level in [0,100] | hard | PASS |  |
| V2a | salience attempts resolve to interactions | soft | PASS | (match=1.000, n=103) |
| V2b | hunger events' exec_interaction_id resolve | soft | PASS | (match=0.993, n=847) |
| V2c | interaction_turns subset of interactions | soft | PASS | (match=1.000) |
| V3a | 8 session-days present | hard | PASS | (days=8, runs=10) |
| V3b | no NULL run_id in interactions | hard | PASS |  |
| V3c | no NULL run_id in hunger events | hard | PASS |  |
| V4a | monotonic_sec non-decreasing within run | hard | PASS | (violations=0) |
| V5a | active cost only on active_cost rows | hard | PASS | (active rows=867) |
| V5b | each action maps to one deterministic cost (== source) | hard | PASS | (varying/mismatch=0) |
| V5c | corpus energy balance reported | soft | PASS | (active_out=2047.0, meal_in=3225.0, net=+1178.0) |

**Corpus energy balance:** active-out 2047.0 vs meal-in 3225.0 (net +1178.0 over 10 runs).