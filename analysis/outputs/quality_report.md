# Quality report

_Generated 2026-07-13 15:17_

| check | result | detail |
|---|---|---|
| v_quality_hunger_invalid_levels empty | PASS | (rows=0) |
| v_quality_interaction_missing_metadata empty | PASS | (rows=0) |
| v_quality_salience_missing_metadata empty | PASS | (rows=0) |
| v_quality_chat_missing_metadata empty | PASS | (rows=0) |
| stomach level in [0,100] | PASS | (min=0.00, max=100.00) |
| no feeding with meal_delta==0 | PASS | (n_feeds=108) |
| meal_delta matches SMALL/MEDIUM/LARGE constants | PASS | (mismatches=0) |
| HS label consistent with level thresholds | PASS | (mismatch rows=0/165460) |
| passive/active level changes are non-increasing | PASS | (violations=0/165336) |