# Reproducibility report

_Generated 2026-07-12 23:40 UTC_

## What can and cannot be reproduced

**The raw data are not in this repository and cannot be.** The SQLite databases hold
participant faces, names and chat transcripts, and `analysis/private/` holds the identity
canonicalisation and role-assignment maps. Both are git-ignored.

Consequently:

- **Full independent reproduction requires controlled access to the excluded data and the
  role mappings.** There is no way around this and no claim to the contrary is made here.
- **Without that data, a third party can still verify**: that the build is deterministic
  (fixed seed, pinned dependencies), that the analysis constants match the pinned controller
  source (`check_constants.py`), and — if they hold the data — that their copy hashes to the
  values in `data_manifest.json`.

## Git state

- commit: `a2b3cd56b0839fc9c9af3ed0127dcd7757da08a8`
- branch: `master`
- dirty: `yes`

## Software

- python: `3.13.9` (Linux x86_64)

| package | version |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.4.4 |
| scipy | 1.18.0 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.9.0 |
| lifelines | 0.30.3 |
| matplotlib | 3.10.8 |
| pyarrow | 23.0.1 |
| nbformat | 5.10.4 |
| nbconvert | 7.17.0 |

- environment lock: `MISSING`
- pinned deps: `analysis/requirements.txt`

## Controller constants

- **UNVERIFIED** (`SKIPPED_UNPINNED`). The controller source was not available, or
  `controller_source.json` still has `commit: UNPINNED`. The constants used by the
  analysis (thresholds, meal sizes, energy costs, affinity EMA) are therefore taken on
  trust. Pin the commit and re-run `make check-constants` to close this.

## Inputs

- manifest: `analysis/data_manifest.json` (67 files, 4,688,350 rows)
- collection dates: 15-06, 16-06, 17-06, 18-06, 22-06, 23-06, 25-06, 29-06
- private maps hashed (contents never published): 3

| file | rows | sha256 (first 16) |
|---|---:|---|
| `data/15-06/data_collection/executive_control.db` | 23989 | `46ee7fe260b5705e` |
| `data/15-06/data_collection/salience_network.db` | 30848 | `f2e480cc1f0701cd` |
| `data/15-06/data_collection/chat_bot.db` | 245 | `fd62d8b646545acd` |
| `data/15-06/data_collection/vision.db` | 83341 | `a7697b21b342e927` |
| `data/16-06/data_collection/executive_control.db` | 47568 | `b34e00c0eab1665f` |
| `data/16-06/data_collection/salience_network.db` | 58342 | `4cc2a694998ef144` |
| `data/16-06/data_collection/chat_bot.db` | 626 | `e552adea6b08ff98` |
| `data/16-06/data_collection/vision.db` | 159404 | `b5a98517c30b1426` |
| `data/17-06/data_collection/executive_control.db` | 71163 | `9381b5262db911ee` |
| `data/17-06/data_collection/salience_network.db` | 83453 | `11b1acd5442c5906` |
| `data/17-06/data_collection/chat_bot.db` | 782 | `0ea45bb89401a987` |
| `data/17-06/data_collection/vision.db` | 230880 | `d7fdfd4d0632fe8e` |
| `data/18-06/data_collection/executive_control.db` | 90878 | `0f854f954908a560` |
| `data/18-06/data_collection/salience_network.db` | 117816 | `c73579d46e60a9f6` |
| `data/18-06/data_collection/chat_bot.db` | 1004 | `e2179920d6a1afa7` |
| `data/18-06/data_collection/vision.db` | 321050 | `1c49ce526ddbac93` |
| `data/22-06/data_collection/executive_control.db` | 109369 | `4a430d854655d10c` |
| `data/22-06/data_collection/salience_network.db` | 149728 | `0bb0eb2f4633b050` |
| `data/22-06/data_collection/chat_bot.db` | 1205 | `1043fcc598ddea39` |
| `data/22-06/data_collection/vision.db` | 414395 | `3ae3bc158b210ad2` |
| `data/23-06/data_collection/executive_control.db` | 129694 | `ca37043879aa1fe2` |
| `data/23-06/data_collection/salience_network.db` | 171954 | `19a2b120379ec75a` |
| `data/23-06/data_collection/chat_bot.db` | 1347 | `b2f9aa0c33d8147e` |
| `data/23-06/data_collection/vision.db` | 479034 | `256cbbf33c54e14a` |
| `data/25-06/data_collection/executive_control.db` | 152561 | `a2b87c4f3030371c` |
| `data/25-06/data_collection/salience_network.db` | 200298 | `c612d0430c9520dd` |
| `data/25-06/data_collection/chat_bot.db` | 1438 | `70f8e54c4baf619b` |
| `data/25-06/data_collection/vision.db` | 550833 | `ce1742dabc9444ad` |
| `data/29-06/data_collection/executive_control.db` | 166078 | `6878a4ba6f211898` |
| `data/29-06/data_collection/salience_network.db` | 222972 | `62b67c3087455a94` |
| `data/29-06/data_collection/chat_bot.db` | 1519 | `b0b28cae015a9b48` |
| `data/29-06/data_collection/vision.db` | 614536 | `0457db1e7184eb27` |

## Notebook execution

- status: **EXECUTED CLEAN**
- code cells: 46, unexecuted: 0, errored: 0

## Outputs

| file | bytes | sha256 (first 16) |
|---|---:|---|
| `outputs/active_cost_table.csv` | 1383 | `5d5b31fd425d4ebb` |
| `outputs/b10_downstream_panel.csv` | 6932 | `696e5727a689499d` |
| `outputs/b10_person_day_exposure.csv` | 3228 | `8a93850cba2c077f` |
| `outputs/b2_detection_check.csv` | 454 | `64a403679e64bd05` |
| `outputs/b2_flapping_events.csv` | 4387 | `ff4ababd2fdd54d2` |
| `outputs/b2_transition_counts.csv` | 250 | `e714f98ec6c59651` |
| `outputs/b3_adjusted_models.csv` | 872 | `ee3cc164557e777b` |
| `outputs/b4_starving_exact.csv` | 218 | `23da55c0d7a57b4b` |
| `outputs/b5_meal_size_by_state.csv` | 273 | `45190f99fbee9d44` |
| `outputs/b5_ping_by_subscriber.csv` | 185 | `18b3a0586af39219` |
| `outputs/b5_ping_response_windows.csv` | 748 | `6c1c626b31fe52b1` |
| `outputs/b6_episode_outcomes.csv` | 360 | `156b792f9a6c61f2` |
| `outputs/b7_stratified_occupancy.csv` | 483 | `90886d67241ababc` |
| `outputs/b7_terminal_segments.csv` | 147 | `e32a35299d1f4eda` |
| `outputs/b9_eligibility_profile.csv` | 445 | `aca512d12ee802c3` |
| `outputs/b9_mechanism_check.csv` | 638 | `0f8b208b25b4a3e8` |
| `outputs/bh_corrected_pvalues.csv` | 3634 | `d830f20b5efa0cca` |
| `outputs/constants_check.json` | 143 | `e27db99a35037b11` |
| `outputs/hs3_episodes.parquet` | 21396 | `7f6737c98d839f26` |
| `outputs/hs_crossing_log_gap.csv` | 137 | `76dfd0907d86b3db` |
| `outputs/hs_transitions.parquet` | 20745 | `20284ac9c61e6ba4` |
| `outputs/hs_transitions_logged_view.parquet` | 24059 | `6d4e74173ec87f0e` |
| `outputs/master_interactions.parquet` | 68980 | `b923300aa6c154b4` |
| `outputs/ml_ablation.csv` | 555 | `0d0a637e19beb371` |
| `outputs/ml_ablation_delta.csv` | 429 | `5553d3c6897a7634` |
| `outputs/ml_model_metrics.csv` | 873 | `fc20cfd24a51d06b` |
| `outputs/multiplicity_table.csv` | 3634 | `d830f20b5efa0cca` |
| `outputs/quality_report.md` | 697 | `3939ba80ae4a3b59` |
| `outputs/results_summary.md` | 20631 | `3c0bc50342e1d892` |
| `outputs/rq3_affinity_repair_robustness.csv` | 297 | `1a98ecc95bec3003` |
| `outputs/rq3_dose_specification_comparison.csv` | 569 | `45dbb73c06300392` |
| `outputs/rq3_memory_crosscheck.csv` | 972 | `bc53166f22a732f7` |
| `outputs/rq3_missingness.csv` | 147 | `25bc3a5e3675cc7f` |
| `outputs/rq3_missingness_model.csv` | 523 | `92941ef29ddb5ee8` |
| `outputs/rq3_model_results.csv` | 2688 | `6baeea2efe971243` |
| `outputs/small_cluster_sensitivity.csv` | 1502 | `ad7422850b6d271d` |
| `outputs/success_criteria.csv` | 9628 | `5553a645958f3d76` |
| `outputs/verification_report.md` | 1442 | `4e4daa654f2360a8` |
| `figures/fig02_drive_timeline.png` | 579246 | `5dbe059deec7f99c` |
| `figures/fig04_deficit_action.png` | 346952 | `1ceb05e3cfafd312` |
| `figures/fig05_prioritisation_heatmap.png` | 187574 | `85b5c4ec49d13e1d` |
| `figures/fig08_remote_loop.png` | 199344 | `e7f3b2c0cb4d54f3` |
| `figures/fig09_steady_state.png` | 222416 | `5f8f28c50eb2f9e2` |
| `figures/fig10_affinity_trajectories.png` | 427492 | `2f99cb3132c548dd` |
| `figures/fig12_role_validation.png` | 219343 | `ed3f2a883f6d1047` |
| `figures/fig13_affinity_dose.png` | 259772 | `1ebece8d247771ca` |

## Determinism

- global seed `SEED=42` set once in the notebook setup cell and used by every bootstrap,
  permutation and model fit.
- databases opened read-only (`mode=ro`).
- `make execute` regenerates every artifact under `analysis/outputs/` and
  `analysis/figures/` from a clean state (`make clean-outputs` first).
