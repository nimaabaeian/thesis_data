# Reproducibility report

_Generated 2026-07-13T12:08:18+00:00_

## Build status: **PASS**

Every acceptance gate passed. The numbers in `results_summary.md` and `README.md`
were produced by this build, from the inputs hashed below, using the constants
verified against the pinned controller commits.

## What can and cannot be reproduced

**The raw data are not in this repository and cannot be.** The SQLite databases hold
participant faces, names and chat transcripts; `analysis/private/` holds the identity,
role and attendance maps. Both are git-ignored.

- **Full independent reproduction requires controlled access to the excluded data.**
  No claim to the contrary is made here.
- **Without it, a third party can still verify** that the build is deterministic, that
  the analysis constants match the pinned controller source, and — if they hold the
  data — that their copy hashes to the values recorded below.

## Git

- branch: `master`
- commit: `13507ed6ae625612c152da6a309c94e3325f7b23`
- working tree dirty: `no`

## Environment

- python `3.13.9` on `Linux-x86_64`
- lock file: `analysis/requirements.lock`

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
| pytest | 9.1.1 |

## Controller constants

**PASS** — every constant verified key-by-key against `social-robot-embodied-behaviour-architecture.git` at each pinned
deployment commit, with no drift between them:

- `b84e2f80fe72`
- `be46774403f2`

## Inputs

- manifest: `analysis/data_manifest.json` (67 files, 4,688,350 rows)
- verified against disk: **yes**
- collection dates: 15-06, 16-06, 17-06, 18-06, 22-06, 23-06, 25-06, 29-06

## Notebook

- status: **EXECUTED CLEAN**
- code cells: 47, unexecuted: 0, errored: 0

## Output hashes

| artifact | sha256 (first 16) |
|---|---|
| `outputs/active_cost_table.csv` | `5d5b31fd425d4ebb` |
| `outputs/b10_downstream_stage_models.csv` | `2f4c094d27b8e536` |
| `outputs/b10_downstream_stages.csv` | `b3f370e617464d3c` |
| `outputs/b10_scheduled_day_panel.csv` | `35c7bf96249be7ca` |
| `outputs/b10_scheduled_day_panel_reconciliation.csv` | `35c7bf96249be7ca` |
| `outputs/b2_detection_check.csv` | `64a403679e64bd05` |
| `outputs/b2_flapping_events.csv` | `ff4ababd2fdd54d2` |
| `outputs/b2_transition_counts.csv` | `e714f98ec6c59651` |
| `outputs/b3_adjusted_models.csv` | `6c011090fa3942ac` |
| `outputs/b4_starving_exact.csv` | `23da55c0d7a57b4b` |
| `outputs/b5_meal_size_by_state.csv` | `45190f99fbee9d44` |
| `outputs/b5_ping_by_subscriber.csv` | `18b3a0586af39219` |
| `outputs/b5_ping_control_pairs.csv` | `71ac3400fd7ae776` |
| `outputs/b5_ping_response_windows.csv` | `58741b59d8d00c12` |
| `outputs/b6_episode_outcomes.csv` | `156b792f9a6c61f2` |
| `outputs/b7_stratified_occupancy.csv` | `6c676f61dd8b0b40` |
| `outputs/b7_terminal_segments.csv` | `e32a35299d1f4eda` |
| `outputs/b9_eligibility_profile.csv` | `aca512d12ee802c3` |
| `outputs/b9_mechanism_check.csv` | `0f8b208b25b4a3e8` |
| `outputs/bh_corrected_pvalues.csv` | `e845c9f732950be4` |
| `outputs/constants_check.json` | `ae3660f35d4923b3` |
| `outputs/d4_feeding_concentration.csv` | `9718d09957d963e9` |
| `outputs/hs3_episodes.parquet` | `79a9ea187e68e70b` |
| `outputs/hs_crossing_log_gap.csv` | `76dfd0907d86b3db` |
| `outputs/hs_transitions.parquet` | `20284ac9c61e6ba4` |
| `outputs/hs_transitions_logged_view.parquet` | `6d4e74173ec87f0e` |
| `outputs/master_interactions.parquet` | `b923300aa6c154b4` |
| `outputs/ml_ablation.csv` | `0d0a637e19beb371` |
| `outputs/ml_ablation_delta.csv` | `5553d3c6897a7634` |
| `outputs/ml_model_metrics.csv` | `fc20cfd24a51d06b` |
| `outputs/multiplicity_table.csv` | `e845c9f732950be4` |
| `outputs/quality_report.md` | `49751acce4a83ebe` |
| `outputs/results_summary.md` | `41600073200aceed` |
| `outputs/rq3_affinity_repair_robustness.csv` | `1a98ecc95bec3003` |
| `outputs/rq3_dose_specification_comparison.csv` | `f49592fe37d71c39` |
| `outputs/rq3_ipw_balance.csv` | `f1e34949feca9e1e` |
| `outputs/rq3_ipw_truncation_sensitivity.csv` | `d2b8307c0fc3a127` |
| `outputs/rq3_memory_crosscheck.csv` | `bc53166f22a732f7` |
| `outputs/rq3_missingness.csv` | `25bc3a5e3675cc7f` |
| `outputs/rq3_missingness_model.csv` | `f367e7c12413cd1e` |
| `outputs/rq3_model_results.csv` | `6baeea2efe971243` |
| `outputs/small_cluster_sensitivity.csv` | `fc2f7d51fc2671ee` |
| `outputs/success_criteria.csv` | `c9092cfdf3d375c2` |
| `outputs/verification_report.md` | `8e61b44e72d517f0` |
| `figures/fig02_drive_timeline.png` | `5dbe059deec7f99c` |
| `figures/fig04_deficit_action.png` | `ae5ba8a6d1dee7c3` |
| `figures/fig05_prioritisation_heatmap.png` | `85b5c4ec49d13e1d` |
| `figures/fig08_remote_loop.png` | `6bdf57ad93017428` |
| `figures/fig09_steady_state.png` | `5f8f28c50eb2f9e2` |
| `figures/fig10_affinity_trajectories.png` | `2f99cb3132c548dd` |
| `figures/fig12_role_validation.png` | `bf46260d735de759` |
| `figures/fig13_affinity_dose.png` | `deaa8ca028414091` |

## Determinism

- global seed `SEED=42`, set once and used by every bootstrap, permutation and fit.
- databases opened read-only (`mode=ro`).
- `make all` deletes `analysis/outputs/`, `figures/` and `cache/` before executing, so
  no stale artifact can survive into a build.
- `make determinism` executes the whole analysis twice from clean and compares every
  generated CSV, Parquet table and report byte-for-byte.
