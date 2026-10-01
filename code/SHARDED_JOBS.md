# Sharded (micro-job) GA jobs

One shard = one `(stage, segment)` search running as its own small PBS job.
Shards cooperate through files; `merge_ga_shards.py` unions them and picks
the winning model family per segment.

## Shard table (21 jobs + 1 merge)

| Stage | Shards (segments) | Per-job shape |
|---|---|---|
| 0 CDAP | `all` (1) | 2cpu/8GB/6h |
| 1 freq | `work edu nonmand` (3) | 2cpu/32GB/12h |
| 2 day-dur | `start end budget` (3) | 2cpu/8GB/12h |
| 3 act-dur | `work edu nonmand` (3, pooled) | 2cpu/16GB/12h |
| 4 timing | 8 purposes (3) | 2cpu/8GB/12h |
| 5 mode+dest | `work edu nonmand` (3) | 4cpu/64GB (nonmand 96GB)/12h |
| 51 trip mode+dest | `work edu nonmand` (3) | 4cpu/32GB/12h |

Memory/walltime notes: right-sized from measured qld_shard1 peaks (s2-s4
<1GB; s1 OOM'd at 16GB -> 32GB; s5 search 34-48GB; s5/nonmand OOM'd at 64GB ->
96GB). The submitter also passes `GA_WALLTIME_SECONDS` per shard (stage-5
workers stop gracefully before PBS kills them) and forwards `PUBLISH_PAPER` to
the merge job so Stage-A paper artifacts refresh automatically.

## Stage 51 — conditional trip-level mode + destination

`ga_stage51` (`pipeline_logger.py`) -> `ga_hypothesis_testing._run_ga_stage5_trip`
searches the trip-level model family that prediction applies:

- **Trip destination** (`stage5b_trip_destination`): a per-purpose MNL for
  INTERMEDIATE STOPS, estimated on HHTS stop trips with the *same* estimator
  prediction uses (`_estimate_mnl_segment`). Conditioning candidate pool
  (`GA_CANDIDATE_VARS`) includes anchors and lag structure:
  `dist_to_anchor` / `log_dist_to_anchor` / `same_as_anchor` (conditional on
  the tour), `same_as_prev` / `is_inbound` (conditional on the previous stop /
  direction), `tour_mode_auto|transit|active` (conditional on tour mode),
  with `detour_dist_km` + purpose-specific `log_size` always in the backbone.
- **Trip mode**: per-purpose nested logit on HHTS trip legs
  (`NestedLogitModeModel`); the search currently selects within the model's
  supported covariate set.
- GA fitness cap: `STAGE51_DEST_EST_OBS` (default 8000) bounds the idca
  expansion per evaluation; Larch's dense `from_idca` unstacks a sampled design
  to cases x all zones, so this cap directly bounds the design matrix.
- **Memory**: the GA materialises the wide `(case, alt)` feature table ONCE per
  purpose with DuckDB (`stage5b_ca_duckdb.py` -> parquet float32) and reads it
  back with column projection per evaluation (no pandas rebuild, only the
  chromosome's active columns). `STAGE5B_CA_BACKEND=off` forces the pandas
  design; missing duckdb falls back automatically.
- Alternative sampling (Manski-Lerman) is implemented
  (`stage5b_trip_destination._sample_alt_pairs`, `STAGE5B_DEST_SAMPLE_SIZE`)
  but **default off**: Larch 6.0.46 cannot fit ragged choice sets safely —
  dense `from_idca` unstacking produces NaN utilities that trip the jax
  gradient guard, and the sparse `from_idce` path zero-gradients on jax and
  segfaults on numba (minimal repro: `_test_larch_sparse.py`). Enable once a
  fixed Larch is installed.

Winners are applied in prediction: `_run_predict_phase.phase_finalize` passes
the stage-51 destination gene vector into `pipeline_logger.stage5b` ->
`run_stage5b` -> `_predict_stop_dest`, so the searched spec lands in the
plan/paper chain.

### Tour-level stage-5 destination memory

`_dest_tree()` takes no genes (CA table, joins, availability are identical for
every chromosome), so the GA now builds it ONCE per segment and reuses it for
all evaluations (`_dest_tree_cached`, env `STAGE5_DEST_TREE_CACHE=1`,
`STAGE5_DEST_TREE_CACHE_MAX=2`). Optional GA-only case cap:
`STAGE5_DEST_EST_CASES` (0 = off; deterministic subsample of tours) — Larch's
dense `from_idca` unstacks to cases x zones, so this directly scales the
estimation matrix. Final/apply estimation is never capped.

## Model-variant shootout (inside each shard, on the winning spec)

- **S4** (`GA_S4_VARIANTS`, default `rrm;tdd`): RRM vs joint-TDD refits, plus
  optional `mnl;ordered;hybrid` (`GA_S4_VARIANTS=all`). Periods are ordinal,
  so ordered logit is legitimate here.
- **S5 mode** (`GA_S5_MODE_VARIANTS`, default `nl;mnl`): nested vs flat,
  plus optional `mxl;gse` (`=all`; GSE rows come from the search's GSE arm).
- **S5 dest** (`GA_S5_DEST_VARIANTS`, default `mnl;nl_dist`): sampled MNL vs
  distance-band NL. `exploded` adds a diagnostic note row only — its
  rank-expanded likelihood is never auto-compared. `ordered` is not offered
  for destinations (zones unordered); it is offered for S4 (periods ordinal).

**Best-model rule:** BIC/n decides *within* a comparability group only
(`s4-period` vs `s4-joint`, `s5mode` vs `s5mode-mxl`, `s5dest` vs
`s5dest-diag`). Cross-group rows sit side by side in the leaderboard.

## Outputs (per shard, then merged)

```
runs/ga_parallel/<tag>/stage<N>/shards/<segment>/
  ga_results.pkl  ga_reports/  plots/ga_all_stages.png
  variants/leaderboard.{csv,html}  variants/winner.json
  variants/winner_spec_{mode,dest}.json  variants/best_model_<v>.pkl (S4)
  shard_info.json
runs/ga_parallel/<tag>/merged/
  ga_results.pkl  ga_reports/  plots/
  shard_leaderboard.csv/html  variant_winners.json
  shard_feature_maps.json  merge_info.json
```

Convergence = per-shard history CSVs + PNGs (same writers as the legacy
path) aggregated by the merge. Models = fitted S4 variant objects + S5
winner specs (full S5 refit happens in predict from the winner spec, as
before).

## Submit

```bash
qsub -v "RUN_TAG=qld_shard1,GA_BUDGET=smoke" job_submit_ga_sharded.pbs
qsub -v "RUN_TAG=qld_shard2,GA_BUDGET=ultrafast,PUBLISH_PAPER=1" job_submit_ga_sharded.pbs
STAGES=4,5 qsub -v "RUN_TAG=s45test" job_submit_ga_sharded.pbs  # subset
DRY_RUN=1 bash job_submit_ga_sharded.pbs                        # preview
```

The QUT `cpu_inter` route is an optional escape from the large batch backlog;
it is the only alternate route currently validated by the submitter.
Use an explicit shard list so its aggregate limits are respected; selected
shards are rejected rather than silently falling back if they exceed 34GB,
8 CPUs, or 12 hours:

```bash
INTER_QUEUE=cpu_inter \
INTER_QUEUE_SHARDS='2:start;2:end;2:budget' \
qsub -v "RUN_TAG=s2_inter,INTER_QUEUE=cpu_inter,INTER_QUEUE_SHARDS=2:start;2:end;2:budget" \
  job_submit_ga_sharded.pbs
```

Stages 0 and 5 do not fit `cpu_inter`; stage 1 and stage 51 fit individually
but use 32GB each, so route them one at a time unless the queue limits have
changed. The route accepts at most 8 queued/running jobs, 34GB of running
memory, and 8 CPUs for this user.

Project accounting: all three scripts carry `#PBS -P FPAXJW8067`
(your DMP RPID) and the orchestrator forwards `-P` to every nested qsub
(children don't inherit it). Override with
`qsub -v "PROJECT_RPID=XXXXXX0000,..." ...`; `PROJECT_RPID=` (empty)
submits without `-P`.

Predict from the result with
`GA_STAGED_GA_DIR=runs/ga_parallel/<tag>/merged` — or chained, after the
merge job finishes:
`qsub -W depend=afterok:<merge_jid> -v "RUN_TAG=<tag>" job_predict_from_merged.pbs`
(which runs ga_staged reuse + validation report + Stage-B paper publish).
To predict with a
non-default winning family, set its switch on the predict job too
(`STAGE4_TIMING_MODEL=tdd`, `STAGE5_MODE_STRUCTURE=flat`,
`STAGE5_DEST_NL_DISTANCE=1`); the merge records which family won where in
`variant_winners.json`.

## Phase-sharded prediction (preferred over the 190GB monolith)

`job_predict_from_merged.pbs` runs the whole prediction in one 8cpu/190GB/20h
job. The phase-sharded chain replaces it with eight smaller jobs, chained
`afterok`, each checkpointing to `runs/ga_parallel/<tag>/predict/_phases/`
(the driver is `_run_predict_phase.py`; phase bodies mirror
`apply_ga_results_to_pipeline` / `GA_STAGED`). Checkpoints are parquet
(compressed; pickle fallback), the all-zones synth frame is never persisted,
and `spatial_plots` runs in its own job (it ignores its `synth` argument):

| Phase | Job | Resources | Produces |
|---|---|---|---|
| `prep` | `ga_pred_prep_<tag>` | 4cpu/64GB/6h | `_phases/travellers.parquet` |
| `s1` | `ga_pred_s1_<tag>` | 4cpu/24GB/4h | `stage1_output.{parquet,csv}` + `stage1_models.pkl` |
| `s23` | `ga_pred_s23_<tag>` | 4cpu/24GB/4h | `stage2/3_output.{parquet,csv}` + models |
| `s4` | `ga_pred_s4_<tag>` | 4cpu/24GB/4h | `stage4_output.{parquet,csv}` + models |
| `s5` | `ga_pred_s5_<tag>` | 4cpu/96GB/12h | `_phases/stage5_output.parquet`, `stage5_models.pkl` |
| `finalize` | `ga_pred_finalize_<tag>` | 4cpu/96GB/6h | Stage 5b + Stage 6, `_phases/plan.parquet`, `ga_staged_validation.csv` |
| `plots` | `ga_pred_plots_<tag>` | 4cpu/32GB/2h | spatial/plan plots under `predict/plots/` |
| `publish` | `ga_predict_publish_<tag>` | 4cpu/32GB/2h | `generate_validation_report.py` + Stage-B paper artifacts |

Sizes are based on measured qld_shard1 peaks (s2–s4 <1GB, s5 search 34–48GB)
plus the pipeline's streaming design (stage-5 zone plans stream to CSV,
`STAGE5_WORKERS=1` default, Stage 6 chunked at 20k persons).

```bash
qsub -v "RUN_TAG=<tag>,MERGE_JID=<merge_jid>" job_submit_predict_sharded.pbs
DRY_RUN=1 bash job_submit_predict_sharded.pbs            # preview
# re-run one phase manually (previous ones already checkpointed):
qsub -v "...same -v...,PREDICT_PHASE=s5" job_predict_phase.pbs
```

`job_predict_from_merged.pbs` remains as the monolithic fallback.
