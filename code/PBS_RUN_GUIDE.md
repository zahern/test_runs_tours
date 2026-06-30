# PBS Run Guide

This document explains the new PBS launcher files and the commands they submit.

## Files

- `submit_pipeline_runs.sh`
  Submits one or more PBS jobs for the pipeline.
- `pbs_pipeline_mode.pbs`
  Generic PBS worker script that runs a single mode.
- `pipeline_logger.py`
  Main entry point for `main`, `ga`, `ga_staged`, `safety_baseline`, and `safety_nosafety`.
- `safety_experiment_analysis.py`
  Post-processing script for the safety experiment comparison.
- `_run_ga_single_stage.py`
  Runs exactly one GA stage (1-5) and writes a one-stage `ga_results.pkl` artifact.
- `merge_ga_results.py`
  Merges per-stage GA artifacts into one combined `ga_results.pkl`.
- `job_ga_stage_single.pbs`
  PBS worker that runs one GA stage (used by fan-out stage jobs).
- `job_merge_ga_results.pbs`
  PBS worker that merges completed stage outputs.
- `job_submit_ga_parallel_chain.pbs`
  PBS submitter job that launches stage 1..5 in parallel, then merge, then `ga_staged` prediction.

## Parallel Stage Chain (GA Fan-out + Merge + Prediction)

This workflow is for running stages `1..5` simultaneously across separate PBS jobs,
joining the stage models, then running `ga_staged` validation/prediction from the
merged `ga_results.pkl`.

### One-command submit

```bash
qsub -v RUN_TAG=exp_sa_parallel_001,GA_ESTIMATOR=sa_bandit,GA_BUDGET=fast,GA_N_RESTARTS=0 job_submit_ga_parallel_chain.pbs
```

What the submitter does:

1. Submits 5 stage jobs in parallel (`job_ga_stage_single.pbs`): stage 1,2,3,4,5.
2. Submits merge job (`job_merge_ga_results.pbs`) with `afterok` dependency on all 5 jobs.
3. Submits prediction job (`pbs_pipeline_mode.pbs` with `MODE=ga_staged`) dependent on merge,
   with `GA_STAGED_REUSE_ESTIMATOR=1` and `GA_STAGED_GA_DIR=<RUN_ROOT>/<RUN_TAG>/merged`.

### Key environment variables

- `RUN_TAG`
  Logical experiment id. Artifacts are written under `RUN_ROOT/<RUN_TAG>/...`.
- `RUN_ROOT`
  Base directory for parallel-stage artifacts (default `runs/ga_parallel`).
- `GA_ESTIMATOR`
  Estimator selection (default `sa_bandit`).
- `GA_BUDGET`
  Budget profile (`smoke|ultrafast|fast|standard|thorough`).
- `GA_N_RESTARTS`
  Restart count override (default `0` in the parallel chain).
- `STAGE_SELECT`, `STAGE_WALLTIME`
  Resource shape for each stage job.
- `PRED_SELECT`, `PRED_WALLTIME`
  Resource shape for the final `ga_staged` prediction job.

### Typical resource override example

```bash
qsub -v RUN_TAG=exp1,GA_ESTIMATOR=sa_bandit,GA_BUDGET=standard,GA_N_RESTARTS=0,STAGE_SELECT=1:ncpus=8:mem=128GB,STAGE_WALLTIME=36:00:00,PRED_SELECT=1:ncpus=8:mem=250GB,PRED_WALLTIME=24:00:00 job_submit_ga_parallel_chain.pbs
```

### Outputs

- Stage artifacts:
  `runs/ga_parallel/<RUN_TAG>/stage1/ ... stage5/`
- Merged artifact for reuse:
  `runs/ga_parallel/<RUN_TAG>/merged/ga_results.pkl`
- Final prediction/validation run:
  standard `ga_staged` run directory under `RUN_ROOT` (managed by `pipeline_logger.py`).

## Run Isolation

PBS-submitted runs now get a unique `PIPELINE_RUN_TAG`, which defaults to the
PBS job id. `pipeline_logger.py` appends that tag to the output folder name.

Examples:

- `main` might write to `runs/main__123456.server/`
- `ga` might write to `runs/ga__123457.server/`
- `main --zone 40` might write to `runs/main_40__123458.server/`

This prevents repeated submissions of the same mode from overwriting earlier
results. Different jobs of the same mode now stay isolated by default.

Local ad hoc runs that do not set `PIPELINE_RUN_TAG` keep the simpler legacy
folder names such as `runs/main/`.

## Modes

- `main`
  Runs the main estimation and synthetic-population pipeline.
- `ga`
  Runs the GA-only feature-selection workflow.
- `ga_staged`
  Runs staged GA plus full population application/validation.
- `safety_baseline`
  Runs the main pipeline with safety skims included in Stage 5.
- `safety_nosafety`
  Runs the main pipeline with safety skims excluded in Stage 5.
- `safety_analysis`
  Runs `safety_experiment_analysis.py` to compare the two safety runs and generate plots/reporting.

## Main Commands

Submit the standard three runs in parallel:

```bash
./submit_pipeline_runs.sh
```

Submit the standard three runs sequentially:

```bash
./submit_pipeline_runs.sh --sequential
```

Submit selected modes only:

```bash
./submit_pipeline_runs.sh --modes "main ga"
```

Submit one or more modes for a specific zone or comma-separated zones:

```bash
./submit_pipeline_runs.sh --modes "main" --zone 40
./submit_pipeline_runs.sh --modes "main ga" --zone 40,124,200
```

Include the safety experiment jobs:

```bash
./submit_pipeline_runs.sh --include-safety --sequential
```

Request more CPUs and some threaded math inside each job:

```bash
./submit_pipeline_runs.sh --cpus 4 --threads 2
```

Submit a GPU-requesting run:

```bash
./submit_pipeline_runs.sh --modes "main" --gpu 1 --accelerator gpu
```

## What Each Option Does

- `--modes "..."`
  Which jobs to submit, in order.
- `--parallel`
  Submit all requested jobs independently.
- `--sequential`
  Submit jobs with `afterok` dependencies so each job starts only after the previous one succeeds.
- `--zone ARG`
  Passes the second positional argument into `pipeline_logger.py`. Use this for a zone number, comma-separated zone list, or a file path.
- `--cpus N`
  Sets the PBS `ncpus` request.
- `--threads N`
  Sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` inside the job.
- `--mem SIZE`
  Sets the PBS memory request.
- `--walltime HH:MM:SS`
  Sets the PBS walltime request.
- `--gpu N`
  Adds `ngpus=N` to the PBS resource request.
- `--accelerator cpu|gpu`
  Controls `PIPELINE_ACCELERATOR`, which `pipeline_logger.py` now maps onto `JAX_PLATFORMS` only when the scheduler explicitly says CPU or GPU.
- `--include-safety`
  Submits `safety_baseline`, `safety_nosafety`, and `safety_analysis` as a dependency chain.
  In `--parallel` mode, that safety chain runs independently of the main/ga/ga_staged jobs.
  In `--sequential` mode, the safety chain starts after the previous job in your main list succeeds.

## Why JAX Was Forced To CPU

Previously, `pipeline_logger.py` always set:

```python
os.environ.setdefault('JAX_PLATFORMS', os.environ.get('JAX_PLATFORMS', 'cpu'))
```

That meant any run that imported JAX/Larch through this entry point was pinned to CPU unless the environment had already overridden it before Python started. In practice, that made cluster GPU requests ineffective for this script.

Now the behavior is:

- If `PIPELINE_ACCELERATOR=cpu`, the script sets `JAX_PLATFORMS=cpu`.
- If `PIPELINE_ACCELERATOR=gpu`, the script sets `JAX_PLATFORMS=gpu`.
- If nothing is declared, the script defaults to CPU for safety.

This keeps CPU-only jobs stable while allowing GPU jobs to actually use the requested accelerator.

## Will More CPUs Help?

Usually a little, not a lot.

What is likely to benefit:

- BLAS / LAPACK math if NumPy or SciPy hits threaded kernels.
- Some Numba/OpenMP-backed work.
- Stage 4 scheduling has an explicit worker-process path.

What is not strongly CPU-scaled right now:

- The outer pipeline orchestration in `pipeline_logger.py`.
- Most GA search logic, which is still a serial fit/evaluate loop.
- Several Larch/JAX estimation flows that are dominated by model compilation or single-process work.

Practical recommendation:

- Start with `--cpus 2 --threads 1` or `--cpus 4 --threads 2`.
- Prefer running `main`, `ga`, and `ga_staged` as separate jobs in parallel over giving one job many CPUs.
- Only request GPU when you also set `--accelerator gpu` and the target path is expected to use JAX on GPU.

## Example Workflows

Standard production submission:

```bash
./submit_pipeline_runs.sh --parallel
```

Standard production plus safety experiment chain:

```bash
./submit_pipeline_runs.sh --include-safety --sequential
```

Zone-specific debugging run:

```bash
./submit_pipeline_runs.sh --modes "main ga_staged" --zone 40 --cpus 2 --threads 1 --mem 96GB
```

GPU test run:

```bash
./submit_pipeline_runs.sh --modes "main" --gpu 1 --accelerator gpu --cpus 4 --threads 2
```

## Monitoring

Monitor jobs with:

```bash
qstat -u $USER
```

Inspect run outputs under:

```bash
runs/<mode>__<jobid>/
```

Common logs to watch:

```bash
tail -f log_main.err
tail -f log_ga.err
tail -f log_ga_staged.err
tail -f log_safety_baseline.err
tail -f log_safety_nosafety.err
tail -f log_safety_analysis.err
```