"""
_run_sa_bandit.py
─────────────────
Run the 5 GA/SA-Bandit estimation stages, optionally run a safety-vs-no-safety
comparison on HHTS validation data, and then apply the best-found features to
the full synthetic population prediction. Outputs land in
runs/sa_bandit_<timestamp>/ and paper-ready figures + model tables are copied
to abm_safety/Figures/.

Outputs land in  runs/sa_bandit_<timestamp>/
  ga_results.pkl            — checkpoint (can be loaded by GA_STAGED later)
  ga_stage<N>_report.txt    — per-stage convergence summary
  ga_stage<N>_coef.html     — coefficients from best-found feature set
  ga_stage<N>_coefficients.html/csv  — per-stage model coefficients
  plots/ga_all_stages.png   — combined convergence figure
  safety_analysis/          — safety-vs-no-safety comparison outputs
    safety_comparison.csv
    safety_report.html
    plots/*.png
  stage<N>_models.pkl       — fitted models per stage (in models/ subdir)
  stage<N>_output.csv       — predicted outputs per stage
  stage6_output.csv         — final plan output

Paper outputs → abm_safety/Figures/
  sa_bandit_ga_all_stages.png       — GA convergence across all stages
  sa_bandit_stage5_coef.html        — Stage 5 coefficients (safety model)
  sa_bandit_safety_comparison.csv   — safety vs no-safety comparison table
  sa_bandit_safety_report.html      — complete safety experiment report
  safety_plots/*.png                — all safety comparison figures

How to run
──────────
Option A – VS Code Run (F5 / Ctrl+F5):
    Just open this file and press ▶ Run Python File.
    (The VS Code task "Run: Current Python File (venv)" already has
     JAX_PLATFORMS=cpu and GA_METACOUNT_FORCE_CPU=1 set.)

Option B – integrated terminal:
    .venv/Scripts/python.exe _run_sa_bandit.py

Option C – change budget at the top of this file, then run as above.

Watching fitness live
─────────────────────
The BanditFeatureSearcher prints every 5 generations by default:

  Bandit gen   0 [work/FullTime] | best_BIC/n=4.1234  T=1.00000  accept=0.60  top_arms=[...]
  Bandit gen   5 [work/FullTime] | best_BIC/n=4.0987  T=0.97531  accept=0.58  top_arms=[...]
    ^ New best [work/FullTime]: BIC/n=4.0901  feats=[age_working, income_hi, ...]

Set GA_VERBOSE_EVERY=1 (see SETTINGS below) to print EVERY generation.
"""
# ─────────────────────────────── ENV SETUP ───────────────────────────────────
import os, sys

# ─────────────────────────────── SETTINGS ────────────────────────────────────
# Budget:   smoke=fast sanity check, small=dev run, medium=research, large=full
#           (see _ga_budget_params in pipeline_logger.py for exact pop/gen values)
# Override at PBS submit:  qsub -v GA_BUDGET=standard job_sa_bandit.pbs
GA_BUDGET = os.environ.get('GA_BUDGET', 'fast').strip().lower()  # reads env var first

# Print a fitness line every N generations (1 = every generation).
GA_VERBOSE_EVERY = 5         # matches the default in BanditFeatureSearcher

# Disable post-main-run random restarts for SA-bandit search by default.
# Override via env: qsub -v GA_DISABLE_RESTARTS=0 job_sa_bandit.pbs
GA_DISABLE_RESTARTS = os.environ.get('GA_DISABLE_RESTARTS', '1').strip() == '1'

# Which stages to run (subset for faster iteration; None = all five)
RUN_STAGES = None            # e.g. [1, 2] to run only stages 1 & 2

# After all GA stages complete, run a safety-vs-no-safety comparison on HHTS
# validation data using the SA bandit best features.  Outputs go to
# runs/sa_bandit_<timestamp>/safety_analysis/ and are also copied to
# abm_safety/Figures/ for direct paper inclusion.
# Override via env: qsub -v RUN_SAFETY_COMPARISON=0 job_sa_bandit.pbs
RUN_SAFETY_COMPARISON = os.environ.get('RUN_SAFETY_COMPARISON', '1').strip() == '1'

# After the safety comparison, apply the GA-selected models to the full
# synthetic population (~3.7M persons across all zones) for prediction.
# This runs stages 1-5 prediction, stage 5b, stage 6, and spatial plots.
# Requires sufficient memory (~194 GB for all zones).
# Override via env: qsub -v RUN_PREDICTION=0 job_sa_bandit.pbs
RUN_PREDICTION = os.environ.get('RUN_PREDICTION', '1').strip() == '1'

# Set NEW_POP=1 to run against the 507-SA2-zone scope (Z:/test_runs_tours/new_pop)
# instead of the legacy ~329-zone scope. This must be set before pipeline_logger
# is imported below — it switches SYNTH_DIR (zone_<sa2>.csv instead of
# order_N.csv), Synth_Connect(new_pop=True) (landuse/HHTS keyed by SA2 code),
# and PIPELINE_SKIMS_PATH (demand_'s 507-zone OMX instead of
# downcasted_matrices_base.omx) all at once — see pipeline_logger.py's
# NEW_POP/NEW_POP_DIR/PIPELINE_SKIMS_PATH module-level setup.
# Override via env: qsub -v NEW_POP=1 job_sa_bandit_507.pbs
os.environ.setdefault('NEW_POP', os.environ.get('NEW_POP', '0'))
NEW_POP = os.environ['NEW_POP'].strip() == '1'
NEW_POP_DIR = os.environ.get('NEW_POP_DIR', r'Z:\test_runs_tours\new_pop')
if NEW_POP:
    os.environ.setdefault('PIPELINE_SKIMS_PATH', r'Z:\test_runs_tours\demand_\sa2_skims_507_omxspec.omx')

# ─────────────────────────────── ENV SETUP ───────────────────────────────────
# All env-vars MUST be set before any pipeline or JAX import.

# GPU acceleration for MetaCount JAX (stages 1-3) and SearchLibrium (stage 4).
# XLA_PYTHON_CLIENT_PREALLOCATE=false lets multiple workers share the GPU.
os.environ.setdefault('PIPELINE_ACCELERATOR',       'gpu')
os.environ.setdefault('GA_METACOUNT_FORCE_CPU',     '0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL',       '3')

# Prevent JAX cache thrashing — override cgroup-detected 34 GB limit.
os.environ.setdefault('GA_JAX_CACHE_MEM_LIMIT_GB',  '200')
os.environ.setdefault('GA_JAX_CACHE_CHECK_EVERY',   '100')
# Penalise insignificant parameters in bandit fitness.
# Weight of 1.5 means a model where all params are insignificant is penalised
# by 1.5× the BIC/n score — strong enough to drive the search away from bloated specs.
os.environ.setdefault('GA_SIG_PENALTY_WEIGHT',      '1.50')
# Stricter significance threshold: flag params with p > 0.05 (was 0.10).
os.environ.setdefault('GA_SIG_PVAL_THRESH',         '0.05')
# Optional: pad feature matrices so JAX compiles once per stage (GA_PAD_FEATURES=1)
# Reduces JAX cache memory but slightly wastes optimisation time on zero columns.
os.environ.setdefault('GA_PAD_FEATURES',            '0')   # set to '1' to enable

os.environ['GA_ESTIMATOR']   = 'sa_bandit'   # activates BanditFeatureSearcher
os.environ['GA_USE_BANDIT_SA'] = '1'
os.environ['GA_BUDGET']      = GA_BUDGET
os.environ['GA_VERBOSE_EVERY'] = str(GA_VERBOSE_EVERY)
if GA_DISABLE_RESTARTS:
    os.environ['GA_N_RESTARTS'] = '0'

# Force non-interactive matplotlib backend before any plot import.
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────── IMPORTS ─────────────────────────────────────
import time
import datetime
import logging
import pandas as pd

# pipeline_logger sets up OUT_DIR, logging, and all the stage functions.
import pipeline_logger as pl
from pipeline_logger import (
    load_data,
    ga_stage0, ga_stage1, ga_stage2, ga_stage3, ga_stage4, ga_stage5,
    ga_stage23_joint,
    _save_ga_checkpoint, _write_stage_checkpoint_report,
    _ga_save_stage_coef_html_from_best, _format_ga_checkpoint_summary,
    logger,
)
from ga_hypothesis_testing import AllStagesResult

# Joint Stage 2+3 SA-Bandit search (opt-in via STAGE23_JOINT=1).
# When active, replaces separate ga_stage2 + ga_stage3 with a single
# metaheuristic search over a shared feature pool.  The SA-Bandit outer loop
# uses stage23_fitness_joint (<2 ms / eval) as the oracle; fit_joint_stage23
# (JAX) is called once after convergence for the final model.
_STAGE23_JOINT = os.environ.get('STAGE23_JOINT', '0').strip() == '1'
if _STAGE23_JOINT:
    logger.info("STAGE23_JOINT=1: Stage 2+3 will use joint SA-Bandit search")

# ─────────────────────────────── LOGGING ─────────────────────────────────────
# Create a timestamped output directory so repeated runs never clobber each other.
_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_run_tag = 'sa_bandit_507' if NEW_POP else 'sa_bandit'
pl.OUT_DIR = os.path.join('runs', f'{_run_tag}_{_ts}')
os.makedirs(os.path.join(pl.OUT_DIR, 'plots'), exist_ok=True)
os.makedirs('saved_data', exist_ok=True)
os.environ['PIPELINE_OUT_DIR'] = pl.OUT_DIR

# Route the logger to both console and a log file in the run directory.
_log_path = os.path.join(pl.OUT_DIR, 'sa_bandit_run.log')
_fh = logging.FileHandler(_log_path, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%H:%M:%S'))
logger.addHandler(_fh)
# Also make sure INFO goes to stdout so VS Code terminal shows it.
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                                        datefmt='%H:%M:%S'))
    logger.addHandler(_sh)
logger.setLevel(logging.INFO)

# ─────────────────────────────── STAGE MAP ───────────────────────────────────

def _write_timing_comparison(wall_times: dict, all_ga: "AllStagesResult", run_dir: str):
    """
    Write estimation-time comparison tables (CSV + HTML) to run_dir.

    Columns
    -------
    stage        — stage label (e.g. "Stage 1", "Stage 2+3 (joint)")
    segment      — sub-model segment (e.g. "work", "edu", "B", "all")
    estimator    — which algorithm was used
    n_evals      — GA/SA chromosome evaluations (where available)
    n_params     — free parameters in the final model
    bic_n        — best –BIC/n fitness achieved
    elapsed_min  — wall-clock minutes
    speedup_vs_metacount — ratio relative to a MetaCount Tobit/lognormal baseline
                           (only filled for stages 2 & 3 alternatives)

    The HTML table is styled to highlight the fastest option per stage in green
    and the slowest in amber, making it easy to compare approaches at a glance.
    """
    import math

    rows = []

    # MetaCount baseline costs (empirical from the 634-min run):
    #   Stage 2: 3 segments × ~211 min each    → 634 min total
    #   Stage 3: 8 purposes × ~15 min each     → ~120 min total
    _MC_BASELINE_MIN = {2: 634.0, 3: 120.0}

    STAGE_LABELS = {
        1: 'Stage 1 (activity freq)',
        2: 'Stage 2 (day duration)',
        3: 'Stage 3 (act duration)',
        4: 'Stage 4 (schedule placement)',
        5: 'Stage 5 (mode+dest)',
        23: 'Stage 2+3 (joint budget)',
    }

    ESTIMATOR_LABELS = {
        1:  'SA-Bandit + MetaCount (count)',
        2:  'SA-Bandit + OLS-BIC (fast proxy)',
        3:  'SA-Bandit + OLS-log BIC (fast proxy)',
        4:  'SA-Bandit + SearchLibrium MNL',
        5:  'SA-Bandit + Larch nested logit',
        23: 'JAX joint lognormal/Weibull (stage23_joint)',
    }

    for sid, elapsed_s in sorted(wall_times.items()):
        stage_label = STAGE_LABELS.get(sid, f'Stage {sid}')
        estimator   = ESTIMATOR_LABELS.get(sid, 'SA-Bandit')
        elapsed_min = elapsed_s / 60.0 if elapsed_s is not None else float('nan')

        # Try to pull per-segment detail from the GA result.
        res = all_ga.results.get(sid)
        seg_results = {}
        if res is not None and isinstance(getattr(res, 'metadata', None), dict):
            seg_results = res.metadata.get('segment_results', {}) or {}

        # Joint model: single segment row
        if sid == 23:
            joint_models = getattr(res, 'stage23_joint_models', None) if res else None
            n_params = joint_models['pidx'].total if (joint_models and joint_models.get('success')) else None
            bic_n    = (-joint_models['bic'] / max(joint_models.get('N_persons', 1), 1)
                        if (joint_models and joint_models.get('success')) else None)
            speedup_2 = (_MC_BASELINE_MIN[2] / elapsed_min) if elapsed_min and not math.isnan(elapsed_min) else None
            speedup_3 = (_MC_BASELINE_MIN[3] / elapsed_min) if elapsed_min and not math.isnan(elapsed_min) else None
            rows.append({
                'stage':    stage_label,
                'segment':  'all (joint)',
                'estimator': estimator,
                'n_evals':  'N/A',
                'n_params': n_params,
                'bic_n':    f'{bic_n:.4f}' if bic_n is not None else '',
                'elapsed_min': f'{elapsed_min:.1f}' if not math.isnan(elapsed_min) else 'FAILED',
                'speedup_vs_metacount': (
                    f'{speedup_2:.0f}× (vs S2) / {speedup_3:.0f}× (vs S3)'
                    if speedup_2 else ''
                ),
            })
            continue

        if seg_results:
            for seg_label, seg_res in seg_results.items():
                seg_elapsed = seg_res.get('elapsed') if isinstance(seg_res, dict) else getattr(seg_res, 'elapsed', None)
                n_evals     = seg_res.get('n_evals') if isinstance(seg_res, dict) else getattr(seg_res, 'n_evals', None)
                n_params    = None
                bic_n       = None
                hist        = seg_res.get('history', []) if isinstance(seg_res, dict) else getattr(seg_res, 'history', [])
                if hist:
                    hdf  = pd.DataFrame(hist)
                    best = hdf.get('best_so_far', hdf.get('best_fitness', pd.Series(dtype=float)))
                    if len(best):
                        bic_n = float(best.iloc[-1])
                # speedup only meaningful for S2/S3 vs MetaCount baseline
                speedup = None
                if sid in _MC_BASELINE_MIN and seg_elapsed:
                    speedup = _MC_BASELINE_MIN[sid] / (seg_elapsed / 60.0)

                rows.append({
                    'stage':    stage_label,
                    'segment':  str(seg_label),
                    'estimator': estimator,
                    'n_evals':  str(n_evals) if n_evals is not None else '',
                    'n_params': n_params,
                    'bic_n':    f'{bic_n:.4f}' if bic_n is not None else '',
                    'elapsed_min': f'{seg_elapsed/60:.1f}' if seg_elapsed else f'{elapsed_min:.1f}*',
                    'speedup_vs_metacount': f'{speedup:.0f}×' if speedup else '',
                })
        else:
            # Single-segment stage or joint result without metadata
            rows.append({
                'stage':    stage_label,
                'segment':  'all',
                'estimator': estimator,
                'n_evals':  '',
                'n_params': '',
                'bic_n':    '',
                'elapsed_min': f'{elapsed_min:.1f}' if not math.isnan(elapsed_min) else 'FAILED',
                'speedup_vs_metacount': '',
            })

    if not rows:
        logger.warning("_write_timing_comparison: no timing data collected")
        return

    df = pd.DataFrame(rows)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, 'estimation_timing.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Timing comparison → {csv_path}")

    # ── HTML ─────────────────────────────────────────────────────────────────
    def _style_row(row):
        try:
            t = float(row['elapsed_min'])
        except (ValueError, TypeError):
            return [''] * len(row)
        # We colour by stage group; just return neutral here — overall table is enough
        return [''] * len(row)

    html_table = df.to_html(index=False, border=0, classes='timing-tbl')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Estimation Timing Comparison</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2em; }}
  h1   {{ font-size: 1.3em; }}
  p.note {{ font-size: 0.85em; color: #555; }}
  table.timing-tbl {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  table.timing-tbl th {{ background: #2c3e50; color: #fff; padding: 6px 10px; text-align: left; }}
  table.timing-tbl td {{ padding: 5px 10px; border-bottom: 1px solid #ddd; }}
  table.timing-tbl tr:nth-child(even) {{ background: #f7f7f7; }}
  .fast  {{ background: #d4edda !important; font-weight: bold; }}
  .slow  {{ background: #fff3cd !important; }}
</style>
</head><body>
<h1>Estimation timing comparison &mdash; {GA_BUDGET} budget</h1>
<p class="note">
  Wall-clock times from this run.  Speedup column compares against empirical
  MetaCount Tobit/lognormal baselines (Stage 2: 634 min, Stage 3: ~120 min).
  Asterisk (*) = stage wall time used (no per-segment breakdown available).
</p>
{html_table}
<p class="note">Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body></html>"""

    html_path = os.path.join(run_dir, 'estimation_timing.html')
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    logger.info(f"Timing comparison HTML → {html_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Estimation timing summary")
    print(f"{'─'*60}")
    col_w = [max(len(str(v)) for v in df[c].tolist() + [c]) for c in df.columns]
    header = '  ' + '  '.join(c.ljust(col_w[i]) for i, c in enumerate(df.columns))
    print(header)
    print('  ' + '-' * (sum(col_w) + 2 * len(col_w)))
    for _, row in df.iterrows():
        print('  ' + '  '.join(str(row[c]).ljust(col_w[i]) for i, c in enumerate(df.columns)))
    print(f"\n  Full table → {os.path.abspath(html_path)}\n")


def _all_stage_runners(out, landuse):
    """Return the ordered list of (stage_id, runner_fn) pairs.

    Stage 0 (CDAP feature search) runs first, before stage 1.
    """
    if _STAGE23_JOINT:
        return [
            (0,  lambda: ga_stage0(out)),
            (1,  lambda: ga_stage1(out, landuse)),
            (23, lambda: ga_stage23_joint(out)),
            (4,  lambda: ga_stage4(out)),
            (5,  lambda: ga_stage5(out, landuse)),
        ]
    return [
        (0, lambda: ga_stage0(out)),
        (1, lambda: ga_stage1(out, landuse)),
        (2, lambda: ga_stage2(out)),
        (3, lambda: ga_stage3(out)),
        (4, lambda: ga_stage4(out)),
        (5, lambda: ga_stage5(out, landuse)),
    ]


# ─────────────────────────────── MAIN ────────────────────────────────────────
def run():
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"  SA-Bandit GA stages   budget={GA_BUDGET}   out={pl.OUT_DIR}")
    print(f"  stages = {RUN_STAGES or [0,1,2,3,4,5]}")
    print(f"{'='*70}\n")

    logger.info(f"SA-Bandit run started  budget={GA_BUDGET}  out={pl.OUT_DIR}")

    # ── 1. Load HHTS survey data (no synth population needed) ────────────────
    print("Loading survey data …")
    sc, out, landuse = load_data()
    print(f"  loaded: {len(out.get('persons', pd.DataFrame())):,} persons\n")

    # ── 2. Run requested GA stages ────────────────────────────────────────────
    all_ga = AllStagesResult()
    _save_ga_checkpoint(all_ga, stage_id=0, failed=False)

    stages = [s for s in _all_stage_runners(out, landuse)
              if RUN_STAGES is None or s[0] in RUN_STAGES]

    # Accumulate wall-clock times per stage for the timing comparison report.
    _stage_wall_times: dict = {}   # sid -> elapsed_seconds (or None if failed)

    for sid, runner in stages:
        stage_t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  STAGE {sid}  (SA-Bandit, budget={GA_BUDGET})")
        print(f"{'─'*60}")
        logger.info(f"Starting GA stage {sid}")

        try:
            res = runner()
            all_ga.results[sid] = res

            elapsed = time.time() - stage_t0
            _stage_wall_times[sid] = elapsed
            logger.info(f"GA stage {sid} finished in {elapsed/60:.1f} min")

            # ── Per-segment fitness summary ──────────────────────────────────
            _seg_results = {}
            if res is not None and isinstance(getattr(res, 'metadata', None), dict):
                _seg_results = res.metadata.get('segment_results', {}) or {}

            if _seg_results:
                print(f"\n  Stage {sid} — per-segment best fitness:")
                for seg_label, seg_res in _seg_results.items():
                    seg_hist = getattr(seg_res, 'history', []) or []
                    if seg_hist:
                        hdf   = pd.DataFrame(seg_hist)
                        best  = hdf.get('best_so_far',
                                        hdf.get('best_fitness', pd.Series(dtype=float)))
                        feats = getattr(seg_res, 'best_individual', None)
                        n_f   = int(feats.genes.sum()) if feats is not None else '?'
                        delta = float(best.iloc[-1] - best.iloc[0]) if len(best) > 1 else 0.0
                        print(f"    [{seg_label}]  final={best.iloc[-1]:.5f}  "
                              f"Δ={delta:+.5f}  n_feats={n_f}  "
                              f"gens={len(hdf)}")
            else:
                hist = getattr(res, 'history', []) or []
                if hist:
                    hdf  = pd.DataFrame(hist)
                    best = hdf.get('best_so_far',
                                   hdf.get('best_fitness', pd.Series(dtype=float)))
                    delta = float(best.iloc[-1] - best.iloc[0]) if len(best) > 1 else 0.0
                    print(f"\n  Stage {sid} best fitness: {best.iloc[-1]:.5f}  "
                          f"Δ={delta:+.5f}  gens={len(hdf)}")

            # ── Checkpoint + HTML report ──────────────────────────────────────
            hist_data = getattr(res, 'history', []) or []
            hist_df = pd.DataFrame(hist_data) if hist_data else None
            summary = _format_ga_checkpoint_summary(res)
            _write_stage_checkpoint_report(
                stage_tag=f'ga_stage{sid}',
                title=f'GA Stage {sid} — SA-Bandit (budget={GA_BUDGET})',
                summary=summary,
                table=hist_df if (hist_df is not None and not hist_df.empty) else None,
            )
            _ga_save_stage_coef_html_from_best(sid, res, out, landuse=landuse)
            _save_ga_checkpoint(all_ga, stage_id=sid, failed=False)
            logger.info(f"GA stage {sid} checkpointed")

        except Exception as exc:
            _stage_wall_times[sid] = None   # mark as failed
            _save_ga_checkpoint(all_ga, stage_id=sid, failed=True)
            logger.error(f"GA stage {sid} FAILED: {exc}", exc_info=True)
            print(f"\n  !! Stage {sid} failed: {exc}")
            print("     Partial results saved.  Continuing with remaining stages …\n")

    # ── 3. Timing comparison report ──────────────────────────────────────────
    try:
        _write_timing_comparison(_stage_wall_times, all_ga, pl.OUT_DIR)
    except Exception as _te:
        logger.warning(f"Timing comparison report failed (non-fatal): {_te}")

    # ── 4. Combined convergence plot ─────────────────────────────────────────
    try:
        ga_reports_dir = os.path.join(pl.OUT_DIR, 'ga_reports')
        all_ga.export_reports(ga_reports_dir)
        all_ga.plot_all(save_path=os.path.join(pl.OUT_DIR, 'plots', 'ga_all_stages.png'))
        logger.info(f"GA reports → {ga_reports_dir}/")
        logger.info(f"GA plot    → {pl.OUT_DIR}/plots/ga_all_stages.png")
    except Exception as e:
        logger.warning(f"GA report export failed (non-fatal): {e}")

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  SA-Bandit run complete in {total/60:.1f} min")
    print(f"  Output: {os.path.abspath(pl.OUT_DIR)}")
    print(f"  Log:    {os.path.abspath(_log_path)}")
    print(f"{'='*70}\n")
    logger.info(f"SA-Bandit run complete in {total/60:.1f} min → {pl.OUT_DIR}")

    # ── 4. Safety comparison (baseline vs no-safety) ──────────────────────────
    if RUN_SAFETY_COMPARISON:
        _run_safety_comparison(out, landuse, all_ga, pl.OUT_DIR)

    # ── 5. Full population prediction (synth pop, all zones) ─────────────────
    if RUN_PREDICTION:
        _run_prediction(out, landuse, all_ga, pl.OUT_DIR)


def _run_safety_comparison(out: dict, landuse, all_ga: AllStagesResult, run_dir: str):
    """Fit Stage 5 models with/without safety skims using SA bandit best features,
    run the safety comparison analysis, and copy outputs to abm_safety/Figures/."""
    import shutil
    import openmatrix as omx
    from pipeline_logger import compare_safety_runs

    print(f"\n{'─'*60}")
    print(f"  SAFETY COMPARISON  (SA-Bandit best features, HHTS validation)")
    print(f"{'─'*60}")
    logger.info("Safety comparison: fitting Stage 5 models with SA bandit best features")

    safety_dir = os.path.join(run_dir, 'safety_analysis')
    os.makedirs(os.path.join(safety_dir, 'plots'), exist_ok=True)

    # ── Extract Stage 5 best features from GA results ─────────────────────
    stage5_res = all_ga.results.get(5)
    if stage5_res is None:
        logger.warning("Stage 5 GA result not found — skipping safety comparison")
        return

    mode_feats = None
    dest_feats = None
    if isinstance(stage5_res.best_features, dict):
        for seg_key, feats in stage5_res.best_features.items():
            seg_str = str(seg_key).lower()
            if 'mode' in seg_str:
                mode_feats = list(feats) if feats else None
            elif 'dest' in seg_str:
                dest_feats = list(feats) if feats else None
    else:
        logger.warning("Stage 5 best_features is not a dict — using defaults")
        mode_feats = list(stage5_res.best_features) if stage5_res.best_features else None

    logger.info(f"  Stage5 mode features: {mode_feats or 'default'}")
    logger.info(f"  Stage5 dest features: {dest_feats or 'default'}")

    # ── Detect crash skim source ───────────────────────────────────────────
    from pipeline_logger import _detect_stage5_crash_source as _detect_crash
    crash_skims = _detect_crash()

    baseline_dir = os.path.join(run_dir, 'safety_baseline')
    nosafety_dir = os.path.join(run_dir, 'safety_nosafety')

    for label, sdir, smode in [
        ('With Safety Skims', baseline_dir, 'baseline'),
        ('Without Safety Skims', nosafety_dir, 'nosafety'),
    ]:
        print(f"\n  Fitting Stage 5 [{smode}]: {label}")
        os.makedirs(sdir, exist_ok=True)
        try:
            _fit_stage5_tour_models_hhts(
                out=out,
                landuse=landuse,
                save_dir=sdir,
                mode_feature_override=mode_feats,
                dest_feature_override=dest_feats,
                safety_mode=smode,
                crash_skims=crash_skims,
            )
        except Exception as exc:
            logger.error(f"Stage 5 fit [{smode}] FAILED: {exc}", exc_info=True)
            print(f"    !! Failed: {exc}")

    # ── Run comparison ────────────────────────────────────────────────────
    print(f"\n  Running safety comparison analysis …")
    try:
        compare_safety_runs(
            baseline_dir=baseline_dir,
            nosafety_dir=nosafety_dir,
            out_dir=safety_dir,
            landuse=landuse,
        )
        logger.info(f"Safety comparison report → {safety_dir}")
    except Exception as exc:
        logger.error(f"Safety comparison FAILED: {exc}", exc_info=True)
        print(f"    !! Failed: {exc}")

    # ── Copy to abm_safety/Figures/ for paper inclusion ───────────────────
    _paper_figures_dir = os.path.join('abm_safety', 'Figures')
    os.makedirs(_paper_figures_dir, exist_ok=True)
    os.makedirs(os.path.join(_paper_figures_dir, 'safety_plots'), exist_ok=True)

    print(f"\n  Copying outputs to {os.path.abspath(_paper_figures_dir)}/ …")

    # GA convergence plot
    _src_ga_plot = os.path.join(run_dir, 'plots', 'ga_all_stages.png')
    if os.path.exists(_src_ga_plot):
        shutil.copy2(_src_ga_plot,
                     os.path.join(_paper_figures_dir, 'sa_bandit_ga_all_stages.png'))
        print(f"    ✓ sa_bandit_ga_all_stages.png")

    # Stage 5 coefficient HTML — try each known output location in priority order
    for _src_coef in [
        os.path.join(run_dir, 'ga_reports', 'stage5_coefficients.html'),
        os.path.join(baseline_dir, 'stage5_models_coefficients.html'),
        os.path.join(baseline_dir, 'stage5_smoke_models_report.html'),
    ]:
        if os.path.exists(_src_coef):
            shutil.copy2(_src_coef,
                         os.path.join(_paper_figures_dir, 'sa_bandit_stage5_coef.html'))
            print(f"    ✓ sa_bandit_stage5_coef.html")
            break

    # Safety comparison CSV
    _src_csv = os.path.join(safety_dir, 'safety_comparison.csv')
    if os.path.exists(_src_csv):
        shutil.copy2(_src_csv,
                     os.path.join(_paper_figures_dir, 'sa_bandit_safety_comparison.csv'))
        print(f"    ✓ sa_bandit_safety_comparison.csv")

    # Safety report HTML
    _src_report = os.path.join(safety_dir, 'safety_report.html')
    if os.path.exists(_src_report):
        shutil.copy2(_src_report,
                     os.path.join(_paper_figures_dir, 'sa_bandit_safety_report.html'))
        print(f"    ✓ sa_bandit_safety_report.html")

    # Safety comparison plots
    _src_plots = os.path.join(safety_dir, 'plots')
    _dst_plots = os.path.join(_paper_figures_dir, 'safety_plots')
    if os.path.isdir(_src_plots):
        for _f in os.listdir(_src_plots):
            if _f.endswith('.png'):
                shutil.copy2(os.path.join(_src_plots, _f), os.path.join(_dst_plots, _f))
                print(f"    ✓ safety_plots/{_f}")

    # Stage 5 models (for paper coefficient tables)
    _src_model_pkl = os.path.join(baseline_dir, 'stage5_models.pkl')
    if os.path.exists(_src_model_pkl):
        shutil.copy2(_src_model_pkl,
                     os.path.join(_paper_figures_dir, 'sa_bandit_stage5_models.pkl'))
        print(f"    ✓ sa_bandit_stage5_models.pkl")

    # Per-stage GA convergence data (CSV)
    _ga_dir = os.path.join(run_dir, 'ga_reports')
    if os.path.isdir(_ga_dir):
        for _f in os.listdir(_ga_dir):
            if _f.endswith('.csv'):
                shutil.copy2(
                    os.path.join(_ga_dir, _f),
                    os.path.join(_paper_figures_dir, f'sa_bandit_{_f}'),
                )
                print(f"    ✓ sa_bandit_{_f}")

    print(f"\n  Paper-ready outputs saved → {os.path.abspath(_paper_figures_dir)}/")
    logger.info(f"Paper outputs → {os.path.abspath(_paper_figures_dir)}/")


def _fit_stage5_tour_models_hhts(
        out: dict,
        landuse,
        save_dir: str,
        mode_feature_override=None,
        dest_feature_override=None,
        safety_mode: str = 'baseline',
        crash_skims=None,
):
    """Estimate Stage 5 tour-level models on HHTS data (no synth population)
    using the given feature sets.  Writes standard artifacts to save_dir."""
    import openmatrix as omx
    from stage5_destination_choice import (
        _skim_flat,
        _prep_hhts_tours,
        _run_segment,
        _save_coefficients,
        save_models as _save_stage5_models,
        save_location_choice_summary,
    )
    from pipeline_logger import _inspect_model_parameters, _write_smoke_models_html

    os.makedirs(save_dir, exist_ok=True)
    # Use the mode-appropriate skims (507-zone OMX under NEW_POP=1, legacy
    # downcasted_matrices_base.omx otherwise) — never hardcode the legacy file.
    from pipeline_logger import PIPELINE_SKIMS_PATH as _skims_default
    skims_path = _skims_default
    if crash_skims is None:
        from pipeline_logger import _detect_stage5_crash_source
        crash_skims = _detect_stage5_crash_source()

    prev_safety_mode = os.environ.get('SAFETY_EXPERIMENT_MODE')
    os.environ['SAFETY_EXPERIMENT_MODE'] = safety_mode

    skims = None
    try:
        skims = omx.open_file(skims_path, 'r')
        skim_df = _skim_flat(skims)

        hhts_tours_src = out.get('tours', pd.DataFrame())
        if hhts_tours_src.empty:
            logger.warning("No HHTS tours available — skipping Stage 5 fit")
            return

        hhts_tours = _prep_hhts_tours(hhts_tours_src, out['persons'])
        logger.info(
            f"Stage 5 [{safety_mode}]: fitting {len(hhts_tours):,} HHTS tours"
            f"  mode_feats={len(mode_feature_override) if mode_feature_override else 'default'}"
            f"  dest_feats={len(dest_feature_override) if dest_feature_override else 'default'}"
        )

        models = {}
        for seg in ('work', 'edu', 'nonmand'):
            try:
                result = _run_segment(
                    seg,
                    hhts_tours,
                    out['persons'],
                    landuse,
                    skims,
                    skim_df,
                    seed=42,
                    mode_feature_override=mode_feature_override,
                    dest_feature_override=dest_feature_override,
                    crash_skims=crash_skims,
                )
                if result:
                    models[seg] = result
            except Exception as exc:
                logger.warning(f"Stage 5 [{safety_mode}] failed for {seg}: {exc}")

        model_path = os.path.join(save_dir, 'stage5_models.pkl')
        coef_path = os.path.join(save_dir, 'stage5_models_coefficients.csv')
        _save_stage5_models(models, model_path)
        _save_coefficients(models, coef_path)
        save_location_choice_summary(
            models=models,
            obs_tours=out.get('tours', pd.DataFrame()),
            landuse=landuse,
            save_dir=save_dir,
            persons=out.get('persons'),
        )
        _inspect_model_parameters(models, 'stage5_tour', save_dir)
        _write_smoke_models_html(
            models,
            save_dir,
            title=f'Stage 5 [{safety_mode}] — SA-Bandit Model Coefficients',
        )
        logger.info(f"Stage 5 [{safety_mode}] models saved → {save_dir}")
        return models
    finally:
        if skims is not None:
            try:
                skims.close()
            except Exception:
                pass
        if prev_safety_mode is None:
            os.environ.pop('SAFETY_EXPERIMENT_MODE', None)
        else:
            os.environ['SAFETY_EXPERIMENT_MODE'] = prev_safety_mode


def _run_prediction(out: dict, landuse, all_ga: AllStagesResult, run_dir: str):
    """Apply GA-selected models to the full synthetic population.

    This mirrors the prediction phase of pipeline_logger.GA_STAGED():
      1. Load synthetic population (all zones)
      2. CDAP + usual locations
      3. apply_ga_results_to_pipeline (stages 1-5 prediction)
      4. Stage 5b (trip-level destination)
      5. Stage 6 (plan finalisation)
      6. Plots + spatial plots
    """
    import openmatrix as omx
    from pipeline_logger import (
        load_synth_all_zones, load_synth,
        cdap, usual_locations,
        stage5b, stage6, plot, spatial_plots,
        SYNTH_DIR, PIPELINE_SKIMS_PATH,
    )
    from ga_hypothesis_testing import apply_ga_results_to_pipeline

    print(f"\n{'='*70}")
    print(f"  PHASE 2 — FULL POPULATION PREDICTION  (SA-Bandit models)")
    print(f"{'='*70}")
    logger.info("=" * 60)
    logger.info("PHASE 2 — POPULATION PREDICTION")
    logger.info("=" * 60)

    # ── Load synthetic population ──────────────────────────────────────
    print("\n  Loading synthetic population (all zones) ...")
    synth = load_synth_all_zones(SYNTH_DIR, landuse=landuse)
    print(f"  Loaded {len(synth):,} synthetic persons")

    # ── Long-term choices then CDAP ───────────────────────────────────
    # usual_locations must run BEFORE cdap: work/school zone accessibility
    # shapes whether a person makes a mandatory tour (CDAP pattern = M).
    synth = usual_locations(synth, landuse, hhts_tours=out.get('tours'))
    synth = cdap(out, synth)
    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} travellers")

    # ── Apply GA-selected features through stages 1-5 ──────────────────
    # apply_ga_results_to_pipeline runs all stages (1-5) internally,
    # re-estimates models with GA features, predicts on synth, and
    # saves coefficients + model bundles.
    from pipeline_logger import _ga_post_estimation_report
    _save_dir = run_dir
    os.environ['GA_STAGE5_FALLBACK_TO_MAIN'] = '1'
    try:
        scheduled_dest, s5_models = apply_ga_results_to_pipeline(
            ga_results = all_ga,
            out        = out,
            synth      = travellers,
            landuse    = landuse,
            skims_path = PIPELINE_SKIMS_PATH,
            save_dir   = _save_dir,
        )
    except Exception as _ape:
        logger.error(f"apply_ga_results_to_pipeline FAILED: {_ape}", exc_info=True)
        print(f"  !! Prediction failed: {_ape}")
        return

    # ── Post-GA diagnostics (coefficients + HHTS validation) ──────────
    _ga_post_estimation_report(s5_models, out, _save_dir)

    # ── Stage 5b (trip-level destination) ──────────────────────────────
    skims   = omx.open_file(PIPELINE_SKIMS_PATH, 'r')
    from stage5_destination_choice import _skim_flat, _skim_lookup
    sk = _skim_lookup(_skim_flat(skims))
    try:
        scheduled_dest = stage5b(out, scheduled_dest, landuse)
    except Exception as _s5b_e:
        import traceback as _tb_s5b
        logger.warning(f"Stage 5b failed (non-fatal): {_s5b_e}")
        logger.debug(_tb_s5b.format_exc())

    # ── Stage 6 (plan finalisation + export) ──────────────────────────
    plan, _ = stage6(out, scheduled_dest, landuse,
                     stage5_models=s5_models, skims=skims, skim_lookup=sk)
    plan.to_csv(os.path.join(run_dir, 'stage6_output.csv'), index=False)
    plot(plan)
    spatial_plots(plan, synth, landuse)
    if skims:
        skims.close()

    # ── Validation summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREDICTION — VALIDATION SUMMARY")
    print("=" * 60)
    try:
        trips_out = plan[plan.get('row_type', pd.Series()) == 'trip'] \
                    if 'row_type' in plan.columns else plan
        if 'mode_name' in trips_out.columns:
            mode_share = trips_out['mode_name'].value_counts(normalize=True) * 100
            print("\n  Mode share (predicted):")
            for mode, share in mode_share.sort_values(ascending=False).items():
                print(f"    {mode:<12} {share:5.1f}%")
    except Exception as _e:
        logger.warning(f"  Mode share validation failed: {_e}")

    try:
        n_trips = int((plan['row_type'] == 'trip').sum()) if 'row_type' in plan.columns else len(plan)
        n_pax   = plan['person_id'].nunique() if 'person_id' in plan.columns else 0
        trips_per = n_trips / max(n_pax, 1)
        print(f"\n  Trips: {n_trips:,}  Persons: {n_pax:,}  Trips/person: {trips_per:.2f}")
    except Exception as _e:
        logger.warning(f"  Trip summary failed: {_e}")

    logger.info("Population prediction complete.")


if __name__ == '__main__':
    run()
