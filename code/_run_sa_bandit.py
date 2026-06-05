"""
_run_sa_bandit.py
─────────────────
Run ONLY the 5 GA/SA-Bandit estimation stages from VS Code — no synth
population, no pipeline prediction, no PBS required.

Outputs land in  runs/sa_bandit_<timestamp>/
  ga_results.pkl            — checkpoint (can be loaded by GA_STAGED later)
  ga_stage<N>_report.txt    — per-stage convergence summary
  ga_stage<N>_coef.html     — coefficients from best-found feature set
  plots/ga_all_stages.png   — combined convergence figure

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
# ─────────────────────────────── SETTINGS ────────────────────────────────────
# Budget:   smoke=fast sanity check, small=dev run, medium=research, large=full
#           (see _ga_budget_params in pipeline_logger.py for exact pop/gen values)
GA_BUDGET = 'fast'           # 'smoke' | 'fast' | 'standard' | 'thorough'

# Print a fitness line every N generations (1 = every generation).
GA_VERBOSE_EVERY = 5         # matches the default in BanditFeatureSearcher

# Which stages to run (subset for faster iteration; None = all five)
RUN_STAGES = None            # e.g. [1, 2] to run only stages 1 & 2

# ─────────────────────────────── ENV SETUP ───────────────────────────────────
# All env-vars MUST be set before any pipeline or JAX import.
import os, sys

os.environ.setdefault('JAX_PLATFORMS',           'cpu')
os.environ.setdefault('GA_METACOUNT_FORCE_CPU',  '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL',    '3')

os.environ['GA_ESTIMATOR']   = 'sa_bandit'   # activates BanditFeatureSearcher
os.environ['GA_USE_BANDIT_SA'] = '1'
os.environ['GA_BUDGET']      = GA_BUDGET
os.environ['GA_VERBOSE_EVERY'] = str(GA_VERBOSE_EVERY)

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
    ga_stage1, ga_stage2, ga_stage3, ga_stage4, ga_stage5,
    _save_ga_checkpoint, _write_stage_checkpoint_report,
    _ga_save_stage_coef_html_from_best, _format_ga_checkpoint_summary,
    logger,
)
from ga_hypothesis_testing import AllStagesResult

# ─────────────────────────────── LOGGING ─────────────────────────────────────
# Create a timestamped output directory so repeated runs never clobber each other.
_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
pl.OUT_DIR = os.path.join('runs', f'sa_bandit_{_ts}')
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
def _all_stage_runners(out, landuse):
    """Return the ordered list of (stage_id, runner_fn) pairs."""
    return [
        (1, lambda: ga_stage1(out)),
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
    print(f"  stages = {RUN_STAGES or [1,2,3,4,5]}")
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
            hist_df = pd.DataFrame(getattr(res, 'history', []) or []) or None
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
            _save_ga_checkpoint(all_ga, stage_id=sid, failed=True)
            logger.error(f"GA stage {sid} FAILED: {exc}", exc_info=True)
            print(f"\n  !! Stage {sid} failed: {exc}")
            print("     Partial results saved.  Continuing with remaining stages …\n")

    # ── 3. Combined convergence plot ─────────────────────────────────────────
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


if __name__ == '__main__':
    run()
