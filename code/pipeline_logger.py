# pipeline_logger.py
import argparse
import time
import logging
import functools
import os
import glob
import gc
import sys
import pickle
import re
import json
import numpy as np
from typing import Optional

# Suppress XLA/TensorFlow verbose logging BEFORE any JAX or Larch import.
# These env vars must be set prior to the first XLA initialisation; setting
# them afterwards has no effect.
#   TF_CPP_MIN_LOG_LEVEL=3 → suppress all XLA/LLVM INFO and WARNING messages
#   XLA_FLAGS=...          → disable slow constant-folding warnings from large
#                            gather/scatter ops during Larch JAX compilation
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# Note: --xla_constant_folding_limit is not a valid XLA flag in the JAX-bundled
# XLA version on HPC — setting it causes a FATAL abort at optimisation time.
# TF_CPP_MIN_LOG_LEVEL=3 (above) already suppresses the XLA INFO/WARNING
# messages that were the original target.
_requested_accelerator = os.environ.get('PIPELINE_ACCELERATOR', '').strip().lower()
if not os.environ.get('JAX_PLATFORMS'):
    if _requested_accelerator in ('cpu', 'gpu'):
        os.environ['JAX_PLATFORMS'] = _requested_accelerator
    else:
        # Safe default for interactive / ad hoc runs when the scheduler did not
        # declare the intended accelerator.
        os.environ['JAX_PLATFORMS'] = 'cpu'

# Optional JAX backend diagnostics. Disabled by default to avoid noisy
# backend/device initialization side-effects in batch runs.
if os.environ.get('DEBUG_JAX_BACKEND', '0') == '1':
    try:
        import jax
        print(jax.default_backend())
        print(jax.devices())
    except Exception as _jax_e:
        print(f"JAX backend probe skipped: {_jax_e}")

# Force non-interactive matplotlib backend BEFORE any plot module imports.
# HPC nodes have no display — without this, savefig() fails silently.
import matplotlib
matplotlib.use('Agg')

# ── Monkey-patch: fix jnp.linalg.inv → scipy.linalg.inv for CUDA 13 ──────
# jaxlib 0.4.30 on CUDA 13 crashes at cusolver_gesvdj_ffi (symbol removed in
# CUDA 13).  Larch's jax_invhess_loglike() calls jnp.linalg.inv(hess) which
# XLA routes to CUDA's SVD.  We replace it with scipy.linalg.inv on CPU.
# This only affects the Hessian inversion — optimisation gradients still use GPU.
def _patch_jax_linalg():
    """Replace jnp.linalg.inv with a scipy-based fallback for CUDA 13 compat."""
    try:
        import jax.numpy as jnp
        import scipy.linalg as _splinalg
        _orig_inv = jnp.linalg.inv

        def _safe_inv(a):
            """jnp.linalg.inv replacement using scipy (CPU) to avoid CUDA 13 SVD crash."""
            arr = np.asarray(a, dtype=np.float64)
            inv_arr = _splinalg.inv(arr)
            return jnp.asarray(inv_arr)

        jnp.linalg.inv = _safe_inv
        # Also patch the jax.scipy.linalg path that some JAX ops use
        import jax.scipy.linalg as _jslinalg
        if hasattr(_jslinalg, 'inv'):
            _jslinalg.inv = _safe_inv
        return True
    except Exception:
        return False

_PATCHED_LINALG = _patch_jax_linalg()
if _PATCHED_LINALG:
    import logging as _logging
    _logging.getLogger('pipeline').info(
        "JAX linalg.inv patched → scipy.linalg.inv (CUDA 13 cusolver_gesvdj_ffi workaround)"
    )

os.makedirs('saved_data', exist_ok=True)

# ---------------------------------------------------------------------------
# OUT_DIR — run-specific output directory
# ---------------------------------------------------------------------------
# All pipeline outputs (CSVs, model PKLs, plots, log) are written under here.
# Set by _setup_logging() before any pipeline function runs, so concurrent or
# sequential runs (main / ga / ga_staged) never overwrite each other.
#
# Layout relative to $PBS_O_WORKDIR (the directory you qsub from):
#
#   runs/
#     main/             ← python pipeline_logger.py main
#     ga/               ← python pipeline_logger.py ga
#     ga_staged/        ← python pipeline_logger.py ga_staged
#     main_40/          ← python pipeline_logger.py main 40
#
#   saved_data/         ← shared read-only inputs, never written by runs
#     zone_ids_sa2/     — synthetic population CSVs
#     osm_cache/        — OSM centroid cache (expensive, reusable)
#     accessibility.csv — derived from skims+landuse (same for all runs)
# ---------------------------------------------------------------------------
OUT_DIR: str = 'runs'   # overwritten by _setup_logging() at __main__
TIMING_ROWS = []

# ── Stage 5 parallel worker state ────────────────────────────────────────────
# Set by predict_stage5 before the fork pool is created; inherited by workers
# at zero copy cost (Linux fork). Workers open their own OMX handle per task.
_S5_WORKER_MODELS    = None
_S5_WORKER_PP        = None
_S5_WORKER_LANDUSE   = None
_S5_WORKER_SK_LOOKUP = None
_S5_WORKER_SKIM_DF   = None
_S5_WORKER_SKIM_PATH = None


def _s5_spawn_initializer(models_path, pp_path, landuse_bytes, skim_path, skim_df_path, sk_bytes):
    """
    Initializer for spawn-mode worker processes.

    Called once per worker at pool creation. Loads large shared objects from
    files/bytes into module-level globals so _s5_zone_worker can access them
    without re-reading on every task.
    """
    import pickle as _pkl
    import pandas as _pd
    global _S5_WORKER_MODELS, _S5_WORKER_PP, _S5_WORKER_LANDUSE
    global _S5_WORKER_SK_LOOKUP, _S5_WORKER_SKIM_DF, _S5_WORKER_SKIM_PATH
    from stage5_destination_choice import load_models as _s5_load
    _S5_WORKER_MODELS    = _s5_load(models_path)
    _S5_WORKER_PP        = _pd.read_parquet(pp_path)
    _S5_WORKER_LANDUSE   = _pkl.loads(landuse_bytes)
    _S5_WORKER_SK_LOOKUP = _pkl.loads(sk_bytes)
    _S5_WORKER_SKIM_DF   = _pd.read_parquet(skim_df_path)
    _S5_WORKER_SKIM_PATH = skim_path


def _s5_zone_worker(task):
    """
    Process one origin zone in a spawned worker process.

    task = (zone_batch_df, seed, otaz)

    Shared objects are loaded once per worker by _s5_spawn_initializer.
    Spawn (not fork) is used so workers start without the parent's GPU/CUDA
    context — fork + CUDA causes CUDA_ERROR_NOT_INITIALIZED in every worker.
    """
    import openmatrix as omx
    from stage5_destination_choice import _build_trip_chain, _attach_skims
    zone_batch, seed, otaz = task
    t0 = time.perf_counter()
    skims_local = omx.open_file(_S5_WORKER_SKIM_PATH, 'r')
    try:
        zone_plan = _build_trip_chain(
            zone_batch, _S5_WORKER_MODELS, _S5_WORKER_PP, _S5_WORKER_LANDUSE,
            skims_local, _S5_WORKER_SK_LOOKUP, seed=seed,
        )
        zone_plan = _attach_skims(zone_plan, _S5_WORKER_SKIM_DF)
        return otaz, zone_plan, None, time.perf_counter() - t0
    except Exception as exc:
        return otaz, None, str(exc), time.perf_counter() - t0
    finally:
        skims_local.close()


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


def _setup_logging(mode: str = 'main', arg: str = None) -> None:
    """Configure logging and set OUT_DIR for this run."""
    global OUT_DIR

    # Build a safe suffix e.g. "ga_staged", "main_40", "main_40,124,200"
    suffix = mode
    if arg:
        safe_arg = arg.replace('/', '_').replace('\\', '_').replace(' ', '')
        suffix = f"{mode}_{safe_arg}"

    ga_estimator = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    _ga_mode = mode in ('ga', 'ga_staged', 'ga_stage5_smoke')
    _use_sa = os.environ.get('GA_USE_SA', '0').strip() == '1'
    _use_bandit_sa = os.environ.get('GA_USE_BANDIT_SA', '0').strip() == '1'
    if _ga_mode:
        # For GA modes: always include estimator name; prefix with sa_ vs ga_ based on SA flags.
        _estim_label = ga_estimator if ga_estimator else 'default'
        if _use_sa or _use_bandit_sa:
            # sa_bandit estimator with SA flags -> 'sa_bandit' (not 'sa_ga_sa_bandit')
            if _estim_label == 'sa_bandit':
                _estim_label = 'bandit'
            _search_prefix = 'sa'
        else:
            _search_prefix = 'ga'
        # mode suffix after the 'ga' base: '' for 'ga', '_staged' for 'ga_staged', etc.
        _mode_tail = mode[2:]  # '' or '_staged' or '_stage5_smoke'
        if arg:
            safe_arg = arg.replace('/', '_').replace('\\', '_').replace(' ', '')
            suffix = f"{_search_prefix}_{_estim_label}{_mode_tail}_{safe_arg}"
        else:
            suffix = f"{_search_prefix}_{_estim_label}{_mode_tail}"
    elif ga_estimator and ga_estimator != 'default':
        suffix = f"{suffix}_{ga_estimator}"

    _run_tag_env = os.environ.get('PIPELINE_RUN_TAG', '').strip()
    _run_tag_pbs = os.environ.get('PBS_JOBID', '').strip()
    if _run_tag_env:
        run_tag = _run_tag_env
        run_tag_source = 'PIPELINE_RUN_TAG'
    elif _run_tag_pbs:
        run_tag = _run_tag_pbs
        run_tag_source = 'PBS_JOBID'
    else:
        run_tag = ''
        run_tag_source = 'none'
    if run_tag:
        safe_tag = run_tag.replace('/', '_').replace('\\', '_').replace(' ', '')
        suffix = f"{suffix}__{safe_tag}"

    run_root = os.environ.get('PIPELINE_RUN_ROOT', 'runs')
    OUT_DIR = os.path.join(run_root, suffix)
    os.makedirs(OUT_DIR,                        exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)

    # ── Clean stale pickles from any previous run in this directory ────────
    # Deleting them ensures Phase 2 reloads freshly estimated models and that
    # validation outputs reflect the current run, not a cached prior run.
    _STALE_PKLS = [
        'stage1_models.pkl', 'stage2_models.pkl', 'stage3_models.pkl',
        'stage4_models.pkl', 'stage5_models.pkl', 'stage5b_models.pkl',
        'ga_results.pkl', '_hhts_persons.pkl', '_hhts_tours.pkl',
    ]
    _cleaned = []
    for _pkl_name in _STALE_PKLS:
        _pkl_path = os.path.join(OUT_DIR, _pkl_name)
        if os.path.exists(_pkl_path):
            try:
                os.remove(_pkl_path)
                _cleaned.append(_pkl_name)
            except OSError as _e:
                pass  # logged after logger is configured
    # Store for post-logger log message
    _cleaned_pkls = _cleaned

    log_path = os.path.join(OUT_DIR, 'pipeline.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            stdout_handler,
            stderr_handler,
            logging.FileHandler(log_path, mode='w'),
        ]
    )
    logging.getLogger('pipeline').info(
        f"Output directory : {OUT_DIR}  (mode={mode}"
        + (f"  arg={arg}" if arg else "") + ")"
    )
    logging.getLogger('pipeline').info(
        f"Run tag source   : {run_tag_source}" + (f"  value={run_tag}" if run_tag else "")
    )

    # Export OUT_DIR as an env var so stage __main__ blocks inherit the
    # correct run directory when called as subprocesses (e.g. --fresh mode).
    os.environ['PIPELINE_OUT_DIR'] = OUT_DIR

    # Write a manifest so it's immediately clear which run owns which directory
    import json, datetime
    manifest = {
        'mode':      mode,
        'arg':       arg,
        'out_dir':   OUT_DIR,
        'started':   datetime.datetime.now().isoformat(timespec='seconds'),
        'pid':       os.getpid(),
    }
    try:
        with open(os.path.join(OUT_DIR, 'run_manifest.json'), 'w') as _mf:
            json.dump(manifest, _mf, indent=2)
    except Exception:
        pass

    # Maintain stable aliases for tooling/users that expect fixed folders
    # (e.g. runs/main and runs/main_ga) while still keeping tagged histories.
    alias_name = None
    if mode == 'main':
        alias_name = 'main'
    elif mode in ('ga', 'ga_staged', 'ga_stage5_smoke'):
        alias_name = 'main_ga'
    if alias_name:
        alias_dir = os.path.join(run_root, alias_name)
        try:
            os.makedirs(alias_dir, exist_ok=True)
            with open(os.path.join(alias_dir, 'LATEST_RUN.txt'), 'w', encoding='utf-8') as _lf:
                _lf.write(OUT_DIR + '\n')
            alias_manifest = dict(manifest)
            alias_manifest.update({'alias': alias_name, 'latest_out_dir': OUT_DIR})
            with open(os.path.join(alias_dir, 'run_manifest.json'), 'w', encoding='utf-8') as _amf:
                json.dump(alias_manifest, _amf, indent=2)
            logging.getLogger('pipeline').info(
                f"Stable alias updated: {alias_dir} -> {OUT_DIR}"
            )
        except Exception as _alias_e:
            logging.getLogger('pipeline').warning(
                f"Could not update stable alias '{alias_name}': {_alias_e}"
            )

    # When mode='ga', also write a per-estimator alias (e.g. main_ga_metacount)
    # so that a following ga_staged job with GA_STAGED_REUSE_ESTIMATOR=1 can
    # locate this run's ga_results.pkl without relying on shared main_ga alias
    # (which would be clobbered by parallel ga chains with different estimators).
    if mode == 'ga' and ga_estimator and ga_estimator not in ('default', ''):
        _est_alias_name = f'main_ga_{ga_estimator}'
        _est_alias_dir = os.path.join(run_root, _est_alias_name)
        try:
            os.makedirs(_est_alias_dir, exist_ok=True)
            with open(os.path.join(_est_alias_dir, 'LATEST_RUN.txt'), 'w', encoding='utf-8') as _lf:
                _lf.write(OUT_DIR + '\n')
            _est_alias_manifest = dict(manifest)
            _est_alias_manifest.update({'alias': _est_alias_name, 'latest_out_dir': OUT_DIR})
            with open(os.path.join(_est_alias_dir, 'run_manifest.json'), 'w', encoding='utf-8') as _amf:
                json.dump(_est_alias_manifest, _amf, indent=2)
            logging.getLogger('pipeline').info(
                f"Estimator alias updated: {_est_alias_dir} -> {OUT_DIR}"
            )
        except Exception as _est_alias_e:
            logging.getLogger('pipeline').warning(
                f"Could not update estimator alias '{_est_alias_name}': {_est_alias_e}"
            )

    # Log pkl cleanup now that the logger is ready
    if _cleaned_pkls:
        logging.getLogger('pipeline').info(
            f"Deleted {len(_cleaned_pkls)} stale pickle(s) from {OUT_DIR}: "
            + ', '.join(_cleaned_pkls)
        )
    else:
        logging.getLogger('pipeline').info(
            f"No stale pickles found in {OUT_DIR} — clean slate."
        )


def _flush_timing_report() -> None:
    if not TIMING_ROWS:
        return
    try:
        timings = pd.DataFrame(TIMING_ROWS)
        timings.to_csv(os.path.join(OUT_DIR, 'timings.csv'), index=False)
        timings.to_html(os.path.join(OUT_DIR, 'timings.html'), index=False)
    except Exception as exc:
        logging.getLogger('pipeline').debug(f"timing report write skipped ({exc})")


def _record_timing(label: str, elapsed: float, status: str) -> None:
    TIMING_ROWS.append({
        'step': label,
        'elapsed_seconds': round(float(elapsed), 3),
        'elapsed_pretty': _fmt(elapsed),
        'status': status,
    })
    _flush_timing_report()


logger = logging.getLogger('pipeline')


def timer(label: str = None):
    """Decorator that logs start, end and elapsed time of any function."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            logger.info(f"▶  {name} started")
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.info(f"✓  {name} finished in {_fmt(elapsed)}")
                _record_timing(name, elapsed, 'ok')
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error(f"✗  {name} FAILED after {_fmt(elapsed)} — {e}")
                _record_timing(name, elapsed, 'failed')
                raise
        return wrapper
    return decorator


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


class PipelineTimer:
    """Context manager for timing named blocks."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        logger.info(f"▶  {self.label} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        if exc_type:
            logger.error(f"✗  {self.label} FAILED after {_fmt(elapsed)}")
            _record_timing(self.label, elapsed, 'failed')
        else:
            logger.info(f"✓  {self.label} finished in {_fmt(elapsed)}")
            _record_timing(self.label, elapsed, 'ok')
        return False


# ---------------------------------------------------------------------------
import pandas as pd
from data_connect import Synth_Connect
from synth_attributes import enrich_synth_population
from stage1_activity_frequency import run_stage1
from stage2_day_duration        import run_stage2
from stage3_activity_duration   import run_stage3
from stage4_schedule_placement  import run_stage4
from stage5_destination_choice  import run_stage5
from stage5b_trip_destination   import run_stage5b
from stage6_plan_finalisation   import run_stage6
from cdap                       import run_cdap
from usual_location             import run_usual_locations
from plot_activity_plans        import plot_all
from spatial_assignment         import (
    run_spatial_assignment, plot_person_movement,
    plot_distance_distribution, plot_desire_lines,
    plot_activity_heatmap, plot_spatial_metrics,
    get_zone_centroids_rich,
)
from read_skims                 import list_omx_matrices
from ga_hypothesis_testing      import run_ga_all_stages


# ---------------------------------------------------------------------------
# SYNTH POPULATION DIRECTORY
# ---------------------------------------------------------------------------
# Point this at the folder containing your per-zone CSV files.
# Files must be named  order_N.csv  (e.g. order_1.csv, order_40.csv).
# The ORDER/OTAZ value is read from the filename — no need to add it manually.
#
# Set NEW_POP=1 to switch to the 507-SA2-zone scope instead: zone_<sa2>.csv
# files (OTAZ = SA2 code directly, no ORDER indirection — see
# export_zone_sa2_files() in synth_pop/run_pipeline.py and
# Synth_Connect(new_pop=True) in data_connect.py), read from
# NEW_POP_DIR/saved_data/zone_ids_sa2 (NEW_POP_DIR defaults to
# Z:\test_runs_tours\new_pop).
NEW_POP = os.environ.get('NEW_POP', '0').strip() == '1'
# Repo-relative default so it resolves on both the HPC
# (/mnt/hpccs01/home/ahernz/test_runs_tours/new_pop) and the Z:-mounted
# Windows checkout - a literal drive letter here broke every NEW_POP=1 run
# executed on Linux.
_TRT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
NEW_POP_DIR = os.environ.get('NEW_POP_DIR') or os.path.join(_TRT_ROOT, 'new_pop')

# NEW_POP_MAPPED=1: load from new_pop 507-zone files but remap OTAZ from SA2
# code → ORDER number via demand_/sa2_to_order_crosswalk.csv, so the existing
# 329-zone ORDER skims (downcasted_matrices_base.omx) remain the destination
# skim while the 324-zone SA1/Lvl4 walk/bike skims supply fine-grain active-
# mode origins.  Run demand_/20_build_sa2_to_order_crosswalk.py first, then
# build_synth_population.py --new-pop-mapped to pre-build the population CSV.
NEW_POP_MAPPED = os.environ.get('NEW_POP_MAPPED', '0').strip() == '1'


def _resolve_507_skims() -> str:
    """Canonical 371-zone skim path (507 SA2 system).

    ZONE MIGRATION (371): the legacy 330-zone downcasted_matrices_base.omx has
    been retired — every mode now uses the 371-zone sa2_skims_507 skims. Prefer
    the omx-spec-compliant re-encode (11_convert_skims_to_openmatrix.py output,
    Blosc2-safe for PyTables) and fall back to the raw aggregation. read_skims
    hard-errors if whatever loads here is not 371×371, so the 330 system can
    never be silently reintroduced.
    """
    _p = os.path.join(_TRT_ROOT, 'demand_', 'sa2_skims_507_omxspec.omx')
    if not os.path.exists(_p):
        _raw = os.path.join(_TRT_ROOT, 'demand_', 'sa2_skims_507.omx')
        if os.path.exists(_raw):
            _p = _raw
    return _p


_SKIMS_507 = _resolve_507_skims()

if NEW_POP:
    SYNTH_DIR = os.path.join(NEW_POP_DIR, 'saved_data', 'zone_ids_sa2')
    DEFAULT_SYNTH = os.path.join(SYNTH_DIR, 'zone_301011001.csv')
    _DEFAULT_SKIMS_PATH = _SKIMS_507
elif NEW_POP_MAPPED:
    # Population: loaded on-the-fly from new_pop zone_<sa2>.csv with SA2→ORDER
    # remapping (build_synth_population.build_synth_population_mapped).
    # ZONE MIGRATION (371): skims are now the 371-zone 507 system (was the
    # 330-zone downcasted_matrices_base.omx). The SA2→ORDER remap targets the
    # 371-zone seq_zone_ids_371.csv ORDER scheme.
    SYNTH_DIR = os.path.join(
        os.environ.get('NEW_POP_DIR', NEW_POP_DIR), 'saved_data', 'zone_ids_sa2'
    )
    DEFAULT_SYNTH = 'saved_data/synth_persons_mapped.csv'
    _DEFAULT_SKIMS_PATH = _SKIMS_507
else:
    SYNTH_DIR = 'saved_data/zone_ids_sa2'
    DEFAULT_SYNTH = f'{SYNTH_DIR}/order_40.csv'
    _DEFAULT_SKIMS_PATH = _SKIMS_507

# Single source of truth for the skims path, overridable per-run via
# PIPELINE_SKIMS_PATH (e.g. qsub -v PIPELINE_SKIMS_PATH=...).
# Use `or` instead of .get(key, default) so an *empty string* from PBS
# (-v "PIPELINE_SKIMS_PATH=" passes '' into os.environ, not a missing key)
# also falls back to the mode-appropriate default rather than resolving to CWD.
PIPELINE_SKIMS_PATH = os.environ.get('PIPELINE_SKIMS_PATH') or _DEFAULT_SKIMS_PATH


# ---------------------------------------------------------------------------
# HELPERS — decorated wrappers so every stage is individually timed
# ---------------------------------------------------------------------------

@timer("process_survey + landuse + accessibility")
def load_data():
    from compute_accessibility import run_accessibility
    sc      = Synth_Connect(new_pop=NEW_POP)
    out     = sc.process_survey(save=False)
    landuse = sc.get_land()
    # ── Landuse patch: fill Primary=0 from Primary_total_priv ─────────────
    # Some zones have Primary=0 (no OSM-mapped public schools) but have
    # significant private-school enrolment in Primary_total_priv.  Using
    # Primary=0 as an attractor results in zero destination-choice probability
    # for those zones (e.g. zone 248 with 26 HHTS primary students predicted=0).
    # Fill: if Primary=0 and Primary_total_priv > 0, set Primary = Primary_total_priv.
    if 'Primary' in landuse.columns and 'Primary_total_priv' in landuse.columns:
        _mask = (landuse['Primary'] == 0) & (landuse['Primary_total_priv'] > 0)
        _n    = _mask.sum()
        if _n > 0:
            landuse.loc[_mask, 'Primary'] = landuse.loc[_mask, 'Primary_total_priv']
            logger.info(f"  Landuse patch: filled Primary=0 from Primary_total_priv "
                        f"for {_n} zone(s) — e.g. {list(landuse.loc[_mask].index[:5])}")
    if 'Secondary' in landuse.columns and 'Secondary_total_priv' in landuse.columns:
        _mask2 = (landuse['Secondary'] == 0) & (landuse['Secondary_total_priv'] > 0)
        if _mask2.sum() > 0:
            landuse.loc[_mask2, 'Secondary'] = landuse.loc[_mask2, 'Secondary_total_priv']
            logger.info(f"  Landuse patch: filled Secondary=0 from Secondary_total_priv "
                        f"for {_mask2.sum()} zone(s)")
    landuse = run_accessibility(
        landuse,
        skims=PIPELINE_SKIMS_PATH,
        save_path='saved_data/accessibility.csv',
        plot_dir=os.path.join(OUT_DIR, 'plots'),
    )
    return sc, out, landuse


def _parse_zone_id(filepath: str):
    """Parse the zone id from order_N.csv (legacy) or zone_<sa2>.csv (new_pop)."""
    import re
    match = re.search(r'order_(\d+)', filepath)
    if match:
        return int(match.group(1))
    match = re.search(r'zone_(\d+)', filepath)
    if match:
        return int(match.group(1))
    return None


@timer("load synthetic population — single zone")
def load_synth(filepath: str = DEFAULT_SYNTH,
               landuse: pd.DataFrame = None):
    """Load one zone's synth CSV.  OTAZ is derived from the filename."""
    synth = pd.read_csv(filepath)

    zone_id = _parse_zone_id(filepath)
    if zone_id is not None:
        synth['OTAZ'] = zone_id
    elif 'ORDER' in synth.columns:
        synth['OTAZ'] = synth['ORDER'].astype(int)
    elif 'OTAZ' in synth.columns:
        synth['OTAZ'] = pd.to_numeric(synth['OTAZ'], errors='coerce').fillna(0).astype(int)
    else:
        raise ValueError(f"Cannot determine OTAZ from {filepath}")

    synth = enrich_synth_population(synth, landuse=landuse)

    logger.info(f"  {len(synth)} persons | OTAZ={synth['OTAZ'].iloc[0]} | "
                f"ptype dist: {synth['ptype'].value_counts().to_dict()}")
    return synth


@timer("load synthetic population — ALL zones")
def load_synth_all_zones(synth_dir: str = SYNTH_DIR,
                         landuse: pd.DataFrame = None,
                         zones: list = None):
    """
    Load every order_N.csv in synth_dir and concatenate into one DataFrame.

    When NEW_POP_MAPPED=1: loads from new_pop zone_<sa2>.csv files and remaps
    OTAZ from SA2 code → ORDER number via demand_/sa2_to_order_crosswalk.csv.
    Non-324-zone people are merged into their nearest 324-zone ORDER neighbour.
    A pre-built CSV (saved_data/synth_persons_mapped.csv) is used when present
    to avoid the full remap on every run; rebuild it with
      python build_synth_population.py --new-pop-mapped

    Parameters
    ----------
    synth_dir : folder containing order_N.csv files (ignored for NEW_POP_MAPPED)
    landuse   : passed to enrich_synth_population for accessibility join
    zones     : optional list of zone ORDER numbers to restrict loading

    Returns
    -------
    pd.DataFrame  — full synthetic population with OTAZ set correctly.
    """
    # ── NEW_POP_MAPPED: load from pre-built mapped CSV or build on-the-fly ──
    if NEW_POP_MAPPED:
        from build_synth_population import (
            build_synth_population_mapped,
            OUT_MAPPED_PERSONS,
        )
        _cached = OUT_MAPPED_PERSONS
        if os.path.exists(_cached):
            logger.info(f"NEW_POP_MAPPED: loading pre-built mapped population from {_cached}")
            synth = pd.read_csv(_cached, low_memory=False)
            logger.info(f"  Loaded {len(synth):,} persons, {synth['OTAZ'].nunique()} ORDER zones")
        else:
            logger.info("NEW_POP_MAPPED: pre-built CSV not found — building from new_pop zone files")
            synth = build_synth_population_mapped(
                new_pop_dir=os.environ.get('NEW_POP_DIR', NEW_POP_DIR),
                attach_accessibility=False,
                save=True,
            )
        # Filter to specific ORDER zones if requested
        if zones is not None:
            synth = synth[synth['OTAZ'].isin(set(zones))].reset_index(drop=True)
            logger.info(f"  Zone filter applied: {synth['OTAZ'].nunique()} zones retained")
        return synth

    files = sorted(glob.glob(os.path.join(synth_dir, 'order_*.csv'))
                    + glob.glob(os.path.join(synth_dir, 'zone_*.csv')))

    if not files:
        raise FileNotFoundError(
            f"No order_*.csv or zone_*.csv files found in '{synth_dir}'.\n"
            f"  Legacy scope expects: order_1.csv, order_2.csv, ..., order_329.csv\n"
            f"  new_pop scope expects: zone_<sa2>.csv (set NEW_POP=1)\n"
            f"  Check that SYNTH_DIR is set correctly at the top of pipeline_logger.py."
        )

    # Filter to specific zones if requested
    if zones is not None:
        zone_set = set(zones)
        files = [f for f in files if _parse_zone_id(f) in zone_set]
        logger.info(f"  Zone filter: loading {len(files)} zones from {len(zone_set)} requested")

    logger.info(f"  Found {len(files)} zone CSV files in {synth_dir}")

    frames = []
    missing = []
    for filepath in files:
        otaz = _parse_zone_id(filepath)
        if otaz is None:
            logger.warning(f"  Skipping {filepath} — cannot parse zone id from filename")
            continue
        try:
            df = pd.read_csv(filepath)
            df['OTAZ'] = otaz       # override/set from filename — always correct
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Failed to load {filepath}: {e}")
            missing.append(otaz)

    if not frames:
        raise ValueError(f"No zone CSVs loaded successfully from {synth_dir}")

    # Concatenate all zones
    synth = pd.concat(frames, ignore_index=True)

    # Assign a globally-unique person_id BEFORE enrichment.
    # Each zone CSV's abspid/PERSID restarts from 0 so after concat they
    # are duplicated across zones — causing "non-unique MultiIndex" errors
    # in Larch's from_idca() in CDAP, stage5, and any other DataTree build.
    if 'person_id' not in synth.columns:
        id_src = next((c for c in ['abspid', 'PERSID', 'person_id']
                       if c in synth.columns), None)
        if id_src:
            # Build globally unique id: OTAZ * 1_000_000 + local_id
            # Each zone has at most ~40k persons so * 1M gives no collisions
            synth['person_id'] = (
                synth['OTAZ'].astype(int) * 1_000_000
                + synth[id_src].astype(int)
            )
        else:
            synth['person_id'] = np.arange(len(synth), dtype=np.int64)
    else:
        # person_id already present — check for duplicates and fix if needed
        if synth['person_id'].duplicated().any():
            n_dup = synth['person_id'].duplicated().sum()
            logger.warning(f"  {n_dup:,} duplicate person_ids detected — reassigning unique ids")
            synth['person_id'] = (
                synth['OTAZ'].astype(int) * 1_000_000
                + synth.groupby('OTAZ').cumcount()
            )
    logger.info(f"  person_id: {synth['person_id'].nunique():,} unique / {len(synth):,} total")

    # Enrich once on the full population — much faster than per-zone
    synth = enrich_synth_population(synth, landuse=landuse)

    # Assign HOME_SA1 for fine-grained walk/bike skim lookups in stage5.
    # order_N.csv (329-zone) files don't carry HOME_SA1; assign here using the
    # SA2_MAIN16 column (present in Census-derived CSVs) → SA1 membership from
    # SA1_SHP/sa1_2021.csv.  In new_pop mode (zone_<sa2>.csv) the column is
    # already present from assign_sa1_to_population.py.
    if 'HOME_SA1' not in synth.columns:
        try:
            _sa1_csv = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'SA1_SHP', 'sa1_2021.csv'))
            _xw_csv  = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'demand_',
                'sa1_to_emme_crosswalk_seqtweed.csv'))
            if os.path.exists(_sa1_csv):
                _sa1_lu = pd.read_csv(_sa1_csv, dtype=str)[['SA1_CODE21', 'SA2_CODE21']]
                if os.path.exists(_xw_csv):
                    _study = set(pd.read_csv(_xw_csv, dtype={'SA1_CODE21': str})['SA1_CODE21'].str.strip())
                    _sa1_lu = _sa1_lu[_sa1_lu['SA1_CODE21'].isin(_study)]
                _sa2_to_sa1s = (
                    _sa1_lu.groupby('SA2_CODE21')['SA1_CODE21']
                    .apply(lambda x: np.array([int(v) for v in x], dtype=np.int64))
                    .to_dict()
                )
                # In new_pop mode OTAZ IS the 2021 SA2 code — use it first so
                # the lookup matches sa1_2021.csv's SA2_CODE21 keys exactly.
                # SA2_MAIN16 is a 2016 code (~93% concordant with 2021) and is
                # only the right key for the legacy Census-derived zone CSVs.
                if NEW_POP:
                    _sa2_col = next((c for c in ['OTAZ', 'SA2_CODE21', 'SA2_MAIN16'] if c in synth.columns), None)
                else:
                    _sa2_col = next((c for c in ['SA2_MAIN16', 'SA2_CODE21'] if c in synth.columns), None)
                if _sa2_col is not None:
                    _rng = np.random.default_rng(seed=42)
                    _sa1_out = np.zeros(len(synth), dtype=np.int64)
                    for _sa2_val, _grp_idx in synth.groupby(_sa2_col).groups.items():
                        _sa1_list = _sa2_to_sa1s.get(str(_sa2_val))
                        if _sa1_list is not None and len(_sa1_list):
                            _sa1_out[_grp_idx] = _rng.choice(_sa1_list, size=len(_grp_idx))
                    synth['HOME_SA1'] = _sa1_out
                    _n_ok = int((_sa1_out > 0).sum())
                    logger.info(f"  HOME_SA1 assigned via {_sa2_col}: "
                                f"{_n_ok:,}/{len(synth):,} persons "
                                f"({100*_n_ok/len(synth):.1f}% coverage)")
                else:
                    logger.warning("HOME_SA1: no SA2 column found — walk/bike fine-skim disabled")
        except Exception as _sa1_err:
            logger.warning(f"HOME_SA1 on-the-fly assignment failed: {_sa1_err}")

    # Per-zone person counts — log summary only (full zone list → DEBUG)
    zone_counts = synth.groupby('OTAZ').size()
    logger.info(
        f"  Loaded {len(synth):,} persons across {zone_counts.shape[0]} zones "
        f"(per-zone min:{zone_counts.min()} mean:{zone_counts.mean():.0f} max:{zone_counts.max()})"
    )
    logger.info(f"  Zones loaded ({zone_counts.shape[0]}): {sorted(zone_counts.index.tolist())}")
    _pt = synth['ptype'].value_counts()
    logger.info(f"  ptype: " + "  ".join(f"{k}={v}" for k, v in _pt.items()))
    if missing:
        logger.warning(f"  Could not load {len(missing)} zones: {missing[:10]}")

    return synth


def _audit_hhts_synth_consistency(hhts_persons: pd.DataFrame,
                                  synth: pd.DataFrame,
                                  landuse: pd.DataFrame,
                                  label: str = "consistency") -> None:
    """Log consistency diagnostics from HHTS generation to synthetic mapping."""
    try:
        if hhts_persons is None or synth is None:
            return
        zone_col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
        valid_zones = set(pd.to_numeric(landuse[zone_col], errors='coerce').dropna().astype(int).tolist())

        _hp = hhts_persons.copy()
        _sp = synth.copy()
        if 'person_id' not in _hp.columns and 'PERSID' in _hp.columns:
            _hp = _hp.rename(columns={'PERSID': 'person_id'})
        if 'person_id' not in _sp.columns and 'PERSID' in _sp.columns:
            _sp = _sp.rename(columns={'PERSID': 'person_id'})

        hhts_n = len(_hp)
        synth_n = len(_sp)
        synth_dup = int(_sp['person_id'].duplicated().sum()) if 'person_id' in _sp.columns else -1

        if 'OTAZ' in _sp.columns:
            _ot = pd.to_numeric(_sp['OTAZ'], errors='coerce').fillna(0).astype(int)
            bad_otaz = int((_ot <= 0).sum() + (_ot.gt(0) & (~_ot.isin(valid_zones))).sum())
        else:
            bad_otaz = -1

        worker_rate_h = float(_hp['WORKSTAT'].isin([1, 2]).mean()) if 'WORKSTAT' in _hp.columns and len(_hp) else np.nan
        worker_rate_s = float(_sp['WORKSTAT'].isin([1, 2]).mean()) if 'WORKSTAT' in _sp.columns and len(_sp) else np.nan
        student_rate_h = float(_hp['STUDYING'].isin([1, 2, 3, 4]).mean()) if 'STUDYING' in _hp.columns and len(_hp) else np.nan
        student_rate_s = float(_sp['STUDYING'].isin([1, 2, 3, 4]).mean()) if 'STUDYING' in _sp.columns and len(_sp) else np.nan

        work_anchor_cov = np.nan
        school_anchor_cov = np.nan
        if {'WORK_TAZ', 'WORKSTAT'}.issubset(_sp.columns):
            _w = _sp['WORKSTAT'].isin([1, 2])
            if _w.any():
                _wt = pd.to_numeric(_sp.loc[_w, 'WORK_TAZ'], errors='coerce').fillna(0).astype(int)
                work_anchor_cov = float((_wt.gt(0) & _wt.isin(valid_zones)).mean())
        if {'SCHOOL_TAZ', 'STUDYING'}.issubset(_sp.columns):
            _s = _sp['STUDYING'].isin([1, 2, 3, 4])
            if _s.any():
                _st = pd.to_numeric(_sp.loc[_s, 'SCHOOL_TAZ'], errors='coerce').fillna(0).astype(int)
                school_anchor_cov = float((_st.gt(0) & _st.isin(valid_zones)).mean())

        logger.info(
            f"[{label}] HHTS→synth audit: hhts_persons={hhts_n:,}, synth_persons={synth_n:,}, "
            f"synth_dup_person_id={synth_dup}, bad_synth_otaz={bad_otaz}"
        )
        logger.info(
            f"[{label}] worker_share hhts={worker_rate_h:.3f} synth={worker_rate_s:.3f} | "
            f"student_share hhts={student_rate_h:.3f} synth={student_rate_s:.3f}"
        )
        logger.info(
            f"[{label}] anchor coverage: work={work_anchor_cov:.3f} school={school_anchor_cov:.3f}"
        )
        if np.isfinite(work_anchor_cov) and work_anchor_cov < 0.95:
            logger.warning(f"[{label}] low WORK_TAZ coverage for workers: {work_anchor_cov:.1%}")
        if np.isfinite(school_anchor_cov) and school_anchor_cov < 0.95:
            logger.warning(f"[{label}] low SCHOOL_TAZ coverage for students: {school_anchor_cov:.1%}")
    except Exception as _audit_e:
        logger.warning(f"[{label}] consistency audit failed: {_audit_e}")


@timer("CDAP — daily activity pattern")
def cdap(out, synth, landuse=None, feature_override=None):
    # ── Impute HHTS-only attributes into the synth population ────────
    # bikes (and ebikes/escooter if HHTS ever carries ownership columns)
    # exist in HHTS but not in the synth files; stage-5 UEC terms
    # reference them on the person table. Stratified sampling from the
    # HHTS conditional distribution (vehicles × household size), one
    # draw per household. Done here because this is the first point
    # where HHTS persons and synth coexist, ahead of every model stage.
    try:
        from synth_attributes import sample_hhts_attributes
        synth = sample_hhts_attributes(synth, out["persons"])
    except Exception as _shx:
        logger.warning(f"HHTS attribute sampling skipped: {_shx}")

    # ── Validate cached model before use ─────────────────────────────
    # The cached cdap_model.pkl may be a null model (dill failed at save
    # time, fallback pickle saved a None or un-fitted object).  If the
    # loaded model has no coef_ or predict attribute it will silently
    # default all persons to M, inflating the traveller pool ~2× and
    # making Stage 3–4 very slow.  Delete the cache and re-estimate.
    # cdap_model.pkl is estimated from HHTS — same result regardless of run mode.
    # Keep it in saved_data/ so all runs share one cache and avoid re-estimating.
    _cache = os.path.join('saved_data', 'cdap_model.pkl')
    if os.path.exists(_cache):
        try:
            import pickle
            # ── Apply __setstate__ patch before loading ────────────────────
            # Older cached Larch Model pickles store 'availability_any' in
            # their __dict__.  Larch 6.x removed that attribute and its
            # __setstate__ now raises "Cannot set 'availability_any' on object
            # of type Model".  Patch __setstate__ before loading so the
            # exception is silently skipped (same patch used in run_cdap).
            try:
                import larch as _lx_cdap
                def _safe_ss_val(self, state):
                    _SKIP = {'availability_any', 'availability_co', 'availability_ca'}
                    for _k, _v in state.items():
                        if _k in _SKIP:
                            continue
                        try:
                            setattr(self, _k, _v)
                        except (TypeError, AttributeError):
                            pass
                for _attr in ('Model', 'NumbaModel'):
                    _cls = getattr(_lx_cdap, _attr, None)
                    if _cls is None:
                        _cls = getattr(getattr(_lx_cdap, 'model', None), _attr, None)
                    if _cls and not getattr(_cls, '_safe_ss_patched', False):
                        _cls.__setstate__ = _safe_ss_val
                        _cls._safe_ss_patched = True
            except Exception:
                pass
            with open(_cache, 'rb') as _f:
                _m = pickle.load(_f)
            # cdap.estimate_cdap() saves a dict: {'mnl': ..., 'mxl': ..., 'features': ...}
            # Older code saved a sklearn model object.  Check both forms.
            _valid = False
            if isinstance(_m, dict):
                # Dict result from cdap.estimate_cdap — valid if mnl key present and non-None
                _valid = (_m.get('mnl') is not None or
                          _m.get('sklearn') is not None)
            elif _m is not None:
                # Legacy sklearn model object
                _valid = (
                    hasattr(_m, 'predict') and
                    (getattr(_m, 'coef_', None) is not None or
                     getattr(_m, 'estimators_', None) is not None or
                     hasattr(_m, 'pf'))
                )
            if not _valid:
                logger.warning(
                    "CDAP cache invalid (null / un-fitted model) — "
                    "deleting and re-estimating"
                )
                os.remove(_cache)
        except Exception as _e:
            logger.warning(f"CDAP cache unreadable ({_e}) — deleting and re-estimating")
            try:
                os.remove(_cache)
            except OSError:
                pass
    synth = run_cdap(out["persons"], synth, feature_override=feature_override)
    synth.to_csv(os.path.join(OUT_DIR, 'cdap_output.csv'), index=False)
    return synth


@timer("Usual Locations — work & school zone")
def usual_locations(synth, landuse, hhts_tours=None):
    return run_usual_locations(
        synth, landuse,
        skims=PIPELINE_SKIMS_PATH,
        save_path=os.path.join(OUT_DIR, 'usual_locations.csv'),
        hhts_tours=hhts_tours,
        plots_dir=os.path.join(OUT_DIR, 'plots'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# COEFFICIENT CSV HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save_stage_coef_csv(models: dict, stage_name: str, save_dir: str):
    """
    Extract model coefficients from any stage's models dict and write a CSV.

    Handles three model flavours:
      • Larch / MNL — model has a .pf DataFrame (value, std_err, t_stat columns)
      • sklearn linear — model has .coef_ / .intercept_
      • BudgetLognormal / dict-based — model is a dict with 'coef', 'std_errors', etc.

    Output: {save_dir}/{stage_name}_coefficients.csv
    """
    import pandas as pd, os
    rows = []

    def _extract(key, obj):
        # ── Larch model (.pf) ──────────────────────────────────────────────
        pf = getattr(obj, 'pf', None)
        if pf is not None and hasattr(pf, 'iterrows'):
            for param, row in pf.iterrows():
                rows.append({
                    'stage':    stage_name,
                    'segment':  key,
                    'param':    param,
                    'value':    row.get('value',   float('nan')),
                    'std_err':  row.get('std_err', float('nan')),
                    't_stat':   row.get('t_stat',  float('nan')),
                    'source':   'larch_pf',
                })
            return
        # ── Dict-based (BudgetLognormal / OLS summary) ─────────────────────
        if isinstance(obj, dict):
            # Handle model-wrapper dicts like stage1's {'model': OrderedLogit, 'mu':..., ...}
            # The actual estimator is at obj['model']; recurse on it directly.
            if 'model' in obj and not obj.get('coef') and not obj.get('coefficients'):
                inner = obj['model']
                # Propagate feature names from wrapper into model object (OrderedLogit stores them)
                if hasattr(inner, 'feature_names_') and not getattr(inner, 'feature_names_', None):
                    inner.feature_names_ = obj.get('features', [])
                _extract(key, inner)
                # Also log OrderedLogit thresholds — these aren't in coef_ but are model params
                if hasattr(inner, 'thresholds_') and inner.thresholds_ is not None:
                    for j, thresh in enumerate(inner.thresholds_):
                        rows.append({'stage': stage_name, 'segment': key,
                                     'param': f'threshold_{j}', 'value': float(thresh),
                                     'std_err': float('nan'), 't_stat': float('nan'),
                                     'p_value': float('nan'), 'source': 'ordered_logit_threshold'})
                return
            coef  = obj.get('coef', obj.get('coefficients', {}))
            # MXL result uses 'std_err' (not 'std_errors'); check all variants
            serr  = obj.get('std_errors', obj.get('std_err', obj.get('se', {})))
            tstat = obj.get('t_stats', obj.get('tstat', obj.get('t_stat', {})))
            if isinstance(coef, dict):
                for param, val in coef.items():
                    rows.append({
                        'stage':   stage_name,
                        'segment': key,
                        'param':   param,
                        'value':   float(val),
                        'std_err': float(serr.get(param, float('nan'))) if serr else float('nan'),
                        't_stat':  float(tstat.get(param, float('nan'))) if tstat else float('nan'),
                        'source':  'dict',
                    })
            elif hasattr(coef, '__iter__'):
                # array-style
                feats = obj.get('features', obj.get('feature_names', [f'x{i}' for i in range(len(coef))]))
                se_arr = list(serr) if hasattr(serr, '__iter__') else []
                for i, (feat, val) in enumerate(zip(feats, coef)):
                    rows.append({
                        'stage':   stage_name,
                        'segment': key,
                        'param':   feat,
                        'value':   float(val),
                        'std_err': float(se_arr[i]) if i < len(se_arr) else float('nan'),
                        't_stat':  float('nan'),
                        'source':  'dict_array',
                    })
            return
        # ── sklearn linear (coef_, intercept_) ────────────────────────────
        coef_arr = getattr(obj, 'coef_export_values_', getattr(obj, 'coef_', None))
        if coef_arr is not None:
            feats = list(getattr(obj, 'coef_export_names_', getattr(obj, 'feature_names_', getattr(obj, 'feature_names_in_', []))))
            coef_flat = np.asarray(coef_arr).ravel()
            std_err = np.asarray(getattr(obj, 'std_err_export_', getattr(obj, 'std_err_', [])), dtype=float).ravel()
            t_stat = np.asarray(getattr(obj, 't_stat_export_', getattr(obj, 't_stat_', [])), dtype=float).ravel()
            p_value = np.asarray(getattr(obj, 'p_value_export_', getattr(obj, 'p_value_', [])), dtype=float).ravel()
            intercept = getattr(obj, 'intercept_', None)

            if intercept is not None:
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   'intercept',
                    'value':   float(intercept.ravel()[0]) if hasattr(intercept, 'ravel') else float(intercept),
                    'std_err': float(std_err[0]) if len(std_err) > 0 else float('nan'),
                    't_stat':  float(t_stat[0]) if len(t_stat) > 0 else float('nan'),
                    'p_value': float(p_value[0]) if len(p_value) > 0 else float('nan'),
                    'source':  'model_object',
                })

            if not feats:
                feats = [f'x{i}' for i in range(len(coef_flat))]
            offset = 1 if len(std_err) >= len(coef_flat) + 1 else 0
            for i, (feat, val) in enumerate(zip(feats, coef_flat)):
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   feat,
                    'value':   float(val),
                    'std_err': float(std_err[i + offset]) if len(std_err) > i + offset else float('nan'),
                    't_stat':  float(t_stat[i + offset]) if len(t_stat) > i + offset else float('nan'),
                    'p_value': float(p_value[i + offset]) if len(p_value) > i + offset else float('nan'),
                    'source':  'model_object',
                })

            return
        # ── sklearn linear (coef_, intercept_) ────────────────────────────
        coef_arr = getattr(obj, 'coef_export_values_', getattr(obj, 'coef_', None))
        if coef_arr is not None:
            feats = list(getattr(obj, 'coef_export_names_', getattr(obj, 'feature_names_', getattr(obj, 'feature_names_in_', []))))
            coef_flat = np.asarray(coef_arr).ravel()
            std_err = np.asarray(getattr(obj, 'std_err_export_', getattr(obj, 'std_err_', [])), dtype=float).ravel()
            t_stat = np.asarray(getattr(obj, 't_stat_export_', getattr(obj, 't_stat_', [])), dtype=float).ravel()
            p_value = np.asarray(getattr(obj, 'p_value_export_', getattr(obj, 'p_value_', [])), dtype=float).ravel()
            intercept = getattr(obj, 'intercept_', None)

            if intercept is not None:
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   'intercept',
                    'value':   float(intercept.ravel()[0]) if hasattr(intercept, 'ravel') else float(intercept),
                    'std_err': float(std_err[0]) if len(std_err) > 0 else float('nan'),
                    't_stat':  float(t_stat[0]) if len(t_stat) > 0 else float('nan'),
                    'p_value': float(p_value[0]) if len(p_value) > 0 else float('nan'),
                    'source':  'model_object',
                })

            if not feats:
                feats = [f'x{i}' for i in range(len(coef_flat))]
            offset = 1 if len(std_err) >= len(coef_flat) + 1 else 0
            for i, (feat, val) in enumerate(zip(feats, coef_flat)):
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   feat,
                    'value':   float(val),
                    'std_err': float(std_err[i + offset]) if len(std_err) > i + offset else float('nan'),
                    't_stat':  float(t_stat[i + offset]) if len(t_stat) > i + offset else float('nan'),
                    'p_value': float(p_value[i + offset]) if len(p_value) > i + offset else float('nan'),
                    'source':  'model_object',
                })

            if hasattr(obj, 'log_sigma_'):
                j = len(coef_flat) + offset
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   'log_sigma',
                    'value':   float(getattr(obj, 'log_sigma_')),
                    'std_err': float(std_err[j]) if len(std_err) > j else float('nan'),
                    't_stat':  float(t_stat[j]) if len(t_stat) > j else float('nan'),
                    'p_value': float(p_value[j]) if len(p_value) > j else float('nan'),
                    'source':  'model_object',
                })

        # ── Custom OrderedLogit ───────────────────────────────────────────
        elif hasattr(obj, 'coef_') and hasattr(obj, 'feature_names_') and hasattr(obj, 'thresholds_'):
            # Custom OrderedLogit from stage1_activity_frequency
            feats = getattr(obj, 'feature_names_', [f'x{i}' for i in range(len(obj.coef_))])
            for i, (feat, val) in enumerate(zip(feats, obj.coef_)):
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   feat,
                    'value':   float(val),
                    'std_err': float('nan'),  # no std err available
                    't_stat':  float('nan'),
                    'p_value': float('nan'),
                    'source':  'custom_ordered_logit',
                })
            # Add thresholds
            for i, thresh in enumerate(obj.thresholds_):
                rows.append({
                    'stage':   stage_name,
                    'segment': key,
                    'param':   f'threshold_{i}',
                    'value':   float(thresh),
                    'std_err': float('nan'),
                    't_stat':  float('nan'),
                    'p_value': float('nan'),
                    'source':  'custom_ordered_logit',
                })
            return

    for key, obj in models.items():
        # obj can itself be a dict of sub-models (e.g. stage1 {purpose → {ptype → mdl}})
        if isinstance(obj, dict) and not obj.get('coef'):
            # check if it looks like a sub-dict of models
            sub_vals = list(obj.values())
            if sub_vals and not isinstance(sub_vals[0], (int, float, str)):
                for sub_key, sub_obj in obj.items():
                    _extract(f'{key}/{sub_key}', sub_obj)
                continue
        _extract(key, obj)

    if not rows:
        logger.warning(f"  {stage_name}: no coefficients extracted — model format unrecognised")
        return

    def _stage_html_explanation(stage_name: str, coef_df: pd.DataFrame) -> str:
        if stage_name == 'stage1':
            return (
                "<p><strong>Stage 1 — ordered logit</strong>: this model predicts activity counts as an ordered outcome. "
                "Each threshold is a cutpoint between adjacent count categories, so many thresholds simply mean "
                "the target has many distinct count values. Thresholds are latent boundaries, not feature slopes. "
                "A positive coefficient increases the latent score, shifting probability toward higher counts.</p>"
                "<p>Example: if counts range from 0 to 5, there are 5 thresholds separating {0,1}, {1,2}, …, {4,5}.</p>"
            )
        elif stage_name == 'stage2':
            return (
                "<p><strong>Stage 2 — day duration</strong>: this model predicts a person-day budget. "
                "Coefficients are estimated on standardized inputs, so large absolute values usually reflect "
                "the internal scaling of the feature rather than an implausible raw effect. "
                "The intercept is the baseline prediction when all standardized predictors are zero. "
                "If the model uses a log transform, the intercept is on the transformed scale.</p>"
            )
        elif stage_name == 'stage3':
            return (
                "<p><strong>Stage 3 — activity duration</strong>: this model uses the Stage 2 budget <code>B</code> in hours. "
                "In the duration dataset, <code>B</code> is the person’s daily budget from Stage 2, and it is also used to compute "
                "<code>budget_per_activity_hr</code> and <code>day_span_hr</code>. "
                "So the <code>B</code> coefficient shows how the estimated day budget influences predicted activity durations. "
                "Because the model standardizes features before fitting, raw coefficient magnitudes can look large.</p>"
            )
        else:
            return ("<p>This report shows the estimated parameters for this stage. "
                    "Check the parameter signs, standard errors, and t-statistics for the most defensible interpretation.</p>")

    coef_path = os.path.join(save_dir, f'{stage_name}_coefficients.csv')
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(coef_path, index=False)

    explanation_html = _stage_html_explanation(stage_name, coef_df)
    html_path = os.path.join(save_dir, f'{stage_name}_coefficients.html')
    html_rows = []
    for seg, seg_df in coef_df.sort_values(['segment', 'param']).groupby('segment', dropna=False):
        html_rows.append(
            f"<h2>{seg}</h2>"
            "<table><tr><th>Parameter</th><th>Value</th><th>Std Err</th>"
            "<th>t-stat</th><th>p-value</th><th>Source</th></tr>"
        )
        for _, r in seg_df.iterrows():
            pval = r.get('p_value', r.get('pval', float('nan')))
            html_rows.append(
                "<tr>"
                f"<td>{r['param']}</td>"
                f"<td style='text-align:right'>{float(r['value']):.6f}</td>"
                f"<td style='text-align:right'>{float(r.get('std_err', float('nan'))):.6f}</td>"
                f"<td style='text-align:right'>{float(r.get('t_stat', float('nan'))):.3f}</td>"
                f"<td style='text-align:right'>{float(pval):.4g}</td>"
                f"<td>{r.get('source', '')}</td>"
                "</tr>"
            )
        html_rows.append("</table>")

    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{stage_name} coefficients</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;font-size:13px}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "th{background:#2c3e50;color:white;padding:6px 10px;text-align:left}"
        "td{border:1px solid #ddd;padding:5px 10px}"
        "tr:nth-child(even){background:#f9f9f9}"
        "h1,h2{color:#2c3e50}</style></head><body>"
        f"<h1>{stage_name} coefficients</h1>"
        f"<p>Exported {len(coef_df)} parameters from {len(coef_df['segment'].astype(str).unique())} segment(s).</p>"
        f"{explanation_html}"
        + "\n".join(html_rows) +
        "</body></html>"
    )
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    logger.info(f"  {stage_name} coefficients → {coef_path} ({len(rows)} params)")
    logger.info(f"  {stage_name} coefficient HTML → {html_path}")


def _auto_prune_enabled(stage_name: str) -> bool:
    """Return True when automatic prune+refit is enabled for this stage."""
    global_flag = os.environ.get('AUTO_PRUNE_REFIT', '1') == '1'
    stage_flag = os.environ.get(f'AUTO_PRUNE_REFIT_{stage_name.upper()}', '1') == '1'
    return global_flag and stage_flag


def _parse_name_set(csv_text: str) -> set[str]:
    if not csv_text:
        return set()
    return {x.strip() for x in csv_text.split(',') if x.strip()}


def _is_significant_row(row: pd.Series,
                        p_thresh: float = 0.10,
                        t_thresh: float = 1.96) -> bool:
    """Significance rule with p-value preferred, t-stat fallback."""
    p_cols = ['p_value', 'pval', 'pvalue']
    t_cols = ['t_stat', 'tstat']

    p_val = np.nan
    t_val = np.nan

    for c in p_cols:
        if c in row.index:
            p_val = pd.to_numeric(row.get(c), errors='coerce')
            if np.isfinite(p_val):
                break
    for c in t_cols:
        if c in row.index:
            t_val = pd.to_numeric(row.get(c), errors='coerce')
            if np.isfinite(t_val):
                break

    if np.isfinite(p_val):
        return bool(p_val <= p_thresh)
    if np.isfinite(t_val):
        return bool(abs(t_val) >= t_thresh)
    return False


def _auto_prune_features_from_coef_csv(
    stage_name: str,
    candidate_features: list[str],
    save_dir: str,
    protected_features: Optional[set[str]] = None,
    p_thresh: float = 0.10,
    t_thresh: float = 1.96,
    min_keep: int = 4,
) -> Optional[list[str]]:
    """
    Build a pruned feature override from an exported stage coefficient CSV.

    Returns a pruned feature list, or None if no safe prune is available.
    """
    coef_path = os.path.join(save_dir, f'{stage_name}_coefficients.csv')
    if not os.path.exists(coef_path):
        return None

    try:
        df = pd.read_csv(coef_path)
    except Exception as exc:
        logger.warning(f"  {stage_name}: could not read coefficient CSV for pruning: {exc}")
        return None

    if df.empty:
        return None

    param_col = 'param' if 'param' in df.columns else ('parameter' if 'parameter' in df.columns else None)
    if not param_col:
        return None

    protected = set(protected_features or set())
    protected |= _parse_name_set(os.environ.get(f'{stage_name.upper()}_PRUNE_PROTECT', ''))

    cand_set = set(candidate_features)
    df = df[df[param_col].astype(str).isin(cand_set)].copy()
    if df.empty:
        return None

    keep = set()
    for _, row in df.iterrows():
        name = str(row.get(param_col, '')).strip()
        if not name:
            continue
        if name in protected or _is_significant_row(row, p_thresh=p_thresh, t_thresh=t_thresh):
            keep.add(name)

    keep |= (cand_set & protected)
    ordered = [f for f in candidate_features if f in keep]

    if len(ordered) < max(min_keep, len(cand_set & protected)):
        logger.info(f"  {stage_name}: prune skipped — too few statistically defensible features ({len(ordered)})")
        return None
    if len(ordered) >= len(candidate_features):
        return None
    return ordered


def _extract_stage5_warm_coefs(models: dict) -> dict:
    """Extract a flat warm-start coefficient dict per segment from stage5 model results."""
    warm = {}
    for seg, res in (models or {}).items():
        if not isinstance(res, dict):
            continue
        coefs = {}
        for src in ('mode_coef', 'coef', 'dest_coef'):
            c = res.get(src, {})
            if isinstance(c, dict):
                coefs.update(c)
        if coefs:
            warm[seg] = coefs
    return warm


def _extract_expr_vars(expr: str) -> set[str]:
    """Extract source variable tokens used in a Larch expression."""
    vars_found = set()
    for token in re.findall(r'od\.([A-Za-z0-9_]+)', str(expr)):
        vars_found.add(token)
    for token in re.findall(r'emp\.([A-Za-z0-9_]+)', str(expr)):
        vars_found.add(token)
    if 'logsum' in str(expr):
        vars_found.add('logsum')
    return vars_found


def _derive_stage5_pruned_overrides(
    coef_csv_path: str,
    p_thresh: float = 0.10,
    t_thresh: float = 1.96,
) -> tuple[list[str], list[str], dict]:
    """
    Convert Stage 5 coefficient significance into mode/destination override lists.

    Protected destination structure terms (distance/crash/size/logsum) are always
    retained so OD behavior does not collapse.
    """
    from stage5_destination_choice import MODE_UEC, DEST_UEC

    if not os.path.exists(coef_csv_path):
        return [], [], {'changed': False, 'reason': 'missing_coefficient_csv'}

    try:
        df = pd.read_csv(coef_csv_path)
    except Exception as exc:
        logger.warning(f"  stage5 prune: failed to read {coef_csv_path}: {exc}")
        return [], [], {'changed': False, 'reason': 'read_failed'}

    if df.empty or 'parameter' not in df.columns or 'model_type' not in df.columns:
        return [], [], {'changed': False, 'reason': 'bad_schema'}

    mode_keep_coeff = set(_parse_name_set(os.environ.get('STAGE5_MODE_PROTECT_COEFFS', '')))
    dest_keep_coeff = set(_parse_name_set(os.environ.get('STAGE5_DEST_PROTECT_COEFFS', '')))

    # Never prune these destination structure terms.
    dest_keep_coeff |= {
        'c_logsum', 'c_log_dist', 'c_dist', 'c_crash',
        'c_size_term', 'c_aggl', 'c_comp',
    }

    for _, row in df.iterrows():
        coeff = str(row.get('parameter', '')).strip()
        mtype = str(row.get('model_type', '')).strip().lower()
        if not coeff or mtype not in ('mode', 'dest'):
            continue
        if _is_significant_row(row, p_thresh=p_thresh, t_thresh=t_thresh):
            if mtype == 'mode':
                mode_keep_coeff.add(coeff)
            else:
                dest_keep_coeff.add(coeff)

    # Map kept coefficient names back to override variable tokens.
    mode_vars = set(_parse_name_set(os.environ.get('STAGE5_MODE_PROTECT_VARS', '')))
    dest_vars = set(_parse_name_set(os.environ.get('STAGE5_DEST_PROTECT_VARS', '')))

    # Critical OD terms protected by default.
    dest_vars |= {'DIST', 'SOVTOLL_DIST__AM', 'SOVTOLL_TIME__AM', 'CRASH',
                  'size_term', 'acdc_aggl', 'acdc_comp', 'logsum'}

    for coeff, (expr, _) in MODE_UEC.items():
        if coeff in mode_keep_coeff:
            mode_vars |= _extract_expr_vars(expr)

    for coeff, (expr, _) in DEST_UEC.items():
        if coeff in dest_keep_coeff:
            dest_vars |= _extract_expr_vars(expr)

    mode_override = sorted(mode_vars)
    dest_override = sorted(dest_vars)
    changed = bool(mode_override or dest_override)
    meta = {
        'changed': changed,
        'mode_n': len(mode_override),
        'dest_n': len(dest_override),
        'mode_coeff_kept': len(mode_keep_coeff),
        'dest_coeff_kept': len(dest_keep_coeff),
    }
    return mode_override, dest_override, meta


# ─────────────────────────────────────────────────────────────────────────────
# STAGE DIAGNOSTICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _log_stage_summary(stage: str, models: dict, label: str = "estimation") -> None:
    """
    Log a compact per-segment diagnostic table after each stage.

    Called after every estimate_stageN() call so the log always shows
    what was actually fitted — not just "saved to path".

    Columns vary by stage:
        Stage 1  — purpose, n_tours, n_params, LL, rho²
        Stage 2  — model_name, converged, RMSE, sigma
        Stage 3  — segment, n, converged, RMSE
        Stage 4  — timing_type, n, converged, RMSE
        Stage 5  — segment, mode_LL, dest_LL, n_tours, frozen_probs,
                   rho2_mode, rho2_dest, walk_pct, bike_pct
    """
    import math
    SEP = "─" * 80
    logger.info(SEP)
    logger.info(f"  STAGE {stage.upper()} SUMMARY  ({label})")
    logger.info(SEP)

    if not models:
        logger.warning(f"  Stage {stage}: models dict is empty — nothing was estimated!")
        logger.info(SEP)
        return

    def _mget(m, key, default=None):
        """Get a field from either a dict entry or an sklearn-style model object.
        If m is a dict without the key, auto-unwraps m['model'] if present."""
        if isinstance(m, dict):
            # dict keys may be 'n_obs'/'n'/'n_', 'converged'/'converged_', etc.
            for k in (key, key.rstrip('_'), key + '_'):
                if k in m:
                    return m[k]
            # auto-unwrap: stages 2/3 store {'model': obj, 'features': ..., ...}
            if 'model' in m:
                inner = m['model']
                return getattr(inner, key,
                       getattr(inner, key.rstrip('_'),
                       getattr(inner, key + '_', default)))
            return default
        return getattr(m, key, getattr(m, key.rstrip('_'), getattr(m, key + '_', default)))

    def _best_entry(m):
        """For Stage-1's nested {ptype: entry} dict, return highest-n_obs entry.
        Handles both {ptype: dict} and {ptype: model_object} structures."""
        if isinstance(m, dict) and m:
            first = next(iter(m.values()))
            if isinstance(first, dict):
                # values are plain dicts — pick highest n_obs
                try:
                    return max(m.values(), key=lambda e: e.get('n_obs', 0))
                except Exception:
                    return first
            elif not isinstance(first, (int, float, str, bool, type(None))):
                # values are model objects — pick the one with most observations
                def _nobs(obj):
                    for attr in ('n_obs_', 'n_obs', 'n_', 'n'):
                        v = getattr(obj, attr, None)
                        if v is not None:
                            try: return int(v)
                            except Exception: pass
                    return 0
                try:
                    return max(m.values(), key=_nobs)
                except Exception:
                    return first
        return m

    if stage == '1':
        # Stage 1: {purpose: {ptype: {'n_obs': n, 'converged': bool, 'll': float, ...}}}
        header = f"  {'Purpose':<15} {'n_tours':>8} {'n_params':>9} {'LL':>12} {'null_LL':>10} {'rho2':>7} {'converged':>10}"
        logger.info(header)
        for seg, m_raw in models.items():
            try:
                m  = _best_entry(m_raw)
                try:
                    n = int(np.asarray(_mget(m, 'n_obs', _mget(m, 'n_', 0))).flat[0])
                except Exception:
                    n = '?'
                # n_params: from coef_+thresholds_ (may be numpy arrays, not lists)
                _coef_raw = _mget(m, 'coef_', None)
                _thr_raw  = _mget(m, 'thresholds_', None)
                try:
                    _coef_len = int(np.asarray(_coef_raw).size) if _coef_raw is not None else 0
                except Exception:
                    _coef_len = 0
                try:
                    _thr_len = int(np.asarray(_thr_raw).size) if _thr_raw is not None else 0
                except Exception:
                    _thr_len = 0
                np_ = (_coef_len + _thr_len) if (_coef_len + _thr_len > 0) else int(_mget(m, 'n_params', 0) or 0)
                try:
                    ll = float(np.asarray(_mget(m, 'll', _mget(m, 'll_', float('nan')))).flat[0])
                except Exception:
                    ll = float('nan')
                try:
                    ll0 = float(np.asarray(_mget(m, 'll0', _mget(m, 'll0_', float('nan')))).flat[0])
                except Exception:
                    ll0 = float('nan')
                rho2 = (1 - ll/ll0) if (math.isfinite(ll) and math.isfinite(ll0) and ll0 != 0) else float('nan')
                conv_raw = _mget(m, 'converged', _mget(m, 'converged_', None))
                try:
                    _conv_bool = bool(np.asarray(conv_raw).flat[0]) if conv_raw is not None else False
                except Exception:
                    _conv_bool = False
                conv = '✓' if _conv_bool else '✗'
                rho2_s = f"{rho2:.4f}" if math.isfinite(rho2) else '     nan'
                ll_s   = f"{ll:.1f}"   if math.isfinite(ll)   else '    nan'
                ll0_s  = f"{ll0:.1f}"  if math.isfinite(ll0)  else '    nan'
                logger.info(f"  {str(seg):<15} {str(n):>8} {str(np_):>9} {ll_s:>12} {ll0_s:>10} {rho2_s:>7} {conv:>10}")
            except Exception as _e:
                logger.info(f"  {str(seg):<15}  (diagnostic failed: {_e})")

    elif stage == '2':
        # Stage 2: {name: {'n_obs': n, 'rmse': float, 'sigma': float, 'll': float, 'converged': bool}}
        header = f"  {'Model':<22} {'n':>7} {'converged':>10} {'RMSE':>8} {'sigma':>8} {'LL':>10}"
        logger.info(header)
        for name, m in models.items():
            try:
                n    = _mget(m, 'n_obs', _mget(m, 'n_', '?'))
                conv_raw = _mget(m, 'converged', _mget(m, 'converged_', True))
                conv = '✓' if bool(np.asarray(conv_raw).flat[0]) else '✗'
                rmse = _mget(m, 'rmse', _mget(m, 'rmse_', float('nan')))
                sig  = _mget(m, 'sigma', _mget(m, 'sigma_', float('nan')))
                ll   = _mget(m, 'll', _mget(m, 'll_', float('nan')))
                rmse_s = f"{rmse:.2f}" if isinstance(rmse, float) and math.isfinite(rmse) else '    nan'
                sig_s  = f"{sig:.2f}"  if isinstance(sig,  float) and math.isfinite(sig)  else '    nan'
                ll_s   = f"{ll:.1f}"   if isinstance(ll,   float) and math.isfinite(ll)   else '    nan'
                logger.info(f"  {str(name):<22} {str(n):>7} {conv:>10} {rmse_s:>8} {sig_s:>8} {ll_s:>10}")
            except Exception as _e:
                logger.info(f"  {str(name):<22}  (diagnostic failed: {_e})")

    elif stage == '3':
        header = f"  {'Segment':<22} {'n':>7} {'converged':>10} {'RMSE':>8} {'sigma':>8}"
        logger.info(header)
        for name, m in models.items():
            try:
                n    = _mget(m, 'n_obs', _mget(m, 'n_', '?'))
                conv_raw = _mget(m, 'converged', _mget(m, 'converged_', True))
                conv = '✓' if bool(np.asarray(conv_raw).flat[0]) else '✗'
                rmse = _mget(m, 'rmse', _mget(m, 'rmse_', float('nan')))
                sig  = _mget(m, 'sigma', _mget(m, 'sigma_', float('nan')))
                rmse_s = f"{rmse:.2f}" if isinstance(rmse, float) and math.isfinite(rmse) else '    nan'
                sig_s  = f"{sig:.2f}"  if isinstance(sig,  float) and math.isfinite(sig)  else '    nan'
                logger.info(f"  {str(name):<22} {str(n):>7} {conv:>10} {rmse_s:>8} {sig_s:>8}")
            except Exception as _e:
                logger.info(f"  {str(name):<22}  (diagnostic failed: {_e})")

    elif stage == '4':
        header = f"  {'Model':<25} {'n':>7} {'converged':>10} {'rho2':>8} {'null_LL':>10} {'LL':>10}"
        logger.info(header)
        for name, m in models.items():
            try:
                n        = _mget(m, 'n_obs', _mget(m, 'n_obs_', _mget(m, 'n_', '?')))
                conv_raw = _mget(m, 'converged', _mget(m, 'converged_', True))
                conv     = '✓' if bool(np.asarray(conv_raw).flat[0]) else '✗'
                rho2     = _mget(m, 'rho2', _mget(m, 'rho2_', float('nan')))
                rho2_s   = f"{float(rho2):.3f}" if isinstance(rho2, (int, float)) and math.isfinite(float(rho2)) else '     nan'
                ll_val   = _mget(m, 'll', _mget(m, 'll_', float('nan')))
                ll_s     = f"{float(ll_val):.1f}" if isinstance(ll_val, (int, float)) and math.isfinite(float(ll_val)) else '   nan'
                null_ll  = _mget(m, 'null_ll', _mget(m, 'null_ll_', float('nan')))
                null_s   = f"{float(null_ll):.1f}" if isinstance(null_ll, (int, float)) and math.isfinite(float(null_ll)) else '   nan'
                logger.info(f"  {str(name):<25} {str(n):>7} {conv:>10} {rho2_s:>8} {null_s:>10} {ll_s:>10}")
            except Exception as _e:
                logger.info(f"  {str(name):<25}  (diagnostic failed: {_e})")

    elif stage == '5':
        # Stage 5: {seg: dict(mode_model, dest_model, mode_ll, dest_ll, ...)}
        header = (f"  {'Seg':<10} {'n_tours':>8} {'mode_LL':>10} {'rho2_m':>8} "
                  f"{'dest_LL':>10} {'rho2_d':>8} {'frozen':>8} {'walk%':>7} {'bike%':>7}")
        logger.info(header)
        for seg, res in models.items():
            try:
                n      = res.get('n_tours', '?')
                mll    = res.get('mode_ll', float('nan'))
                dll    = res.get('dest_ll', float('nan'))
                frz    = '✓' if res.get('frozen_probs') is not None else '✗'
                # rho² for mode model
                mm = res.get('mode_model')
                dm = res.get('dest_model')
                def _rho2(mdl):
                    try:
                        pf = getattr(mdl, '_final_pf', None) or getattr(mdl, 'pf', None)
                        if pf is not None and 'value' in pf.columns:
                            ll_val  = float(pf.loc['rho_sq_adj', 'value']) if 'rho_sq_adj' in pf.index else float('nan')
                            return ll_val
                    except Exception:
                        pass
                    # fallback: derive from ll and null_ll stored on model
                    ll  = getattr(mdl, 'rho2_', float('nan'))
                    return ll if isinstance(ll, float) else float('nan')
                rho2_m = _rho2(mm) if mm else float('nan')
                rho2_d = _rho2(dm) if dm else float('nan')
                # mode share diagnostics
                fp = res.get('frozen_probs')
                walk_pct = bike_pct = float('nan')
                if fp is not None and hasattr(fp, 'columns'):
                    for walk_col in ['fp_Walk', 'p_Walk', 'Walk']:
                        if walk_col in fp.columns:
                            walk_pct = float(fp[walk_col].mean() * 100)
                            break
                    for bike_col in ['fp_Bike', 'p_Bike', 'Bike']:
                        if bike_col in fp.columns:
                            bike_pct = float(fp[bike_col].mean() * 100)
                            break
                def _fs(v): return f"{v:.1f}" if isinstance(v, float) and math.isfinite(v) else '  nan'
                logger.info(
                    f"  {str(seg):<10} {str(n):>8} {_fs(mll):>10} {_fs(rho2_m):>8} "
                    f"{_fs(dll):>10} {_fs(rho2_d):>8} {frz:>8} {_fs(walk_pct):>7} {_fs(bike_pct):>7}"
                )
            except Exception as _e:
                logger.info(f"  {str(seg):<10}  (diagnostic failed: {_e})")

    else:
        # Generic: just log keys and sizes
        for k, v in models.items():
            logger.info(f"  {k}: {type(v).__name__}")

    logger.info(SEP)


def _write_stage_checkpoint_report(
    stage_tag: str,
    title: str,
    summary: dict,
    table: Optional[pd.DataFrame] = None,
    max_rows: int = 200,
) -> None:
    """
    Persist a compact stage checkpoint in machine-readable and paper-ready forms.

    Files written under OUT_DIR/stage_reports:
      - <stage_tag>.json
      - <stage_tag>.html
      - <stage_tag>.tex
      - <stage_tag>_table.csv   (optional)
    For GA stages, includes selected features prominently in the report.
    BIC values shown as positive (minimized, lower = better).
    """
    try:
        rep_dir = os.path.join(OUT_DIR, 'stage_reports')
        os.makedirs(rep_dir, exist_ok=True)

        def _jsonable(v):
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (list, tuple)):
                return [_jsonable(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _jsonable(x) for k, x in v.items()}
            try:
                return float(v)
            except Exception:
                return str(v)

        clean_summary = {str(k): _jsonable(v) for k, v in (summary or {}).items()}
        clean_summary['stage_tag'] = stage_tag
        clean_summary['title'] = title
        clean_summary['timestamp'] = pd.Timestamp.now().isoformat()

        json_path = os.path.join(rep_dir, f'{stage_tag}.json')
        with open(json_path, 'w', encoding='utf-8') as _jf:
            json.dump(clean_summary, _jf, indent=2)

        summary_df = pd.DataFrame([
            {'metric': k, 'value': v}
            for k, v in clean_summary.items()
            if k not in (
                'stage_tag', 'title', 'timestamp',
                'selected_features', 'selected_features_list', 'selected_features_by_block'
            )
        ])

        table_html = '<p>No table preview available.</p>'
        table_latex = ''
        table_title = 'Table Preview'
        csv_written = None
        if isinstance(table, pd.DataFrame) and not table.empty:
            tbl = table.head(max_rows).copy()
            csv_path = os.path.join(rep_dir, f'{stage_tag}_table.csv')
            tbl.to_csv(csv_path, index=False)
            csv_written = csv_path
            table_html = tbl.to_html(index=False)
            try:
                table_latex = tbl.to_latex(index=False, escape=True)
            except Exception:
                table_latex = ''
            cols_lower = {str(c).lower() for c in tbl.columns}
            if 'generation' in cols_lower:
                table_title = 'Convergence History (Generations)'

        selected_features = str(clean_summary.get('selected_features', '') or '').strip()
        selected_features_by_block = clean_summary.get('selected_features_by_block', {}) or {}
        selected_features_html = ''
        selected_features_tex = ''
        if selected_features:
            block_html = ''
            if isinstance(selected_features_by_block, dict) and selected_features_by_block:
                rows = []
                for block, vals in selected_features_by_block.items():
                    vals_str = ', '.join(str(v) for v in vals) if isinstance(vals, list) else str(vals)
                    rows.append(
                        "<tr><td><strong>" + str(block) + "</strong></td><td>" + vals_str + "</td></tr>"
                    )
                block_html = (
                    "<h3>Selected Variables by Block</h3>"
                    "<table><thead><tr><th>Block</th><th>Variables</th></tr></thead><tbody>"
                    + ''.join(rows)
                    + "</tbody></table>"
                )

            selected_features_html = (
                "<h2>Selected Variables / Model Specification</h2>"
                "<pre style='background:#f6f8fa;padding:10px;border:1px solid #ddd;white-space:pre-wrap'>"
                f"{selected_features}"
                "</pre>"
                + block_html
            )

            selected_features_tex = (
                "\n\\subsection*{Selected Variables}\n"
                "\\begin{quote}\n"
                + selected_features.replace('_', '\\_')
                + "\n\\end{quote}\n"
            )

        html_path = os.path.join(rep_dir, f'{stage_tag}.html')
        html_content = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:18px}"
            "table{border-collapse:collapse;width:100%;margin-top:10px}"
            "th{background:#2c3e50;color:#fff;padding:6px;text-align:left}"
            "td{border:1px solid #ddd;padding:6px;vertical-align:top}"
            "h2{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:5px}"
            "h3{color:#1f3a56;margin-top:16px}"
            "</style></head><body>"
            f"<h1>{title}</h1>"
            f"{selected_features_html}"
            "<h2>Summary Metrics</h2>"
            f"{summary_df.to_html(index=False)}"
            f"<h2>{table_title}</h2>"
            f"{table_html}"
            "</body></html>"
        )
        with open(html_path, 'w', encoding='utf-8') as _hf:
            _hf.write(html_content)

        tex_path = os.path.join(rep_dir, f'{stage_tag}.tex')
        tex_content = (
            "\\section*{" + str(title).replace('_', '\\_') + "}\n"
            "\\subsection*{Summary Metrics}\n"
            + summary_df.to_latex(index=False, escape=True)
            + selected_features_tex
        )
        if table_latex:
            tex_content += "\n\\subsection*{" + table_title.replace('_', '\\_') + "}\n"
            tex_content += table_latex

        with open(tex_path, 'w', encoding='utf-8') as _tf:
            _tf.write(tex_content)

        if csv_written:
            logger.info(f"  Stage report table CSV saved → {csv_written}")
        logger.info(f"  Stage report bundle saved → {rep_dir}/{stage_tag}.*")
    except Exception as _sr_e:
        logger.warning(f"  Stage checkpoint report failed ({stage_tag}): {_sr_e}")


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATION WRAPPERS  (Phase 1 — HHTS only)
# Each returns the fitted models dict and saves PKL + HTML to OUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

@timer("Estimate Stage 1 — activity frequency")
def estimate_stage1(out, landuse=None):
    from stage1_activity_frequency import estimate_frequency_models, save_models, prepare_frequency_data
    import pandas as pd
    logger.info("  Preparing HHTS frequency data...")
    models = estimate_frequency_models(out["tours"], out["persons"], landuse=landuse)
    path = os.path.join(OUT_DIR, 'stage1_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage1', OUT_DIR)
    if _auto_prune_enabled('stage1'):
        logger.info("  Stage 1 auto-prune skipped: ordered-logit exports do not provide reliable p-values for feature-level pruning.")
    logger.info(f"  Stage 1 models → {path}  (HTML report alongside)")
    _log_stage_summary('1', models)
    _coef = pd.read_csv(os.path.join(OUT_DIR, 'stage1_coefficients.csv')) \
        if os.path.exists(os.path.join(OUT_DIR, 'stage1_coefficients.csv')) else None
    _write_stage_checkpoint_report(
        stage_tag='estimate_stage1',
        title='Estimate Stage 1 - Activity Frequency',
        summary={
            'n_segments': len(models),
            'model_path': path,
            'coefficients_csv': os.path.join(OUT_DIR, 'stage1_coefficients.csv'),
        },
        table=_coef,
    )
    return models


@timer("Estimate Stage 2 — day duration")
def estimate_stage2(out):
    from stage2_day_duration import (
        estimate_day_models,
        save_models,
        aggregate_person_day,
        PERSON_FEATURES,
    )
    logger.info("  Preparing HHTS person-day data...")
    person_day = aggregate_person_day(out["tours"], out["persons"])
    logger.info(f"  person_day rows: {len(person_day):,}  "
                f"cols: {list(person_day.columns[:8])}")
    models = estimate_day_models(person_day)
    path = os.path.join(OUT_DIR, 'stage2_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage2', OUT_DIR)

    if _auto_prune_enabled('stage2'):
        protected = {'incp', 'vehrd', 'has_vehicle', 'WORKSTAT', 'STUDYING'}
        pruned = _auto_prune_features_from_coef_csv(
            stage_name='stage2',
            candidate_features=list(PERSON_FEATURES),
            save_dir=OUT_DIR,
            protected_features=protected,
            p_thresh=float(os.environ.get('STAGE2_PRUNE_P_THRESHOLD', '0.10')),
            t_thresh=float(os.environ.get('STAGE2_PRUNE_T_THRESHOLD', '1.96')),
            min_keep=int(os.environ.get('STAGE2_PRUNE_MIN_KEEP', '6')),
        )
        if pruned:
            logger.info(f"  Stage 2 prune+refit: {len(PERSON_FEATURES)} → {len(pruned)} person features")
            models = estimate_day_models(person_day, feature_override=pruned)
            save_models(models, path)
            _save_stage_coef_csv(models, 'stage2', OUT_DIR)

    logger.info(f"  Stage 2 models → {path}  (HTML report alongside)")
    _log_stage_summary('2', models)
    _coef = pd.read_csv(os.path.join(OUT_DIR, 'stage2_coefficients.csv')) \
        if os.path.exists(os.path.join(OUT_DIR, 'stage2_coefficients.csv')) else None
    _write_stage_checkpoint_report(
        stage_tag='estimate_stage2',
        title='Estimate Stage 2 - Day Duration',
        summary={
            'n_models': len(models),
            'person_day_rows': len(person_day),
            'model_path': path,
            'coefficients_csv': os.path.join(OUT_DIR, 'stage2_coefficients.csv'),
        },
        table=_coef,
    )
    return models


@timer("Estimate Stage 2 — joint participation + duration (Tobit-I / Heckman)")
def estimate_stage2_joint(out, s2_models=None):
    """
    Optional companion to estimate_stage2().

    Estimates two joint participation + duration models as alternatives to the
    sequential two-stage approach, then writes a comparison HTML report:

        A.  Tobit Type-I  — left-censor B at 0, include non-travellers
        B.  Heckman (Heckit) — probit selection + OLS outcome with IMR

    Controlled by env-var JOINT_PARTICIPATION_MODEL=1 (default: 0 = skip).

    Outputs
    -------
    stage2_joint_models.pkl          : {tobit1, heckman} model objects
    stage2_joint_comparison.csv      : per-model fit metrics
    stage2_joint_model_comparison.html : full comparison report
    plots/joint_participation_comparison.png
    plots/joint_budget_scatter.png
    plots/joint_coefficient_imr.png
    """
    if os.environ.get('JOINT_PARTICIPATION_MODEL', '0') != '1':
        logger.info("  JOINT_PARTICIPATION_MODEL not set — skipping joint estimation "
                    "(set JOINT_PARTICIPATION_MODEL=1 to enable)")
        return None

    from stage2_joint_participation import (
        aggregate_person_day_with_nontravellers,
        estimate_joint_models,
        compare_models,
        plot_joint_comparison,
        write_joint_html_report,
        save_joint_models,
    )

    logger.info("  Building full person-day dataset (including non-travellers)...")
    try:
        full_pd = aggregate_person_day_with_nontravellers(out["tours"], out["persons"])
    except Exception as e:
        logger.warning(f"  Could not build full person-day with non-travellers: {e}")
        logger.warning("  Falling back to travellers-only dataset.")
        from stage2_day_duration import aggregate_person_day
        full_pd = aggregate_person_day(out["tours"], out["persons"])
        full_pd['participated'] = 1

    nt_pct = 100 * (1 - full_pd['participated'].mean())
    logger.info(f"  Full dataset: {len(full_pd):,} persons  "
                f"({nt_pct:.1f}% non-travellers)")

    logger.info("  Estimating joint models (Tobit-I + Heckman)...")
    joint_res = estimate_joint_models(full_pd)

    # save joint model objects
    joint_pkl = os.path.join(OUT_DIR, 'stage2_joint_models.pkl')
    save_joint_models(joint_res, joint_pkl)
    logger.info(f"  Joint models → {joint_pkl}")

    # comparison metrics vs two-stage
    logger.info("  Computing comparison metrics...")
    comparison = compare_models(
        joint_res['tobit1'],
        joint_res['heckman'],
        s2_models,   # may be None if called without two-stage
        full_pd,
    )

    # save metrics CSV
    csv_path = os.path.join(OUT_DIR, 'stage2_joint_comparison.csv')
    comparison['metrics'].to_csv(csv_path, index=False)
    logger.info(f"  Comparison metrics → {csv_path}")

    # plots
    try:
        os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)
        plot_joint_comparison(comparison, OUT_DIR)
    except Exception as e:
        logger.warning(f"  Joint comparison plots failed: {e}")

    # HTML report
    try:
        html_path = write_joint_html_report(comparison, OUT_DIR, s2_models)
        logger.info(f"  Joint comparison report → {html_path}")
    except Exception as e:
        logger.warning(f"  Joint comparison HTML failed: {e}")

    # log recommendation
    from stage2_joint_participation import joint_model_recommendation
    rec_html = joint_model_recommendation(comparison)
    # strip tags for plain-text log
    import re as _re
    rec_plain = _re.sub(r'<[^>]+>', ' ', rec_html).strip()
    logger.info(f"\n{'='*60}\n  JOINT MODEL RECOMMENDATION:\n{rec_plain[:800]}\n{'='*60}")

    _write_stage_checkpoint_report(
        stage_tag='estimate_stage2_joint',
        title='Estimate Stage 2 Joint - Tobit and Heckman',
        summary={
            'full_person_day_rows': len(full_pd),
            'comparison_csv': csv_path,
            'joint_models_pkl': joint_pkl,
            'report_html': os.path.join(OUT_DIR, 'stage2_joint_model_comparison.html'),
        },
        table=comparison.get('metrics') if isinstance(comparison, dict) else None,
    )

    return joint_res


@timer("Estimate Stage 3 — activity duration")
def estimate_stage3(out):
    from stage3_activity_duration import estimate_duration_models, save_models, PERSON_FEATURES
    logger.info("  Preparing HHTS duration data...")
    _use_mdcev = os.environ.get('STAGE3_USE_MDCEV', '0') == '1'
    models = estimate_duration_models(
        out["tours"], out["persons"], trips=out["trips"], use_mdcev=_use_mdcev
    )
    path = os.path.join(OUT_DIR, 'stage3_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage3', OUT_DIR)

    if _auto_prune_enabled('stage3'):
        protected = {'incp', 'vehrd', 'has_vehicle', 'WORKSTAT', 'STUDYING'}
        pruned = _auto_prune_features_from_coef_csv(
            stage_name='stage3',
            candidate_features=list(PERSON_FEATURES),
            save_dir=OUT_DIR,
            protected_features=protected,
            p_thresh=float(os.environ.get('STAGE3_PRUNE_P_THRESHOLD', '0.10')),
            t_thresh=float(os.environ.get('STAGE3_PRUNE_T_THRESHOLD', '1.96')),
            min_keep=int(os.environ.get('STAGE3_PRUNE_MIN_KEEP', '6')),
        )
        if pruned:
            logger.info(f"  Stage 3 prune+refit: {len(PERSON_FEATURES)} → {len(pruned)} person features")
            models = estimate_duration_models(
                out["tours"], out["persons"], trips=out["trips"],
                use_mdcev=_use_mdcev,
                feature_override=pruned,
            )
            save_models(models, path)
            _save_stage_coef_csv(models, 'stage3', OUT_DIR)

    logger.info(f"  Stage 3 models → {path}  (HTML report alongside)")
    _log_stage_summary('3', models)
    _coef = pd.read_csv(os.path.join(OUT_DIR, 'stage3_coefficients.csv')) \
        if os.path.exists(os.path.join(OUT_DIR, 'stage3_coefficients.csv')) else None
    _write_stage_checkpoint_report(
        stage_tag='estimate_stage3',
        title='Estimate Stage 3 - Activity Duration',
        summary={
            'n_models': len(models),
            'model_path': path,
            'coefficients_csv': os.path.join(OUT_DIR, 'stage3_coefficients.csv'),
        },
        table=_coef,
    )
    return models


@timer("Estimate Stage 4 — schedule placement")
def estimate_stage4(out):
    from stage4_schedule_placement import estimate_timing_models, save_models, PERSON_FEATURES
    logger.info("  Preparing HHTS timing data...")
    models = estimate_timing_models(out["trips"], out["tours"], out["persons"])
    path = os.path.join(OUT_DIR, 'stage4_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage4', OUT_DIR)

    if _auto_prune_enabled('stage4'):
        protected = {'incp', 'vehrd', 'has_vehicle', 'WORKSTAT', 'STUDYING'}
        pruned = _auto_prune_features_from_coef_csv(
            stage_name='stage4',
            candidate_features=list(PERSON_FEATURES),
            save_dir=OUT_DIR,
            protected_features=protected,
            p_thresh=float(os.environ.get('STAGE4_PRUNE_P_THRESHOLD', '0.10')),
            t_thresh=float(os.environ.get('STAGE4_PRUNE_T_THRESHOLD', '1.96')),
            min_keep=int(os.environ.get('STAGE4_PRUNE_MIN_KEEP', '6')),
        )
        if pruned:
            logger.info(f"  Stage 4 prune+refit: {len(PERSON_FEATURES)} → {len(pruned)} person features")
            models = estimate_timing_models(
                out["trips"], out["tours"], out["persons"],
                feature_override=pruned,
            )
            save_models(models, path)
            _save_stage_coef_csv(models, 'stage4', OUT_DIR)

    logger.info(f"  Stage 4 models → {path}  (HTML report alongside)")
    _log_stage_summary('4', models)
    _inspect_model_parameters(models, 'stage4', OUT_DIR)
    _coef = pd.read_csv(os.path.join(OUT_DIR, 'stage4_coefficients.csv')) \
        if os.path.exists(os.path.join(OUT_DIR, 'stage4_coefficients.csv')) else None
    _write_stage_checkpoint_report(
        stage_tag='estimate_stage4',
        title='Estimate Stage 4 - Schedule Placement',
        summary={
            'n_models': len(models),
            'model_path': path,
            'coefficients_csv': os.path.join(OUT_DIR, 'stage4_coefficients.csv'),
        },
        table=_coef,
    )
    return models


@timer("Estimate Stage 5 — destination + mode choice")
def estimate_stage5(out, landuse):
    """Estimate Stage 5 models from HHTS only (no synth population)."""
    import openmatrix as omx
    from stage5_destination_choice import (
        _skim_flat, _skim_lookup, _prep_hhts_tours, _run_segment,
        _clear_sharrow_cache,
    )
    from read_skims import list_omx_matrices
    skims_path = PIPELINE_SKIMS_PATH
    list_omx_matrices(skims_path)
    if os.environ.get('STAGE5_CLEAR_SHARROW_CACHE', '0') == '1':
        _clear_sharrow_cache()

    skims      = omx.open_file(skims_path, 'r')
    skim_df    = _skim_flat(skims)
    sk_lookup  = _skim_lookup(skim_df)

    logger.info("  Preparing HHTS tours for Stage 5 estimation...")
    # Pass the HHTS tours table — it already has tour_mode from build_as_tours()
    # which calls _modal_mode() on the trip-level mode column (MAINMODE → DA/Walk/etc.)
    # out["trips"] does NOT have tour_mode, so passing it always defaults to DA.
    _hhts_tours_src = out["tours"]
    if "tour_mode" not in _hhts_tours_src.columns or _hhts_tours_src["tour_mode"].isna().all():
        # Fallback: derive dominant trip mode from out["trips"]
        _mode_col = next(
            (c for c in ["mode", "MAINMODE", "TRAVMODE", "LINKMODE"]
             if c in out["trips"].columns), None)
        if _mode_col:
            _dom = (out["trips"]
                    .groupby(["person_id", "tour_id"])[_mode_col]
                    .agg(lambda x: x.value_counts().idxmax() if len(x) else "DA")
                    .reset_index()
                    .rename(columns={_mode_col: "tour_mode"}))
            _hhts_tours_src = _hhts_tours_src.copy()
            if "tour_mode" in _hhts_tours_src.columns:
                _hhts_tours_src = _hhts_tours_src.drop(columns=["tour_mode"])
            _hhts_tours_src = _hhts_tours_src.merge(_dom, on=["person_id", "tour_id"], how="left")
            n_got = _hhts_tours_src["tour_mode"].notna().sum()
            logger.info(f"  tour_mode derived from trips['{_mode_col}']: {n_got}/{len(_hhts_tours_src)} tours")
    hhts_tours = _prep_hhts_tours(_hhts_tours_src, out["persons"])
    logger.info(f"  {len(hhts_tours)} HHTS tours available")

    # ── Warm-start from GA results or a previous stage5_models.pkl ──────────
    # If GA has been run, its best model coefficients give a much better
    # starting point than the hand-coded SEGMENT_COEFFICIENTS defaults.
    # This typically reduces estimation time and improves fit (fewer local optima).
    _warm_coefs: dict = {}   # {seg: {coef_name: value}}
    try:
        from stage5_destination_choice import SEGMENT_COEFFICIENTS as _SC
        import importlib
        # Priority 1: GA results pkl in the current run directory
        _ga_search_paths = [
            os.path.join(OUT_DIR, 'ga_results.pkl'),
            os.path.join(os.path.dirname(OUT_DIR), 'ga_staged', 'ga_results.pkl'),
            os.path.join(os.path.dirname(OUT_DIR), 'ga', 'ga_results.pkl'),
            'saved_data/ga_results.pkl',
        ]
        _ga_loaded = None
        for _gp in _ga_search_paths:
            if os.path.isfile(_gp):
                try:
                    with open(_gp, 'rb') as _f:
                        _ga_loaded = pickle.load(_f)
                    logger.info(f"  GA warm-start: loaded {_gp}")
                    break
                except Exception as _ge:
                    logger.debug(f"  GA pkl load failed ({_gp}): {_ge}")

        if _ga_loaded is not None and hasattr(_ga_loaded, 'results'):
            _s5_ga = _ga_loaded.results.get(5)
            if _s5_ga is not None and hasattr(_s5_ga, 'best_features'):
                _best_fit = float(getattr(_s5_ga, 'best_fitness', float('nan')))
                # GA can emit heavily-penalized sentinels (e.g. -1e9) when no
                # feasible feature set was found. Do not apply those overrides.
                if np.isfinite(_best_fit) and _best_fit <= -1.0e8:
                    logger.info(
                        "  GA warm-start stage 5 best_fitness indicates penalized/invalid "
                        f"solution ({_best_fit:.3f}); ignoring GA feature overrides."
                    )
                    _ga_mode_feats = None
                    _ga_dest_feats = None
                    _mode_feats = None
                    _dest_feats = None
                else:
                # GA result has feature lists; extract mode/dest overrides
                    _mode_feats = (_s5_ga.best_features.get('mode')
                                   if isinstance(_s5_ga.best_features, dict)
                                   else _s5_ga.best_features)
                    _dest_feats = (_s5_ga.best_features.get('dest')
                                   if isinstance(_s5_ga.best_features, dict)
                                   else None)
                    logger.info(
                        f"  GA warm-start: stage 5 best mode feats={_mode_feats} "
                        f"dest feats={_dest_feats}  fitness={_best_fit:.4f}"
                    )
                    # Store feature overrides; these are passed to _run_segment below
                    _ga_mode_feats = _mode_feats
                    _ga_dest_feats = _dest_feats
            else:
                _ga_mode_feats = None
                _ga_dest_feats = None
        else:
            _ga_mode_feats = None
            _ga_dest_feats = None

        # Priority 2: previous stage5_models.pkl — extract fitted coefficients
        # as warm starts even if no GA results are present.
        _prev_pkl_paths = [
            os.path.join(OUT_DIR, 'stage5_models.pkl'),
        ]
        for _pp in _prev_pkl_paths:
            if os.path.isfile(_pp) and os.path.getsize(_pp) > 1000:
                try:
                    from stage5_destination_choice import load_models as _s5_load
                    _prev = _s5_load(_pp)
                    if _prev:
                        for _seg, _res in _prev.items():
                            _coefs = {}
                            # Mode model coefficients
                            for _src in ('mode_coef', 'coef'):
                                _c = _res.get(_src, {})
                                if isinstance(_c, dict) and _c:
                                    _coefs.update(_c)
                                    break
                            # Dest model coefficients
                            _dc = _res.get('dest_coef', {})
                            if isinstance(_dc, dict):
                                _coefs.update(_dc)
                            if _coefs:
                                _warm_coefs[_seg] = _coefs
                        if _warm_coefs:
                            logger.info(
                                f"  Warm-start coefficients loaded from previous run: "
                                f"{_pp}  ({', '.join(f'{s}:{len(c)} coefs' for s, c in _warm_coefs.items())})"
                            )
                        break
                except Exception as _we:
                    logger.debug(f"  Warm-start pkl load failed ({_pp}): {_we}")
    except Exception as _ws_e:
        logger.debug(f"  Warm-start setup failed: {_ws_e}")
        _ga_mode_feats = None
        _ga_dest_feats = None

    segments = ['work', 'edu', 'nonmand']
    models = {}
    for seg in segments:
        logger.info(f"  ▶  Estimating segment: {seg}")
        _seg_warm = _warm_coefs.get(seg)
        if _seg_warm:
            logger.info(f"     warm-start: {len(_seg_warm)} coefficients from previous run")
        try:
            result = _run_segment(
                seg, hhts_tours, out["persons"], landuse,
                skims, skim_df, seed=42,
                mode_feature_override=_ga_mode_feats,
                dest_feature_override=_ga_dest_feats,
                coef_warm_start=_seg_warm,     # NEW: inject previous coefficients
            )
            if result:
                models[seg] = result
                logger.info(f"  ✓  {seg}: mode LL={result.get('mode_ll','?'):.1f}  "
                            f"dest LL={result.get('dest_ll','?'):.1f}  "
                            f"n_tours={result.get('n_tours','?')}")
            else:
                logger.warning(f"  ✗  {seg}: _run_segment returned empty dict "
                               "(fewer than MIN_TOURS tours in this segment?)")
        except Exception as e:
            import traceback as _tb
            logger.warning(f"  ✗  {seg} estimation failed: {e}")
            logger.warning(f"  {seg} traceback:\n{_tb.format_exc()}")

    # Save models (cloudpickle → dill → pickle fallback)
    from stage5_destination_choice import save_models as s5_save
    path = os.path.join(OUT_DIR, 'stage5_models.pkl')
    s5_save(models, path)

    # Also save the HTML coefficient report
    from stage5_destination_choice import _save_coefficients
    coef_path = path.replace('.pkl', '_coefficients.csv')
    _save_coefficients(models, coef_path)

    if _auto_prune_enabled('stage5'):
        mode_override, dest_override, meta = _derive_stage5_pruned_overrides(
            coef_csv_path=coef_path,
            p_thresh=float(os.environ.get('STAGE5_PRUNE_P_THRESHOLD', '0.10')),
            t_thresh=float(os.environ.get('STAGE5_PRUNE_T_THRESHOLD', '1.96')),
        )
        if meta.get('changed') and (mode_override or dest_override):
            logger.info(
                "  Stage 5 prune+refit: "
                f"mode_vars={len(mode_override)} dest_vars={len(dest_override)} "
                f"(critical DIST/CRASH/size/logsum terms protected)"
            )
            _warm_refit = _extract_stage5_warm_coefs(models)
            _pruned_models = {}
            for seg in segments:
                try:
                    _r = _run_segment(
                        seg, hhts_tours, out["persons"], landuse,
                        skims, skim_df, seed=42,
                        mode_feature_override=mode_override,
                        dest_feature_override=dest_override,
                        coef_warm_start=_warm_refit.get(seg),
                    )
                    if _r:
                        _pruned_models[seg] = _r
                except Exception as _pr_err:
                    logger.warning(f"  Stage5 prune-refit failed for {seg}: {_pr_err}")

            if _pruned_models:
                models = _pruned_models
                s5_save(models, path)
                _save_coefficients(models, coef_path)
                pd.DataFrame([
                    {'group': 'mode_override', 'variable': v} for v in mode_override
                ] + [
                    {'group': 'dest_override', 'variable': v} for v in dest_override
                ]).to_csv(os.path.join(OUT_DIR, 'stage5_pruned_overrides.csv'), index=False)
                logger.info("  Stage 5 prune+refit models saved (stage5_pruned_overrides.csv written)")
            else:
                logger.warning("  Stage 5 prune+refit produced no valid segment models; keeping original estimates")

    logger.info(f"  Stage 5 models → {path}  (HTML report alongside)")
    _log_stage_summary('5', models)

    # ActivitySim-style location choice summary (coefficients + validation)
    try:
        from stage5_destination_choice import save_location_choice_summary
        _obs_tours = out.get('tours', pd.DataFrame()) if isinstance(out, dict) else pd.DataFrame()
        save_location_choice_summary(
            models=models,
            obs_tours=_obs_tours,
            landuse=landuse,
            save_dir=OUT_DIR,
            persons=out.get('persons') if isinstance(out, dict) else None,
        )
    except Exception as _loc_e:
        logger.warning(f"  Location choice summary (post-estimation) failed: {_loc_e}")

    # ── P-value / zero-coefficient diagnostics ──────────────────────────────
    _inspect_model_parameters(models, 'stage5_tour', OUT_DIR)

    _coef = pd.read_csv(os.path.join(OUT_DIR, 'stage5_models_coefficients.csv')) \
        if os.path.exists(os.path.join(OUT_DIR, 'stage5_models_coefficients.csv')) else None
    _write_stage_checkpoint_report(
        stage_tag='estimate_stage5',
        title='Estimate Stage 5 - Mode and Destination Choice',
        summary={
            'n_segments': len(models),
            'model_path': path,
            'coefficients_csv': os.path.join(OUT_DIR, 'stage5_models_coefficients.csv'),
        },
        table=_coef,
    )

    return models, skims, sk_lookup


# ─────────────────────────────────────────────────────────────────────────────
# TRIP-LEVEL MODE + DESTINATION MODELS  (Stage 5t)
# ─────────────────────────────────────────────────────────────────────────────

@timer("Estimate Stage 5t — trip-level mode + destination choice")
def estimate_stage5_trips(out, landuse, skims=None, skim_df=None):
    """
    Estimate trip-level mode choice (NestedLogitModeModel per purpose) and
    trip-level destination choice (Larch MNL per segment) on HHTS trip legs.

    Unlike the tour-level Stage 5 models, every trip leg is one observation
    here.  This gives more data for estimation and captures within-tour mode
    variability (e.g. walk on the return leg even when the outbound was car).

    Outputs
    -------
    stage5_trip_mode_models.pkl       — nested logit mode models by purpose
    stage5_trip_dest_models.pkl       — destination MNL models by segment
    stage5_trip_models_coefficients.csv   — combined coefficient table
    stage5_trip_models_coefficients.html  — HTML report (mode + dest)
    stage5_trip_models_coefficients_mode.html
    stage5_trip_models_coefficients_dest.html
    """
    import openmatrix as omx
    from stage6_mode_choice import (
        estimate_mode_models as _est_mode,
        save_models as _s6_save,
        _save_s6_coefficients,
        _write_s6_html_report,
    )
    from stage5_destination_choice import (
        _prep_hhts_tours,
        _run_segment,
        save_models as _s5_save,
        _save_coefficients as _s5_save_coeff,
    )

    trips   = out.get('trips',   pd.DataFrame())
    persons = out.get('persons', pd.DataFrame())

    # ── Trip-level mode choice ──────────────────────────────────────────────
    logger.info("  [5t] Estimating trip-level mode choice models...")
    trip_mode_models = {}
    try:
        trip_mode_models = _est_mode(trips, persons, landuse)
        mode_pkl = os.path.join(OUT_DIR, 'stage5_trip_mode_models.pkl')
        _s6_save(trip_mode_models, mode_pkl)

        # Coefficient CSV + HTML (reuse stage6_mode_choice functions)
        mode_coef_csv  = os.path.join(OUT_DIR, 'stage5_trip_mode_coefficients.csv')
        mode_coef_html = os.path.join(OUT_DIR, 'stage5_trip_mode_coefficients.html')
        _save_s6_coefficients(trip_mode_models, mode_coef_csv)
        # Save full coefficient HTML report so analysts can inspect the model,
        # not just the mode-share validation table.
        try:
            _write_s6_html_report(trip_mode_models, mode_coef_html)
            logger.info(f"  [5t] Trip mode coefficient HTML → {mode_coef_html}")
        except Exception as _html_me:
            logger.debug(f"  [5t] Trip mode HTML report failed: {_html_me}")

        logger.info(f"  [5t] Trip mode models: {list(trip_mode_models.keys())} → {mode_pkl}")

        # Validation: predicted vs observed mode shares (table + bar charts)
        _s5t_mode_validation(trip_mode_models, trips, persons, OUT_DIR)
        _s5t_mode_validation_plots(trip_mode_models, trips, persons, OUT_DIR)

    except Exception as _me:
        import traceback as _tb3
        logger.warning(f"  [5t] Trip mode estimation failed: {_me}")
        logger.debug(_tb3.format_exc())

    # ── Trip-level destination choice ───────────────────────────────────────
    # Treat each trip leg as an independent destination choice.  Map trip
    # purposes to the three tour segments (work / edu / nonmand) then run
    # _run_segment on the trip table so the same Larch MNL pipeline applies.
    logger.info("  [5t] Estimating trip-level destination choice models...")
    trip_dest_models = {}
    try:
        if skim_df is None and skims is not None:
            from stage5_destination_choice import _skim_flat as _sf
            skim_df = _sf(skims)

        # Convert trips to a tours-like table: each trip leg is one observation
        # with its own OTAZ, DTAZ, and a segment label.
        _trip_like = _trips_to_seg_table(trips, persons)
        if not _trip_like.empty:
            for seg in ['work', 'edu', 'nonmand']:
                try:
                    result = _run_segment(
                        seg, _trip_like, persons, landuse,
                        skims, skim_df, seed=42,
                    )
                    if result:
                        trip_dest_models[seg] = result
                        logger.info(
                            f"  [5t] dest/{seg}: mode LL={result.get('mode_ll','?'):.1f}  "
                            f"dest LL={result.get('dest_ll','?'):.1f}  "
                            f"n={result.get('n_tours','?')}"
                        )
                    else:
                        logger.warning(f"  [5t] dest/{seg}: _run_segment returned empty")
                except Exception as _se:
                    import traceback as _tb4
                    logger.warning(f"  [5t] dest/{seg} failed: {_se}")
                    logger.debug(_tb4.format_exc())

        if trip_dest_models:
            dest_pkl  = os.path.join(OUT_DIR, 'stage5_trip_dest_models.pkl')
            _s5_save(trip_dest_models, dest_pkl)
            dest_csv  = os.path.join(OUT_DIR, 'stage5_trip_dest_coefficients.csv')
            _s5_save_coeff(trip_dest_models, dest_csv)
            logger.info(f"  [5t] Trip dest models → {dest_pkl}")
            logger.info(f"  [5t] Trip dest HTML   → {dest_csv.replace('.csv', '.html')}")
            logger.info(f"  [5t] Trip dest mode-only HTML → {dest_csv.replace('.csv', '_mode.html')}")
            logger.info(f"  [5t] Trip dest dest-only HTML → {dest_csv.replace('.csv', '_dest.html')}")
            # P-value diagnostics for trip dest models
            _inspect_model_parameters(trip_dest_models, 'stage5_trip_dest', OUT_DIR)

        # ── Per-purpose destination models for non-mandatory sub-purposes ─────
        # The three-segment model (work/edu/nonmand) lumps all non-mandatory
        # trips together.  Here we additionally fit a destination model per
        # individual non-mandatory purpose (eat, shopping, othdiscr, etc.)
        # where enough observations exist (min 50).
        # Key: pre-filter _trip_like to one purpose, rename primary_purpose
        # to 'nonmand' so _filter_seg passes it through unaltered.
        NONMAND_PURPOSES = ['social', 'escort', 'shopping', 'othmaint', 'othdiscr', 'eat']
        trip_dest_models_per_purpose = {}
        if not _trip_like.empty:
            purp_col_tl = next(
                (c for c in ['primary_purpose', 'main_purpose', 'purpose']
                 if c in _trip_like.columns), None
            )
            if purp_col_tl:
                for _purp in NONMAND_PURPOSES:
                    try:
                        _sub = _trip_like[
                            _trip_like[purp_col_tl].astype(str).str.lower() == _purp
                        ].copy()
                        if len(_sub) < 50:
                            logger.info(f"  [5t] dest/{_purp}: only {len(_sub)} obs — skipping")
                            continue
                        # Rename to 'nonmand' so _filter_seg lets it through
                        _sub[purp_col_tl] = 'nonmand'
                        result = _run_segment(
                            'nonmand', _sub, persons, landuse,
                            skims, skim_df, seed=42,
                        )
                        if result:
                            result['purpose_label'] = _purp  # tag for reporting
                            trip_dest_models_per_purpose[_purp] = result
                            logger.info(
                                f"  [5t] dest/{_purp}: dest LL={result.get('dest_ll','?')}  "
                                f"n={result.get('n_tours','?')}"
                            )
                    except Exception as _pe:
                        logger.debug(f"  [5t] dest/{_purp} skipped: {_pe}")

        if trip_dest_models_per_purpose:
            pdest_pkl = os.path.join(OUT_DIR, 'stage5_trip_dest_per_purpose_models.pkl')
            _s5_save(trip_dest_models_per_purpose, pdest_pkl)
            pdest_csv = os.path.join(OUT_DIR, 'stage5_trip_dest_per_purpose_coefficients.csv')
            _s5_save_coeff(trip_dest_models_per_purpose, pdest_csv)
            logger.info(
                f"  [5t] Per-purpose dest models: {list(trip_dest_models_per_purpose.keys())} → {pdest_pkl}"
            )
            logger.info(f"  [5t] Per-purpose dest HTML (combined) → {pdest_csv.replace('.csv', '.html')}")
            logger.info(f"  [5t] Per-purpose dest-only HTML       → {pdest_csv.replace('.csv', '_dest.html')}")

        # ── Combined HTML report: mode + dest (3-segment + per-purpose) ───────
        try:
            _all_dest = {**trip_dest_models, **trip_dest_models_per_purpose}
            _write_trip_models_html_report(trip_mode_models, _all_dest, OUT_DIR)
        except Exception as _html_e:
            logger.debug(f"  [5t] HTML report failed: {_html_e}")

    except Exception as _de:
        import traceback as _tb5
        logger.warning(f"  [5t] Trip destination estimation failed: {_de}")
        logger.debug(_tb5.format_exc())

    return trip_mode_models, trip_dest_models


def _write_trip_models_html_report(
    mode_models: dict,
    dest_models: dict,
    save_dir: str,
) -> None:
    """
    Write a comprehensive HTML report for trip-level mode + destination models.

    One section per purpose/segment.  Non-mandatory purposes (eat, shopping,
    othdiscr, social, escort, othmaint) each get their own section so the
    reader can see destination / mode choice behaviour per activity type,
    rather than having them all lumped under 'nonmand'.

    Outputs
    -------
    {save_dir}/stage5t_models_report.html     — combined report
    {save_dir}/stage5t_mode_{purpose}.html    — per-purpose mode model
    
    FIX (2026-05-15): Added diagnostic logging when models dict is empty
    to catch cases where trip-level estimation produced no models.
    """
    import math as _math

    def _sig(t):
        if np.isnan(t):  return ''
        if abs(t) >= 3.0: return '***'
        if abs(t) >= 2.0: return '**'
        if abs(t) >= 1.65: return '*'
        return ''

    def _coef_table_html(coef_dict, pf):
        """Build an HTML coefficient table from a coefficient dict + pf DataFrame."""
        rows_h = ("<table><tr><th>Parameter</th><th>Value</th>"
                  "<th>Std&nbsp;Err</th><th>t-stat</th><th>Sig</th></tr>")
        params = list(pf.index) if pf is not None and len(pf) > 0 else list(coef_dict.keys())
        for p in params:
            val = coef_dict.get(str(p), float('nan'))
            se  = tstat = float('nan')
            if pf is not None and p in pf.index:
                row = pf.loc[p]
                for c in ('std_err', 'std_err_opg', 'se'):
                    if c in row.index and np.isfinite(float(row[c])):
                        se = float(row[c]); break
                for c in ('t_stat', 't_stat_opg', 'tstat'):
                    if c in row.index and np.isfinite(float(row[c])):
                        tstat = float(row[c]); break
            if np.isnan(tstat) and np.isfinite(se) and se > 0:
                tstat = float(val) / se
            _style = " style='color:#c0392b;font-weight:bold'" if abs(tstat) < 1.65 and np.isfinite(tstat) else ""
            rows_h += (
                f"<tr><td{_style}>{p}</td>"
                f"<td style='text-align:right'>{val:.4f}</td>"
                f"<td style='text-align:right'>{se:.4f if np.isfinite(se) else '—'}</td>"
                f"<td style='text-align:right'>{tstat:.2f if np.isfinite(tstat) else '—'}</td>"
                f"<td style='text-align:center'>{_sig(tstat)}</td></tr>"
            )
        return rows_h + "</table>"

    def _mode_section_html(purpose, model):
        """Build one purpose section for trip-level mode models."""
        ll   = getattr(model, 'll_final_', float('nan'))
        rho2 = getattr(model, 'rho2_', float('nan'))
        n    = getattr(model, 'n_obs_', '?')
        conv = getattr(model, 'converged_', False)
        coef = model.coef_ if hasattr(model, 'coef_') else []
        pnames = getattr(model, 'PARAM_NAMES', [f'p{i}' for i in range(len(coef))])
        se    = getattr(model, 'se_',    np.full(len(coef), np.nan))
        tstat = getattr(model, 'tstat_', np.full(len(coef), np.nan))
        rows_h = ""
        for pn, c, s, t in zip(pnames, coef, se, tstat):
            if np.isnan(t) and np.isfinite(s) and s > 0:
                t = float(c) / s
            _style = " style='color:#c0392b;font-weight:bold'" if np.isfinite(t) and abs(t) < 1.65 else ""
            rows_h += (
                f"<tr><td{_style}>{pn}</td>"
                f"<td style='text-align:right'>{c:.4f}</td>"
                f"<td style='text-align:right'>{s:.4f if np.isfinite(s) else '—'}</td>"
                f"<td style='text-align:right'>{t:.2f if np.isfinite(t) else '—'}</td>"
                f"<td style='text-align:center'>{_sig(t)}</td></tr>"
            )
        return (
            f"<div class='model-section'>"
            f"<h2>Mode Choice — {purpose.capitalize()}</h2>"
            f"<div class='stats'>LL={ll:.1f if np.isfinite(ll) else '—'} &nbsp; "
            f"ρ²={rho2:.3f if np.isfinite(rho2) else '—'} &nbsp; "
            f"N={n} &nbsp; {'✓ converged' if conv else '✗ not converged'}</div>"
            f"<table><tr><th>Parameter</th><th>Value</th><th>Std&nbsp;Err</th>"
            f"<th>t-stat</th><th>Sig</th></tr>{rows_h}</table>"
            f"<p class='sig-note'>*** |t|≥3 &nbsp; ** |t|≥2 &nbsp; * |t|≥1.65 &nbsp; "
            f"<span style='color:#c0392b'>red = |t|&lt;1.65 (not significant)</span></p>"
            f"</div>"
        )

    def _dest_section_html(seg, res):
        """Build one segment section for trip-level destination models."""
        label = res.get('purpose_label', seg)
        n     = res.get('n_tours', '?')
        dest_ll = res.get('dest_ll', float('nan'))
        mode_ll = res.get('mode_ll', float('nan'))
        dest_mdl = res.get('dest_model')
        mode_mdl = res.get('mode_model')
        sections = []
        for mtype, mdl, ll_val in [('Mode', mode_mdl, mode_ll), ('Destination', dest_mdl, dest_ll)]:
            if mdl is None:
                continue
            try:
                from stage5_destination_choice import _coef_dict_from_model as _cdfm
                coef_dict = _cdfm(mdl)
                pf = None
                if hasattr(mdl, '_final_pf') and isinstance(mdl._final_pf, pd.DataFrame):
                    pf = mdl._final_pf
                else:
                    try:
                        pf = mdl.pf
                    except Exception:
                        pass
                sections.append(
                    f"<h3>{mtype} model &nbsp; <small>LL={ll_val:.1f if np.isfinite(float(str(ll_val).replace('?','nan'))) else '—'}  N={n}</small></h3>"
                    + _coef_table_html(coef_dict, pf)
                )
            except Exception as _ce:
                sections.append(f"<h3>{mtype} model — could not render: {_ce}</h3>")
        return (
            f"<div class='model-section'>"
            f"<h2>Destination Choice — {label.capitalize()}</h2>"
            + '\n'.join(sections) +
            f"<p class='sig-note'>*** |t|≥3 &nbsp; ** |t|≥2 &nbsp; * |t|≥1.65 &nbsp; "
            f"<span style='color:#c0392b'>red = |t|&lt;1.65 (not significant)</span></p>"
            f"</div>"
        )

    CSS = (
        "body{font-family:Arial,sans-serif;margin:40px;color:#222;max-width:1100px}"
        "h1{border-bottom:2px solid #1565c0;padding-bottom:8px;color:#1a3a5c}"
        "h2{color:#1565c0;margin-top:32px}"
        "h3{color:#2c5f8a;margin-top:18px;font-size:1em}"
        ".model-section{margin-bottom:36px;border:1px solid #ccc;border-radius:6px;padding:20px}"
        ".stats{background:#e3f2fd;padding:8px 12px;border-radius:4px;font-size:.95em;margin-bottom:12px}"
        "table{border-collapse:collapse;width:100%;margin-bottom:12px}"
        "th{background:#1565c0;color:white;padding:8px 12px;text-align:left;font-size:.9em}"
        "td{padding:5px 10px;border-bottom:1px solid #eee;font-size:.88em}"
        "tr:hover td{background:#f5f5f5}"
        ".sig-note{font-size:.8em;color:#555;margin-top:6px}"
        ".warning{background:#fff3cd;padding:12px;border:1px solid #ffc107;border-radius:4px;margin-bottom:16px}"
    )

    # Build combined report body
    mode_sections = [_mode_section_html(p, m) for p, m in mode_models.items() if m is not None]
    dest_sections = [_dest_section_html(s, r) for s, r in dest_models.items() if r is not None]

    # FIX (2026-05-15): Log and handle case where no models were successfully estimated
    if not mode_sections and not dest_sections:
        logger.warning(
            f"  [5t] HTML report: no valid models found. "
            f"mode_models keys={list(mode_models.keys())} (None count={sum(1 for m in mode_models.values() if m is None)}); "
            f"dest_models keys={list(dest_models.keys())} (None count={sum(1 for d in dest_models.values() if d is None)})"
        )
        # Create a placeholder warning message so the report is not completely empty
        warning_msg = (
            "<div class='warning'><strong>⚠ No trip-level models estimated</strong><p>"
            "All trip-level mode and destination models failed to estimate. "
            "This may indicate insufficient data or model specification issues. "
            "Check the run logs for detailed error messages.</p></div>"
        )
    else:
        warning_msg = ""

    body = (
        warning_msg
        + "<h2 style='margin-top:0;color:#1a3a5c'>Trip-Level Mode Choice Models</h2>"
        + ('\n'.join(mode_sections) if mode_sections else "<p><em>No mode models available.</em></p>")
        + "<hr><h2 style='color:#1a3a5c;margin-top:32px'>Trip-Level Destination Choice Models</h2>"
        + ('\n'.join(dest_sections) if dest_sections else "<p><em>No destination models available.</em></p>")
    )

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>Stage 5t — Trip-Level Mode & Destination Models</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>Stage 5t — Trip-Level Mode &amp; Destination Models</h1>"
        "<p>Each non-mandatory activity purpose (eat, shopping, othdiscr, social, "
        "escort, othmaint) is reported separately so that behavioural differences "
        "across activity types are visible. Red parameter names have |t-stat| &lt; 1.65 "
        "(not significant at 10% level).</p>"
        f"{body}"
        "</body></html>"
    )

    report_path = os.path.join(save_dir, 'stage5t_models_report.html')
    os.makedirs(save_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as _f:
        _f.write(html)
    logger.info(f"  [5t] Combined HTML report → {report_path}")

    # Write per-purpose mode model HTMLs
    for purpose, model in mode_models.items():
        if model is None:
            continue
        try:
            one_section = _mode_section_html(purpose, model)
            one_html = (
                "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
                f"<title>Stage 5t Mode — {purpose.capitalize()}</title>"
                f"<style>{CSS}</style></head><body>"
                f"<h1>Trip-Level Mode Choice: {purpose.capitalize()}</h1>"
                f"{one_section}</body></html>"
            )
            pp = os.path.join(save_dir, f'stage5t_mode_{purpose}.html')
            with open(pp, 'w', encoding='utf-8') as _f:
                _f.write(one_html)
        except Exception as _pe:
            logger.debug(f"  [5t] per-purpose HTML for {purpose} failed: {_pe}")

    logger.info(f"  [5t] Per-purpose mode HTML files → {save_dir}/stage5t_mode_{{purpose}}.html")

    # Write per-purpose destination model HTMLs
    for seg, res in dest_models.items():
        if res is None:
            continue
        try:
            one_dest_section = _dest_section_html(seg, res)
            label = res.get('purpose_label', seg)
            one_html = (
                "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
                f"<title>Stage 5t Destination \u2014 {label.capitalize()}</title>"
                f"<style>{CSS}</style></head><body>"
                f"<h1>Trip-Level Destination Choice: {label.capitalize()}</h1>"
                f"{one_dest_section}</body></html>"
            )
            pp = os.path.join(save_dir, f'stage5t_dest_{label}.html')
            with open(pp, 'w', encoding='utf-8') as _f:
                _f.write(one_html)
        except Exception as _dpe:
            logger.debug(f"  [5t] per-purpose dest HTML for {seg} failed: {_dpe}")

    logger.info(f"  [5t] Per-purpose dest HTML files → {save_dir}/stage5t_dest_{{purpose}}.html")


def _trips_to_seg_table(trips: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    """
    Convert HHTS trip legs into a tours-like DataFrame for _run_segment.

    Each trip leg becomes one observation with:
      - OTAZ, DTAZ from the trip
      - tour_mode from the trip mode column
      - primary_purpose derived from trip purpose
      - person attributes merged from persons
    """
    from stage5_destination_choice import _prep_hhts_tours, _purpose_to_seg
    if trips.empty:
        return pd.DataFrame()

    t = trips.copy()

    # Standardise mode column → tour_mode integer
    mode_col = next((c for c in ['mode', 'MAINMODE', 'TRAVMODE', 'LINKMODE']
                     if c in t.columns), None)
    from stage5_destination_choice import MODE_MAP
    if mode_col and 'tour_mode' not in t.columns:
        t['tour_mode'] = t[mode_col].map(MODE_MAP).fillna(2).astype(int)
    elif 'tour_mode' not in t.columns:
        t['tour_mode'] = 1

    # Standardise purpose → primary_purpose (for _filter_seg compatibility)
    purp_col = next((c for c in ['purpose', 'main_purpose', 'TRIPTYPE', 'trip_purpose']
                     if c in t.columns), None)
    if purp_col and 'primary_purpose' not in t.columns:
        t['primary_purpose'] = t[purp_col].astype(str).str.lower().str.strip()

    # Map trip purposes to tour segments
    _purp_remap = {
        'work':      'work',   'work_related': 'work',
        'edu':       'edu',    'school': 'edu',   'education': 'edu',
        'social':    'nonmand', 'escort': 'nonmand', 'shopping': 'nonmand',
        'othmaint':  'nonmand', 'othdiscr': 'nonmand', 'eat': 'nonmand',
        'recreation':'nonmand', 'personal': 'nonmand', 'other': 'nonmand',
    }
    if 'primary_purpose' in t.columns:
        t['primary_purpose'] = t['primary_purpose'].map(
            lambda p: _purp_remap.get(str(p).lower(), 'nonmand')
        )

    # Ensure OTAZ / DTAZ
    for zcol, alts in [('OTAZ', ['ORIGZONE', 'orig_taz', 'ORDER']),
                       ('DTAZ', ['DESTZONE', 'dest_taz'])]:
        if zcol not in t.columns:
            for a in alts:
                if a in t.columns:
                    t[zcol] = t[a]
                    break

    for zcol in ['OTAZ', 'DTAZ']:
        if zcol in t.columns:
            t[zcol] = pd.to_numeric(t[zcol], errors='coerce').fillna(0).astype(int)

    # Merge person attributes if not already present
    if persons is not None and not persons.empty:
        pid_col = next((c for c in ['person_id', 'PERSONID', 'PERSNUM']
                        if c in persons.columns), None)
        t_pid   = next((c for c in ['person_id', 'PERSONID', 'PERSNUM']
                        if c in t.columns), None)
        if pid_col and t_pid:
            _person_cols = [pid_col] + [
                c for c in persons.columns
                if c != pid_col and c not in t.columns
                and c in ['ptype', 'WORKSTAT', 'STUDYING', 'income_hi', 'income_lo',
                           'vehrd', 'has_vehicle', 'has_licence', 'indp', 'indp_group',
                           'income_band', 'age_group', 'veh_bin', 'age']
            ]
            t = t.merge(persons[_person_cols], left_on=t_pid,
                        right_on=pid_col, how='left', suffixes=('', '_person'))

    # _run_segment needs a 'weight' column (use 1.0 for trips — not weighted)
    if 'weight' not in t.columns:
        t['weight'] = 1.0

    return t.reset_index(drop=True)


def _s5t_mode_validation(models: dict, trips: pd.DataFrame,
                         persons: pd.DataFrame, save_dir: str) -> None:
    """
    Compare observed vs predicted mode shares for trip-level models.
    Saves a CSV + HTML to save_dir.
    """
    try:
        from stage6_mode_choice import prepare_mode_data, PURPOSES
        mode_data = prepare_mode_data(trips, persons)
        rows = []
        for purpose, model in models.items():
            subset = mode_data[mode_data['purpose'] == purpose]
            if subset.empty:
                continue
            obs_share = (subset['chosen_mode'].value_counts(normalize=True)
                         .rename('obs_share').reset_index()
                         .rename(columns={'index': 'mode', 'chosen_mode': 'mode'}))
            # predicted shares from frozen probabilities or model
            if model.frozen_probs_ is not None and not model.frozen_probs_.empty:
                fp = model.frozen_probs_.copy()
                pred_cols = [c for c in fp.columns if c.startswith('p_')]
                if pred_cols:
                    pred_shares = fp[pred_cols].mean().to_dict()
                    for mc, ps in pred_shares.items():
                        mode_name = mc.replace('p_', '')
                        obs_row = obs_share[obs_share['mode'] == mode_name]
                        obs_val = float(obs_row['obs_share'].iloc[0]) if len(obs_row) else float('nan')
                        rows.append({
                            'purpose': purpose,
                            'mode': mode_name,
                            'obs_share': round(obs_val * 100, 2),
                            'pred_share': round(float(ps) * 100, 2),
                            'n_obs': len(subset),
                        })

        if rows:
            df_val = pd.DataFrame(rows)
            csv_path  = os.path.join(save_dir, 'stage5_trip_mode_validation.csv')
            html_path = os.path.join(save_dir, 'stage5_trip_mode_validation.html')
            df_val.to_csv(csv_path, index=False)

            # Build simple HTML table
            _html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Stage 5t mode validation</title>"
                "<style>body{font-family:Arial,sans-serif;margin:20px;font-size:13px}"
                "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
                "th{background:#2c3e50;color:white;padding:6px 10px}"
                "td{border:1px solid #ddd;padding:5px 10px}"
                "tr:nth-child(even){background:#f9f9f9}</style></head><body>"
                "<h1>Stage 5t — Trip-level mode share validation (HHTS)</h1>"
                "<table><tr><th>Purpose</th><th>Mode</th>"
                "<th>Observed %</th><th>Predicted %</th><th>N obs</th></tr>"
            )
            for _, r in df_val.iterrows():
                _html += (
                    f"<tr><td>{r['purpose']}</td><td>{r['mode']}</td>"
                    f"<td style='text-align:right'>{r['obs_share']:.1f}</td>"
                    f"<td style='text-align:right'>{r['pred_share']:.1f}</td>"
                    f"<td style='text-align:right'>{r['n_obs']}</td></tr>"
                )
            _html += "</table></body></html>"
            with open(html_path, 'w', encoding='utf-8') as _hf:
                _hf.write(_html)
            logger.info(f"  [5t] Mode share validation → {csv_path}")
    except Exception as _e:
        logger.debug(f"  [5t] Mode validation failed: {_e}")


def _s5t_mode_validation_plots(models: dict, trips: pd.DataFrame,
                                persons: pd.DataFrame, save_dir: str) -> None:
    """
    Bar-chart visualisation of observed vs predicted trip-level mode shares,
    one panel per trip purpose, saved to save_dir/plots/stage5t_mode_shares.png.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import math as _math
        from stage6_mode_choice import prepare_mode_data, PURPOSES

        MODE_COLORS = {'DA': '#3498db', 'SR': '#e74c3c', 'Walk': '#2ecc71',
                       'Bike': '#f39c12', 'Transit': '#9b59b6'}
        MODE_NAMES  = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}

        mode_data = prepare_mode_data(trips, persons)
        plot_data  = {}   # {purpose: {mode: (obs, pred)}}

        for purpose, model in models.items():
            subset = mode_data[mode_data['purpose'] == purpose]
            if subset.empty:
                continue
            obs_raw = subset['chosen_mode'].map(MODE_NAMES)
            obs_counts = obs_raw.value_counts(normalize=True)

            pred_shares = {}
            if model.frozen_probs_ is not None and not model.frozen_probs_.empty:
                fp = model.frozen_probs_.copy()
                pred_cols = [c for c in fp.columns if c.startswith('p_')]
                if pred_cols:
                    for pc in pred_cols:
                        mode_name = pc.replace('p_', '')
                        pred_shares[mode_name] = float(fp[pc].mean())

            if pred_shares:
                all_modes = sorted(set(list(obs_counts.index) + list(pred_shares.keys())))
                plot_data[purpose] = {
                    m: (float(obs_counts.get(m, 0.0)) * 100,
                        pred_shares.get(m, 0.0) * 100)
                    for m in all_modes
                }

        if not plot_data:
            logger.debug("  [5t] No plot data for mode share bar charts")
            return

        purposes = list(plot_data.keys())
        n    = len(purposes)
        cols = min(3, n)
        rows = _math.ceil(n / max(cols, 1))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows),
                                 facecolor='white', squeeze=False)
        fig.suptitle('Stage 5t — Trip-level Mode Shares: Observed vs Predicted',
                     fontsize=12, fontweight='bold')
        axes_flat = axes.flatten()

        for i, purp in enumerate(purposes):
            ax      = axes_flat[i]
            d       = plot_data[purp]
            modes   = list(d.keys())
            obs_v   = [d[m][0] for m in modes]
            pred_v  = [d[m][1] for m in modes]
            x       = np.arange(len(modes))
            w       = 0.35
            bars_o  = ax.bar(x - w/2, obs_v,  width=w, label='Observed',
                             color='#2c3e50', alpha=0.8)
            bars_p  = ax.bar(x + w/2, pred_v, width=w, label='Predicted',
                             color='#e74c3c', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(modes, fontsize=9)
            ax.set_ylabel('Share (%)', fontsize=9)
            ax.set_title(purp.capitalize(), fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_ylim(0, max(max(obs_v + pred_v, default=10) * 1.2, 10))
            # Value labels
            for b in list(bars_o) + list(bars_p):
                h = b.get_height()
                if h > 0.5:
                    ax.text(b.get_x() + b.get_width() / 2, h + 0.5,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=7)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plot_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, 'stage5t_mode_shares.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  [5t] Mode share bar charts → {out_path}")
    except Exception as _e:
        import traceback as _tb_ms
        logger.debug(f"  [5t] Mode share bar charts failed: {_e}\n{_tb_ms.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER DIAGNOSTICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _inspect_model_parameters(models: dict, label: str, save_dir: str) -> None:
    """
    Inspect estimated models for:
      - Zero / near-zero coefficients  (|value| < 1e-4)
      - Insignificant parameters       (|t-stat| < 1.96)
      - High p-values                  (p > 0.10)
      - Wrong behavioural signs

    Logs warnings and writes a diagnostic HTML to save_dir.
    Also calls _check_behavioural_signs from stage5_destination_choice for
    structural sign checks against expected economic theory.
    """
    try:
        from stage5_destination_choice import _check_behavioural_signs, _save_coefficients
    except ImportError:
        return

    SEP = "─" * 70
    logger.info(SEP)
    logger.info(f"  PARAMETER DIAGNOSTICS — {label}")
    logger.info(SEP)

    # ── Behavioural sign checks ──────────────────────────────────────────
    sign_results = []
    try:
        sign_results = _check_behavioural_signs(models)
        failures = [r for r in sign_results if r.get('pass') is False]
        if failures:
            logger.warning(f"  {len(failures)} WRONG-SIGN coefficient(s):")
            for r in failures:
                logger.warning(
                    f"    [{r['segment']}/{r['model']}] {r['parameter']}={r['value']:.4f} "
                    f"(expected {r['expected']}) — {r['note']}"
                )
        else:
            logger.info("  All checked coefficients have correct behavioural signs.")
    except Exception as _sc_e:
        logger.debug(f"  _check_behavioural_signs failed: {_sc_e}")

    # ── Zero / near-zero / insignificant coefficients ────────────────────
    diag_rows = []
    for seg, res in models.items():
        if not isinstance(res, dict):
            continue
        for mtype, mdl_key in [('mode', 'mode_model'), ('dest', 'dest_model')]:
            mdl = res.get(mdl_key)
            if mdl is None:
                continue
            try:
                pf = None
                if hasattr(mdl, '_final_pf') and isinstance(mdl._final_pf, pd.DataFrame):
                    pf = mdl._final_pf
                else:
                    try:
                        pf = mdl.pf
                    except AttributeError:
                        pass

                from stage5_destination_choice import _coef_dict_from_model
                coef = _coef_dict_from_model(mdl)
                param_list = list(pf.index) if pf is not None and len(pf) > 0 else list(coef.keys())

                for pname in param_list:
                    val = coef.get(str(pname), float('nan'))
                    if not np.isfinite(val):
                        continue

                    se    = float('nan')
                    tstat = float('nan')
                    pval  = float('nan')
                    holdfast = False
                    if pf is not None and pname in pf.index:
                        row_pf = pf.loc[pname]
                        holdfast = bool(row_pf.get('holdfast', False))
                        for _c in ('std_err', 'std_err_opg', 'se', 'stderr'):
                            if _c in row_pf.index:
                                _v = float(row_pf[_c])
                                if np.isfinite(_v) and _v >= 0:
                                    se = _v
                                    break
                        for _c in ('t_stat', 't_stat_opg', 'tstat'):
                            if _c in row_pf.index:
                                _v = float(row_pf[_c])
                                if np.isfinite(_v):
                                    tstat = _v
                                    break
                        if np.isnan(tstat) and np.isfinite(se) and se > 0:
                            tstat = val / se
                        for _c in ('pval', 'p_value', 'pvalue', 'pval_opg'):
                            if _c in row_pf.index:
                                _v = float(row_pf[_c])
                                if np.isfinite(_v):
                                    pval = _v
                                    break
                        if np.isnan(pval) and np.isfinite(tstat):
                            from scipy.stats import norm as _sn
                            pval = float(2.0 * _sn.sf(abs(tstat)))

                    # Classify issues
                    issues = []
                    if not holdfast and abs(val) < 1e-4:
                        issues.append('ZERO')
                    if not holdfast and np.isfinite(tstat) and abs(tstat) < 1.0:
                        issues.append('INSIG_LOW_T')
                    elif not holdfast and np.isfinite(pval) and pval > 0.10:
                        issues.append('HIGH_PVAL')
                    if np.isfinite(se) and se == 0.0 and not holdfast:
                        issues.append('SE_ZERO')

                    if issues:
                        issue_str = ','.join(issues)
                        logger.warning(
                            f"  [{seg}/{mtype}] {pname}: "
                            f"val={val:.4f} t={tstat:.2f} p={pval:.3g} → {issue_str}"
                        )
                    diag_rows.append({
                        'label':     label,
                        'segment':   seg,
                        'model':     mtype,
                        'parameter': pname,
                        'value':     round(val, 6),
                        'std_err':   round(se,   6) if np.isfinite(se)    else None,
                        'tstat':     round(tstat, 3) if np.isfinite(tstat) else None,
                        'pval':      round(pval,  4) if np.isfinite(pval)  else None,
                        'holdfast':  holdfast,
                        'issues':    issue_str if issues else '',
                    })
            except Exception as _ie:
                logger.debug(f"  param inspect failed [{seg}/{mtype}]: {_ie}")

    # Summary counts
    if diag_rows:
        df_diag = pd.DataFrame(diag_rows)
        n_zero    = int((df_diag['issues'].str.contains('ZERO')).sum())
        n_insig   = int((df_diag['issues'].str.contains('INSIG|HIGH')).sum())
        n_flagged = int((df_diag['issues'] != '').sum())
        logger.info(
            f"  Diagnostics: {len(diag_rows)} params total | "
            f"{n_zero} near-zero | {n_insig} insignificant | {n_flagged} flagged"
        )
        if n_flagged:
            logger.warning(
                f"  → Consider removing or holdfasting flagged parameters to improve "
                f"model parsimony and numerical stability."
            )

        # Save diagnostic HTML
        os.makedirs(save_dir, exist_ok=True)
        _write_param_diagnostic_html(df_diag, sign_results, label, save_dir)
    logger.info(SEP)


def _write_param_diagnostic_html(df_diag: pd.DataFrame,
                                  sign_results: list,
                                  label: str,
                                  save_dir: str) -> None:
    """Write a diagnostic HTML report for parameter quality."""
    try:
        html_path = os.path.join(save_dir, f'{label}_param_diagnostics.html')
        flagged   = df_diag[df_diag['issues'] != ''].copy()

        def _fmt(v, decimals=4):
            try:
                return f'{float(v):.{decimals}f}' if v is not None and np.isfinite(float(v)) else 'n/a'
            except Exception:
                return str(v)

        def _row_bg(issue):
            if 'ZERO' in str(issue):
                return '#fff3cd'  # yellow
            if 'INSIG' in str(issue) or 'HIGH_PVAL' in str(issue):
                return '#fce4e4'  # red
            return ''

        # Flagged parameter rows
        flagged_html = ''
        for _, r in flagged.sort_values(['segment', 'model', 'issues']).iterrows():
            bg = _row_bg(r['issues'])
            flagged_html += (
                f"<tr style='background:{bg}'>"
                f"<td>{r['segment']}</td><td>{r['model']}</td>"
                f"<td><b>{r['parameter']}</b></td>"
                f"<td style='text-align:right'>{_fmt(r['value'])}</td>"
                f"<td style='text-align:right'>{_fmt(r['std_err'])}</td>"
                f"<td style='text-align:right'>{_fmt(r['tstat'], 2)}</td>"
                f"<td style='text-align:right'>{_fmt(r['pval'], 4)}</td>"
                f"<td style='color:{'#b71c1c' if r['issues'] else '#1a7a1a'}'>"
                f"<b>{r['issues']}</b></td>"
                "</tr>\n"
            )

        # Sign check rows
        sign_html = ''
        for r in sign_results:
            ok  = r.get('pass')
            bg  = '' if ok else '#fce4e4'
            icon = '✓' if ok else ('✗' if ok is False else '?')
            sign_html += (
                f"<tr style='background:{bg}'>"
                f"<td>{r['segment']}</td><td>{r['model']}</td>"
                f"<td>{r['parameter']}</td>"
                f"<td style='text-align:right'>{r['value']:.4f}</td>"
                f"<td>{r['expected']}</td>"
                f"<td style='color:{'#1a7a1a' if ok else '#b71c1c'};font-weight:bold'>{icon}</td>"
                f"<td>{r['note']}</td>"
                "</tr>\n"
            )

        # All parameters (full table)
        all_html = ''
        for _, r in df_diag.sort_values(['segment', 'model', 'parameter']).iterrows():
            bg = _row_bg(r['issues'])
            insig_color = '#b71c1c' if r['issues'] else '#1a7a1a'
            all_html += (
                f"<tr style='background:{bg}'>"
                f"<td>{r['segment']}</td><td>{r['model']}</td>"
                f"<td>{r['parameter']}</td>"
                f"<td style='text-align:right'>{_fmt(r['value'])}</td>"
                f"<td style='text-align:right'>{_fmt(r['std_err'])}</td>"
                f"<td style='text-align:right'>{_fmt(r['tstat'], 2)}</td>"
                f"<td style='text-align:right'>{_fmt(r['pval'], 4)}</td>"
                f"<td style='color:{insig_color}'>{r['issues'] or '—'}</td>"
                "</tr>\n"
            )

        n_zero  = int((df_diag['issues'].str.contains('ZERO')).sum())
        n_insig = int((df_diag['issues'].str.contains('INSIG|HIGH')).sum())
        n_sign  = len([r for r in sign_results if r.get('pass') is False])

        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>{label} parameter diagnostics</title>
<style>
body{{font-family:Arial,sans-serif;margin:20px;font-size:13px}}
table{{border-collapse:collapse;width:100%;margin-bottom:28px}}
th{{background:#2c3e50;color:white;padding:6px 10px;text-align:left}}
td{{border:1px solid #ddd;padding:5px 8px}}
h1,h2{{color:#2c3e50}} .box{{border:1px solid #ddd;padding:10px;margin:12px 0;border-radius:4px}}
.warn{{color:#b71c1c}} .ok{{color:#1a7a1a}} .badge{{font-size:12px;font-weight:bold;
padding:2px 7px;border-radius:3px}}
.badge-zero{{background:#fff3cd}} .badge-insig{{background:#fce4e4}}
</style></head><body>
<h1>{label} — Parameter Diagnostics</h1>
<div class='box'>
  <b>Summary:</b>
  {len(df_diag)} total parameters &nbsp;|&nbsp;
  <span class='warn'><b>{n_zero} near-zero</b></span> &nbsp;|&nbsp;
  <span class='warn'><b>{n_insig} insignificant (|t|&lt;1.96 or p&gt;0.10)</b></span> &nbsp;|&nbsp;
  <span class='warn'><b>{n_sign} wrong-sign</b></span>
  <p>Recommendations for flagged parameters:
  <ul>
    <li><b>ZERO</b> — parameter stuck at or near zero. Consider removing it from the spec 
    to improve parsimony, or check if the variable has variation in the estimation data.</li>
    <li><b>INSIG_LOW_T / HIGH_PVAL</b> — not significantly different from zero at 90% confidence.
    Consider removing to reduce BIC penalty.</li>
    <li><b>SE_ZERO</b> — standard error is exactly zero, likely a degenerate direction. 
    Holdfast or remove this parameter.</li>
    <li><b>Wrong sign</b> — coefficient has a theoretically incorrect sign. 
    Check for collinearity with other variables, or constrain via holdfast.</li>
  </ul></p>
</div>

<h2>Flagged Parameters ({len(flagged)})</h2>
<table>
  <tr><th>Segment</th><th>Model</th><th>Parameter</th><th>Value</th>
      <th>Std Err</th><th>t-stat</th><th>p-value</th><th>Issue</th></tr>
{flagged_html if flagged_html else '<tr><td colspan=8 style="color:#1a7a1a">No flagged parameters</td></tr>'}
</table>

<h2>Behavioural Sign Checks ({len(sign_results)})</h2>
<table>
  <tr><th>Segment</th><th>Model</th><th>Parameter</th><th>Value</th>
      <th>Expected</th><th>Pass</th><th>Note</th></tr>
{sign_html if sign_html else '<tr><td colspan=7 style="color:#888">No sign checks available</td></tr>'}
</table>

<h2>All Parameters</h2>
<table>
  <tr><th>Segment</th><th>Model</th><th>Parameter</th><th>Value</th>
      <th>Std Err</th><th>t-stat</th><th>p-value</th><th>Flag</th></tr>
{all_html}
</table>
</body></html>"""

        with open(html_path, 'w', encoding='utf-8') as _hf:
            _hf.write(html)
        logger.info(f"  Parameter diagnostics → {html_path}")
    except Exception as _e:
        logger.debug(f"  _write_param_diagnostic_html failed: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION WRAPPERS  (Phase 2 — synth population, one zone at a time)
# Each loads previously estimated models and applies them to the synth.
# ─────────────────────────────────────────────────────────────────────────────

@timer("Predict Stage 1 — activity frequency")
def predict_stage1(synth, models=None, hhts_persons=None, hhts_tours=None):
    from stage1_activity_frequency import (
        predict_activity_frequency,
        calibrate_activity_rates,
        load_models,
    )
    if models is None:
        models = load_models(os.path.join(OUT_DIR, 'stage1_models.pkl'))
    logger.info(f"  Predicting activity frequency for {len(synth):,} persons...")
    result = predict_activity_frequency(synth, models)
    if hhts_persons is not None and hhts_tours is not None:
        try:
            result = calibrate_activity_rates(
                result,
                hhts_persons,
                hhts_tours,
                rng=np.random.default_rng(43),
            )
            logger.info("  Stage 1 calibrated to HHTS ptype-purpose rates")
        except Exception as _cal_e:
            logger.warning(f"  Stage 1 calibration skipped (non-fatal): {_cal_e}")
    result.to_csv(os.path.join(OUT_DIR, 'stage1_output.csv'), index=False)
    logger.info(f"  ✓  Stage 1 output: {len(result):,} rows → {OUT_DIR}/stage1_output.csv")
    _write_stage_checkpoint_report(
        stage_tag='predict_stage1',
        title='Predict Stage 1 - Activity Frequency',
        summary={
            'n_rows': len(result),
            'n_persons': int(result['person_id'].nunique()) if 'person_id' in result.columns else None,
            'output_csv': os.path.join(OUT_DIR, 'stage1_output.csv'),
        },
        table=result,
    )
    return result


@timer("Predict Stage 2 — day duration")
def predict_stage2(synth_s1, models=None):
    from stage2_day_duration import predict_day_budget, load_models
    if models is None:
        models = load_models(os.path.join(OUT_DIR, 'stage2_models.pkl'))
    logger.info(f"  Predicting day duration for {len(synth_s1):,} persons...")
    result = predict_day_budget(synth_s1, models)
    result.to_csv(os.path.join(OUT_DIR, 'stage2_output.csv'), index=False)
    logger.info(f"  ✓  Stage 2 output: {len(result):,} rows → {OUT_DIR}/stage2_output.csv")
    _write_stage_checkpoint_report(
        stage_tag='predict_stage2',
        title='Predict Stage 2 - Day Duration',
        summary={
            'n_rows': len(result),
            'n_persons': int(result['person_id'].nunique()) if 'person_id' in result.columns else None,
            'output_csv': os.path.join(OUT_DIR, 'stage2_output.csv'),
        },
        table=result,
    )
    return result


@timer("Predict Stage 3 — activity duration")
def predict_stage3(synth_s2, models=None):
    from stage3_activity_duration import predict_activity_durations, load_models
    if models is None:
        models = load_models(os.path.join(OUT_DIR, 'stage3_models.pkl'))
    _use_mdcev = os.environ.get('STAGE3_USE_MDCEV', '0') == '1'
    logger.info(f"  Predicting activity durations for {len(synth_s2):,} persons...")
    result = predict_activity_durations(synth_s2, models, use_mdcev=_use_mdcev)
    result.to_csv(os.path.join(OUT_DIR, 'stage3_output.csv'), index=False)
    logger.info(f"  ✓  Stage 3 output: {len(result):,} rows "
                f"({result['person_id'].nunique():,} persons) → {OUT_DIR}/stage3_output.csv")
    _write_stage_checkpoint_report(
        stage_tag='predict_stage3',
        title='Predict Stage 3 - Activity Duration',
        summary={
            'n_rows': len(result),
            'n_persons': int(result['person_id'].nunique()) if 'person_id' in result.columns else None,
            'output_csv': os.path.join(OUT_DIR, 'stage3_output.csv'),
        },
        table=result,
    )
    return result


@timer("Predict Stage 4 — schedule placement")
def predict_stage4(activities, models=None):
    from stage4_schedule_placement import predict_schedule, load_models
    if models is None:
        models = load_models(os.path.join(OUT_DIR, 'stage4_models.pkl'))
    logger.info(f"  Building schedules for {len(activities):,} activities "
                f"({activities['person_id'].nunique():,} persons)...")
    result = predict_schedule(activities, models)
    result.to_csv(os.path.join(OUT_DIR, 'stage4_output.csv'), index=False)
    logger.info(f"  ✓  Stage 4 output: {len(result):,} rows → {OUT_DIR}/stage4_output.csv")
    _write_stage_checkpoint_report(
        stage_tag='predict_stage4',
        title='Predict Stage 4 - Schedule Placement',
        summary={
            'n_rows': len(result),
            'n_persons': int(result['person_id'].nunique()) if 'person_id' in result.columns else None,
            'output_csv': os.path.join(OUT_DIR, 'stage4_output.csv'),
        },
        table=result,
    )
    return result


@timer("Predict Stage 5 — destination + mode (one origin zone at a time)")
def predict_stage5(activities, models=None, out=None, landuse=None, skims=None, sk=None,
                   persons=None):
    """
    Apply Stage 5 destination + mode models to the synth population.

    Processes ONE origin zone (OTAZ) at a time to keep peak memory minimal.
    Each zone's activities (~12k persons, ~40k rows) is independently predicted
    and its results appended to the output CSV.  Total memory per iteration:
    ~300 MB vs ~6 GB for the full population at once.
    """
    from stage5_destination_choice import (
        _build_trip_chain, _attach_skims, _skim_flat, _skim_lookup,
        compute_validation_metrics,
        load_models as s5_load
    )
    import openmatrix as omx

    if models is None:
        models = s5_load(os.path.join(OUT_DIR, 'stage5_models.pkl'))

    skims_path = PIPELINE_SKIMS_PATH
    if skims is None:
        skims = omx.open_file(skims_path, 'r')
    if sk is None:
        sk = _skim_lookup(_skim_flat(skims))
    pp = persons if persons is not None else (out["persons"] if out else pd.DataFrame())

    # Ensure synthetic persons have usual work/school anchors before Stage 5.
    if not pp.empty and landuse is not None:
        missing_anchors = False
        zone_col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
        valid_zone_set = set(pd.to_numeric(landuse[zone_col], errors='coerce').dropna().astype(int).tolist())
        if 'WORK_TAZ' not in pp.columns or 'SCHOOL_TAZ' not in pp.columns:
            missing_anchors = True
        else:
            if 'WORKSTAT' in pp.columns:
                workers_no_anchor = pp.loc[pp['WORKSTAT'].isin([1, 2]), 'WORK_TAZ'].fillna(0).astype(int).le(0)
                if workers_no_anchor.any():
                    missing_anchors = True
            if 'STUDYING' in pp.columns:
                students_no_anchor = pp.loc[pp['STUDYING'].isin([1, 2, 3, 4]), 'SCHOOL_TAZ'].fillna(0).astype(int).le(0)
                if students_no_anchor.any():
                    missing_anchors = True
            # Invalidate anchors not present in landuse zones to prevent impossible destinations.
            _work_ok = pp['WORK_TAZ'].fillna(0).astype(int)
            _school_ok = pp['SCHOOL_TAZ'].fillna(0).astype(int)
            invalid_work = _work_ok.gt(0) & (~_work_ok.isin(valid_zone_set))
            invalid_school = _school_ok.gt(0) & (~_school_ok.isin(valid_zone_set))
            if invalid_work.any() or invalid_school.any():
                logger.warning(
                    "Stage 5 prediction: found invalid usual-location anchors "
                    f"(WORK_TAZ invalid={int(invalid_work.sum())}, "
                    f"SCHOOL_TAZ invalid={int(invalid_school.sum())})"
                )
                missing_anchors = True
        if missing_anchors:
            logger.warning("Stage 5 prediction: missing WORK_TAZ/SCHOOL_TAZ anchors in synthetic persons; recomputing usual locations.")
            from usual_location import run_usual_locations
            pp = run_usual_locations(pp, landuse, skims=skims, save_path=None)

    # Sort activities by origin zone
    acts = activities.copy()
    acts['OTAZ'] = pd.to_numeric(acts.get('OTAZ', 1), errors='coerce').fillna(1).astype(int)
    zones = sorted(acts['OTAZ'].unique().tolist())
    n_zones = len(zones)
    zone_counts = (
        acts.groupby('OTAZ')
        .agg(n_activities=('OTAZ', 'size'), n_persons=('person_id', 'nunique'))
        .reset_index()
        .sort_values('OTAZ')
    )
    zone_counts.to_csv(os.path.join(OUT_DIR, 'stage5_zones_input.csv'), index=False)

    out_path = os.path.join(OUT_DIR, 'stage5_output.csv')
    header_written = False
    total_trips = 0
    zone_stats = []

    # Build skim_df ONCE before the loop — _skim_flat reads full OMX (~100 MB)
    # and calling it per zone would read the file 325 times unnecessarily.
    logger.info("  Building flat skim table for travel-time attachment...")
    _pred_skim_df = _skim_flat(skims)
    logger.info(f"  Skim table: {len(_pred_skim_df):,} OD pairs")

    n_workers = int(os.environ.get('STAGE5_WORKERS', '1'))
    logger.info(f"  Predicting {len(acts):,} activities across {n_zones} origin zones "
                f"({n_workers} worker{'s' if n_workers > 1 else ''})...")

    def _record_zone(otaz, z_idx, n_acts, n_persons, zone_plan, elapsed, error=None):
        nonlocal header_written, total_trips
        if error:
            logger.warning(f"  Zone {otaz} failed: {error} — skipping")
            zone_stats.append({'OTAZ': int(otaz), 'zone_idx': int(z_idx),
                               'n_persons': int(n_persons), 'n_activities': int(n_acts),
                               'n_trips': 0, 'elapsed_sec': float(elapsed),
                               'status': f'failed: {error}'})
            return
        zone_plan.to_csv(out_path,
                         mode='w' if not header_written else 'a',
                         header=not header_written, index=False)
        header_written = True
        zone_trips = int((zone_plan['row_type'] == 'trip').sum())
        total_trips += zone_trips
        zone_stats.append({'OTAZ': int(otaz), 'zone_idx': int(z_idx),
                           'n_persons': int(n_persons), 'n_activities': int(n_acts),
                           'n_trips': int(zone_trips), 'elapsed_sec': float(elapsed),
                           'status': 'ok'})
        if z_idx == 1 or z_idx % 50 == 0 or z_idx == n_zones:
            logger.info(f"  Zone {z_idx:4d}/{n_zones}  OTAZ={otaz:3d} "
                        f"({n_persons} persons)  trips so far: {total_trips:,}")

    if n_workers > 1:
        import multiprocessing as _mp
        import concurrent.futures as _cf
        import pickle as _pkl_s5
        import shutil as _shutil
        # Write large shared DataFrames to parquet so spawned workers can load
        # them from disk instead of receiving them through the IPC pipe.
        _s5_tmp = os.path.join(OUT_DIR, '_s5_worker_tmp')
        os.makedirs(_s5_tmp, exist_ok=True)
        _pp_path  = os.path.join(_s5_tmp, 'pp.parquet')
        _sdf_path = os.path.join(_s5_tmp, 'skim_df.parquet')
        try:
            pp.to_parquet(_pp_path)
            _pred_skim_df.to_parquet(_sdf_path)
        except Exception as _pq_e:
            logger.warning(f"  Parquet write failed ({_pq_e}); falling back to pickle for worker data")
            _pp_path  = os.path.join(_s5_tmp, 'pp.pkl')
            _sdf_path = os.path.join(_s5_tmp, 'skim_df.pkl')
            pp.to_pickle(_pp_path)
            _pred_skim_df.to_pickle(_sdf_path)
        _lu_bytes = _pkl_s5.dumps(landuse)
        _sk_bytes = _pkl_s5.dumps(sk)
        _models_path = os.path.join(OUT_DIR, 'stage5_models.pkl')

        # Hide GPU from spawned children — set in parent before spawn so the
        # env is inherited at process start (before any CUDA/JAX initialisation).
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'

        logger.info(f"  Parallel Stage 5: {n_workers} workers via spawn pool (GPU hidden)")
        tasks = [
            (acts[acts['OTAZ'] == otaz].copy(), 1000 + otaz, otaz)
            for otaz in zones
        ]
        zone_index = {otaz: z_idx for z_idx, otaz in enumerate(zones, 1)}
        ctx = _mp.get_context('spawn')
        initargs = (_models_path, _pp_path, _lu_bytes, skims_path, _sdf_path, _sk_bytes)
        try:
            with _cf.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_s5_spawn_initializer, initargs=initargs,
            ) as pool:
                futures = {pool.submit(_s5_zone_worker, t): t[2] for t in tasks}
                for fut in _cf.as_completed(futures):
                    otaz_done = futures[fut]
                    z_idx = zone_index[otaz_done]
                    try:
                        otaz_r, zone_plan, err, elapsed = fut.result()
                    except Exception as exc:
                        _record_zone(otaz_done, z_idx, 0, 0, None, 0.0, str(exc))
                        continue
                    n_acts_done = int((acts['OTAZ'] == otaz_done).sum())
                    n_pers_done = acts.loc[acts['OTAZ'] == otaz_done, 'person_id'].nunique() \
                                  if 'person_id' in acts.columns else 0
                    _record_zone(otaz_done, z_idx, n_acts_done, n_pers_done,
                                 zone_plan, elapsed, err)
        finally:
            _shutil.rmtree(_s5_tmp, ignore_errors=True)
    else:
        for z_idx, otaz in enumerate(zones, 1):
            zone_t0 = time.perf_counter()
            zone_mask  = acts['OTAZ'] == otaz
            zone_batch = acts[zone_mask].copy()
            n_p = zone_batch['person_id'].nunique() if 'person_id' in zone_batch.columns else 0
            try:
                zone_plan = _build_trip_chain(
                    zone_batch, models, pp, landuse, skims, sk,
                    seed=1000 + otaz
                )
                zone_plan = _attach_skims(zone_plan, _pred_skim_df)
                _record_zone(otaz, z_idx, len(zone_batch), n_p, zone_plan,
                             time.perf_counter() - zone_t0)
                del zone_plan
            except Exception as e:
                _record_zone(otaz, z_idx, len(zone_batch), n_p, None,
                             time.perf_counter() - zone_t0, str(e))
            del zone_batch

    del _pred_skim_df
    if zone_stats:
        zone_stats_df = pd.DataFrame(zone_stats)
        zone_stats_df.to_csv(os.path.join(OUT_DIR, 'stage5_zone_timings.csv'), index=False)
        slow = zone_stats_df[zone_stats_df['status'] == 'ok'].nlargest(10, 'elapsed_sec')
        if not slow.empty:
            logger.info(
                "  Slowest Stage 5 zones (sec): " +
                ', '.join(f"{int(r.OTAZ)}={r.elapsed_sec:.1f}" for r in slow.itertuples(index=False))
            )

    n_ok = sum(1 for s in zone_stats if s.get('status') == 'ok')
    logger.info(f"  ✓  Stage 5 complete: {n_ok}/{n_zones} zones succeeded, "
                f"{total_trips:,} trips → {out_path}")
    if not os.path.exists(out_path):
        failed = [s for s in zone_stats if s.get('status', '').startswith('failed')]
        sample = '; '.join(f"OTAZ {s['OTAZ']}: {s['status']}" for s in failed[:3])
        raise RuntimeError(
            f"Stage 5 produced no output — all {n_zones} zones failed. "
            f"First failures: {sample}"
        )
    result = pd.read_csv(out_path)

    # ActivitySim-style location-choice coverage metrics:
    # workplace, school, tour-level, and trip-level consistency.
    try:
        _obs_tours = out.get('tours', pd.DataFrame()) if isinstance(out, dict) else pd.DataFrame()
        _val_path = os.path.join(OUT_DIR, 'stage5_validation_metrics.csv')
        compute_validation_metrics(
            obs_tours=_obs_tours,
            pred_plan=result,
            models=models,
            persons=pp,
            save_path=_val_path,
        )
    except Exception as _s5_val_e:
        logger.debug(f"Stage 5 post-predict validation skipped ({_s5_val_e})")

    _write_stage_checkpoint_report(
        stage_tag='predict_stage5',
        title='Predict Stage 5 - Destination and Mode Choice',
        summary={
            'n_rows': len(result),
            'n_persons': int(result['person_id'].nunique()) if 'person_id' in result.columns else None,
            'n_zones': int(n_zones),
            'n_zones_ok': int(n_ok),
            'stage5_output_csv': out_path,
            'stage5_zone_timings_csv': os.path.join(OUT_DIR, 'stage5_zone_timings.csv'),
            'stage5_validation_csv': os.path.join(OUT_DIR, 'stage5_validation_metrics.csv'),
        },
        table=result,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY COMBINED WRAPPERS  (used by GA_main, GA_STAGED, and direct calls)
# These keep the old estimate+predict-together behaviour for the GA modes.
# ─────────────────────────────────────────────────────────────────────────────

@timer("Stage 1 — activity frequency")
def stage1(out, synth):
    synth_s1, models = run_stage1(out["tours"], out["persons"], synth,
                                    save_path=os.path.join(OUT_DIR, 'stage1_models.pkl'))
    _save_stage_coef_csv(models, 'stage1', OUT_DIR)
    synth_s1.to_csv(os.path.join(OUT_DIR, 'stage1_output.csv'), index=False)
    _write_stage_checkpoint_report(
        stage_tag='legacy_stage1',
        title='Legacy Stage 1 - Estimate and Predict',
        summary={'n_rows': len(synth_s1), 'n_models': len(models)},
        table=synth_s1,
    )
    return synth_s1, models


@timer("Stage 2 — day duration")
def stage2(out, synth_s1):
    synth_s2, models = run_stage2(out["tours"], out["persons"], synth_s1,
                                    save_path=os.path.join(OUT_DIR, 'stage2_models.pkl'))
    _save_stage_coef_csv(models, 'stage2', OUT_DIR)
    synth_s2.to_csv(os.path.join(OUT_DIR, 'stage2_output.csv'), index=False)
    _write_stage_checkpoint_report(
        stage_tag='legacy_stage2',
        title='Legacy Stage 2 - Estimate and Predict',
        summary={'n_rows': len(synth_s2), 'n_models': len(models)},
        table=synth_s2,
    )
    return synth_s2, models


@timer("Stage 3 — activity duration")
def stage3(out, synth_s2):
    activities, models = run_stage3(
        out["tours"], out["persons"], synth_s2, trips=out["trips"],
        save_path=os.path.join(OUT_DIR, 'stage3_models.pkl'),
    )
    _save_stage_coef_csv(models, 'stage3', OUT_DIR)
    activities.to_csv(os.path.join(OUT_DIR, 'stage3_output.csv'), index=False)
    _write_stage_checkpoint_report(
        stage_tag='legacy_stage3',
        title='Legacy Stage 3 - Estimate and Predict',
        summary={'n_rows': len(activities), 'n_models': len(models)},
        table=activities,
    )
    return activities, models


@timer("Stage 4 — schedule placement")
def stage4(out, activities):
    scheduled, models = run_stage4(
        out["trips"], out["tours"], out["persons"], activities,
        save_path=os.path.join(OUT_DIR, 'stage4_models.pkl'),
    )
    _save_stage_coef_csv(models, 'stage4', OUT_DIR)
    scheduled.to_csv(os.path.join(OUT_DIR, 'stage4_output.csv'), index=False)
    _write_stage_checkpoint_report(
        stage_tag='legacy_stage4',
        title='Legacy Stage 4 - Estimate and Predict',
        summary={'n_rows': len(scheduled), 'n_models': len(models)},
        table=scheduled,
    )
    return scheduled, models


@timer("Stage 5 — destination choice")
def stage5(out, synth_persons, scheduled, landuse):
    import openmatrix as omx
    from stage5_destination_choice import _skim_flat, _skim_lookup
    skims_path = PIPELINE_SKIMS_PATH
    # Require Stage 5 to fit cleanly by default; fallback paths are opt-in.
    os.environ.setdefault('STAGE5_STRICT_ESTIMATION', '1')
    # Keep destination-estimation requirement enabled by default.
    os.environ.setdefault('STAGE5_REQUIRE_DEST_ESTIMATION', '1')

    crash_skims = _detect_stage5_crash_source()
    if crash_skims:
        logger.info(f"Stage 5 crash source detected: {crash_skims}")
    else:
        logger.warning("Stage 5 crash source not found; crash utility terms will use zeros")

    list_omx_matrices(skims_path)
    scheduled_dest, models = run_stage5(
        out["trips"], synth_persons, scheduled,
        skims=skims_path,
        landuse=landuse,
        tours=out["tours"],
        save_path=os.path.join(OUT_DIR, 'stage5_models.pkl'),
        crash_skims=crash_skims,
    )
    scheduled_dest.to_csv(os.path.join(OUT_DIR, 'stage5_output.csv'), index=False)
    _write_stage_checkpoint_report(
        stage_tag='legacy_stage5',
        title='Legacy Stage 5 - Estimate and Predict',
        summary={
            'n_rows': len(scheduled_dest),
            'n_persons': int(scheduled_dest['person_id'].nunique()) if 'person_id' in scheduled_dest.columns else None,
            'n_segments': len(models),
        },
        table=scheduled_dest,
    )
    skims      = omx.open_file(skims_path, 'r')
    skim_df    = _skim_flat(skims)
    sk_lookup  = _skim_lookup(skim_df)
    return scheduled_dest, models, skims, sk_lookup


def _resolve_stage5b_reuse_path() -> Optional[str]:
    """Locate a previously-estimated stage5b_models.pkl to reuse for predict-only mode.

    Mirrors the GA_STAGED_REUSE_ESTIMATOR resolution used for ga_results.pkl
    (see main_ga_staged): a full-population run's Stage 5b would otherwise
    re-estimate a fresh MNL per purpose on the *entire* predicted trip table
    (`_estimate_mnl_segment` builds a dense observations x zones design
    matrix), which for millions of stop trips exceeds typical job memory
    budgets (job 23661073: OOM-killed at ~261GB estimating n=2,600,393 home
    stops). When a prior lighter estimation run already produced
    stage5b_models.pkl, reuse it and predict-only instead.
    """
    if os.environ.get('GA_STAGED_REUSE_ESTIMATOR', '0').strip() != '1':
        return None
    candidates = []
    _ga_dir = os.environ.get('GA_STAGED_GA_DIR', '').strip()
    if _ga_dir:
        candidates.append(os.path.join(_ga_dir, 'stage5b_models.pkl'))
    candidates.append(os.path.join(OUT_DIR, 'stage5b_models.pkl'))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@timer("Stage 5b — trip-level destination choice")
def stage5b(out, scheduled_dest, landuse, reuse_models_path=None):
    """Estimate + predict trip-level intermediate-stop destination choice.

    Used by GA_main / GA_STAGED (combined estimate+predict pattern).
    In the two-phase main() pipeline, stage5b is called inline with travellers
    as the persons source.

    When GA_STAGED_REUSE_ESTIMATOR=1 and a prior stage5b_models.pkl can be
    resolved, skips re-estimation and predicts using the reused coefficients.
    """
    if reuse_models_path is None:
        reuse_models_path = _resolve_stage5b_reuse_path()
    result = run_stage5b(
        trips              = scheduled_dest,
        persons            = out["persons"],
        skims_path         = PIPELINE_SKIMS_PATH,
        landuse            = landuse,
        output_dir         = OUT_DIR,
        reuse_models_path  = reuse_models_path,
    )
    # run_stage5b always returns a dict {'trips': df, ...}; extract the DataFrame
    # so callers (stage6, GA paths) always receive a DataFrame, not a dict.
    if isinstance(result, dict):
        return result.get('trips', scheduled_dest)
    return result


@timer("Stage 6 — plan validation + Aimsun export")
def stage6(out, scheduled_dest, landuse,
           stage5_models=None, skims=None, skim_lookup=None):
    # Build zone centroid table for coordinate pinning.
    # Priority: cached centroids from get_zone_centroids_rich (uses ABS API or
    # SA4 anchor fallback) -> build_sa2_centroids_from_landuse (always works,
    # no network needed).
    sa2_geo = None
    try:
        sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
        sa2_geo = sa2_geo[['ORDER', 'lat', 'lon']].copy()
        logger.info(f"  Zone centroids: {sa2_geo['lat'].notna().sum()}/{len(sa2_geo)} zones")
    except Exception as e:
        logger.warning(f"get_zone_centroids_rich failed ({e}) — using landuse fallback")
    if sa2_geo is None or sa2_geo['lat'].isna().all():
        try:
            from spatial_assignment import build_sa2_centroids_from_landuse
            sa2_geo = build_sa2_centroids_from_landuse(landuse)
            sa2_geo = sa2_geo.rename(columns={'zone_id': 'ORDER'})                 if 'zone_id' in sa2_geo.columns else sa2_geo
            logger.info(f"  Zone centroids (landuse fallback): {len(sa2_geo)} zones")
        except Exception as e2:
            logger.warning(f"build_sa2_centroids_from_landuse also failed ({e2}) — no coords")
            sa2_geo = None

    final_plan, trips_aimsun = run_stage6(
        plan               = scheduled_dest,
        persons            = out["persons"],
        landuse            = landuse,
        stage5_models      = stage5_models,
        skims              = skims,
        skim_lookup        = skim_lookup,
        sa2_geo            = sa2_geo,
        repair_dwells      = stage5_models is not None,
        dynus_sim_start_min= 360,
        dynus_value_of_time= 15.0,
        save_dir           = OUT_DIR,
    )
    logger.info(f"  Per-zone folders: {OUT_DIR}/zones/zone_*/")
    _write_stage_checkpoint_report(
        stage_tag='legacy_stage6',
        title='Legacy Stage 6 - Finalisation and Export',
        summary={
            'n_rows': len(final_plan),
            'n_persons': int(final_plan['person_id'].nunique()) if 'person_id' in final_plan.columns else None,
            'n_trips_for_aimsun': int(len(trips_aimsun)),
            'stage6_output_csv': os.path.join(OUT_DIR, 'stage6_output.csv'),
            'aimsun_csv': os.path.join(OUT_DIR, 'trips_for_aimsun.csv'),
        },
        table=final_plan,
    )
    return final_plan, trips_aimsun



@timer("plot activity plans")
def plot(plan):
    #todo there is an error here with the pltos
    import sys
    plots_dir = os.path.join(OUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    try:
        plot_all(plan, n_sample=8, save_dir=plots_dir)
        logger.info(f"  Activity plan plots saved → {plots_dir}/")
    except Exception as e:
        logger.warning(f"  plot_all failed ({e}) — skipping activity plots")
    sys.stdout.flush()


@timer("spatial plots")
def spatial_plots(plan, synth, landuse):
    import os
    os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)
    os.makedirs('saved_data/osm_cache', exist_ok=True)

    # Build zone centroid table — no OSM/network access needed.
    sa2_geo = None
    try:
        sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
        sa2_geo = sa2_geo[['ORDER', 'lat', 'lon']].rename(columns={'ORDER': 'zone_id'})
        logger.info(f"  Zone centroids for spatial plots: {sa2_geo['lat'].notna().sum()}/{len(sa2_geo)} zones")
    except Exception as e:
        logger.warning(f"get_zone_centroids_rich failed ({e}) — trying landuse fallback")
    if sa2_geo is None or sa2_geo['lat'].isna().all():
        try:
            from spatial_assignment import build_sa2_centroids_from_landuse
            sa2_geo = build_sa2_centroids_from_landuse(landuse)
            logger.info(f"  Zone centroids (landuse fallback): {len(sa2_geo)} zones")
        except Exception as e2:
            logger.warning(f"Zone centroid build failed ({e2}) — using Brisbane anchor")
            sa2_geo = landuse[['ORDER']].rename(columns={'ORDER': 'zone_id'}).copy()
            sa2_geo['lat'] = -27.47
            sa2_geo['lon'] = 153.03

    # assign_coordinates uses zone centroids (+small jitter) — no OSM calls needed.
    try:
        from spatial_assignment import assign_coordinates
        _sa2 = sa2_geo.copy()
        if 'ORDER' in _sa2.columns and 'zone_id' not in _sa2.columns:
            _sa2 = _sa2.rename(columns={'ORDER': 'zone_id'})
        plan_coords = assign_coordinates(plan, _sa2, seed=42)
        logger.info(
            f"  Coordinates assigned: "
            f"{plan_coords['dist_km'].notna().sum() if 'dist_km' in plan_coords.columns else 0}"
            f"/{(plan_coords.get('row_type', pd.Series('trip')) == 'trip').sum()} trip rows"
        )
    except Exception as e:
        logger.warning(f"assign_coordinates failed ({e}) — plots will have no spatial coords")
        plan_coords = plan.copy()

    import sys
    _pdir = os.path.join(OUT_DIR, 'plots')

    # Each plot is wrapped individually so one failure doesn't block the rest.
    _saved = []
    for _fn, _func, _kw in [
        ('distance_distribution.png', plot_distance_distribution,
            {'save_path': os.path.join(_pdir, 'distance_distribution.png')}),
        ('desire_lines.png', plot_desire_lines,
            {'save_path': os.path.join(_pdir, 'desire_lines.png')}),
        ('activity_heatmap.png', plot_activity_heatmap,
            {'save_path': os.path.join(_pdir, 'activity_heatmap.png')}),
        ('spatial_metrics.png', plot_spatial_metrics,
            {'save_path': os.path.join(_pdir, 'spatial_metrics.png')}),
    ]:
        try:
            _func(plan_coords, **_kw)
            _saved.append(_fn)
            logger.info(f"  Plot saved → {_kw['save_path']}")
        except Exception as _pe:
            logger.warning(f"  Plot '{_fn}' failed: {_pe}")
        sys.stdout.flush()

    # Sample person-movement plots — representative travellers (not outliers)
    try:
        if 'person_id' in plan_coords.columns:
            rt_col = 'row_type' if 'row_type' in plan_coords.columns else None
            trip_rows = plan_coords[plan_coords[rt_col] == 'trip'] if rt_col else plan_coords
            # Use haversine dist_km for ranking (always computed; skim dist may have
            # artefacts for edge OD pairs that produce unrealistically long trips).
            _dist_col = ('dist_km' if 'dist_km' in trip_rows.columns
                         else ('dist' if 'dist' in trip_rows.columns else None))
            if _dist_col and len(trip_rows) > 0:
                _dist_per_person = (trip_rows.groupby('person_id')[_dist_col]
                                    .sum().dropna())
                _p10 = float(_dist_per_person.quantile(0.10))
                _p90 = float(_dist_per_person.quantile(0.90))
                # Select from the bulk of travellers, not extreme outliers
                _repr = _dist_per_person[
                    (_dist_per_person >= _p10) & (_dist_per_person <= _p90)
                ]
                if len(_repr) == 0:
                    _repr = _dist_per_person
                sample_pids = _repr.sample(
                    min(4, len(_repr)), random_state=42
                ).index.tolist()
                logger.info(
                    f"  Movement plots: {len(sample_pids)} representative travellers "
                    f"(daily dist_km p10={_p10:.1f}-p90={_p90:.1f} km)"
                )
            else:
                sample_pids = []
            for pid in sample_pids:
                try:
                    _sp = os.path.join(_pdir, f'person_{pid}_movement.png')
                    plot_person_movement(plan_coords, pid, save_path=_sp)
                    _saved.append(f'person_{pid}_movement.png')
                    logger.info(f"  Movement plot -> {_sp}")
                except Exception as _mpe:
                    logger.warning(f"  Movement plot for {pid} failed: {_mpe}")
    except Exception as _e:
        logger.warning(f"  Person movement plots failed: {_e}")

    # Distance-linked sample plans: re-draw Gantt charts with dist_km annotations
    try:
        from plot_activity_plans import plot_sample_plans as _psp
        _spdist_path = os.path.join(_pdir, 'sample_plans_with_distances.png')
        _psp(plan_coords, n=8, save_path=_spdist_path, seed=42)
        _saved.append('sample_plans_with_distances.png')
        logger.info(f"  Distance-linked sample plans -> {_spdist_path}")
    except Exception as _spe:
        logger.warning(f"  sample_plans_with_distances failed: {_spe}")

    logger.info(f"  Spatial plots complete: {len(_saved)} files in {_pdir}/")
    sys.stdout.flush()
    return plan_coords


# ---------------------------------------------------------------------------
# GA WRAPPERS  (unchanged)
# ---------------------------------------------------------------------------

@timer("GA — all stages")
def ga_all(out, landuse):
    # Baseline-first policy: estimator/stage fallbacks must be explicitly enabled.
    os.environ.setdefault('GA_ESTIMATOR_FALLBACK', '0')
    os.environ.setdefault('GA_STAGE5_FALLBACK_TO_MAIN', '0')
    os.environ.setdefault('GA_STAGE5_PREFIT_BASELINE', '1')
    _run_ga_stages_with_checkpoints(out, landuse)


def _save_ga_checkpoint(all_ga, stage_id=None, failed=False):
    """Persist GA progress so long runs always leave usable artifacts."""
    import json
    status = {
        'completed_stages': sorted(int(k) for k in all_ga.results.keys()),
        'last_stage': int(stage_id) if stage_id is not None else None,
        'failed': bool(failed),
        'out_dir': OUT_DIR,
        'timestamp': pd.Timestamp.now().isoformat(),
    }

    # Keep a rolling partial plus the canonical ga_results.pkl.
    for _path in (
        os.path.join(OUT_DIR, 'ga_results_checkpoint.pkl'),
        os.path.join(OUT_DIR, 'ga_results.pkl'),
    ):
        try:
            with open(_path, 'wb') as _f:
                pickle.dump(all_ga, _f)
        except Exception as _e:
            logger.warning(f"Could not save GA checkpoint {_path}: {_e}")

    try:
        with open(os.path.join(OUT_DIR, 'ga_progress.json'), 'w', encoding='utf-8') as _f:
            json.dump(status, _f, indent=2)
    except Exception as _e:
        logger.warning(f"Could not write ga_progress.json: {_e}")

    # Keep lightweight CSV/TXT reports in sync with every checkpoint so users
    # can inspect best-found models even when a later stage fails.
    try:
        _reports_dir = os.path.join(OUT_DIR, 'ga_reports')
        all_ga.export_reports(_reports_dir)
    except Exception as _e:
        logger.warning(f"Could not export GA checkpoint reports: {_e}")


def _format_ga_checkpoint_summary(res) -> dict:
    """
    Extract and format GA result for checkpoint reporting.
    
    Shows:
    - Actual BIC (positive, to be minimized) not negative fitness
    - Selected variables (best_features)
    - Key metrics
    """
    if res is None:
        return {}
    
    best_fitness = float(getattr(res, 'best_fitness', np.nan))
    fitness_metric = str(getattr(res, 'fitness_metric', 'unknown'))
    best_features = getattr(res, 'best_features', [])

    selected_features_text = ""
    selected_features_list = []
    selected_features_by_block = {}
    n_features = 0

    if isinstance(best_features, dict):
        for block, vals in best_features.items():
            if vals is None:
                vals_list = []
            elif isinstance(vals, (list, tuple, set)):
                vals_list = [str(v) for v in vals]
            else:
                vals_list = [str(vals)]
            selected_features_by_block[str(block)] = vals_list
            for v in vals_list:
                selected_features_list.append(f"{block}:{v}")

        n_features = sum(len(v) for v in selected_features_by_block.values())
        lines = []
        for block, vals in selected_features_by_block.items():
            if vals:
                lines.append(f"{block}: " + ", ".join(vals))
            else:
                lines.append(f"{block}: (none)")
        selected_features_text = "\n".join(lines)
    else:
        if best_features is None:
            selected_features_list = []
        elif isinstance(best_features, (list, tuple, set)):
            selected_features_list = [str(v) for v in best_features]
        else:
            selected_features_list = [str(best_features)]
        n_features = len(selected_features_list)
        selected_features_text = ", ".join(selected_features_list)

    summary = {
        'n_selected_features': n_features,
        'selected_features': selected_features_text,
        'selected_features_list': selected_features_list,
        'selected_features_by_block': selected_features_by_block,
        'n_generations': len(getattr(res, 'history', [])),
        'n_candidates_evaluated': len(getattr(res, 'all_candidates', []) or []),
        'fitness_metric': fitness_metric,
    }
    
    # Report actual BIC (positive, minimized) not negative fitness
    if 'BIC' in fitness_metric and np.isfinite(best_fitness):
        # fitness = -BIC/n, so BIC/n = -fitness
        bic_per_obs = -best_fitness
        summary['best_BIC_per_obs'] = f"{bic_per_obs:.6f}"
        summary['best_fitness_note'] = "(BIC/n to minimize, lower = better)"
    else:
        summary['best_fitness'] = f"{best_fitness:.6f}"
        summary['fitness_note'] = "(higher = better)"
    
    return summary


def _ga_save_stage_coef_html_from_best(sid: int, res, out: dict, landuse=None) -> None:
    """
    Re-estimate stage `sid` using the GA's best-selected features and write
    coefficient HTML + models pkl to OUT_DIR (mirrors the main pipeline layout)
    and also to the ga_reports/ subdir.
    """
    from ga_hypothesis_testing import _ga_save_stage_coef_csv
    if res is None:
        return
    best_feats = getattr(res, 'best_features', None)
    save_dir = os.path.join(OUT_DIR, 'ga_reports')
    os.makedirs(save_dir, exist_ok=True)

    def _persist(models, stage_name):
        """Save coefficients to both ga_reports/ and OUT_DIR, plus pkl to OUT_DIR, plus HTML."""
        _ga_save_stage_coef_csv(models, stage_name, save_dir)
        _ga_save_stage_coef_csv(models, stage_name, OUT_DIR)
        _pkl_path = os.path.join(OUT_DIR, f'{stage_name}_models.pkl')
        try:
            with open(_pkl_path, 'wb') as _f:
                pickle.dump(models, _f)
            logger.info(f"GA {stage_name} models → {_pkl_path}")
        except Exception as _pe:
            logger.warning(f"GA {stage_name} pkl save failed: {_pe}")
        # Write HTML coefficient table from the CSV data
        _html_path = os.path.join(OUT_DIR, f'{stage_name}_coefficients.html')
        try:
            _csv_path = os.path.join(OUT_DIR, f'{stage_name}_coefficients.csv')
            if os.path.exists(_csv_path):
                import pandas as _pd_html
                _df_coef = _pd_html.read_csv(_csv_path)
                def _p_style(p):
                    try:
                        pf = float(p)
                        if pf < 0.001: return "color:#1a7a1a;font-weight:bold"
                        if pf < 0.01:  return "color:#2e7d32;font-weight:bold"
                        if pf < 0.05:  return "color:#558b2f"
                        if pf < 0.10:  return "color:#f57c00"
                        return "color:#b71c1c"
                    except Exception:
                        return ""
                rows_html = ""
                for _, row in _df_coef.iterrows():
                    p = row.get('p_value', row.get('pval', float('nan')))
                    style = _p_style(p)
                    cells = "".join(f"<td style='{style if c in ('p_value','pval','t_stat','t_stat_opg') else ''}'>{v}</td>"
                                    for c, v in row.items())
                    rows_html += f"<tr>{cells}</tr>"
                headers = "".join(f"<th>{c}</th>" for c in _df_coef.columns)
                html_body = (f"<table border='1' style='border-collapse:collapse;font-size:0.9em'>"
                             f"<thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>")
                with open(_html_path, 'w', encoding='utf-8') as _fh:
                    _fh.write(f"<html><body><h2>GA {stage_name} coefficients</h2>{html_body}</body></html>")
                logger.info(f"GA {stage_name} coefficients HTML → {_html_path}")
        except Exception as _he:
            logger.warning(f"GA {stage_name} HTML write failed: {_he}")

    try:
        if sid == 0:
            # Stage 0: CDAP feature search — write a simple JSON + text summary.
            # The CDAP model itself is saved via run_cdap's own cache path.
            import json as _json
            _s0_path = os.path.join(OUT_DIR, 'stage0_cdap_best_features.json')
            _s0_summary = {
                'best_features': list(best_feats) if best_feats else [],
                'best_fitness':  float(getattr(res, 'best_fitness', float('nan'))),
                'fitness_metric': str(getattr(res, 'fitness_metric', '')),
                'n_features':    len(best_feats) if best_feats else 0,
            }
            with open(_s0_path, 'w', encoding='utf-8') as _f:
                _json.dump(_s0_summary, _f, indent=2, default=str)
            logger.info(f"GA stage0 (CDAP) best features → {_s0_path}")
            # Also write a plain-text report alongside the checkpoint
            _s0_txt = os.path.join(OUT_DIR, 'stage0_cdap_report.txt')
            with open(_s0_txt, 'w', encoding='utf-8') as _f:
                _f.write(f"CDAP GA stage 0 — best feature set\n")
                _f.write(f"fitness: {_s0_summary['best_fitness']:.6f}  ({_s0_summary['fitness_metric']})\n")
                _f.write(f"n_features: {_s0_summary['n_features']}\n\n")
                for feat in (_s0_summary['best_features'] or []):
                    _f.write(f"  {feat}\n")
            logger.info(f"GA stage0 (CDAP) text report → {_s0_txt}")
        elif sid == 1:
            from stage1_activity_frequency import estimate_frequency_models
            models = estimate_frequency_models(
                out['tours'], out['persons'],
                feature_override=best_feats if isinstance(best_feats, list) else None,
                landuse=landuse,
            )
            _persist(models, 'stage1')
        elif sid == 23:
            # Joint Stage 2+3 Weibull AFT model — already fitted by ga_stage23_joint.
            # Re-persist from the metadata dict so coefficients land in OUT_DIR.
            _j23_models = (res.metadata or {}).get('stage23_joint_models') if res else None
            if _j23_models and _j23_models.get('success'):
                _j23_pkl = os.path.join(OUT_DIR, 'stage23_joint_models.pkl')
                with open(_j23_pkl, 'wb') as _f:
                    pickle.dump(_j23_models, _f)
                logger.info(f"GA stage23_joint models → {_j23_pkl}")
                # Also copy to saved_data/ for downstream consumption
                import shutil as _sh
                _sd_pkl = os.path.join('saved_data', 'stage23_joint_models.pkl')
                try:
                    _sh.copy2(_j23_pkl, _sd_pkl)
                    logger.info(f"GA stage23_joint models → {_sd_pkl}")
                except Exception as _ce:
                    logger.warning(f"stage23_joint copy to saved_data failed: {_ce}")
            else:
                logger.warning("_ga_save_stage_coef_html_from_best sid=23: no joint model in metadata")
        elif sid == 2:
            from stage2_day_duration import aggregate_person_day, estimate_day_models
            pd_df = aggregate_person_day(out['tours'], out['persons'])
            models = estimate_day_models(
                pd_df,
                feature_override=best_feats if isinstance(best_feats, list) else None,
            )
            _persist(models, 'stage2')
        elif sid == 3:
            from stage3_activity_duration import estimate_duration_models
            models = estimate_duration_models(
                out['tours'], out['persons'], trips=out.get('trips'),
                feature_override=best_feats if isinstance(best_feats, list) else None,
            )
            _persist(models, 'stage3')
        elif sid == 4:
            from stage4_schedule_placement import estimate_timing_models
            models = estimate_timing_models(
                out['trips'], out['tours'], out['persons'],
                feature_override=best_feats,
            )
            _persist(models, 'stage4')
        elif sid == 5:
            import openmatrix as omx
            from stage5_destination_choice import (
                _prep_hhts_tours, _filter_seg, _mode_tree, _build_mode_model,
                _estimate, _logsums, _attach_size_term, _attach_acdc_terms,
                _dest_tree, _build_dest_model, _skim_flat,
            )
            _skims_path = PIPELINE_SKIMS_PATH
            if not os.path.isabs(_skims_path):
                _skims_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _skims_path)
            _s5_mode_feats = (best_feats.get('mode') if isinstance(best_feats, dict)
                              else (best_feats or []))
            _s5_dest_feats = (best_feats.get('dest') if isinstance(best_feats, dict)
                              else (best_feats or []))
            _hhts_input = out.get('tours')
            if _hhts_input is None:
                _hhts_input = out.get('trips')
            # _prep_hhts_tours adds ptour_id (and standardises column names) which
            # _mode_tree / _dest_tree require.  Without this call, _mode_tree raises
            # KeyError: 'ptour_id' for every segment.
            try:
                _hhts_input = _prep_hhts_tours(_hhts_input, out['persons'])
            except Exception as _prep_e:
                logger.warning(f"GA stage 5 coef HTML: _prep_hhts_tours failed: {_prep_e}")
            _s5_models = {}
            _skims_hdl = omx.open_file(_skims_path, 'r')
            try:
                try:
                    _skim_df = _skim_flat(_skims_hdl)
                except Exception as _se:
                    import pandas as _pd
                    _skim_df = _pd.DataFrame()
                    logger.warning(f"GA stage 5 coef HTML: skim_flat failed: {_se}")
                _lu = landuse if landuse is not None else getattr(res, 'metadata', {}).get('landuse')
                for _seg in ['work', 'edu', 'nonmand']:
                    try:
                        _st = _filter_seg(_hhts_input, _seg)
                        if not isinstance(_st, (pd.DataFrame, type(None))):
                            _st = pd.DataFrame()
                        if isinstance(_st, pd.DataFrame) and len(_st) < 50:
                            continue
                        _dt_m = _mode_tree(_st, out['persons'], _lu, _skims_hdl)
                        _mode_mdl = _build_mode_model(_dt_m, _seg,
                                                       custom_mode_vars=_s5_mode_feats)
                        _mode_mdl = _estimate(_mode_mdl)
                        # Capture the mode model immediately — the dest re-estimate
                        # below can hit a sharrow/numba flow-compile error on the
                        # logsum/walk_feasible column (1D indexed 2D) for some
                        # segments; that must NOT drop the mode coefficients too.
                        _s5_models[f'{_seg}_mode'] = _mode_mdl
                        try:
                            _ls = _logsums(_mode_mdl, _dt_m, skims=_skims_hdl, verbose=False)
                            _lu_st = _attach_size_term(_lu, _seg)
                            _lu_st, _seg_acdc = _attach_acdc_terms(_lu_st, _st, _skim_df, _seg)
                            _dt_d = _dest_tree(_seg_acdc, out['persons'], _lu_st, _skims_hdl, _ls)
                            _dest_mdl = _build_dest_model(_dt_d, _seg,
                                                           custom_dest_vars=_s5_dest_feats)
                            _dest_mdl = _estimate(_dest_mdl)
                            _s5_models[f'{_seg}_dest'] = _dest_mdl
                        except Exception as _dest_exc:
                            logger.warning(
                                f"GA stage 5 coef HTML [{_seg}]: dest re-estimate "
                                f"skipped (mode report kept); non-fatal — the saved "
                                f"stage5_models.pkl has the real models: {_dest_exc}")
                    except Exception as _seg_exc:
                        logger.warning(f"GA stage 5 coef HTML [{_seg}]: {_seg_exc}")
            finally:
                _skims_hdl.close()
            if _s5_models:
                # Convert flat {seg_mode, seg_dest} → nested {seg: {mode_model, dest_model}}
                # required by save_models() and _write_smoke_models_html()
                _s5_nested = {}
                for _seg in ['work', 'edu', 'nonmand']:
                    _mm = _s5_models.get(f'{_seg}_mode')
                    _dm = _s5_models.get(f'{_seg}_dest')
                    if _mm is not None or _dm is not None:
                        _s5_nested[_seg] = {'mode_model': _mm, 'dest_model': _dm}
                if _s5_nested:
                    from stage5_destination_choice import save_models as _s5_save
                    _s5_pkl = os.path.join(OUT_DIR, 'stage5_models.pkl')
                    try:
                        _s5_save(_s5_nested, _s5_pkl)
                        logger.info(f"GA stage5 models → {_s5_pkl}")
                    except Exception as _sp:
                        logger.warning(f"GA stage5 save_models failed: {_sp}")
                        try:
                            with open(_s5_pkl, 'wb') as _f:
                                pickle.dump(_s5_nested, _f)
                        except Exception:
                            pass
                    try:
                        _write_smoke_models_html(
                            _s5_nested, OUT_DIR,
                            title='GA Stage 5 \u2014 Best Feature Model Coefficients',
                        )
                    except Exception as _sh:
                        logger.warning(f"GA stage5 HTML write failed: {_sh}")
            # Fallback: always write a basic CSV from GA best-features summary
            # even when model re-estimation fails, so downstream consumers can
            # see what features were selected.
            try:
                _s5_basic_rows = []
                if isinstance(best_feats, dict):
                    for _k, _v in best_feats.items():
                        _s5_basic_rows.append({'segment': str(_k), 'param': 'GA_selected',
                                                'value': ', '.join(map(str, _v)) if isinstance(_v, (list, tuple)) else str(_v)})
                elif isinstance(best_feats, (list, tuple, set)) and best_feats:
                    _s5_basic_rows.append({'segment': 'all', 'param': 'GA_selected',
                                            'value': ', '.join(map(str, best_feats))})
                if _s5_basic_rows:
                    import pandas as _pd
                    _basic_csv = os.path.join(OUT_DIR, 'stage5_coefficients.csv')
                    _pd.DataFrame(_s5_basic_rows).to_csv(_basic_csv, index=False)
                    logger.info(f"GA stage5 feature summary CSV → {_basic_csv}")
                    _html_path = os.path.join(OUT_DIR, 'stage5_coefficients.html')
                    try:
                        _pd.DataFrame(_s5_basic_rows).to_html(_html_path, index=False,
                            border=1, justify='left',
                            classes='table', escape=True)
                        logger.info(f"GA stage5 feature summary HTML → {_html_path}")
                    except Exception:
                        pass
            except Exception as _fb:
                logger.warning(f"GA stage5 fallback summary write failed: {_fb}")
    except Exception as _exc:
        logger.warning(f"GA stage {sid} coef HTML from best features failed: {_exc}")


def _run_ga_stages_with_checkpoints(out, landuse):
    """Run GA stages sequentially with after-stage checkpointing.

    Supports resume: if ga_results_checkpoint.pkl already contains completed
    stages (e.g. from a prior PBS run that was killed mid-way), those stages
    are reloaded and skipped so the job continues from where it left off.
    Per-purpose progress within each stage is checkpointed via the
    GA_CHECKPOINT_DIR env var, which is read by the stage-level run functions
    in ga_hypothesis_testing.py.
    """
    from ga_hypothesis_testing import AllStagesResult

    # ── Resume detection ──────────────────────────────────────────────────
    _ckpt_path = os.path.join(OUT_DIR, 'ga_results_checkpoint.pkl')
    all_ga = AllStagesResult()
    _skip_stages: set = set()
    if os.path.exists(_ckpt_path):
        try:
            with open(_ckpt_path, 'rb') as _f:
                _prev = pickle.load(_f)
            _prev_completed = set(
                int(k) for k in getattr(_prev, 'results', {}).keys()
            )
            if _prev_completed:
                all_ga = _prev
                _skip_stages = _prev_completed
                logger.info(
                    f"Resuming GA: reloaded checkpoint with completed stages "
                    f"{sorted(_skip_stages)} from {_ckpt_path}"
                )
        except Exception as _e:
            logger.warning(f"Could not load GA checkpoint for resume: {_e} — starting fresh")
    if not _skip_stages:
        _save_ga_checkpoint(all_ga, stage_id=0, failed=False)

    # Expose OUT_DIR to stage functions so per-purpose checkpoints can be
    # saved incrementally (and reloaded on resume).
    os.environ['GA_CHECKPOINT_DIR'] = OUT_DIR

    _stages = [
        (0, lambda: ga_stage0(out)),
        (1, lambda: ga_stage1(out, landuse)),
        (2, lambda: ga_stage2(out)),
        (3, lambda: ga_stage3(out)),
        (4, lambda: ga_stage4(out)),
        (5, lambda: ga_stage5(out, landuse)),
    ]

    for sid, runner in _stages:
        if sid in _skip_stages:
            logger.info(f"GA stage {sid} skipped (results loaded from checkpoint)")
            continue
        try:
            all_ga.results[sid] = runner()
            _res = all_ga.results[sid]
            _hist = getattr(_res, 'history', []) if _res is not None else []
            _hist_df = pd.DataFrame(_hist) if _hist else None

            _seg_results = {}
            if _res is not None and isinstance(getattr(_res, 'metadata', None), dict):
                _seg_results = _res.metadata.get('segment_results', {}) or {}
            if _seg_results:
                for _seg_label, _seg_res in _seg_results.items():
                    _seg_hist = getattr(_seg_res, 'history', []) or []
                    if _seg_hist:
                        _hdf = pd.DataFrame(_seg_hist)
                        _best = _hdf.get('best_so_far', _hdf.get('best_fitness', pd.Series(dtype=float)))
                        _div = _hdf.get('diversity', pd.Series(dtype=float))
                        _n_gen = int(len(_hdf))
                        _div_max = float(pd.to_numeric(_div, errors='coerce').max()) if len(_div) else float('nan')
                        _impr = float(_best.iloc[-1] - _best.iloc[0]) if len(_best) > 1 else 0.0
                        logger.info(
                            f"GA stage {sid} segment '{_seg_label}': gens={_n_gen} "
                            f"max_diversity={_div_max:.3f} best_so_far_delta={_impr:.6f}"
                        )
                    else:
                        logger.info(f"GA stage {sid} segment '{_seg_label}': no history rows recorded")
            elif _hist_df is not None and not _hist_df.empty:
                _best = _hist_df.get('best_so_far', _hist_df.get('best_fitness', pd.Series(dtype=float)))
                _div = _hist_df.get('diversity', pd.Series(dtype=float))
                _impr = float(_best.iloc[-1] - _best.iloc[0]) if len(_best) > 1 else 0.0
                _div_max = float(pd.to_numeric(_div, errors='coerce').max()) if len(_div) else float('nan')
                logger.info(
                    f"GA stage {sid}: gens={len(_hist_df)} max_diversity={_div_max:.3f} "
                    f"best_so_far_delta={_impr:.6f}"
                )
            
            # Build enhanced summary with model details and proper BIC reporting
            summary = _format_ga_checkpoint_summary(_res)
            
            _write_stage_checkpoint_report(
                stage_tag=f'ga_stage{sid}',
                title=f'GA Stage {sid} - Selected Model & Convergence',
                summary=summary,
                table=_hist_df,
            )
            # Re-estimate with the best features and save coefficient HTML
            # so users can inspect named coefficients in the GA search run dir.
            _ga_save_stage_coef_html_from_best(sid, _res, out, landuse=landuse)
            _save_ga_checkpoint(all_ga, stage_id=sid, failed=False)
            logger.info(
                f"GA checkpoint saved after stage {sid} "
                f"({len(all_ga.results)} stage(s) complete)"
            )
        except Exception:
            _save_ga_checkpoint(all_ga, stage_id=sid, failed=True)
            logger.error(
                f"GA failed during stage {sid}. "
                "Partial results were checkpointed to ga_results.pkl"
            )
            raise

    return all_ga


def _ga_budget_params() -> dict:
    """
    Read GA_BUDGET env var and return pop/gens/restarts.

    Total evaluation steps = pop_size * (n_generations + n_restarts * (n_generations // 3)).
    All levels below have been tuned so that reducing restarts redistributes those
    steps back into the main run (same total compute, longer uninterrupted convergence).

    GA_BUDGET=smoke      pop=8,   gens=2,   restarts=0  (sanity check only)
    GA_BUDGET=ultrafast  pop=25,  gens=30,  restarts=0  total_gens_equiv=30  (~45 min/purpose for MetaCount)
    GA_BUDGET=fast       pop=40,  gens=75,  restarts=1  total_gens_equiv=100
    GA_BUDGET=standard   pop=60,  gens=263, restarts=1  total_gens_equiv=350
    GA_BUDGET=thorough   pop=100, gens=540, restarts=2  total_gens_equiv=900

    Fewer restarts means the SA-Bandit main run has more generations to warm up
    the temperature schedule and sharpen the Beta posteriors before being reset.
    One restart is kept (fast/standard) to recover from bad random initialisations.

    For Stage 5 specifically, each fitness eval fits a Larch NL model (~2-5 s).
    Stages 1-4 are much faster (seconds/eval); budget only matters for Stage 5.
    """
    level = os.environ.get('GA_BUDGET', 'standard').strip().lower()
    budgets = {
        'smoke':     dict(pop_size=8,   n_generations=2,   n_restarts=0),
        'ultrafast': dict(pop_size=25,  n_generations=30,  n_restarts=0),
        'fast':      dict(pop_size=40,  n_generations=75,  n_restarts=1),
        'standard':  dict(pop_size=60,  n_generations=263, n_restarts=1),
        'thorough':  dict(pop_size=100, n_generations=540, n_restarts=2),
    }
    if level not in budgets:
        logger.warning(f"Unknown GA_BUDGET='{level}'; using 'standard'. "
                   f"Valid choices: smoke, ultrafast, fast, standard, thorough")
        level = 'standard'
    params = dict(budgets[level])

    est = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    # MetaCount / SearchLibrium evaluations are 5-30× more expensive than
    # default statsmodels, so using the standard budget (>15k evals per
    # purpose) causes multi-day runs.  We override the budget level to
    # 'ultrafast' (~750 evals) unless the user explicitly chose a budget.
    # Set GA_BUDGET to 'fast' or 'standard' to opt in to heavier search.
    if est in ('metacount', 'searchlibrium', 'both', 'hybrid'):
        _user_budget = os.environ.get('GA_BUDGET', '').strip().lower()
        if not _user_budget:
            os.environ['GA_BUDGET'] = 'ultrafast'
            params = dict(budgets['ultrafast'])
            logger.info(
                f"GA estimator={est!r}: MetaCount is 5-30× slower per eval; "
                f"auto-setting GA_BUDGET='ultrafast' "
                f"(pop={params['pop_size']} gens={params['n_generations']} "
                f"restarts={params['n_restarts']}).  "
                f"Set GA_BUDGET=fast/standard/thorough explicitly to override."
            )
        else:
            logger.info(
                f"GA estimator={est!r}: using user-requested budget '{_user_budget}' "
                f"pop={params['pop_size']} gens={params['n_generations']} "
                f"restarts={params['n_restarts']}"
            )

    # Optional hard override for restart count (useful for uninterrupted SA-bandit runs).
    _n_restarts_env = os.environ.get('GA_N_RESTARTS', '').strip()
    if _n_restarts_env:
        try:
            params['n_restarts'] = max(0, int(_n_restarts_env))
        except ValueError:
            logger.warning(f"Ignoring invalid GA_N_RESTARTS={_n_restarts_env!r}; expected integer >= 0")

    logger.info(f"GA budget level: {level!r} → pop={params['pop_size']}  "
                f"gens={params['n_generations']}  restarts={params['n_restarts']}")
    return params


def _ga_post_estimation_report(s5_models: dict, out: dict, save_dir: str) -> None:
    """
    Called after apply_ga_results_to_pipeline in GA_main and GA_STAGED.

    Saves:
    - stage5_models_coefficients.html (if not already saved by apply_ga_results_to_pipeline)
    - stage5_tour_param_diagnostics.html  (p-value / zero-coeff inspection)
    - ga_hhts_validation/                 (full in-sample HHTS validation for all stages)

    All of these are already generated inside apply_ga_results_to_pipeline itself,
    but this function ensures the pipeline_logger OUT_DIR always has them too (the
    GA function saves to save_dir=OUT_DIR which matches here).
    """
    logger.info("  [GA post-estimation] running coefficient diagnostics...")
    _inspect_model_parameters(s5_models, 'stage5_ga_tour', save_dir)

    # Log a compact summary of tour model fit (LL + rho² per segment)
    logger.info("  [GA] Stage 5 tour model summary:")
    for seg, res in s5_models.items():
        if not isinstance(res, dict):
            continue
        for mtype in ('mode', 'dest'):
            ll  = res.get(f'{mtype}_ll', float('nan'))
            rho2 = float('nan')
            mdl  = res.get(f'{mtype}_model')
            if mdl is not None:
                try:
                    rho2 = float(mdl.rho_sq_null())
                except Exception:
                    pass
            logger.info(
                f"    [{seg}/{mtype}] LL={ll:.1f}  ρ²={rho2:.3f}"
                if np.isfinite(rho2) else
                f"    [{seg}/{mtype}] LL={ll:.1f}  ρ²=n/a"
            )

    # ── HHTS in-sample validation with GA-selected models ─────────────────
    # Load all stage models saved by apply_ga_results_to_pipeline into
    # save_dir/models/, then run the full run_validation() for obs vs pred
    # comparison plots before the synthetic population is applied.
    try:
        import pickle as _pkl_val
        from validate_hhts import run_validation as _run_val

        models_dir = os.path.join(save_dir, 'models')
        _load_pkl = lambda fn: _pkl_val.load(open(os.path.join(models_dir, fn), 'rb')) \
            if os.path.isfile(os.path.join(models_dir, fn)) else {}

        s1_m = _load_pkl('stage1_models.pkl')
        s2_m = _load_pkl('stage2_models.pkl')
        s3_m = _load_pkl('stage3_models.pkl')
        s4_m = _load_pkl('stage4_models.pkl')
        # s5_models already passed in — use directly (avoids re-loading large pkl)
        s5_m = s5_models or _load_pkl('stage5_models.pkl')

        if any([s1_m, s2_m, s3_m, s4_m, s5_m]):
            logger.info("  [GA post-estimation] running HHTS in-sample validation...")
            _val_out_dir = os.path.join(save_dir, 'ga_hhts_validation')
            os.makedirs(os.path.join(_val_out_dir, 'plots', 'validation'), exist_ok=True)
            _run_val(
                out       = out,
                s1_models = s1_m,
                s2_models = s2_m,
                s3_models = s3_m,
                s4_models = s4_m,
                s5_models = s5_m,
                out_dir   = _val_out_dir,
            )
            logger.info(f"  [GA] HHTS validation → {_val_out_dir}/plots/validation/")
        else:
            logger.warning(
                "  [GA post-estimation] no stage models found in models/ directory; "
                "skipping HHTS validation (apply_ga_results_to_pipeline may not have run yet)"
            )
    except Exception as _ve:
        import traceback as _tb_val
        logger.warning(f"  [GA post-estimation] HHTS validation failed: {_ve}")
        logger.debug(_tb_val.format_exc())


def _ga_plot_stage_result(res, save_path: str) -> None:
    """
    Plot GA convergence for one stage, unfolding per-segment sub-results.

    Majority-vote results from _run_ga_stageN store a synthetic single-entry
    aggregate history (to avoid showing inflated per-segment values in one
    combined line) and put the actual per-segment GAResult objects in
    res.metadata['segment_results'].  This helper detects that structure and
    renders a multi-panel figure — one panel per segment — so the convergence
    history is not blank.
    """
    import math as _math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seg_results = {}
    if hasattr(res, 'metadata') and isinstance(res.metadata, dict):
        seg_results = res.metadata.get('segment_results', {})

    if seg_results:
        plot_items = list(seg_results.items())   # [(label, GAResult), ...]
    else:
        plot_items = [('', res)]

    n    = len(plot_items)
    cols = min(3, n)
    rows = _math.ceil(n / max(cols, 1))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(7 * cols, 4.5 * rows),
                             facecolor='white', squeeze=False)
    axes_flat = axes.flatten()
    for i, (label, r) in enumerate(plot_items):
        if i < len(axes_flat):
            r.plot_convergence(ax=axes_flat[i], segment_label=label)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  GA plot saved → {save_path}  ({n} segment(s))")


@timer("GA — Stage 0 (CDAP activity pattern)")
def ga_stage0(out):
    from census_bridge import run_ga_cdap
    _bp = _ga_budget_params()
    res = run_ga_cdap(out["persons"], **_bp)
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage0.png'))
    return res


@timer("GA — Stage 1 (activity frequency)")
def ga_stage1(out, landuse=None):
    from census_bridge import run_ga_stage1
    # MULTIVARIATE option: GA_ESTIMATOR=metacount_mv also fits MetaCount's joint
    # multivariate activity-count model (per-purpose counts estimated together
    # with a copula) and writes stage1_multivariate_* reports to OUT_DIR. It runs
    # alongside the normal per-purpose search (which still produces the models
    # prediction needs); it's fully guarded so it can't break the stage.
    if os.environ.get('GA_ESTIMATOR', '').strip().lower() == 'metacount_mv':
        try:
            from stage1_multivariate import fit_stage1_multivariate
            logger.info("Stage 1: fitting MetaCount multivariate activity-count model "
                        "(GA_ESTIMATOR=metacount_mv) ...")
            fit_stage1_multivariate(out["tours"], out["persons"], landuse=landuse,
                                    out_dir=OUT_DIR)
        except Exception as _mv_e:
            logger.error(f"Stage 1 multivariate fit skipped: {_mv_e!r}")
    _bp = _ga_budget_params()
    res = run_ga_stage1(out["tours"], out["persons"], landuse=landuse, **_bp)
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage1.png'))
    return res


@timer("GA — Stage 2 (day duration)")
def ga_stage2(out):
    from census_bridge import run_ga_stage2
    _bp = _ga_budget_params()
    res = run_ga_stage2(out["tours"], out["persons"], **_bp)
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage2.png'))
    return res


@timer("GA — Stage 3 (activity duration)")
def ga_stage3(out):
    from census_bridge import run_ga_stage3
    _bp = _ga_budget_params()
    res = run_ga_stage3(out["tours"], out["persons"], trips=out["trips"], **_bp)
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage3.png'))
    return res


@timer("GA — Stage 2+3 Joint (budget-constrained)")
def ga_stage23_joint(out):
    """
    Joint SA-Bandit search over a shared Stage 2+3 feature pool.

    Activated via STAGE23_JOINT=1.  Replaces the separate ga_stage2 + ga_stage3
    calls with a single metaheuristic search that:
      1. Optimises a shared feature set against a fast OLS-BIC joint oracle
         (stage23_fitness_joint — <2 ms per eval, ~3000 evals in <5 min)
      2. After convergence, runs fit_joint_stage23 once with the winning features
         to produce the final JAX-optimised budget-constrained model.

    Saves stage23_joint_models.pkl and stage23_joint_summary.txt to OUT_DIR.
    """
    import pickle
    from ga_hypothesis_testing import _run_ga_stage23_joint
    _bp = _ga_budget_params()
    _n_restarts = _bp.pop('n_restarts', 2)
    res = _run_ga_stage23_joint(
        tours=out["tours"], persons=out["persons"],
        trips=out.get("trips"),
        ga_kwargs=_bp,
        n_restarts=_n_restarts,
    )
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage23_joint.png'))

    # Persist the final JAX joint model
    final_models = (res.metadata or {}).get('stage23_joint_models')
    if final_models:
        pkl_path = os.path.join(OUT_DIR, 'stage23_joint_models.pkl')
        with open(pkl_path, 'wb') as fh:
            pickle.dump(final_models, fh)
        logger.info(f"Joint Stage 2+3 model → {pkl_path}")

        summary_path = os.path.join(OUT_DIR, 'stage23_joint_summary.txt')
        from stage23_joint import joint_model_summary
        with open(summary_path, 'w') as fh:
            fh.write(joint_model_summary(final_models))

    out['stage23_joint_models'] = final_models
    return res


@timer("GA — Stage 4 (schedule placement)")
def ga_stage4(out):
    from census_bridge import run_ga_stage4
    _bp = _ga_budget_params()
    res = run_ga_stage4(out["trips"], out["tours"], out["persons"], **_bp)
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage4.png'))
    return res


@timer("GA — Stage 5 (skim + landuse)")
def ga_stage5(out, landuse):
    from census_bridge import run_ga_stage5
    _bp = _ga_budget_params()
    res = run_ga_stage5(
        trips=out["trips"], persons=out["persons"],
        landuse=landuse,
        skims=PIPELINE_SKIMS_PATH,
        tours=out.get("tours"),
        **_bp,
    )
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage5.png'))
    print(f"  mode:    {res.best_features.get('mode')}")
    print(f"  dest:    {res.best_features.get('dest')}")
    return res


def _detect_stage5_crash_source() -> Optional[str]:
    """Prefer an OMX with CRASH matrix, then fallback to tabular crash files."""
    import openmatrix as omx

    omx_candidates = [
        'downcasted_matrices_base.omx',
    ]
    for path in omx_candidates:
        if not os.path.isfile(path):
            continue
        try:
            with omx.open_file(path, 'r') as cf:
                if 'CRASH' in cf.listMatrices():
                    return path
        except Exception as exc:
            logger.warning(f"Could not inspect crash OMX source {path}: {exc}")

    crash_candidates = [
        'OD_Crashes.csv',
        'SA2_OD_Crash_Matrix_PT.xlsx',
        'SA2_OD_Crash_Matrix.xlsx',
    ]
    return next((path for path in crash_candidates if os.path.isfile(path)), None)


def _write_smoke_models_html(models: dict, save_dir: str, title: str = 'Stage 5 Smoke — Model Coefficients') -> None:
        """Write a full HTML coefficient report for all smoke-test stage 5 models."""
        try:
                from stage5_destination_choice import _coef_dict_from_model
        except ImportError:
                return
        from scipy.stats import norm as _sn

        def _fmt(v, fmt='.4f'):
                try:
                        f = float(v)
                        return format(f, fmt) if np.isfinite(f) else '—'
                except Exception:
                        return '—'

        def _pval_from_t(t):
                try:
                        return float(2.0 * _sn.sf(abs(float(t))))
                except Exception:
                        return float('nan')

        def _p_style(p):
                try:
                        pf = float(p)
                        if not np.isfinite(pf): return ''
                        if pf < 0.001: return "color:#1a7a1a;font-weight:bold"
                        if pf < 0.01:  return "color:#2e7d32;font-weight:bold"
                        if pf < 0.05:  return "color:#558b2f"
                        if pf < 0.10:  return "color:#f57c00"
                        return "color:#b71c1c"
                except Exception:
                        return ''

        def _coef_rows_html(mdl):
                try:
                        coef = _coef_dict_from_model(mdl)
                        pf = None
                        try:
                                _pf = getattr(mdl, '_final_pf', None)
                                if isinstance(_pf, pd.DataFrame):
                                        pf = _pf
                        except Exception:
                                pass
                        if pf is None:
                                try:
                                        _pf2 = mdl.pf
                                        if isinstance(_pf2, pd.DataFrame):
                                                pf = _pf2
                                except Exception:
                                        pass
                        params = list(pf.index) if isinstance(pf, pd.DataFrame) and len(pf) > 0 else list(coef.keys())
                        rows = ""
                        for p in params:
                                val = coef.get(str(p), float('nan'))
                                se = tstat = pval = float('nan')
                                if pf is not None and p in pf.index:
                                        row = pf.loc[p]
                                        for c in ('std_err', 'std_err_opg', 'se'):
                                                if c in row.index and np.isfinite(float(row[c])): se = float(row[c]); break
                                        for c in ('t_stat', 't_stat_opg', 'tstat'):
                                                if c in row.index and np.isfinite(float(row[c])): tstat = float(row[c]); break
                                        for c in ('pval', 'p_value', 'pvalue', 'pval_opg'):
                                                if c in row.index and np.isfinite(float(row[c])): pval = float(row[c]); break
                                if np.isnan(tstat) and np.isfinite(se) and se > 0:
                                        tstat = val / se
                                if np.isnan(pval) and np.isfinite(tstat):
                                        pval = _pval_from_t(tstat)
                                t_style = _p_style(pval)
                                p_style = _p_style(pval)
                                rows += (
                                        f"<tr>"
                                        f"<td style='{t_style}'>{p}</td>"
                                        f"<td style='text-align:right'>{_fmt(val)}</td>"
                                        f"<td style='text-align:right'>{_fmt(se)}</td>"
                                        f"<td style='text-align:right;{t_style}'>{_fmt(tstat, '.2f')}</td>"
                                        f"<td style='text-align:right;{p_style}'>{_fmt(pval, '.4f')}</td>"
                                        f"</tr>"
                                )
                        return rows
                except Exception as e:
                        return f"<tr><td colspan='5'>Could not render: {e}</td></tr>"

        def _model_stats_html(mdl, ll, seg, mtype):
                """Extract rho2 and n_obs from live model attributes."""
                try:
                        rho2 = getattr(mdl, 'rho2_', float('nan'))
                        if not np.isfinite(rho2):
                                rho2 = getattr(mdl, '_rho2', float('nan'))
                except Exception:
                        rho2 = float('nan')
                try:
                        n = getattr(mdl, 'n_obs_', None) or getattr(mdl, '_n_obs', None)
                        if n is None and hasattr(mdl, 'pf'):
                                n = '?'
                except Exception:
                        n = '?'
                ll_s   = _fmt(ll,   '.2f')
                rho2_s = _fmt(rho2, '.4f')
                try:
                        try:
                                _pf = getattr(mdl, '_final_pf', None)
                        except Exception:
                                _pf = None
                        if isinstance(_pf, pd.DataFrame):
                                pf = _pf
                        else:
                                try:
                                        pf = mdl.pf
                                except Exception:
                                        pf = None
                        if isinstance(pf, pd.DataFrame) and 'holdfast' in pf.columns:
                                n_free = int((pf['holdfast'].fillna(True).astype(bool) == False).sum())
                        else:
                                n_free = '?'
                except Exception:
                        n_free = '?'
                return (f"<p class='stats'>"
                        f"<b>Log-likelihood:</b> {ll_s} &nbsp;|&nbsp;"
                        f"<b>McFadden ρ²:</b> {rho2_s} &nbsp;|&nbsp;"
                        f"<b>N obs:</b> {n} &nbsp;|&nbsp;"
                        f"<b>Free params:</b> {n_free}"
                        f"</p>")

        sections = ""
        for seg, res in models.items():
                if not isinstance(res, dict):
                        continue
                n = res.get('n_tours', '?')
                sections += (
                        f"<div class='seg-header'>"
                        f"<h2>Segment: {seg.upper()}</h2>"
                        f"<p>HHTS estimation tours: <b>{n}</b></p>"
                        f"</div>"
                )
                for mtype, mdl_key, ll_key in [
                        ('Destination choice', 'dest_model', 'dest_ll'),
                        ('Mode choice',        'mode_model', 'mode_ll'),
                ]:
                        mdl = res.get(mdl_key)
                        ll  = float(res.get(ll_key, float('nan')))
                        if mdl is None:
                                sections += f"<div class='model-section'><h3>{mtype} — not estimated</h3></div>"
                                continue
                        sections += (
                                f"<div class='model-section'>"
                                f"<h3>{mtype} &nbsp; <span class='seg-tag'>{seg}</span></h3>"
                                + _model_stats_html(mdl, ll, seg, mtype) +
                                "<table><thead><tr>"
                                "<th>Parameter</th><th>Value</th><th>Std&nbsp;Err</th>"
                                "<th>t-stat</th><th>p-value</th>"
                                "</tr></thead><tbody>"
                                + _coef_rows_html(mdl) +
                                "</tbody></table>"
                                "<p class='sig-note'>"
                                "<span style='color:#1a7a1a;font-weight:bold'>p&lt;0.001</span> "
                                "<span style='color:#2e7d32;font-weight:bold'>p&lt;0.01</span> "
                                "<span style='color:#558b2f'>p&lt;0.05</span> "
                                "<span style='color:#f57c00'>p&lt;0.10</span> "
                                "<span style='color:#b71c1c'>p&ge;0.10 (not significant)</span>"
                                "</p></div>"
                        )

        os.makedirs(save_dir, exist_ok=True)
        html_path = os.path.join(save_dir, 'stage5_smoke_models_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html lang='en'><head><meta charset='utf-8'>
<title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:30px;font-size:13px;max-width:1300px;color:#222}}
h1{{color:#1a3a5c;border-bottom:3px solid #1a3a5c;padding-bottom:10px;font-size:1.6rem}}
h2{{color:#1565c0;margin:32px 0 4px;font-size:1.2rem}}
h3{{color:#2c5f8a;margin:0 0 6px;font-size:1rem}}
.seg-header{{background:#e3f2fd;border-left:5px solid #1565c0;padding:10px 16px;margin:28px 0 0;border-radius:4px}}
.model-section{{border:1px solid #ccc;border-radius:6px;padding:16px;margin:10px 0 20px}}
.seg-tag{{background:#1565c0;color:white;font-size:0.75rem;padding:2px 8px;border-radius:10px;vertical-align:middle;margin-left:6px}}
.stats{{background:#e8f5e9;padding:8px 14px;border-radius:4px;font-size:0.92em;margin:6px 0 12px;border-left:4px solid #43a047}}
.sig-note{{font-size:0.78em;color:#666;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}}
th{{background:#1565c0;color:white;padding:7px 10px;text-align:left;white-space:nowrap}}
td{{padding:5px 10px;border-bottom:1px solid #eee}}
tr:hover td{{background:#f5f9ff}}
tr:nth-child(even) td{{background:#fafafa}}
tr:nth-child(even):hover td{{background:#f0f4ff}}
</style></head><body>
<h1>{title}</h1>
<p style='color:#555;font-size:12px'>
  Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  All segments: work &middot; edu &middot; nonmand &nbsp;|&nbsp;
  All model types: destination choice &middot; mode choice
</p>
{sections}
</body></html>""")
        logger.info(f"  Smoke model coefficient HTML -> {html_path}")


def _fit_stage5_smoke_models(
        out: dict,
        landuse: pd.DataFrame,
        save_dir: str,
        mode_feature_override: Optional[list] = None,
        dest_feature_override: Optional[list] = None,
        safety_mode: str = 'baseline',
        use_gpu: bool = False,
) -> dict:
        """Estimate Stage 5 HHTS models for the smoke run and write main-run style artifacts."""
        import openmatrix as omx
        from stage5_destination_choice import (
                _skim_flat,
                _prep_hhts_tours,
                _run_segment,
                _save_coefficients,
                save_models as _save_stage5_models,
                save_location_choice_summary,
        )

        os.makedirs(save_dir, exist_ok=True)
        skims_path = PIPELINE_SKIMS_PATH
        crash_skims = _detect_stage5_crash_source()

        prev_safety_mode = os.environ.get('SAFETY_EXPERIMENT_MODE')
        os.environ['SAFETY_EXPERIMENT_MODE'] = safety_mode

        # GPU: force JAX engine (XLA can use CUDA if available) and prevent
        # pre-allocation to avoid OOM on shared HPC nodes.
        prev_larch_engine = os.environ.get('GA_LARCH_ENGINE')
        if use_gpu:
                os.environ['GA_LARCH_ENGINE'] = 'jax'
                os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
                logger.info(f"Stage 5 smoke [{safety_mode}]: GPU mode — forcing JAX engine (XLA/CUDA)")

        skims = None
        try:
            skims = omx.open_file(skims_path, 'r')
            skim_df = _skim_flat(skims)

            _hhts_tours_src = out['tours']
            if 'tour_mode' not in _hhts_tours_src.columns or _hhts_tours_src['tour_mode'].isna().all():
                    _mode_col = next(
                            (c for c in ['mode', 'MAINMODE', 'TRAVMODE', 'LINKMODE'] if c in out['trips'].columns),
                            None,
                    )
                    if _mode_col:
                            _dom = (
                                    out['trips']
                                    .groupby(['person_id', 'tour_id'])[_mode_col]
                                    .agg(lambda x: x.value_counts().idxmax() if len(x) else 'DA')
                                    .reset_index()
                                    .rename(columns={_mode_col: 'tour_mode'})
                            )
                            _hhts_tours_src = _hhts_tours_src.copy()
                            if 'tour_mode' in _hhts_tours_src.columns:
                                    _hhts_tours_src = _hhts_tours_src.drop(columns=['tour_mode'])
                            _hhts_tours_src = _hhts_tours_src.merge(_dom, on=['person_id', 'tour_id'], how='left')

            hhts_tours = _prep_hhts_tours(_hhts_tours_src, out['persons'])
            logger.info(
                    f"Stage 5 smoke [{safety_mode}]: estimating selected models for {len(hhts_tours):,} HHTS tours"
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
                            logger.warning(f"Stage 5 smoke [{safety_mode}] failed for {seg}: {exc}")

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
                    title=f'Stage 5 Smoke [{safety_mode}] — Model Coefficients',
            )
            logger.info(f"Stage 5 smoke [{safety_mode}] artifacts saved -> {save_dir}")
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
                if use_gpu:
                        if prev_larch_engine is None:
                                os.environ.pop('GA_LARCH_ENGINE', None)
                        else:
                                os.environ['GA_LARCH_ENGINE'] = prev_larch_engine


def _write_interactive_safety_dashboard(
        baseline_dir: str,
        nosafety_dir: str,
        out_dir: str,
        landuse: Optional[pd.DataFrame] = None,
        comparison_df: Optional[pd.DataFrame] = None,
        hhts_obs: Optional[pd.DataFrame] = None,
        dest_zone_shift: Optional[list] = None,
) -> Optional[str]:
        """Create an interactive HTML dashboard for the Stage 5 safety comparison."""
        try:
                import geopandas as gpd
                from stage5_destination_choice import _read_crash_od_table
        except Exception as exc:
                logger.warning(f"Interactive safety dashboard skipped: {exc}")
                return None

        def _load_location_summary(run_dir: str) -> pd.DataFrame:
                path = os.path.join(run_dir, 'location_choice', 'location_choice_summary.csv')
                if os.path.exists(path):
                        return pd.read_csv(path)
                return pd.DataFrame()

        def _clean_for_json(df: pd.DataFrame) -> list[dict]:
                if df is None or df.empty:
                        return []
                clean = df.where(pd.notna(df), None)
                return clean.to_dict(orient='records')

        coef_base = _load_location_summary(baseline_dir)
        coef_nosa = _load_location_summary(nosafety_dir)
        if coef_base.empty:
                logger.warning("Interactive safety dashboard skipped: baseline location_choice summary not found")
                return None

        if landuse is None or getattr(landuse, 'empty', True):
                candidate_landuse_paths = [
                        os.path.join('saved_data', 'accessibility.csv'),
                        'pddopulation_by_order_with_osm_and_landuse.csv',
                ]
                for candidate in candidate_landuse_paths:
                        if os.path.exists(candidate):
                                try:
                                        landuse = pd.read_csv(candidate)
                                        break
                                except Exception:
                                        continue
        if landuse is None or getattr(landuse, 'empty', True):
                logger.warning("Interactive safety dashboard skipped: landuse data unavailable")
                return None

        zone_map = pd.read_csv('seq_zone_ids_.csv')

        def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        def _normalize_sa2_code(series: pd.Series) -> pd.Series:
            s = series.astype(str).str.strip()
            s = s.str.replace(r'\.0$', '', regex=True)
            s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
            return s

        zone_code_col = _pick_column(zone_map, ['SA2_MAIN16', 'SA2_CODE16', 'SA2_MAIN21', 'SA2_CODE21', 'SA2_CODE'])
        zone_name_col = _pick_column(zone_map, ['SA2_NAME16', 'SA2_NAME21', 'SA2_NAME'])
        if zone_code_col is None:
            raise KeyError(f"No SA2 code column found in seq_zone_ids_.csv. Columns: {list(zone_map.columns)}")

        zone_map['ORDER'] = pd.to_numeric(zone_map['ORDER'], errors='coerce')
        zone_map['SA2_CODE'] = _normalize_sa2_code(zone_map[zone_code_col])
        zone_map['SA2_NAME'] = zone_map[zone_name_col].astype(str) if zone_name_col else 'Unknown'

        lu = landuse.copy()
        if 'ORDER' not in lu.columns:
                for col in ('zone_id', 'ORDER_rel'):
                        if col in lu.columns:
                                lu = lu.rename(columns={col: 'ORDER'})
                                break
        lu['ORDER'] = pd.to_numeric(lu['ORDER'], errors='coerce')
        zone_df = zone_map[['ORDER', 'SA2_CODE', 'SA2_NAME', 'latitude', 'longitude']].merge(
                lu,
                on='ORDER',
                how='left',
        )
        zone_df['SA2_NAME16'] = zone_df['SA2_NAME']

        crash_source = _detect_stage5_crash_source()
        crash_df = _read_crash_od_table(crash_source, zone_ids=zone_df['ORDER'].dropna().astype(int).tolist()) \
                if crash_source else pd.DataFrame(columns=['OTAZ', 'DTAZ', 'CRASH'])
        if not crash_df.empty:
                crash_df['log_crash'] = np.log(pd.to_numeric(crash_df['CRASH'], errors='coerce').fillna(0.0) + 1.0)
                crash_zone = (
                        crash_df.groupby('DTAZ', as_index=False)
                        .agg(
                                inbound_crash_mean=('CRASH', 'mean'),
                                inbound_crash_log_mean=('log_crash', 'mean'),
                                inbound_crash_total=('CRASH', 'sum'),
                        )
                        .rename(columns={'DTAZ': 'ORDER'})
                )
                zone_df = zone_df.merge(crash_zone, on='ORDER', how='left')
        else:
                zone_df['inbound_crash_mean'] = np.nan
                zone_df['inbound_crash_log_mean'] = np.nan
                zone_df['inbound_crash_total'] = np.nan

        crash_coef_rows = coef_base[
                coef_base['coefficient_name'].astype(str).str.lower().isin(['c_crash', 'c_crash_veh'])
        ].copy()
        crash_coef_rows['value'] = pd.to_numeric(crash_coef_rows['value'], errors='coerce')
        if 'p_value' in crash_coef_rows.columns:
                crash_coef_rows['p_value'] = pd.to_numeric(crash_coef_rows['p_value'], errors='coerce')
        else:
                crash_coef_rows['p_value'] = np.nan

        for seg in ('work', 'edu', 'nonmand'):
                _seg_dest = crash_coef_rows[
                        (crash_coef_rows['segment'] == seg) & (crash_coef_rows['model_type'] == 'dest')
                ]
                _seg_mode = crash_coef_rows[
                        (crash_coef_rows['segment'] == seg) & (crash_coef_rows['model_type'] == 'mode')
                ]
                dest_beta = float(_seg_dest['value'].iloc[0]) if len(_seg_dest) else np.nan
                mode_beta = float(_seg_mode['value'].iloc[0]) if len(_seg_mode) else np.nan
                zone_df[f'{seg}_dest_safety_penalty'] = pd.to_numeric(
                        zone_df.get('inbound_crash_log_mean'),
                        errors='coerce',
                ) * dest_beta
                zone_df[f'{seg}_mode_safety_penalty'] = pd.to_numeric(
                        zone_df.get('inbound_crash_log_mean'),
                        errors='coerce',
                ) * mode_beta

        penalty_cols = [c for c in zone_df.columns if c.endswith('_dest_safety_penalty')]
        zone_df['max_dest_safety_penalty'] = zone_df[penalty_cols].min(axis=1) if penalty_cols else np.nan
        zone_df['mean_dest_safety_penalty'] = zone_df[penalty_cols].mean(axis=1) if penalty_cols else np.nan

        feature_candidates = [
                'TOTPOP', 'TOTEMP', 'sa2_area', 'residential_y', 'commercial_y', 'retail_y',
                'park_y', 'recreation_ground_y', 'education_y', 'trees_y', 'childcare_y',
                'government_y', 'industrial_y', 'sport_y', 'utility_y', 'village_green_y',
        ]
        corr_rows = []
        target = pd.to_numeric(zone_df['max_dest_safety_penalty'], errors='coerce')
        for col in feature_candidates:
                if col not in zone_df.columns:
                        continue
                vals = pd.to_numeric(zone_df[col], errors='coerce')
                pair = pd.DataFrame({'x': vals, 'y': target}).dropna()
                if len(pair) < 8 or pair['x'].std() == 0 or pair['y'].std() == 0:
                        continue
                corr_rows.append({
                        'feature': col,
                        'correlation': float(pair['x'].corr(pair['y'])),
                        'mean_value': float(pair['x'].mean()),
                })
        corr_df = pd.DataFrame(corr_rows)
        if not corr_df.empty:
                corr_df = corr_df.sort_values('correlation')
        else:
                corr_df = pd.DataFrame(columns=['feature', 'correlation', 'mean_value'])

        top_zone_cols = [
                'ORDER', 'SA2_NAME16', 'inbound_crash_mean', 'inbound_crash_total',
                'max_dest_safety_penalty', 'mean_dest_safety_penalty',
                'TOTPOP', 'TOTEMP', 'commercial_y', 'retail_y', 'residential_y',
                'education_y', 'park_y', 'recreation_ground_y', 'trees_y',
        ]
        top_zone_cols = [c for c in top_zone_cols if c in zone_df.columns]
        top_zones = zone_df.sort_values('max_dest_safety_penalty').head(25)[top_zone_cols].copy()
        top_zones.to_csv(os.path.join(out_dir, 'safety_dashboard_top_zones.csv'), index=False)
        corr_df.to_csv(os.path.join(out_dir, 'safety_dashboard_feature_correlations.csv'), index=False)

        shp_path = os.path.join(
                'saved_data',
                'SA2_2021_AUST_SHP_GDA2020',
                'SA2_2021_AUST_GDA2020.shp',
        )
        geo_df = gpd.read_file(shp_path)
        geo_code_col = _pick_column(geo_df, ['SA2_MAIN16', 'SA2_CODE16', 'SA2_MAIN21', 'SA2_CODE21', 'SA2_CODE'])
        geo_name_col = _pick_column(geo_df, ['SA2_NAME16', 'SA2_NAME21', 'SA2_NAME'])
        if geo_code_col is None:
            raise KeyError(f"No SA2 code column found in shapefile. Columns: {list(geo_df.columns)}")

        geo_df['SA2_CODE'] = _normalize_sa2_code(geo_df[geo_code_col])
        if geo_name_col and 'SA2_NAME16' not in geo_df.columns:
            geo_df['SA2_NAME16'] = geo_df[geo_name_col]

        geo_df = geo_df[geo_df['SA2_CODE'].isin(zone_df['SA2_CODE'].dropna())].copy()
        geo_df = geo_df.merge(zone_df, on='SA2_CODE', how='left')
        geo_df = geo_df.to_crs(epsg=4326)
        geojson = json.loads(geo_df.to_json())

        # ── Load comparison CSV for mode/destination shift panels ────────────
        _cmp_df = comparison_df
        if _cmp_df is None or getattr(_cmp_df, 'empty', True):
                _cmp_csv = os.path.join(out_dir, 'safety_comparison.csv')
                if os.path.exists(_cmp_csv):
                        try:
                                _cmp_df = pd.read_csv(_cmp_csv)
                        except Exception:
                                _cmp_df = pd.DataFrame()
                else:
                        _cmp_df = pd.DataFrame()

        mode_share_rows = []
        dest_shift_rows = []
        if not _cmp_df.empty:
                _ms = _cmp_df[_cmp_df['category'] == 'mode_share'] if 'category' in _cmp_df.columns else pd.DataFrame()
                for _, r in _ms.iterrows():
                        mode_share_rows.append({
                                'mode':          str(r.get('parameter', '')),
                                'with_safety':   float(r['beta'])   if pd.notna(r.get('beta'))    else 0,
                                'without_safety':float(r['se'])     if pd.notna(r.get('se'))      else 0,
                                'delta':         float(r['t_ratio'])if pd.notna(r.get('t_ratio')) else 0,
                        })
                _ds = _cmp_df[_cmp_df['category'] == 'trip_distance'] if 'category' in _cmp_df.columns else pd.DataFrame()
                for _, r in _ds.iterrows():
                        dest_shift_rows.append({
                                'stat':          str(r.get('parameter', '')),
                                'with_safety':   float(r['beta'])   if pd.notna(r.get('beta'))    else None,
                                'without_safety':float(r['se'])     if pd.notna(r.get('se'))      else None,
                                'delta':         float(r['t_ratio'])if pd.notna(r.get('t_ratio')) else None,
                        })

        # ── Build OD flow network from HHTS observations ───────────────────
        # Load zone centroids for lat/lon lookup
        od_flows = []
        try:
                _zone_centroids = zone_map.set_index('ORDER')[['latitude', 'longitude']].dropna()
                _obs = hhts_obs if (hhts_obs is not None and not getattr(hhts_obs, 'empty', True)) else None
                if _obs is None:
                        _obs_path = os.path.join(baseline_dir, 'location_choice', 'location_choice_observations.csv')
                        if os.path.exists(_obs_path):
                                _obs = pd.read_csv(_obs_path)

                if _obs is not None and len(_obs) > 0:
                        _mode_names = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
                        _mode_col = next((c for c in ['tour_mode', 'mode', 'TOURMODE'] if c in _obs.columns), None)
                        _od = _obs.groupby(['OTAZ', 'choice']).size().reset_index(name='flow')
                        if _mode_col:
                                _od_mode = (
                                        _obs.groupby(['OTAZ', 'choice', _mode_col])
                                        .size()
                                        .reset_index(name='n')
                                )
                                _dom_mode = _od_mode.loc[_od_mode.groupby(['OTAZ', 'choice'])['n'].idxmax()]
                                _od = _od.merge(_dom_mode[['OTAZ', 'choice', _mode_col]], on=['OTAZ', 'choice'], how='left')
                                _od['mode_name'] = pd.to_numeric(_od[_mode_col], errors='coerce').map(_mode_names).fillna('DA')
                        else:
                                _od['mode_name'] = 'DA'

                        # Only keep top OD pairs by flow to keep JSON small
                        _od = _od.nlargest(500, 'flow')
                        for _, r in _od.iterrows():
                                otaz = int(r['OTAZ'])
                                dtaz = int(r['choice'])
                                if otaz not in _zone_centroids.index or dtaz not in _zone_centroids.index:
                                        continue
                                olat, olon = float(_zone_centroids.at[otaz, 'latitude']), float(_zone_centroids.at[otaz, 'longitude'])
                                dlat, dlon = float(_zone_centroids.at[dtaz, 'latitude']), float(_zone_centroids.at[dtaz, 'longitude'])
                                od_flows.append({
                                        'o': otaz, 'd': dtaz,
                                        'oLat': olat, 'oLon': olon,
                                        'dLat': dlat, 'dLon': dlon,
                                        'flow': int(r['flow']),
                                        'mode': str(r['mode_name']),
                                })
        except Exception as _od_e:
                logger.debug(f"OD flow network build failed: {_od_e}")

        dashboard_payload = {
                'geojson': geojson,
                'coefficients_with_safety': _clean_for_json(coef_base),
                'coefficients_without_safety': _clean_for_json(coef_nosa),
                'crash_coefficients': _clean_for_json(crash_coef_rows),
                'feature_correlations': _clean_for_json(corr_df.head(15)),
                'top_zones': _clean_for_json(top_zones),
                'mode_share': mode_share_rows,
                'dest_shift': dest_shift_rows,
                'od_flows': od_flows,
                'dest_zone_shift': dest_zone_shift if dest_zone_shift else [],
        }
        with open(os.path.join(out_dir, 'safety_dashboard_data.json'), 'w', encoding='utf-8') as handle:
                json.dump(dashboard_payload, handle)

        html_path = os.path.join(out_dir, 'safety_dashboard.html')
        html = """<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>Stage 5 Safety Dashboard</title>
    <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
    <script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
    <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
    <style>
        :root {{
            --bg: #f5efe3;
            --panel: rgba(255,255,255,0.88);
            --ink: #17313e;
            --muted: #5f6e75;
            --accent: #ad3f2f;
            --accent-2: #1f6a73;
            --line: rgba(23,49,62,0.12);
        }}
        body {{ margin: 0; font-family: Georgia, 'Palatino Linotype', serif; background:
            radial-gradient(circle at top left, #f6d8a8 0%, #f5efe3 35%, #d8ebe8 100%); color: var(--ink); }}
        .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
        h1 {{ font-size: 2.2rem; margin: 0 0 8px; }}
        h2 {{ margin: 0 0 10px; font-size: 1.1rem; }}
        p {{ color: var(--muted); line-height: 1.45; }}
        .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }}
        .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 16px 40px rgba(23,49,62,0.08); }}
        .card {{ padding: 16px 18px; }}
        .card .label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
        .card .value {{ font-size: 1.55rem; margin-top: 6px; font-weight: 700; }}
        .grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; }}
        .panel {{ padding: 16px; }}
        #map {{ height: 640px; border-radius: 14px; }}
        .controls {{ display: flex; gap: 10px; margin: 12px 0 14px; flex-wrap: wrap; }}
        select {{ border-radius: 999px; border: 1px solid var(--line); padding: 8px 14px; background: white; color: var(--ink); }}
        .plot {{ height: 320px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }}
        th {{ color: var(--muted); font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }}
        @media (max-width: 1100px) {{ .grid, .cards {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class='wrap'>
        <h1>Stage 5 Safety Dashboard</h1>
        <p>Interactive comparison of the selected Stage 5 smoke-test model with and without safety terms. The map shows which SA2 destinations are penalised most strongly by crash exposure, while the charts link that penalty back to coefficients, crash burden, and land-use context.</p>
        <div class='cards'>
            <div class='card'><div class='label'>Baseline Run</div><div class='value'>__BASELINE_LABEL__</div></div>
            <div class='card'><div class='label'>No-Safety Run</div><div class='value'>__NOSAFETY_LABEL__</div></div>
            <div class='card'><div class='label'>Zones Covered</div><div class='value'>__ZONE_COUNT__</div></div>
            <div class='card'><div class='label'>Crash Source</div><div class='value'>__CRASH_SOURCE__</div></div>
        </div>
        <div class='grid'>
            <div class='panel'>
                <div class='controls'>
                    <label>Map metric
                        <select id='metricSelect'>
                            <option value='max_dest_safety_penalty'>Max destination safety penalty</option>
                            <option value='mean_dest_safety_penalty'>Mean destination safety penalty</option>
                            <option value='work_dest_safety_penalty'>Work destination penalty</option>
                            <option value='edu_dest_safety_penalty'>Education destination penalty</option>
                            <option value='nonmand_dest_safety_penalty'>Non-mandatory destination penalty</option>
                            <option value='inbound_crash_total'>Inbound crash total</option>
                            <option value='inbound_crash_mean'>Inbound crash mean</option>
                        </select>
                    </label>
                    <label style='margin-left:10px'>
                        <input type='checkbox' id='odToggle' checked> HHTS OD flows
                    </label>
                    <label style='margin-left:10px'>
                        OD mode
                        <select id='odModeFilter'>
                            <option value='all'>All modes</option>
                            <option value='DA'>Drive Alone</option>
                            <option value='SR'>Shared Ride</option>
                            <option value='Walk'>Walk</option>
                            <option value='Bike'>Bike</option>
                            <option value='Transit'>Transit</option>
                        </select>
                    </label>
                </div>
                <div id='map'></div>
            </div>
            <div style='display:grid; gap:14px;'>
                <div class='panel'><div id='coefPlot' class='plot'></div></div>
                <div class='panel'><div id='corrPlot' class='plot'></div></div>
                <div class='panel'><div id='zonePlot' class='plot'></div></div>
            </div>
        </div>
        <div class='grid' style='margin-top:14px;'>
            <div class='panel'>
                <h2>Top Penalised SA2 Destinations</h2>
                <table id='zoneTable'></table>
            </div>
            <div class='panel'>
                <h2>Safety Coefficients</h2>
                <table id='coefTable'></table>
            </div>
        </div>
        <div style='margin-top:14px;'>
            <h2 style='font-size:1.05rem;margin:0 0 10px'>Behaviour Shift: With vs Without Safety Skims</h2>
            <div class='grid'>
                <div class='panel'><div id='modeSharePlot' class='plot'></div></div>
                <div class='panel'><div id='destShiftPlot' class='plot'></div></div>
            </div>
            <div class='panel' style='margin-top:14px;'>
                <div id='modeDeltaPlot' class='plot'></div>
            </div>
            <div class='panel' style='margin-top:14px;'>
                <div id='destZoneShiftPlot' style='height:420px;'></div>
            </div>
        </div>
    </div>
    <script>
        const payload = __PAYLOAD_JSON__;
        const crashRows = payload.crash_coefficients || [];
        const corrRows = payload.feature_correlations || [];
        const topZones = payload.top_zones || [];
        const modeShareRows = payload.mode_share || [];
        const destShiftRows = payload.dest_shift || [];
        const odFlows = payload.od_flows || [];
        const destZoneShift = payload.dest_zone_shift || [];
        const metricSelect = document.getElementById('metricSelect');
        const odToggle = document.getElementById('odToggle');
        const odModeFilter = document.getElementById('odModeFilter');
        const map = L.map('map').setView([-27.47, 153.03], 8.7);
        L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);

        function valuesFor(metric) {{
            return payload.geojson.features
                .map(f => Number(f.properties?.[metric]))
                .filter(v => Number.isFinite(v));
        }}

        function colorScale(value, minVal, maxVal) {{
            if (!Number.isFinite(value)) return '#c9d4d8';
            const span = (maxVal - minVal) || 1;
            const t = Math.max(0, Math.min(1, (value - minVal) / span));
            const r = Math.round(25 + t * 190);
            const g = Math.round(114 + (1 - t) * 90);
            const b = Math.round(110 + (1 - t) * 70);
            return `rgb(${{r}},${{g}},${{b}})`;
        }}

        let layer;
        function renderMap(metric) {{
            const vals = valuesFor(metric);
            const minVal = vals.length ? Math.min(...vals) : 0;
            const maxVal = vals.length ? Math.max(...vals) : 1;
            if (layer) map.removeLayer(layer);
            layer = L.geoJSON(payload.geojson, {{
                style: feature => {{
                    const value = Number(feature.properties?.[metric]);
                    return {{ color: '#ffffff', weight: 0.6, fillOpacity: 0.74, fillColor: colorScale(value, minVal, maxVal) }};
                }},
                onEachFeature: (feature, lyr) => {{
                    const p = feature.properties || {{}};
                    const zoneName = p.SA2_NAME16 || p.SA2_NAME16_x || p.SA2_NAME16_y || p.SA2_NAME || 'Unknown SA2';
                    lyr.bindPopup(`
                        <strong>${{zoneName}}</strong><br>
                        ORDER: ${{p.ORDER ?? 'n/a'}}<br>
                        ${metric}: ${{Number.isFinite(Number(p[metric])) ? Number(p[metric]).toFixed(3) : 'n/a'}}<br>
                        inbound_crash_total: ${{Number.isFinite(Number(p.inbound_crash_total)) ? Number(p.inbound_crash_total).toFixed(1) : 'n/a'}}<br>
                        TOTEMP: ${{Number.isFinite(Number(p.TOTEMP)) ? Number(p.TOTEMP).toFixed(0) : 'n/a'}}<br>
                        TOTPOP: ${{Number.isFinite(Number(p.TOTPOP)) ? Number(p.TOTPOP).toFixed(0) : 'n/a'}}
                    `);
                }}
            }}).addTo(map);
        }}

        function renderCoeffPlot() {{
            const order = ['work dest', 'edu dest', 'nonmand dest', 'work mode', 'edu mode', 'nonmand mode'];
            const label = r => `${{r.segment}} ${{r.model_type}}`;
            const sorted = [...crashRows].sort((a, b) => order.indexOf(label(a)) - order.indexOf(label(b)));
            Plotly.newPlot('coefPlot', [{
                x: sorted.map(r => label(r)),
                y: sorted.map(r => Number(r.value)),
                type: 'bar',
                marker: {{ color: sorted.map(r => r.model_type === 'dest' ? '#ad3f2f' : '#1f6a73') }},
                customdata: sorted.map(r => [r.coefficient_name, r.p_value]),
                hovertemplate: '%{x}<br>coef=%{y:.4f}<br>param=%{customdata[0]}<br>p=%{customdata[1]:.4g}<extra></extra>'
            }], {{
                title: 'Crash-related coefficients kept in the smoke model',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ l: 40, r: 10, t: 40, b: 80 }}
            }}, {{responsive: true}});
        }}

        function renderCorrPlot() {{
            const rows = corrRows.slice(0, 12).reverse();
            Plotly.newPlot('corrPlot', [{
                x: rows.map(r => Number(r.correlation)),
                y: rows.map(r => String(r.feature).replace(/_y$/, '').replace(/_/g, ' ')),
                type: 'bar', orientation: 'h', marker: {{ color: rows.map(r => Number(r.correlation) < 0 ? '#ad3f2f' : '#1f6a73') }}
            }], {{
                title: 'Features most associated with the safety penalty map',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ l: 130, r: 10, t: 40, b: 40 }}
            }}, {{responsive: true}});
        }}

        function renderZonePlot() {{
            const rows = topZones.slice(0, 20).reverse();
            Plotly.newPlot('zonePlot', [{
                x: rows.map(r => Number(r.max_dest_safety_penalty)),
                y: rows.map(r => r.SA2_NAME16 || `Zone ${{r.ORDER}}`),
                type: 'bar', orientation: 'h', marker: {{ color: '#ad3f2f' }}
            }], {{
                title: 'Most penalised destination zones',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ l: 170, r: 10, t: 40, b: 40 }}
            }}, {{responsive: true}});
        }}

        function renderTables() {{
            const zTable = document.getElementById('zoneTable');
            zTable.innerHTML = '<tr><th>SA2</th><th>Penalty</th><th>Crash</th><th>TOTEMP</th><th>TOTPOP</th></tr>' +
                topZones.slice(0, 15).map(r => `
                    <tr>
                        <td>${{r.SA2_NAME16 || r.ORDER}}</td>
                        <td>${{Number(r.max_dest_safety_penalty).toFixed(3)}}</td>
                        <td>${{Number.isFinite(Number(r.inbound_crash_total)) ? Number(r.inbound_crash_total).toFixed(1) : 'n/a'}}</td>
                        <td>${{Number.isFinite(Number(r.TOTEMP)) ? Number(r.TOTEMP).toFixed(0) : 'n/a'}}</td>
                        <td>${{Number.isFinite(Number(r.TOTPOP)) ? Number(r.TOTPOP).toFixed(0) : 'n/a'}}</td>
                    </tr>`).join('');

            const cTable = document.getElementById('coefTable');
            cTable.innerHTML = '<tr><th>Segment</th><th>Model</th><th>Parameter</th><th>Coef</th><th>p-value</th></tr>' +
                crashRows.map(r => `
                    <tr>
                        <td>${{r.segment}}</td>
                        <td>${{r.model_type}}</td>
                        <td>${{r.coefficient_name}}</td>
                        <td>${{Number(r.value).toFixed(4)}}</td>
                        <td>${{Number.isFinite(Number(r.p_value)) ? Number(r.p_value).toFixed(4) : 'n/a'}}</td>
                    </tr>`).join('');
        }}

        function renderModeSharePlot() {{
            if (!modeShareRows.length) return;
            const modes = modeShareRows.map(r => r.mode);
            Plotly.newPlot('modeSharePlot', [
                {{
                    name: 'With safety skims',
                    x: modes, y: modeShareRows.map(r => Number(r.with_safety)),
                    type: 'bar', marker: {{color: '#2C6B9C'}},
                    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>with safety</extra>'
                }},
                {{
                    name: 'Without safety skims',
                    x: modes, y: modeShareRows.map(r => Number(r.without_safety)),
                    type: 'bar', marker: {{color: '#E07030'}},
                    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>without safety</extra>'
                }}
            ], {{
                title: 'Mode share: with vs without safety skims',
                barmode: 'group',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                yaxis: {{title: 'Mode share (%)'}},
                legend: {{orientation: 'h', y: -0.2}},
                margin: {{l: 50, r: 10, t: 50, b: 60}}
            }}, {{responsive: true}});
        }}

        function renderModeDeltaPlot() {{
            if (!modeShareRows.length) return;
            const modes = modeShareRows.map(r => r.mode);
            const deltas = modeShareRows.map(r => Number(r.delta));
            Plotly.newPlot('modeDeltaPlot', [{{
                x: modes, y: deltas, type: 'bar',
                marker: {{color: deltas.map(d => d < 0 ? '#2ECC71' : '#E74C3C')}},
                hovertemplate: '%{{x}}: %{{y:+.2f}} pp<extra></extra>',
                text: deltas.map(d => `${{d > 0 ? '+' : ''}}${{d.toFixed(2)}}pp`),
                textposition: 'outside'
            }}], {{
                title: 'Mode share shift from removing safety skims (positive = increases without crash penalty)',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                yaxis: {{title: 'Δ share (pp)', zeroline: true}},
                margin: {{l: 50, r: 10, t: 50, b: 60}}
            }}, {{responsive: true}});
        }}

        function renderDestShiftPlot() {{
            if (!destShiftRows.length) return;
            const stats = destShiftRows.map(r => r.stat.replace('dist_', '').replace('_km', ''));
            const bVals = destShiftRows.map(r => r.with_safety);
            const nVals = destShiftRows.map(r => r.without_safety);
            const deltas = destShiftRows.map(r => r.delta);
            Plotly.newPlot('destShiftPlot', [
                {{
                    name: 'With safety', x: stats, y: bVals, type: 'bar',
                    marker: {{color: '#2C6B9C'}},
                    hovertemplate: '%{{x}}: %{{y:.2f}} km<extra>with safety</extra>'
                }},
                {{
                    name: 'Without safety', x: stats, y: nVals, type: 'bar',
                    marker: {{color: '#E07030'}},
                    hovertemplate: '%{{x}}: %{{y:.2f}} km<extra>without safety</extra>'
                }},
                {{
                    name: 'Δ (km)', x: stats, y: deltas, type: 'scatter', mode: 'markers+text',
                    marker: {{color: '#1f6a73', size: 10, symbol: 'diamond'}},
                    text: deltas.map(d => d !== null ? `${{d > 0 ? '+' : ''}}${{Number(d).toFixed(2)}}km` : ''),
                    textposition: 'top center',
                    yaxis: 'y2'
                }}
            ], {{
                title: 'Trip distance distribution: with vs without safety skims',
                barmode: 'group',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                yaxis: {{title: 'Distance (km)'}},
                yaxis2: {{title: 'Δ km', overlaying: 'y', side: 'right', showgrid: false}},
                legend: {{orientation: 'h', y: -0.2}},
                margin: {{l: 50, r: 60, t: 50, b: 60}}
            }}, {{responsive: true}});
        }}

        // ── OD flow network (HHTS observed tours as lines) ─────────────────
        const modeColors = {{
            DA: '#E74C3C', SR: '#E67E22', Walk: '#27AE60',
            Bike: '#3498DB', Transit: '#9B59B6'
        }};
        let odLayer = null;
        function renderOdFlows() {{
            if (odLayer) {{ map.removeLayer(odLayer); odLayer = null; }}
            if (!odToggle.checked || !odFlows.length) return;
            const modeFilter = odModeFilter.value;
            const maxFlow = Math.max(...odFlows.map(r => r.flow));
            const lines = odFlows
                .filter(r => modeFilter === 'all' || r.mode === modeFilter)
                .map(r => {{
                    const w = Math.max(0.8, (r.flow / maxFlow) * 6);
                    const col = modeColors[r.mode] || '#888';
                    return L.polyline(
                        [[r.oLat, r.oLon], [r.dLat, r.dLon]],
                        {{ color: col, weight: w, opacity: 0.55 }}
                    ).bindPopup(`OTAZ ${{r.o}} → DTAZ ${{r.d}}<br>Mode: ${{r.mode}}<br>HHTS flows: ${{r.flow}}`);
                }});
            odLayer = L.layerGroup(lines).addTo(map);
        }}

        // ── Destination zone shift Plotly chart ────────────────────────────
        function renderDestZoneShiftPlot() {{
            if (!destZoneShift.length) return;
            const el = document.getElementById('destZoneShiftPlot');
            if (!el) return;
            const segs = [...new Set(destZoneShift.map(r => r.segment))];
            const traces = segs.map(seg => {{
                const rows = destZoneShift
                    .filter(r => r.segment === seg)
                    .sort((a, b) => a.delta_pred - b.delta_pred)
                    .slice(0, 30);
                return {{
                    name: seg,
                    x: rows.map(r => Number(r.delta_pred)),
                    y: rows.map(r => `Zone ${{r.zone}}`),
                    type: 'bar', orientation: 'h',
                    hovertemplate: `${{seg}} Zone %{{y}}<br>Δ pred share: %{{x:.4f}}<extra></extra>`
                }};
            }});
            Plotly.newPlot('destZoneShiftPlot', traces, {{
                title: 'Destination zone probability shift (baseline pred − nosafety pred)',
                barmode: 'overlay',
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: {{title: 'Δ predicted share (baseline − nosafety)'}},
                margin: {{l: 80, r: 10, t: 50, b: 50}}
            }}, {{responsive: true}});
        }}

        metricSelect.addEventListener('change', e => renderMap(e.target.value));
        odToggle.addEventListener('change', renderOdFlows);
        odModeFilter.addEventListener('change', renderOdFlows);
        renderMap(metricSelect.value);
        renderOdFlows();
        renderCoeffPlot();
        renderCorrPlot();
        renderZonePlot();
        renderTables();
        renderModeSharePlot();
        renderModeDeltaPlot();
        renderDestShiftPlot();
        renderDestZoneShiftPlot();
    </script>
</body>
</html>"""
        html = (
                html.replace('__PAYLOAD_JSON__', json.dumps(dashboard_payload))
                .replace('__BASELINE_LABEL__', os.path.basename(baseline_dir))
                .replace('__NOSAFETY_LABEL__', os.path.basename(nosafety_dir))
                .replace('__ZONE_COUNT__', str(int(zone_df['ORDER'].nunique())))
                .replace('__CRASH_SOURCE__', os.path.basename(crash_source) if crash_source else 'none')
        )
        with open(html_path, 'w', encoding='utf-8') as handle:
                handle.write(html)
        logger.info(f"Interactive safety dashboard -> {html_path}")
        return html_path


@timer("GA — Stage 6 (person + accessibility)")
def ga_stage6(out, landuse):
    from census_bridge import run_ga_extended
    _bp = _ga_budget_params()
    res = run_ga_extended(
        stage=6, trips=out["trips"], persons=out["persons"],
        landuse=landuse, purpose='work', **_bp,
    )
    _ga_plot_stage_result(res, os.path.join(OUT_DIR, 'plots', 'ga_stage6.png'))
    print(f"  person: {res.best_features.get('person')}")
    print(f"  acc:    {res.best_features.get('accessibility')}")
    return res


# ---------------------------------------------------------------------------
# MAIN PIPELINES
# ---------------------------------------------------------------------------

def main(filepath: str = None):
    """
    Standard pipeline — full synthetic population across all zones.

    If filepath is given, runs on that single zone CSV only (legacy mode).
    Otherwise loads every order_N.csv from SYNTH_DIR.

    Usage
    -----
    python pipeline_logger.py                          # all zones
    python pipeline_logger.py main                     # all zones
    python pipeline_logger.py main order_40.csv        # single zone (legacy)
    python pipeline_logger.py main 40,124,200          # comma-separated zone list
    """
    # ── Data quality checks ─────────────────────────────────────────────
    # Runs before any estimation — catches mapping errors, wrong column values,
    # etc. so you see clear warnings before waiting hours for a crash.
    try:
        from data_checks import run_data_checks
        run_data_checks(raise_on_critical=False)
    except ImportError:
        logger.debug("data_checks.py not on path — skipping startup checks")
    except Exception as _dc_e:
        logger.warning(f"Data checks failed (non-fatal): {_dc_e}")

    sc, out, landuse = load_data()

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — MODEL ESTIMATION (HHTS only, no synth population needed)
    # Estimate all stage models, save PKL + HTML reports.
    # The synth population is NOT loaded until Phase 2.
    # This keeps peak memory low: HHTS (~17k persons) vs synth (3.7M persons).
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1 — MODEL ESTIMATION (HHTS)")
    logger.info("=" * 60)

    def _mem_mb():
        """Current process RSS memory in MB — for progress tracking on HPC."""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            try:
                import psutil, os as _os
                return psutil.Process(_os.getpid()).memory_info().rss / 1e6
            except Exception:
                return float('nan')

    logger.info("▶  Estimating Stage 1 models (activity frequency)...")
    s1_models = estimate_stage1(out, landuse)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 1]")

    logger.info("▶  Estimating Stage 2 models (day duration)...")
    s2_models = estimate_stage2(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 2]")

    # Joint participation + duration models (Tobit-I / Heckman).
    # Always run in main so the HTML comparison report is available for every
    # run.  The recommendation block in estimate_stage2_joint prints which
    # model (two-stage lognormal vs Tobit-I vs Heckman) fits better.
    os.environ['JOINT_PARTICIPATION_MODEL'] = '1'
    logger.info("▶  Estimating Stage 2 joint models (Tobit-I + Heckman — model comparison)...")
    estimate_stage2_joint(out, s2_models=s2_models)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 2 joint]")

    logger.info("▶  Estimating Stage 3 models (activity duration)...")
    s3_models = estimate_stage3(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 3]")

    logger.info("▶  Estimating Stage 4 models (schedule placement)...")
    s4_models = estimate_stage4(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 4]")

    logger.info("▶  Estimating Stage 5 models (destination + mode choice)...")
    s5_models_est, skims, sk = estimate_stage5(out, landuse)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 5]")

    logger.info("▶  Estimating Stage 5t models (trip-level mode + destination choice)...")
    _require_s5t = os.environ.get('STAGE5T_REQUIRED', '1') == '1'
    try:
        estimate_stage5_trips(out, landuse, skims)
    except Exception as _s5t_e:
        import traceback as _tb_s5t
        logger.error(f"  Trip-level Stage 5 estimation failed: {_s5t_e}")
        logger.debug(_tb_s5t.format_exc())
        if _require_s5t:
            raise RuntimeError(
                "Stage 5t (trip mode + trip destination) is required but failed. "
                "Set STAGE5T_REQUIRED=0 to continue without trip-level model artifacts."
            ) from _s5t_e
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 5t]")

    logger.info("✓  All stage models estimated and saved.")
    logger.info(f"   PKL + HTML reports in {OUT_DIR}/")

    # ── In-sample validation on HHTS training data ─────────────────────────
    # Runs BEFORE freeing models so all fitted objects are still in memory.
    # Produces plots + HTML report in {OUT_DIR}/plots/validation/
    try:
        from validate_hhts import run_validation
        run_validation(
            out       = out,
            s1_models = s1_models,
            s2_models = s2_models,
            s3_models = s3_models,
            s4_models = s4_models,
            s5_models = s5_models_est,
            out_dir   = OUT_DIR,
        )
    except Exception as _val_err:
        import traceback as _tb
        logger.warning(f"HHTS validation failed (non-fatal): {_val_err}")
        logger.debug(_tb.format_exc())

    # Free Phase 1 model objects from memory — Phase 2 reloads from disk
    # one at a time so peak memory stays low during synth load.
    del s1_models, s2_models, s3_models, s4_models, s5_models_est
    gc.collect()
    logger.info("  Phase 1 model objects freed from memory (all saved to disk)")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — POPULATION PREDICTION (synth population, one zone at a time)
    #
    # Memory strategy:
    #   - Save HHTS persons to disk for Stage 5 prediction, then del out entirely
    #   - Load each stage's model just before use, del immediately after
    #   - Stage 5 processes one origin zone at a time (~300 MB peak per zone)
    #   - Stage 6 processes 20k persons at a time (~500 MB peak per chunk)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2 — POPULATION PREDICTION")
    logger.info("=" * 60)
    # Save HHTS persons + tours to disk — needed by Stage 5 and usual_location
    # validation respectively; keeping full 'out' dict wastes 3-5 GB during synth load
    _hhts_persons_path = os.path.join(OUT_DIR, '_hhts_persons.pkl')
    _hhts_tours_path   = os.path.join(OUT_DIR, '_hhts_tours.pkl')
    import pickle as _pkl
    with open(_hhts_persons_path, 'wb') as _pf:
        _pkl.dump(out['persons'], _pf)
        _pf.flush()
        os.fsync(_pf.fileno())
    if not os.path.exists(_hhts_persons_path):
        raise RuntimeError(
            f"HHTS persons file was not written: {_hhts_persons_path}\n"
            "Check disk quota — run 'quota -s' or 'df -h $HOME'."
        )
    logger.info(f"  HHTS persons saved to {_hhts_persons_path} ({os.path.getsize(_hhts_persons_path)//1024:,} KB)")
    try:
        with open(_hhts_tours_path, 'wb') as _tf:
            _pkl.dump(out['tours'], _tf)
        logger.info(f"  HHTS tours saved to {_hhts_tours_path} for usual-location validation")
    except Exception as _te:
        logger.debug(f"  Could not save HHTS tours ({_te}) — validation plots will be skipped")
        _hhts_tours_path = None

    # Free HHTS data — not needed until Stage 5 prediction
    # Keep landuse (small) and skims handle (file reference only)
    _out_persons_ref = _hhts_persons_path   # path to reload when needed
    del out
    gc.collect()
    logger.info("  HHTS 'out' dict freed — loading synthetic population...")

    if filepath is not None:
        if os.path.isfile(filepath):
            synth = load_synth(filepath, landuse=landuse)
        elif filepath.replace(',', '').replace(' ', '').isdigit():
            zones = [int(z.strip()) for z in filepath.split(',')]
            synth = load_synth_all_zones(SYNTH_DIR, landuse=landuse, zones=zones)
        else:
            synth = load_synth(os.path.join(SYNTH_DIR, filepath), landuse=landuse)
    else:
        synth = load_synth_all_zones(SYNTH_DIR, landuse=landuse)

    # Reconstruct minimal out dict for cdap and usual_locations
    _out_for_cdap = {'persons': _pkl.load(open(_out_persons_ref, 'rb'))}
    # Load HHTS tours for usual-location validation plots (work/school destinations)
    _hhts_tours_for_val = None
    _hhts_tours_stage5_val = None
    if _hhts_tours_path and os.path.exists(_hhts_tours_path):
        try:
            _hhts_tours_for_val = _pkl.load(open(_hhts_tours_path, 'rb'))
            try:
                from stage5_destination_choice import _prep_hhts_tours as _prep_stage5_val_tours
                _hhts_persons_for_val = _pkl.load(open(_out_persons_ref, 'rb'))
                _hhts_tours_stage5_val = _prep_stage5_val_tours(
                    _hhts_tours_for_val,
                    _hhts_persons_for_val,
                )
                logger.info(
                    f"  Prepared {len(_hhts_tours_stage5_val):,} HHTS tours for Stage 5 validation"
                )
            except Exception as _pve:
                logger.debug(f"  Could not prepare Stage 5 validation tours ({_pve})")
        except Exception as _tle:
            logger.debug(f"  Could not load HHTS tours for validation ({_tle})")
    # usual_locations must run BEFORE cdap so work/school zones are known
    synth      = usual_locations(synth, landuse, hhts_tours=_hhts_tours_for_val)
    synth      = cdap(_out_for_cdap, synth)
    _audit_hhts_synth_consistency(_out_for_cdap.get('persons'), synth, landuse,
                                  label='post_usual_locations')

    # ── Location Choice Summary (ActivitySim-style output) ────────────────
    # Save coefficient tables + validation right after CDAP / usual_locations,
    # following ActivitySim's workplace_location / school_location protocol.
    try:
        from stage5_destination_choice import (
            save_location_choice_summary as _save_loc_summary,
            load_models as _s5_load_loc,
        )
        _s5_path = os.path.join(OUT_DIR, 'stage5_models.pkl')
        if os.path.isfile(_s5_path):
            _s5_models = _s5_load_loc(_s5_path)
            _save_loc_summary(
                models=_s5_models,
                obs_tours=_hhts_tours_stage5_val if _hhts_tours_stage5_val is not None else pd.DataFrame(),
                landuse=landuse,
                save_dir=OUT_DIR,
                persons=_out_for_cdap.get('persons'),
            )
            del _s5_models
            logger.info("  Location choice summary saved (post-CDAP)")
        else:
            logger.info("  No stage5_models.pkl found — skipping location choice summary")
    except Exception as _loc_e:
        logger.warning(f"  Location choice summary failed: {_loc_e}")

    del _out_for_cdap
    gc.collect()

    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    logger.info(f"CDAP filter: {len(travellers):,}/{len(synth):,} travellers")

    # Normalise ptype labels: enrich_synth_population may produce old-style labels
    # (School, Student, JointWorkStudy) — remap to the labels the models were
    # estimated on (SchoolChild, UniversityStudent, WorkingStudent).
    _PTYPE_REMAP = {
        'School':            'SchoolChild',
        'Student':           'UniversityStudent',
        'JointWorkStudy':    'WorkingStudent',
        'Unemployed / Other':'NonWorker',
        # Already-correct labels pass through unchanged
        'FullTimeWorker':    'FullTimeWorker',
        'PartTimeWorker':    'PartTimeWorker',
        'SchoolChild':       'SchoolChild',
        'UniversityStudent': 'UniversityStudent',
        'WorkingStudent':    'WorkingStudent',
        'Retired':           'Retired',
        'Preschool':         'Preschool',
        'NonWorker':         'NonWorker',
    }
    if 'ptype' in travellers.columns:
        before = travellers['ptype'].value_counts().to_dict()
        travellers['ptype'] = travellers['ptype'].map(_PTYPE_REMAP).fillna('NonWorker')

        # Guardrail: prevent over-assignment of Preschool when upstream coded
        # fields are noisy. Preschool should be very young children only.
        if 'age5p' in travellers.columns:
            _age5 = pd.to_numeric(travellers['age5p'], errors='coerce').fillna(6).astype(int)
            _preschool = travellers['ptype'] == 'Preschool'
            _too_old = _preschool & (_age5 > 1)
            if _too_old.any():
                travellers.loc[_too_old & (_age5 <= 5), 'ptype'] = 'SchoolChild'
                travellers.loc[_too_old & (_age5 > 5),  'ptype'] = 'NonWorker'
                logger.info(
                    f"  ptype guardrail: reassigned {int(_too_old.sum())} over-age Preschool persons"
                )

        if 'STUDYING' in travellers.columns:
            _stud = pd.to_numeric(travellers['STUDYING'], errors='coerce').fillna(0).astype(int)
            _work = pd.to_numeric(travellers.get('WORKSTAT', 0), errors='coerce').fillna(0).astype(int)
            _ps = travellers['ptype'] == 'Preschool'
            _school = _ps & _stud.isin([2])
            _uni = _ps & _stud.isin([3, 4])
            if _school.any() or _uni.any():
                travellers.loc[_school, 'ptype'] = 'SchoolChild'
                travellers.loc[_uni, 'ptype'] = 'UniversityStudent'
            _working_student = _stud.isin([3, 4]) & _work.isin([1, 2])
            _uni_only = _stud.isin([3, 4]) & ~_work.isin([1, 2])
            if _working_student.any():
                travellers.loc[_working_student, 'ptype'] = 'WorkingStudent'
            if _uni_only.any():
                travellers.loc[_uni_only, 'ptype'] = 'UniversityStudent'

        after = travellers['ptype'].value_counts().to_dict()
        changed = {k: (before.get(k,0), after.get(k,0)) for k in set(before)|set(after) if before.get(k) != after.get(k)}
        if changed:
            logger.info(f"  ptype labels normalised (old→new): {changed}")
        else:
            logger.info(f"  ptype labels already normalised")
    if 'ptype' in synth.columns:
        synth['ptype'] = synth['ptype'].map(_PTYPE_REMAP).fillna('NonWorker')

    # Save lightweight home-zone table for spatial plots later
    _synth_home = (synth[['person_id', 'OTAZ']].copy()
                   if 'person_id' in synth.columns else None)
    del synth
    gc.collect()

    # ── Stage 1 predict — load model, predict, del model ──────────────────
    logger.info("▶  Phase 2 — Stage 1 prediction")
    _s1 = __import__('stage1_activity_frequency',
                     fromlist=['load_models']).load_models(
                         os.path.join(OUT_DIR, 'stage1_models.pkl'))
    _hhts_persons_s1 = None
    _hhts_tours_s1 = None
    try:
        _hhts_persons_s1 = _pkl.load(open(_out_persons_ref, 'rb'))
    except Exception as _hp_e:
        logger.warning(f"  Could not load HHTS persons for Stage 1 calibration: {_hp_e}")
    if _hhts_tours_path and os.path.exists(_hhts_tours_path):
        try:
            _hhts_tours_s1 = _pkl.load(open(_hhts_tours_path, 'rb'))
        except Exception as _ht_e:
            logger.warning(f"  Could not load HHTS tours for Stage 1 calibration: {_ht_e}")
    synth_s1 = predict_stage1(
        travellers,
        _s1,
        hhts_persons=_hhts_persons_s1,
        hhts_tours=_hhts_tours_s1,
    )
    del _s1, _hhts_persons_s1, _hhts_tours_s1
    gc.collect()

    # ── Stage 2 predict ───────────────────────────────────────────────────
    logger.info("▶  Phase 2 — Stage 2 prediction")
    _s2 = __import__('stage2_day_duration',
                     fromlist=['load_models']).load_models(
                         os.path.join(OUT_DIR, 'stage2_models.pkl'))
    synth_s2 = predict_stage2(synth_s1, _s2); del _s2, synth_s1; gc.collect()

    # ── Stage 3 predict ───────────────────────────────────────────────────
    logger.info("▶  Phase 2 — Stage 3 prediction")
    _s3 = __import__('stage3_activity_duration',
                     fromlist=['load_models']).load_models(
                         os.path.join(OUT_DIR, 'stage3_models.pkl'))
    activities = predict_stage3(synth_s2, _s3); del _s3, synth_s2; gc.collect()

    # ── Stage 4 predict ───────────────────────────────────────────────────
    logger.info("▶  Phase 2 — Stage 4 prediction")
    _stage4_path = os.path.join(OUT_DIR, 'stage4_models.pkl')
    if not os.path.exists(_stage4_path):
        _run_contents = os.listdir(OUT_DIR) if os.path.isdir(OUT_DIR) else ['<dir missing>']
        logger.error(
            f"stage4_models.pkl is missing from {OUT_DIR}\n"
            f"  Expected: {_stage4_path}\n"
            f"  Run directory contents: {_run_contents}\n"
            "Most likely causes: (1) Stage 4 estimation failed earlier in this run; "
            "(2) run-directory contents were cleared by a concurrent re-submission; "
            "(3) disk quota exceeded during model writes."
        )
        raise FileNotFoundError(
            f"stage4_models.pkl not found at '{_stage4_path}'. "
            "Check pipeline.log for upstream Stage 4 estimation errors and run-directory cleanup events."
        )
    _s4 = __import__('stage4_schedule_placement',
                     fromlist=['load_models']).load_models(
                         _stage4_path)
    scheduled = predict_stage4(activities, _s4); del _s4, activities; gc.collect()

    # ── Stage 5 predict (one zone at a time) ──────────────────────────────
    logger.info("▶  Phase 2 — Stage 5 prediction (one origin zone per iteration)")
    from stage5_destination_choice import load_models as _s5_load
    _s5 = _s5_load(os.path.join(OUT_DIR, 'stage5_models.pkl'))
    # Reconstruct minimal out dict for Stage 5 predict. Use the synthetic
    # person table, not HHTS estimation persons, otherwise most Stage 5
    # person/home lookups miss and collapse back to zone 1.
    _out_s5 = {'persons': travellers, 'trips': pd.DataFrame(),
               'tours': _hhts_tours_stage5_val if _hhts_tours_stage5_val is not None else pd.DataFrame()}
    scheduled_dest = predict_stage5(
        scheduled, _s5, _out_s5, landuse, skims, sk, persons=travellers
    )
    del scheduled, _s5, _out_s5; gc.collect()

    # ── Stage 5b predict — trip-level intermediate-stop destinations ──────
    logger.info("▶  Phase 2 — Stage 5b prediction (trip-level destination choice)")
    try:
        _out_s5b = {'persons': travellers}
        scheduled_dest = stage5b(_out_s5b, scheduled_dest, landuse)
        logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 5b]")
    except Exception as _s5b_e:
        import traceback as _tb_s5b
        logger.warning(
            f"Stage 5b failed (non-fatal): {_s5b_e!r}\n{_tb_s5b.format_exc(limit=8)}"
        )
    gc.collect()

    # ── Stage 6 (chunked by persons) ─────────────────────────────────────
    logger.info("▶  Phase 2 — Stage 6 validation + export (chunked)")
    if not os.path.exists(_out_persons_ref):
        _run_contents = os.listdir(OUT_DIR) if os.path.isdir(OUT_DIR) else ['<dir missing>']
        logger.error(
            f"_hhts_persons.pkl is missing from {OUT_DIR}\n"
            f"  Expected: {_out_persons_ref}\n"
            f"  Run directory contents: {_run_contents}\n"
            "Most likely causes: (1) the run directory was cleaned by a concurrent "
            "re-submission while this job was executing — never re-submit while a job "
            "is running; (2) disk quota exceeded during Stage 5 — check 'quota -s'."
        )
        raise FileNotFoundError(
            f"_hhts_persons.pkl not found at '{_out_persons_ref}'. "
            "See pipeline.log for diagnostic info."
        )
    # Minimal 'out' for stage6 — only persons needed for validate step
    _out_s6 = {'persons': _pkl.load(open(_out_persons_ref, 'rb'))}
    plan, _ = stage6(_out_s6, scheduled_dest, landuse,
                     stage5_models=None,
                     skims=skims, skim_lookup=sk)
    del scheduled_dest, _out_s6; gc.collect()
    logger.info(f"  Stage 6 complete — outputs in {OUT_DIR}/")
    logger.info(f"    Flat CSVs  : {OUT_DIR}/stage6_output.csv")
    logger.info(f"                 {OUT_DIR}/trips_for_aimsun.csv")

    # ── Multi-level validation: trip / tour / work / edu assignment ────────
    # Compare predicted plan against HHTS observed tours for all four levels.
    try:
        from stage5_destination_choice import compute_multilevel_validation
        _hhts_val_tours = None
        if _hhts_tours_path and os.path.exists(_hhts_tours_path):
            try:
                import pickle as _pkl2
                _hhts_val_tours = _pkl2.load(open(_hhts_tours_path, 'rb'))
                logger.info(f"  Multilevel validation: loaded HHTS tours ({len(_hhts_val_tours)} rows)")
            except Exception as _tle2:
                logger.debug(f"  Could not load HHTS tours for multilevel validation: {_tle2}")
        _hhts_val_tours_stage5 = _hhts_tours_stage5_val
        if _hhts_val_tours_stage5 is not None and not _hhts_val_tours_stage5.empty:
            _ml_val_dir = os.path.join(OUT_DIR, 'validation')
            compute_multilevel_validation(
                obs_tours=_hhts_val_tours_stage5,
                pred_plan=plan,
                landuse=landuse,
                save_dir=_ml_val_dir,
            )
            logger.info(f"  Multilevel validation outputs → {_ml_val_dir}/")
        else:
            logger.info("  Multilevel validation: no HHTS tours available, skipping")
    except Exception as _mlv_e:
        logger.warning(f"  Multilevel validation failed: {_mlv_e}")
        import traceback as _tb2; logger.debug(_tb2.format_exc())

    plot(plan)
    _synth_spatial = (_synth_home if _synth_home is not None
                      else pd.DataFrame(columns=['person_id', 'OTAZ']))
    spatial_plots(plan, _synth_spatial, landuse)
    del plan, _synth_spatial, _synth_home
    gc.collect()
    if skims:
        skims.close()
    logger.info("Pipeline complete.")
    logger.info(f"  Per-zone folders : {OUT_DIR}/zones/zone_{{1..N}}/")
    logger.info(f"    Each folder contains stage6_output.csv + trips_for_aimsun.csv for that origin zone.")

    # ── Auto-run safety comparison if both experiment arms have completed ──
    _safety_mode = os.environ.get('SAFETY_EXPERIMENT_MODE', '').strip()
    if _safety_mode in ('baseline', 'nosafety'):
        _safe_base_dir = os.path.join('runs', 'safety_baseline')
        _safe_nosa_dir = os.path.join('runs', 'safety_nosafety')
        _safe_out_dir  = os.path.join('runs', 'safety_analysis')
        # Only compare if the OTHER arm has already finished (has a plan CSV or coef CSV)
        _arm_done = lambda d: (
            os.path.isfile(os.path.join(d, 'stage5_models_coefficients.csv')) or
            os.path.isfile(os.path.join(d, 'stage5_output.csv'))
        )
        if _arm_done(_safe_base_dir) and _arm_done(_safe_nosa_dir):
            logger.info("Both safety experiment arms detected — running safety comparison...")
            try:
                compare_safety_runs(
                    baseline_dir=_safe_base_dir,
                    nosafety_dir=_safe_nosa_dir,
                    out_dir=_safe_out_dir,
                )
                logger.info(f"  Safety analysis → {_safe_out_dir}/safety_report.html")
            except Exception as _sfe:
                import traceback as _tb_sf
                logger.warning(f"Safety comparison failed: {_sfe}")
                logger.debug(_tb_sf.format_exc())
        else:
            _missing_arm = 'baseline' if not _arm_done(_safe_base_dir) else 'nosafety'
            logger.info(
                f"Safety experiment '{_safety_mode}' completed. "
                f"Run 'safety_{_missing_arm}' then 'safety_compare' to generate the report."
            )


def GA_main(filepath: str = None):
    """Run GA across all stages, then apply GA-selected features to the pipeline.

    GA estimation uses only HHTS data — synth is NOT loaded until after GA
    completes.  Pipeline prediction uses apply_ga_results_to_pipeline() which
    passes each stage's GA-selected feature set as feature_override.
    """
    import pickle
    sc, out, landuse = load_data()

    # ── GA estimation (HHTS only) ─────────────────────────────────────
    ga_all(out, landuse)

    # ── Load GA results ───────────────────────────────────────────────
    ga_path = os.path.join(OUT_DIR, 'ga_results.pkl')
    if not os.path.isfile(ga_path):
        logger.warning(f"GA results not found at {ga_path} — running pipeline without feature overrides")
        ga_results = None
    else:
        with open(ga_path, 'rb') as _f:
            ga_results = pickle.load(_f)
        logger.info(f"GA results loaded: {len(ga_results.results)} stages")

    # ── Load synth only after GA completes ────────────────────────────
    synth = (load_synth(filepath, landuse=landuse) if filepath and os.path.isfile(filepath)
             else load_synth_all_zones(SYNTH_DIR, landuse=landuse))

    synth = usual_locations(synth, landuse, hhts_tours=out.get('tours'))
    _s0_res = ga_results.results.get(0) if ga_results is not None else None
    synth = cdap(out, synth, feature_override=getattr(_s0_res, 'best_features', None))
    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} persons are travellers")

    if ga_results is not None:
        # ── Apply GA-selected features through stages 1–5 ─────────────
        from ga_hypothesis_testing import apply_ga_results_to_pipeline
        _prev_larch_engine = os.environ.get('GA_LARCH_ENGINE')
        _prev_strict_s5 = os.environ.get('STAGE5_STRICT_ESTIMATION')
        try:
            # GA search can use numba, but final Stage 5 application is more
            # stable on JAX for large idca dataflows on Linux/HPC.
            if os.name != 'nt':
                os.environ['GA_LARCH_ENGINE'] = 'jax'
            # Require strict Stage 5 fit during GA apply.
            os.environ['STAGE5_STRICT_ESTIMATION'] = '1'
            scheduled_dest, s5_models = apply_ga_results_to_pipeline(
                ga_results   = ga_results,
                out          = out,
                synth        = travellers,
                landuse      = landuse,
                skims_path   = PIPELINE_SKIMS_PATH,
                save_dir     = OUT_DIR,
            )
        finally:
            if _prev_larch_engine is None:
                os.environ.pop('GA_LARCH_ENGINE', None)
            else:
                os.environ['GA_LARCH_ENGINE'] = _prev_larch_engine
            if _prev_strict_s5 is None:
                os.environ.pop('STAGE5_STRICT_ESTIMATION', None)
            else:
                os.environ['STAGE5_STRICT_ESTIMATION'] = _prev_strict_s5
        # ── Post-GA diagnostics (coefficients + HHTS validation) ──────
        _ga_post_estimation_report(s5_models, out, OUT_DIR)
    else:
        # Fallback: run without feature overrides (same as main)
        synth_s1, _                          = stage1(out, travellers)
        synth_s2, _                          = stage2(out, synth_s1)
        activities, _                        = stage3(out, synth_s2)
        scheduled, _                         = stage4(out, activities)
        scheduled_dest, s5_models, _s5_skims, _ = stage5(out, travellers, scheduled, landuse)
        if _s5_skims: _s5_skims.close()  # skims reopened below for stage6

    import openmatrix as omx
    from stage5_destination_choice import _skim_flat, _skim_lookup
    skims     = omx.open_file('downcasted_matrices_base.omx', 'r')
    sk        = _skim_lookup(_skim_flat(skims))
    try:
        scheduled_dest = stage5b(out, scheduled_dest, landuse)
    except Exception as _s5b_e:
        import traceback as _tb_s5b
        logger.warning(
            f"Stage 5b failed (non-fatal): {_s5b_e!r}\n{_tb_s5b.format_exc(limit=8)}"
        )
    plan, _   = stage6(out, scheduled_dest, landuse,
                       stage5_models=s5_models, skims=skims, skim_lookup=sk)
    plot(plan)
    spatial_plots(plan, synth, landuse)
    if skims:
        skims.close()
    logger.info("GA pipeline complete.")


def GA_STAGED(filepath: str = None):
    """Run GA stage-by-stage, then run full pipeline.

    Each ga_stageN call uses only HHTS data.  Synth population is loaded
    once after all GA estimation completes, before pipeline prediction.

    NOTE — load order matters.
    An earlier version of this function loaded the synthetic population
    BEFORE running GA stages, then interleaved GA estimation and pipeline
    prediction per stage (GA-1 → pipeline-1 → GA-2 → pipeline-2 → ...).
    That pattern wastes memory and walltime because Stage 3 alone takes
    ~52 min on 3.7 M persons.  The correct order enforced here is:
        1. load_data (HHTS survey only)
        2. all GA stages          — HHTS only, no synth needed
        3. load_synth_all_zones   — once, AFTER GA completes
        4. full pipeline stages   — in sequence on synth

    Walltime note (PBS select=1:ncpus=1:mem=94GB, walltime=23:59:00):
    GA stages 1-5 ≈ 3.5 h | synth load ≈ 6 m | pipeline stages ≈ 4-5 h
    Total ≈ 9-10 h — well within the 24 h walltime for all-zone runs.
    """
    sc, out, landuse = load_data()

    # ── GA estimation for all stages (HHTS only) ──────────────────────
    # When GA_STAGED_REUSE_ESTIMATOR=1, load ga_results.pkl from a prior 'ga'
    # run dir instead of re-running estimation.  This lets the estimation job
    # use a lighter memory allocation (no synth population) while this job only
    # pays the 250 GB cost once for prediction.  Resolution order:
    #   1. GA_STAGED_GA_DIR env var (explicit path)
    #   2. runs/main_ga_<estimator>/LATEST_RUN.txt  (per-estimator alias)
    #   3. runs/main_ga/LATEST_RUN.txt              (shared alias fallback)
    # Falls back to full re-estimation if no valid pkl is found.
    _reuse_est = os.environ.get('GA_STAGED_REUSE_ESTIMATOR', '0').strip() == '1'
    if _reuse_est:
        _ga_dir = os.environ.get('GA_STAGED_GA_DIR', '').strip()
        if not _ga_dir:
            _est_key = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
            _run_root_env = os.environ.get('PIPELINE_RUN_ROOT', 'runs')
            for _alias_cand in [
                os.path.join(_run_root_env, f'main_ga_{_est_key}', 'LATEST_RUN.txt'),
                os.path.join(_run_root_env, 'main_ga', 'LATEST_RUN.txt'),
            ]:
                if os.path.isfile(_alias_cand):
                    with open(_alias_cand, 'r', encoding='utf-8') as _f:
                        _ga_dir = _f.read().strip()
                    if _ga_dir:
                        logger.info(f"[GA_STAGED] resolved prior ga dir via {_alias_cand}")
                        break
        _ga_pkl = os.path.join(_ga_dir, 'ga_results.pkl') if _ga_dir else ''
        if _ga_pkl and os.path.isfile(_ga_pkl):
            logger.info(
                f"[GA_STAGED] GA_STAGED_REUSE_ESTIMATOR=1: loading ga_results.pkl "
                f"from prior 'ga' run — skipping re-estimation.\n"
                f"  source: {_ga_pkl}"
            )
            with open(_ga_pkl, 'rb') as _f:
                all_ga = pickle.load(_f)
        else:
            # Secondary fallback: if GA_STAGED_GA_DIR points to a 'merged' subdir
            # but that pkl is missing, try to auto-merge from adjacent stage PKLs.
            # This handles the case where merge_ga_results.py didn't run (e.g. PBS
            # job killed) but all per-stage ga_results.pkl files exist.
            _auto_merged = False
            if _ga_dir and os.path.basename(_ga_dir) == 'merged':
                _stage_base = os.path.dirname(_ga_dir)
                from ga_hypothesis_testing import AllStagesResult as _ASR
                _all_ga_try = _ASR()
                for _sid in range(6):
                    _sp = os.path.join(_stage_base, f'stage{_sid}', 'ga_results.pkl')
                    if not os.path.isfile(_sp):
                        continue
                    try:
                        with open(_sp, 'rb') as _sf:
                            _sobj = pickle.load(_sf)
                        if isinstance(_sobj, _ASR) and _sid in _sobj.results:
                            _all_ga_try.results[_sid] = _sobj.results[_sid]
                        elif isinstance(_sobj, _ASR) and len(_sobj.results) == 1:
                            _all_ga_try.results[_sid] = next(iter(_sobj.results.values()))
                    except Exception as _sme:
                        logger.warning(f"[GA_STAGED] auto-merge stage {_sid}: {_sme}")
                if _all_ga_try.results:
                    all_ga = _all_ga_try
                    _auto_merged = True
                    # Save the merged pkl so the merge step is skipped next time
                    os.makedirs(_ga_dir, exist_ok=True)
                    with open(_ga_pkl, 'wb') as _mf:
                        pickle.dump(all_ga, _mf)
                    logger.info(
                        f"[GA_STAGED] auto-merged {len(all_ga.results)} stage PKLs "
                        f"→ {_ga_pkl}"
                    )
            if not _auto_merged:
                logger.warning(
                    f"[GA_STAGED] GA_STAGED_REUSE_ESTIMATOR=1 requested but no valid "
                    f"ga_results.pkl found (tried: {_ga_pkl!r}) — "
                    f"falling back to full re-estimation"
                )
                all_ga = _run_ga_stages_with_checkpoints(out, landuse)
    else:
        # Checkpoints are written after every stage to avoid all-or-nothing runs.
        all_ga = _run_ga_stages_with_checkpoints(out, landuse)

    # ── Export per-stage history CSVs + combined convergence plot ─────────
    try:
        ga_reports_dir = os.path.join(OUT_DIR, 'ga_reports')
        all_ga.export_reports(ga_reports_dir)
        # Combined multi-panel convergence figure (all stages, all segments)
        all_ga.plot_all(save_path=os.path.join(OUT_DIR, 'plots', 'ga_all_stages.png'))
        logger.info(f"GA reports exported → {ga_reports_dir}/")
        logger.info(f"GA combined plot   → {OUT_DIR}/plots/ga_all_stages.png")
    except Exception as _gae:
        logger.warning(f"GA report export failed: {_gae}")

    # ── Load synth only after all GA stages complete ───────────────────
    synth = (load_synth(filepath, landuse=landuse) if filepath and os.path.isfile(filepath)
             else load_synth_all_zones(SYNTH_DIR, landuse=landuse))

    synth = usual_locations(synth, landuse, hhts_tours=out.get('tours'))
    _s0_res = all_ga.results.get(0) if all_ga is not None else None
    synth = cdap(out, synth, feature_override=getattr(_s0_res, 'best_features', None))
    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} persons are travellers")

    # ── Apply GA-selected features through stages 1–5 ─────────────────
    from ga_hypothesis_testing import apply_ga_results_to_pipeline
    _prev_larch_engine = os.environ.get('GA_LARCH_ENGINE')
    _prev_strict_s5 = os.environ.get('STAGE5_STRICT_ESTIMATION')
    try:
        if os.name != 'nt':
            os.environ['GA_LARCH_ENGINE'] = 'jax'
        os.environ['STAGE5_STRICT_ESTIMATION'] = '1'
        scheduled_dest, s5_models = apply_ga_results_to_pipeline(
            ga_results = all_ga,
            out        = out,
            synth      = travellers,
            landuse    = landuse,
            skims_path = PIPELINE_SKIMS_PATH,
            save_dir   = OUT_DIR,
        )
    finally:
        if _prev_larch_engine is None:
            os.environ.pop('GA_LARCH_ENGINE', None)
        else:
            os.environ['GA_LARCH_ENGINE'] = _prev_larch_engine
        if _prev_strict_s5 is None:
            os.environ.pop('STAGE5_STRICT_ESTIMATION', None)
        else:
            os.environ['STAGE5_STRICT_ESTIMATION'] = _prev_strict_s5
    # ── Post-GA diagnostics (coefficients + HHTS validation) ──────────
    _ga_post_estimation_report(s5_models, out, OUT_DIR)

    import openmatrix as omx
    from stage5_destination_choice import _skim_flat, _skim_lookup
    skims   = omx.open_file(PIPELINE_SKIMS_PATH, 'r')
    sk      = _skim_lookup(_skim_flat(skims))
    try:
        scheduled_dest = stage5b(out, scheduled_dest, landuse)
    except Exception as _s5b_e:
        import traceback as _tb_s5b
        logger.warning(
            f"Stage 5b failed (non-fatal): {_s5b_e!r}\n{_tb_s5b.format_exc(limit=8)}"
        )
    plan, _ = stage6(out, scheduled_dest, landuse,
                     stage5_models=s5_models, skims=skims, skim_lookup=sk)
    plot(plan)
    spatial_plots(plan, synth, landuse)
    if skims:
        skims.close()

    # ── GA_STAGED validation summary ──────────────────────────────────────
    # After the full pipeline runs, print and save a validation summary so
    # the quality of the GA-selected model can be judged in one place.
    print("\n" + "=" * 60)
    print("GA STAGED — VALIDATION SUMMARY")
    print("=" * 60)
    val_rows = []
    try:
        # Mode share validation
        trips_out = plan[plan.get('row_type', pd.Series()) == 'trip'] \
                    if 'row_type' in plan.columns else plan
        if 'mode_name' in trips_out.columns:
            mode_share = trips_out['mode_name'].value_counts(normalize=True) * 100
            print("\n  Mode share (predicted):")
            for mode, share in mode_share.sort_values(ascending=False).items():
                print(f"    {mode:<12} {share:5.1f}%")
                val_rows.append({'metric': f'mode_share_{mode}', 'value': round(share, 2), 'unit': '%'})
    except Exception as _e:
        logger.warning(f"  Mode share validation failed: {_e}")

    try:
        # Zone validation
        from stage6_plan_finalisation import compute_zone_validation_metrics
        zv = compute_zone_validation_metrics(plan, verbose=True)
        for k, v in zv.items():
            if k != 'catchall_by_mode' and isinstance(v, (int, float)):
                val_rows.append({'metric': f'zone_{k}', 'value': v, 'unit': ''})
    except Exception as _e:
        logger.warning(f"  Zone validation failed: {_e}")

    try:
        # Trip count / activity count summary
        n_trips = int((plan['row_type'] == 'trip').sum()) if 'row_type' in plan.columns else len(plan)
        n_acts  = int((plan['row_type'] == 'activity').sum()) if 'row_type' in plan.columns else 0
        n_pax   = plan['person_id'].nunique() if 'person_id' in plan.columns else 0
        trips_per = n_trips / max(n_pax, 1)
        print(f"\n  Trips: {n_trips:,}  Activities: {n_acts:,}  Persons: {n_pax:,}")
        print(f"  Avg trips/person: {trips_per:.2f}")
        val_rows += [
            {'metric': 'n_trips',           'value': n_trips,              'unit': 'trips'},
            {'metric': 'n_activities',       'value': n_acts,               'unit': 'activities'},
            {'metric': 'n_persons',          'value': n_pax,                'unit': 'persons'},
            {'metric': 'trips_per_person',   'value': round(trips_per, 3),  'unit': 'trips/person'},
        ]
    except Exception as _e:
        logger.warning(f"  Trip/activity summary failed: {_e}")

    try:
        # Travel time summary
        tt_col = next((c for c in ['travtime', 'tt_avg'] if c in plan.columns), None)
        if tt_col:
            trip_rows = plan[plan['row_type'] == 'trip'] if 'row_type' in plan.columns else plan
            tt = pd.to_numeric(trip_rows[tt_col], errors='coerce').dropna()
            if tt.median() > 200:
                tt = tt / 60.0
            print(f"\n  Travel time (trips): mean={tt.mean():.1f}min  median={tt.median():.1f}min  "
                  f"p90={tt.quantile(0.9):.1f}min")
            val_rows += [
                {'metric': 'travtime_mean_min',   'value': round(float(tt.mean()),   2), 'unit': 'min'},
                {'metric': 'travtime_median_min', 'value': round(float(tt.median()), 2), 'unit': 'min'},
                {'metric': 'travtime_p90_min',    'value': round(float(tt.quantile(0.9)), 2), 'unit': 'min'},
            ]
    except Exception as _e:
        logger.warning(f"  Travel time summary failed: {_e}")

    # Save validation CSV
    if val_rows:
        val_path = os.path.join(OUT_DIR, 'ga_staged_validation.csv')
        pd.DataFrame(val_rows).to_csv(val_path, index=False)
        print(f"\n  Validation saved → {val_path}")

    print("=" * 60)
    logger.info("GA staged pipeline complete.")


# ---------------------------------------------------------------------------
def GA_STAGE5_SMOKE(filepath: str = None):
    """Run only GA Stage 5 with an ultra-small budget for wiring checks.

    This mode is intentionally minimal and does not run full synth prediction.
    It exists to quickly verify Stage 5 GA setup, checkpoint writing, and
    report generation without paying the full runtime of ga/ga_staged.
    """
    import pickle

    logger.info("Starting Stage 5-only GA smoke test (no full pipeline run).")
    sc, out, landuse = load_data()

    prev_budget = os.environ.get('GA_BUDGET')
    os.environ['GA_BUDGET'] = 'smoke'
    try:
        res = ga_stage5(out, landuse)
    finally:
        if prev_budget is None:
            os.environ.pop('GA_BUDGET', None)
        else:
            os.environ['GA_BUDGET'] = prev_budget

    from ga_hypothesis_testing import AllStagesResult
    all_ga = AllStagesResult()
    all_ga.results[5] = res

    # Save smoke artifacts without overwriting the main staged output file.
    smoke_pkl = os.path.join(OUT_DIR, 'ga_results_stage5_smoke.pkl')
    with open(smoke_pkl, 'wb') as _f:
        pickle.dump(all_ga, _f)
    logger.info(f"Stage 5 smoke GA results saved -> {smoke_pkl}")

    # Standard checkpoint/report bundle for parity with full staged runs.
    _save_ga_checkpoint(all_ga, stage_id=5, failed=False)
    _hist = getattr(res, 'history', []) if res is not None else []
    _hist_df = pd.DataFrame(_hist) if _hist else None
    _write_stage_checkpoint_report(
        stage_tag='ga_stage5_smoke',
        title='GA Stage 5 Smoke Test - Selected Model & Convergence',
        summary=_format_ga_checkpoint_summary(res),
        table=_hist_df,
    )

    try:
        ga_reports_dir = os.path.join(OUT_DIR, 'ga_reports')
        all_ga.export_reports(ga_reports_dir)
        logger.info(f"GA smoke reports exported -> {ga_reports_dir}/")
    except Exception as _gae:
        logger.warning(f"GA smoke report export failed: {_gae}")

    try:
        mode_feats = res.best_features.get('mode') if getattr(res, 'best_features', None) else None
        dest_feats = res.best_features.get('dest') if getattr(res, 'best_features', None) else None

        logger.info("Stage 5 smoke: fitting final selected Stage 5 models with safety terms...")
        _fit_stage5_smoke_models(
            out=out,
            landuse=landuse,
            save_dir=OUT_DIR,
            mode_feature_override=mode_feats,
            dest_feature_override=dest_feats,
            safety_mode='baseline',
            use_gpu=True,
        )

        nosafety_dir = os.path.join(OUT_DIR, 'safety_nosafety')
        logger.info("Stage 5 smoke: refitting the same selected Stage 5 models without safety terms...")
        _fit_stage5_smoke_models(
            out=out,
            landuse=landuse,
            save_dir=nosafety_dir,
            mode_feature_override=mode_feats,
            dest_feature_override=dest_feats,
            use_gpu=True,
            safety_mode='nosafety',
        )

        safety_out_dir = os.path.join(OUT_DIR, 'safety_analysis')
        compare_safety_runs(
            baseline_dir=OUT_DIR,
            nosafety_dir=nosafety_dir,
            out_dir=safety_out_dir,
            landuse=landuse,
        )
    except Exception as _s5e:
        import traceback as _tb_s5
        logger.warning(f"Stage 5 smoke final estimation/reporting failed: {_s5e}")
        logger.debug(_tb_s5.format_exc())

    logger.info("Stage 5-only GA smoke test complete.")


# ---------------------------------------------------------------------------
def compare_safety_runs(
    baseline_dir: str = 'runs/safety_baseline',
    nosafety_dir: str = 'runs/safety_nosafety',
    out_dir:      str = 'runs/safety_analysis',
    landuse: Optional[pd.DataFrame] = None,
) -> None:
    """
    Core paper analysis: compare Stage 5 models estimated WITH vs WITHOUT
    crash/safety skims.

    Produces:
      {out_dir}/safety_comparison.csv     — numeric table of all metrics
      {out_dir}/safety_report.html        — full HTML report (figures inline)
      {out_dir}/plots/*.png               — individual figures

    Statistics reported
    -------------------
    1. Likelihood Ratio Test (LRT) for each crash coefficient:
          LRT = 2 × (LL_with_crash − LL_without_crash)
          df  = number of crash parameters
          Follows χ²(df) under H₀ (crash vars add no fit)
          p < 0.05 → safety skims improve model fit significantly.

    2. McFadden ρ² with vs without crash:
          ρ² = 1 − LL_model / LL_null
          Δρ² = ρ²_with − ρ²_without  (positive = crash helps)

    3. Crash coefficient t-ratios — are they significant?
          t = β / se(β)  → p = 2 × Φ(−|t|)
          Reports β, se, t, p for CRASH and CRASHPT in each segment.

    4. Mode share shift — how much does crash change mode share?
          Δ(Walk %) = Walk%_with − Walk%_without
          Δ(Bike %) = Bike%_with − Bike%_without
          Positive Δ = safer routes attract more active travel.

    5. Destination distance distribution shift:
          Δ(mean dist) and Δ(dist p90) — does perceived safety
          shift where people go?

    6. Zone-level accessibility change:
          Active-mode accessibility with vs without safety penalisation.
    """
    import scipy.stats as _stats

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)

    logger.info(f"Safety comparison: {baseline_dir!r} vs {nosafety_dir!r} → {out_dir!r}")

    # ── Load coefficient CSVs ───────────────────────────────────────────────
    def _load_coef(d):
        for fn in ('stage5_models_coefficients.csv', 'stage5_tour_coefficients.csv'):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                return pd.read_csv(p)
        return pd.DataFrame()

    def _load_loc_obs(d):
        """Load HHTS location-choice observations (OTAZ, DTAZ, mode, dist_km)."""
        p = os.path.join(d, 'location_choice', 'location_choice_observations.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            logger.info(f"  Loaded location_choice_observations from {d}: {len(df):,} rows")
            return df
        return pd.DataFrame()

    def _load_mode_validation(d):
        """Load mode_share_validation.csv from location_choice dir."""
        p = os.path.join(d, 'location_choice', 'mode_share_validation.csv')
        if os.path.exists(p):
            return pd.read_csv(p)
        return pd.DataFrame()

    def _load_dest_validation(d):
        """Load location_choice_validation.csv (zone-level pred_share)."""
        p = os.path.join(d, 'location_choice', 'location_choice_validation.csv')
        if os.path.exists(p):
            return pd.read_csv(p)
        return pd.DataFrame()

    coef_base = _load_coef(baseline_dir)
    coef_nosa = _load_coef(nosafety_dir)
    obs_base  = _load_loc_obs(baseline_dir)
    obs_nosa  = _load_loc_obs(nosafety_dir)
    ms_val_base = _load_mode_validation(baseline_dir)
    ms_val_nosa = _load_mode_validation(nosafety_dir)
    dv_base = _load_dest_validation(baseline_dir)
    dv_nosa = _load_dest_validation(nosafety_dir)

    rows = []    # accumulate result rows

    # ── 1. Crash coefficient statistics ────────────────────────────────────
    crash_cols = [c for c in getattr(coef_base, 'columns', [])
                  if 'crash' in str(c).lower() or 'CRASH' in str(c)]

    if not coef_base.empty:
        _coef_id  = next((c for c in coef_base.columns
                          if c.lower() in ('parameter', 'coef_name', 'coefficient_name', 'name', 'variable')),
                         coef_base.columns[0])
        _pairs = [(coef_base, 'with_safety')]
        if not coef_nosa.empty:
            _pairs.append((coef_nosa, 'without_safety'))
        for _df, _lbl in _pairs:
            _coef_id_local = next((c for c in _df.columns
                                   if c.lower() in ('parameter', 'coef_name', 'coefficient_name', 'name', 'variable')),
                                  _df.columns[0])
            crash_rows = _df[_df[_coef_id_local].str.lower().str.contains('crash', na=False)]
            for _, cr in crash_rows.iterrows():
                _beta = pd.to_numeric(cr.get('value', cr.get('coef', cr.get('estimate', np.nan))),
                                       errors='coerce')
                _se   = pd.to_numeric(cr.get('std_err', cr.get('se', cr.get('stderr', np.nan))),
                                       errors='coerce')
                _t    = _beta / _se if _se and _se > 0 else np.nan
                _p    = float(2 * _stats.norm.sf(abs(_t))) if np.isfinite(_t) else np.nan
                rows.append({
                    'category': 'crash_coef',
                    'run':      _lbl,
                    'segment':  str(cr.get('segment', cr.get('purpose', 'all'))),
                    'parameter': str(cr[_coef_id_local]),
                    'beta':     round(float(_beta), 4) if np.isfinite(_beta) else None,
                    'se':       round(float(_se),   4) if np.isfinite(_se)   else None,
                    't_ratio':  round(float(_t),    3) if np.isfinite(_t)    else None,
                    'p_value':  round(float(_p),    4) if np.isfinite(_p)    else None,
                    'significant': bool(_p < 0.05)     if np.isfinite(_p)    else None,
                })
        logger.info(f"  Crash coefficient rows extracted: {len(crash_rows)}")
    else:
        logger.warning("  Could not load coefficient CSVs from one or both run dirs")

    # ── 2. Log-likelihood and rho² comparison ──────────────────────────────
    def _load_ll(d):
        """Extract LL values from stage5_validation_metrics.csv if present."""
        p = os.path.join(d, 'stage5_validation_metrics.csv')
        if os.path.exists(p):
            return pd.read_csv(p)
        return pd.DataFrame()

    ll_base = _load_ll(baseline_dir)
    ll_nosa = _load_ll(nosafety_dir)

    if not ll_base.empty and not ll_nosa.empty:
        _ll_col = next((c for c in ll_base.columns if 'll' in c.lower()), None)
        _rho_col = next((c for c in ll_base.columns if 'rho' in c.lower()), None)
        _seg_col = next((c for c in ll_base.columns if 'seg' in c.lower()
                         or 'purp' in c.lower()), None)

        if _ll_col and _seg_col:
            for seg in ll_base[_seg_col].unique():
                _b = ll_base[ll_base[_seg_col] == seg]
                _n = ll_nosa[ll_nosa[_seg_col] == seg] if not ll_nosa.empty else pd.DataFrame()
                ll_b  = float(_b[_ll_col].iloc[0]) if not _b.empty else np.nan
                ll_n  = float(_n[_ll_col].iloc[0]) if not _n.empty else np.nan
                lrt   = 2 * (ll_b - ll_n) if np.isfinite(ll_b) and np.isfinite(ll_n) else np.nan
                # df = number of crash params; approximate as 2 (CRASH + CRASHPT)
                p_lrt = float(_stats.chi2.sf(lrt, df=2)) if np.isfinite(lrt) else np.nan
                rho_b = float(_b[_rho_col].iloc[0]) if _rho_col and not _b.empty else np.nan
                rho_n = float(_n[_rho_col].iloc[0]) if _rho_col and not _n.empty else np.nan
                rows.append({
                    'category':  'model_fit',
                    'run':       'comparison',
                    'segment':   str(seg),
                    'parameter': 'LL_ratio_test',
                    'beta':      round(ll_b,  2) if np.isfinite(ll_b)  else None,
                    'se':        round(ll_n,  2) if np.isfinite(ll_n)  else None,
                    't_ratio':   round(lrt,   3) if np.isfinite(lrt)   else None,
                    'p_value':   round(p_lrt, 4) if np.isfinite(p_lrt) else None,
                    'significant': bool(p_lrt < 0.05) if np.isfinite(p_lrt) else None,
                })
                delta_rho = (rho_b - rho_n) if np.isfinite(rho_b) and np.isfinite(rho_n) else np.nan
                rows.append({
                    'category':  'model_fit',
                    'run':       'comparison',
                    'segment':   str(seg),
                    'parameter': 'delta_rho2',
                    'beta':      round(rho_b, 4) if np.isfinite(rho_b) else None,
                    'se':        round(rho_n, 4) if np.isfinite(rho_n) else None,
                    't_ratio':   round(delta_rho, 4) if np.isfinite(delta_rho) else None,
                    'p_value':   None,
                    'significant': bool(delta_rho > 0) if np.isfinite(delta_rho) else None,
                })

    # ── 3. Mode share shift (from location_choice/mode_share_validation.csv) ─
    # Prefer predicted mode shares from model validation; fall back to HHTS observed.
    def _extract_ms_from_validation(mv_df):
        """Return {mode_name: pred_share_pct} from mode_share_validation.csv."""
        if mv_df.empty or 'mode_name' not in mv_df.columns:
            return {}
        # Only include rows where at least one run has non-zero predicted share
        col = 'pred_share_pct' if 'pred_share_pct' in mv_df.columns else 'obs_share_pct'
        result = {}
        for _, r in mv_df.iterrows():
            mode = str(r['mode_name'])
            val = float(r[col]) if pd.notna(r.get(col)) else 0.0
            result[mode] = result.get(mode, 0.0) + val / mv_df['segment'].nunique()
        return result

    def _obs_ms_from_validation(mv_df):
        """Return {mode_name: obs_share_pct} averaged across segments."""
        if mv_df.empty or 'mode_name' not in mv_df.columns:
            return {}
        result = {}
        for _, r in mv_df.iterrows():
            mode = str(r['mode_name'])
            val = float(r['obs_share_pct']) if pd.notna(r.get('obs_share_pct')) else 0.0
            result[mode] = result.get(mode, 0.0) + val / mv_df['segment'].nunique()
        return result

    # baseline: use predicted shares; nosafety: use observed (since nosafety predictions
    # may be zero if the model DataTree wasn't live during validation)
    ms_base_pred = _extract_ms_from_validation(ms_val_base)
    ms_nosa_pred = _extract_ms_from_validation(ms_val_nosa)
    ms_base_obs  = _obs_ms_from_validation(ms_val_base)

    # If nosafety predictions are all zero, fall back to obs for nosafety
    # and show baseline_pred vs baseline_obs (effect of including safety)
    nosa_has_preds = any(v > 0 for v in ms_nosa_pred.values())
    if not nosa_has_preds and ms_base_pred:
        logger.info("  Nosafety mode shares are zero — showing baseline pred vs HHTS observed mode shift")
        ms_with   = ms_base_pred
        ms_without = ms_base_obs
        ms_label  = 'predicted_vs_observed'
    else:
        ms_with   = ms_base_pred
        ms_without = ms_nosa_pred
        ms_label  = 'with_vs_without_safety'

    all_modes = sorted(set(ms_with.keys()) | set(ms_without.keys()))
    for mode in all_modes:
        b_val = float(ms_with.get(mode, 0))
        n_val = float(ms_without.get(mode, 0))
        rows.append({
            'category':   'mode_share',
            'run':        ms_label,
            'segment':    'all',
            'parameter':  str(mode),
            'beta':       round(b_val, 2),
            'se':         round(n_val, 2),
            't_ratio':    round(n_val - b_val, 2),
            'p_value':    None,
            'significant': None,
        })
    active_b = sum(float(ms_with.get(m, 0)) for m in ['Walk', 'Bike'])
    active_n = sum(float(ms_without.get(m, 0)) for m in ['Walk', 'Bike'])
    rows.append({
        'category': 'active_travel',
        'run': ms_label, 'segment': 'all',
        'parameter': 'active_mode_share_pct',
        'beta': round(active_b, 2), 'se': round(active_n, 2),
        't_ratio': round(active_n - active_b, 2),
        'p_value': None, 'significant': None,
    })
    logger.info(f"  Active travel ({ms_label}): with={active_b:.1f}%  without={active_n:.1f}%  Δ={active_n-active_b:+.2f}pp")

    # ── 4. Destination zone shift (from location_choice_validation.csv) ───────
    # Compare predicted zone share between baseline and nosafety models.
    dest_zone_shift_rows = []
    if not dv_base.empty:
        _dv = dv_base.copy()
        if not dv_nosa.empty and not dv_nosa['pred_share'].eq(0).all():
            _dv = _dv.merge(
                dv_nosa[['segment', 'zone', 'pred_share']].rename(columns={'pred_share': 'pred_share_nosa'}),
                on=['segment', 'zone'], how='left',
            )
            _dv['delta_pred'] = _dv['pred_share_nosa'].fillna(0) - _dv['pred_share']
        else:
            # Nosafety predictions unavailable — compare model pred vs HHTS obs
            _dv['pred_share_nosa'] = _dv['obs_share']
            _dv['delta_pred'] = _dv['obs_share'] - _dv['pred_share']
        for _, r in _dv.iterrows():
            dest_zone_shift_rows.append({
                'segment': str(r['segment']),
                'zone': int(r['zone']) if pd.notna(r['zone']) else None,
                'obs_share': float(r['obs_share']),
                'pred_share_base': float(r['pred_share']),
                'pred_share_nosa': float(r.get('pred_share_nosa', 0)),
                'delta_pred': float(r.get('delta_pred', 0)),
            })

    # Distance stats from HHTS observations
    def _dist_stats(obs_df):
        dc = next((c for c in ['trip_dist', 'dist_km', 'dist', 'DIST'] if c in obs_df.columns), None)
        if obs_df.empty or not dc:
            return {}
        vals = pd.to_numeric(obs_df[dc], errors='coerce').dropna()
        if vals.empty:
            return {}
        return {
            'mean': vals.mean(), 'median': vals.median(),
            'p25': vals.quantile(0.25), 'p75': vals.quantile(0.75),
            'p90': vals.quantile(0.90),
        }

    ds_base = _dist_stats(obs_base)
    ds_nosa = _dist_stats(obs_nosa)
    for stat in ['mean', 'median', 'p75', 'p90']:
        b_val = ds_base.get(stat, np.nan)
        n_val = ds_nosa.get(stat, np.nan)
        delta = n_val - b_val if np.isfinite(b_val) and np.isfinite(n_val) else np.nan
        rows.append({
            'category': 'trip_distance', 'run': 'comparison', 'segment': 'all',
            'parameter': f'dist_{stat}_km',
            'beta': round(b_val, 3) if np.isfinite(b_val) else None,
            'se':   round(n_val, 3) if np.isfinite(n_val) else None,
            't_ratio': round(delta, 3) if np.isfinite(delta) else None,
            'p_value': None, 'significant': None,
        })
    logger.info(f"  Mean trip dist (HHTS obs): {ds_base.get('mean', np.nan):.2f} km")

    # ── Write outputs ──────────────────────────────────────────────────────
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(os.path.join(out_dir, 'safety_comparison.csv'), index=False)

    # ── Generate figures ────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    # Figure 1 — mode share comparison
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        _ms_df = df_rows[df_rows['category'] == 'mode_share'].copy()
        modes = _ms_df['parameter'].tolist()
        xs = np.arange(len(modes))
        w  = 0.35
        ax1.bar(xs - w/2, _ms_df['beta'],   w, color='#2C6B9C', alpha=0.85, label='With safety skims')
        ax1.bar(xs + w/2, _ms_df['se'],     w, color='#E07030', alpha=0.85, label='Without safety skims')
        ax1.set_xticks(xs); ax1.set_xticklabels(modes, rotation=20, ha='right', fontsize=9)
        ax1.set_ylabel('Mode share (%)'); ax1.set_title('Mode share: safety vs no-safety')
        ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
        deltas = _ms_df['t_ratio'].tolist()
        ax2.barh(modes, deltas,
                 color=['#2ECC71' if d < 0 else '#E74C3C' for d in deltas], alpha=0.85)
        ax2.axvline(0, color='black', lw=0.8)
        ax2.set_xlabel('Δ share pp (negative = crash penalises mode)'); ax2.grid(axis='x', alpha=0.3)
        ax2.set_title('Mode share shift from removing safety skims')
        for i, d in enumerate(deltas):
            if abs(d) > 0.05:
                ax2.text(d + (0.05 if d >= 0 else -0.05), i, f'{d:+.2f}pp',
                         va='center', ha='left' if d >= 0 else 'right', fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'plots', 'safety_mode_share.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as _fe:
        logger.warning(f"  Mode share figure failed: {_fe}")

    # Figure 2 — crash coefficient forest plot
    try:
        _cc = df_rows[(df_rows['category'] == 'crash_coef') &
                      df_rows['beta'].notna() & df_rows['t_ratio'].notna()]
        if not _cc.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, len(_cc) * 0.5 + 1.5)), facecolor='white')
            labels  = [f"{r['run'][:4]}|{r['segment']}|{r['parameter']}"
                       for _, r in _cc.iterrows()]
            betas   = _cc['beta'].tolist()
            ses     = _cc['se'].fillna(0).tolist()
            colors  = ['#2C6B9C' if 'with' in lb else '#E07030' for lb in labels]
            ys      = np.arange(len(labels))
            ax.barh(ys, betas, xerr=ses, color=colors, alpha=0.8, capsize=3)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Coefficient β  (negative = crash deters travel on that OD)')
            ax.set_title('Crash/safety coefficient estimates')
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color='#2C6B9C', label='With safety'),
                                Patch(color='#E07030', label='Without safety')],
                      fontsize=9)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, 'plots', 'safety_crash_coefs.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
    except Exception as _fe:
        logger.warning(f"  Crash coef figure failed: {_fe}")

    # ── Figure 3 — Activity level comparison (with vs without safety) ────────
    # Shows how safety skims shift the activity-type distribution:
    # if crash-penalised ODs make certain activity zones less attractive,
    # those trip purposes will be suppressed.
    try:
        def _activity_dist(plan):
            """Return activity type counts (normalised %) from a plan DataFrame."""
            if plan.empty:
                return pd.Series(dtype=float)
            purp_col = next(
                (c for c in ['purpose', 'primary_purpose', 'main_purpose', 'activity']
                 if c in plan.columns), None
            )
            if purp_col is None:
                return pd.Series(dtype=float)
            # Count trip (not activity) rows for mode share consistency
            if 'row_type' in plan.columns:
                rows = plan[plan['row_type'].isin(['trip', 'activity'])]
            else:
                rows = plan
            return rows[purp_col].value_counts(normalize=True) * 100

        act_base = _activity_dist(obs_base)
        act_nosa = _activity_dist(obs_nosa)

        if not act_base.empty or not act_nosa.empty:
            all_acts = sorted(set(act_base.index) | set(act_nosa.index))
            # Filter to meaningful activity types (drop 'home', NaN, numeric)
            all_acts = [a for a in all_acts if isinstance(a, str) and a.lower() not in ('home', 'nan', '')]
            if all_acts:
                b_vals = [float(act_base.get(a, 0)) for a in all_acts]
                n_vals = [float(act_nosa.get(a, 0)) for a in all_acts]
                deltas = [n - b for n, b in zip(n_vals, b_vals)]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(all_acts) * 0.55 + 1.5)),
                                               facecolor='white')
                xs = np.arange(len(all_acts))
                w  = 0.35
                ax1.barh(xs - w/2, b_vals, w, color='#2C6B9C', alpha=0.85, label='With safety skims')
                ax1.barh(xs + w/2, n_vals, w, color='#E07030', alpha=0.85, label='Without safety skims')
                ax1.set_yticks(xs)
                ax1.set_yticklabels([a.replace('_', ' ').title() for a in all_acts], fontsize=9)
                ax1.set_xlabel('Activity share (%)')
                ax1.set_title('Activity type distribution\n(with vs without safety skims)', fontsize=10)
                ax1.legend(fontsize=9)
                ax1.grid(axis='x', alpha=0.3)

                # Delta panel — highlight which activities are pushed by safety
                colors_d = ['#27AE60' if d < -0.05 else '#E74C3C' if d > 0.05 else '#95A5A6'
                            for d in deltas]
                ax2.barh(xs, deltas, color=colors_d, alpha=0.85)
                ax2.axvline(0, color='black', lw=0.8)
                ax2.set_yticks(xs)
                ax2.set_yticklabels([a.replace('_', ' ').title() for a in all_acts], fontsize=9)
                ax2.set_xlabel('Δ activity share pp\n(positive = more without safety; green = safety suppresses)')
                ax2.set_title('Safety skim effect on activity distribution', fontsize=10)
                for i, d in enumerate(deltas):
                    if abs(d) > 0.1:
                        ax2.text(d + (0.05 if d >= 0 else -0.05), i, f'{d:+.2f}pp',
                                 va='center', ha='left' if d >= 0 else 'right', fontsize=8)
                ax2.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, 'plots', 'safety_activity_levels.png'),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info("  Activity-level safety figure saved.")
    except Exception as _fe3:
        logger.warning(f"  Activity level figure failed: {_fe3}")

    # ── Figure 4 — Zone derivation: which zones are pushed by safety skims ───
    # Identify zones where removing safety skims changes predicted mode share
    # or trip volume the most.  Rank zones by |Δ active travel share| and
    # plot against their OSM/landuse attributes to expose the driver.
    _zone_df = pd.DataFrame()
    try:
        def _zone_active_share(plan):
            """Return per-zone active-travel share."""
            if plan.empty:
                return pd.Series(dtype=float)
            t = plan[plan['row_type'] == 'trip'] if 'row_type' in plan.columns else plan
            zcol = next((c for c in ['OTAZ', 'origin_zone', 'ORIGZONE'] if c in t.columns), None)
            mcol = next((c for c in ['mode', 'mode_name'] if c in t.columns), None)
            if zcol is None or mcol is None:
                return pd.Series(dtype=float)
            m = t[mcol].map(lambda x: x if isinstance(x, str) else
                             {1:'DA',2:'SR',3:'Walk',4:'Bike',5:'Transit'}.get(int(x),'DA'))
            is_active = m.isin(['Walk', 'Bike'])
            return t.groupby(zcol).apply(lambda g: is_active.loc[g.index].mean() * 100)

        za_base = _zone_active_share(obs_base)
        za_nosa = _zone_active_share(obs_nosa)

        if not za_base.empty and not za_nosa.empty:
            common = za_base.index.intersection(za_nosa.index)
            _zone_df = pd.DataFrame({
                'zone':         common,
                'active_with':  za_base.reindex(common).values,
                'active_without': za_nosa.reindex(common).values,
            })
            _zone_df['delta_active'] = _zone_df['active_without'] - _zone_df['active_with']
            _zone_df['abs_delta']    = _zone_df['delta_active'].abs()
            _zone_df = _zone_df.sort_values('abs_delta', ascending=False)

            # Try to merge landuse attributes (OSM safe features, crash exposure)
            _lu_path = os.path.join('saved_data', 'accessibility.csv')
            if not os.path.exists(_lu_path):
                _lu_path = os.path.join('pddopulation_by_order_with_osm_and_landuse.csv')
            if os.path.exists(_lu_path):
                _lu = pd.read_csv(_lu_path)
                _zone_id_col = next(
                    (c for c in ['ORDER', 'zone_id', 'ZONE_ID', 'SA1_7DIGITCODE_2021']
                     if c in _lu.columns), _lu.columns[0]
                )
                _lu = _lu.rename(columns={_zone_id_col: 'zone'})
                _lu['zone'] = pd.to_numeric(_lu['zone'], errors='coerce')
                _zone_df = _zone_df.merge(_lu, on='zone', how='left')

            # Safe/walkable features from OSM columns
            _safe_cols = [c for c in _zone_df.columns if any(
                kw in c.lower() for kw in
                ('park', 'walk', 'bike', 'cycl', 'crash', 'safe', 'footway',
                 'pedestrian', 'recreational', 'recreation_ground', 'residential')
            )]

            # Write zone ranking CSV
            _zone_df.to_csv(os.path.join(out_dir, 'safety_zone_ranking.csv'), index=False)
            logger.info(f"  Zone ranking saved → {out_dir}/safety_zone_ranking.csv  "
                        f"({len(_zone_df)} zones)")

            # Plot: top-20 zones most affected by safety skims vs their OSM features
            top20 = _zone_df.head(20)
            n_feat = min(len(_safe_cols), 4)

            if n_feat > 0:
                fig, axes = plt.subplots(1, n_feat + 1,
                                         figsize=(4 * (n_feat + 1), 5),
                                         facecolor='white')
                axes = np.atleast_1d(axes)
                # Left panel: delta active travel
                ax0 = axes[0]
                colors_z = ['#27AE60' if d < 0 else '#E74C3C' for d in top20['delta_active']]
                ax0.barh(range(len(top20)), top20['delta_active'], color=colors_z, alpha=0.85)
                ax0.axvline(0, color='black', lw=0.8)
                ax0.set_yticks(range(len(top20)))
                ax0.set_yticklabels([f"Zone {z}" for z in top20['zone']], fontsize=7)
                ax0.set_xlabel('Δ active travel (pp)')
                ax0.set_title('Active travel shift\n(top-20 zones)', fontsize=9)
                ax0.grid(axis='x', alpha=0.3)

                for k, feat in enumerate(_safe_cols[:n_feat]):
                    ax = axes[k + 1]
                    vals = pd.to_numeric(top20[feat], errors='coerce').fillna(0)
                    ax.barh(range(len(top20)), vals, color='#8E44AD', alpha=0.7)
                    ax.set_yticks(range(len(top20)))
                    ax.set_yticklabels([])
                    ax.set_xlabel(feat.replace('_', ' '), fontsize=8)
                    ax.set_title(feat.replace('_', ' ').title(), fontsize=8)
                    ax.grid(axis='x', alpha=0.3)

                plt.suptitle('Zones most affected by safety skims — landuse / OSM attributes',
                             fontsize=10, fontweight='bold')
                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, 'plots', 'safety_zone_derivation.png'),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info("  Zone derivation figure saved.")
            else:
                # Simpler plot if no safe feature columns found
                fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
                colors_z = ['#27AE60' if d < 0 else '#E74C3C' for d in top20['delta_active']]
                ax.barh(range(len(top20)), top20['delta_active'], color=colors_z, alpha=0.85)
                ax.axvline(0, color='black', lw=0.8)
                ax.set_yticks(range(len(top20)))
                ax.set_yticklabels([f"Zone {z}" for z in top20['zone']], fontsize=8)
                ax.set_xlabel('Δ active travel share (pp)')
                ax.set_title('Top-20 zones: active travel shift from removing safety skims\n'
                             '(green = safety suppresses active travel; red = safety promotes it)',
                             fontsize=9)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, 'plots', 'safety_zone_derivation.png'),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
    except Exception as _fe4:
        import traceback as _tb_z
        logger.warning(f"  Zone derivation figure failed: {_fe4}")
        logger.debug(_tb_z.format_exc())

    # ── HTML report ────────────────────────────────────────────────────────
    _img = lambda fn: (
        f"<img src='plots/{fn}' style='max-width:100%;margin-bottom:20px'>"
        if os.path.exists(os.path.join(out_dir, 'plots', fn)) else ""
    )

    # Crash coeff section
    _cc_rows = df_rows[df_rows['category'] == 'crash_coef']
    _cc_html = ""
    if not _cc_rows.empty:
        _cc_html = "<table><tr><th>Run</th><th>Segment</th><th>Parameter</th>" \
                   "<th>β</th><th>SE</th><th>t</th><th>p</th><th>Sig?</th></tr>"
        for _, r in _cc_rows.iterrows():
            _sig = "✓ yes" if r.get('significant') else ("✗ no" if r.get('significant') is not None else "—")
            _p_style = " style='color:green;font-weight:bold'" if r.get('significant') else ""
            _cc_html += (f"<tr><td>{r['run']}</td><td>{r['segment']}</td>"
                         f"<td>{r['parameter']}</td>"
                         f"<td style='text-align:right'>{r['beta'] if r['beta'] is not None else '—'}</td>"
                         f"<td style='text-align:right'>{r['se'] if r['se'] is not None else '—'}</td>"
                         f"<td style='text-align:right'>{r['t_ratio'] if r['t_ratio'] is not None else '—'}</td>"
                         f"<td style='text-align:right'{_p_style}>{r['p_value'] if r['p_value'] is not None else '—'}</td>"
                         f"<td>{_sig}</td></tr>")
        _cc_html += "</table>"

    # Model fit (LRT) section
    _lrt_rows = df_rows[df_rows['category'] == 'model_fit']
    _lrt_html = ""
    if not _lrt_rows.empty:
        _lrt_html = ("<table><tr><th>Segment</th><th>Metric</th>"
                     "<th>With safety</th><th>Without safety</th><th>Δ / LRT stat</th>"
                     "<th>p (χ²)</th><th>Interpretation</th></tr>")
        for _, r in _lrt_rows.iterrows():
            if r['parameter'] == 'LL_ratio_test':
                _interp = (f"LRT={r['t_ratio']:.2f}, p={r['p_value']:.4f}. "
                            "Crash vars <b>significantly improve</b> model fit." 
                            if r.get('significant') else
                            "Crash vars do <b>not</b> significantly improve fit.")
                _lrt_html += (f"<tr><td>{r['segment']}</td><td>LL (Ratio Test)</td>"
                              f"<td>{r['beta']:.1f}</td><td>{r['se']:.1f}</td>"
                              f"<td>{r['t_ratio']:.2f}</td>"
                              f"<td{' style=color:green;font-weight:bold' if r.get('significant') else ''}>"
                              f"{r['p_value']:.4f}</td><td>{_interp}</td></tr>")
            elif r['parameter'] == 'delta_rho2':
                _interp = ("Δρ² > 0: safety skims improve model fit." 
                            if r.get('significant') else "Δρ² ≤ 0: no improvement from crash vars.")
                _lrt_html += (f"<tr><td>{r['segment']}</td><td>McFadden ρ²</td>"
                              f"<td>{r['beta']:.4f}</td><td>{r['se']:.4f}</td>"
                              f"<td>{r['t_ratio']:+.4f}</td><td>—</td>"
                              f"<td>{_interp}</td></tr>")
        _lrt_html += "</table>"

    # Mode share section
    _ms_df_html = df_rows[df_rows['category'] == 'mode_share']
    _ms_html = ""
    if not _ms_df_html.empty:
        _ms_html = ("<table><tr><th>Mode</th><th>With safety (%)</th>"
                    "<th>Without safety (%)</th><th>Δ (pp)</th>"
                    "<th>Interpretation</th></tr>")
        for _, r in _ms_df_html.iterrows():
            delta = r['t_ratio']
            _interp = ("↑ removing crash increases share" if delta > 0.1
                       else "↓ crash penalisation reduces share" if delta < -0.1
                       else "≈ negligible change")
            _ms_html += (f"<tr><td>{r['parameter']}</td>"
                         f"<td style='text-align:right'>{r['beta']:.1f}</td>"
                         f"<td style='text-align:right'>{r['se']:.1f}</td>"
                         f"<td style='text-align:right'>{delta:+.2f}</td>"
                         f"<td>{_interp}</td></tr>")
        _ms_html += "</table>"

    _active_row = df_rows[df_rows['parameter'] == 'active_mode_share_pct']
    _act_summary = ""
    if not _active_row.empty:
        r = _active_row.iloc[0]
        _act_summary = (f"<p><strong>Active mode share (Walk+Bike):</strong> "
                        f"With safety = {r['beta']:.1f}%  |  "
                        f"Without safety = {r['se']:.1f}%  |  "
                        f"Δ = {r['t_ratio']:+.2f} pp. "
                        f"{'Removing crash skims increases active travel share (counter-intuitive — suggests the safety penalty is correctly discouraging unsafe routes).' if r['t_ratio'] < 0 else 'Removing crash skims reduces active travel share (consistent with hypothesis: crash-free zones attract active modes).'}"
                        f"</p>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Safety Skim Experiment — Model Comparison Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:30px;font-size:13px;max-width:1100px}}
h1{{color:#1a3a5c;border-bottom:2px solid #1a3a5c;padding-bottom:8px}}
h2{{color:#2c5f8a;margin-top:32px}}
h3{{color:#3a7abf}}
table{{border-collapse:collapse;width:100%;margin-bottom:20px;font-size:12px}}
th{{background:#2c5f8a;color:white;padding:7px 10px;text-align:left}}
td{{border:1px solid #ddd;padding:5px 10px}}
tr:nth-child(even){{background:#f5f8fc}}
.box{{background:#eaf2fb;border-left:4px solid #2c5f8a;padding:12px 16px;margin:16px 0}}
.highlight{{background:#d4efdf;border-left:4px solid #27ae60;padding:12px 16px;margin:16px 0}}
.warn{{background:#fdecea;border-left:4px solid #e74c3c;padding:12px 16px;margin:16px 0}}
img{{border:1px solid #ddd;border-radius:4px;margin-bottom:16px}}
</style></head><body>

<h1>Safety Skim Experiment — Model Comparison Report</h1>
<div class="box">
<strong>Research question:</strong> Do crash/safety OD skims improve the tour-level 
destination and mode choice models? Does penalising unsafe route combinations shift 
predicted travel behaviour toward active modes and shorter, safer trips?<br><br>
<strong>Runs compared:</strong><br>
&nbsp;• <em>With safety skims</em> ({baseline_dir})<br>
&nbsp;• <em>Without safety skims</em> ({nosafety_dir})
</div>

<h2>1. Crash Coefficient Estimates</h2>
<p>Negative β for CRASH/CRASHPT means travellers are less likely to choose 
destinations/modes with high crash exposure. A significant negative β 
(<em>p</em> &lt; 0.05, coloured green below) confirms safety skims carry 
behavioural information beyond the road network skim.</p>
{_cc_html or '<p>Coefficient CSV not found in run directories.</p>'}
{_img('safety_crash_coefs.png')}

<h2>2. Model Fit — Likelihood Ratio Test</h2>
<p>LRT = 2 × (LL<sub>with</sub> − LL<sub>without</sub>), χ²(<em>df</em>=2).
Reject H₀ (crash adds no fit) when <em>p</em> &lt; 0.05.</p>
{_lrt_html or '<p>stage5_validation_metrics.csv not found in run directories.</p>'}

<h2>3. Mode Share Implications</h2>
<p>If safety skims correctly penalise crash-prone ODs, we expect the <em>with-safety</em>
model to produce lower Walk and Bike shares on dangerous corridors and higher shares on 
safe ones — but aggregate mode share may shift either way depending on the SEQ network.</p>
{_ms_html or '<p>Plan output not found.</p>'}
{_act_summary}
{_img('safety_mode_share.png')}

<h2>4. Activity Level Implications</h2>
<p>Safety skims affect <em>where</em> people go, which indirectly affects <em>what</em>
activities they pursue (if crash-penalised zones host certain activity types, those
activity purposes are suppressed in the with-safety prediction).  The chart below
compares the predicted activity-type distribution across both model runs.
Green bars indicate activity types whose share is reduced by safety penalisation
(crash skims discourage travel to those activity zones).</p>
{_img('safety_activity_levels.png')}

<h2>5. Zone Derivation — Which Zones Are Being Pushed by Safety Skims?</h2>
<p>The chart below ranks the 20 zones with the largest absolute change in predicted
active-travel share (Walk + Bike) between the two runs.  Zones where removing safety
skims <em>increases</em> active travel (red bars) are zones that the crash skim was
discouraging — likely high-crash ODs.  OSM and landuse attributes (park area, cycling
infrastructure, residential density) are plotted alongside to show whether safe
physical features align with the zones being suppressed.</p>
{_img('safety_zone_derivation.png')}
<p>Full zone-level data saved to <code>safety_zone_ranking.csv</code>.</p>

<h2>6. Summary and Research Implications</h2>
<div class="highlight">
<ul>
<li>If crash coefficients are negative and significant: safety skims are <strong>behaviourally
valid</strong> — model it.</li>
<li>If Δρ² &gt; 0.002: meaningful improvement in destination choice predictive power.</li>
<li>If active travel share drops with safety skims: crash penalisation is selecting
safer but lower-speed routes, as hypothesised.</li>
<li>If LRT <em>p</em> &lt; 0.01 across all segments: strong evidence for inclusion
in the final published model.</li>
<li>Zone derivation: zones with high OSM walkability/cycling features and large
negative Δ active travel are the zones where the safety skim is actively suppressing
behaviour — these are the zones to highlight in the paper.</li>
</ul>
</div>

<p style="color:#888;font-size:11px;margin-top:40px">
Generated by pipeline_logger.py compare_safety_runs() — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</p>
</body></html>"""

    html_path = os.path.join(out_dir, 'safety_report.html')
    with open(html_path, 'w', encoding='utf-8') as _hf:
        _hf.write(html)
    logger.info(f"Safety report → {html_path}")
    logger.info(f"Safety CSV   → {os.path.join(out_dir, 'safety_comparison.csv')}")

    try:
        _write_interactive_safety_dashboard(
            baseline_dir=baseline_dir,
            nosafety_dir=nosafety_dir,
            out_dir=out_dir,
            landuse=landuse,
            comparison_df=df_rows,
            hhts_obs=obs_base,
            dest_zone_shift=dest_zone_shift_rows,
        )
    except Exception as _dash_e:
        logger.warning(f"Interactive safety dashboard failed: {_dash_e}")

    try:
        spatial_path = _write_spatial_destination_dashboard(
            baseline_dir=baseline_dir,
            nosafety_dir=nosafety_dir,
            out_dir=out_dir,
            landuse=landuse,
        )
        if spatial_path:
            print(f"  Spatial dashboard -> {spatial_path}")
    except Exception as _sp_e:
        import traceback as _tb_sp
        logger.warning(f"Spatial destination dashboard failed: {_sp_e}")
        logger.debug(_tb_sp.format_exc())

    print(f"\n{'='*60}")
    print(f"Safety comparison complete.")
    print(f"  HTML report -> {html_path}")
    print(f"  CSV data    -> {os.path.join(out_dir, 'safety_comparison.csv')}")
    print(f"  Figures     -> {os.path.join(out_dir, 'plots')}/")


def _write_spatial_destination_dashboard(
    baseline_dir: str,
    nosafety_dir: str,
    out_dir: str,
    landuse: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """
    Full interactive spatial dashboard: observed destination flows, person-type
    attraction, land-use context, and safety-induced redistribution.

    Uses baseline location_choice_observations (HHTS observed tours with OTAZ/DTAZ,
    mode, purpose, person attributes) plus zone-level model pred_share vs obs_share
    to show where safety skims shift destination probabilities and who is most affected.
    """
    import geopandas as gpd
    os.makedirs(out_dir, exist_ok=True)

    # ── Zone centroid lookup ──────────────────────────────────────────────────
    zone_centroids = pd.read_csv('seq_zone_ids_.csv')
    zone_centroids['ORDER'] = pd.to_numeric(zone_centroids['ORDER'], errors='coerce')
    zone_centroids = zone_centroids.dropna(subset=['ORDER', 'latitude', 'longitude'])
    zone_centroids = zone_centroids.set_index('ORDER')
    zname_col = next((c for c in ['SA2_NAME16', 'SA2_NAME21', 'SA2_NAME'] if c in zone_centroids.columns), None)

    def _zone_name(order):
        try:
            return str(zone_centroids.at[int(order), zname_col]) if zname_col else f'Zone {int(order)}'
        except Exception:
            return f'Zone {int(order) if order else "?"}'

    # ── Load baseline HHTS observations ──────────────────────────────────────
    obs_path = os.path.join(baseline_dir, 'location_choice', 'location_choice_observations.csv')
    if not os.path.exists(obs_path):
        logger.warning(f"Spatial dashboard: observations not found at {obs_path}")
        return None
    obs = pd.read_csv(obs_path)
    obs['DTAZ']     = pd.to_numeric(obs['choice'],     errors='coerce')
    obs['OTAZ']     = pd.to_numeric(obs['OTAZ'],       errors='coerce')
    obs['tour_mode']= pd.to_numeric(obs['tour_mode'],  errors='coerce')
    obs['trip_dist']= pd.to_numeric(obs['trip_dist'],  errors='coerce')
    obs = obs.dropna(subset=['DTAZ', 'OTAZ'])

    # ── Derive person-type groupings from obs columns ─────────────────────────
    # HHTS obs uses case/person IDs that don't match the synthetic population
    # persons file. Derive meaningful traveller types from obs fields directly.
    MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
    obs['_mode_name'] = obs['tour_mode'].map(MODE_NAMES).fillna('DA')

    # Traveller type from segment + study_level
    def _traveller_type(row):
        seg = str(row.get('segment', ''))
        sl  = int(row['study_level']) if pd.notna(row.get('study_level')) else 0
        if seg == 'work':
            return 'Worker'
        if seg == 'edu' or sl > 0:
            return 'Student'
        return 'Leisure'
    obs['_traveller_type'] = obs.apply(_traveller_type, axis=1)

    # Mobility type from mode
    def _mobility_type(mode_code):
        m = MODE_NAMES.get(int(mode_code) if pd.notna(mode_code) else 1, 'DA')
        if m in ('Walk', 'Bike'):    return 'Active'
        if m == 'Transit':           return 'Transit'
        return 'Private car'
    obs['_mobility_type'] = obs['tour_mode'].apply(_mobility_type)

    # Study level label
    STUDY_LABELS = {0: 'Non-student', 1: 'Primary', 2: 'Secondary', 3: 'Tertiary', 4: 'Postgrad'}
    obs['_study_level'] = obs['study_level'].map(STUDY_LABELS).fillna('Non-student')

    # ── Load validation CSV (pred_share vs obs_share per zone/segment) ────────
    val_path = os.path.join(baseline_dir, 'location_choice', 'location_choice_validation.csv')
    val_df   = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()

    # Aggregate pred/obs share across segments per zone
    if not val_df.empty:
        zone_val = (
            val_df.groupby('zone', as_index=False)
            .agg(obs_tours_total=('obs_tours','sum'),
                 obs_share_mean=('obs_share','mean'),
                 pred_share_mean=('pred_share','mean'))
        )
        zone_val['delta_share'] = zone_val['pred_share_mean'] - zone_val['obs_share_mean']
        zone_val['zone'] = pd.to_numeric(zone_val['zone'], errors='coerce')
        zone_val_dict = zone_val.set_index('zone').to_dict('index')
    else:
        zone_val_dict = {}

    # ── Load accessibility ────────────────────────────────────────────────────
    acc_df = pd.DataFrame()
    for acc_path in ['saved_data/accessibility.csv', 'accessibility.csv']:
        if os.path.exists(acc_path):
            acc_df = pd.read_csv(acc_path)
            break
    acc_id = next((c for c in ['zone_id', 'ORDER', 'zone'] if c in acc_df.columns), None) if not acc_df.empty else None
    if not acc_df.empty and acc_id:
        acc_df[acc_id] = pd.to_numeric(acc_df[acc_id], errors='coerce')
        acc_dict = acc_df.set_index(acc_id).to_dict('index')
    else:
        acc_dict = {}

    # ── Load safety penalty from existing dashboard JSON ──────────────────────
    dash_json_path = os.path.join(out_dir, 'safety_dashboard_data.json')
    crash_by_zone  = {}
    if os.path.exists(dash_json_path):
        try:
            with open(dash_json_path, encoding='utf-8') as _jf:
                _dash = json.load(_jf)
            for feat in _dash.get('geojson', {}).get('features', []):
                p = feat.get('properties', {})
                z = p.get('ORDER')
                if z is not None:
                    crash_by_zone[int(z)] = {
                        'crash_total': p.get('inbound_crash_total'),
                        'safety_penalty': p.get('max_dest_safety_penalty'),
                        'crash_log_mean': p.get('inbound_crash_log_mean'),
                    }
        except Exception as _je:
            logger.debug(f"Could not load safety dashboard JSON: {_je}")

    # ── Aggregate per destination zone ────────────────────────────────────────
    def _safe_vc(series, normalize=False):
        if series.dropna().empty:
            return {}
        vc = series.dropna().value_counts(normalize=normalize)
        if normalize:
            return {str(k): round(float(v)*100, 1) for k, v in vc.items()}
        return {str(k): int(v) for k, v in vc.items()}

    zone_detail = {}
    for dtaz, grp in obs.groupby('DTAZ'):
        dtaz_int = int(dtaz)
        top_origins = (
            grp.groupby('OTAZ').size()
            .nlargest(8)
            .reset_index(name='count')
            .assign(name=lambda d: d['OTAZ'].apply(_zone_name))
            .to_dict('records')
        )
        vd = zone_val_dict.get(dtaz, {})
        ad = acc_dict.get(dtaz_int, {})
        cd = crash_by_zone.get(dtaz_int, {})

        record = {
            'name':          _zone_name(dtaz_int),
            'obs_tours':     len(grp),
            'mean_dist':     round(float(grp['trip_dist'].dropna().mean()), 2) if not grp['trip_dist'].dropna().empty else None,
            'mode_share':    _safe_vc(grp['_mode_name'], normalize=True),
            'purpose_share': _safe_vc(grp['purpose'],    normalize=True),
            'segment_share': _safe_vc(grp['segment'],    normalize=True),
            'top_origins':   top_origins,
            'obs_share_mean':  round(float(vd.get('obs_share_mean', 0)), 5) if vd else None,
            'pred_share_mean': round(float(vd.get('pred_share_mean', 0)), 5) if vd else None,
            'delta_share':     round(float(vd.get('delta_share', 0)), 5) if vd else None,
            'crash_total':   cd.get('crash_total'),
            'safety_penalty':cd.get('safety_penalty'),
            'crash_log_mean':cd.get('crash_log_mean'),
            'acc_auto_work': ad.get('auto_work_acc'),
            'acc_walk':      ad.get('walk_acc'),
            'acc_transit':   ad.get('transit_acc'),
            'acc_bike':      ad.get('bike_acc'),
        }
        record['traveller_type'] = _safe_vc(grp['_traveller_type'])
        record['mobility_type']  = _safe_vc(grp['_mobility_type'])
        record['study_level']    = _safe_vc(grp['_study_level'])
        zone_detail[dtaz_int] = record

    # ── Build OD flow network (top 600 pairs) ─────────────────────────────────
    od_grp = obs.groupby(['OTAZ', 'DTAZ', '_mode_name', 'segment']).size().reset_index(name='flow')
    od_top = od_grp.nlargest(600, 'flow')
    od_flows = []
    max_flow = float(od_top['flow'].max()) if len(od_top) else 1.0
    for _, r in od_top.iterrows():
        o, d = int(r['OTAZ']), int(r['DTAZ'])
        if o not in zone_centroids.index or d not in zone_centroids.index:
            continue
        od_flows.append({
            'o': o, 'd': d,
            'oLat': float(zone_centroids.at[o, 'latitude']),
            'oLon': float(zone_centroids.at[o, 'longitude']),
            'dLat': float(zone_centroids.at[d, 'latitude']),
            'dLon': float(zone_centroids.at[d, 'longitude']),
            'flow': int(r['flow']),
            'mode': str(r['_mode_name']),
            'segment': str(r['segment']),
            'w': round(max(0.6, float(r['flow']) / max_flow * 8), 2),
        })

    # ── Build GeoJSON with all zone attributes ────────────────────────────────
    shp_path = os.path.join('saved_data', 'SA2_2021_AUST_SHP_GDA2020', 'SA2_2021_AUST_GDA2020.shp')
    if not os.path.exists(shp_path):
        logger.warning(f"Spatial dashboard: shapefile not found at {shp_path}")
        return None
    geo = gpd.read_file(shp_path).to_crs(epsg=4326)
    geo_code = next((c for c in ['SA2_MAIN16','SA2_CODE16','SA2_MAIN21','SA2_CODE21'] if c in geo.columns), None)
    zmap = zone_centroids.reset_index()[['ORDER', 'SA2_CODE'] if 'SA2_CODE' in zone_centroids.columns
                                        else ['ORDER']].copy()
    # Build ORDER → SA2_CODE lookup
    zmap2 = pd.read_csv('seq_zone_ids_.csv')
    zmap2['ORDER'] = pd.to_numeric(zmap2['ORDER'], errors='coerce')
    sa2_col = next((c for c in ['SA2_MAIN16','SA2_CODE16','SA2_MAIN21'] if c in zmap2.columns), None)
    if sa2_col:
        zmap2['_SA2_KEY'] = zmap2[sa2_col].astype(str).str.strip().str.replace(r'\.0$','',regex=True)
        order_to_sa2 = zmap2.set_index('ORDER')['_SA2_KEY'].to_dict()
    else:
        order_to_sa2 = {}

    if geo_code:
        geo['_SA2_KEY'] = geo[geo_code].astype(str).str.strip().str.replace(r'\.0$','',regex=True)
        sa2_to_order = {v: k for k, v in order_to_sa2.items()}
        geo['ORDER'] = geo['_SA2_KEY'].map(sa2_to_order)
        geo = geo[geo['ORDER'].notna()].copy()
        geo['ORDER'] = geo['ORDER'].astype(int)

        def _merge_zone(row):
            o = int(row['ORDER'])
            zd = zone_detail.get(o, {})
            vd = zone_val_dict.get(o, {})
            ad = acc_dict.get(o, {})
            cd = crash_by_zone.get(o, {})
            return {
                'ORDER': o,
                'zone_name': zd.get('name', _zone_name(o)),
                'obs_tours': zd.get('obs_tours', 0),
                'mean_dist': zd.get('mean_dist'),
                'delta_share': zd.get('delta_share'),
                'pred_share_mean': zd.get('pred_share_mean'),
                'obs_share_mean':  zd.get('obs_share_mean'),
                'crash_total':   cd.get('crash_total'),
                'safety_penalty': cd.get('safety_penalty'),
                'acc_auto_work': ad.get('auto_work_acc'),
                'acc_walk':      ad.get('walk_acc'),
                'acc_transit':   ad.get('transit_acc'),
                'pct_walk_bike': zd.get('mode_share', {}).get('Walk', 0) + zd.get('mode_share', {}).get('Bike', 0),
                'dominant_mode': max(zd.get('mode_share', {'DA': 100}), key=zd.get('mode_share', {'DA': 100}).get) if zd.get('mode_share') else 'DA',
                'dominant_purpose': max(zd.get('purpose_share', {'unknown': 100}), key=zd.get('purpose_share', {'unknown': 100}).get) if zd.get('purpose_share') else 'unknown',
            }

        geo_props = geo.apply(_merge_zone, axis=1, result_type='expand')
        for col in geo_props.columns:
            geo[col] = geo_props[col]
        geo_json = json.loads(geo.to_json())
    else:
        geo_json = {'type': 'FeatureCollection', 'features': []}

    # ── Payload ───────────────────────────────────────────────────────────────
    def _clean(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v

    def _clean_dict(d):
        return {k: _clean(v) if not isinstance(v, dict) else _clean_dict(v) for k, v in d.items()}

    zone_detail_clean = {k: _clean_dict(v) for k, v in zone_detail.items()}

    payload = {
        'geojson':     geo_json,
        'zone_detail': zone_detail_clean,
        'od_flows':    od_flows,
        'run_label':   os.path.basename(baseline_dir),
    }

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_path = os.path.join(out_dir, 'spatial_destination_dashboard.html')
    _html = _build_spatial_dashboard_html(payload)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(_html)
    logger.info(f"Spatial destination dashboard -> {html_path}")
    return html_path


def _build_spatial_dashboard_html(payload: dict) -> str:
    """Render the full interactive spatial destination dashboard HTML."""
    payload_json = json.dumps(payload, allow_nan=False)
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Spatial Destination Dashboard — Safety Analysis</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,Arial,sans-serif;background:#1a1e2e;color:#e8eaf0;height:100vh;display:flex;flex-direction:column}}
#topbar{{background:#242840;padding:10px 18px;display:flex;align-items:center;gap:16px;border-bottom:1px solid #353a5a;flex-shrink:0;flex-wrap:wrap}}
#topbar h1{{font-size:1rem;font-weight:700;color:#7eb8f7;margin-right:8px;white-space:nowrap}}
.ctrl-group{{display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#9aa3c4}}
select,input[type=range]{{background:#1a1e2e;color:#e8eaf0;border:1px solid #353a5a;border-radius:6px;padding:4px 8px;font-size:0.78rem;cursor:pointer}}
select:focus{{outline:1px solid #7eb8f7}}
.badge{{background:#353a5a;border-radius:12px;padding:2px 8px;font-size:0.72rem;color:#9aa3c4}}
#layout{{display:flex;flex:1;overflow:hidden}}
#map-wrap{{flex:1;position:relative;min-width:0}}
#map{{height:100%;width:100%}}
#sidebar{{width:380px;background:#242840;border-left:1px solid #353a5a;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}}
#zone-header{{padding:12px 14px;background:#2d3250;border-bottom:1px solid #353a5a;flex-shrink:0}}
#zone-header h2{{font-size:0.95rem;color:#7eb8f7}}
#zone-header .meta{{font-size:0.75rem;color:#9aa3c4;margin-top:3px}}
#charts{{flex:1;overflow-y:auto;padding:8px}}
.chart-panel{{background:#1a1e2e;border-radius:8px;margin-bottom:8px;padding:4px;border:1px solid #353a5a}}
.chart-panel h4{{font-size:0.73rem;color:#9aa3c4;padding:6px 8px 0;text-transform:uppercase;letter-spacing:.06em}}
.chart-panel .plotly-chart{{height:170px}}
#legend{{position:absolute;bottom:24px;left:10px;background:rgba(26,30,46,.88);border-radius:8px;padding:10px 14px;font-size:0.72rem;color:#9aa3c4;z-index:900;border:1px solid #353a5a;min-width:160px}}
#legend h4{{color:#7eb8f7;font-size:0.75rem;margin-bottom:6px}}
.leg-row{{display:flex;align-items:center;gap:7px;margin:3px 0}}
.leg-swatch{{width:16px;height:10px;border-radius:2px;flex-shrink:0}}
#tooltip{{position:absolute;top:10px;left:50%;transform:translateX(-50%);background:rgba(26,30,46,.94);border:1px solid #353a5a;border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#e8eaf0;z-index:900;pointer-events:none;max-width:300px;text-align:center;display:none}}
.od-legend{{position:absolute;bottom:24px;right:10px;background:rgba(26,30,46,.88);border-radius:8px;padding:8px 12px;font-size:0.71rem;color:#9aa3c4;z-index:900;border:1px solid #353a5a}}
</style>
</head>
<body>

<div id='topbar'>
  <h1>Destination Spatial Dashboard</h1>
  <div class='ctrl-group'>
    <label>Zone colour</label>
    <select id='metricSel'>
      <option value='obs_tours'>Observed flows (HHTS)</option>
      <option value='delta_share'>Safety model shift (pred − obs share)</option>
      <option value='pred_share_mean'>Predicted destination share</option>
      <option value='obs_share_mean'>Observed destination share</option>
      <option value='pct_walk_bike'>Active mode share (%)</option>
      <option value='safety_penalty'>Safety utility penalty</option>
      <option value='crash_total'>Inbound crash exposure</option>
      <option value='acc_walk'>Walk accessibility</option>
      <option value='acc_transit'>Transit accessibility</option>
      <option value='acc_auto_work'>Auto/work accessibility</option>
    </select>
  </div>
  <div class='ctrl-group'>
    <label>OD flows</label>
    <select id='odSeg'>
      <option value='all'>All segments</option>
      <option value='work'>Work</option>
      <option value='edu'>Education</option>
      <option value='nonmand'>Non-mandatory</option>
    </select>
    <select id='odMode'>
      <option value='all'>All modes</option>
      <option value='DA'>Drive alone</option>
      <option value='SR'>Shared ride</option>
      <option value='Walk'>Walk</option>
      <option value='Bike'>Bike</option>
      <option value='Transit'>Transit</option>
    </select>
  </div>
  <div class='ctrl-group'>
    <label><input type='checkbox' id='odToggle' checked> Show OD flows</label>
  </div>
  <div class='ctrl-group'>
    <label>Flow opacity <input type='range' id='odAlpha' min='10' max='90' value='45' style='width:70px'></label>
  </div>
  <span class='badge' id='flowCount'></span>
</div>

<div id='layout'>
  <div id='map-wrap'>
    <div id='map'></div>
    <div id='tooltip'></div>
    <div id='legend'>
      <h4 id='leg-title'>Scale</h4>
      <canvas id='leg-canvas' width='140' height='14' style='border-radius:3px;display:block;margin-bottom:4px'></canvas>
      <div style='display:flex;justify-content:space-between;font-size:0.68rem'>
        <span id='leg-min'></span><span id='leg-max'></span>
      </div>
    </div>
    <div class='od-legend' id='od-legend'>
      <div style='font-size:0.74rem;color:#7eb8f7;margin-bottom:5px;font-weight:600'>OD flow mode</div>
    </div>
  </div>
  <div id='sidebar'>
    <div id='zone-header'>
      <h2 id='zone-title'>Click a zone to explore</h2>
      <div class='meta' id='zone-meta'>Hover over the map to highlight a zone, click to pin details</div>
    </div>
    <div id='charts'>
      <div class='chart-panel'><h4>Mode share</h4><div id='modeChart' class='plotly-chart'></div></div>
      <div class='chart-panel'><h4>Trip purpose</h4><div id='purposeChart' class='plotly-chart'></div></div>
      <div class='chart-panel'><h4>Traveller type (segment)</h4><div id='ageChart' class='plotly-chart'></div></div>
      <div class='chart-panel'><h4>Mobility type (mode class)</h4><div id='incChart' class='plotly-chart'></div></div>
      <div class='chart-panel'><h4>Study level</h4><div id='licChart' class='plotly-chart'></div></div>
      <div class='chart-panel'><h4>Top origin zones</h4><div id='originChart' class='plotly-chart'></div></div>
      <div class='chart-panel'><h4>Zone accessibility profile</h4><div id='accChart' class='plotly-chart'></div></div>
    </div>
  </div>
</div>

<script>
const DATA = {payload_json};
const zd   = DATA.zone_detail;
const geo  = DATA.geojson;
const ods  = DATA.od_flows;

// ── Leaflet map ────────────────────────────────────────────────────────────
const map = L.map('map', {{zoomSnap:0.5}}).setView([-27.5, 153.0], 9);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; OpenStreetMap, &copy; CartoDB', subdomains:'abcd', maxZoom:19
}}).addTo(map);

// ── Colour scales ──────────────────────────────────────────────────────────
const METRIC_META = {{
  obs_tours:      {{label:'Observed HHTS flows',  diverging:false, unit:'tours'}},
  delta_share:    {{label:'Safety model shift (pred−obs)', diverging:true,  unit:'share Δ'}},
  pred_share_mean:{{label:'Predicted share',       diverging:false, unit:'share'}},
  obs_share_mean: {{label:'Observed share',        diverging:false, unit:'share'}},
  pct_walk_bike:  {{label:'Active mode (%)',        diverging:false, unit:'%'}},
  safety_penalty: {{label:'Safety penalty',        diverging:false, unit:'util'}},
  crash_total:    {{label:'Inbound crash exposure',diverging:false, unit:'crashes'}},
  acc_walk:       {{label:'Walk accessibility',    diverging:false, unit:'index'}},
  acc_transit:    {{label:'Transit accessibility', diverging:false, unit:'index'}},
  acc_auto_work:  {{label:'Auto/work accessibility',diverging:false,unit:'index'}},
}};

function lerp3(t, c0, c1, c2) {{
  if (t <= 0.5) {{
    const s = t*2;
    return [c0[0]+(c1[0]-c0[0])*s, c0[1]+(c1[1]-c0[1])*s, c0[2]+(c1[2]-c0[2])*s];
  }} else {{
    const s = (t-0.5)*2;
    return [c1[0]+(c2[0]-c1[0])*s, c1[1]+(c2[1]-c1[1])*s, c1[2]+(c2[2]-c1[2])*s];
  }}
}}
function toHex(rgb) {{
  return '#' + rgb.map(v=>Math.round(Math.max(0,Math.min(255,v))).toString(16).padStart(2,'0')).join('');
}}
function colourSeq(t) {{ return toHex(lerp3(t,[30,70,180],[0,160,130],[220,80,30]])); }}
function colourDiv(t) {{ return toHex(lerp3(t,[220,50,50],[240,240,240],[30,120,200]])); }}

let _vals, _min, _max, _diverging, odLayer=null, chorLayer=null, selectedOrder=null;

function buildScale(metric) {{
  _diverging = METRIC_META[metric]?.diverging;
  const props = geo.features.map(f=>f.properties[metric]).filter(v=>v!=null&&isFinite(v));
  if (!props.length) {{ _min=0; _max=1; return; }}
  _min=Math.min(...props); _max=Math.max(...props);
  drawLegend(metric);
}}

function getColour(val, metric) {{
  if (val==null||!isFinite(val)) return '#2d3250';
  const span=(_max-_min)||1;
  const t=Math.max(0,Math.min(1,(val-_min)/span));
  return _diverging ? colourDiv(t) : colourSeq(t);
}}

function drawLegend(metric) {{
  const meta=METRIC_META[metric]||{{}};
  document.getElementById('leg-title').textContent=meta.label||metric;
  document.getElementById('leg-min').textContent=_min.toFixed(_min<0.01?4:2);
  document.getElementById('leg-max').textContent=_max.toFixed(_max<0.01?4:2);
  const canvas=document.getElementById('leg-canvas');
  const ctx=canvas.getContext('2d');
  for(let i=0;i<140;i++) {{
    ctx.fillStyle=_diverging?colourDiv(i/139):colourSeq(i/139);
    ctx.fillRect(i,0,1,14);
  }}
}}

function renderChoropleth(metric) {{
  buildScale(metric);
  if (chorLayer) map.removeLayer(chorLayer);
  chorLayer = L.geoJSON(geo, {{
    style: f => {{
      const v=f.properties[metric];
      return {{fillColor:getColour(v,metric), fillOpacity:0.75, color:'#1a1e2e', weight:0.8}};
    }},
    onEachFeature: (f,lyr) => {{
      lyr.on('mouseover', () => showTooltip(f.properties));
      lyr.on('mouseout',  () => hideTooltip());
      lyr.on('click',     () => selectZone(f.properties.ORDER));
    }}
  }}).addTo(map);
}}

// ── OD flows ───────────────────────────────────────────────────────────────
const MODE_COLS = {{DA:'#e74c3c',SR:'#e67e22',Walk:'#27ae60',Bike:'#3498db',Transit:'#9b59b6'}};
// Build OD legend
const odLeg=document.getElementById('od-legend');
Object.entries(MODE_COLS).forEach(([m,c])=>{{
  odLeg.innerHTML+=`<div style='display:flex;align-items:center;gap:5px;margin:2px 0'>
    <span style='width:22px;height:3px;background:${{c}};border-radius:2px;display:inline-block'></span>
    <span>${{m}}</span></div>`;
}});

function renderOD() {{
  if (odLayer) {{ map.removeLayer(odLayer); odLayer=null; }}
  if (!document.getElementById('odToggle').checked) return;
  const segF=document.getElementById('odSeg').value;
  const modF=document.getElementById('odMode').value;
  const alpha=document.getElementById('odAlpha').value/100;
  const filtered=ods.filter(r=>(segF==='all'||r.segment===segF)&&(modF==='all'||r.mode===modF));
  document.getElementById('flowCount').textContent=`${{filtered.length}} OD pairs`;
  const lines=filtered.map(r=>L.polyline([[r.oLat,r.oLon],[r.dLat,r.dLon]],{{
    color:MODE_COLS[r.mode]||'#888', weight:r.w, opacity:alpha
  }}).bindTooltip(`${{r.segment}} | ${{r.mode}} | ${{r.flow}} flows`,{{sticky:true}}));
  odLayer=L.layerGroup(lines).addTo(map);
}}

// ── Tooltip ────────────────────────────────────────────────────────────────
const tt=document.getElementById('tooltip');
function showTooltip(p) {{
  const metric=document.getElementById('metricSel').value;
  const val=p[metric];
  const valStr=val!=null?Number(val).toFixed(val<0.01&&val>-0.01?5:2):'n/a';
  tt.innerHTML=`<strong>${{p.zone_name||'Zone '+p.ORDER}}</strong><br>
    ${{METRIC_META[metric]?.label||metric}}: <b>${{valStr}}</b><br>
    HHTS tours: ${{p.obs_tours||0}} | Dominant mode: ${{p.dominant_mode||'?'}}`;
  tt.style.display='block';
}}
function hideTooltip() {{ if (!selectedOrder) tt.style.display='none'; }}

// ── Zone selection & sidebar ───────────────────────────────────────────────
const PLOTLY_CFG={{responsive:true, displayModeBar:false}};
const DARK={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#9aa3c4',size:10}},
  margin:{{l:0,r:0,t:8,b:0}},
  xaxis:{{color:'#353a5a',gridcolor:'#2d3250',tickfont:{{size:9}}}},
  yaxis:{{color:'#353a5a',gridcolor:'#2d3250',tickfont:{{size:9}}}},
}};
const DARK_H={{...DARK,margin:{{l:80,r:10,t:8,b:20}}}};

function barH(el, labels, vals, colours, title) {{
  Plotly.newPlot(el, [{{x:vals,y:labels,type:'bar',orientation:'h',
    marker:{{color:colours}},textposition:'inside',text:vals.map(v=>v>0?v.toFixed(1)+'%':'')
  }}], {{...DARK_H, title:{{text:'',font:{{size:9}}}}}}, PLOTLY_CFG);
}}

function selectZone(order) {{
  selectedOrder=order;
  const d=zd[order];
  if (!d) {{ document.getElementById('zone-title').textContent='No detail for zone '+order; return; }}

  document.getElementById('zone-title').textContent=d.name||`Zone ${{order}}`;
  const distStr=d.mean_dist!=null?`avg dist: ${{d.mean_dist}} km`:'';
  const penStr=d.safety_penalty!=null?` | safety penalty: ${{Number(d.safety_penalty).toFixed(3)}}`:'';
  const deltaStr=d.delta_share!=null?` | pred−obs: ${{Number(d.delta_share)>0?'+':''}}${{Number(d.delta_share).toFixed(4)}}`:'';
  document.getElementById('zone-meta').textContent=
    `${{d.obs_tours}} HHTS tours | ${{distStr}}${{penStr}}${{deltaStr}}`;

  // Mode share
  const ms=d.mode_share||{{}};
  const mkeys=['DA','SR','Walk','Bike','Transit'];
  const mvals=mkeys.map(k=>ms[k]||0);
  const mcols=['#e74c3c','#e67e22','#27ae60','#3498db','#9b59b6'];
  barH('modeChart', mkeys, mvals, mcols);

  // Purpose share
  const ps=d.purpose_share||{{}};
  const pkeys=Object.keys(ps).sort((a,b)=>ps[b]-ps[a]).slice(0,8);
  barH('purposeChart', pkeys.map(k=>k.replace(/_/g,' ')), pkeys.map(k=>ps[k]||0),
    pkeys.map((_,i)=>['#7eb8f7','#5ec4a0','#f7c67e','#e88aaa','#b09ef7','#f78e5e','#78d4e8','#a8e8a0'][i%8]));

  // Traveller type (Worker / Student / Leisure)
  const tt=d.traveller_type||{{}};
  const ttkeys=['Worker','Student','Leisure'];
  barH('ageChart', ttkeys, ttkeys.map(k=>tt[k]||0),
    ['#7eb8f7','#5ec4a0','#f7c67e']);

  // Mobility type (Active / Transit / Private car)
  const mt=d.mobility_type||{{}};
  const mtkeys=['Active','Transit','Private car'];
  barH('incChart', mtkeys, mtkeys.map(k=>mt[k]||0),
    ['#27ae60','#9b59b6','#e74c3c']);

  // Study level
  const sl=d.study_level||{{}};
  const slkeys=['Non-student','Primary','Secondary','Tertiary','Postgrad'];
  barH('licChart', slkeys, slkeys.map(k=>sl[k]||0),
    ['#7eb8f7','#5ec4a0','#f7c67e','#e88aaa','#b09ef7']);

  // Top origin zones
  const tops=(d.top_origins||[]).slice(0,8);
  if (tops.length) {{
    Plotly.newPlot('originChart', [{{
      x:tops.map(r=>r.count), y:tops.map(r=>r.name||`Zone ${{r.OTAZ}}`),
      type:'bar', orientation:'h', marker:{{color:'#7eb8f7'}}
    }}], {{...DARK_H}}, PLOTLY_CFG);
  }}

  // Accessibility
  const accKeys=['acc_auto_work','acc_walk','acc_transit','acc_bike'];
  const accLabels=['Auto/work','Walk','Transit','Bike'];
  const accVals=accKeys.map(k=>d[k]);
  const hasAcc=accVals.some(v=>v!=null);
  if (hasAcc) {{
    Plotly.newPlot('accChart', [{{
      r:accVals.map(v=>v??0), theta:accLabels,
      fill:'toself', type:'scatterpolar',
      marker:{{color:'#7eb8f7'}}, line:{{color:'#7eb8f7'}}
    }}], {{
      ...DARK, margin:{{l:20,r:20,t:10,b:20}},
      polar:{{bgcolor:'rgba(0,0,0,0)',
        angularaxis:{{color:'#353a5a',tickfont:{{size:9}}}},
        radialaxis:{{color:'#353a5a',tickfont:{{size:8}},visible:true}}
      }}
    }}, PLOTLY_CFG);
  }}

  // Highlight on map
  showTooltip({{
    zone_name:d.name, ORDER:order,
    obs_tours:d.obs_tours, dominant_mode:Object.keys(d.mode_share||{{}})[0]||'?',
    [document.getElementById('metricSel').value]: d[document.getElementById('metricSel').value]
  }});
}}

// ── Wiring ────────────────────────────────────────────────────────────────
document.getElementById('metricSel').addEventListener('change', e=>renderChoropleth(e.target.value));
['odSeg','odMode','odToggle','odAlpha'].forEach(id=>
  document.getElementById(id).addEventListener('change', renderOD));

renderChoropleth('obs_tours');
renderOD();

// Click blank map to deselect
map.on('click', () => {{ selectedOrder=null; tt.style.display='none'; }});
</script>
</body>
</html>"""


def _get_model_zoo_searches() -> dict:
    try:
        from model_zoo import get_search_presets
        return get_search_presets()
    except Exception as exc:
        logger.warning(f"Unable to load model_zoo search presets: {exc}")
        return {}


def _print_model_zoo_searches() -> None:
    searches = _get_model_zoo_searches()
    if not searches:
        print("No model_zoo search presets available.")
        return
    print("Available model_zoo / HHTS search presets:")
    for key, spec in searches.items():
        print(f"  {key:18s} {spec.get('pipeline_mode', 'n/a'):7s}  {spec.get('description', '')}")


def run_hhts_optional_pipeline(
    search_name: str,
    force_mode: Optional[str] = None,
    n_iter: Optional[int] = None,
    temperature: Optional[float] = None,
    search_model: Optional[str] = None,
) -> None:
    searches = _get_model_zoo_searches()
    preset = searches.get(search_name)
    if preset is None:
        raise ValueError(
            f"Unknown HHTS/model_zoo search preset '{search_name}'. "
            f"Available presets: {', '.join(sorted(searches))}"
        )

    try:
        from main_2 import run_core as _run_core, run_search as _run_search
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the legacy HHTS stage0-4 runner from main_2.py"
        ) from exc

    mode = (force_mode or preset.get("pipeline_mode", "core")).lower().strip()
    sa_iter = int(n_iter if n_iter is not None else preset.get("n_iter", 400))
    sa_temp = float(temperature if temperature is not None else preset.get("temperature", 2.0))
    sa_model = (search_model or preset.get("search_model", "nested")).lower().strip()

    logger.info("=" * 70)
    logger.info("Optional HHTS/model_zoo pipeline requested via pipeline_logger")
    logger.info(f"  preset       : {search_name}")
    logger.info(f"  description  : {preset.get('description', '')}")
    logger.info(f"  mode         : {mode}")
    logger.info(f"  search_model : {sa_model}")
    logger.info(f"  n_iter       : {sa_iter}")
    logger.info(f"  temperature  : {sa_temp}")
    logger.info("=" * 70)

    old_env = {
        "MODE": os.environ.get("MODE"),
        "SA_N_ITER": os.environ.get("SA_N_ITER"),
        "SA_TEMP": os.environ.get("SA_TEMP"),
        "SA_MODEL": os.environ.get("SA_MODEL"),
    }
    try:
        os.environ["MODE"] = mode
        os.environ["SA_N_ITER"] = str(sa_iter)
        os.environ["SA_TEMP"] = str(sa_temp)
        os.environ["SA_MODEL"] = sa_model
        if mode == "search":
            _run_search(n_iter=sa_iter, temperature=sa_temp, search_model=sa_model)
        else:
            _run_core()
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    try:
        import numpy as _np
        if isinstance(value, (_np.integer, _np.floating)):
            return value.item()
        if isinstance(value, _np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


@timer("MCR structure search")
def run_mcr_search(data_arg: Optional[str] = None) -> None:
    """
    Run a standalone MetaCount structure search job.

    Family mapping:
      count    -> crash frequency count family
      duration -> survival/AFT-style positive-duration family
      linear   -> Gaussian continuous-outcome family
      cmf      -> Crash Modification Function family

    Random-parameter capability is enabled by exposing roles 2/3/4 in
    default_roles for the search evaluator.
    """
    try:
        from metacountregressor import (
            ExperimentBuilder,
            ModelConstraints,
            load_example16_3_model_data,
            load_example_duration_data,
            load_example_linear_data,
        )
    except Exception as exc:
        raise RuntimeError(
            "metacountregressor is required for mcr_search mode"
        ) from exc

    family = os.environ.get("MCR_FAMILY", "count").strip().lower()
    algo = os.environ.get("MCR_ALGO", "sa").strip().lower()
    max_iter = int(os.environ.get("MCR_MAX_ITER", "2000"))
    draws_r = int(os.environ.get("MCR_R", "200"))

    if family not in {"count", "duration", "linear", "cmf"}:
        raise ValueError(
            f"Unsupported MCR_FAMILY '{family}'. "
            "Use one of: count, duration, linear, cmf"
        )
    if algo not in {"sa", "de", "hs"}:
        raise ValueError(
            f"Unsupported MCR_ALGO '{algo}'. Use one of: sa, de, hs"
        )

    logger.info("=" * 70)
    logger.info("MCR structure search")
    logger.info(f"  family   : {family}")
    logger.info(f"  algo     : {algo}")
    logger.info(f"  max_iter : {max_iter}")
    logger.info(f"  R draws  : {draws_r}")
    logger.info("  random-parameter roles enabled: [2, 3, 4]")
    logger.info("=" * 70)

    base_df = None
    if data_arg and os.path.isfile(data_arg):
        base_df = pd.read_csv(data_arg)
        logger.info(f"Using external dataset from arg: {data_arg}")

    if family == "count":
        df = base_df.copy() if base_df is not None else load_example16_3_model_data()
        if "OFFSET" not in df.columns and {"AADT", "LENGTH"}.issubset(df.columns):
            df["OFFSET"] = np.log(np.clip(df["AADT"] * df["LENGTH"] * 365 / 1e8, 1e-12, None))
        y_col = "FREQ"
        id_col = "ID"
        offset_col = "OFFSET" if "OFFSET" in df.columns else None
        variables = [v for v in ["AADT", "LENGTH", "SPEED", "CURVES", "WIDTH", "SHOULDERW", "SLOPE", "SHOULDER"] if v in df.columns]
        family_kwargs = {}
        # SHOULDERW (continuous shoulder width) and SHOULDER (shoulder presence)
        # are two encodings of the same road-geometry feature - the CMF branch
        # below already treats them as alternatives (one goes to local_vars,
        # the other to baseline_vars). Use metacountregressor's native
        # mutual_exclusion so the count-family structure search never puts
        # both in a spec at once (redundant df, inflates BIC for no LL gain).
        mcr_constraints = None
        if {"SHOULDERW", "SHOULDER"}.issubset(set(variables)):
            mcr_constraints = ModelConstraints().mutual_exclusion(["SHOULDERW", "SHOULDER"])
    elif family == "duration":
        # Survival models in MetaCount are searched under model_family="duration".
        df = base_df.copy() if base_df is not None else load_example_duration_data()
        if "DURATION" not in df.columns:
            df["DURATION"] = np.exp(
                0.5
                + 0.01 * df.get("WIDTH", 0)
                + 0.02 * df.get("CURVES", 0)
                + 0.00002 * df.get("AADT", 0)
            )
        if "B" not in df.columns:
            df["B"] = np.clip(df.get("LENGTH", 1.0), 1e-6, None)
        y_col = "DURATION"
        id_col = "ID" if "ID" in df.columns else "obs_id"
        if id_col == "obs_id":
            df[id_col] = np.arange(len(df))
        offset_col = None
        variables = [v for v in ["AADT", "LENGTH", "SPEED", "CURVES", "WIDTH", "SHOULDERW", "SLOPE"] if v in df.columns]
        family_kwargs = {"budget_col": "B"}
        mcr_constraints = None
    elif family == "linear":
        df = base_df.copy() if base_df is not None else load_example_linear_data()
        if "LINEAR_TARGET" not in df.columns:
            df["LINEAR_TARGET"] = (
                0.1 * df.get("WIDTH", 0)
                + 0.2 * df.get("CURVES", 0)
                + 0.05 * df.get("SLOPE", 0)
                + 0.01 * df.get("AADT", 0) / 100.0
            )
        y_col = "LINEAR_TARGET"
        id_col = "ID" if "ID" in df.columns else "obs_id"
        if id_col == "obs_id":
            df[id_col] = np.arange(len(df))
        offset_col = None
        variables = [v for v in ["AADT", "LENGTH", "SPEED", "CURVES", "WIDTH", "SHOULDERW", "SLOPE"] if v in df.columns]
        family_kwargs = {}
        mcr_constraints = None
    else:  # cmf
        df = base_df.copy() if base_df is not None else load_example16_3_model_data()
        if "OFFSET" not in df.columns and {"AADT", "LENGTH"}.issubset(df.columns):
            df["OFFSET"] = np.log(np.clip(df["AADT"] * df["LENGTH"] * 365 / 1e8, 1e-12, None))
        y_col = "FREQ"
        id_col = "ID"
        offset_col = "OFFSET" if "OFFSET" in df.columns else None
        variables = [v for v in ["AADT", "LENGTH", "SPEED", "CURVES", "WIDTH", "SHOULDERW", "SLOPE"] if v in df.columns]
        baseline_vars = [v for v in ["LENGTH", "SPEED", "CURVES", "WIDTH", "SLOPE", "SHOULDER", "SHOULDERW"] if v in df.columns]
        local_vars = [v for v in ["SHOULDERW", "SHOULDER", "CURVES", "SLOPE"] if v in df.columns]
        if not local_vars and baseline_vars:
            local_vars = [baseline_vars[-1]]
            baseline_vars = baseline_vars[:-1]
        family_kwargs = {
            "aadt_col": "AADT",
            "baseline_vars": baseline_vars,
            "local_vars": local_vars,
        }
        # cmf routes through build_search() (model_family != "count"), which
        # does not forward `constraints`/`mutual_exclusion` in this package
        # version - SHOULDERW/SHOULDER exclusivity is instead handled above
        # by construction (each goes to at most one of baseline_vars/local_vars).
        mcr_constraints = None

    builder = ExperimentBuilder(
        df=df,
        id_col=id_col,
        y_col=y_col,
        offset_col=offset_col,
    )

    # Roles include random independent/correlated/grouped parameters.
    default_roles = [0, 1, 2, 3, 4, 5, 6]
    evaluator = builder.build_evaluator(
        variables=variables,
        model_family=family,
        default_roles=default_roles,
        max_latent_classes=1,
        R=draws_r,
        constraints=mcr_constraints,
        **family_kwargs,
    )

    # Compatibility across MetaCount API variants:
    # - count family returns a StructureEvaluatorLC (builder.run)
    # - duration/linear/cmf may return a *SearchProblem object with .run()
    if hasattr(evaluator, "vars"):
        result = builder.run(
            evaluator,
            algo=algo,
            max_iter=max_iter,
            n_jobs=1,
            seed=42,
        )
    elif hasattr(evaluator, "run"):
        result = evaluator.run(algo=algo, max_iter=max_iter, seed=42)
    else:
        raise TypeError(
            f"Unsupported evaluator object for family '{family}': {type(evaluator).__name__}"
        )

    if isinstance(result, dict):
        best_score = result.get("best_score", result.get("best_obj", result.get("best_objective")))
        best_spec = result.get("best_spec", result.get("best_solution", result.get("spec")))
    else:
        best_score = getattr(result, "best_score", None)
        best_spec = getattr(result, "best_spec", None)
    result_payload = {
        "family": family,
        "algo": algo,
        "max_iter": max_iter,
        "R": draws_r,
        "n_rows": int(len(df)),
        "y_col": y_col,
        "id_col": id_col,
        "offset_col": offset_col,
        "variables": variables,
        "default_roles": default_roles,
        "random_parameters_enabled": True,
        "best_score": _to_jsonable(best_score),
    }

    result_json_path = os.path.join(OUT_DIR, "mcr_search_result.json")
    best_spec_path = os.path.join(OUT_DIR, "best_spec.json")
    summary_html_path = os.path.join(OUT_DIR, "mcr_search_summary.html")

    with open(result_json_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(result_payload), fh, indent=2)

    with open(best_spec_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(best_spec), fh, indent=2)

    variable_descriptions = {
        "ID": "Segment identifier",
        "FREQ": "Observed crash frequency on the segment",
        "AADT": "Annual Average Daily Traffic",
        "LENGTH": "Segment length",
        "SPEED": "Posted or operating speed",
        "WIDTH": "Roadway width",
        "CURVES": "Horizontal curvature measure",
        "SLOPE": "Vertical grade/slope",
        "SHOULDER": "Shoulder indicator/measure",
        "SHOULDERW": "Shoulder width",
        "OFFSET": "Log exposure term log(AADT * LENGTH * 365 / 1e8)",
        "DURATION": "Positive duration response used for survival-duration examples",
        "B": "Budget/reference positive duration covariate for duration-family search",
        "LINEAR_TARGET": "Continuous target for Gaussian/linear-family search",
    }

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    describe_html = df[numeric_cols].describe().to_html(classes="table table-sm", border=0)
    vars_html = "".join(
        f"<tr><td>{col}</td><td>{variable_descriptions.get(col, 'No description provided')}</td></tr>"
        for col in df.columns
    )
    html = (
        "<html><head><meta charset='utf-8'><title>MCR Search Summary</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;width:100%;margin:12px 0;}"
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left;}"
        "th{background:#f6f8fa;} code{background:#f2f4f7;padding:1px 4px;border-radius:3px;}"
        "</style></head><body>"
        f"<h1>MCR Search Summary ({family})</h1>"
        "<p><strong>Survival mapping:</strong> survival/AFT search is run under "
        "<code>model_family='duration'</code>.</p>"
        "<p><strong>Random parameters:</strong> enabled via roles 2/3/4 in "
        "<code>default_roles=[0,1,2,3,4,5,6]</code>.</p>"
        f"<p><strong>Algorithm:</strong> {algo} &nbsp; <strong>Max iterations:</strong> {max_iter} "
        f"&nbsp; <strong>R:</strong> {draws_r}</p>"
        f"<p><strong>Best score:</strong> {best_score}</p>"
        "<h2>Example 16-3 Data Properties</h2>"
        f"{describe_html}"
        "<h2>Variable Descriptions</h2>"
        "<table><thead><tr><th>Variable</th><th>Description</th></tr></thead><tbody>"
        f"{vars_html}"
        "</tbody></table>"
        "</body></html>"
    )
    with open(summary_html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    logger.info(f"MCR search result JSON  -> {result_json_path}")
    logger.info(f"MCR best spec JSON      -> {best_spec_path}")
    logger.info(f"MCR summary HTML        -> {summary_html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEQ pipeline runner with optional HHTS/model_zoo presets.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="main",
        choices=[
            "main",
            "debug",
            "ga",
            "ga_staged",
            "ga_stage5_smoke",
            "mcr_search",
            "safety_baseline",
            "safety_nosafety",
            "safety_compare",
            "hhts_core",
            "hhts_search",
            "list_searches",
        ],
    )
    parser.add_argument(
        "arg",
        nargs="?",
        default=None,
        help="Zone selector / file path for main modes.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Optional HHTS/model_zoo preset name. Use list_searches to inspect choices.",
    )
    parser.add_argument("--sa-iter", type=int, default=None, help="Override SA iterations for HHTS search modes.")
    parser.add_argument("--sa-temp", type=float, default=None, help="Override SA temperature for HHTS search modes.")
    parser.add_argument(
        "--sa-model",
        choices=["nested", "mnl"],
        default=None,
        help="Override the Stage-4 search model for HHTS search modes.",
    )
    args = parser.parse_args()

    mode = 'main' if args.mode == 'debug' else args.mode
    arg = args.arg

    _setup_logging(mode, arg or args.search)

    if mode == 'main':
        main(arg)
    elif mode == 'ga':
        GA_main(arg)
    elif mode == 'ga_staged':
        GA_STAGED(arg)
    elif mode == 'ga_stage5_smoke':
        GA_STAGE5_SMOKE(arg)
    elif mode == 'mcr_search':
        run_mcr_search(arg)
    elif mode == 'hhts_core':
        run_hhts_optional_pipeline(
            search_name=args.search or 'core_fixed',
            force_mode='core',
            n_iter=args.sa_iter,
            temperature=args.sa_temp,
            search_model=args.sa_model,
        )
    elif mode == 'hhts_search':
        run_hhts_optional_pipeline(
            search_name=args.search or 'nested_standard',
            force_mode='search',
            n_iter=args.sa_iter,
            temperature=args.sa_temp,
            search_model=args.sa_model,
        )
    elif mode == 'list_searches':
        _print_model_zoo_searches()
    elif mode in ('safety_baseline', 'safety_nosafety'):
        # Safety-skim experiment modes.
        # safety_baseline  → run normally (safety skims INCLUDED in stage 5)
        # safety_nosafety  → exclude CRASH + CRASHPT from stage 5 estimation
        os.environ['SAFETY_EXPERIMENT_MODE'] = mode.replace('safety_', '')
        logger.info(f"Safety experiment mode: {os.environ['SAFETY_EXPERIMENT_MODE']}")
        main(arg)
    elif mode == 'safety_compare':
        # Compare safety_baseline vs safety_nosafety runs and write a report.
        # Run AFTER both safety_baseline and safety_nosafety have completed.
        compare_safety_runs(
            baseline_dir = os.path.join('runs', 'safety_baseline'),
            nosafety_dir = os.path.join('runs', 'safety_nosafety'),
            out_dir      = os.path.join('runs', 'safety_analysis'),
        )
    else:
        parser.print_help()


