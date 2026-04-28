# pipeline_logger.py
import time
import logging
import functools
import os
import glob
import gc
import sys
import pickle
import numpy as np

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
        'stage4_models.pkl', 'stage5_models.pkl', 'ga_results.pkl',
        '_hhts_persons.pkl', '_hhts_tours.pkl',
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
SYNTH_DIR = 'saved_data/zone_ids_sa2'

# Single-zone fallback (used by old CLI invocations and unit tests)
DEFAULT_SYNTH = f'{SYNTH_DIR}/order_40.csv'


# ---------------------------------------------------------------------------
# HELPERS — decorated wrappers so every stage is individually timed
# ---------------------------------------------------------------------------

@timer("process_survey + landuse + accessibility")
def load_data():
    from compute_accessibility import run_accessibility
    sc      = Synth_Connect()
    out     = sc.process_survey(save=False)
    landuse = sc.get_land()
    landuse = run_accessibility(
        landuse,
        skims='downcasted_matrices_base.omx',
        save_path='saved_data/accessibility.csv',
    )
    return sc, out, landuse


@timer("load synthetic population — single zone")
def load_synth(filepath: str = DEFAULT_SYNTH,
               landuse: pd.DataFrame = None):
    """Load one zone's synth CSV.  OTAZ is derived from the filename."""
    import re
    synth = pd.read_csv(filepath)

    match = re.search(r'order_(\d+)', filepath)
    if match:
        synth['OTAZ'] = int(match.group(1))
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

    Each file contributes one zone's worth of synthetic persons.  OTAZ is
    set from the filename (order_N.csv → OTAZ = N).

    Parameters
    ----------
    synth_dir : folder containing order_N.csv files
    landuse   : passed to enrich_synth_population for accessibility join
    zones     : optional list of zone ORDER numbers to restrict loading
                (e.g. [40, 124, 200]).  None = load every file found.

    Returns
    -------
    pd.DataFrame  — full synthetic population across all loaded zones,
                    with OTAZ column set correctly per person.
    """
    import re

    pattern = os.path.join(synth_dir, 'order_*.csv')
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No order_*.csv files found in '{synth_dir}'.\n"
            f"  Expected files like: order_1.csv, order_2.csv, ..., order_329.csv\n"
            f"  Check that SYNTH_DIR is set correctly at the top of pipeline_logger.py."
        )

    # Filter to specific zones if requested
    if zones is not None:
        zone_set = set(zones)
        files = [
            f for f in files
            if (m := re.search(r'order_(\d+)', f)) and int(m.group(1)) in zone_set
        ]
        logger.info(f"  Zone filter: loading {len(files)} zones from {len(zone_set)} requested")

    logger.info(f"  Found {len(files)} zone CSV files in {synth_dir}")

    frames = []
    missing = []
    for filepath in files:
        match = re.search(r'order_(\d+)', filepath)
        if not match:
            logger.warning(f"  Skipping {filepath} — cannot parse ORDER from filename")
            continue
        otaz = int(match.group(1))
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
def cdap(out, synth, landuse=None):
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
    synth = run_cdap(out["persons"], synth)
    synth.to_csv(os.path.join(OUT_DIR, 'cdap_output.csv'), index=False)
    return synth


@timer("Usual Locations — work & school zone")
def usual_locations(synth, landuse, hhts_tours=None):
    return run_usual_locations(
        synth, landuse,
        skims='downcasted_matrices_base.omx',
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


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATION WRAPPERS  (Phase 1 — HHTS only)
# Each returns the fitted models dict and saves PKL + HTML to OUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

@timer("Estimate Stage 1 — activity frequency")
def estimate_stage1(out):
    from stage1_activity_frequency import estimate_frequency_models, save_models, prepare_frequency_data
    import pandas as pd
    logger.info("  Preparing HHTS frequency data...")
    models = estimate_frequency_models(out["tours"], out["persons"])
    path = os.path.join(OUT_DIR, 'stage1_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage1', OUT_DIR)
    logger.info(f"  Stage 1 models → {path}  (HTML report alongside)")
    _log_stage_summary('1', models)
    return models


@timer("Estimate Stage 2 — day duration")
def estimate_stage2(out):
    from stage2_day_duration import estimate_day_models, save_models, aggregate_person_day
    logger.info("  Preparing HHTS person-day data...")
    person_day = aggregate_person_day(out["tours"], out["persons"])
    logger.info(f"  person_day rows: {len(person_day):,}  "
                f"cols: {list(person_day.columns[:8])}")
    models = estimate_day_models(person_day)
    path = os.path.join(OUT_DIR, 'stage2_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage2', OUT_DIR)
    logger.info(f"  Stage 2 models → {path}  (HTML report alongside)")
    _log_stage_summary('2', models)
    return models


@timer("Estimate Stage 3 — activity duration")
def estimate_stage3(out):
    from stage3_activity_duration import estimate_duration_models, save_models
    logger.info("  Preparing HHTS duration data...")
    models = estimate_duration_models(out["tours"], out["persons"], trips=out["trips"])
    path = os.path.join(OUT_DIR, 'stage3_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage3', OUT_DIR)
    logger.info(f"  Stage 3 models → {path}  (HTML report alongside)")
    _log_stage_summary('3', models)
    return models


@timer("Estimate Stage 4 — schedule placement")
def estimate_stage4(out):
    from stage4_schedule_placement import estimate_timing_models, save_models
    logger.info("  Preparing HHTS timing data...")
    models = estimate_timing_models(out["trips"], out["tours"], out["persons"])
    path = os.path.join(OUT_DIR, 'stage4_models.pkl')
    save_models(models, path)
    _save_stage_coef_csv(models, 'stage4', OUT_DIR)
    logger.info(f"  Stage 4 models → {path}  (HTML report alongside)")
    _log_stage_summary('4', models)
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
    skims_path = 'downcasted_matrices_base.omx'
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

    return models, skims, sk_lookup


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION WRAPPERS  (Phase 2 — synth population, one zone at a time)
# Each loads previously estimated models and applies them to the synth.
# ─────────────────────────────────────────────────────────────────────────────

@timer("Predict Stage 1 — activity frequency")
def predict_stage1(synth, models=None):
    from stage1_activity_frequency import predict_activity_frequency, load_models
    if models is None:
        models = load_models(os.path.join(OUT_DIR, 'stage1_models.pkl'))
    logger.info(f"  Predicting activity frequency for {len(synth):,} persons...")
    result = predict_activity_frequency(synth, models)
    result.to_csv(os.path.join(OUT_DIR, 'stage1_output.csv'), index=False)
    logger.info(f"  ✓  Stage 1 output: {len(result):,} rows → {OUT_DIR}/stage1_output.csv")
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
    return result


@timer("Predict Stage 3 — activity duration")
def predict_stage3(synth_s2, models=None):
    from stage3_activity_duration import predict_activity_durations, load_models
    if models is None:
        models = load_models(os.path.join(OUT_DIR, 'stage3_models.pkl'))
    logger.info(f"  Predicting activity durations for {len(synth_s2):,} persons...")
    result = predict_activity_durations(synth_s2, models)
    result.to_csv(os.path.join(OUT_DIR, 'stage3_output.csv'), index=False)
    logger.info(f"  ✓  Stage 3 output: {len(result):,} rows "
                f"({result['person_id'].nunique():,} persons) → {OUT_DIR}/stage3_output.csv")
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

    skims_path = 'downcasted_matrices_base.omx'
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
    return synth_s1, models


@timer("Stage 2 — day duration")
def stage2(out, synth_s1):
    synth_s2, models = run_stage2(out["tours"], out["persons"], synth_s1,
                                    save_path=os.path.join(OUT_DIR, 'stage2_models.pkl'))
    _save_stage_coef_csv(models, 'stage2', OUT_DIR)
    synth_s2.to_csv(os.path.join(OUT_DIR, 'stage2_output.csv'), index=False)
    return synth_s2, models


@timer("Stage 3 — activity duration")
def stage3(out, synth_s2):
    activities, models = run_stage3(
        out["tours"], out["persons"], synth_s2, trips=out["trips"],
        save_path=os.path.join(OUT_DIR, 'stage3_models.pkl'),
    )
    _save_stage_coef_csv(models, 'stage3', OUT_DIR)
    activities.to_csv(os.path.join(OUT_DIR, 'stage3_output.csv'), index=False)
    return activities, models


@timer("Stage 4 — schedule placement")
def stage4(out, activities):
    scheduled, models = run_stage4(
        out["trips"], out["tours"], out["persons"], activities,
        save_path=os.path.join(OUT_DIR, 'stage4_models.pkl'),
    )
    _save_stage_coef_csv(models, 'stage4', OUT_DIR)
    scheduled.to_csv(os.path.join(OUT_DIR, 'stage4_output.csv'), index=False)
    return scheduled, models


@timer("Stage 5 — destination choice")
def stage5(out, synth_persons, scheduled, landuse):
    import openmatrix as omx
    from stage5_destination_choice import _skim_flat, _skim_lookup
    skims_path = 'downcasted_matrices_base.omx'
    # Allow Stage 5 to complete with diagnostics/artifacts even when one
    # segment has unstable estimation. This preserves HTML/KPI traces.
    os.environ.setdefault('STAGE5_STRICT_ESTIMATION', '0')
    # Keep destination-estimation requirement enabled by default.
    os.environ.setdefault('STAGE5_REQUIRE_DEST_ESTIMATION', '1')

    crash_candidates = [
        'OD_Crashes.csv',
        'SA2_OD_Crash_Matrix_PT.xlsx',
        'SA2_OD_Crash_Matrix.xlsx',
    ]
    crash_skims = next((p for p in crash_candidates if os.path.isfile(p)), None)
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
    skims      = omx.open_file(skims_path, 'r')
    skim_df    = _skim_flat(skims)
    sk_lookup  = _skim_lookup(skim_df)
    return scheduled_dest, models, skims, sk_lookup


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

    # Sample person-movement plots (top travellers by distance)
    try:
        if 'person_id' in plan_coords.columns:
            rt_col = 'row_type' if 'row_type' in plan_coords.columns else None
            trip_rows = plan_coords[plan_coords[rt_col] == 'trip'] if rt_col else plan_coords
            if 'dist_km' in trip_rows.columns and len(trip_rows) > 0:
                top_pids = (trip_rows.groupby('person_id')['dist_km']
                            .sum().sort_values(ascending=False)
                            .head(20).index.tolist())
                import numpy as np
                rng = np.random.default_rng(42)
                sample_pids = rng.choice(top_pids, size=min(4, len(top_pids)),
                                         replace=False).tolist()
                for pid in sample_pids:
                    try:
                        _sp = os.path.join(_pdir, f'person_{pid}_movement.png')
                        plot_person_movement(plan_coords, pid, save_path=_sp)
                        _saved.append(f'person_{pid}_movement.png')
                        logger.info(f"  Movement plot → {_sp}")
                    except Exception as _mpe:
                        logger.warning(f"  Movement plot for {pid} failed: {_mpe}")
    except Exception as _e:
        logger.warning(f"  Person movement plots failed: {_e}")

    logger.info(f"  Spatial plots complete: {len(_saved)} files in {_pdir}/")
    sys.stdout.flush()
    return plan_coords


# ---------------------------------------------------------------------------
# GA WRAPPERS  (unchanged)
# ---------------------------------------------------------------------------

@timer("GA — all stages")
def ga_all(out, landuse):
    # pop_size / n_generations must match the staged wrappers below.
    # The defaults in run_ga_all_stages (pop_size=40, n_generations=60,
    # n_restarts=2) are far too large for stages 5/6 which call Larch NL
    # models — each individual writes a fresh flow code, so thousands of
    # evaluations are needed and the GA never finishes within walltime.
    run_ga_all_stages(
        tours=out["tours"],
        trips=out["trips"],
        persons=out["persons"],
        landuse=landuse,
        skims='downcasted_matrices_base.omx',
        pop_size=30,
        n_generations=40,
        n_restarts=1,
        save_path=os.path.join(OUT_DIR, 'ga_results.pkl'),
    )


@timer("GA — Stage 1 (activity frequency)")
def ga_stage1(out):
    from census_bridge import run_ga_stage1
    res = run_ga_stage1(out["tours"], out["persons"], pop_size=30, n_generations=40)
    res.plot_convergence(save_path=os.path.join(OUT_DIR, 'plots', 'ga_stage1.png'))
    return res


@timer("GA — Stage 2 (day duration)")
def ga_stage2(out):
    from census_bridge import run_ga_stage2
    res = run_ga_stage2(out["tours"], out["persons"], pop_size=30, n_generations=40)
    res.plot_convergence(save_path=os.path.join(OUT_DIR, 'plots', 'ga_stage2.png'))
    return res


@timer("GA — Stage 3 (activity duration)")
def ga_stage3(out):
    from census_bridge import run_ga_stage3
    res = run_ga_stage3(out["tours"], out["persons"], trips=out["trips"], pop_size=30, n_generations=40)
    res.plot_convergence(save_path=os.path.join(OUT_DIR, 'plots', 'ga_stage3.png'))
    return res


@timer("GA — Stage 4 (schedule placement)")
def ga_stage4(out):
    from census_bridge import run_ga_stage4
    res = run_ga_stage4(out["trips"], out["tours"], out["persons"],
                        pop_size=30, n_generations=40)
    res.plot_convergence(save_path=os.path.join(OUT_DIR, 'plots', 'ga_stage4.png'))
    return res


@timer("GA — Stage 5 (skim + landuse)")
def ga_stage5(out, landuse):
    from census_bridge import run_ga_stage5
    res = run_ga_stage5(
        trips=out["trips"], persons=out["persons"],
        landuse=landuse,
        skims='downcasted_matrices_base.omx',
        pop_size=30, n_generations=40,
    )
    res.plot_convergence(save_path=os.path.join(OUT_DIR, 'plots', 'ga_stage5.png'))
    print(f"  mode:    {res.best_features.get('mode')}")
    print(f"  dest:    {res.best_features.get('dest')}")
    return res


@timer("GA — Stage 6 (person + accessibility)")
def ga_stage6(out, landuse):
    from census_bridge import run_ga_extended
    res = run_ga_extended(
        stage=6, trips=out["trips"], persons=out["persons"],
        landuse=landuse, purpose='work', pop_size=30, n_generations=40,
    )
    res.plot_convergence(save_path=os.path.join(OUT_DIR, 'plots', 'ga_stage6.png'))
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
    s1_models = estimate_stage1(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 1]")

    logger.info("▶  Estimating Stage 2 models (day duration)...")
    s2_models = estimate_stage2(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 2]")

    logger.info("▶  Estimating Stage 3 models (activity duration)...")
    s3_models = estimate_stage3(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 3]")

    logger.info("▶  Estimating Stage 4 models (schedule placement)...")
    s4_models = estimate_stage4(out)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 4]")

    logger.info("▶  Estimating Stage 5 models (destination + mode choice)...")
    s5_models_est, skims, sk = estimate_stage5(out, landuse)
    logger.info(f"   [mem: {_mem_mb():.0f} MB after Stage 5]")

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
    logger.info(f"  HHTS persons saved to {_hhts_persons_path} for Stage 5")
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
    synth      = cdap(_out_for_cdap, synth)
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
    synth      = usual_locations(synth, landuse, hhts_tours=_hhts_tours_for_val)
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
    synth_s1 = predict_stage1(travellers, _s1); del _s1; gc.collect()

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
    _s4 = __import__('stage4_schedule_placement',
                     fromlist=['load_models']).load_models(
                         os.path.join(OUT_DIR, 'stage4_models.pkl'))
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

    # ── Stage 6 (chunked by persons) ─────────────────────────────────────
    logger.info("▶  Phase 2 — Stage 6 validation + export (chunked)")
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

    synth = cdap(out, synth)
    synth = usual_locations(synth, landuse, hhts_tours=out.get('tours'))
    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} persons are travellers")

    if ga_results is not None:
        # ── Apply GA-selected features through stages 1–5 ─────────────
        from ga_hypothesis_testing import apply_ga_results_to_pipeline
        scheduled_dest, s5_models = apply_ga_results_to_pipeline(
            ga_results   = ga_results,
            out          = out,
            synth        = travellers,
            landuse      = landuse,
            skims_path   = 'downcasted_matrices_base.omx',
            save_dir     = OUT_DIR,
        )
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
    # Capture results so we can pass feature_override to the pipeline stages.
    from ga_hypothesis_testing import AllStagesResult
    import pickle

    all_ga = AllStagesResult()
    all_ga.results[1] = ga_stage1(out)
    all_ga.results[2] = ga_stage2(out)
    all_ga.results[3] = ga_stage3(out)
    all_ga.results[4] = ga_stage4(out)
    all_ga.results[5] = ga_stage5(out, landuse)

    # Save combined results for reference
    ga_path = os.path.join(OUT_DIR, 'ga_results.pkl')
    try:
        with open(ga_path, 'wb') as _f:
            pickle.dump(all_ga, _f)
        logger.info(f"GA staged results saved → {ga_path}")
    except Exception as _e:
        logger.warning(f"Could not save GA staged results: {_e}")

    # ── Load synth only after all GA stages complete ───────────────────
    synth = (load_synth(filepath, landuse=landuse) if filepath and os.path.isfile(filepath)
             else load_synth_all_zones(SYNTH_DIR, landuse=landuse))

    synth = cdap(out, synth)
    synth = usual_locations(synth, landuse, hhts_tours=out.get('tours'))
    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} persons are travellers")

    # ── Apply GA-selected features through stages 1–5 ─────────────────
    from ga_hypothesis_testing import apply_ga_results_to_pipeline
    scheduled_dest, s5_models = apply_ga_results_to_pipeline(
        ga_results = all_ga,
        out        = out,
        synth      = travellers,
        landuse    = landuse,
        skims_path = 'downcasted_matrices_base.omx',
        save_dir   = OUT_DIR,
    )

    import openmatrix as omx
    from stage5_destination_choice import _skim_flat, _skim_lookup
    skims   = omx.open_file('downcasted_matrices_base.omx', 'r')
    sk      = _skim_lookup(_skim_flat(skims))
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
if __name__ == "__main__":
    import sys
    
    # ── Usage ──────────────────────────────────────────────────────────────
    # python pipeline_logger.py                      → full population, all zones
    # python pipeline_logger.py main                 → full population, all zones
    # python pipeline_logger.py main 40              → zone 40 only
    # python pipeline_logger.py main 40,124,200      → zones 40, 124, 200
    # python pipeline_logger.py main order_40.csv    → single file (legacy)
    # python pipeline_logger.py ga                   → GA + full population
    # python pipeline_logger.py ga_staged            → staged GA + full population
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'main'
    # 'debug' is an alias for 'main' with verbose logging already enabled
    # via _setup_logging; no separate branch is needed.
    #TODO allow the debug mode to only sample one zone 40
    
    #TODO allow the arguments to be selectivabel with defaults (e.g. --mode=ga --zones=40,124,200)
    
    
    if mode == 'debug':
        mode = 'main'
    arg  = sys.argv[2] if len(sys.argv) > 2 else None

    _setup_logging(mode, arg)

    if mode == 'main':
        main(arg)
    elif mode == 'ga':
        GA_main(arg)
    elif mode == 'ga_staged':
        GA_STAGED(arg)
    elif mode in ('safety_baseline', 'safety_nosafety'):
        # Safety-skim experiment modes.
        # safety_baseline  → run normally (safety skims INCLUDED in stage 5)
        # safety_nosafety  → exclude CRASH + CRASHPT from stage 5 estimation
        os.environ['SAFETY_EXPERIMENT_MODE'] = mode.replace('safety_', '')
        logger.info(f"Safety experiment mode: {os.environ['SAFETY_EXPERIMENT_MODE']}")
        main(arg)
    else:
        print("Usage: python pipeline_logger.py [main|ga|ga_staged|safety_baseline|safety_nosafety] [zone]")
        print("  zone_or_file can be:")
        print("    (empty)          — load all zones from SYNTH_DIR")
        print("    40               — single zone number")
        print("    40,124,200       — comma-separated zone numbers")
        print("    order_40.csv     — specific CSV filename")
        print("    path/to/file.csv — full path to CSV")


