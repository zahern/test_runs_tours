"""
equilibrium_loop.py
===================
Outer equilibrium loop: iterates the prediction chain (Stage 1 → 6) until
activity-generation patterns converge or a maximum iteration count is reached.

Wired into pipeline_logger.main() behind EQUILIBRIUM_LOOP=1.  The loop
re-runs the same prediction functions that the flat Phase 2 chain uses,
each iteration getting a fresh random seed so stochastic draws can settle
toward a stable distribution.  Convergence is measured on Stage-1 per-person
activity counts (n_{purpose}_out columns).

Env vars
--------
EQUILIBRIUM_LOOP     : '1' to enable (default '0')
EQUILIBRIUM_N_ITER   : max iterations (default 3)
EQUILIBRIUM_TOL      : relative change tolerance (default 0.01)
"""

import os, sys, gc, logging
import numpy as np
import pandas as pd
import pickle as _pkl

logger = logging.getLogger(__name__)

ENV_LOOP   = 'EQUILIBRIUM_LOOP'
ENV_N_ITER = 'EQUILIBRIUM_N_ITER'
ENV_TOL    = 'EQUILIBRIUM_TOL'

# ── convergence helper ──────────────────────────────────────────────────────

def _stage1_convergence(prev_s1: pd.DataFrame, curr_s1: pd.DataFrame,
                        tol: float = 0.01) -> tuple:
    """
    Compare Stage-1 activity frequency per-purpose means between iterations.

    Returns (converged: bool, max_relative_change: float).
    """
    _out_cols = [c for c in prev_s1.columns
                 if c.startswith('n_') and c.endswith('_out')]
    if not _out_cols:
        return True, 0.0
    max_rel = 0.0
    for col in _out_cols:
        if col not in curr_s1.columns:
            continue
        prev_mean = prev_s1[col].mean()
        curr_mean = curr_s1[col].mean()
        denom = max(abs(prev_mean), 1.0)
        max_rel = max(max_rel, abs(curr_mean - prev_mean) / denom)
    return max_rel < tol, max_rel


# ── main loop ───────────────────────────────────────────────────────────────

def run_equilibrium_loop(
    *,
    travellers: pd.DataFrame,
    landuse: pd.DataFrame,
    skims,
    sk: dict,
    OUT_DIR: str,
    _hhts_tours_path: str = None,
    _out_persons_ref: str = None,
    _hhts_tours_stage5_val = None,
    _synth_home: pd.DataFrame = None,
):
    """
    Run Phase 2 prediction chain iteratively until activity counts converge.

    Re-imports pipeline_logger module at runtime (no circular-dependency risk
    because this is called from inside main(), after all pipeline_logger
    module-level globals are set).

    Returns (plan, scheduled_dest, converged_flag).
    """
    import pipeline_logger as _pl

    n_iter = int(os.environ.get(ENV_N_ITER, '3'))
    tol    = float(os.environ.get(ENV_TOL, '0.01'))

    # ── Load all stage models once ──────────────────────────────────────
    _s1_load = __import__('stage1_activity_frequency',
                          fromlist=['load_models']).load_models
    _s2_load = __import__('stage2_day_duration',
                          fromlist=['load_models']).load_models
    _s3_load = __import__('stage3_activity_duration',
                          fromlist=['load_models']).load_models
    _s4_load = __import__('stage4_schedule_placement',
                          fromlist=['load_models']).load_models
    _s5_load = __import__('stage5_destination_choice',
                          fromlist=['load_models']).load_models

    _s1_models = _s1_load(os.path.join(OUT_DIR, 'stage1_models.pkl'))
    _s2_models = _s2_load(os.path.join(OUT_DIR, 'stage2_models.pkl'))
    _s3_models = _s3_load(os.path.join(OUT_DIR, 'stage3_models.pkl'))
    _s4_path   = os.path.join(OUT_DIR, 'stage4_models.pkl')
    if not os.path.exists(_s4_path):
        raise FileNotFoundError(f"stage4_models.pkl not found at '{_s4_path}'")
    _s4_models = _s4_load(_s4_path)
    _s5_models = _s5_load(os.path.join(OUT_DIR, 'stage5_models.pkl'))

    # HHTS calibration data (load once)
    _hhts_persons_s1 = None
    _hhts_tours_s1   = None
    try:
        if _out_persons_ref and os.path.exists(_out_persons_ref):
            _hhts_persons_s1 = _pkl.load(open(_out_persons_ref, 'rb'))
    except Exception as _e:
        logger.warning(f"  Could not load HHTS persons for calib: {_e}")
    if _hhts_tours_path and os.path.exists(_hhts_tours_path):
        try:
            _hhts_tours_s1 = _pkl.load(open(_hhts_tours_path, 'rb'))
        except Exception as _e:
            logger.warning(f"  Could not load HHTS tours for calib: {_e}")

    prev_s1     = None
    converged   = False
    final_plan  = None
    final_sched = None

    for eq_i in range(n_iter):
        logger.info(f"═══ Equilibrium iteration {eq_i+1}/{n_iter} ═══")

        # ── Stage 1 ────────────────────────────────────────────────────
        synth_s1 = _pl.predict_stage1(
            travellers, _s1_models,
            hhts_persons=_hhts_persons_s1,
            hhts_tours=_hhts_tours_s1,
        )

        if prev_s1 is not None:
            conv, diff = _stage1_convergence(prev_s1, synth_s1, tol)
            logger.info(
                f"  Stage 1 relative change = {diff:.6f} "
                f"({'converged' if conv else 'not converged'})")
            if conv:
                converged = True
                # Use the most recent synth_s1 for downstream stages
                break

        prev_s1 = synth_s1.copy()

        # ── Stage 2 ────────────────────────────────────────────────────
        synth_s2 = _pl.predict_stage2(synth_s1, _s2_models)

        # ── Stage 3 ────────────────────────────────────────────────────
        activities = _pl.predict_stage3(synth_s2, _s3_models)

        # ── Stage 4 ────────────────────────────────────────────────────
        scheduled = _pl.predict_stage4(activities, _s4_models)

        # ── Stage 5 ────────────────────────────────────────────────────
        _out_s5 = {
            'persons': travellers,
            'trips': pd.DataFrame(),
            'tours': (_hhts_tours_stage5_val
                      if _hhts_tours_stage5_val is not None
                      else pd.DataFrame()),
        }
        try:
            scheduled_dest = _pl.predict_stage5(
                scheduled, _s5_models, _out_s5, landuse, skims, sk,
                persons=travellers)
        except Exception as _s5e:
            logger.error(f"  Stage 5 prediction failed at eq iter {eq_i+1}: {_s5e}")
            raise

        # ── Stage 5b ───────────────────────────────────────────────────
        try:
            _out_s5b = {'persons': travellers}
            scheduled_dest = _pl.stage5b(_out_s5b, scheduled_dest, landuse)
        except Exception as _s5be:
            logger.warning(f"  Stage 5b failed (non-fatal) at eq iter {eq_i+1}: {_s5be}")

        gc.collect()

        final_sched = scheduled_dest

        # Convergence not detected — continue to next iteration

    # ── Stage 6 (run once after the loop) ──────────────────────────────
    if final_sched is None:
        logger.error("  No scheduled_dest produced by equilibrium loop")
        return None, None, False

    if _out_persons_ref and os.path.exists(_out_persons_ref):
        _out_s6 = {'persons': _pkl.load(open(_out_persons_ref, 'rb'))}
    else:
        _out_s6 = {'persons': travellers}

    try:
        final_plan, _ = _pl.stage6(
            _out_s6, final_sched, landuse,
            stage5_models=None,
            skims=skims, skim_lookup=sk,
        )
    except Exception as _s6e:
        logger.error(f"  Stage 6 failed in equilibrium loop: {_s6e}")
        raise

    logger.info(
        f"═══ Equilibrium {'converged' if converged else 'finished'} "
        f"in {eq_i+1}/{n_iter} iterations ═══")

    # ── Post-processing (mirrors pipeline_logger.main) ────────────────────
    try:
        from stage5_destination_choice import compute_multilevel_validation
        if _hhts_tours_stage5_val is not None and not _hhts_tours_stage5_val.empty:
            _ml_val_dir = os.path.join(OUT_DIR, 'validation')
            compute_multilevel_validation(
                obs_tours=_hhts_tours_stage5_val,
                pred_plan=final_plan,
                landuse=landuse,
                save_dir=_ml_val_dir,
            )
            logger.info(f"  Multilevel validation outputs → {_ml_val_dir}/")
        else:
            logger.info("  Multilevel validation: no HHTS tours available, skipping")
    except Exception as _mlv_e:
        import traceback as _tb2; logger.debug(_tb2.format_exc())
        logger.warning(f"  Multilevel validation failed: {_mlv_e}")

    try:
        _pl.plot(final_plan)
        _synth_spatial = (_synth_home if _synth_home is not None
                          else pd.DataFrame(columns=['person_id', 'OTAZ']))
        _pl.spatial_plots(final_plan, _synth_spatial, landuse)
    except Exception as _plt_e:
        logger.warning(f"  Plotting failed in equilibrium loop: {_plt_e}")

    if skims:
        try:
            skims.close()
        except Exception:
            pass

    logger.info("Equilibrium pipeline complete.")
    return final_plan, final_sched, converged
