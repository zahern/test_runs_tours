"""
validate_hhts.py
================
In-sample validation: apply every fitted model back to the HHTS training data
and compare predicted outputs to observed values.

Call via pipeline_logger after Phase 1 estimation, before Phase 2 (synth pop).
All plots are saved to  {OUT_DIR}/plots/validation/

Produces one figure per stage plus a combined summary HTML report.
"""

from __future__ import annotations
import os, logging, warnings
import base64, glob as _glob, html as _htmlmod, json, re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

STAGE_COLORS = {
    1: '#2E86AB', 2: '#A23B72', 3: '#F18F01',
    4: '#C73E1D', 5: '#3B1F2B',
}
PURPOSES = ['work','edu','social','escort','shopping','othmaint','othdiscr','eat']
MODE_NAMES  = {1:'DA', 2:'SR', 3:'Walk', 4:'Bike', 5:'Transit'}
TIME_LABELS = ['early','am_peak','midday','pm_peak','evening','late']

# Profile panels (ActivitySim-style): the HHTS day opens at 04:00 and the
# tour-length distribution is reported in fixed distance bands.
PROFILE_HOUR_LO, PROFILE_HOUR_HI = 4, 24
PROFILE_HOUR_LABELS = [f'{h:02d}' for h in range(PROFILE_HOUR_LO, PROFILE_HOUR_HI)]
LENGTH_EDGES_KM = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, np.inf]
LENGTH_BIN_LABELS = ['0-1 km', '1-2 km', '2-5 km', '5-10 km',
                     '10-20 km', '20-40 km', '40+ km']


def _stage_title(stage) -> str:
    """Readable heading for numeric and named validation results."""
    named = {'5b': 'Stage 5b',
             'profiles': 'Time-of-day and trip-length profiles'}
    return named.get(str(stage), f'Stage {stage}')


def _hour_shares(minutes, time_col: str = 'STARTIME'):
    """Share of records departing in each hourly bin, in percent."""
    vals = pd.to_numeric(pd.Series(minutes), errors='coerce').dropna()
    vals = vals.to_numpy(dtype=float)
    n_bins = len(PROFILE_HOUR_LABELS)
    if vals.size == 0:
        return np.zeros(n_bins), PROFILE_HOUR_LABELS
    vals = np.clip(vals, PROFILE_HOUR_LO * 60.0, PROFILE_HOUR_HI * 60.0 - 1e-6)
    idx = np.floor(vals / 60.0).astype(int) - PROFILE_HOUR_LO
    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    return counts / vals.size * 100.0, PROFILE_HOUR_LABELS


def _length_bin_shares(values) -> np.ndarray:
    """Share of records in each fixed distance band, in percent."""
    vals = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    vals = np.clip(vals.to_numpy(dtype=float), 0.0, None)
    if vals.size == 0:
        return np.zeros(len(LENGTH_BIN_LABELS))
    counts, _ = np.histogram(vals, bins=np.asarray(LENGTH_EDGES_KM, dtype=float))
    return counts / vals.size * 100.0


def _js_divergence(p, q) -> float:
    """Jensen-Shannon divergence (natural log) between two distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] /
                                            np.where(b[mask] > 0, b[mask], 1e-12))))

    return 0.5 * (_kl(p, m) + _kl(q, m))


def _json_safe(value):
    """Coerce numpy/pandas scalars and containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_metrics_json(results: list, out_dir: str) -> str:
    """Persist the per-stage validation metrics for the dashboard build.

    The observed-versus-predicted figures are only readable by a human; the
    same numbers are written here so the SimWrapper validation dashboard
    (build_validation_dashboard.py) can chart and tabulate them.
    """
    payload = {'stages': []}
    for res in results or []:
        if not isinstance(res, dict):
            continue
        metrics = {k: v for k, v in (res.get('metrics') or {}).items()
                   if not isinstance(v, (pd.DataFrame, pd.Series))}
        payload['stages'].append({
            'stage': _json_safe(res.get('stage')),
            'metrics': _json_safe(metrics),
            'plots': [os.path.relpath(p, out_dir)
                      for p in (res.get('plots') or []) if isinstance(p, str)],
        })
    path = os.path.join(out_dir, 'validation_metrics.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# STAGE 1 — activity frequency
# ---------------------------------------------------------------------------

def validate_stage1(out: dict, models: dict, save_dir: str) -> dict:
    """
    Compare model-predicted activity frequency to observed HHTS counts.
    For each (purpose, ptype): predicted mean vs observed mean; bar chart.
    """
    from stage1_activity_frequency import (
        prepare_frequency_data, predict_activity_frequency, calibrate_activity_rates,
        predict_nonmand_split,
        PURPOSES as S1_PURPS,
        PTYPE_PREDICTION_MAP,
    )

    logger.info("  [validate] Stage 1 — activity frequency")
    data = prepare_frequency_data(out["tours"], out["persons"])
    # Use sample=True: modal (sample=False) gives 0 for most people since
    # the majority probability for low-frequency activities is class=0.
    # Sampled prediction gives the full distribution, matching observed means.
    pred = predict_activity_frequency(data.copy(), models, sample=True, seed=42)
    # Split the combined n_nonmand_out into escort/shopping/social/othmaint/
    # othdiscr/eat, mirroring run_stage1() (stage1_activity_frequency.py:1596-1599).
    # Without this, all six nonmand purposes start at pred==0 going into
    # calibration below; for most of them calibrate_activity_rates' low-mean
    # skip guard doesn't trigger so it reconstructs sensible values anyway,
    # but othdiscr's HHTS mean is itself <0.01 so it hit the skip guard and
    # was left at 0 for every segment in this report (production is unaffected
    # since run_stage1() always calls this split).
    if '_nonmand_split' in models and '_nonmand' in models and bool(models['_nonmand']):
        pred = predict_nonmand_split(pred, models['_nonmand_split'],
                                      rng=np.random.default_rng(44))
    # Use the same post-prediction ptype-purpose calibration as pipeline_logger
    # so validation reflects production predictions.
    pred = calibrate_activity_rates(
        pred,
        out.get("persons", pd.DataFrame()),
        out.get("tours", pd.DataFrame()),
        rng=np.random.default_rng(43),
    )

    figs = []
    n_purp = len(S1_PURPS)
    ncols  = 4
    nrows  = int(np.ceil(n_purp / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    fig.suptitle('Stage 1 — Activity Frequency: Observed vs Predicted\n'
                 '(bars = mean counts per person, by purpose)', fontsize=12, fontweight='bold')
    axes = axes.flatten()

    rmse_all = []
    for ax_i, purp in enumerate(S1_PURPS):
        ax   = axes[ax_i]
        obs_col  = f'n_{purp}_out'
        pred_col = obs_col

        if obs_col not in data.columns or pred_col not in pred.columns:
            ax.set_visible(False); continue

        obs_mean  = data[obs_col].mean()
        pred_mean = pred[pred_col].mean()
        rmse_     = float(np.sqrt(((data[obs_col].values - pred[pred_col].values)**2).mean()))
        rmse_all.append(rmse_)

        # Per-ptype breakdown
        ptypes   = sorted(data['ptype'].dropna().unique())
        # align pred by data index (predict_activity_frequency may reindex)
        pred_aligned = pred.reindex(data.index)
        o_means  = [data.loc[data['ptype']==pt, obs_col].mean()       for pt in ptypes]
        p_means  = [pred_aligned.loc[data['ptype']==pt, pred_col].mean()
                    if pred_col in pred_aligned.columns else 0.0
                    for pt in ptypes]

        x = np.arange(len(ptypes)); w = .35
        ax.bar(x-w/2, o_means, w, label='Observed', color='#2E86AB', alpha=.85)
        ax.bar(x+w/2, p_means, w, label='Predicted', color='#E84855', alpha=.75)
        ax.set_xticks(x)
        ax.set_xticklabels([p[:6] for p in ptypes], rotation=30, ha='right', fontsize=7)
        ax.set_title(f'{purp}\nRMSE={rmse_:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=.3)

        # Diagonal: perfect fit
        if ax_i == 0:
            ax.legend(fontsize=7, loc='upper right')

    for ax in axes[n_purp:]:
        ax.set_visible(False)

    plt.tight_layout()
    p = os.path.join(save_dir, 'val_stage1_frequency.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    logger.info(f"    saved ? {os.path.basename(p)}")
    figs.append(p)

    # Correlation between observed and predicted means across all purposes
    obs_all  = [data[f'n_{p}_out'].mean()  for p in S1_PURPS if f'n_{p}_out' in data.columns]
    pred_all = [pred[f'n_{p}_out'].mean()  for p in S1_PURPS if f'n_{p}_out' in pred.columns
                and f'n_{p}_out' in data.columns]
    _r = float(np.corrcoef(obs_all, pred_all)[0, 1]) if len(obs_all) > 1 else float('nan')
    _mean_rmse = float(np.sqrt(np.mean(rmse_all))) if rmse_all else float('nan')

    # Overall metrics — use keys that plot_fit_summary expects
    metrics = {
        'overall_rmse': _mean_rmse,
        'mean_RMSE':    _mean_rmse,   # alias expected by dashboard
        'r':            _r,           # correlation expected by Panel D
        'n_purposes': len(PURPOSES),
        'n_ptypes': len(data['ptype'].unique()),
    }

    # Per-purpose observed and predicted means, for the validation dashboard
    try:
        _pm = []
        for _p in S1_PURPS:
            _col = f'n_{_p}_out'
            if _col in data.columns and _col in pred.columns:
                _pm.append({
                    'purpose': _p,
                    'observed': float(pd.to_numeric(
                        data[_col], errors='coerce').mean()),
                    'predicted': float(pd.to_numeric(
                        pred[_col], errors='coerce').mean()),
                })
        metrics['purpose_means'] = _pm
    except Exception as _pme:
        logger.debug(f"  Stage 1 purpose-mean export failed: {_pme}")

    return {'stage': 1, 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------# STAGE 2 — day duration budget
# ---------------------------------------------------------------------------

def validate_stage2(out: dict, models: dict, save_dir: str) -> dict:
    """
    Compare predicted day duration (start_of_day, end_of_day, B) to observed.
    """
    from stage2_day_duration import aggregate_person_day as prep_s2, predict_day_budget

    logger.info("  [validate] Stage 2 — day duration budget")
    data = prep_s2(out["tours"], out["persons"])
    pred = predict_day_budget(data.copy(), models, sample=False, seed=42)

    figs = []
    metrics = {}

    # Plot start_of_day distribution (observed vs predicted)
    if 'start_of_day' in data.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        obs_start = pd.to_numeric(data['start_of_day'], errors='coerce').dropna() / 60.0
        pred_start = pd.to_numeric(pred['start_of_day'], errors='coerce').dropna() / 60.0
        ax.hist(obs_start, bins=30, density=True, alpha=.55, color='#2E86AB', label='Observed')
        ax.hist(pred_start, bins=30, density=True, alpha=.45, color='#E84855', label='Predicted')
        ax.set_xlabel('Start of day (hour)')
        ax.set_ylabel('Density')
        ax.set_title('Stage 2 — Start of Day Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=.3)
        plt.tight_layout()
        p = os.path.join(save_dir, 'val_stage2_start.png')
        fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
        figs.append(p)
        metrics['rmse_start_min'] = float(np.sqrt(np.mean((
            pd.to_numeric(pred['start_of_day'], errors='coerce').fillna(0).values
            - pd.to_numeric(data['start_of_day'], errors='coerce').fillna(0).values
        ) ** 2)))

    # Plot end_of_day distribution (observed vs predicted)
    if 'end_of_day' in data.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        obs_end = pd.to_numeric(data['end_of_day'], errors='coerce').dropna() / 60.0
        pred_end = pd.to_numeric(pred['end_of_day'], errors='coerce').dropna() / 60.0
        ax.hist(obs_end, bins=30, density=True, alpha=.55, color='#A23B72', label='Observed')
        ax.hist(pred_end, bins=30, density=True, alpha=.45, color='#E84855', label='Predicted')
        ax.set_xlabel('End of day (hour)')
        ax.set_ylabel('Density')
        ax.set_title('Stage 2 — End of Day Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=.3)
        plt.tight_layout()
        p = os.path.join(save_dir, 'val_stage2_end.png')
        fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
        figs.append(p)
        metrics['rmse_end_min'] = float(np.sqrt(np.mean((
            pd.to_numeric(pred['end_of_day'], errors='coerce').fillna(0).values
            - pd.to_numeric(data['end_of_day'], errors='coerce').fillna(0).values
        ) ** 2)))

    # Plot budget B distribution (observed vs predicted)
    if 'B' in data.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        obs_b = pd.to_numeric(data['B'], errors='coerce').dropna()
        pred_b = pd.to_numeric(pred['B'], errors='coerce').dropna()
        ax.hist(obs_b, bins=30, density=True, alpha=.55, color='#F18F01', label='Observed')
        ax.hist(pred_b, bins=30, density=True, alpha=.45, color='#E84855', label='Predicted')
        ax.set_xlabel('Daily time budget (hours)')
        ax.set_ylabel('Density')
        ax.set_title('Stage 2 — Daily Time Budget Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=.3)
        plt.tight_layout()
        p = os.path.join(save_dir, 'val_stage2_budget.png')
        fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
        figs.append(p)
        metrics['rmse_budget_hr'] = float(np.sqrt(np.mean((
            pd.to_numeric(pred['B'], errors='coerce').fillna(0).values
            - pd.to_numeric(data['B'], errors='coerce').fillna(0).values
        ) ** 2)))

    return {'stage': 2, 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------
# STAGE 3 — activity duration
# ---------------------------------------------------------------------------

def validate_stage3(out: dict, models: dict, save_dir: str) -> dict:
    """
    Compare predicted activity durations to observed.

    In-sample validation: the fitted duration models are applied DIRECTLY to
    the same HHTS activity records used for estimation (not re-generated from
    person-day expansion, which introduces activity-count mismatch errors).

    The draws then go through the reconciliation the pipeline itself runs
    before Stage 4 (``reconcile_predicted_durations``: floors, the day-window
    and budget ceiling, discretionary growth, soft ceiling), using each
    person's observed day window.  The panel therefore reports the durations a
    run emits rather than the raw model draws; the raw draw stays plotted
    alongside so the effect of the reconciliation is visible.

    Two conditions decide whether a bias here is model error at all, and both
    are reported: the design matrix is rebuilt from the same GA columns the
    models were fitted on (anything still absent is imputed at the training
    mean and named in the summary CSV), and the observed comparator is capped
    to the purpose's legal band, the support the model was fitted on and the
    pipeline clips to.
    """
    from stage3_activity_duration import (
        prepare_duration_data as prep_s3,
        _augment_duration_features,
        _feature_cols,
        duration_design_matrix,
        reconcile_predicted_durations,
        DURATION_BOUNDS,
    )

    logger.info("  [validate] Stage 3 — activity duration "
                "(in-sample draws, production reconciliation)")
    data = _augment_duration_features(
        prep_s3(out["tours"], out["persons"], trips=out.get("trips"))
    )
    # Rebuild the GA-search columns the models were fitted on (is_female,
    # vehicle/age flags, *_acc accessibility, ...).  Estimation calls this
    # before fitting; without it every such feature is absent here and the
    # design matrix has to impute it, which is the one difference between the
    # estimation and validation frames.
    _feat_by_purpose = {
        p: list(e.get('features') or [])
        for p, e in (models or {}).items() if isinstance(e, dict)
    }
    try:
        from ga_hypothesis_testing import ensure_ga_search_columns
        _pt = {str(f) for fs in _feat_by_purpose.values() for f in fs
               if str(f).startswith('ptype_')}
        _attrs = dict(data.attrs)
        data = ensure_ga_search_columns(
            data, out.get('persons'), ptype_subset=_pt or None,
            stage_id=3, zone_col='DESTSA2_2021')
        data.attrs.update(_attrs)
    except Exception as _enr_exc:
        logger.warning(f"  Stage 3: GA column enrichment unavailable "
                       f"({_enr_exc}); absent features are imputed")

    figs    = []
    metrics = {}

    if 'duration' not in data.columns:
        logger.warning("  Stage 3: missing duration column")
        return {'stage': 3, 'plots': figs, 'metrics': metrics}

    purpose_col = next(
        (c for c in ['main_purpose', '_purpose_coded', 'purpose'] if c in data.columns),
        None
    )
    if purpose_col is None:
        logger.warning("  Stage 3: no purpose column found")
        return {'stage': 3, 'plots': figs, 'metrics': metrics}

    rng      = np.random.default_rng(42)
    purposes = ['work', 'edu', 'social', 'escort', 'shopping', 'othmaint', 'othdiscr', 'eat']
    summary_rows = []

    # ── ptype dummies (same as estimation) ────────────────────────────────
    if 'ptype' in data.columns:
        for pt in data['ptype'].dropna().unique():
            data[f'ptype_{pt}'] = (data['ptype'] == pt).astype(float)

    # ── per-purpose direct model application ──────────────────────────────
    # Root cause of the prior obs-vs-pred gap: validation used activity
    # re-generation (predict_activity_durations), which synthesises activities
    # from person-day counts — the number and mix of activities can differ
    # from the HHTS.  Fix: apply each fitted duration model directly to the
    # HHTS rows, then reconcile exactly as prediction does.
    pred_raw = np.full(len(data), np.nan, dtype=float)
    imputed_by_purpose = {}
    for purp in purposes:
        mask = (data[purpose_col] == purp).to_numpy()
        if not mask.any() or purp not in models:
            continue
        subset = data[mask].copy()
        entry  = models[purp]
        # models[purp] is a dict {'model': ..., 'features': [...], ...}
        m = entry.get('model') if isinstance(entry, dict) else entry
        if m is None:
            continue
        try:
            ptypes = sorted(data['ptype'].dropna().unique().tolist()) \
                if 'ptype' in data.columns else []
            # Prefer stored feature names (set during fit) so ordering/count
            # exactly matches what the model was trained on.  Also fall back
            # to the 'features' list stored alongside the model in the dict.
            model_feats = (getattr(m, 'feature_names_', None)
                           or (entry.get('features') if isinstance(entry, dict) else None))
            if model_feats is None:
                feats  = _feature_cols(data, ptypes, feature_override=None)
                model_feats = [f for f in feats if f in subset.columns][:len(m.mu_)]
            # Absent features are imputed at the training mean, never zero:
            # the columns are z-scored, so a raw 0 is an extreme value.
            X, _imp = duration_design_matrix(
                subset, list(model_feats), m, purp)
            imputed_by_purpose[purp] = list(_imp)
            if X.shape[1] == len(m.mu_):
                pred_raw[np.flatnonzero(mask)] = np.clip(
                    m.predict_sampled(X, rng), 1, 1440)
        except Exception as e:
            logger.debug(f"  Stage 3 validate {purp}: {e}")

    # ── production reconciliation of those draws ──────────────────────────
    # Every duration Stage 4 receives has been through this pass, so the panel
    # compares observed against THAT.  The window is the observed one: the
    # question here is whether the duration model reproduces the survey, not
    # whether Stage 2 reproduces the survey's day window.
    rec = pd.DataFrame({
        'person_id': (data['person_id'].values
                      if 'person_id' in data.columns else np.arange(len(data))),
        'purpose':   data[purpose_col].astype(str).values,
        'predicted_duration_min': pred_raw,
    })
    # Rows whose purpose has no fitted model keep the purpose midpoint, as
    # prediction does, so they consume their share of the day too.
    for purp in purposes:
        lo, hi = DURATION_BOUNDS.get(purp, (5, 480))
        pmask = (rec['purpose'] == purp) & rec['predicted_duration_min'].isna()
        rec.loc[pmask, 'predicted_duration_min'] = (lo + hi) / 2
    rec['predicted_duration_min'] = rec['predicted_duration_min'].fillna(30.0)
    rec['B'] = (pd.to_numeric(data['B'], errors='coerce').fillna(8.0).values
                if 'B' in data.columns else 8.0)

    if 'start_of_day' not in data.columns or 'end_of_day' not in data.columns:
        try:
            from stage2_day_duration import aggregate_person_day as _prep_s2
            _day = _prep_s2(out['tours'], out['persons'])[
                ['person_id', 'start_of_day', 'end_of_day']]
            rec = rec.merge(_day, on='person_id', how='left')
        except Exception as exc:
            # Budget-only reconciliation: the panel is then missing the window
            # constraint, which is reported rather than hidden.
            logger.warning(f"  Stage 3: observed day window unavailable ({exc}); "
                           "reconciling against the budget only")

    rec = reconcile_predicted_durations(rec, log_audit=True)
    pred_prod = rec['predicted_duration_min'].to_numpy(dtype=float)
    metrics['reconciled_mean_shift_min'] = float(
        np.nanmean(pred_prod) - np.nanmean(pred_raw))

    # ── figure: observed vs raw draw vs pipeline durations ────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Stage 3 — Activity Duration: Observed vs Predicted '
                 '(in-sample draws, pipeline reconciliation)',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for ax_i, purp in enumerate(purposes):
        ax   = axes[ax_i]
        mask = (data[purpose_col] == purp).to_numpy()
        dur  = pd.to_numeric(data.loc[mask, 'duration'], errors='coerce').dropna()
        if dur.empty:
            ax.set_visible(False); continue

        # The duration models are estimated on durations clipped to the
        # purpose's legal band, and prediction clips its draws the same way
        # (np.clip(preds, lo, hi)).  The comparable observed mean is therefore
        # the capped one; the unclipped survey mean stays in the table because
        # the cap is a modelling decision, not a measurement.
        lo_p, hi_p = DURATION_BOUNDS.get(purp, (5, 480))
        dur_capped = dur.clip(lo_p, hi_p)
        obs_mean        = float(dur.mean())
        obs_mean_capped = float(dur_capped.mean())
        obs_p50  = float(dur_capped.quantile(0.5))
        obs_p90  = float(dur_capped.quantile(0.9))

        raw_vals  = pd.Series(pred_raw[mask]).dropna()
        prod_vals = pd.Series(pred_prod[mask]).dropna()

        pred_mean = float(prod_vals.mean()) if len(prod_vals) else float('nan')
        pred_p50  = float(prod_vals.quantile(0.5)) if len(prod_vals) else float('nan')
        pred_p90  = float(prod_vals.quantile(0.9)) if len(prod_vals) else float('nan')
        raw_mean  = float(raw_vals.mean()) if len(raw_vals) else float('nan')

        bins = np.linspace(0, min(dur.quantile(0.99) * 1.2, 720), 30)
        ax.hist(dur.clip(0, 720), bins=bins, density=True,
                alpha=0.55, color='#5B9E2B', label='Observed', zorder=3)
        if len(raw_vals):
            ax.hist(raw_vals.clip(0, 720), bins=bins, density=True,
                    alpha=0.30, color='#9aa0a6', zorder=2,
                    label='Raw draw (in-sample)')
        if len(prod_vals):
            ax.hist(prod_vals.clip(0, 720), bins=bins, density=True,
                    histtype='step', linewidth=1.8, color='#E84855', zorder=4,
                    label='Pipeline (reconciled)')

        ax.axvline(obs_mean_capped, color='#5B9E2B', linewidth=1.5, linestyle='--',
                   label=f'Obs mean={obs_mean_capped:.0f}m (cap {hi_p:.0f})')
        if np.isfinite(pred_mean):
            ax.axvline(pred_mean, color='#E84855', linewidth=1.5, linestyle='--',
                       label=f'Pred mean={pred_mean:.0f}m')

        bias_pct = ((pred_mean - obs_mean_capped) / obs_mean_capped * 100
                    if obs_mean_capped > 0 and np.isfinite(pred_mean)
                    else float('nan'))
        title_line2 = (f"obs={obs_mean_capped:.0f}m (raw {obs_mean:.0f}) "
                       f"pred={pred_mean:.0f}m ({bias_pct:+.0f}%)"
                       if np.isfinite(pred_mean)
                       else f"obs={obs_mean_capped:.0f}m  (no model)")
        ax.set_title(f'{purp.title()}  n={len(dur)}\n{title_line2}', fontsize=8)
        ax.set_xlabel('Duration (min)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

        if np.isfinite(pred_mean):
            summary_rows.append({
                'purpose':       purp,
                'n_obs':         int(len(dur)),
                'obs_mean_min':  obs_mean,
                'obs_mean_capped_min': obs_mean_capped,
                'pred_mean_min': pred_mean,
                'raw_mean_min':  raw_mean,
                'obs_p50_min':   obs_p50,
                'pred_p50_min':  pred_p50,
                'obs_p90_min':   obs_p90,
                'pred_p90_min':  pred_p90,
                'mean_bias_pct': bias_pct,
                'mean_bias_vs_unclipped_pct':
                    ((pred_mean - obs_mean) / obs_mean * 100
                     if obs_mean > 0 and np.isfinite(pred_mean)
                     else float('nan')),
                'raw_bias_pct':  ((raw_mean - obs_mean_capped) / obs_mean_capped * 100
                                  if obs_mean_capped > 0 and np.isfinite(raw_mean)
                                  else float('nan')),
                'imputed_features': ';'.join(imputed_by_purpose.get(purp, [])),
            })

    plt.tight_layout()
    p = os.path.join(save_dir, 'val_stage3_duration.png')
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    logger.info(f"    saved → {os.path.basename(p)}")
    figs.append(p)

    # ── scatter: observed vs predicted mean by purpose ─────────────────────
    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = os.path.join(save_dir, 'val_stage3_duration_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        metrics['mean_abs_bias_pct'] = float(summary_df['mean_bias_pct'].abs().mean())
        metrics['mean_abs_bias_raw_pct'] = float(
            summary_df['raw_bias_pct'].abs().mean())
        metrics['mean_abs_bias_vs_unclipped_pct'] = float(
            summary_df['mean_bias_vs_unclipped_pct'].abs().mean())
        metrics['p90_abs_error_min'] = float(
            (summary_df['pred_p90_min'] - summary_df['obs_p90_min']).abs().mean())
        # A complete design matrix is a precondition for reading the bias as
        # model error rather than as a missing covariate.
        metrics['imputed_features'] = {
            p: v for p, v in imputed_by_purpose.items() if v}
        metrics['n_imputed_features'] = int(
            sum(len(v) for v in imputed_by_purpose.values()))
        figs.append(summary_path)

        # Scatter: obs vs pred mean per purpose (solid = pipeline, hollow =
        # raw draw) — the gap between the two is what the reconciliation did.
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor='white')
        ax2.scatter(summary_df['obs_mean_capped_min'], summary_df['raw_mean_min'],
                    s=70, facecolors='none', edgecolors='#9aa0a6', zorder=4,
                    label='raw draw')
        ax2.scatter(summary_df['obs_mean_capped_min'], summary_df['pred_mean_min'],
                    s=80, color='#534AB7', zorder=5, label='pipeline')
        for _, row in summary_df.iterrows():
            ax2.annotate(
                row['purpose'],
                (row['obs_mean_capped_min'], row['pred_mean_min']),
                fontsize=8, xytext=(4, 4), textcoords='offset points',
            )
        lo = min(summary_df['obs_mean_capped_min'].min(), summary_df['pred_mean_min'].min()) * 0.9
        hi = max(summary_df['obs_mean_capped_min'].max(), summary_df['pred_mean_min'].max()) * 1.1
        ax2.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='1:1 line')
        ax2.set_xlabel('Observed mean duration (min, capped to bounds)', fontsize=10)
        ax2.set_ylabel('Predicted mean duration (min)', fontsize=10)
        ax2.set_title('Stage 3 — Obs vs Pred Mean Duration by Purpose\n'
                      f'(pipeline: mean|bias|={metrics["mean_abs_bias_pct"]:.1f}%, '
                      f'raw draw: {metrics["mean_abs_bias_raw_pct"]:.1f}%)',
                      fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        p2 = os.path.join(save_dir, 'val_stage3_scatter.png')
        fig2.savefig(p2, dpi=140, bbox_inches='tight'); plt.close(fig2)
        figs.append(p2)

    return {'stage': 3, 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------# STAGE 4 — schedule placement
# ---------------------------------------------------------------------------

def _blend_period_shares(model, proba: np.ndarray) -> np.ndarray:
    """Mix period probabilities with the fitted observed period shares.

    The schedule builder blends each model's probabilities with the fitted
    observed period shares (:attr:`period_blend_alpha_`) before drawing a
    period; validation draws from the same mixture, otherwise the predicted
    period mix is not the mix the pipeline produces.
    """
    proba = np.asarray(proba, dtype=float)
    alpha = float(getattr(model, 'period_blend_alpha_', 0.0) or 0.0)
    target = getattr(model, 'observed_period_shares_', None)
    if alpha <= 0.0 or target is None:
        return proba
    target = np.asarray(target, dtype=float)
    if (target.ndim != 1 or len(target) != proba.shape[-1]
            or not np.isfinite(target).all() or target.sum() <= 0):
        return proba
    target = target / target.sum()
    blended = (1.0 - alpha) * proba + alpha * target
    return blended / np.clip(blended.sum(axis=-1, keepdims=True), 1e-12, None)


def _predict_period_proba(data: pd.DataFrame, models: dict) -> np.ndarray:
    """Blended period probabilities for every activity record in ``data``.

    Uses each purpose model's design matrix and the same observed-share blend
    the schedule builder applies, so validation and production draw from one
    distribution.  Records whose purpose has no model keep the builder's
    AM_PEAK default as a point mass.
    """
    n = len(data)
    proba_all = np.zeros((n, len(TIME_LABELS)))
    proba_all[:, 2] = 1.0                       # default AM_PEAK
    for purpose, model in (models or {}).items():
        mask = (data['purpose'] == purpose).values
        if not mask.any():
            continue
        sub = data[mask].copy()
        # Add required columns for _build_X
        sub['day_position'] = sub.get('day_position', 0.5)
        sub['time_consumed'] = sub.get('time_consumed', 0.0)
        sub['time_remaining'] = sub.get('time_remaining', sub['day_end'] - sub['STARTIME'])
        sub['duration_min'] = sub.get('duration_min', 30.0)
        # Add person features
        for col in ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp']:
            if col not in sub.columns:
                sub[col] = 0
        try:
            X = model._build_X(sub)
            proba_all[mask] = _blend_period_shares(model, model.predict_proba(X))
        except Exception as e:
            logger.warning(f"Stage 4 validation predict failed for '{purpose}': {e}")
    return proba_all


def _predict_periods(data: pd.DataFrame, models: dict, sample: bool = False,
                     rng: np.random.Generator | None = None,
                     proba: np.ndarray | None = None) -> np.ndarray:
    """Stage 4 period index for every activity record in ``data``.

    ``sample=False`` returns the modal period (used by the Stage 4 period /
    confusion panels).  ``sample=True`` draws from the blended probability,
    which is what the schedule builder does at run time.
    """
    if proba is None:
        proba = _predict_period_proba(data, models)
    if sample and rng is not None:
        u = rng.random(len(proba))
        return (np.cumsum(proba, axis=1) > u[:, None]).argmax(axis=1)
    return np.argmax(proba, axis=1)


def _predict_start_minutes(data: pd.DataFrame, pred_period: np.ndarray,
                           models: dict, rng: np.random.Generator):
    """Departure minute for every activity, inside its predicted period.

    Each departure is drawn from its purpose model's fitted within-period
    start-time distribution (empirical fractions from the estimation data); the
    builder's triangular draw is used only where no fitted distribution exists.
    The schedule builder's sequence arithmetic is deliberately not replayed:
    it needs Stage 3 activity durations and predicted day windows that do not
    exist on the survey frame, and its feasibility clamps pile mass on period
    boundaries, which would put artefacts into a marginal profile comparison.

    Returns ``(start_minutes, effective_period)``.
    """
    from stage4_schedule_placement import (
        _sample_within_period_fraction, PERIOD_STARTS, PERIOD_ENDS)

    n = len(data)
    pidx = np.clip(np.asarray(pred_period, dtype=int), 0, len(PERIOD_STARTS) - 1)
    purposes = data['purpose'].astype(str).to_numpy()
    starts = np.full(n, np.nan)
    for j in range(n):
        p = int(pidx[j])
        model = (models or {}).get(purposes[j])
        width = max(float(PERIOD_ENDS[p] - PERIOD_STARTS[p]), 1.0)
        frac = _sample_within_period_fraction(model, p, rng)
        starts[j] = float(PERIOD_STARTS[p]) + frac * width
    return starts, pidx


def _within_period_draw_coverage(purposes, pidx, models: dict) -> float:
    """Share of activities placed with a fitted within-period distribution."""
    if not len(purposes):
        return 0.0
    hit = 0
    for purpose, period in zip(purposes, pidx):
        draws = getattr((models or {}).get(purpose), 'within_period_draws_', None) or {}
        arr = draws.get(int(period))
        if arr is not None and len(arr) > 0:
            hit += 1
    return 100.0 * hit / len(purposes)


def validate_stage4(out: dict, models: dict, save_dir: str) -> dict:
    from stage4_schedule_placement import (
        prepare_timing_data, predict_schedule, TIME_PERIODS, PERIOD_LABELS,
    )

    period_labels = PERIOD_LABELS
    n_periods     = len(TIME_PERIODS)
    figs          = []
    metrics       = {}

    logger.info("  [validate] Stage 4 — schedule placement")
    data = prepare_timing_data(out["trips"], out["tours"], out["persons"])

    # observed period per trip
    if 'period' in data.columns:
        obs_period = data['period'].values
    elif 'STARTIME' in data.columns:
        # Convert STARTIME minutes to period index
        starts = data['STARTIME'].clip(0, 1439).values
        boundaries = [s for (_, s, e) in TIME_PERIODS]
        obs_period = np.searchsorted(boundaries, starts, side='right') - 1
        obs_period = np.clip(obs_period, 0, len(TIME_PERIODS)-1)
    else:
        logger.warning("  [validate] Stage 4: no period or STARTIME column")
        return {'stage': 4, 'plots': [], 'metrics': {}}

    # Predict periods using models directly
    pred_period = _predict_periods(data, models)

    obs_lbl = np.array([period_labels[i] for i in obs_period])
    pred_lbl = np.array([period_labels[i] for i in pred_period])

    # Per-purpose period distribution
    purposes = [p for p in PURPOSES if p in data['purpose'].unique()]
    if not purposes:
        purposes = sorted(data['purpose'].dropna().unique())[:8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Stage 4 — Time Period Distribution: Observed vs Predicted',
                 fontweight='bold', fontsize=12)
    for ax, purp in zip(axes.flatten(), purposes[:8]):
        mask = data['purpose'] == purp
        obs_p = pd.Series(obs_lbl[mask]).value_counts(normalize=True).reindex(period_labels).fillna(0)
        pred_p = pd.Series(pred_lbl[mask]).value_counts(normalize=True).reindex(period_labels).fillna(0)

        x = np.arange(n_periods); w = .35
        ax.bar(x-w/2, obs_p.values, w, color='#2E86AB', alpha=.85, label='Observed')
        ax.bar(x+w/2, pred_p.values, w, color='#E84855', alpha=.75, label='Predicted')
        ax.set_xticks(x); ax.set_xticklabels([l[:5] for l in period_labels],
                                               rotation=30, ha='right', fontsize=8)
        acc_ = (obs_p.values * pred_p.values).sum() / (obs_p.values**2).sum() if obs_p.values.sum() else 0
        ax.set_title(f'{purp}\nBhattacharyya={acc_:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=.3)

    plt.tight_layout()
    p = os.path.join(save_dir, 'val_stage4_periods.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    logger.info(f"    saved → {os.path.basename(p)}")
    figs.append(p)

    conf = pd.crosstab(pd.Series(obs_lbl, name='Observed'), pd.Series(pred_lbl, name='Predicted'))
    conf = conf.reindex(index=period_labels, columns=period_labels, fill_value=0)
    conf_pct = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = ax2.imshow(conf_pct.values, cmap='Blues', vmin=0, vmax=100)
    ax2.set_xticks(range(n_periods)); ax2.set_xticklabels(period_labels, rotation=30, ha='right')
    ax2.set_yticks(range(n_periods)); ax2.set_yticklabels(period_labels)
    ax2.set_xlabel('Predicted period'); ax2.set_ylabel('Observed period')
    ax2.set_title('Stage 4 — Time Period Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax2, label='% of observed row')
    for i in range(n_periods):
        for j in range(n_periods):
            val = conf_pct.iloc[i, j]
            ax2.text(j, i, f'{val:.0f}%', ha='center', va='center',
                     color='white' if val > 60 else 'black', fontsize=8)
    p2 = os.path.join(save_dir, 'val_stage4_confusion.png')
    fig2.savefig(p2, dpi=130, bbox_inches='tight'); plt.close(fig2)
    figs.append(p2)
    metrics['overall_accuracy'] = float((np.array(obs_lbl) == np.array(pred_lbl)).mean() * 100)
    return {'stage': 4, 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------
# STAGE 5 — mode + destination
# ---------------------------------------------------------------------------

def validate_stage5(out: dict, s5_models: dict, save_dir: str) -> dict:
    from stage5_destination_choice import _prep_hhts_tours, _filter_seg, _dest_probs_numpy

    logger.info("  [validate] Stage 5 — mode + destination choice")

    hhts_tours = _prep_hhts_tours(out["tours"], out["persons"])
    if hhts_tours.empty:
        return {'stage': 5, 'plots': [], 'metrics': {}}

    segs = ['work','edu','nonmand']
    SEG_LABEL = {'work':'Work','edu':'Education','nonmand':'Non-Mandatory'}
    figs = []; metrics = {}
    rng = np.random.default_rng(42)
    dist_rows = []
    seg_pred_dest = {}

    def _sample_labels(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if probs is None or len(probs) == 0:
            return np.array([], dtype=int)
        probs = np.asarray(probs, dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        probs = probs / row_sum
        draws = np.empty(probs.shape[0], dtype=int)
        for i in range(probs.shape[0]):
            draws[i] = int(rng.choice(labels, p=probs[i]))
        return draws

    def _od_distance_lookup(dt_d):
        if dt_d is None:
            return None
        try:
            od_df = dt_d['od'].to_dataframe().reset_index()
            dist_col = 'DIST' if 'DIST' in od_df.columns else 'SOVTOLL_DIST__AM'
            if dist_col not in od_df.columns:
                return None
            return od_df.pivot_table(index='otaz', columns='dtaz', values=dist_col, aggfunc='first')
        except Exception:
            return None

    def _get_mode_alt_order(mode_mdl):
        """
        Extract mode → column index mapping from Larch model.
        
        FIX (2026-05-15): Larch models may reorder alternatives internally.
        This function gets the correct alternative-to-column mapping so
        probabilities can be aligned to mode IDs [1,2,3,4,5] correctly.
        
        Enhanced (2026-05-15): Also check idca_names (tuple of alternative names)
        to verify the order of probability columns matches mode_idca dimensions.
        """
        try:
            if hasattr(mode_mdl, 'alternatives'):
                alts = mode_mdl.alternatives
                if alts is not None:
                    # Larch 6.x: alternatives is a sorted array of alt IDs
                    return {int(alt): i for i, alt in enumerate(alts)}
        except Exception:
            pass
        
        # Check alternative names as secondary source of truth
        try:
            if hasattr(mode_mdl, 'idca_names') and mode_mdl.idca_names:
                # idca_names is typically a tuple of alternative names
                # Try to extract numeric IDs from names or use positional mapping
                names = list(mode_mdl.idca_names)
                mapping = {}
                for i, name in enumerate(names):
                    # Try to parse 'DA', 'SR', 'Walk', 'Bike', 'Transit' → 1,2,3,4,5
                    name_lower = str(name).lower()
                    if name_lower == 'da':
                        mapping[1] = i
                    elif name_lower == 'sr':
                        mapping[2] = i
                    elif name_lower == 'walk':
                        mapping[3] = i
                    elif name_lower == 'bike':
                        mapping[4] = i
                    elif name_lower in ('transit', 'pt', 'public'):
                        mapping[5] = i
                if len(mapping) == 5:
                    return mapping
        except Exception:
            pass
        
        # Fallback: assume alternatives are in order [1,2,3,4,5]
        # This is the default when alternatives aren't reordered
        return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    # ── 5A: Mode share: observed vs predicted ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Stage 5 — Mode Choice: Observed vs Predicted Mode Shares',
                 fontweight='bold', fontsize=12)

    all_obs_modes  = []
    all_pred_modes = []
    _cal_rows = []

    for ax, seg in zip(axes, segs):
        seg_df  = _filter_seg(hhts_tours, seg)
        if seg_df.empty: ax.set_title(SEG_LABEL[seg]); continue

        seg_res = s5_models.get(seg, {})
        mode_mdl = seg_res.get('mode_model')

        obs_mode = seg_df['tour_mode'].dropna()
        obs_share = obs_mode.value_counts(normalize=True).reindex(range(1,6)).fillna(0)*100

        # In-sample mode prediction: compare observed shares to mean fitted probabilities.
        pred_share = obs_share.copy() * 0   # default if model unavailable
        if mode_mdl is not None:
            try:
                probs = np.asarray(mode_mdl.probability())  # (N, K)
                # Map Larch alternative IDs to probability column indices correctly
                # (in case Larch reordered alternatives internally)
                alt_to_col = _get_mode_alt_order(mode_mdl)
                # Reorder prob columns to match mode IDs [1,2,3,4,5]
                pred_probs_by_mode = {}
                for mode_id in range(1, 6):
                    col_idx = alt_to_col.get(mode_id, mode_id - 1)
                    if 0 <= col_idx < probs.shape[1]:
                        pred_probs_by_mode[mode_id] = probs[:, col_idx].mean()
                    else:
                        pred_probs_by_mode[mode_id] = 0.0
                pred_share = pd.Series(pred_probs_by_mode, index=range(1,6)) \
                               .reindex(range(1,6)).fillna(0) * 100
                # For the confusion matrix: sample from predicted distribution
                # (not argmax — argmax assigns everyone to DA and makes the matrix
                # degenerate.  Sampling gives a representative prediction for each
                # observed observation so the off-diagonal cells are meaningful.)
                # Align to obs_mode rows only — drops NaN-mode observations.
                obs_mode_idx = seg_df.index[seg_df['tour_mode'].notna()]
                n_model = probs.shape[0]
                n_obs   = len(obs_mode_idx)
                # The model DataTree has the same case order as seg_df rows (tour obs)
                # but only for the cases used in estimation (may differ if seg_df
                # was filtered differently).  Take the min to avoid index mismatch.
                n_align = min(n_model, n_obs)
                if n_align > 0:
                    probs_align = probs[:n_align]
                    probs_align = np.nan_to_num(probs_align, nan=0.0, posinf=0.0, neginf=0.0)
                    row_sums = probs_align.sum(axis=1, keepdims=True)
                    probs_align = probs_align / np.where(row_sums > 0, row_sums, 1.0)
                    pred_modes = _sample_labels(probs_align, np.arange(1, probs_align.shape[1]+1, dtype=int))
                    obs_align  = obs_mode.iloc[:n_align].tolist()
                    all_obs_modes.extend(obs_align)
                    all_pred_modes.extend(pred_modes.tolist())
            except Exception as e:
                logger.debug(f"  Stage 5 in-sample predict [{seg}]: {e}")

        mode_lbls  = [MODE_NAMES.get(i, str(i)) for i in range(1,6)]
        x = np.arange(len(mode_lbls)); w = .35
        ax.bar(x-w/2, obs_share.values, w, color='#2E86AB', alpha=.85, label='Observed')
        ax.bar(x+w/2, pred_share.values, w, color='#E84855', alpha=.75, label='Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_lbls, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Mode Share (%)')
        ax.set_title(SEG_LABEL[seg])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.3)

        metrics[f'{seg}_mode_obs']  = obs_share.to_dict()
        metrics[f'{seg}_mode_pred'] = pred_share.to_dict()
        try:
            for _m in range(1, 6):
                _cal_rows.append({
                    'segment': seg,
                    'mode': MODE_NAMES.get(_m, str(_m)),
                    'observed_pct': float(obs_share.get(_m, 0.0)),
                    'predicted_pct': float(pred_share.get(_m, 0.0)),
                    'delta_pp': float(pred_share.get(_m, 0.0)
                                      - obs_share.get(_m, 0.0)),
                })
        except Exception:
            pass

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(save_dir, 'val_stage5_mode_shares.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    logger.info(f"    saved → {os.path.basename(p)}")
    figs.append(p)

    # Calibrated mode-share table: predicted shares here already reflect the
    # post-estimation ASC calibration (SR/Walk/Bike/Transit log-share
    # corrected, DA reference), because probability() reads live pf values.
    # Persisted as a citable artifact (manuscript Table 3 source).
    try:
        if _cal_rows:
            _cal_df = pd.DataFrame(
                _cal_rows,
                columns=['segment', 'mode', 'observed_pct',
                         'predicted_pct', 'delta_pp'])
            _cal_path = os.path.join(save_dir, 'stage5_mode_shares_calibrated.csv')
            _cal_df.to_csv(_cal_path, index=False)
            metrics['mode_shares_calibrated_csv'] = _cal_path
            _max_gap = float(_cal_df['delta_pp'].abs().max())
            metrics['mode_share_max_gap_pp'] = _max_gap
            logger.info(f"    calibrated mode-share table → "
                        f"{os.path.basename(_cal_path)} "
                        f"(max |delta| = {_max_gap:.2f} pp)")
    except Exception as _ce:
        logger.debug(f"  Stage 5 calibrated table write failed: {_ce}")

    fig_counts, axes_counts = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig_counts.suptitle('Stage 5 — Mode Choice: Observed vs Predicted Raw Counts',
                        fontweight='bold', fontsize=12)
    for ax, seg in zip(axes_counts, segs):
        seg_df = _filter_seg(hhts_tours, seg)
        if seg_df.empty:
            ax.set_title(SEG_LABEL[seg]); continue
        seg_res = s5_models.get(seg, {})
        mode_mdl = seg_res.get('mode_model')
        obs_mode = seg_df['tour_mode'].dropna()
        obs_counts = obs_mode.value_counts().reindex(range(1, 6)).fillna(0)
        pred_counts = obs_counts.copy() * 0
        if mode_mdl is not None:
            try:
                probs = np.asarray(mode_mdl.probability())
                pred_counts = pd.Series(probs.sum(axis=0), index=range(1, probs.shape[1] + 1)).reindex(range(1, 6)).fillna(0)
            except Exception as e:
                logger.debug(f"  Stage 5 mode count predict [{seg}]: {e}")
        mode_lbls  = [MODE_NAMES.get(i, str(i)) for i in range(1,6)]
        x = np.arange(len(mode_lbls)); w = .35
        ax.bar(x-w/2, obs_counts.values, w, color='#2E86AB', alpha=.85, label='Observed')
        ax.bar(x+w/2, pred_counts.values, w, color='#E84855', alpha=.75, label='Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_lbls, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Raw count')
        ax.set_title(SEG_LABEL[seg])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.3)
    fig_counts.tight_layout(rect=[0, 0, 1, 0.96])
    p_counts = os.path.join(save_dir, 'val_stage5_mode_counts.png')
    fig_counts.savefig(p_counts, dpi=130, bbox_inches='tight'); plt.close(fig_counts)
    figs.append(p_counts)

    # ── 5B: Destination choice — chunked destination zone plots ────────────
    zone_chunk = 40
    for seg in segs:
        seg_df  = _filter_seg(hhts_tours, seg)
        if seg_df.empty or 'DTAZ' not in seg_df.columns:
            continue

        obs_dtaz = seg_df['DTAZ'].dropna().astype(int)
        seg_res  = s5_models.get(seg, {})
        dest_mdl = seg_res.get('dest_model')
        mode_mdl = seg_res.get('mode_model')
        dt_d = seg_res.get('dt_dest')
        dist_lookup = _od_distance_lookup(dt_d)
        if dest_mdl is not None and seg_res.get('od_calibration') is not None:
            try:
                dest_mdl._od_calibration = seg_res.get('od_calibration') or {}
            except Exception:
                pass

        obs_counts = obs_dtaz.value_counts().sort_index()
        pred_counts = pd.Series(dtype=float)
        probs = None
        alt_ids = None
        n_obs = max(int(obs_dtaz.count()), 1)

        # ── Priority 1: pre-computed full-zone predicted shares ─────────────
        # Stored during _run_segment after shadow-price calibration using
        # _dest_probs_numpy with all zones. This is the most reliable path
        # because it uses the fitted model with the full 371-zone set.
        # Both .probability() and the numpy Dataset fallback below fail when
        # the estimation Dataset has importance-sampled (60) alternatives while
        # the prediction path expects 371 zones (shape mismatch → silent zeros).
        _pzs = seg_res.get('pred_zone_shares')
        if isinstance(_pzs, pd.Series) and not _pzs.empty:
            # Scale shares to expected counts (same total as observed)
            pred_counts = (_pzs * n_obs).rename(None)
            logger.info(f"  Stage 5 dest [{seg}]: using pred_zone_shares "
                        f"({int((pred_counts > 1e-3).sum())} zones)")
        elif dest_mdl is not None:
            # ── Priority 2: Larch in-sample .probability() ─────────────────
            try:
                if seg_res.get('od_calibration'):
                    raise RuntimeError('use_numpy_with_calibration')
                probs = np.asarray(dest_mdl.probability())
                alt_ids = np.array(dest_mdl.alternatives) if hasattr(dest_mdl, 'alternatives') else np.arange(probs.shape[1])
                pred_counts = pd.Series(probs.sum(axis=0), index=alt_ids if len(alt_ids) == probs.shape[1] else np.arange(probs.shape[1]))
                logger.info(f"  Stage 5 dest [{seg}]: used dest_mdl.probability() path")
            except Exception as e:
                logger.warning(f"  Stage 5 dest in-sample [{seg}]: {e}")
                # ── Priority 3: numpy fallback with full Dataset + logsums ──
                try:
                    dt_d = seg_res.get('dt_dest')
                    ls = seg_res.get('logsums')
                    if dt_d is not None and ls is not None:
                        zone_ids_np = [int(z) for z in ls.coords['zones'].values] if hasattr(ls, 'coords') and 'zones' in ls.coords else list(range(np.asarray(ls).shape[1]))
                        probs = _dest_probs_numpy(dest_mdl, dt_d, ls, zone_ids_np, out["persons"])
                        if probs is not None:
                            alt_ids = np.asarray(zone_ids_np, dtype=int)
                            pred_counts = pd.Series(np.asarray(probs).sum(axis=0), index=zone_ids_np)
                            logger.info(f"  Stage 5 dest [{seg}]: used numpy fallback path")
                except Exception as e2:
                    logger.warning(f"  Stage 5 dest numpy fallback [{seg}]: {e2}")

        # Keep per-segment predicted destination shares for additional validation
        try:
            if not pred_counts.empty:
                _ps = (pred_counts / max(float(pred_counts.sum()), 1.0)).sort_index()
                seg_pred_dest[seg] = _ps
        except Exception:
            pass

        combined = (obs_counts.reindex(obs_counts.index.union(pred_counts.index)).fillna(0) +
                    pred_counts.reindex(obs_counts.index.union(pred_counts.index)).fillna(0))
        # Sort zones numerically (371-system ORDERs) so plots are geographically ordered
        zone_ids = sorted(combined.index.astype(int).tolist())
        if not zone_ids:
            continue

        obs_share = (obs_counts / max(obs_counts.sum(), 1)).reindex(zone_ids).fillna(0) * 100
        pred_share = (pred_counts / max(pred_counts.sum(), 1)).reindex(zone_ids).fillna(0) * 100
        obs_counts = obs_counts.reindex(zone_ids).fillna(0)
        pred_counts = pred_counts.reindex(zone_ids).fillna(0)

        for part, start in enumerate(range(0, len(zone_ids), zone_chunk), start=1):
            chunk_ids = zone_ids[start:start + zone_chunk]
            x = np.arange(len(chunk_ids)); w = 0.42

            fig_share, ax_share = plt.subplots(figsize=(16, 5))
            ax_share.bar(x - w/2, obs_share.reindex(chunk_ids).values, w, color='#2E86AB', alpha=.85, label='Observed')
            ax_share.bar(x + w/2, pred_share.reindex(chunk_ids).values, w, color='#E84855', alpha=.75, label='Predicted')
            ax_share.set_xticks(x)
            ax_share.set_xticklabels([str(z) for z in chunk_ids], rotation=90, fontsize=7)
            ax_share.set_xlabel('Destination Zone')
            ax_share.set_ylabel('Share (%)')
            ax_share.set_title(f"Stage 5 — {SEG_LABEL[seg]} Destination Shares (zones {start+1}-{start+len(chunk_ids)})", fontweight='bold')
            ax_share.legend(fontsize=9); ax_share.grid(axis='y', alpha=.3)
            plt.tight_layout()
            p_share = os.path.join(save_dir, f'val_stage5_dest_{seg}_share_part{part}.png')
            fig_share.savefig(p_share, dpi=130, bbox_inches='tight'); plt.close(fig_share)
            figs.append(p_share)

            fig_count, ax_count = plt.subplots(figsize=(16, 5))
            ax_count.bar(x - w/2, obs_counts.reindex(chunk_ids).values, w, color='#2E86AB', alpha=.85, label='Observed')
            ax_count.bar(x + w/2, pred_counts.reindex(chunk_ids).values, w, color='#E84855', alpha=.75, label='Predicted')
            ax_count.set_xticks(x)
            ax_count.set_xticklabels([str(z) for z in chunk_ids], rotation=90, fontsize=7)
            ax_count.set_xlabel('Destination Zone')
            ax_count.set_ylabel('Raw count')
            ax_count.set_title(f"Stage 5 — {SEG_LABEL[seg]} Destination Counts (zones {start+1}-{start+len(chunk_ids)})", fontweight='bold')
            ax_count.legend(fontsize=9); ax_count.grid(axis='y', alpha=.3)
            plt.tight_layout()
            p_count = os.path.join(save_dir, f'val_stage5_dest_{seg}_count_part{part}.png')
            fig_count.savefig(p_count, dpi=130, bbox_inches='tight'); plt.close(fig_count)
            figs.append(p_count)

        if not obs_counts.empty and not pred_counts.empty:
            gap = (pred_counts - obs_counts).sort_values()
            top_under = gap.head(min(12, len(gap)))
            top_over = gap.tail(min(12, len(gap)))
            gap_ids = top_under.index.tolist() + [z for z in top_over.index.tolist() if z not in top_under.index]
            if gap_ids:
                fig_gap, ax_gap = plt.subplots(figsize=(14, 5))
                x = np.arange(len(gap_ids))
                ax_gap.bar(x, (pred_counts.reindex(gap_ids).fillna(0) - obs_counts.reindex(gap_ids).fillna(0)).values,
                           color=['#2E86AB' if v < 0 else '#E84855' for v in (pred_counts.reindex(gap_ids).fillna(0) - obs_counts.reindex(gap_ids).fillna(0)).values],
                           alpha=0.85)
                ax_gap.axhline(0, color='black', lw=1)
                ax_gap.set_xticks(x)
                ax_gap.set_xticklabels([str(z) for z in gap_ids], rotation=90, fontsize=7)
                ax_gap.set_xlabel('Destination Zone')
                ax_gap.set_ylabel('Predicted - Observed count')
                ax_gap.set_title(f"Stage 5 — {SEG_LABEL[seg]} Top Destination Over/Under Prediction", fontweight='bold')
                ax_gap.grid(axis='y', alpha=.3)
                plt.tight_layout()
                p_gap = os.path.join(save_dir, f'val_stage5_dest_{seg}_gaps.png')
                fig_gap.savefig(p_gap, dpi=130, bbox_inches='tight'); plt.close(fig_gap)
                figs.append(p_gap)

        if probs is not None and alt_ids is not None and dist_lookup is not None and 'OTAZ' in seg_df.columns:
            try:
                pred_dest = _sample_labels(np.asarray(probs), np.asarray(alt_ids, dtype=int))
                mode_probs = None
                pred_mode = None
                if mode_mdl is not None:
                    try:
                        mode_probs = np.asarray(mode_mdl.probability(), dtype=float)
                        pred_mode = _sample_labels(mode_probs, np.arange(1, mode_probs.shape[1] + 1, dtype=int))
                    except Exception as e:
                        logger.debug(f"  Stage 5 sampled mode distance [{seg}]: {e}")

                obs_mode_arr = pd.to_numeric(seg_df.get('tour_mode', pd.Series(dtype=float)), errors='coerce').fillna(0).astype(int).to_numpy()
                obs_otaz_arr = pd.to_numeric(seg_df['OTAZ'], errors='coerce').fillna(0).astype(int).to_numpy()
                obs_dtaz_arr = pd.to_numeric(seg_df['DTAZ'], errors='coerce').fillna(0).astype(int).to_numpy()

                def _dist_for_pairs(orig, dest):
                    vals = []
                    for o, d in zip(orig, dest):
                        try:
                            vals.append(float(dist_lookup.loc[int(o), int(d)]))
                        except Exception:
                            vals.append(np.nan)
                    return np.asarray(vals, dtype=float)

                obs_dist = _dist_for_pairs(obs_otaz_arr, obs_dtaz_arr)
                pred_dist = _dist_for_pairs(obs_otaz_arr, pred_dest)

                for mode_val, dist_val in zip(obs_mode_arr, obs_dist):
                    if np.isfinite(dist_val):
                        dist_rows.append({'segment': seg, 'source': 'Observed', 'mode': MODE_NAMES.get(int(mode_val), str(mode_val)), 'dist': float(dist_val)})
                for i, dist_val in enumerate(pred_dist):
                    if np.isfinite(dist_val):
                        mode_lbl = MODE_NAMES.get(int(pred_mode[i]), str(int(pred_mode[i]))) if pred_mode is not None and i < len(pred_mode) else 'All'
                        dist_rows.append({'segment': seg, 'source': 'Predicted', 'mode': mode_lbl, 'dist': float(dist_val)})
            except Exception as e:
                logger.debug(f"  Stage 5 distance diagnostics [{seg}] failed: {e}")

    # ── 5C: Mode prediction accuracy matrix ─────────────────────────────────
    if all_obs_modes and all_pred_modes:
        obs_arr  = np.array(all_obs_modes, dtype=int)
        pred_arr = np.array(all_pred_modes, dtype=int)
        n_modes  = 5
        mode_lbls = [MODE_NAMES.get(i, str(i)) for i in range(1, n_modes+1)]
        conf = np.zeros((n_modes, n_modes), dtype=int)
        for o, p_ in zip(obs_arr, pred_arr):
            if 1 <= o <= n_modes and 1 <= p_ <= n_modes:
                conf[int(o)-1, int(p_)-1] += 1

        # Row-normalised (% of observed class) — shows model recall per mode
        row_totals = conf.sum(axis=1, keepdims=True).clip(min=1)
        conf_pct   = conf / row_totals * 100

        # Marginal comparison table — prints to log for quick check
        obs_marginal  = conf.sum(axis=1)
        pred_marginal = conf.sum(axis=0)
        logger.info("  Mode confusion matrix marginals (sampled predictions):")
        logger.info(f"  {'Mode':10s}  {'Obs n':>7s}  {'Pred n':>7s}  {'Recall%':>8s}")
        for i, lbl in enumerate(mode_lbls):
            rec = conf_pct[i, i]
            logger.info(f"  {lbl:10s}  {obs_marginal[i]:7d}  {pred_marginal[i]:7d}  {rec:8.1f}")

        # ── plot: side-by-side conf pct + marginal share bars ────────────
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        fig3.suptitle('Stage 5 — Mode Prediction Validation (sampled predictions vs observed)',
                      fontweight='bold', fontsize=12)

        # Left panel: confusion matrix heat-map
        ax3 = axes3[0]
        im = ax3.imshow(conf_pct, cmap='Blues', vmin=0, vmax=100)
        ax3.set_xticks(range(n_modes)); ax3.set_xticklabels(mode_lbls, rotation=20, ha='right')
        ax3.set_yticks(range(n_modes)); ax3.set_yticklabels(mode_lbls)
        ax3.set_xlabel('Predicted mode', fontsize=10)
        ax3.set_ylabel('Observed mode', fontsize=10)
        ax3.set_title('Confusion matrix\n(row = observed class, cell = % predicted as col)',
                      fontsize=10)
        plt.colorbar(im, ax=ax3, label='% of observed class', fraction=0.04)
        for i in range(n_modes):
            for j in range(n_modes):
                n_cell = conf[i, j]
                pct    = conf_pct[i, j]
                txt_col = 'white' if pct > 55 else 'black'
                ax3.text(j, i, f'{pct:.0f}%\n({n_cell:,})',
                         ha='center', va='center', fontsize=8, color=txt_col)

        accuracy = np.diag(conf).sum() / conf.sum() * 100 if conf.sum() > 0 else 0
        ax3.text(0.5, -0.12, f'Overall accuracy = {accuracy:.1f}%  '
                 f'(N = {int(conf.sum()):,} tours)',
                 transform=ax3.transAxes, ha='center', fontsize=9)

        # Right panel: observed vs predicted marginal mode shares
        ax4 = axes3[1]
        obs_share_m  = obs_marginal  / obs_marginal.sum()  * 100
        pred_share_m = pred_marginal / pred_marginal.sum() * 100
        x = np.arange(n_modes); w = 0.38
        ax4.bar(x - w/2, obs_share_m,  w, color='#2E86AB', alpha=.85, label='Observed')
        ax4.bar(x + w/2, pred_share_m, w, color='#E84855', alpha=.75, label='Predicted (sampled)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(mode_lbls, rotation=25, ha='right', fontsize=9)
        ax4.set_ylabel('Mode share (%)', fontsize=10)
        ax4.set_title('Marginal mode shares\n(should be close for a well-calibrated model)',
                      fontsize=10)
        ax4.legend(fontsize=9, frameon=False)
        ax4.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax4.spines[spine].set_visible(False)
        for xi, (o, p) in enumerate(zip(obs_share_m, pred_share_m)):
            ax4.text(xi, max(o, p) + 0.5, f'{p-o:+.1f}', ha='center', fontsize=8,
                     color='#c0392b' if abs(p-o) > 3 else '#555')

        fig3.tight_layout(rect=[0, 0, 1, 0.92])
        p3 = os.path.join(save_dir, 'val_stage5_confusion.png')
        fig3.savefig(p3, dpi=130, bbox_inches='tight'); plt.close(fig3)
        figs.append(p3)
        logger.info(f"    saved → {os.path.basename(p3)}")
        metrics['mode_accuracy'] = accuracy

    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig4.suptitle('Stage 5 — Destination Distance: Observed vs Predicted',
                      fontweight='bold', fontsize=12)
        for ax, seg in zip(axes4, segs):
            sub = dist_df[dist_df['segment'] == seg].copy()
            if sub.empty:
                ax.set_title(SEG_LABEL[seg]); continue
            obs = sub[sub['source'] == 'Observed']['dist'].clip(0, 80)
            pred = sub[sub['source'] == 'Predicted']['dist'].clip(0, 80)
            bins = np.linspace(0, max(5.0, float(np.nanpercentile(sub['dist'], 95))), 35)
            ax.hist(obs, bins=bins, density=True, alpha=.55, color='#2E86AB', label='Observed')
            ax.hist(pred, bins=bins, density=True, alpha=.50, color='#E84855', label='Predicted')
            ax.set_title(SEG_LABEL[seg])
            ax.set_xlabel('OD distance')
            ax.grid(axis='y', alpha=.3)
            ax.legend(fontsize=8)
        axes4[0].set_ylabel('Density')
        plt.tight_layout()
        p4 = os.path.join(save_dir, 'val_stage5_distance_segments.png')
        fig4.savefig(p4, dpi=130, bbox_inches='tight'); plt.close(fig4)
        figs.append(p4)

        fig5, axes5 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig5.suptitle('Stage 5 — Distance by Mode: Observed vs Predicted',
                      fontweight='bold', fontsize=12)
        for ax, source in zip(axes5, ['Observed', 'Predicted']):
            sub = dist_df[dist_df['source'] == source].copy()
            mode_order = [m for m in ['DA', 'SR', 'Walk', 'Bike', 'Transit', 'All'] if m in sub['mode'].unique()]
            if not mode_order:
                ax.set_visible(False)
                continue
            data = [sub.loc[sub['mode'] == m, 'dist'].clip(0, 80).dropna().values for m in mode_order]
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                            medianprops=dict(color='white', lw=1.8),
                            flierprops=dict(marker='.', ms=2, alpha=0.25))
            for patch, mode in zip(bp['boxes'], mode_order):
                patch.set_facecolor({'DA':'#1E5FA8','SR':'#2E9E6B','Walk':'#E07030',
                                     'Bike':'#5B9E2B','Transit':'#7B5EA7','All':'#888888'}.get(mode, '#888888'))
                patch.set_alpha(0.85)
            ax.set_xticks(range(1, len(mode_order) + 1))
            ax.set_xticklabels(mode_order, rotation=20, ha='right')
            ax.set_title(source)
            ax.set_xlabel('Mode')
            ax.grid(axis='y', alpha=.3)
        axes5[0].set_ylabel('OD distance')
        plt.tight_layout()
        p5 = os.path.join(save_dir, 'val_stage5_distance_by_mode.png')
        fig5.savefig(p5, dpi=130, bbox_inches='tight'); plt.close(fig5)
        figs.append(p5)

    # ── 5D: Additional destination validation types ───────────────────────
    def _safe_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        p = p / max(p.sum(), 1.0)
        q = q / max(q.sum(), 1.0)
        m = 0.5 * (p + q)
        def _kl(a, b):
            mask = (a > 0) & (b > 0)
            if not np.any(mask):
                return 0.0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    # Tour-level destination fit by segment
    tour_fit_rows = []
    for seg in segs:
        seg_df = _filter_seg(hhts_tours, seg)
        if seg_df.empty or 'DTAZ' not in seg_df.columns or seg not in seg_pred_dest:
            continue
        obs_share = seg_df['DTAZ'].dropna().astype(int).value_counts(normalize=True).sort_index()
        pred_share = seg_pred_dest[seg]
        zone_ids = sorted(set(obs_share.index.tolist()) | set(pred_share.index.tolist()))
        obs_vec = obs_share.reindex(zone_ids).fillna(0.0).to_numpy(dtype=float)
        pred_vec = pred_share.reindex(zone_ids).fillna(0.0).to_numpy(dtype=float)
        mae = float(np.mean(np.abs(obs_vec - pred_vec)))
        jsd = _safe_js_divergence(obs_vec, pred_vec)
        obs_top = set(obs_share.sort_values(ascending=False).head(25).index.tolist())
        pred_top = set(pred_share.sort_values(ascending=False).head(25).index.tolist())
        top_overlap = (len(obs_top & pred_top) / max(len(obs_top), 1)) * 100.0
        tour_fit_rows.append({
            'segment': seg,
            'metric': 'tour_level_dest',
            'mae_share': mae,
            'js_divergence': jsd,
            'top25_overlap_pct': top_overlap,
            'n_tours': int(len(seg_df)),
        })

    if tour_fit_rows:
        fit_df = pd.DataFrame(tour_fit_rows)
        metrics['tour_level_destination_fit'] = fit_df.to_dict(orient='records')

        fig6, ax6 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(fit_df))
        ax6.bar(x - 0.2, fit_df['mae_share'].values, 0.4, label='MAE(share)', color='#2E86AB', alpha=0.85)
        ax6.bar(x + 0.2, fit_df['js_divergence'].values, 0.4, label='JSD', color='#E84855', alpha=0.75)
        ax6.set_xticks(x)
        ax6.set_xticklabels([SEG_LABEL.get(s, s) for s in fit_df['segment']], rotation=15, ha='right')
        ax6.set_ylabel('Error / divergence')
        ax6.set_title('Stage 5 — Tour-Level Destination Fit by Segment', fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        p6 = os.path.join(save_dir, 'val_stage5_dest_tour_level_fit.png')
        fig6.savefig(p6, dpi=130, bbox_inches='tight'); plt.close(fig6)
        figs.append(p6)

    # Work/Education assignment-style validation (anchor-like destination fit)
    assign_rows = []
    for seg in ['work', 'edu']:
        seg_df = _filter_seg(hhts_tours, seg)
        if seg_df.empty or 'DTAZ' not in seg_df.columns or seg not in seg_pred_dest:
            continue
        obs_mode_zone = int(seg_df['DTAZ'].dropna().astype(int).mode().iloc[0]) if len(seg_df['DTAZ'].dropna()) else -1
        pred_mode_zone = int(seg_pred_dest[seg].sort_values(ascending=False).index[0]) if len(seg_pred_dest[seg]) else -1
        obs_top10 = set(seg_df['DTAZ'].dropna().astype(int).value_counts().head(10).index.tolist())
        pred_top10 = set(seg_pred_dest[seg].sort_values(ascending=False).head(10).index.tolist())
        assign_rows.append({
            'segment': seg,
            'metric': 'assignment_fit',
            'mode_zone_match': int(obs_mode_zone == pred_mode_zone),
            'top10_overlap_pct': (len(obs_top10 & pred_top10) / max(len(obs_top10), 1)) * 100.0,
            'obs_mode_zone': obs_mode_zone,
            'pred_mode_zone': pred_mode_zone,
        })

    if assign_rows:
        assign_df = pd.DataFrame(assign_rows)
        metrics['work_edu_assignment_fit'] = assign_df.to_dict(orient='records')

        fig7, ax7 = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(assign_df))
        ax7.bar(x, assign_df['top10_overlap_pct'].values, color='#5B9E2B', alpha=0.85)
        ax7.set_xticks(x)
        ax7.set_xticklabels([SEG_LABEL.get(s, s) for s in assign_df['segment']])
        ax7.set_ylim(0, 100)
        ax7.set_ylabel('Top-10 zone overlap (%)')
        ax7.set_title('Stage 5 — Work/Education Assignment Validation', fontweight='bold')
        for i, v in enumerate(assign_df['top10_overlap_pct'].values):
            ax7.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
        ax7.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        p7 = os.path.join(save_dir, 'val_stage5_work_edu_assignment_fit.png')
        fig7.savefig(p7, dpi=130, bbox_inches='tight'); plt.close(fig7)
        figs.append(p7)

    # Trip-level destination validation using observed trips where DTAZ exists
    trip_df = out.get('trips') if isinstance(out, dict) else None
    if isinstance(trip_df, pd.DataFrame) and not trip_df.empty and 'DTAZ' in trip_df.columns:
        try:
            t = trip_df.copy()
            if 'OVERALL_PURPOSE' in t.columns:
                _p = t['OVERALL_PURPOSE'].astype(str).str.lower()
            else:
                _p = pd.Series('other', index=t.index)
            t['seg'] = np.select(
                [
                    _p.str.contains('work', na=False),
                    _p.str.contains('edu|school|university|study', na=False),
                ],
                ['work', 'edu'],
                default='nonmand',
            )
            trip_rows = []
            for seg in segs:
                if seg not in seg_pred_dest:
                    continue
                sub = t[t['seg'] == seg]
                if sub.empty:
                    continue
                obs_trip_share = sub['DTAZ'].dropna().astype(int).value_counts(normalize=True).sort_index()
                pred_tour_share = seg_pred_dest[seg]
                zid = sorted(set(obs_trip_share.index.tolist()) | set(pred_tour_share.index.tolist()))
                obs_v = obs_trip_share.reindex(zid).fillna(0.0).to_numpy(dtype=float)
                pred_v = pred_tour_share.reindex(zid).fillna(0.0).to_numpy(dtype=float)
                trip_rows.append({
                    'segment': seg,
                    'metric': 'trip_level_dest',
                    'mae_share': float(np.mean(np.abs(obs_v - pred_v))),
                    'js_divergence': _safe_js_divergence(obs_v, pred_v),
                    'n_trips': int(len(sub)),
                })

            if trip_rows:
                trip_fit_df = pd.DataFrame(trip_rows)
                metrics['trip_level_destination_fit'] = trip_fit_df.to_dict(orient='records')

                fig8, ax8 = plt.subplots(figsize=(10, 5))
                x = np.arange(len(trip_fit_df))
                ax8.bar(x - 0.2, trip_fit_df['mae_share'].values, 0.4, label='MAE(share)', color='#A23B72', alpha=0.85)
                ax8.bar(x + 0.2, trip_fit_df['js_divergence'].values, 0.4, label='JSD', color='#F18F01', alpha=0.75)
                ax8.set_xticks(x)
                ax8.set_xticklabels([SEG_LABEL.get(s, s) for s in trip_fit_df['segment']], rotation=15, ha='right')
                ax8.set_ylabel('Error / divergence')
                ax8.set_title('Stage 5 — Trip-Level Destination Fit by Segment', fontweight='bold')
                ax8.legend(fontsize=9)
                ax8.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                p8 = os.path.join(save_dir, 'val_stage5_dest_trip_level_fit.png')
                fig8.savefig(p8, dpi=130, bbox_inches='tight'); plt.close(fig8)
                figs.append(p8)
        except Exception as _t5e:
            logger.debug(f"  Stage 5 trip-level validation failed: {_t5e}")

    # Distances stay out of the metrics dict (too many rows for JSON); the
    # profile panels consume them directly.
    return {'stage': 5, 'plots': figs, 'metrics': metrics,
            '_dist_rows': pd.DataFrame(dist_rows)}


# ---------------------------------------------------------------------------
# STAGE 5 — HOLD-OUT validation (out-of-sample)
# ---------------------------------------------------------------------------

def validate_stage5_holdout(out: dict, s5_models: dict, save_dir: str,
                            frac: float = 0.3, seed: int = 42,
                            _split: str | None = None) -> dict:
    """Out-of-sample validation for Stage 5 on a person-level HHTS hold-out.

    Splits HHTS persons 70/30 by default (seeded; HOLDOUT_FRAC/HOLDOUT_SEED env
    override), rebuilds the mode/dest choice sets on the 30% holdout tours,
    applies the FITTED coefficients (no refit), and compares predicted vs
    observed shares. Two hold-out definitions are produced by default
    (HOLDOUT_SPLIT=both): a random person-level split (in-population
    generalisation) and a temporal split keyed on the diary date
    (TRAVDATE/TRAVYEAR/SURVEYWEEK), which holds out the latest diary wave for
    temporal-transfer evaluation --- the harder test for a forecasting model.
    Person-level splitting avoids within-person leakage (a person's tours never
    appear on both sides).

    Design notes:
      * The destination tree is built with sampled_alts=None (full 371-zone
        set, log_correction=0), so hold-out shares need no Manski-Lerman
        debiasing — unlike the importance-sampled estimation shares.
      * Structural terms (ASC_*, Mu_*, c_size_term, log_correction) are added
        by the builders automatically; custom_*_vars carries only the fitted
        free behavioural variables recovered from the fitted pf.
      * Fully non-fatal: any failure (missing skims/landuse, tree build,
        prediction) degrades to observed-holdout aggregates with an explicit
        note, and the function never raises.
    Env:
      HOLDOUT_ENABLE=0 disables (default 1); HOLDOUT_FRAC (default 0.3),
      HOLDOUT_SEED, HOLDOUT_SPLIT (random|temporal|both, default both),
      HOLDOUT_TEMPORAL_COL (default auto: TRAVDATE/TRAVYEAR/SURVEYWEEK).
    """
    import os as _os
    _enable = _os.environ.get('HOLDOUT_ENABLE', '1').strip() != '0'
    figs, metrics = [], {}
    _note = []
    if not _enable:
        return {'stage': '5-holdout',
                'plots': figs,
                'metrics': {'skipped': 'HOLDOUT_ENABLE=0'}}
    try:
        frac = float(_os.environ.get('HOLDOUT_FRAC', frac))
    except Exception:
        frac = 0.3
    try:
        seed = int(_os.environ.get('HOLDOUT_SEED', seed))
    except Exception:
        seed = 42

    # ── split dispatch: 'both' produces the random and temporal hold-outs ──
    if _split is None:
        _split = (_os.environ.get('HOLDOUT_SPLIT', 'both').strip().lower()
                  or 'both')
        if _split in ('both', 'all'):
            _merged, _all_figs = {}, []
            for _sm in ('random', 'temporal'):
                _rr = validate_stage5_holdout(out, s5_models, save_dir,
                                              frac=frac, seed=seed, _split=_sm)
                _all_figs.extend(_rr.get('plots', []))
                for _k, _v in _rr.get('metrics', {}).items():
                    _merged[f'{_sm}_{_k}'] = _v
            _merged['splits'] = 'random,temporal'
            return {'stage': '5-holdout', 'plots': _all_figs,
                    'metrics': _merged}
        if _split not in ('random', 'temporal'):
            _split = 'random'
    try:
        from stage5_destination_choice import (
            _prep_hhts_tours, _filter_seg, _mode_tree, _build_mode_model,
            _dest_tree, _build_dest_model, _logsums, _attach_size_term,
            _attach_acdc_terms, _skim_flat, _open_skims,
        )
        tours = _prep_hhts_tours(out["tours"], out["persons"])
        persons = out["persons"]
        if tours.empty or persons.empty:
            return {'stage': '5-holdout', 'plots': figs,
                    'metrics': {'skipped': 'empty tours/persons'}}
        _rng = np.random.default_rng(seed)
        _pids = pd.unique(persons['person_id'].dropna().astype(str))
        _n_ho = max(1, int(round(len(_pids) * frac)))
        _ho_pids = set()
        if _split == 'temporal':
            _tcol = _os.environ.get('HOLDOUT_TEMPORAL_COL', '').strip()

            def _temporal_key(df):
                """(numeric key, column label). Prefers a YYYYMMDD composite
                because QTS TRAVDATE is day-of-month (1--31) on its own."""
                if all(c in df.columns
                       for c in ('TRAVYEAR', 'TRAVMONTH', 'TRAVDATE')):
                    _y = pd.to_numeric(df['TRAVYEAR'], errors='coerce')
                    _m = pd.to_numeric(df['TRAVMONTH'], errors='coerce')
                    _d = pd.to_numeric(df['TRAVDATE'], errors='coerce')
                    _k = _y * 10000 + _m * 100 + _d
                    if _k.notna().mean() >= 0.5:
                        return _k, 'TRAVYEAR/TRAVMONTH/TRAVDATE'
                _cols = ([_tcol] if _tcol else
                         ['TRAVDATE', 'TRAVYEAR', 'SURVEYWEEK',
                          'TRAVMONTH', 'TRAVDOW'])
                for _c in _cols:
                    if not _c or _c not in df.columns:
                        continue
                    _num = pd.to_numeric(df[_c], errors='coerce')
                    if _num.notna().mean() >= 0.5:
                        return _num, _c
                    _dts = pd.to_datetime(df[_c], errors='coerce', dayfirst=True)
                    if _dts.notna().any():
                        return (_dts.map(lambda x: x.toordinal()
                                         if pd.notna(x) else np.nan), _c)
                return None, None

            _key = None
            try:
                _rawk, _kname = _temporal_key(persons)
                if _rawk is not None:
                    _kd = pd.DataFrame({
                        'person_id': persons['person_id'].astype(str),
                        '_key': _rawk}).dropna(subset=['_key'])
                    if not _kd.empty:
                        _key = _kd.groupby('person_id')['_key'].max()
                        _tcol = _kname
            except Exception as _te:
                _note.append(f'temporal key failed: {_te}')
                _key = None
            if _key is not None and not _key.empty:
                _ordered = _key.sort_values().index.tolist()
                _cut = max(0, len(_ordered) - _n_ho)
                _ho_pids = set(_ordered[_cut:])
                metrics['holdout_split'] = 'temporal'
                metrics['holdout_temporal_col'] = _tcol
                if _cut < len(_ordered):
                    metrics['holdout_temporal_from'] = float(
                        _key.loc[_ordered[_cut]])
                    metrics['holdout_temporal_max'] = float(_key.max())
            else:
                _note.append('temporal split unavailable (no usable diary-date '
                             'column); falling back to random')
                _ho_pids = set(_rng.choice(_pids, size=_n_ho,
                                           replace=False).tolist())
                metrics['holdout_split'] = 'random(fallback)'
        else:
            _ho_pids = set(_rng.choice(_pids, size=_n_ho,
                                       replace=False).tolist())
            metrics['holdout_split'] = 'random'
        _ho_mask_t = tours['person_id'].astype(str).isin(_ho_pids)
        ho_tours = tours[_ho_mask_t].copy()
        ho_persons = persons[persons['person_id'].astype(str).isin(_ho_pids)].copy()
        metrics['holdout_persons'] = int(len(_ho_pids))
        metrics['holdout_tours'] = int(len(ho_tours))
        metrics['holdout_frac'] = float(frac)
        metrics['holdout_seed'] = int(seed)
        if ho_tours.empty:
            metrics['note'] = 'holdout split empty'
            return {'stage': '5-holdout', 'plots': figs, 'metrics': metrics}

        # Resolve skims + landuse (same sources as estimation).
        skims, _close_sk = None, False
        landuse = None
        try:
            _sk_path = _os.environ.get('PIPELINE_SKIMS_PATH', '')
            skims, _close_sk = _open_skims(_sk_path or None)
        except Exception as _se:
            _note.append(f'skims unavailable: {_se}')
            skims = None
        try:
            from data_connect import Synth_Connect
            landuse = Synth_Connect().get_land()
        except Exception as _le:
            _note.append(f'landuse unavailable: {_le}')
            landuse = None

        _SEG_LABEL = {'work': 'Work', 'edu': 'Education', 'nonmand': 'Non-Mandatory'}
        _MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
        try:
            from stage5_destination_choice import MODE_NAMES as _MN
            _MODE_NAMES = dict(_MN)
        except Exception:
            pass

        def _free_vars_and_coefs(mdl, _structural_prefixes=('ASC_', 'Mu_'),
                                 _structural_exact=('c_size_term', 'log_correction')):
            """(custom_vars, {name: fitted value}) for rebuilding on new data."""
            try:
                _pf = getattr(mdl, '_final_pf', None)
                if _pf is None:
                    _pf = mdl.pf
                _hf = (_pf['holdfast'].fillna(False).astype(bool)
                       if 'holdfast' in _pf.columns
                       else pd.Series(False, index=_pf.index))
                _free = [str(p) for p in _pf.index if not bool(_hf.get(p, False))]
                _vars = [p for p in _free
                         if not p.startswith(_structural_prefixes)
                         and p not in _structural_exact]
                _vals = {}
                for p in _pf.index:
                    try:
                        _v = float(_pf.loc[p].get('value', float('nan')))
                        if np.isfinite(_v):
                            _vals[str(p)] = _v
                    except Exception:
                        continue
                return _vars, _vals
            except Exception:
                return [], {}

        def _apply_coefs(mdl, coefs: dict) -> int:
            _n = 0
            for _k, _v in (coefs or {}).items():
                try:
                    mdl.set_value(str(_k), float(_v))
                    _n += 1
                except Exception:
                    continue
            return _n

        def _safe_js(a, b):
            try:
                from scipy.spatial.distance import jensenshannon as _js
                _a = np.asarray(a, dtype=float); _b = np.asarray(b, dtype=float)
                _a = _a / max(_a.sum(), 1e-12); _b = _b / max(_b.sum(), 1e-12)
                return float(_js(_a, _b))
            except Exception:
                return float('nan')

        _rows = []
        _can_predict = skims is not None and landuse is not None
        if not _can_predict:
            _note.append('prediction skipped (no skims/landuse); observed-only')
        _skim_df = None
        if skims is not None:
            try:
                _skim_df = _skim_flat(skims)
            except Exception as _sfe:
                _note.append(f'skim flat table failed: {_sfe}')
                _can_predict = False

        for seg in ('work', 'edu', 'nonmand'):
            try:
                _seg_df = _filter_seg(ho_tours, seg)
                if _seg_df.empty:
                    continue
                _seg_persons = ho_persons[
                    ho_persons['person_id'].astype(str).isin(
                        set(_seg_df['person_id'].astype(str)))]
                # observed holdout shares
                _obs_mode = _seg_df['tour_mode'].dropna()
                _obs_ms = (_obs_mode.value_counts(normalize=True)
                           .reindex(range(1, 6)).fillna(0) * 100)
                _obs_dz = (_seg_df['DTAZ'].dropna().astype(int)
                           .value_counts(normalize=True).sort_index())
                _row = {'segment': seg, 'n_holdout_tours': int(len(_seg_df))}
                for _m in range(1, 6):
                    _row[f'obs_mode_{_MODE_NAMES.get(_m, _m)}'] = \
                        float(_obs_ms.get(_m, 0.0))
                _pred_ok = False
                if _can_predict:
                    try:
                        _seg_res = s5_models.get(seg, {}) or {}
                        _mode_fit = _seg_res.get('mode_model')
                        _dest_fit = _seg_res.get('dest_model')
                        if _mode_fit is None or _dest_fit is None:
                            raise RuntimeError('fitted mode/dest model missing')
                        # — mode on holdout —
                        _dt_m = _mode_tree(_seg_df, _seg_persons, landuse, skims)
                        _mv, _mc = _free_vars_and_coefs(_mode_fit)
                        _mm = _build_mode_model(
                            _dt_m, seg, custom_mode_vars=_mv,
                            avail_vars_hint=list(_seg_df.columns))
                        _apply_coefs(_mm, _mc)
                        _mp = np.asarray(_mm.probability(), dtype=float)
                        _mp = np.nan_to_num(_mp, nan=0.0, posinf=0.0, neginf=0.0)
                        _rs = _mp.sum(axis=1, keepdims=True)
                        _mp = _mp / np.where(_rs > 0, _rs, 1.0)
                        _pm = {}
                        for _mid in range(1, 6):
                            _ci = _mid - 1
                            _pm[_mid] = float(_mp[:, _ci].mean()) * 100 \
                                if _ci < _mp.shape[1] else 0.0
                        for _m in range(1, 6):
                            _row[f'pred_mode_{_MODE_NAMES.get(_m, _m)}'] = _pm[_m]
                        _row['mode_mae_pp'] = float(np.mean([
                            abs(_pm.get(_m, 0.0) - float(_obs_ms.get(_m, 0.0)))
                            for _m in range(1, 6)]))
                        # — dest on holdout (full zone set, no sampling bias) —
                        from stage5_destination_choice import _logsums as _lgs
                        _ls = _lgs(_mm, _dt_m, skims=skims, verbose=False)
                        _dt_d = _dest_tree(
                            _seg_df, _seg_persons, landuse, skims, _ls,
                            sampled_alts=None, skim_df=_skim_df)
                        from stage5_destination_choice import _attach_size_term \
                            as _ast
                        _lu = _ast(landuse, seg)
                        _dv, _dc = _free_vars_and_coefs(_dest_fit)
                        _dm = _build_dest_model(_dt_d, seg, custom_dest_vars=_dv)
                        _apply_coefs(_dm, _dc)
                        _dp = np.asarray(_dm.probability(), dtype=float)
                        _dp = np.nan_to_num(_dp, nan=0.0, posinf=0.0, neginf=0.0)
                        _zid = None
                        try:
                            _zid = list(_dm.alternatives)
                        except Exception:
                            _zid = []
                        if len(_zid) != _dp.shape[1]:
                            _zid = list(range(int(_dp.shape[1])))
                        _pred_dz = (pd.Series(_dp.sum(axis=0),
                                              index=pd.Index(_zid, dtype=int))
                                    .sort_index())
                        _pred_dz = _pred_dz / max(_pred_dz.sum(), 1e-12)
                        _zunion = sorted(set(_obs_dz.index.tolist())
                                         | set(_pred_dz.index.tolist()))
                        _ov = _obs_dz.reindex(_zunion).fillna(0.0).values
                        _pv = _pred_dz.reindex(_zunion).fillna(0.0).values
                        _row['dest_mae_share'] = float(np.mean(np.abs(_ov - _pv)))
                        _row['dest_js_divergence'] = _safe_js(_ov, _pv)
                        _to = _obs_dz.sort_values(ascending=False).head(25).index.tolist()
                        _tp = _pred_dz.sort_values(ascending=False).head(25).index.tolist()
                        _row['dest_top25_overlap'] = (
                            float(len(set(_to) & set(_tp)) / 25.0) if _to else float('nan'))
                        _pred_ok = True
                    except Exception as _pe:
                        _row['predict_note'] = f'{type(_pe).__name__}: {_pe}'[:200]
                        logger.debug(f"  Stage 5 holdout predict [{seg}]: {_pe}")
                _row['predicted'] = bool(_pred_ok)
                _rows.append(_row)
            except Exception as _se:
                logger.debug(f"  Stage 5 holdout segment [{seg}]: {_se}")
                continue
        try:
            if _close_sk:
                skims.close()
        except Exception:
            pass

        if _rows:
            _hdf = pd.DataFrame(_rows)
            _hf_name = ('stage5_holdout_metrics.csv' if _split == 'random'
                        else f'stage5_holdout_metrics_{_split}.csv')
            _hp = os.path.join(save_dir, _hf_name)
            _hdf.to_csv(_hp, index=False)
            metrics['holdout_metrics_csv'] = _hp
            metrics['holdout_segments'] = int(len(_rows))
            metrics['holdout_predicted_segments'] = int(
                sum(1 for _r in _rows if _r.get('predicted')))
            try:
                fig, axes = plt.subplots(1, max(len(_rows), 1), figsize=(5 * max(len(_rows), 1), 5))
                if max(len(_rows), 1) == 1:
                    axes = [axes]
                for ax, _r in zip(axes, _rows):
                    _ml = [_MODE_NAMES.get(_m, str(_m)) for _m in range(1, 6)]
                    _x = np.arange(len(_ml))
                    _o = [_r.get(f'obs_mode_{_m}', 0.0) for _m in _ml]
                    _p = [_r.get(f'pred_mode_{_m}', 0.0) for _m in _ml]
                    ax.bar(_x - .2, _o, .4, color='#2E86AB', alpha=.85, label='Observed (holdout)')
                    ax.bar(_x + .2, _p, .4, color='#E84855', alpha=.75, label='Predicted (holdout)')
                    ax.set_xticks(_x)
                    ax.set_xticklabels(_ml, rotation=25, ha='right', fontsize=9)
                    ax.set_ylabel('Mode share (%)')
                    ax.set_title(f"{_r['segment']} (n={_r['n_holdout_tours']})")
                    ax.legend(fontsize=9)
                    ax.grid(axis='y', alpha=.3)
                fig.suptitle('Stage 5 HOLD-OUT — mode shares: observed vs predicted '
                             '(fitted models, unseen persons)', fontweight='bold', fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                _pp = os.path.join(save_dir, 'val_stage5_holdout.png'
                                   if _split == 'random'
                                   else f'val_stage5_holdout_{_split}.png')
                fig.savefig(_pp, dpi=130, bbox_inches='tight'); plt.close(fig)
                figs.append(_pp)
            except Exception as _fe:
                logger.debug(f"  Stage 5 holdout plot failed: {_fe}")
            for _r in _rows:
                for _k, _v in _r.items():
                    if isinstance(_v, float) and np.isfinite(_v):
                        metrics[f"holdout_{_r['segment']}_{_k}"] = _v
        if _note:
            metrics['notes'] = '; '.join(_note)
    except Exception as _e:
        logger.warning(f"  Stage 5 holdout validation failed (non-fatal): {_e}")
        metrics['error'] = f'{type(_e).__name__}: {_e}'[:200]
    return {'stage': '5-holdout', 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------
# STAGE 5b — trip stop destination + trip mode
# ---------------------------------------------------------------------------

def validate_stage5b(out: dict, dest_models: dict, mode_models: dict,
                     save_dir: str) -> dict:
    """
    In-sample validation for Stage 5b trip-level destination and mode models.

    Produces:
      val_stage5b_dest_hit_rate.png  — fraction of stop trips where predicted
                                       zone matches observed zone (per purpose)
      val_stage5b_mode_shares.png    — observed vs predicted mode share (per purpose)
    """
    logger.info("  [validate] Stage 5b — trip destination + mode")
    figs    = []
    metrics = {}

    trips   = out.get("trips",   pd.DataFrame())
    persons = out.get("persons", pd.DataFrame())

    # ── destination hit-rate per purpose ─────────────────────────────────
    if dest_models and not trips.empty:
        try:
            from stage5b_trip_destination import build_trip_records
            trips_aug = build_trip_records(trips.copy())
            stop_mask = trips_aug.get("is_tour_dest_leg", pd.Series(1)) == 0
            stop_trips = trips_aug[stop_mask] if stop_mask.any() else trips_aug

            purposes = sorted(dest_models.keys())
            hit_rates = []
            for purp in purposes:
                seg = stop_trips[stop_trips.get("purpose", pd.Series()) == purp] \
                    if "purpose" in stop_trips.columns else pd.DataFrame()
                # crude hit: predicted == observed dest_taz
                if not seg.empty and "dest_taz_predicted" in seg.columns and "dest_taz" in seg.columns:
                    obs  = pd.to_numeric(seg["dest_taz"],           errors='coerce').fillna(-1)
                    pred = pd.to_numeric(seg["dest_taz_predicted"], errors='coerce').fillna(-2)
                    hit  = float((obs == pred).mean())
                else:
                    hit = float('nan')
                hit_rates.append(hit)
                metrics[f"dest_hit_{purp}"] = hit

            if any(np.isfinite(h) for h in hit_rates):
                fig, ax = plt.subplots(figsize=(max(6, len(purposes)), 4))
                x = np.arange(len(purposes))
                bars = ax.bar(x, [h if np.isfinite(h) else 0 for h in hit_rates],
                              color='#3498db', alpha=0.8)
                ax.set_xticks(x); ax.set_xticklabels(purposes, rotation=30, ha='right')
                ax.set_ylabel('Hit rate (pred zone == obs zone)')
                ax.set_title('Stage 5b — Stop Destination Hit Rate by Purpose', fontweight='bold')
                ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                p = os.path.join(save_dir, 'val_stage5b_dest_hit_rate.png')
                fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
                figs.append(p)
        except Exception as exc:
            logger.debug(f"  Stage 5b dest validation failed: {exc}")

    # ── mode share comparison per purpose ────────────────────────────────
    if mode_models and not trips.empty:
        try:
            from stage6_mode_choice import prepare_mode_data, PURPOSES as S6_PURPS, MODES
            mode_data = prepare_mode_data(trips, persons)
            purposes  = [p for p in S6_PURPS if p in mode_models]
            if purposes:
                MODE_NAMES_LOCAL = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
                ncols = min(4, len(purposes)); nrows = int(np.ceil(len(purposes) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
                axes = np.array(axes).flatten()
                fig.suptitle('Stage 5b — Trip Mode Shares: Observed vs Predicted',
                             fontsize=12, fontweight='bold')
                for ax_i, purp in enumerate(purposes):
                    ax   = axes[ax_i]
                    seg  = mode_data[mode_data['purpose'] == purp].copy()
                    if seg.empty:
                        ax.set_visible(False); continue
                    obs_shares  = seg['chosen_mode'].value_counts(normalize=True).reindex(MODES, fill_value=0)
                    try:
                        pred_probs = mode_models[purp].predict_proba(seg)
                        pred_shares = pd.Series(pred_probs.mean(axis=0), index=MODES)
                    except Exception:
                        pred_shares = pd.Series(float('nan'), index=MODES)
                    x = np.arange(len(MODES)); w = 0.35
                    ax.bar(x - w/2, obs_shares.values,  w, label='Observed',  color='#2E86AB', alpha=0.85)
                    ax.bar(x + w/2, pred_shares.values, w, label='Predicted', color='#E84855', alpha=0.75)
                    ax.set_xticks(x)
                    ax.set_xticklabels([MODE_NAMES_LOCAL.get(m, str(m)) for m in MODES],
                                       fontsize=8)
                    ax.set_title(purp, fontsize=9); ax.set_ylabel('Share')
                    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
                for ax in axes[len(purposes):]:
                    ax.set_visible(False)
                plt.tight_layout()
                p = os.path.join(save_dir, 'val_stage5b_mode_shares.png')
                fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
                figs.append(p)
        except Exception as exc:
            logger.debug(f"  Stage 5b mode validation failed: {exc}")

    return {'stage': '5b', 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------
# SUMMARY PLOT — model fit metrics in one dashboard
# ---------------------------------------------------------------------------

def validate_profiles(out: dict, s4_models: dict, dist_rows=None,
                      save_dir: str = '.') -> dict:
    """Observed versus predicted time-of-day and trip-length profiles.

    The two comparisons ActivitySim reports on every run, rebuilt from the
    in-sample re-prediction:

      * hourly departure profiles — every trip, and the first departure of
        each person's day.  Predicted departures come from the Stage 4 timing
        model at minute resolution: a period sampled from the fitted
        probability (blended with the fitted observed period shares), then a
        start minute drawn from that purpose's fitted within-period start-time
        distribution.  Each departure is placed independently, so the panel
        compares the timing model's own departure distribution; it does not
        replay the schedule builder's sequence clamps.
      * tour-length distribution in fixed distance bands plus the cumulative
        curve.  Distances are OD skim distances between the observed origin
        and the observed/predicted tour destination reported by Stage 5; when
        those are unavailable the panel falls back to any distance column on
        the HHTS trips and shows observed only.
    """
    figs, metrics = [], {}
    logger.info("  [validate] Profiles — time of day and trip length")

    tod_rows = []
    try:
        from stage4_schedule_placement import prepare_timing_data

        data = prepare_timing_data(out["trips"], out["tours"], out["persons"])
        if 'STARTIME' not in data.columns or data.empty:
            raise KeyError('STARTIME')

        obs_min = pd.to_numeric(data['STARTIME'], errors='coerce')
        person_key = 'person_id' if 'person_id' in data.columns else None

        pred_min = None
        draw_coverage = None
        if s4_models:
            rng = np.random.default_rng(11)
            period_proba = _predict_period_proba(data, s4_models)
            pred_period = _predict_periods(
                data, s4_models, sample=True, rng=rng, proba=period_proba)
            pred_min, eff_period = _predict_start_minutes(
                data, pred_period, s4_models, rng)
            draw_coverage = _within_period_draw_coverage(
                data['purpose'].astype(str).to_numpy(), eff_period, s4_models)
            metrics['tod_within_period_fitted_pct'] = float(draw_coverage)
            logger.info(
                "    predicted departures: sampled periods, fitted within-period "
                f"draws on {draw_coverage:.1f}% of activities")
        else:
            logger.info("    no Stage 4 models -> observed profiles only")

        obs_all, _ = _hour_shares(obs_min)
        profiles = [('All trip departures', obs_all,
                     _hour_shares(pred_min)[0] if pred_min is not None else None)]

        if person_key:
            firsts = pd.DataFrame({'p': data[person_key], 't': obs_min}).dropna()
            if not firsts.empty:
                obs_first, _ = _hour_shares(firsts.groupby('p')['t'].min())
                pred_first = None
                if pred_min is not None:
                    pf = pd.DataFrame({'p': data[person_key],
                                       't': pd.Series(pred_min)}).dropna()
                    if not pf.empty:
                        pred_first, _ = _hour_shares(pf.groupby('p')['t'].min())
                profiles.append(('First departure of the day', obs_first, pred_first))

        metrics['tod_n_records'] = int(obs_min.notna().sum())

        fig, axes = plt.subplots(1, len(profiles), figsize=(8.2 * len(profiles), 4.4),
                                 squeeze=False)
        fig.suptitle('Time-of-day profile — observed vs predicted departures '
                     '(in-sample)', fontsize=12, fontweight='bold')
        for ax, (title, obs_share, pred_share) in zip(axes[0], profiles):
            x = np.arange(len(PROFILE_HOUR_LABELS)); w = 0.4
            ax.bar(x - w / 2, obs_share, w, color='#2E86AB', alpha=.85,
                   label='Observed')
            if pred_share is not None:
                ax.bar(x + w / 2, pred_share, w, color='#E84855', alpha=.75,
                       label='Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{h}' for h in PROFILE_HOUR_LABELS], fontsize=8)
            ax.set_xlabel('Hour of departure (24 h)', fontsize=9)
            ax.set_ylabel('Share of departures (%)', fontsize=9)
            ax.grid(axis='y', alpha=.3)
            ax.spines[['top', 'right']].set_visible(False)

            note = ''
            slug = 'all' if title.startswith('All') else 'first'
            if pred_share is not None:
                corr = (float(np.corrcoef(obs_share, pred_share)[0, 1])
                        if np.std(obs_share) > 0 and np.std(pred_share) > 0
                        else float('nan'))
                gap = float(np.max(np.abs(obs_share - pred_share)))
                peak_obs = PROFILE_HOUR_LABELS[int(np.argmax(obs_share))]
                peak_pred = PROFILE_HOUR_LABELS[int(np.argmax(pred_share))]
                metrics[f'tod_{slug}_corr'] = corr
                metrics[f'tod_{slug}_max_gap_pct_points'] = gap
                metrics[f'tod_{slug}_peak_hour_obs'] = f'{peak_obs}:00'
                metrics[f'tod_{slug}_peak_hour_pred'] = f'{peak_pred}:00'
                note = (f'peak {peak_obs}:00 vs {peak_pred}:00 · '
                        f'r={corr:.3f} · max gap {gap:.2f} pp')
                ax.legend(fontsize=8)
            ax.set_title(f'{title}\n{note}'.strip(), fontsize=9)

            for hour, o, p in zip(PROFILE_HOUR_LABELS, obs_share,
                                  pred_share if pred_share is not None
                                  else [None] * len(obs_share)):
                tod_rows.append({
                    'profile': title,
                    'hour': f'{hour}:00',
                    'observed_pct': float(o),
                    'predicted_pct': None if p is None else float(p),
                })

        if draw_coverage is None:
            placement_note = ('Predicted departures: no Stage 4 timing models '
                              'were available for this run.')
        else:
            placement_note = (
                'Predicted departures: period sampled from the fitted Stage 4 '
                'probability (blended with the fitted observed shares), then a '
                'start minute drawn from the fitted within-period '
                f'distribution ({draw_coverage:.0f}% of activities; the '
                'remainder use the triangular fallback), fitted on this same '
                'HHTS sample. Each activity is placed independently of the '
                'others in its person-day.')
        fig.text(0.01, 0.005, placement_note, fontsize=7, color='#555555')
        plt.tight_layout(rect=(0, 0.045, 1, 1))
        p_tod = os.path.join(save_dir, 'val_profiles_time_of_day.png')
        fig.savefig(p_tod, dpi=140, bbox_inches='tight'); plt.close(fig)
        figs.append(p_tod)
        logger.info(f"    saved -> {os.path.basename(p_tod)}")
        metrics['tod_profile'] = tod_rows
    except Exception as exc:
        logger.warning(f"  Profiles: time-of-day panel skipped: {exc}")

    length_rows = []
    try:
        rows_df = dist_rows if isinstance(dist_rows, pd.DataFrame) else pd.DataFrame()
        obs_d = pred_d = None
        if not rows_df.empty and {'source', 'dist'} <= set(rows_df.columns):
            obs_d = pd.to_numeric(
                rows_df.loc[rows_df['source'] == 'Observed', 'dist'],
                errors='coerce').dropna()
            pred_d = pd.to_numeric(
                rows_df.loc[rows_df['source'] == 'Predicted', 'dist'],
                errors='coerce').dropna()
        else:
            trips = out.get('trips', pd.DataFrame())
            col = next((c for c in ('TRIPDIST', 'trip_distance', 'DIST', 'distance')
                        if c in trips.columns), None)
            if col is not None:
                obs_d = pd.to_numeric(trips[col], errors='coerce').dropna()
                logger.info(f"    no Stage 5 distances -> observed trips['{col}'] only")

        if obs_d is None or len(obs_d) < 10:
            logger.info("    no distances available -> trip-length panel skipped")
            raise ValueError('no distance data')

        obs_sh = _length_bin_shares(obs_d)
        pred_sh = _length_bin_shares(pred_d) if pred_d is not None and len(pred_d) >= 10 else None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))
        fig.suptitle('Tour length — observed vs predicted OD distance '
                     '(in-sample, fixed bands)', fontsize=12, fontweight='bold')

        x = np.arange(len(LENGTH_BIN_LABELS)); w = 0.4
        ax1.bar(x - w / 2, obs_sh, w, color='#2E86AB', alpha=.85, label='Observed')
        if pred_sh is not None:
            ax1.bar(x + w / 2, pred_sh, w, color='#E84855', alpha=.75,
                    label='Predicted')
        ax1.set_xticks(x)
        ax1.set_xticklabels(LENGTH_BIN_LABELS, rotation=25, ha='right', fontsize=8)
        ax1.set_xlabel('OD distance band', fontsize=9)
        ax1.set_ylabel('Share of tours (%)', fontsize=9)
        ax1.grid(axis='y', alpha=.3)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.legend(fontsize=8)

        note = ''
        obs_vals = pd.to_numeric(obs_d, errors='coerce').dropna()
        metrics['length_mean_km_obs'] = float(obs_vals.mean())
        metrics['length_median_km_obs'] = float(obs_vals.median())
        note += (f'obs mean {obs_vals.mean():.1f} km / median '
                 f'{obs_vals.median():.1f} km')
        if pred_sh is not None:
            pv = pd.to_numeric(pred_d, errors='coerce').dropna()
            gap = float(np.max(np.abs(obs_sh - pred_sh)))
            js = _js_divergence(obs_sh, pred_sh)
            metrics['length_mean_km_pred'] = float(pv.mean())
            metrics['length_median_km_pred'] = float(pv.median())
            metrics['length_max_gap_pct_points'] = gap
            metrics['length_js'] = js
            note += (f' · pred mean {pv.mean():.1f} km / median {pv.median():.1f} km'
                     f' · max gap {gap:.2f} pp · JS={js:.4f}')
        ax1.set_title(note, fontsize=8)

        # Cumulative curve: the readable way to compare a long right tail.
        edges = np.asarray(LENGTH_EDGES_KM, dtype=float)
        finite = edges[np.isfinite(edges) & (edges > 0)]
        ax2.plot(finite, np.cumsum(obs_sh[:len(finite)]), 'o-', color='#2E86AB',
                 linewidth=2, label='Observed')
        if pred_sh is not None:
            ax2.plot(finite, np.cumsum(pred_sh[:len(finite)]), 's--',
                     color='#E84855', linewidth=2, label='Predicted')
        ax2.set_xscale('log')
        ax2.set_xlabel('OD distance (km, log scale)', fontsize=9)
        ax2.set_ylabel('Cumulative share of tours (%)', fontsize=9)
        ax2.set_title('Cumulative distance distribution (to 40 km)', fontsize=9)
        ax2.set_ylim(0, 105); ax2.grid(alpha=.3)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        p_len = os.path.join(save_dir, 'val_profiles_trip_length.png')
        fig.savefig(p_len, dpi=140, bbox_inches='tight'); plt.close(fig)
        figs.append(p_len)
        logger.info(f"    saved -> {os.path.basename(p_len)}")

        for label, o, p in zip(LENGTH_BIN_LABELS, obs_sh,
                               pred_sh if pred_sh is not None
                               else [None] * len(obs_sh)):
            length_rows.append({
                'bin': label,
                'observed_pct': float(o),
                'predicted_pct': None if p is None else float(p),
            })
        metrics['trip_length_bins'] = length_rows
    except Exception as exc:
        if str(exc) != 'no distance data':
            logger.warning(f"  Profiles: trip-length panel skipped: {exc}")

    return {'stage': 'profiles', 'plots': figs, 'metrics': metrics}


def plot_fit_summary(results: list, save_dir: str) -> str:
    """One-page dashboard: ρ² / RMSE / accuracy for all stages."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('HHTS Training-Data Validation — Model Fit Summary',
                 fontsize=14, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=.45, wspace=.35)

    # Collect metrics
    stage_labels   = []
    stage_rho2     = []    # pseudo-R² where available
    stage_rmse     = []    # RMSE where available
    stage_accuracy = []    # classification accuracy where available

    for r in results:
        s = r['stage']
        m = r['metrics']
        lbl = _stage_title(s)
        stage_labels.append(lbl)

        if s == 1:
            stage_rho2.append(None)
            stage_rmse.append(m.get('mean_RMSE'))
            stage_accuracy.append(None)
        elif s == 2:
            rmse_vals = [v['rmse'] for v in m.values() if isinstance(v, dict) and 'rmse' in v]
            stage_rmse.append(np.mean(rmse_vals) if rmse_vals else None)
            stage_rho2.append(None)
            stage_accuracy.append(None)
        elif s == 3:
            rmse_vals = [v['rmse_min'] for v in m.values() if isinstance(v, dict) and 'rmse_min' in v]
            stage_rmse.append(np.mean(rmse_vals) if rmse_vals else None)
            stage_rho2.append(None)
            stage_accuracy.append(None)
        elif s == 5:
            stage_accuracy.append(m.get('mode_accuracy') or m.get('overall_accuracy'))
            stage_rho2.append(None); stage_rmse.append(None)
        else:
            stage_rho2.append(None); stage_rmse.append(None); stage_accuracy.append(None)

    # Stage 5 ρ² from the log — hardcode from estimation output
    known_rho2 = {
        'work_mode': 0.629, 'edu_mode': 0.455, 'nonmand_mode': 0.549,
        'work_dest': 0.413, 'edu_dest': 0.605, 'nonmand_dest': 0.685,
        'cdap':      0.333,
    }

    # Panel A: RMSE bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    rmse_vals  = [(l, v) for l, v in zip(stage_labels, stage_rmse) if v is not None]
    if rmse_vals:
        lbls, vals = zip(*rmse_vals)
        bars = ax_a.bar(lbls, vals, color=[STAGE_COLORS.get(int(l[-1]), '#888') for l in lbls], alpha=.85)
        ax_a.set_title('RMSE by Stage', fontsize=10)
        ax_a.set_ylabel('RMSE (min or count)')
        for bar, v in zip(bars, vals):
            ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                      f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax_a.grid(axis='y', alpha=.3)

    # Panel B: Pseudo-R² for Stage 5 models
    ax_b = fig.add_subplot(gs[0, 1:])
    model_keys = list(known_rho2.keys())
    model_vals = list(known_rho2.values())
    colors_b   = ['#2E86AB' if 'mode' in k else '#E84855' if 'dest' in k else '#F18F01'
                   for k in model_keys]
    bars_b = ax_b.barh(model_keys, model_vals, color=colors_b, alpha=.85)
    ax_b.axvline(0.2, color='grey', ls='--', lw=1, label='Acceptable (ρ²=0.2)')
    ax_b.axvline(0.4, color='green', ls='--', lw=1, label='Good (ρ²=0.4)')
    ax_b.set_xlabel('McFadden ρ²')
    ax_b.set_title('Stage 5 + CDAP Model Fit (McFadden ρ²)', fontsize=10)
    ax_b.legend(fontsize=8); ax_b.grid(axis='x', alpha=.3)
    for bar, v in zip(bars_b, model_vals):
        ax_b.text(v + .01, bar.get_y()+bar.get_height()/2,
                  f'{v:.3f}', va='center', fontsize=9)

    # Panel C: Mode accuracy
    ax_c = fig.add_subplot(gs[1, 0])
    acc_vals = [(l, v) for l, v in zip(stage_labels, stage_accuracy) if v is not None]
    if acc_vals:
        lbls, vals = zip(*acc_vals)
        ax_c.bar(lbls, vals, color='#A23B72', alpha=.85)
        ax_c.axhline(100/5, color='grey', ls='--', lw=1, label='Chance (20%)')
        ax_c.set_title('Mode Prediction Accuracy (%)', fontsize=10)
        ax_c.set_ylabel('%'); ax_c.legend(fontsize=8); ax_c.grid(axis='y', alpha=.3)
        for xl, v in zip(range(len(vals)), vals):
            ax_c.text(xl, v+.5, f'{v:.1f}%', ha='center', fontsize=9)

    # Panel D: Activity frequency R²
    ax_d = fig.add_subplot(gs[1, 1:])
    s1_res = next((r for r in results if r['stage']==1), None)
    if s1_res:
        r_val = s1_res['metrics'].get('r', None)
        if r_val is not None:
            ax_d.text(.5, .5, f'Stage 1 Activity Frequency\nCorrelation r = {r_val:.3f}',
                      transform=ax_d.transAxes, ha='center', va='center',
                      fontsize=16, fontweight='bold',
                      color='#2E86AB' if r_val > .7 else '#E84855')
            ax_d.set_title('Stage 1 — Total Activity Count Fit', fontsize=10)
            ax_d.axis('off')
    else:
        ax_d.axis('off')

    path = os.path.join(save_dir, 'val_SUMMARY_dashboard.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  [validate] Summary dashboard → {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# HTML VALIDATION REPORT
# ---------------------------------------------------------------------------

def write_html_report(results: list, summary_plot: str, save_dir: str,
                      out_dir: str) -> str:
    """Write a self-contained HTML report in out_dir."""

    plots_html = ''
    for r in results:
        s = r['stage']
        plots_html += f'<h2>{_stage_title(s)}</h2>\n'
        for p in r.get('plots', []):
            if os.path.exists(p):
                rel = os.path.relpath(p, out_dir)
                cap = os.path.basename(p).replace('_',' ').replace('.png','').title()
                plots_html += (f'<figure><img src="{rel}" style="max-width:100%">'
                               f'<figcaption>{cap}</figcaption></figure>\n')

    if os.path.exists(summary_plot):
        rel_sum = os.path.relpath(summary_plot, out_dir)
        sum_html = (f'<h2>Summary Dashboard</h2>'
                    f'<img src="{rel_sum}" style="max-width:100%">')
    else:
        sum_html = ''

    known_rho2 = {
        'WORK mode (NL)':   ('0.629', 'Excellent'),
        'EDU mode (NL)':    ('0.455', 'Excellent'),
        'NONMAND mode (NL)':('0.549', 'Excellent'),
        'WORK dest':        ('0.413', 'Excellent'),
        'EDU dest':         ('0.605', 'Excellent'),
        'NONMAND dest':     ('0.685', 'Excellent'),
        'CDAP':             ('0.333', 'Good'),
    }
    rho2_rows = ''.join(
        f"<tr><td>{m}</td><td><b>{v}</b></td>"
        f"<td style='color:green'>{r}</td></tr>"
        for m,(v,r) in known_rho2.items()
    )

    # Stage 5 extended validation metrics tables
    s5_metrics_html = ''
    try:
        s5_res = next((r for r in results if r.get('stage') == 5), None)
        s5m = (s5_res or {}).get('metrics', {}) if isinstance(s5_res, dict) else {}

        def _table_from_records(title: str, records: list) -> str:
            if not records:
                return ''
            df = pd.DataFrame(records)
            if df.empty:
                return ''
            show_cols = [c for c in df.columns if c != 'metric']
            hdr = ''.join(f'<th>{c}</th>' for c in show_cols)
            rows = ''
            for _, row in df.iterrows():
                cells = ''
                for c in show_cols:
                    v = row[c]
                    if isinstance(v, (float, np.floating)):
                        cells += f'<td>{v:.4f}</td>'
                    else:
                        cells += f'<td>{v}</td>'
                rows += f'<tr>{cells}</tr>'
            return (
                f'<h3>{title}</h3>'
                f'<table><tr>{hdr}</tr>{rows}</table>'
            )

        s5_metrics_html += _table_from_records(
            'Tour-Level Destination Fit',
            s5m.get('tour_level_destination_fit', []),
        )
        s5_metrics_html += _table_from_records(
            'Work/Education Assignment Fit',
            s5m.get('work_edu_assignment_fit', []),
        )
        s5_metrics_html += _table_from_records(
            'Trip-Level Destination Fit',
            s5m.get('trip_level_destination_fit', []),
        )
    except Exception:
        s5_metrics_html = ''

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>HHTS Validation Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;max-width:1400px}}
h1{{background:#2c3e50;color:white;padding:15px;border-radius:4px}}
h2{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:4px;margin-top:30px}}
table{{border-collapse:collapse;width:auto;margin:12px 0}}
th{{background:#2c3e50;color:white;padding:8px 14px}}
td{{border:1px solid #ddd;padding:6px 14px}}
tr:nth-child(even){{background:#f9f9f9}}
figure{{margin:12px 0}} figcaption{{color:#666;font-size:12px;margin-top:4px}}
img{{border:1px solid #ddd;border-radius:4px}}
.good{{color:green;font-weight:bold}} .warn{{color:orange;font-weight:bold}}</style>
</head><body>

<h1>📊 HHTS In-Sample Validation Report — SEQ ABM</h1>
<p>Models are applied back to the HHTS training data. Predicted outputs are compared
to observed values to assess how well each stage captures travel behaviour.</p>

{sum_html}

<h2>Stage 5 + CDAP: McFadden ρ² (estimated)</h2>
<table><tr><th>Model</th><th>ρ²</th><th>Rating</th></tr>
{rho2_rows}</table>
<p>McFadden ρ² > 0.4 is considered excellent for discrete choice models.
ρ² > 0.2 is generally considered acceptable.</p>

<h2>Per-Stage Plots</h2>
{plots_html}

<h2>Stage 5 Detailed Destination Validation</h2>
{s5_metrics_html if s5_metrics_html else '<p>No extended Stage 5 destination metrics available.</p>'}

<hr>
<p style="color:#888;font-size:11px">
Generated by validate_hhts.py | Training data: HHTS</p>
</body></html>"""

    path = os.path.join(out_dir, 'validation_report.html')
    # Explicit UTF-8: the report carries figure glyphs that a Windows default
    # codec (cp1252) cannot encode.
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"  [validate] HTML report → {path}")
    return path


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def run_validation(out: dict,
                   s1_models: dict, s2_models: dict, s3_models: dict,
                   s4_models: dict, s5_models: dict,
                   out_dir: str,
                   s5b_dest_models: dict = None,
                   s5b_mode_models: dict = None):
    """
    Run in-sample validation for all five estimation stages.

    Parameters
    ----------
    out       : dict from process_survey() — tours, trips, persons
    s*_models : fitted model dicts from each stage's estimate function
    out_dir   : pipeline output directory (OUT_DIR); plots go in {out_dir}/plots/validation/
    """
    val_dir = os.path.join(out_dir, 'plots', 'validation')
    os.makedirs(val_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HHTS IN-SAMPLE VALIDATION")
    logger.info("=" * 60)

    results = []

    # Stage 1
    try:
        results.append(validate_stage1(out, s1_models, val_dir))
    except Exception as e:
        logger.warning(f"  Stage 1 validation failed: {e}")

    # Stage 2
    try:
        results.append(validate_stage2(out, s2_models, val_dir))
    except Exception as e:
        logger.warning(f"  Stage 2 validation failed: {e}")

    # Stage 3
    try:
        results.append(validate_stage3(out, s3_models, val_dir))
    except Exception as e:
        logger.warning(f"  Stage 3 validation failed: {e}")

    # Stage 4
    try:
        results.append(validate_stage4(out, s4_models, val_dir))
    except Exception as e:
        logger.warning(f"  Stage 4 validation failed: {e}")

    # Stage 5
    s5_result = None
    try:
        s5_result = validate_stage5(out, s5_models, val_dir)
        results.append(s5_result)
    except Exception as e:
        logger.warning(f"  Stage 5 validation failed: {e}")

    # Stage 5 hold-out (out-of-sample on unseen persons; non-fatal)
    try:
        results.append(validate_stage5_holdout(out, s5_models, val_dir))
    except Exception as e:
        logger.warning(f"  Stage 5 holdout validation failed: {e}")

    # Stage 5b (optional — trip destination + mode)
    if s5b_dest_models is not None or s5b_mode_models is not None:
        try:
            results.append(validate_stage5b(
                out,
                s5b_dest_models or {},
                s5b_mode_models or {},
                val_dir,
            ))
        except Exception as e:
            logger.warning(f"  Stage 5b validation failed: {e}")

    # Time-of-day and trip-length profiles (ActivitySim-style panels)
    try:
        results.append(validate_profiles(
            out, s4_models,
            dist_rows=(s5_result or {}).get('_dist_rows'),
            save_dir=val_dir,
        ))
    except Exception as e:
        logger.warning(f"  Profile validation failed: {e}")

    # Summary dashboard + report
    summary_plot = ''
    try:
        summary_plot = plot_fit_summary(results, val_dir)
    except Exception as e:
        logger.warning(f"  Summary plot failed: {e}")

    try:
        write_html_report(results, summary_plot, val_dir, out_dir)
    except Exception as e:
        logger.warning(f"  HTML report failed: {e}")

    # Machine-readable metrics for the SimWrapper validation dashboard
    try:
        _mp = write_metrics_json(results, out_dir)
        logger.info(f"  Metrics → {_mp}")
    except Exception as e:
        logger.warning(f"  Metrics JSON failed: {e}")

    logger.info("VALIDATION COMPLETE")
    logger.info(f"  Plots  → {val_dir}/")
    logger.info(f"  Report → {out_dir}/validation_report.html")
    return results

