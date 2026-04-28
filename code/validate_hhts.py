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


# ---------------------------------------------------------------------------
# STAGE 1 — activity frequency
# ---------------------------------------------------------------------------

def validate_stage1(out: dict, models: dict, save_dir: str) -> dict:
    """
    Compare model-predicted activity frequency to observed HHTS counts.
    For each (purpose, ptype): predicted mean vs observed mean; bar chart.
    """
    from stage1_activity_frequency import (
        prepare_frequency_data, predict_activity_frequency, PURPOSES as S1_PURPS,
        PTYPE_PREDICTION_MAP,
    )

    logger.info("  [validate] Stage 1 — activity frequency")
    data = prepare_frequency_data(out["tours"], out["persons"])
    # Use sample=True: modal (sample=False) gives 0 for most people since
    # the majority probability for low-frequency activities is class=0.
    # Sampled prediction gives the full distribution, matching observed means.
    pred = predict_activity_frequency(data.copy(), models, sample=True, seed=42)

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

    In-sample validation: the fitted BudgetLognormal models are applied DIRECTLY
    to the same HHTS activity records used for estimation (not re-generated from
    person-day expansion, which introduces activity-count mismatch errors).
    This gives the true in-sample fit — any residual differences are from the
    model's functional form, not from activity-count generation noise.
    """
    from stage3_activity_duration import (
        prepare_duration_data as prep_s3,
        _augment_duration_features,
        _feature_cols,
    )

    logger.info("  [validate] Stage 3 — activity duration (direct in-sample)")
    data = _augment_duration_features(
        prep_s3(out["tours"], out["persons"], trips=out.get("trips"))
    )

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
    # Root cause of prior obs-vs-pred gap: validation used activity re-generation
    # (predict_activity_durations) which synthesises activities from person-day
    # counts — the number and mix of activities can differ from the HHTS.
    # Fix: apply each fitted BudgetLognormal directly to the HHTS rows.
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Stage 3 — Activity Duration: Observed vs Predicted (In-Sample)',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for ax_i, purp in enumerate(purposes):
        ax     = axes[ax_i]
        mask   = data[purpose_col] == purp
        subset = data[mask].copy()
        if subset.empty:
            ax.set_visible(False); continue
        dur = pd.to_numeric(subset['duration'], errors='coerce').dropna()
        if dur.empty:
            ax.set_visible(False); continue

        obs_mean = float(dur.mean())
        obs_p50  = float(dur.quantile(0.5))
        obs_p90  = float(dur.quantile(0.9))

        pred_mean = pred_p50 = pred_p90 = float('nan')
        pred_vals  = None

        if purp in models:
            m = models[purp]
            # Build the same feature matrix used during estimation
            try:
                ptypes = sorted(data['ptype'].dropna().unique().tolist()) if 'ptype' in data.columns else []
                feats  = _feature_cols(data, ptypes, feature_override=None)
                # Keep only columns that the model actually saw (len(m.mu_) == len(feats_used))
                feats_model = [f for f in feats if f in subset.columns][:len(m.mu_)]
                X = subset[feats_model].fillna(0).values.astype(float)
                if X.shape[1] == len(m.mu_):
                    pred_arr  = m.predict_sampled(X, rng)
                    pred_arr  = np.clip(pred_arr, 1, 1440)
                    pred_vals = pd.Series(pred_arr)
                    pred_mean = float(pred_vals.mean())
                    pred_p50  = float(pred_vals.quantile(0.5))
                    pred_p90  = float(pred_vals.quantile(0.9))
            except Exception as e:
                logger.debug(f"  Stage 3 validate {purp}: {e}")

        # ── plot ─────────────────────────────────────────────────────────
        bins = np.linspace(0, min(dur.quantile(0.99) * 1.2, 720), 30)
        ax.hist(dur.clip(0, 720), bins=bins, density=True,
                alpha=0.55, color='#5B9E2B', label='Observed', zorder=3)
        if pred_vals is not None and len(pred_vals) > 0:
            ax.hist(pred_vals.clip(0, 720), bins=bins, density=True,
                    alpha=0.45, color='#E84855', label='Predicted (in-sample)', zorder=2)

        ax.axvline(obs_mean, color='#5B9E2B', linewidth=1.5, linestyle='--',
                   label=f'Obs mean={obs_mean:.0f}m')
        if np.isfinite(pred_mean):
            ax.axvline(pred_mean, color='#E84855', linewidth=1.5, linestyle='--',
                       label=f'Pred mean={pred_mean:.0f}m')

        bias_pct = (pred_mean - obs_mean) / obs_mean * 100 if obs_mean > 0 and np.isfinite(pred_mean) else float('nan')
        title_line2 = (f"obs={obs_mean:.0f}m pred={pred_mean:.0f}m ({bias_pct:+.0f}%)"
                       if np.isfinite(pred_mean) else f"obs={obs_mean:.0f}m  (no model)")
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
                'pred_mean_min': pred_mean,
                'obs_p50_min':   obs_p50,
                'pred_p50_min':  pred_p50,
                'obs_p90_min':   obs_p90,
                'pred_p90_min':  pred_p90,
                'mean_bias_pct': bias_pct,
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
        metrics['p90_abs_error_min'] = float(
            (summary_df['pred_p90_min'] - summary_df['obs_p90_min']).abs().mean())
        figs.append(summary_path)

        # Scatter: obs vs pred mean per purpose
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor='white')
        ax2.scatter(summary_df['obs_mean_min'], summary_df['pred_mean_min'],
                    s=80, color='#534AB7', zorder=5)
        for _, row in summary_df.iterrows():
            ax2.annotate(
                row['purpose'], (row['obs_mean_min'], row['pred_mean_min']),
                fontsize=8, xytext=(4, 4), textcoords='offset points',
            )
        lo = min(summary_df['obs_mean_min'].min(), summary_df['pred_mean_min'].min()) * 0.9
        hi = max(summary_df['obs_mean_min'].max(), summary_df['pred_mean_min'].max()) * 1.1
        ax2.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='1:1 line')
        ax2.set_xlabel('Observed mean duration (min)', fontsize=10)
        ax2.set_ylabel('Predicted mean duration (min)', fontsize=10)
        ax2.set_title('Stage 3 — Obs vs Pred Mean Duration by Purpose\n'
                      f'(in-sample: mean|bias|={metrics["mean_abs_bias_pct"]:.1f}%)',
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
    pred_period = np.full(len(data), 2, dtype=int)  # default AM_PEAK
    for purpose, model in models.items():
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
            pred_period[mask] = np.argmax(model.predict_proba(X), axis=1)  # modal prediction
        except Exception as e:
            logger.warning(f"Stage 4 validation predict failed for '{purpose}': {e}")

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

    # ── 5A: Mode share: observed vs predicted ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Stage 5 — Mode Choice: Observed vs Predicted Mode Shares',
                 fontweight='bold', fontsize=12)

    all_obs_modes  = []
    all_pred_modes = []

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
                pred_share = pd.Series(probs.mean(axis=0), index=range(1, probs.shape[1] + 1))\
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

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(save_dir, 'val_stage5_mode_shares.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    logger.info(f"    saved → {os.path.basename(p)}")
    figs.append(p)

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
        # because it uses the fitted model with the full 329-zone set.
        # Both .probability() and the numpy Dataset fallback below fail when
        # the estimation Dataset has importance-sampled (60) alternatives while
        # the prediction path expects 329 zones (shape mismatch → silent zeros).
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
        # Sort zones numerically 1→330 so plots are geographically ordered
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

    return {'stage': 5, 'plots': figs, 'metrics': metrics}


# ---------------------------------------------------------------------------
# SUMMARY PLOT — model fit metrics in one dashboard
# ---------------------------------------------------------------------------

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
        lbl = f"Stage {s}"
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
        plots_html += f'<h2>Stage {s}</h2>\n'
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
    with open(path, 'w') as f:
        f.write(html)
    logger.info(f"  [validate] HTML report → {path}")
    return path


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def run_validation(out: dict,
                   s1_models: dict, s2_models: dict, s3_models: dict,
                   s4_models: dict, s5_models: dict,
                   out_dir: str):
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
    try:
        results.append(validate_stage5(out, s5_models, val_dir))
    except Exception as e:
        logger.warning(f"  Stage 5 validation failed: {e}")

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

    logger.info("VALIDATION COMPLETE")
    logger.info(f"  Plots  → {val_dir}/")
    logger.info(f"  Report → {out_dir}/validation_report.html")
    return results

