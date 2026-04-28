from __future__ import annotations
"""
plot_activity_plans.py
======================
Visualisation for the SEQ ABM pipeline output plan DataFrame.

Two public functions:
  plot_all(plan, n_sample, save_dir)  — population summary + sample plans
  plot_sample_plans(plan, n, save_dir)— just the individual plan charts

Plan DataFrame expected columns:
  row_type   : 'trip' | 'activity'
  person_id  : person identifier
  purpose    : purpose string ('work', 'edu', 'travel', 'home', ...)
  STARTIME   : minutes from midnight
  ENDTIME    : minutes from midnight
  mode       : int code  0=dwell, 1=DA, 2=SR, 3=Walk, 4=Bike, 5=Transit
  mode_name  : str  ('DA', 'SR', 'Walk', 'Bike', 'Transit', or '' for dwell)
  OTAZ, DTAZ : zone codes
  ptype      : person type string
  vehrd      : vehicles in HH
"""

import os, math as _math_module, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


warnings.filterwarnings('ignore')

# ── Colour palettes ────────────────────────────────────────────────────────
PURPOSE_COLORS = {
    'work':     '#2E86AB',
    'edu':      '#1B998B',
    'social':   '#E84855',
    'escort':   '#9B5C3A',
    'shopping': '#C73E1D',
    'othmaint': '#7B2D8B',
    'othdiscr': '#A9A9A9',
    'eat':      '#3D6B35',
    'home':     '#DDDDDD',
    'travel':   '#CCCCCC',   # trip legs — never shown as dwell
}
MODE_COLORS = {
    'DA':      '#2E86AB',
    'SR':      '#E67E22',
    'Walk':    '#2ECC71',
    'Bike':    '#3498DB',
    'Transit': '#9B59B6',
    '':        '#DDDDDD',    # blank = dwell
}


def _hm(minutes: float) -> str:
    """Convert minutes-from-midnight to HH:MM string."""
    try:
        m = int(round(float(minutes))) % (24 * 60)
        return f"{m//60}:{m%60:02d}"
    except Exception:
        return '?'


def _trips_only(plan: pd.DataFrame) -> pd.DataFrame:
    if 'row_type' in plan.columns:
        return plan[plan['row_type'] == 'trip'].copy()
    return plan.copy()


def _activities_only(plan: pd.DataFrame) -> pd.DataFrame:
    if 'row_type' in plan.columns:
        return plan[plan['row_type'] == 'activity'].copy()
    return plan.copy()


# ---------------------------------------------------------------------------
# FIGURE 1 — Population summary
# ---------------------------------------------------------------------------

def plot_population_summary(plan: pd.DataFrame, save_path: str):
    #FIXME plot summary plan clipps travel time distribution at 90 mins, it shows nothing, oi want to see the total tour day time of activities as a distributions, ttha is my total time across heach perosn,
    trips = _trips_only(plan)
    acts  = _activities_only(plan)
    n_persons = plan['person_id'].nunique() if 'person_id' in plan.columns else len(plan)
    n_acts    = len(acts)

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(f'Activity plan summary  —  {n_persons:,} persons, {n_acts:,} activities',
                 fontsize=14, fontweight='bold', y=0.99)
    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30,
                  height_ratios=[1.4, 1, 1, 1])

    purp_order = ['work','edu','social','escort','shopping','othmaint','othdiscr','eat']
    hours      = list(range(4, 25))   # 4am → midnight

    # ── Panel 1: Departures heatmap (act starts by hour × purpose) ──────────
    ax_heat = fig.add_subplot(gs[0, :])
    heat_data = {}
    for purp in purp_order:
        purp_acts = acts[acts.get('purpose', pd.Series()) == purp] \
                    if 'purpose' in acts.columns else pd.DataFrame()
        if purp_acts.empty or 'STARTIME' not in purp_acts.columns:
            heat_data[purp] = [0] * len(hours)
            continue
        start_hr = (purp_acts['STARTIME'] / 60).clip(4, 24).astype(int)
        counts   = start_hr.value_counts().reindex(hours, fill_value=0)
        heat_data[purp] = counts.values.tolist()

    heat_df = pd.DataFrame(heat_data, index=hours).T.reindex(purp_order).fillna(0)
    max_val = heat_df.values.max() if heat_df.values.max() > 0 else 1

    for row_i, purp in enumerate(purp_order):
        base_color = matplotlib.colors.to_rgb(PURPOSE_COLORS.get(purp, '#888888'))
        for col_i, hr in enumerate(hours):
            val  = heat_df.loc[purp, hr]
            intens = float(val) / max_val
            cell_color = tuple(1.0 - (1.0 - c) * intens for c in base_color)
            ax_heat.add_patch(mpatches.Rectangle(
                (col_i - 0.5, row_i - 0.5), 1, 1,
                facecolor=cell_color, edgecolor='white', linewidth=0.5
            ))
            if val > 0 and intens > 0.15:
                ax_heat.text(col_i, row_i, f'{int(val):,}',
                             ha='center', va='center', fontsize=6.5,
                             color='white' if intens > 0.6 else '#333')

    ax_heat.set_xlim(-0.5, len(hours) - 0.5)
    ax_heat.set_ylim(-0.5, len(purp_order) - 0.5)
    ax_heat.set_xticks(range(len(hours)))
    ax_heat.set_xticklabels([f'{h}{"am" if h<12 else "pm"}' if h != 12 else '12pm'
                              for h in hours], fontsize=8)
    ax_heat.set_yticks(range(len(purp_order)))
    ax_heat.set_yticklabels(purp_order, fontsize=9)
    ax_heat.set_xlabel('Hour of day', fontsize=10)
    ax_heat.set_title('Activity departures by purpose and hour', fontsize=10)

    # legend patches
    legend_handles = [mpatches.Patch(color=PURPOSE_COLORS.get(p,'#888'), label=p)
                      for p in purp_order]
    ax_heat.legend(handles=legend_handles, loc='upper right', ncol=4,
                   fontsize=7, framealpha=0.8)

    # ── Panel 2: Mode share by purpose ───────────────────────────────────────
    ax_mode = fig.add_subplot(gs[1, 0])
    mode_names_ordered = ['DA', 'SR', 'Transit', 'Walk', 'Bike']

    # Build mode share per purpose — only use trips for that purpose,
    # normalise to sum=1 across motorised+active modes (exclude dwell/empty).
    mode_rows = {}
    for purp in purp_order:
        purp_trips = (trips[trips['purpose'].fillna('') == purp]
                      if 'purpose' in trips.columns else pd.DataFrame())
        if purp_trips.empty:
            mode_rows[purp] = {m: 0.0 for m in mode_names_ordered}
            continue

        _INT_MODE_MAP = {1:'DA', 2:'SR', 3:'Walk', 4:'Bike', 5:'Transit'}
        # Try columns in priority order: mode_name → mode → pred_mode
        _mode_raw = None
        for _mc in ('mode_name', 'mode', 'pred_mode'):
            if _mc not in purp_trips.columns:
                continue
            _col = purp_trips[_mc]
            if _col.dtype == object or str(_col.dtype) == 'object':
                _mode_raw = _col.dropna()
            else:
                _mode_raw = _col.map(_INT_MODE_MAP).dropna()
            if len(_mode_raw) > 0:
                break
        if _mode_raw is None or len(_mode_raw) == 0:
            mode_rows[purp] = {m: 0.0 for m in mode_names_ordered}
            continue
        raw = _mode_raw

        # keep only recognised mode names (drops dwell/''/NaN)
        valid = raw[raw.isin(mode_names_ordered)]
        if len(valid) == 0:
            mode_rows[purp] = {m: 0.0 for m in mode_names_ordered}
            continue

        mc = valid.value_counts(normalize=True)
        mode_rows[purp] = {m: float(mc.get(m, 0.0)) for m in mode_names_ordered}

    mode_df = pd.DataFrame(mode_rows).T.reindex(purp_order).fillna(0)

    # Stacked horizontal bars
    lefts = np.zeros(len(purp_order))
    _any_mode_data = False
    for mode in mode_names_ordered:
        vals = mode_df[mode].values if mode in mode_df.columns else np.zeros(len(purp_order))
        if vals.sum() > 0:
            _any_mode_data = True
        bars = ax_mode.barh(purp_order, vals * 100, left=lefts * 100,
                            color=MODE_COLORS.get(mode, '#888'), alpha=0.85,
                            label=mode)
        lefts += vals

    ax_mode.set_xlabel('Share (%)')
    ax_mode.set_xlim(0, 100)
    ax_mode.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax_mode.set_title('Mode share by purpose')
    ax_mode.legend(ncol=3, fontsize=8, loc='lower right')
    ax_mode.grid(axis='x', alpha=0.3)
    if not _any_mode_data:
        _mn_vals = trips['mode_name'].value_counts().head(5).to_dict() if 'mode_name' in trips.columns else {}
        _mode_vals = trips['mode'].value_counts().head(5).to_dict() if 'mode' in trips.columns else {}
        ax_mode.text(50, len(purp_order)/2,
                     f'No mode data\nmode_name uniq: {_mn_vals}\nmode uniq: {_mode_vals}',
                     ha='center', va='center', fontsize=8, color='#c00',
                     transform=ax_mode.transData)

    # ── Panel 3: Distribution of daily activities per person ─────────────────
    ax_dist = fig.add_subplot(gs[1, 1])
    if 'person_id' in acts.columns:
        acts_per = acts.groupby('person_id').size()
        mean_acts = acts_per.mean()
        ax_dist.hist(acts_per.clip(0, 30), bins=30, color='#7B2D8B',
                     alpha=0.75, edgecolor='white', linewidth=0.5)
        ax_dist.axvline(mean_acts, color='#C73E1D', ls='--', lw=2,
                        label=f'Mean {mean_acts:.1f}')
        ax_dist.set_xlabel('Activities per person')
        ax_dist.set_ylabel('Persons')
        ax_dist.set_title('Distribution of daily activities')
        ax_dist.legend(fontsize=9)
        ax_dist.grid(alpha=0.3)

    # ── Panel 4: Average trip travel time by mode (bar chart) ─────────────
    ax_tt = fig.add_subplot(gs[2, 0])
    tt_col = next((c for c in ['travtime', 'tt_avg'] if c in trips.columns), None)
    if tt_col and 'mode_name' in trips.columns:
        tt_raw = pd.to_numeric(trips[tt_col], errors='coerce')
        # Guard against seconds encoding
        if tt_raw.median() > 200:
            tt_raw = tt_raw / 60.0
        trips_tt = trips.copy()
        trips_tt['_tt'] = tt_raw
        # Only include recognised modes, exclude dwell rows
        trips_tt = trips_tt[trips_tt['mode_name'].isin(mode_names_ordered)]
        tt_by_mode = (trips_tt.groupby('mode_name')['_tt']
                      .agg(['mean', 'median', 'count'])
                      .reindex(mode_names_ordered)
                      .dropna(subset=['mean']))
        if not tt_by_mode.empty:
            colors = [MODE_COLORS.get(m, '#888') for m in tt_by_mode.index]
            ax_tt.bar(range(len(tt_by_mode)), tt_by_mode['mean'],
                      color=colors, alpha=0.85, edgecolor='white', label='Mean')
            ax_tt.scatter(range(len(tt_by_mode)), tt_by_mode['median'],
                          color='white', edgecolor='#333', s=50, zorder=5, label='Median')
            ax_tt.set_xticks(range(len(tt_by_mode)))
            ax_tt.set_xticklabels(tt_by_mode.index, fontsize=9)
            ax_tt.set_ylabel('Travel time (min)')
            ax_tt.set_title('Average trip travel time by mode')
            ax_tt.legend(fontsize=8)
            ax_tt.grid(axis='y', alpha=0.3)
            for i, (_, row) in enumerate(tt_by_mode.iterrows()):
                ax_tt.text(i, row['mean'] + 0.3, f'n={int(row["count"]):,}',
                           ha='center', va='bottom', fontsize=7, color='#333')
        else:
            ax_tt.set_title('Average trip travel time by mode')
            ax_tt.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax_tt.transAxes, fontsize=11, color='#999')
    elif tt_col:
        # Fallback: histogram without mode breakdown
        tt_raw = pd.to_numeric(trips[tt_col], errors='coerce').dropna()
        if tt_raw.median() > 200:
            tt_raw = tt_raw / 60.0
        tt = tt_raw.clip(0, 120)
        ax_tt.hist(tt, bins=48, color='#2E86AB', alpha=0.75, edgecolor='white', linewidth=0.4)
        ax_tt.axvline(tt.median(), color='red', ls='--', lw=1.5,
                      label=f'Median {tt.median():.1f} min')
        ax_tt.set_xlabel('Travel time (min)')
        ax_tt.set_ylabel('Trips')
        ax_tt.set_title('Travel time distribution')
        ax_tt.legend(fontsize=9)
    ax_tt.grid(axis='y', alpha=0.3)

    # ── Panel 5: Total daily tour time distribution ────────────────────────
    ax_tour = fig.add_subplot(gs[2, 1])
    if 'person_id' in trips.columns and tt_col:
        # Travel time — always convert to HOURS explicitly.
        # travtime is in minutes (skim output); dividing here keeps units consistent
        # regardless of total daily travel magnitude.
        raw_tt = pd.to_numeric(trips[tt_col], errors='coerce')
        # Detect seconds encoding only (rare): median per-trip > 3600 sec → /60/60
        if raw_tt.median() > 3600:
            raw_tt = raw_tt / 3600.0
        elif raw_tt.median() > 60:
            raw_tt = raw_tt / 60.0   # minutes → hours
        # else already in hours (skim returns hours) — leave as-is
        person_travtime = raw_tt.groupby(trips['person_id']).sum()

        # Add non-home activity dwell times — use predicted_duration_min if
        # available (direct model output, no ENDTIME artefact issues).
        if 'person_id' in acts.columns:
            acts_nh = acts.copy()
            if 'purpose' in acts_nh.columns:
                acts_nh = acts_nh[acts_nh['purpose'].fillna('') != 'home']

            if 'predicted_duration_min' in acts_nh.columns and \
               'STARTIME' in acts_nh.columns and 'ENDTIME' in acts_nh.columns:
                # Prefer ENDTIME - STARTIME (reflects actual scheduled placement);
                # only fall back to predicted_duration_min when ENDTIME is absent
                # or identical to STARTIME (unscheduled placeholder rows).
                raw_dwell = (pd.to_numeric(acts_nh['ENDTIME'], errors='coerce') -
                             pd.to_numeric(acts_nh['STARTIME'], errors='coerce')).clip(0)
                # Detect whether the dwell times actually vary.  If std < 0.1 min
                # (i.e. they're all the same constant), fall back to predicted_duration_min.
                _dwell_std = float(raw_dwell.std()) if len(raw_dwell) > 1 else 0.0
                if _dwell_std < 0.1:
                    dwell_min = pd.to_numeric(
                        acts_nh['predicted_duration_min'], errors='coerce').fillna(0)
                    acts_nh['_dwell'] = (dwell_min / 60.0).clip(0, 12)
                else:
                    if raw_dwell.median() > 3600:
                        acts_nh['_dwell'] = (raw_dwell / 3600.0).clip(0, 12)
                    else:
                        acts_nh['_dwell'] = (raw_dwell / 60.0).clip(0, 10)
            elif 'predicted_duration_min' in acts_nh.columns:
                dwell_min = pd.to_numeric(
                    acts_nh['predicted_duration_min'], errors='coerce').fillna(0)
                acts_nh['_dwell'] = (dwell_min / 60.0).clip(0, 12)
            elif 'STARTIME' in acts_nh.columns and 'ENDTIME' in acts_nh.columns:
                raw_dwell = (pd.to_numeric(acts_nh['ENDTIME'], errors='coerce') -
                             pd.to_numeric(acts_nh['STARTIME'], errors='coerce')).clip(0)
                if raw_dwell.median() > 3600:
                    acts_nh['_dwell'] = (raw_dwell / 3600.0).clip(0, 12)
                else:
                    acts_nh['_dwell'] = (raw_dwell / 60.0).clip(0, 10)
            else:
                acts_nh['_dwell'] = 0.0
                acts_nh['_dwell'] = 0.0

            person_dwell = acts_nh.groupby('person_id')['_dwell'].sum()
            tour_time = person_travtime.add(person_dwell, fill_value=0).clip(0, 18)
        else:
            tour_time = person_travtime.clip(0, 18)
        if len(tour_time) > 5:
            med = tour_time.median()
            mn  = tour_time.mean()
            ax_tour.hist(tour_time, bins=36, color='#3D6B35', alpha=0.75,
                         edgecolor='white', linewidth=0.4, range=(0, 18))
            ax_tour.axvline(med, color='red', ls='--', lw=1.8, label=f'Median {med:.1f}h')
            ax_tour.axvline(mn,  color='orange', ls='--', lw=1.5, label=f'Mean {mn:.1f}h')
            ax_tour.set_xlabel('Total active time out-of-home (hours)')
            ax_tour.set_ylabel('Persons')
            ax_tour.set_title('Daily out-of-home activity + travel time')
            ax_tour.set_xlim(0, 18)
            ax_tour.legend(fontsize=8)
            ax_tour.grid(alpha=0.3)
        else:
            ax_tour.set_title('Total daily tour time distribution')
            ax_tour.text(0.5, 0.5, 'No data', ha='center', va='center',
                         transform=ax_tour.transAxes, fontsize=11, color='#999')
    else:
        ax_tour.set_title('Total daily tour time distribution')
        ax_tour.text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=ax_tour.transAxes, fontsize=11, color='#999')

    # ── Panel 6: Activity start times by purpose (box) ────────────────────
    ax_dep = fig.add_subplot(gs[3, 0])
    dep_data = []
    dep_labels = []
    for purp in purp_order:
        if 'purpose' in acts.columns:
            mask = acts['purpose'].fillna('') == purp
        else:
            mask = pd.Series([True] * len(acts), index=acts.index)
        st = acts.loc[mask, 'STARTIME'].dropna() / 60 if 'STARTIME' in acts.columns else pd.Series()
        if len(st) > 10:
            dep_data.append(st.clip(4, 24).values)
            dep_labels.append(purp)

    if dep_data:
        bp = ax_dep.boxplot(dep_data, vert=False, patch_artist=True, widths=0.6,
                            medianprops={'color':'white','lw':2},
                            whiskerprops={'color':'#555'},
                            capprops={'color':'#555'})
        for patch, purp in zip(bp['boxes'], dep_labels):
            patch.set_facecolor(PURPOSE_COLORS.get(purp, '#888'))
            patch.set_alpha(0.8)
        ax_dep.set_yticks(range(1, len(dep_labels)+1))
        ax_dep.set_yticklabels(dep_labels, fontsize=9)
        ax_dep.set_xlabel('Hour of day')
        ax_dep.set_title('Activity start times by purpose')
        ax_dep.set_xlim(4, 24)
        ax_dep.set_xticks([6,8,10,12,14,16,18,20,22])
        ax_dep.set_xticklabels(['6am','8am','10am','12pm','2pm','4pm','6pm','8pm','10pm'])
        ax_dep.grid(axis='x', alpha=0.3)

    # ── Panel 7: Activity end times by purpose (box) ──────────────────────
    ax_end = fig.add_subplot(gs[3, 1])
    end_data = []
    end_labels = []
    end_col = next((c for c in ['ENDTIME', 'arrtime'] if c in acts.columns), None)
    if end_col:
        for purp in purp_order:
            if 'purpose' in acts.columns:
                mask = acts['purpose'].fillna('') == purp
            else:
                mask = pd.Series([True] * len(acts), index=acts.index)
            et = pd.to_numeric(acts.loc[mask, end_col], errors='coerce').dropna() / 60
            if len(et) > 10:
                end_data.append(et.clip(4, 28).values)   # allow past midnight (28h)
                end_labels.append(purp)
    if end_data:
        bp2 = ax_end.boxplot(end_data, vert=False, patch_artist=True, widths=0.6,
                             medianprops={'color':'white','lw':2},
                             whiskerprops={'color':'#555'},
                             capprops={'color':'#555'})
        for patch, purp in zip(bp2['boxes'], end_labels):
            patch.set_facecolor(PURPOSE_COLORS.get(purp, '#888'))
            patch.set_alpha(0.8)
        ax_end.set_yticks(range(1, len(end_labels)+1))
        ax_end.set_yticklabels(end_labels, fontsize=9)
        ax_end.set_xlabel('Hour of day')
        ax_end.set_title('Activity end times by purpose')
        ax_end.set_xlim(4, 28)
        ax_end.set_xticks([6,8,10,12,14,16,18,20,22,24])
        ax_end.set_xticklabels(['6am','8am','10am','12pm','2pm','4pm','6pm','8pm','10pm','12am'])
        ax_end.grid(axis='x', alpha=0.3)
    else:
        ax_end.set_title('Activity end times by purpose')
        ax_end.text(0.5, 0.5, 'ENDTIME not available', ha='center', va='center',
                    transform=ax_end.transAxes, fontsize=11, color='#999')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Population summary → {save_path}")


# ---------------------------------------------------------------------------
# FIGURE 2 — Sample individual activity plans
# ---------------------------------------------------------------------------

def plot_sample_plans(plan: pd.DataFrame, n: int = 8, save_path: str = None,
                      seed: int = 42):
    """
    Draw n randomly sampled person plans as Gantt charts.

    Each person gets one row per purpose; trip legs are grey arrows,
    dwell bars are coloured by purpose (NOT labelled with mode).
    """
    if 'person_id' not in plan.columns:
        return

    rng = np.random.default_rng(seed)
    pids = plan['person_id'].unique()

    # Sample diverse ptypes if available
    if 'ptype' in plan.columns:
        by_ptype = plan.groupby('ptype')['person_id'].apply(lambda x: x.unique())
        sample_pids = []
        for pt, grp_ids in by_ptype.items():
            k = max(1, n // len(by_ptype))
            sample_pids.extend(rng.choice(grp_ids, size=min(k, len(grp_ids)), replace=False))
        if len(sample_pids) < n:
            extra = rng.choice(pids, size=n - len(sample_pids), replace=False)
            sample_pids.extend(extra)
        sample_pids = list(set(sample_pids))[:n]
    else:
        sample_pids = rng.choice(pids, size=min(n, len(pids)), replace=False).tolist()

    ncols = 2
    nrows = _math_module.ceil(len(sample_pids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*4))
    fig.suptitle(f'Sample activity plans  —  {len(sample_pids)} persons',
                 fontsize=13, fontweight='bold')
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    purp_non_travel = [p for p in PURPOSE_COLORS if p not in ('home','travel')]

    for ax, pid in zip(axes_flat, sample_pids):
        person = plan[plan['person_id'] == pid].sort_values('STARTIME')
        if person.empty:
            ax.set_visible(False)
            continue

        trips_p = person[person['row_type'] == 'trip'] if 'row_type' in person.columns \
                  else pd.DataFrame()
        acts_p  = person[person['row_type'] == 'activity'] if 'row_type' in person.columns \
                  else person

        # Collect unique non-trivial purposes for y-axis — include 'home'
        # so the morning and evening home dwell periods are visible.
        if 'purpose' in acts_p.columns:
            purposes = [p for p in acts_p['purpose'].unique()
                        if p not in ('travel','') and pd.notna(p)]
            if 'home' not in purposes:
                purposes = ['home'] + purposes
            else:
                purposes = ['home'] + [p for p in purposes if p != 'home']
        else:
            purposes = ['home'] + purp_non_travel

        if not purposes:
            ax.set_visible(False)
            continue

        purpose_y = {p: i for i, p in enumerate(purposes)}
        y_max = len(purposes)

        x_min = max(4.0,  (person['STARTIME'].min() / 60 - 0.5) if 'STARTIME' in person.columns else 6.0)
        x_max = min(25.0, (person['ENDTIME'].max()  / 60 + 0.5) if 'ENDTIME'  in person.columns else 23.0)

        # Infer home dwell as the complement of trips + non-home activities.
        home_blocks = []
        occupied = []
        if 'STARTIME' in person.columns and 'ENDTIME' in person.columns:
            if not trips_p.empty:
                for _, tr in trips_p.iterrows():
                    occupied.append((
                        float(tr.get('STARTIME', 0)) / 60.0,
                        float(tr.get('ENDTIME', 0)) / 60.0,
                    ))
            if not acts_p.empty:
                for _, act in acts_p.iterrows():
                    purp = str(act.get('purpose', ''))
                    if purp == 'home':
                        continue
                    occupied.append((
                        float(act.get('STARTIME', 0)) / 60.0,
                        float(act.get('ENDTIME', 0)) / 60.0,
                    ))
        occupied = [(max(x_min, s), min(x_max, e)) for s, e in occupied if e > s]
        occupied.sort()
        cursor = x_min
        for s, e in occupied:
            if s > cursor:
                home_blocks.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < x_max:
            home_blocks.append((cursor, x_max))

        # ── Draw trip legs as coloured bars in a shared row at y=-0.3 ───────
        if not trips_p.empty and 'STARTIME' in trips_p.columns:
            for _, tr in trips_p.iterrows():
                t_start = float(tr.get('STARTIME', 0)) / 60
                t_end   = float(tr.get('ENDTIME',  t_start + 0.5)) / 60
                raw_mn  = tr.get('mode_name', '')
                # Guard against float NaN coming from the DataFrame
                mode_n  = str(raw_mn) if (raw_mn is not None and
                              not (isinstance(raw_mn, float) and
                                   _math_module.isnan(raw_mn))) else ''
                color   = MODE_COLORS.get(mode_n, '#CCCCCC')
                if t_end > t_start:
                    ax.barh(y=-0.3, width=t_end-t_start, left=t_start,
                            height=0.25, color=color, alpha=0.7, zorder=2)
                    if (t_end - t_start) > 0.15 and mode_n:
                        ax.text(t_start + (t_end - t_start)/2, -0.3,
                                mode_n, ha='center', va='center',
                                fontsize=6, color='white', fontweight='bold',
                                zorder=3)

        # ── Draw activity dwells (coloured by purpose) ────────────────────
        legend_purps = set()
        if 'home' in purpose_y:
            y_home = purpose_y['home']
            for h_start, h_end in home_blocks:
                if h_end <= h_start:
                    continue
                dur_hr = h_end - h_start
                ax.barh(y=y_home, width=dur_hr, left=h_start,
                        height=0.7, color=PURPOSE_COLORS.get('home', '#DDDDDD'),
                        alpha=0.95, zorder=1, edgecolor='white', linewidth=0.5)
                if dur_hr > 0.35:
                    ax.text(h_start + dur_hr/2, y_home - 0.42,
                            f'{int(round(dur_hr * 60))}m',
                            ha='center', va='top', fontsize=6, color='#555', zorder=4)
            if home_blocks:
                legend_purps.add('home')
        if not acts_p.empty and 'STARTIME' in acts_p.columns:
            for _, act in acts_p.iterrows():
                purp    = str(act.get('purpose', ''))
                a_start = float(act.get('STARTIME', 0)) / 60
                a_end   = float(act.get('ENDTIME',  a_start + 0.5)) / 60
                dur_hr  = a_end - a_start

                if purp not in purpose_y or dur_hr <= 0:
                    continue

                y_pos   = purpose_y[purp]
                color   = PURPOSE_COLORS.get(purp, '#888888')
                ax.barh(y=y_pos, width=dur_hr, left=a_start,
                        height=0.7, color=color, alpha=0.85, zorder=3,
                        edgecolor='white', linewidth=0.5)

                # Duration annotation inside bar
                if dur_hr > 0.3:
                    dur_min = int(round(dur_hr * 60))
                    ax.text(a_start + dur_hr/2, y_pos - 0.42,
                            f'{dur_min}m', ha='center', va='top', fontsize=6,
                            color='#555', zorder=4)
                legend_purps.add(purp)

        # ── Axes formatting ────────────────────────────────────────────────
        ax.set_yticks(range(len(purposes)))
        ax.set_yticklabels(purposes, fontsize=9)
        ax.set_ylim(-0.7, y_max - 0.3)

        # x-axis: hours
        ax.set_xlim(x_min, x_max)
        tick_hours = [h for h in range(4, 26) if x_min <= h <= x_max]
        ax.set_xticks(tick_hours)
        ax.set_xticklabels([f'{h}{"am" if h<12 else "pm"}' if h != 12 else '12pm'
                            for h in tick_hours], fontsize=8, rotation=30)
        ax.set_xlabel('Time of day', fontsize=8)
        ax.grid(axis='x', alpha=0.25)

        # Title with person summary
        ptype = str(person['ptype'].iloc[0]) if 'ptype' in person.columns else ''
        vehs  = int(person['vehrd'].iloc[0]) if 'vehrd' in person.columns else 0
        n_a   = len(acts_p)
        t_start_str = _hm(person['STARTIME'].min()) if 'STARTIME' in person.columns else '?'
        t_end_str   = _hm(person['ENDTIME'].max())  if 'ENDTIME'  in person.columns else '?'

        # Mode summary from trip rows
        if not trips_p.empty and 'mode_name' in trips_p.columns:
            mc = trips_p['mode_name'].value_counts()
            mode_summary = '  '.join(f'{m}×{c}' for m, c in mc.items() if m)
        else:
            mode_summary = ''

        ax.set_title(
            f'Person {pid}  ·  {ptype}  ·  vehicles: {vehs}  ·  '
            f'{n_a} activities  ·  {t_start_str}–{t_end_str}'
            + (f'  ·  modes: {mode_summary}' if mode_summary else ''),
            fontsize=8, pad=4
        )

        # Legend for purposes in this person's plan
        handles = [mpatches.Patch(color=PURPOSE_COLORS.get(p,'#888'), label=p, alpha=0.85)
                   for p in sorted(legend_purps)]
        if not trips_p.empty and 'mode_name' in trips_p.columns:
            for mn in trips_p['mode_name'].dropna().unique():
                if mn:
                    handles.append(mpatches.Patch(color=MODE_COLORS.get(mn,'#CCC'),
                                                   label=mn, alpha=0.5))
        if handles:
            ax.legend(handles=handles, fontsize=7, loc='upper right',
                      ncol=max(1, len(handles)//2), framealpha=0.7)

    for ax in axes_flat[len(sample_pids):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"  Sample plans → {save_path}")
    else:
        return fig


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def plot_all(plan: pd.DataFrame, n_sample: int = 8, save_dir: str = '.'):
    """
    Generate both summary and sample plan plots.

    Parameters
    ----------
    plan      : full pipeline plan DataFrame (trips + activities interleaved)
    n_sample  : number of individual plans to show
    save_dir  : directory to save PNG files into
    """
    os.makedirs(save_dir, exist_ok=True)

    # Drop rows with no meaningful time data
    if 'STARTIME' in plan.columns:
        plan = plan[plan['STARTIME'].notna() & (plan['STARTIME'] < 1440 * 2)].copy()

    plot_population_summary(plan, os.path.join(save_dir, 'population_summary.png'))
    plot_sample_plans(plan, n=n_sample, save_path=os.path.join(save_dir, 'sample_plans.png'))
