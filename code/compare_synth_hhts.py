"""
compare_synth_hhts.py — Compare synthetic population outputs against HHTS survey data.

Reads the most recent safety_baseline run's stage6_output.csv (synth) and the
HHTS tour/trip data (via pipeline_logger.load_data) to produce a comparison
report showing what the synthetic population adds beyond the survey sample.

Produces:
  safety_analysis/plots/synth_vs_hhts_mode_share.png
  safety_analysis/plots/synth_vs_hhts_trip_counts.png
  safety_analysis/plots/synth_vs_hhts_distance.png
  safety_analysis/synth_vs_hhts_comparison.csv
  safety_analysis/synth_population_report.html
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

MODE_NAMES = {1: 'Drive Alone', 2: 'Shared Ride', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
MODE_COLORS = {'Drive Alone': '#E74C3C', 'Shared Ride': '#E67E22',
               'Walk': '#2ECC71', 'Bike': '#3498DB', 'Transit': '#9B59B6'}
SEGMENT_LABELS = {'work': 'Work', 'edu': 'Education', 'nonmand': 'Non-Mandatory'}


def _latest_dir(prefix, base='runs'):
    cand = sorted(
        glob.glob(os.path.join(base, prefix + '__*')),
        key=os.path.getmtime, reverse=True,
    )
    cand = [c for c in cand if os.path.isdir(c)]
    return cand[0] if cand else os.path.join(base, prefix)


def main():
    print("=" * 60)
    print("SYNTHETIC POPULATION vs HHTS SURVEY COMPARISON")
    print("=" * 60)

    # ── Find run directories ──────────────────────────────────────────────
    synth_dir = _latest_dir('safety_baseline')
    out_dir = _latest_dir('safety_analysis')
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n  Synthetic pop dir: {synth_dir}")
    print(f"  Output dir:        {out_dir}")

    # ── Load synth data ────────────────────────────────────────────────────
    s6_path = os.path.join(synth_dir, 'stage6_output.csv')
    if not os.path.exists(s6_path):
        s6_path = os.path.join(synth_dir, 'trips_for_aimsun.csv')
    if not os.path.exists(s6_path):
        print(f"  ERROR: stage6_output.csv not found in {synth_dir}")
        sys.exit(1)

    synth_plan = pd.read_csv(s6_path)
    synth_trips = synth_plan[synth_plan['row_type'] == 'trip'] if 'row_type' in synth_plan.columns else synth_plan
    print(f"  Synth trips: {len(synth_trips):,}")

    # ── Load HHTS data ─────────────────────────────────────────────────────
    sys.path.insert(0, '.')
    from pipeline_logger import load_data
    sc, hhts_out, landuse = load_data()
    hhts_tours = hhts_out.get('tours', pd.DataFrame())
    hhts_trips = hhts_out.get('trips', pd.DataFrame())
    hhts_persons = hhts_out.get('persons', pd.DataFrame())
    print(f"  HHTS persons: {len(hhts_persons):,}")
    print(f"  HHTS tours: {len(hhts_tours):,}")
    print(f"  HHTS trips: {len(hhts_trips):,}")

    # ── Derive HHTS mode and segment ──────────────────────────────────────
    if not hhts_trips.empty:
        if 'MODE' in hhts_trips.columns:
            hhts_trips['mode_num'] = pd.to_numeric(hhts_trips['MODE'], errors='coerce')
            hhts_trips['mode_name'] = hhts_trips['mode_num'].map(MODE_NAMES)
        if 'main_purpose' in hhts_trips.columns:
            _map = {'Work': 'work', 'Education': 'edu'}
            hhts_trips['segment'] = hhts_trips['main_purpose'].map(_map).fillna('nonmand')
        if 'trip_dist' in hhts_trips.columns:
            hhts_trips['dist_km'] = pd.to_numeric(hhts_trips['trip_dist'], errors='coerce')
        elif 'DIST' in hhts_trips.columns:
            hhts_trips['dist_km'] = pd.to_numeric(hhts_trips['DIST'], errors='coerce')

    # ── Standardise synth columns ──────────────────────────────────────────
    if 'mode' in synth_trips.columns:
        synth_trips['mode_num'] = pd.to_numeric(synth_trips['mode'], errors='coerce')
        synth_trips['mode_name'] = synth_trips['mode_num'].map(MODE_NAMES)
    if 'segment' not in synth_trips.columns and 'purpose' in synth_trips.columns:
        synth_trips['segment'] = synth_trips['purpose']

    # ── Scale factor ──────────────────────────────────────────────────────
    # HHTS = sample (~20k persons), synth = full population (3.7M persons)
    n_hhts_persons = len(hhts_persons) if not hhts_persons.empty else 1
    n_synth = 3_700_000  # approximate
    scale_hhts_pct = (n_synth / max(n_hhts_persons, 1)) / n_synth  # per-person scaling

    rows = []

    # ── 1. Mode shares ──────────────────────────────────────────────────────
    def _ms(df, label):
        if df.empty or 'mode_name' not in df.columns:
            return {}
        valid = df['mode_name'].dropna()
        if valid.empty:
            return {}
        return valid.value_counts(normalize=True).to_dict()

    ms_synth = _ms(synth_trips, 'Synth')
    ms_hhts = _ms(hhts_trips, 'HHTS')

    all_modes = sorted(set(list(ms_synth.keys()) + list(ms_hhts.keys())))
    for m in all_modes:
        rows.append({
            'category': 'mode_share', 'run': 'synth_population',
            'parameter': str(m), 'beta': round(float(ms_synth.get(m, 0)) * 100, 2),
            'se': round(float(ms_hhts.get(m, 0)) * 100, 2), 't_ratio': None, 'p_value': None,
        })

    # Active travel share
    active_synth = sum(ms_synth.get(m, 0) for m in ['Walk', 'Bike']) * 100
    active_hhts = sum(ms_hhts.get(m, 0) for m in ['Walk', 'Bike']) * 100
    rows.append({
        'category': 'active_mode', 'run': 'comparison',
        'parameter': 'active_share_pct',
        'beta': round(active_synth, 2), 'se': round(active_hhts, 2),
        't_ratio': round(active_synth - active_hhts, 2), 'p_value': None,
    })

    # ── 2. Trip counts per person ──────────────────────────────────────────
    n_synth_persons = synth_trips['person_id'].nunique() if 'person_id' in synth_trips.columns else len(synth_trips)
    tpp_synth = len(synth_trips) / max(n_synth_persons, 1)
    tpp_hhts = len(hhts_trips) / max(len(hhts_persons), 1) if not hhts_trips.empty else 0
    rows.append({
        'category': 'trip_count', 'run': 'comparison',
        'parameter': 'trips_per_person',
        'beta': round(tpp_synth, 2), 'se': round(tpp_hhts, 2),
        't_ratio': round(tpp_synth - tpp_hhts, 2), 'p_value': None,
    })
    rows.append({
        'category': 'trip_count', 'run': 'comparison',
        'parameter': 'total_trips',
        'beta': len(synth_trips), 'se': len(hhts_trips),
        't_ratio': round(len(synth_trips) / max(len(hhts_trips), 1), 1), 'p_value': None,
    })
    rows.append({
        'category': 'trip_count', 'run': 'comparison',
        'parameter': 'total_persons',
        'beta': n_synth_persons, 'se': len(hhts_persons),
        't_ratio': None, 'p_value': None,
    })

    # ── 3. Segment-level mode shares ────────────────────────────────────────
    for seg in ['work', 'edu', 'nonmand']:
        for m in all_modes:
            s_val = 0
            h_val = 0
            if not synth_trips.empty and 'segment' in synth_trips.columns:
                s_mask = synth_trips['segment'] == seg
                if s_mask.any():
                    s_counts = synth_trips.loc[s_mask, 'mode_name'].value_counts(normalize=True)
                    s_val = float(s_counts.get(m, 0)) * 100
            if not hhts_trips.empty and 'segment' in hhts_trips.columns:
                h_mask = hhts_trips['segment'] == seg
                if h_mask.any():
                    h_counts = hhts_trips.loc[h_mask, 'mode_name'].value_counts(normalize=True)
                    h_val = float(h_counts.get(m, 0)) * 100
            rows.append({
                'category': f'mode_share_{seg}',
                'run': 'synth_population',
                'parameter': str(m),
                'beta': round(s_val, 2), 'se': round(h_val, 2),
                't_ratio': round(s_val - h_val, 2), 'p_value': None,
            })

    # ── 4. Distance stats ────────────────────────────────────────────────────
    def _dists(df):
        if 'dist_km' in df.columns:
            vals = pd.to_numeric(df['dist_km'], errors='coerce').dropna()
            if len(vals) > 0:
                return {'mean': vals.mean(), 'median': vals.median(),
                        'p90': vals.quantile(0.90)}
        return {}
    ds = _dists(synth_trips)
    dh = _dists(hhts_trips)
    for stat in ['mean', 'median', 'p90']:
        rows.append({
            'category': 'distance', 'run': 'comparison',
            'parameter': f'dist_{stat}_km',
            'beta': round(ds.get(stat, 0), 2), 'se': round(dh.get(stat, 0), 2),
            't_ratio': round(ds.get(stat, 0) - dh.get(stat, 0), 2), 'p_value': None,
        })

    # ── 5. Spatial coverage ──────────────────────────────────────────────────
    synth_ods = 0
    hhts_ods = 0
    if 'OTAZ' in synth_trips.columns and 'DTAZ' in synth_trips.columns:
        synth_ods = synth_trips[['OTAZ', 'DTAZ']].drop_duplicates().shape[0]
    if 'OTAZ' in hhts_trips.columns and 'DTAZ' in hhts_trips.columns:
        hhts_ods = hhts_trips[['OTAZ', 'DTAZ']].drop_duplicates().shape[0]
    rows.append({
        'category': 'spatial_coverage', 'run': 'comparison',
        'parameter': 'unique_od_pairs',
        'beta': synth_ods, 'se': hhts_ods,
        't_ratio': round(synth_ods / max(hhts_ods, 1), 1), 'p_value': None,
    })

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'synth_vs_hhts_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Comparison CSV → {csv_path}")

    # ── Plot 1: Mode share comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Synthetic Population vs HHTS Survey — Mode Shares', fontweight='bold')

    # Overall
    ax = axes[0]
    modes = [m for m in all_modes if m in ms_synth or m in ms_hhts]
    x = np.arange(len(modes))
    w = 0.35
    cols = [MODE_COLORS.get(m, '#888') for m in modes]
    ax.bar(x - w/2, [ms_synth.get(m, 0) * 100 for m in modes], w,
           color=cols, alpha=0.85, label='Synthetic Population\n(3.7M persons)')
    ax.bar(x + w/2, [ms_hhts.get(m, 0) * 100 for m in modes], w,
           color=cols, alpha=0.45, hatch='//', edgecolor='grey', label='HHTS Survey\n(~20k persons)')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha='right')
    ax.set_ylabel('Mode Share (%)')
    ax.set_title('Overall Mode Share')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for i, m in enumerate(modes):
        sv = ms_synth.get(m, 0) * 100
        hv = ms_hhts.get(m, 0) * 100
        if sv > 0.5:
            ax.text(i - w/2, sv + 0.3, f'{sv:.1f}%', ha='center', fontsize=8)
        if hv > 0.5:
            ax.text(i + w/2, hv + 0.3, f'{hv:.1f}%', ha='center', fontsize=8)

    # By segment
    ax = axes[1]
    seg_labels = ['Work', 'Education', 'Non-Mandatory']
    seg_active_s = []
    seg_active_h = []
    for seg in ['work', 'edu', 'nonmand']:
        s_val = 0
        h_val = 0
        if not synth_trips.empty and 'segment' in synth_trips.columns:
            sm = synth_trips[synth_trips['segment'] == seg]
            if not sm.empty:
                s_val = sm['mode_name'].isin(['Walk', 'Bike']).mean() * 100
        if not hhts_trips.empty and 'segment' in hhts_trips.columns:
            hm = hhts_trips[hhts_trips['segment'] == seg]
            if not hm.empty:
                h_val = hm['mode_name'].isin(['Walk', 'Bike']).mean() * 100
        seg_active_s.append(s_val)
        seg_active_h.append(h_val)
    x2 = np.arange(len(seg_labels))
    ax.bar(x2 - w/2, seg_active_s, w, color='#2ECC71', alpha=0.85, label='Synthetic Population')
    ax.bar(x2 + w/2, seg_active_h, w, color='#2ECC71', alpha=0.35,
           hatch='//', edgecolor='grey', label='HHTS Survey')
    ax.set_xticks(x2)
    ax.set_xticklabels(seg_labels)
    ax.set_ylabel('Active Mode Share (%)')
    ax.set_title('Active Travel by Purpose Segment')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, 'synth_vs_hhts_mode_share.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {os.path.basename(p1)}")

    # ── Plot 2: Trip count comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ['Total Trips', 'Trips per\nPerson', 'Unique OD\nPairs']
    synth_vals = [len(synth_trips), tpp_synth, synth_ods]
    hhts_vals = [len(hhts_trips), tpp_hhts, hhts_ods]
    x3 = np.arange(len(categories))
    ax.bar(x3 - w/2, synth_vals, w, color='#2E86AB', alpha=0.85, label='Synthetic Population')
    ax.bar(x3 + w/2, hhts_vals, w, color='#E84855', alpha=0.75, label='HHTS Survey')
    ax.set_xticks(x3)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Count')
    ax.set_title('Scale Comparison: Synthetic Population vs HHTS Survey')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for i, (sv, hv) in enumerate(zip(synth_vals, hhts_vals)):
        ax.text(i - w/2, sv + max(synth_vals) * 0.02, f'{sv:,.0f}', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
        ax.text(i + w/2, hv + max(synth_vals) * 0.02, f'{hv:,.0f}', ha='center', fontsize=9, color='#E84855')
    plt.tight_layout()
    p2 = os.path.join(plots_dir, 'synth_vs_hhts_trip_counts.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {os.path.basename(p2)}")

    # ── Plot 3: Distance distributions ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'dist_km' in synth_trips.columns:
        sv = pd.to_numeric(synth_trips['dist_km'], errors='coerce').clip(0, 60).dropna()
        ax.hist(sv, bins=50, density=True, alpha=0.5, color='#2E86AB',
                label=f'Synthetic Population\n(mean={sv.mean():.1f} km, trips={len(sv):,})')
    if 'dist_km' in hhts_trips.columns:
        hv = pd.to_numeric(hhts_trips['dist_km'], errors='coerce').clip(0, 60).dropna()
        ax.hist(hv, bins=50, density=True, alpha=0.5, color='#E84855',
                label=f'HHTS Survey\n(mean={hv.mean():.1f} km, trips={len(hv):,})')
    ax.set_xlabel('Trip Distance (km)')
    ax.set_ylabel('Density')
    ax.set_title('Trip Distance Distribution: Synthetic Population vs HHTS Survey')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, 'synth_vs_hhts_distance.png')
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot → {os.path.basename(p3)}")

    # ── HTML report ───────────────────────────────────────────────────────────
    scale_factor = round(n_synth / max(len(hhts_persons), 1))
    synth_zones = synth_trips['OTAZ'].nunique() if 'OTAZ' in synth_trips.columns else '?'
    hhts_zones = hhts_trips['OTAZ'].nunique() if 'OTAZ' in hhts_trips.columns else '?'

    report_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Synthetic Population vs HHTS Comparison</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;max-width:1200px}}
h1{{background:#2c3e50;color:white;padding:15px;border-radius:4px}}
h2{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}
th{{background:#2c3e50;color:white;padding:8px 12px}}
td{{border:1px solid #ddd;padding:6px 12px}}
tr:nth-child(even){{background:#f9f9f9}}
.box{{background:#eaf4fb;border-left:4px solid #2980b9;padding:12px;margin:10px 0}}
</style></head><body>
<h1>Synthetic Population vs HHTS Survey — Scale & Coverage</h1>

<div class="box">
<strong>Why the synthetic population matters:</strong>
The Queensland Household Travel Survey (HHTS) provides high-quality trip diaries for
approximately {len(hhts_persons):,} persons.  However, survey data alone covers only
{hhts_zones} of the 371 SA2 zones in SEQ and represents a sample — not the full population.
The synthetic population expands this to {n_synth:,.0f} persons across all {synth_zones} zones,
generating {len(synth_trips):,} daily trips versus the {len(hhts_trips):,} trip records in
the HHTS sample.  This ~{scale_factor}&times; scale-up enables:
<ul>
  <li><b>Zone-level policy evaluation</b> — mode shares and accessibility can be
  computed for every SA2 zone, not just sampled zones</li>
  <li><b>Spatial coverage</b> — {synth_ods:,} unique OD pairs are covered vs {hhts_ods:,}
  in the survey, enabling network-wide assignment</li>
  <li><b>Distributional analysis</b> — equity impacts across demographic groups and
  geographic areas can be assessed at census-tract resolution</li>
  <li><b>Counterfactual experiments</b> — safety investment scenarios can be
  evaluated at the full network scale rather than on a sparse sample</li>
</ul>
</div>

<h2>Key Metrics</h2>
<table>
<tr><th>Metric</th><th>Synthetic Population</th><th>HHTS Survey</th><th>Ratio</th></tr>
<tr><td>Persons</td><td>{n_synth:,.0f}</td><td>{len(hhts_persons):,}</td><td>{scale_factor}&times;</td></tr>
<tr><td>Daily Trips</td><td>{len(synth_trips):,}</td><td>{len(hhts_trips):,}</td><td>{len(synth_trips)/max(len(hhts_trips),1):.1f}&times;</td></tr>
<tr><td>Trips per Person</td><td>{tpp_synth:.2f}</td><td>{tpp_hhts:.2f}</td><td>{tpp_synth/max(tpp_hhts,0.01):.1f}&times;</td></tr>
<tr><td>Unique OD Pairs</td><td>{synth_ods:,}</td><td>{hhts_ods:,}</td><td>{synth_ods/max(hhts_ods,1):.1f}&times;</td></tr>
<tr><td>Active Mode Share</td><td>{active_synth:.1f}%</td><td>{active_hhts:.1f}%</td><td>Δ={active_synth-active_hhts:+.1f}pp</td></tr>
<tr><td>Mean Trip Distance</td><td>{ds.get('mean',0):.1f} km</td><td>{dh.get('mean',0):.1f} km</td><td>Δ={ds.get('mean',0)-dh.get('mean',0):+.1f} km</td></tr>
</table>

<h2>Visual Comparison</h2>
<img src="plots/synth_vs_hhts_mode_share.png" style="max-width:100%"><br>
<img src="plots/synth_vs_hhts_trip_counts.png" style="max-width:100%"><br>
<img src="plots/synth_vs_hhts_distance.png" style="max-width:100%"><br>

<hr><p style="color:#888;font-size:11px">
Generated by compare_synth_hhts.py — Synth dir: {synth_dir}</p>
</body></html>"""

    html_path = os.path.join(out_dir, 'synth_population_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(report_html)
    print(f"\n  HTML report → {html_path}")
    print("=" * 60)
    return out_dir


if __name__ == '__main__':
    main()
