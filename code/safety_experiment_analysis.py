"""
safety_experiment_analysis.py — SEQ ABM safety skim experiment comparison.
Reads runs/safety_baseline/ and runs/safety_nosafety/ and produces:
  runs/safety_analysis/safety_report.html
  runs/safety_analysis/safety_comparison.csv
  runs/safety_analysis/plots/*.png
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DIR_BASE  = 'runs/safety_baseline'
DIR_NOSA  = 'runs/safety_nosafety'
DIR_OUT   = 'runs/safety_analysis'
PLOTS_DIR = os.path.join(DIR_OUT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODE_NAMES  = {1:'Drive Alone',2:'Shared Ride',3:'Walk',4:'Bike',5:'Transit'}
MODE_COLORS = {'Drive Alone':'#E74C3C','Shared Ride':'#E67E22',
               'Walk':'#2ECC71','Bike':'#3498DB','Transit':'#9B59B6'}
LANDUSE_CANDIDATES = [
    'zone_order_and_landuse.csv',
    os.path.join('saved_data', 'zone_order_and_landuse.csv'),
]

def _csv(d, f, n=None):
    p = os.path.join(d, f)
    if os.path.exists(p):
        df = pd.read_csv(p, nrows=n); print(f"  ✓ {f}: {len(df):,} rows"); return df
    print(f"  ✗ {f} not found"); return pd.DataFrame()

def load_run(d, label):
    print(f"\nLoading '{label}' from {d}")
    return dict(label=label, dir=d,
                plan  = _csv(d,'stage6_output.csv'),
                s5    = _csv(d,'stage5_output.csv'),
                aimsun= _csv(d,'trips_for_aimsun.csv'),
                coef  = _csv(d,'stage5_models_coefficients.csv'))

BASE = load_run(DIR_BASE, 'With Safety Skims')
NOSA = load_run(DIR_NOSA, 'Without Safety Skims')


def load_landuse():
    for p in LANDUSE_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                print(f"  ✓ landuse: {p} ({len(df):,} zones)")
                return df
            except Exception as e:
                print(f"  × landuse failed from {p}: {e}")
    return pd.DataFrame()


LANDUSE = load_landuse()

# ── helpers ──────────────────────────────────────────────────────────────────
def trips(run):
    df = run['plan']
    return df[df['row_type']=='trip'] if 'row_type' in df.columns else df

def mode_shares(run):
    t = trips(run)
    if t.empty or 'mode' not in t.columns: return pd.Series(dtype=float)
    return t['mode'].map(MODE_NAMES).value_counts(normalize=True)*100

def active_pct(run):
    t = trips(run)
    if t.empty or 'mode' not in t.columns: return float('nan')
    return (t['mode'].isin([3,4])).mean()*100

def mean_dist(run):
    t = trips(run)
    return t['dist_km'].mean() if 'dist_km' in t.columns else float('nan')

def seg_trips(run, seg):
    t = trips(run)
    return t[t['segment']==seg] if 'segment' in t.columns else t

# ── Figure 1: mode shares ─────────────────────────────────────────────────────
def fig_mode_shares(path):
    bs, ns = mode_shares(BASE), mode_shares(NOSA)
    if bs.empty and ns.empty: return
    modes = [m for m in list(MODE_NAMES.values()) if m in bs.index or m in ns.index]
    x, w = np.arange(len(modes)), 0.35
    fig, (a1,a2) = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle('Mode Share: Safety Skims vs No Safety Skims', fontweight='bold')
    bv = [bs.get(m,0) for m in modes]
    nv = [ns.get(m,0) for m in modes]
    cols = [MODE_COLORS.get(m,'#888') for m in modes]
    a1.bar(x-w/2, bv, w, color=cols, alpha=.85, label='With Safety')
    a1.bar(x+w/2, nv, w, color=cols, alpha=.45, hatch='//', edgecolor='grey', label='Without Safety')
    a1.set_xticks(x); a1.set_xticklabels(modes, rotation=20, ha='right')
    a1.set_ylabel('Share (%)'); a1.legend(); a1.grid(axis='y',alpha=.3)
    for xp, v in zip(x-w/2, bv):
        if v>.5: a1.text(xp, v+.3, f'{v:.1f}%', ha='center', fontsize=8)
    for xp, v in zip(x+w/2, nv):
        if v>.5: a1.text(xp, v+.3, f'{v:.1f}%', ha='center', fontsize=8)
    deltas = [nv[i]-bv[i] for i in range(len(modes))]
    a2.barh(modes, deltas, color=['#2ECC71' if d<0 else '#E74C3C' for d in deltas], alpha=.85)
    a2.axvline(0, color='black', lw=.8)
    a2.set_xlabel('Δ Share (pp)'); a2.set_title('Shift from removing safety skims')
    a2.grid(axis='x',alpha=.3)
    for i,d in enumerate(deltas):
        if abs(d)>.05: a2.text(d+(0.05 if d>=0 else -.05), i, f'{d:+.2f}pp', va='center',
                                ha='left' if d>=0 else 'right', fontsize=9)
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  saved → {os.path.basename(path)}")

# ── Figure 2: distance distributions ─────────────────────────────────────────
def fig_distance(path):
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle('Trip Distance Distribution by Segment', fontweight='bold')
    for ax, seg, lab in zip(axes,['work','edu','nonmand'],['Work','Education','Non-Mandatory']):
        for run, col, lbl in [(BASE,'#2E86AB','With Safety'),(NOSA,'#E84855','Without Safety')]:
            v = seg_trips(run,seg).get('dist_km',pd.Series()).clip(0,60).dropna()
            if len(v)>50:
                ax.hist(v, bins=40, density=True, alpha=.45, color=col, label=lbl)
                ax.axvline(v.median(), color=col, lw=2, ls='--',
                           label=f'Median {v.median():.1f}km')
        ax.set_title(lab); ax.set_xlabel('Distance (km)'); ax.set_ylabel('Density')
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  saved → {os.path.basename(path)}")

# ── Figure 3: active mode by distance band ────────────────────────────────────
def fig_active_by_dist(path):
    bins   = [0,2,5,10,20,50,999]
    labels = ['<2km','2-5km','5-10km','10-20km','20-50km','>50km']
    fig, axes = plt.subplots(1,3,figsize=(16,5),sharey=True)
    fig.suptitle('Active Mode Share (Walk + Bike) by Distance Band', fontweight='bold')
    for ax, seg, lab in zip(axes,['work','edu','nonmand'],['Work','Education','Non-Mandatory']):
        for run, col, ls, lbl in [(BASE,'#2ECC71','-','With Safety'),
                                   (NOSA,'#E74C3C','--','Without Safety')]:
            df = seg_trips(run, seg).copy()
            if df.empty or 'dist_km' not in df.columns or 'mode' not in df.columns: continue
            df['band']   = pd.cut(df['dist_km'], bins=bins, labels=labels, right=False)
            df['active'] = df['mode'].isin([3,4])
            g = df.groupby('band',observed=True)['active'].mean()*100
            ax.plot(range(len(g)), g.values, marker='o', ls=ls, color=col, label=lbl, lw=2)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=30,ha='right',fontsize=9)
        ax.set_title(lab); ax.set_ylabel('Active Mode Share (%)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  saved → {os.path.basename(path)}")

# ── Figure 4: coefficient comparison ─────────────────────────────────────────
def fig_coef(path):
    b, n = BASE['coef'], NOSA['coef']
    if b.empty or n.empty: return
    pc = b.columns[0]; vc = next(c for c in b.columns if 'value' in c.lower())
    focus = ['c_crash_veh','c_crash','c_ivt_auto','c_ivt_transit',
             'c_log_dist','c_logsum','c_aggl','c_comp']
    m = pd.merge(b[[pc,'segment',vc]].rename(columns={vc:'base'}),
                 n[[pc,'segment',vc]].rename(columns={vc:'nosafety'}),
                 on=[pc,'segment'],how='outer').fillna(0)
    m = m[m[pc].isin(focus)]
    if m.empty: return
    fig, axes = plt.subplots(1,3,figsize=(16,6))
    fig.suptitle('Key Coefficient Comparison', fontweight='bold')
    for ax, (seg, lab) in zip(axes, [('work','Work'),('edu','Education'),('nonmand','Non-Mandatory')]):
        s = m[m['segment']==seg] if 'segment' in m.columns else m
        if s.empty: ax.set_title(lab); continue
        params=s[pc].tolist(); bv=s['base'].tolist(); nv=s['nosafety'].tolist()
        x=np.arange(len(params)); w=.35
        ax.barh(x-w/2, bv, w, color='#2E86AB', alpha=.85, label='With Safety')
        ax.barh(x+w/2, nv, w, color='#E84855', alpha=.75, label='Without Safety')
        ax.set_yticks(x); ax.set_yticklabels(params, fontsize=9)
        ax.axvline(0, color='black', lw=.8); ax.set_title(lab)
        ax.legend(fontsize=8); ax.grid(axis='x',alpha=.3)
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  saved → {os.path.basename(path)}")

# ── Figure 5: zone-level active shift ────────────────────────────────────────
def fig_zone_shift(path):
    for run in [BASE, NOSA]:
        if run['plan'].empty or 'OTAZ' not in run['plan'].columns: return
    b = BASE['plan'].copy(); n = NOSA['plan'].copy()
    if 'mode' not in b.columns or 'mode' not in n.columns: return
    bz = b.assign(active=b['mode'].isin([3,4])).groupby('OTAZ')['active'].mean()*100
    nz = n.assign(active=n['mode'].isin([3,4])).groupby('OTAZ')['active'].mean()*100
    delta = (nz-bz).dropna().sort_values()
    top = pd.concat([delta.head(10), delta.tail(10)])
    fig, ax = plt.subplots(figsize=(10,8))
    colors = ['#2ECC71' if v<0 else '#E74C3C' for v in top.values]
    ax.barh(top.index.astype(str), top.values, color=colors, alpha=.85)
    ax.axvline(0, color='black', lw=.8)
    ax.set_xlabel('Δ Active Mode Share (pp) — removing safety skims')
    ax.set_ylabel('Origin Zone (OTAZ)')
    ax.set_title('Top 10 zones: most and least affected by safety skim removal')
    ax.grid(axis='x',alpha=.3)
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
    print(f"  saved → {os.path.basename(path)}")

# ── Summary table ─────────────────────────────────────────────────────────────
def build_summary():
    rows = []
    for run in [BASE, NOSA]:
        t = trips(run)
        row = {'Run': run['label'], 'Total trips': f"{len(t):,}",
               'Active mode %': f"{active_pct(run):.2f}%",
               'Mean dist km': f"{mean_dist(run):.2f}"}
        s = mode_shares(run)
        for m in ['Drive Alone','Walk','Bike','Transit']:
            row[m] = f"{s.get(m,0):.2f}%"
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) == 2:
        diff = {'Run':'Δ (nosafety − baseline)'}
        for c in df.columns:
            if c == 'Run': continue
            try:
                v0 = float(str(df.iloc[0][c]).replace('%','').replace(',',''))
                v1 = float(str(df.iloc[1][c]).replace('%','').replace(',',''))
                diff[c] = f"{v1-v0:+.2f}"
            except: diff[c] = '—'
        df = pd.concat([df, pd.DataFrame([diff])], ignore_index=True)
    return df


def build_safety_proxy_summary():
    if LANDUSE.empty:
        return {}
    lu = LANDUSE.copy()
    for col in ['EATOUT', 'Shopping', 'Entertainment', 'community', 'civic',
                'park', 'sport', 'childcare', 'Healthcare', 'commercial',
                'retail', 'industrial', 'TOTPOP']:
        if col not in lu.columns:
            lu[col] = 0.0
        lu[col] = pd.to_numeric(lu[col], errors='coerce').fillna(0.0)
    lu['activity_frontage_proxy'] = (
        lu['EATOUT'] + lu['Shopping'] + lu['Entertainment'] +
        0.0001 * (lu['commercial'] + lu['retail'])
    )
    lu['community_presence_proxy'] = (
        lu['community'] + lu['civic'] + lu['childcare'] + lu['Healthcare']
    )
    lu['open_space_proxy'] = lu['park'] + lu['sport']
    lu['exposure_proxy'] = lu['TOTPOP'] + 0.25 * lu['industrial']
    return {
        'frontage_top3': lu.nlargest(3, 'activity_frontage_proxy')[['ORDER', 'activity_frontage_proxy']]
                           if 'ORDER' in lu.columns else pd.DataFrame(),
        'community_top3': lu.nlargest(3, 'community_presence_proxy')[['ORDER', 'community_presence_proxy']]
                            if 'ORDER' in lu.columns else pd.DataFrame(),
        'openspace_top3': lu.nlargest(3, 'open_space_proxy')[['ORDER', 'open_space_proxy']]
                            if 'ORDER' in lu.columns else pd.DataFrame(),
        'lighting_note': (
            "Direct OSM lighting or surveillance fields are not present in the current "
            "land-use inputs, so the report uses built-environment proxies instead."
        ),
    }

# ── HTML report ───────────────────────────────────────────────────────────────
SAFETY_COEFFS = {
    'WORK mode  c_crash_veh': -0.0782,  'EDU mode c_crash_veh':  0.0056,
    'NONMAND    c_crash_veh': -0.0182,  'WORK dest c_crash':     -0.1863,
    'EDU dest   c_crash':     -0.1995,  'NONMAND dest c_crash':  -0.2361,
}

def write_report(summary, plot_files):
    out_path = os.path.join(DIR_OUT, 'safety_report.html')
    coef_rows = ''.join(
        f"<tr><td>{n}</td>"
        f"<td style='font-weight:bold;color:{'green' if v<-0.01 else 'red' if v>0.001 else 'orange'}'>{v:.4f}</td>"
        f"<td>{'✓ deters travel' if v<-0.01 else '⚠ wrong sign' if v>0.001 else '~ negligible'}</td></tr>"
        for n, v in SAFETY_COEFFS.items()
    )
    plots_html = ''.join(
        f'<h3>{os.path.basename(p).replace("_"," ").replace(".png","").title()}</h3>'
        f'<img src="{os.path.relpath(p,DIR_OUT)}" style="max-width:100%"><br>'
        for p in plot_files if os.path.exists(p)
    )
    narrative = ""
    proxy_html = ""
    try:
        if len(summary) >= 3:
            _base = summary.iloc[0]
            _delta = summary.iloc[2]
            _active = str(_delta.get('Active mode %', '0')).replace('%', '')
            _walk = str(_delta.get('Walk', '0')).replace('%', '')
            _bike = str(_delta.get('Bike', '0')).replace('%', '')
            _da = str(_delta.get('Drive Alone', '0')).replace('%', '')
            narrative = (
                "<div class='box'><strong>What the safety skims are doing:</strong> "
                f"removing them changes active mode share by {_active} percentage points, "
                f"with Walk shifting {_walk} pp, Bike shifting {_bike} pp, and Drive Alone "
                f"shifting {_da} pp relative to the baseline run. The sign and size of the "
                "crash coefficients indicate that the safety skims are working mainly as a "
                "destination deterrent, then secondarily as a mode deterrent.</div>"
            )
    except Exception:
        narrative = ""
    try:
        proxy = build_safety_proxy_summary()
        if proxy:
            def _rows(df, value_col):
                if df is None or df.empty:
                    return '<li>Not available</li>'
                return ''.join(
                    f"<li>Zone {int(r.ORDER)}: {float(getattr(r, value_col)):.2f}</li>"
                    for r in df.itertuples(index=False)
                )
            proxy_html = (
                "<div class='box'><strong>Built-environment safety context:</strong> "
                "the current pipeline does not ingest direct OSM lighting fields, so this "
                "report summarises zone-level safety context using available land-use "
                "proxies for activity frontage, community presence, and open-space access."
                f"<br><br>{proxy['lighting_note']}"
                "<ul>"
                f"<li><strong>Highest activity-frontage proxy zones:</strong><ul>{_rows(proxy.get('frontage_top3'), 'activity_frontage_proxy')}</ul></li>"
                f"<li><strong>Highest community-presence proxy zones:</strong><ul>{_rows(proxy.get('community_top3'), 'community_presence_proxy')}</ul></li>"
                f"<li><strong>Highest open-space proxy zones:</strong><ul>{_rows(proxy.get('openspace_top3'), 'open_space_proxy')}</ul></li>"
                "</ul></div>"
            )
    except Exception:
        proxy_html = ""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Safety Skim Experiment</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;max-width:1400px}}
h1{{background:#2c3e50;color:white;padding:15px;border-radius:4px}}
h2{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:15px 0}}
th{{background:#2c3e50;color:white;padding:8px 12px}}
td{{border:1px solid #ddd;padding:6px 12px}}
tr:nth-child(even){{background:#f9f9f9}}
.box{{background:#eaf4fb;border-left:4px solid #2980b9;padding:12px;margin:10px 0}}</style>
</head><body>
<h1>🛡️ Safety Skim Experiment — SEQ ABM</h1>
<div class="box"><strong>Design:</strong> Baseline includes CRASH + CRASHPT skims in Stage 5.
No-safety excludes them. All else identical.</div>
{narrative}
{proxy_html}
<h2>Summary</h2>{summary.to_html(index=False,border=0,escape=False)}
<h2>Safety Coefficients (baseline model)</h2>
<table><tr><th>Parameter</th><th>Value</th><th>Sign check</th></tr>{coef_rows}</table>
<div class="box">Destination-level crash deterrence (c_crash ≈ −0.19 to −0.24) is stronger
than mode-level (c_crash_veh ≈ −0.01 to −0.08). Travellers avoid crash-prone destination
zones more than crash-prone routes, consistent with zone reputation effects.</div>
<h2>Plots</h2>{plots_html}
<hr><p style="color:#888;font-size:11px">
Baseline: {DIR_BASE} | No-safety: {DIR_NOSA}</p>
</body></html>"""
    with open(out_path,'w') as f: f.write(html)
    print(f"\n  HTML report → {out_path}")
    return out_path

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('='*60)
    print('SAFETY SKIM EXPERIMENT ANALYSIS')
    print(f'  Baseline  : {DIR_BASE}')
    print(f'  No-safety : {DIR_NOSA}')
    print(f'  Output    : {DIR_OUT}')
    print('='*60)

    plots = []
    for name, fn in [
        ('01_mode_shares.png',       fig_mode_shares),
        ('02_distance_dist.png',     fig_distance),
        ('03_active_by_distance.png',fig_active_by_dist),
        ('04_coef_comparison.png',   fig_coef),
        ('05_zone_active_shift.png', fig_zone_shift),
    ]:
        p = os.path.join(PLOTS_DIR, name)
        fn(p); plots.append(p)

    summary = build_summary()
    csv_p = os.path.join(DIR_OUT, 'safety_comparison.csv')
    summary.to_csv(csv_p, index=False)
    print(f'\nSummary:\n{summary.to_string(index=False)}')

    write_report(summary, plots)
    print(f'\n{"="*60}\nDONE — see {DIR_OUT}/safety_report.html\n{"="*60}')
