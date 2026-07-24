"""
generate_validation_report.py
==============================
Self-contained interactive validation report with:

  - Stage tabs (0–5)
  - Per-segment toggles (work / edu / nonmand for stage 5)
  - Sortable mode-share tables
  - Run-picker for side-by-side comparison of two runs
  - validation_summary.json for the deck gallery

No CDN — all CSS and JS are inlined. All images are base64-embedded.
The output is a single HTML file that works offline and survives CSP blocks.

Usage:
    python generate_validation_report.py                          # newest run
    python generate_validation_report.py --run runs/ga_parallel/sa_bandit__guided03
    python generate_validation_report.py --run A --compare B     # side-by-side
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import glob
import io
import json
import os
import pickle
import re
import sys
from pathlib import Path

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE_NAMES = {
    0: 'CDAP (Activity Pattern)',
    1: 'Activity Frequency',
    2: 'Day Duration Budget',
    3: 'Activity Duration',
    4: 'Schedule Placement',
    5: 'Destination + Mode',
    '5b': 'Trip Stops',
    '6': 'Trip Mode (Finalisation)',
}

SEGMENTS = ['work', 'edu', 'nonmand']
MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}


def find_newest_run(base_dir: str = None) -> str | None:
    if base_dir is None:
        base_dir = os.path.join(CODE_DIR, 'runs')
    cands = []
    for d in glob.glob(os.path.join(base_dir, '*', '*')):
        if not os.path.isdir(d):
            continue
        has = glob.glob(os.path.join(d, 'plots', '*.png')) or \
              glob.glob(os.path.join(d, 'ga_hhts_validation', 'plots', 'validation', '*.png')) or \
              glob.glob(os.path.join(d, 'stage_reports', '*.html'))
        if has:
            cands.append((os.path.getmtime(d), d))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


def collect_run_artifacts(run_dir: str) -> dict:
    """Collect all validation artifacts from a run directory."""
    art = {
        'run_dir': run_dir,
        'run_name': os.path.basename(run_dir),
        'stages': {},
        'coefficients': {},
        'timings': None,
        'outline': None,
        'progress': None,
        'summary': {},
    }

    # Load ga_progress.json
    gp = os.path.join(run_dir, 'ga_progress.json')
    if os.path.exists(gp):
        with open(gp) as f:
            art['progress'] = json.load(f)

    # Load run_outline.txt
    ro = os.path.join(run_dir, 'run_outline.txt')
    if os.path.exists(ro):
        with open(ro) as f:
            art['outline'] = f.read().strip()

    # Load timings.csv
    tc = os.path.join(run_dir, 'timings.csv')
    if os.path.exists(tc):
        try:
            art['timings'] = list(csv.DictReader(open(tc)))
        except Exception:
            pass

    # Coefficient tables
    for stage_file in glob.glob(os.path.join(run_dir, 'stage*_coefficients.csv')):
        sn = re.search(r'stage(\d+|[\w]+)_coefficients', os.path.basename(stage_file))
        if sn:
            stage_key = sn.group(1)
            try:
                art['coefficients'][stage_key] = list(csv.DictReader(open(stage_file)))
            except Exception:
                pass

    html_coef = glob.glob(os.path.join(run_dir, 'stage*_coefficients.html'))
    for hf in html_coef:
        sn = re.search(r'stage(\d+|[\w]+)_coefficients', os.path.basename(hf))
        if sn:
            art['coefficients'][f"{sn.group(1)}_html"] = hf

    # Per-stage reports
    for report_dir in ['stage_reports', 'ga_reports']:
        rd = os.path.join(run_dir, report_dir)
        if os.path.isdir(rd):
            for f in glob.glob(os.path.join(rd, '*.html')):
                bn = os.path.basename(f)
                art['stages'].setdefault(bn, {})['html'] = f
            for f in glob.glob(os.path.join(rd, '*.csv')):
                bn = os.path.basename(f)
                art['stages'].setdefault(bn, {})['csv'] = f
            for f in glob.glob(os.path.join(rd, '*.png')):
                bn = os.path.basename(f)
                art['stages'].setdefault(bn, {})['png'] = f

    # Validation plots
    for val_path in glob.glob(os.path.join(run_dir, 'plots', 'validation', '*.png')):
        bn = os.path.basename(val_path)
        art['stages'].setdefault(f'validation_{bn}', {})['png'] = val_path
    for val_path in glob.glob(os.path.join(run_dir, 'ga_hhts_validation', 'plots', 'validation', '*.png')):
        bn = os.path.basename(val_path)
        art['stages'].setdefault(f'validation_{bn}', {})['png'] = val_path

    # Location choice plots
    for lc_path in glob.glob(os.path.join(run_dir, 'location_choice', '*.png')):
        bn = os.path.basename(lc_path)
        art['stages'].setdefault(f'location_{bn}', {})['png'] = lc_path

    # GA convergence plots
    for ga_path in glob.glob(os.path.join(run_dir, 'plots', 'ga_*.png')):
        bn = os.path.basename(ga_path)
        art['stages'].setdefault(f'ga_{bn}', {})['png'] = ga_path

    # Load selected_features_summary.json
    sfs = os.path.join(run_dir, 'selected_features_summary.json')
    if os.path.exists(sfs):
        with open(sfs) as f:
            art['selected_features'] = json.load(f)

    # Extract headline metrics for the deck gallery
    _extract_headline_metrics(art)

    return art


def _extract_headline_metrics(art: dict):
    """Extract headline metrics from coefficient CSVs and logs."""
    s = art['summary']

    # From timings
    if art['timings']:
        for row in art['timings']:
            step = row.get('step', '')
            status = row.get('status', '')
            wall = row.get('wall', '')
            if 'all stages' in step.lower():
                s['ga_wall_time'] = wall
                s['ga_status'] = status

    # From progress
    if art['progress']:
        s['completed_stages'] = art['progress'].get('completed_stages', [])
        s['failed'] = art['progress'].get('failed', None)

    # Mode share and rho2 from stage 5 coefficients
    s5_coefs = art['coefficients'].get('5', [])
    mode_rho2 = {}
    for row in s5_coefs:
        seg = row.get('segment', row.get('SEGMENT', ''))
        rho2 = row.get('rho_sq', row.get('RHO2', ''))
        if seg and rho2:
            try:
                mode_rho2[seg.lower()] = float(rho2)
            except (ValueError, TypeError):
                pass
    if mode_rho2:
        s['stage5_rho2'] = mode_rho2

    # Shadow price statistics from logs
    try:
        plog = os.path.join(art['run_dir'], 'pipeline_run.log')
        if os.path.exists(plog):
            with open(plog) as f:
                content = f.read()
            for seg in ['edu', 'nonmand', 'work']:
                m = re.search(
                    rf'_fit_destination_zone_shadow_prices:\s+(\d+)/(\d+)\s+zones\s+adjusted.*?max_abs=([\d.]+).*?remaining_fail_zones=(\d+)',
                    content, re.DOTALL
                )
                # Try per-segment
                pattern = rf'\[{seg}\].*?_fit_destination_zone_shadow_prices:\s+(\d+)/(\d+)\s+zones\s+adjusted.*?max_abs=([\d.]+).*?remaining_fail_zones=(\d+)'
                m = re.search(pattern, content, re.DOTALL)
                if not m:
                    # Try without segment prefix
                    m = re.search(
                        rf'_fit_destination_zone_shadow_prices:\s+(\d+)/(\d+)\s+zones\s+adjusted.*?max_abs=([\d.]+).*?remaining_fail_zones=(\d+)',
                        content
                    )
                if m:
                    s.setdefault('shadow_prices', {})[seg] = {
                        'adjusted': int(m.group(1)),
                        'total': int(m.group(2)),
                        'max_abs': float(m.group(3)),
                        'fail_zones': int(m.group(4)),
                    }
    except Exception:
        pass

    art['summary'] = s


def encode_image_base64(path: str, max_size: int = 400_000) -> str | None:
    """Encode a PNG to a base64 data URI, with size limit."""
    if not os.path.exists(path):
        return None
    try:
        size = os.path.getsize(path)
        if size > max_size:
            return None  # Skip large files
        with open(path, 'rb') as f:
            return f'data:image/png;base64,{base64.b64encode(f.read()).decode()}'
    except Exception:
        return None


def build_html_report(run_a: dict, run_b: dict = None,
                      output_path: str = None) -> str:
    """Build a self-contained interactive HTML validation report."""

    def _stage_tab_content(run: dict, stage: int) -> str:
        """Generate HTML content for one stage tab."""
        parts = []

        # Coefficients table
        coefs = run['coefficients'].get(str(stage), [])
        if stage == 5:
            coefs.extend(run['coefficients'].get('5_mode', []))
            coefs.extend(run['coefficients'].get('5_dest', []))

        if stage == 5:
            parts.append('<div class="segment-toggles">')
            parts.append('<button class="seg-btn active" onclick="filterSeg(this,\'all\')">All</button>')
            for seg in SEGMENTS:
                parts.append(f'<button class="seg-btn" onclick="filterSeg(this,\'{seg}\')">{seg.title()}</button>')
            parts.append('</div>')

        # Mode share table for stage 5
        if stage == 5:
            parts.append('<h4>Mode Shares</h4>')
            parts.append('<table class="sortable mode-table" id="mode-table">')
            parts.append('<thead><tr><th onclick="sortTable(0)">Segment</th>')
            for m in MODE_NAMES.values():
                parts.append(f'<th onclick="sortTable({list(MODE_NAMES.values()).index(m)+1})">{m}</th>')
            parts.append('<th>Obs | Pred</th></tr></thead><tbody>')

            for seg in SEGMENTS:
                seg_rows = [r for r in coefs if r.get('segment', '').lower() == seg]
                mode_shares = {}
                for r in seg_rows:
                    for mode_name in MODE_NAMES.values():
                        key = f'pred_share_{mode_name}'
                        if key in r:
                            mode_shares[mode_name] = float(r[key])
                if mode_shares:
                    parts.append(f'<tr class="seg-{seg}"><td>{seg.title()}</td>')
                    for m in MODE_NAMES.values():
                        parts.append(f'<td>{mode_shares.get(m, 0):.3f}</td>')
                    parts.append('<td>--</td></tr>')
            parts.append('</tbody></table>')

        # Coefficient details
        if coefs:
            parts.append('<h4>Coefficients</h4>')
            parts.append('<table class="sortable coef-table"><thead><tr>')
            headers = list(coefs[0].keys()) if coefs else []
            for i, h in enumerate(headers):
                parts.append(f'<th onclick="sortTable({i})">{h}</th>')
            parts.append('</tr></thead><tbody>')
            for row in coefs:
                seg_cls = f'seg-{row.get("segment", "").lower()}' if 'segment' in row else ''
                parts.append(f'<tr class="{seg_cls}">')
                for h in headers:
                    v = row.get(h, '')
                    try:
                        v = f"{float(v):.4f}"
                    except (ValueError, TypeError):
                        pass
                    parts.append(f'<td>{v}</td>')
                parts.append('</tr>')
            parts.append('</tbody></table>')

        # Plots for this stage
        stage_plots = []
        for key, val in run['stages'].items():
            if 'png' in val and f'stage{stage}' in key.lower():
                stage_plots.append(val['png'])
            elif 'png' in val and f'val_stage{stage}' in key.lower():
                stage_plots.append(val['png'])

        for plot_path in sorted(set(stage_plots))[:12]:
            img_data = encode_image_base64(plot_path)
            if img_data:
                bn = os.path.basename(plot_path)
                parts.append(f'<figure><img src="{img_data}" loading="lazy" '
                             f'alt="{bn}"><figcaption>{bn}</figcaption></figure>')

        return '\n'.join(parts)

    # ── Build HTML ────────────────────────────────────────────────────────
    html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SEQ ABM Validation Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#f5f6fa;color:#2c3e50}}
.header{{background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;padding:18px 24px}}
.header h1{{font-size:1.3em}} .header p{{opacity:.8;font-size:.85em;margin-top:4px}}
.tabs{{display:flex;background:#fff;border-bottom:3px solid #3498db;padding:0 12px;position:sticky;top:0;z-index:10}}
.tab-btn{{padding:12px 20px;border:none;background:none;cursor:pointer;font-size:.9em;color:#7f8c8d;border-bottom:3px solid transparent;margin-bottom:-3px;transition:all .2s}}
.tab-btn:hover{{color:#2c3e50}}
.tab-btn.active{{color:#3498db;border-bottom-color:#3498db;font-weight:600}}
.tab-content{{display:none;padding:20px}}
.tab-content.active{{display:block}}
.segment-toggles{{margin:10px 0;display:flex;gap:6px}}
.seg-btn{{padding:6px 14px;border:1px solid #ddd;border-radius:20px;background:#fff;cursor:pointer;font-size:.82em;transition:all .2s}}
.seg-btn.active,.seg-btn:hover{{background:#3498db;color:#fff;border-color:#3498db}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.85em}}
th{{background:#2c3e50;color:#fff;padding:10px 12px;text-align:left;cursor:pointer;position:sticky;top:0;user-select:none}}
th:hover{{background:#34495e}}
td{{padding:8px 12px;border-bottom:1px solid #eee}}
tr:hover{{background:#f0f4ff}}
.run-picker{{display:flex;gap:10px;align-items:center;padding:14px 24px;background:#fff;border-bottom:1px solid #eee;flex-wrap:wrap}}
.run-picker input{{flex:1;min-width:200px;padding:8px 12px;border:1px solid #ddd;border-radius:6px;font-size:.9em}}
.run-picker button{{padding:8px 18px;background:#3498db;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:.9em}}
.run-picker button:hover{{background:#2980b9}}
.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.comparison-panel{{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.comparison-panel h3{{color:#3498db;margin-bottom:10px;font-size:1em}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin:12px 0}}
.metric-card{{background:#fff;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.06);border-left:4px solid #3498db}}
.metric-card .label{{font-size:.75em;color:#7f8c8d;text-transform:uppercase;letter-spacing:.5px}}
.metric-card .value{{font-size:1.4em;font-weight:700;color:#2c3e50;margin-top:2px}}
.metric-card.warn{{border-left-color:#e67e22}}
.metric-card.bad{{border-left-color:#e74c3c}}
figure{{margin:16px 0;max-width:100%}}
figure img{{max-width:100%;border:1px solid #eee;border-radius:4px}}
figcaption{{font-size:.78em;color:#888;margin-top:4px;text-align:center}}
.status-ok{{color:#27ae60}} .status-fail{{color:#e74c3c}}
tr.hidden{{display:none}}
.footer{{text-align:center;padding:20px;color:#888;font-size:.78em}}
@media(max-width:768px){{.comparison{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<div class="header">
  <h1>SEQ ABM — Validation Report</h1>
  <p>Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
''']
    # Add header with run info
    if run_a.get('run_name'):
        rn = run_a['run_name']
        html_parts.append(f'<div class="run-info" style="padding:8px 24px;background:#ecf0f1;font-size:.85em">Run: <b>{rn}</b> | GA stages completed: {run_a.get("summary",{}).get("completed_stages",[])} | Wall time: {run_a.get("summary",{}).get("ga_wall_time","?")}</div>')

    # Tabs
    html_parts.append('<div class="tabs">')
    for stage_num in sorted([k for k in STAGE_NAMES if isinstance(k, int)]):
        name = STAGE_NAMES[stage_num]
        active = 'active' if stage_num == 5 else ''
        html_parts.append(f'<button class="tab-btn {active}" onclick="showTab({stage_num})">Stage {stage_num}<br><small>{name}</small></button>')
    html_parts.append('</div>')

    # Tab content
    for stage_num in sorted([k for k in STAGE_NAMES if isinstance(k, int)]):
        active = 'active' if stage_num == 5 else ''
        html_parts.append(f'<div class="tab-content {active}" id="tab-{stage_num}">')
        html_parts.append(f'<h2>{STAGE_NAMES[stage_num]}</h2>')

        if stage_num == 5:
            # Summary metrics
            s5_sum = run_a['summary'].get('stage5_rho2', {})
            if s5_sum:
                html_parts.append('<div class="metric-grid">')
                for seg in SEGMENTS:
                    rho = s5_sum.get(seg, None)
                    if rho is not None:
                        cls = '' if rho > 0.3 else 'warn' if rho > 0.15 else 'bad'
                        html_parts.append(f'<div class="metric-card {cls}"><div class="label">{seg} rho&sup2;</div><div class="value">{rho:.3f}</div></div>')
                html_parts.append('</div>')

            # Shadow price summary
            sp = run_a['summary'].get('shadow_prices', {})
            if sp:
                html_parts.append('<div class="metric-grid">')
                for seg in SEGMENTS:
                    if seg in sp:
                        s = sp[seg]
                        fail_pct = s['fail_zones'] / max(s['total'], 1) * 100
                        cls = '' if fail_pct < 20 else 'warn' if fail_pct < 50 else 'bad'
                        html_parts.append(f'<div class="metric-card {cls}"><div class="label">{seg} shadow price fail %</div><div class="value">{fail_pct:.0f}%</div><div style="font-size:.7em;color:#888">{s["adjusted"]}/{s["total"]} adjusted, max={s["max_abs"]}</div></div>')
                html_parts.append('</div>')

        # Comparison mode
        if run_b:
            html_parts.append('<div class="comparison">')
            html_parts.append(f'<div class="comparison-panel"><h3>Run A: {run_a["run_name"]}</h3>')
            html_parts.append(_stage_tab_content(run_a, stage_num))
            html_parts.append('</div>')
            html_parts.append(f'<div class="comparison-panel"><h3>Run B: {run_b["run_name"]}</h3>')
            html_parts.append(_stage_tab_content(run_b, stage_num))
            html_parts.append('</div></div>')
        else:
            html_parts.append(_stage_tab_content(run_a, stage_num))

        html_parts.append('</div>')

    # JavaScript
    html_parts.append('''
<script>
function showTab(n){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.querySelectorAll('.tab-btn')[n].classList.add('active');
  document.getElementById('tab-'+n).classList.add('active');
}
function filterSeg(btn,seg){
  document.querySelectorAll('.seg-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('tr[class*="seg-"]').forEach(r=>{
    if(seg==='all') r.classList.remove('hidden');
    else r.classList.toggle('hidden',!r.className.includes('seg-'+seg));
  });
}
function sortTable(n){
  var table=document.querySelector('.sortable'),rows,i,x,y,dir=table.getAttribute('data-sort-dir')||'asc';
  rows=Array.from(table.tBodies[0].rows);
  dir=(dir==='asc')?'desc':'asc';
  table.setAttribute('data-sort-dir',dir);
  rows.sort(function(a,b){
    x=a.cells[n].textContent.trim();y=b.cells[n].textContent.trim();
    var xn=parseFloat(x),yn=parseFloat(y);
    if(!isNaN(xn)&&!isNaN(yn)) return dir==='asc'?xn-yn:yn-xn;
    return dir==='asc'?x.localeCompare(y):y.localeCompare(x);
  });
  rows.forEach(r=>table.tBodies[0].appendChild(r));
}
</script>
<div class="footer">SEQ ABM Validation Report &mdash; Self-contained | No external dependencies</div>
</body></html>''')

    html = '\n'.join(html_parts)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Validation report -> {output_path}")

    return html


def write_validation_summary(run: dict, output_path: str) -> None:
    """Write validation_summary.json for the deck gallery."""
    summary = {
        'run_name': run['run_name'],
        'generated': datetime.datetime.now().isoformat(),
        'stages_completed': run.get('summary', {}).get('completed_stages', []),
        'ga_wall_time': run.get('summary', {}).get('ga_wall_time'),
        'ga_status': run.get('summary', {}).get('ga_status'),
        'stage5_rho2': run.get('summary', {}).get('stage5_rho2', {}),
        'shadow_prices': run.get('summary', {}).get('shadow_prices', {}),
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Validation summary -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate interactive validation report')
    parser.add_argument('--run', type=str, help='Run directory (default: newest)')
    parser.add_argument('--compare', type=str, help='Second run directory for comparison')
    parser.add_argument('--output', type=str, help='HTML output path')
    parser.add_argument('--summary', type=str, help='JSON summary output path')
    args = parser.parse_args()

    run_dir_a = args.run or find_newest_run()
    if not run_dir_a:
        print("No run directory found and no --run specified.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading run A: {run_dir_a}")
    run_a = collect_run_artifacts(run_dir_a)

    run_b = None
    if args.compare:
        print(f"Loading run B: {args.compare}")
        run_b = collect_run_artifacts(args.compare)

    out_html = args.output or os.path.join(run_dir_a, 'validation_report_interactive.html')
    build_html_report(run_a, run_b, output_path=out_html)

    out_json = args.summary or os.path.join(run_dir_a, 'validation_summary.json')
    write_validation_summary(run_a, out_json)


if __name__ == '__main__':
    main()
