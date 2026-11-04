"""
generate_person_day_samples.py
==============================
Extract example person-days from the NEWEST completed pipeline outputs so the
zones dashboard (demand_/09_osmx_network_zones_dashboard.py, "Person Day" tab)
and the model-review deck can show a chosen person type living out their
simulated day - and, when the safety counterfactual run exists, the same
person's day with crash skims zeroed, side by side.

Only runs meaningfully AFTER the full model chain has produced stage6 output;
with no output it exits 0 doing nothing (safe to call from the PBS tail like
generate_sa2_pattern_slide.py).

Outputs
-------
  demand_/person_day_samples.json     - {arm: {ptype: [person plans]}} for the
                                        dashboard (small, ~N_PER_PTYPE persons
                                        per ptype per arm)
  abm_safety/model_review_deck.html   - "A day in the life" slide injected
                                        between PERSON_DAY_SLIDE sentinels:
                                        one exemplar per major ptype, with
                                        safety-vs-nosafety timelines side by
                                        side when both arms exist.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_ROOT = os.path.normpath(os.path.join(CODE_DIR, '..'))
OUT_JSON = os.path.join(TRT_ROOT, 'demand_', 'person_day_samples.json')
DECK_HTML = os.path.join(TRT_ROOT, 'abm_safety', 'model_review_deck.html')
ZONE_REF = os.path.join(TRT_ROOT, 'demand_', 'sa2_zones_reference_357.csv')

N_PER_PTYPE = 25
MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
USECOLS = ['person_id', 'row_type', 'seq', 'purpose', 'OTAZ', 'DTAZ',
           'STARTIME', 'ENDTIME', 'mode', 'travtime', 'ptype', 'dist',
           'agep', 'sexp', 'vehrd', 'CARLICENCE']

ARM_PATTERNS = {
    'safety':   ['runs/ga_parallel/sa_bandit_staged__*/stage6_output.csv',
                 'runs/ga_parallel/safety_baseline__*/stage6_output.csv',
                 'runs/*/stage6_output.csv'],
    'nosafety': ['runs/ga_parallel/safety_nosafety__*/stage6_output.csv',
                 'runs/safety_nosafety__*/stage6_output.csv'],
}


def _newest(patterns):
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(os.path.join(CODE_DIR, pat)))
    cands = [c for c in cands if os.path.getsize(c) > 1_000_000]
    return max(cands, key=os.path.getmtime) if cands else None


def _extract_samples(source: str, person_filter: set = None,
                     chunksize: int = 2_000_000) -> dict:
    """{ptype: {person_id: [row dicts]}} - keeps the first N_PER_PTYPE persons
    per ptype whose day contains >= 2 trips (or all persons in person_filter,
    so the nosafety arm can be matched to the SAME people as the safety arm)."""
    picked: dict = {}
    rows: dict = {}
    header = pd.read_csv(source, nrows=0)
    usecols = [c for c in USECOLS if c in header.columns]
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunksize,
                             low_memory=True, on_bad_lines='skip'):
        chunk['person_id'] = chunk['person_id'].astype(str)
        if person_filter is not None:
            sel = chunk[chunk['person_id'].isin(person_filter)]
        else:
            # sample: persons not yet full for their ptype
            sel = chunk
        for pid, grp in sel.groupby('person_id', sort=False):
            pt = str(grp['ptype'].iloc[0]) if 'ptype' in grp.columns else 'Unknown'
            if person_filter is None:
                bucket = picked.setdefault(pt, set())
                if pid not in bucket:
                    if len(bucket) >= N_PER_PTYPE:
                        continue
                    n_trips = int((grp['row_type'] == 'trip').sum())
                    if n_trips < 2:
                        continue
                    bucket.add(pid)
            rows.setdefault(pid, []).append(grp)
        if person_filter is None and picked and \
                all(len(v) >= N_PER_PTYPE for v in picked.values()) and len(picked) >= 6:
            break
    out: dict = {}
    keep = (person_filter if person_filter is not None
            else {p for b in picked.values() for p in b})
    for pid in keep:
        parts = rows.get(pid)
        if not parts:
            continue
        day = pd.concat(parts).sort_values('seq' if 'seq' in parts[0].columns else 'STARTIME')
        pt = str(day['ptype'].iloc[0]) if 'ptype' in day.columns else 'Unknown'
        recs = []
        for _, r in day.iterrows():
            recs.append({
                'row_type': str(r.get('row_type', '')),
                'purpose':  str(r.get('purpose', '')),
                'OTAZ':     int(r['OTAZ']) if pd.notna(r.get('OTAZ')) else None,
                'DTAZ':     int(r['DTAZ']) if pd.notna(r.get('DTAZ')) else None,
                'start':    round(float(r['STARTIME']), 1) if pd.notna(r.get('STARTIME')) else None,
                'end':      round(float(r['ENDTIME']), 1) if pd.notna(r.get('ENDTIME')) else None,
                'mode':     MODE_NAMES.get(int(r['mode']) if pd.notna(r.get('mode')) else 0, ''),
                'dist':     round(float(r['dist']), 1) if pd.notna(r.get('dist')) else None,
            })
        out.setdefault(pt, {})[pid] = recs
    return out


def _fmt_time(minutes):
    if minutes is None:
        return '?'
    h, m = int(minutes // 60) % 24, int(minutes % 60)
    return f'{h:02d}:{m:02d}'


def _timeline_html(recs, color='#1C7293'):
    """Compact horizontal 24h timeline strip for one person-day."""
    segs = []
    for r in recs:
        if r['start'] is None or r['end'] is None:
            continue
        left = max(0.0, min(100.0, r['start'] / 1440.0 * 100))
        width = max(0.4, min(100.0 - left, (r['end'] - r['start']) / 1440.0 * 100))
        if r['row_type'] == 'trip':
            label = f"{r['mode']} {_fmt_time(r['start'])}→{_fmt_time(r['end'])}" \
                    + (f" · {r['dist']}km" if r['dist'] else '')
            bg = '#C85A14'
        else:
            label = f"{r['purpose']} {_fmt_time(r['start'])}–{_fmt_time(r['end'])}"
            bg = color
        segs.append(
            f"<div title=\"{label}\" style=\"position:absolute;left:{left:.2f}%;"
            f"width:{width:.2f}%;top:{2 if r['row_type']=='trip' else 12}px;"
            f"height:{8 if r['row_type']=='trip' else 12}px;background:{bg};"
            f"border-radius:2px;\"></div>")
    return ("<div style='position:relative;height:28px;background:var(--surface-tint-2);"
            "border-radius:4px;'>" + ''.join(segs) + "</div>")


_SLIDE_START = '<!--PERSON_DAY_SLIDE_START-->'
_SLIDE_END = '<!--PERSON_DAY_SLIDE_END-->'


def inject_deck_slide(samples: dict):
    if not os.path.exists(DECK_HTML):
        print(f'[person-day] deck not found: {DECK_HTML}')
        return
    have_nosafety = bool(samples.get('nosafety'))
    cards = []
    for pt in ['FullTimeWorker', 'SchoolChild', 'Retired', 'UniversityStudent']:
        base = samples.get('safety', {}).get(pt) or {}
        if not base:
            continue
        pid = sorted(base)[0]
        rows = [f"<p style='margin:0 0 2px;font-size:0.66rem;color:var(--text-muted);'>"
                f"{pt} · person {pid} · crash skims ON</p>",
                _timeline_html(base[pid])]
        cf = samples.get('nosafety', {}).get(pt, {}).get(pid)
        if cf:
            rows.append("<p style='margin:6px 0 2px;font-size:0.66rem;color:var(--text-muted);'>"
                        "same person · crash skims OFF</p>")
            rows.append(_timeline_html(cf, color='#2E7D6B'))
        cards.append("<div class='card' style='min-height:0;'>" + ''.join(rows) + "</div>")
    note = '' if have_nosafety else (
        "<p class='sub' style='font-size:0.7rem;'>Counterfactual (no-safety) arm not run "
        "yet - rerun after the safety experiment chain completes to see side-by-side days.</p>")
    slide = f"""{_SLIDE_START}
      <section class="slide" id="person-day-slide">
        <h2 class="slide-kicker">Simulation Output · auto-refreshed by generate_person_day_samples.py</h2>
        <h1 class="slide-title">A day in the life: simulated activity plans by person type</h1>
        <div class="body-area" style="gap:1vh;">
          <div class="grid" style="grid-template-columns: 1fr 1fr; gap:0.8em;">{''.join(cards)}</div>
          {note}
          <p class="sub" style="font-family:var(--font-mono);font-size:0.62rem;text-align:center;">
          Interactive person picker: demand_/09_osmx_network_zones_dashboard.py · "Person Day" tab</p>
        </div>
      </section>
      {_SLIDE_END}"""
    with open(DECK_HTML, 'r', encoding='utf-8') as fh:
        html = fh.read()
    pat = re.compile(re.escape(_SLIDE_START) + '.*?' + re.escape(_SLIDE_END), re.DOTALL)
    if pat.search(html):
        html = pat.sub(slide, html)
    else:
        idx = html.rfind('</section>')
        if idx < 0:
            print('[person-day] no </section> in deck - skipped')
            return
        insert_at = idx + len('</section>')
        html = html[:insert_at] + '\n\n' + slide + html[insert_at:]
    with open(DECK_HTML, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f'[person-day] deck slide updated: {DECK_HTML}')


def main():
    src_safety = _newest(ARM_PATTERNS['safety'])
    if not src_safety:
        print('[person-day] no completed stage6 output found - nothing to do')
        return 0
    print(f'[person-day] safety arm: {src_safety}')
    samples = {'safety': _extract_samples(src_safety)}
    n = sum(len(v) for v in samples['safety'].values())
    print(f'[person-day] sampled {n} persons across {len(samples["safety"])} ptypes')

    src_nosafety = _newest(ARM_PATTERNS['nosafety'])
    if src_nosafety:
        print(f'[person-day] nosafety arm: {src_nosafety}')
        wanted = {pid for pt in samples['safety'].values() for pid in pt}
        samples['nosafety'] = _extract_samples(src_nosafety, person_filter=wanted)
    else:
        print('[person-day] nosafety arm not found yet - single-arm samples only')

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(samples, fh)
    print(f'[person-day] samples written: {OUT_JSON}')

    inject_deck_slide(samples)
    return 0


if __name__ == '__main__':
    sys.exit(main())
