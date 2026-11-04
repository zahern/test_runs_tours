#!/usr/bin/env python3
"""ActivitySim-style dashboard for the HHTS re-prediction validation.

After a run has been estimated, the fitted stages are applied back to the
HHTS persons, tours and trips they were estimated on (and to a held-out
person split) and the predicted behaviour is compared with the observed
behaviour, stage by stage.  That comparison is produced by
``validate_hhts.run_validation``, which writes one figure per stage plus a
``validation_metrics.json`` summary.

This module turns the same artifacts into the pages an ActivitySim analyst
expects to find in a SimWrapper dashboard:

  * an at-a-glance sheet of observed-versus-predicted fit indicators,
  * tour-mode-share, activity-frequency and destination-fit charts,
  * a self-contained gallery of every stage figure (no CDN, base64 images),
  * SimWrapper YAML tabs so the study area maps and the validation charts
    sit side by side in one ``simwrapper open asim`` session.

Inputs (written by validate_hhts):
    <validation_dir>/validation_metrics.json
    <validation_dir>/plots/validation/*.png
    <validation_dir>/plots/validation/stage5_mode_shares_calibrated.csv (optional)

Outputs (default: <run>/summarize):
    dashboard-2-validation.yaml   SimWrapper tab: charts + link panel
    topsheet-validation.yaml      at-a-glance sheet
    val_kpis.csv                  one row per scalar fit metric
    val_mode_share.csv            tour mode shares, observed vs predicted
    val_mode_<segment>.csv        wide per-segment slices for the bar charts
    val_purpose_means.csv         activity frequency by purpose
    val_destination_fit.csv       destination and assignment fit by segment
    val_tod_profile.csv           hourly departure profile, observed vs predicted
    val_trip_length.csv           tour-length distribution by distance band
    validation_gallery.html       self-contained figure gallery

Usage:
    python build_validation_dashboard.py                        # newest run
    python build_validation_dashboard.py runs/ga_parallel/<tag>  # one run
    python build_validation_dashboard.py --validation-dir <dir>
    python build_validation_dashboard.py --self-test            # fixture

Running it twice is safe: every output file is replaced.
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import html
import json
import math
import os
import sys
import tempfile

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

MODE_NAMES = {1: 'Drive alone', 2: 'Shared ride', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
SEGMENTS = {'work': 'Work', 'edu': 'Education', 'nonmand': 'Non-mandatory'}
STAGE_LABELS = {
    0: 'CDAP',
    1: 'Activity frequency',
    2: 'Day budget',
    3: 'Activity duration',
    4: 'Timing',
    5: 'Mode and destination',
    '5': 'Mode and destination',
    '5b': 'Trip destination',
    'profiles': 'Time of day and trip length',
}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# artifact discovery and normalisation
# --------------------------------------------------------------------------- #
def find_validation_dir(run_dir: str):
    """Return the directory holding the validation artifacts, or None."""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    direct = [
        os.path.join(run_dir, 'ga_hhts_validation'),
        run_dir,
    ]
    for cand in direct:
        if os.path.isfile(os.path.join(cand, 'validation_metrics.json')):
            return cand
        if os.path.isdir(os.path.join(cand, 'plots', 'validation')):
            return cand
    for pattern in ('*/validation_metrics.json',
                    '*/*/validation_metrics.json',
                    '*/plots/validation',
                    '*/*/plots/validation'):
        hits = sorted(glob.glob(os.path.join(run_dir, pattern)))
        if hits:
            candidate = hits[0]
            if candidate.endswith('validation_metrics.json'):
                return os.path.dirname(candidate)
            return os.path.dirname(os.path.dirname(candidate))
    return None


def load_metrics(validation_dir: str) -> list:
    """Load the per-stage validation summary; empty list when absent."""
    path = os.path.join(validation_dir, 'validation_metrics.json')
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
    except Exception as exc:                      # corrupt or mid-write file
        log(f'  [validation] could not read {path}: {exc}')
        return []
    if isinstance(payload, dict):
        payload = payload.get('stages') or payload.get('results') or []
    return [r for r in payload if isinstance(r, dict)]


def _num(value):
    """Coerce to a finite float, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _round(value, digits: int = 4):
    """Round for publication-ready CSV output; None passes through."""
    return None if value is None else round(value, digits)


def _stage_label(stage) -> str:
    if stage in STAGE_LABELS:
        return STAGE_LABELS[stage]
    try:
        return STAGE_LABELS.get(int(stage), f'Stage {stage}')
    except (TypeError, ValueError):
        return f'Stage {stage}'


def _sort_mode(key):
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def scalar_rows(results: list) -> list:
    """One row per finite scalar metric, with the stage it belongs to."""
    rows = []
    for res in results:
        metrics = res.get('metrics') or {}
        if not isinstance(metrics, dict):
            continue
        label = _stage_label(res.get('stage'))
        for key, value in metrics.items():
            number = _num(value)
            if number is None:
                continue
            rows.append({'stage': label, 'metric': key,
                         'value': _round(number, 6)})
    return rows


def mode_share_rows(results: list, validation_dir: str) -> list:
    """Long table of observed and predicted tour mode shares (percent)."""
    rows = []
    for res in results:
        metrics = res.get('metrics') or {}
        if not isinstance(metrics, dict):
            continue
        for seg, seg_label in SEGMENTS.items():
            obs = metrics.get(f'{seg}_mode_obs')
            pred = metrics.get(f'{seg}_mode_pred')
            if not isinstance(obs, dict) or not isinstance(pred, dict):
                continue
            for key in sorted(obs, key=_sort_mode):
                o = _num(obs.get(key))
                p = _num(pred.get(key))
                if o is None and p is None:
                    continue
                try:
                    mode = MODE_NAMES[int(key)]
                except (KeyError, TypeError, ValueError):
                    mode = str(key)
                rows.append({
                    'segment': seg_label,
                    'mode': mode,
                    'observed_pct': _round(o),
                    'predicted_pct': _round(p),
                    'delta_pp': None if (o is None or p is None)
                    else _round(p - o),
                })
    if rows:
        return rows

    # Fall back to the calibrated table written by stage 5, if present.
    for pattern in ('plots/validation/stage5_mode_shares_calibrated.csv',
                    'stage5_mode_shares_calibrated.csv'):
        path = os.path.join(validation_dir, pattern)
        if not os.path.isfile(path):
            continue
        with open(path, 'r', encoding='utf-8', newline='') as fh:
            for rec in csv.DictReader(fh):
                seg = SEGMENTS.get(str(rec.get('segment', '')).strip(),
                                   str(rec.get('segment', '')).strip())
                o = _num(rec.get('observed_pct'))
                p = _num(rec.get('predicted_pct'))
                rows.append({
                    'segment': seg,
                    'mode': str(rec.get('mode', '')).strip(),
                    'observed_pct': _round(o),
                    'predicted_pct': _round(p),
                    'delta_pp': None if (o is None or p is None)
                    else _round(p - o),
                })
        if rows:
            return rows
    return rows


def purpose_rows(results: list) -> list:
    """Activity-frequency comparison by purpose (stage 1)."""
    rows = []
    for res in results:
        metrics = res.get('metrics') or {}
        for rec in (metrics.get('purpose_means') or []):
            if not isinstance(rec, dict):
                continue
            rows.append({
                'purpose': str(rec.get('purpose', '')),
                'observed': _round(_num(rec.get('observed'))),
                'predicted': _round(_num(rec.get('predicted'))),
            })
    return rows


def destination_rows(results: list) -> list:
    """Destination and assignment fit records, one row per segment/metric."""
    rows = []
    keys = {
        'tour_level_destination_fit': 'Tour destination',
        'trip_level_destination_fit': 'Trip destination',
        'work_edu_assignment_fit': 'Assignment',
    }
    for res in results:
        metrics = res.get('metrics') or {}
        for key, kind in keys.items():
            for rec in (metrics.get(key) or []):
                if not isinstance(rec, dict):
                    continue
                seg = str(rec.get('segment', '')).strip()
                rows.append({
                    'kind': kind,
                    'segment': SEGMENTS.get(seg, seg),
                    'mae_share': _round(_num(rec.get('mae_share'))),
                    'js_divergence': _round(_num(rec.get('js_divergence'))),
                    'top_overlap_pct': _round(_num(
                        rec.get('top25_overlap_pct',
                                rec.get('top10_overlap_pct'))), 2),
                    'n': _num(rec.get('n_tours')),
                })
    return rows


def figure_paths(results: list, validation_dir: str) -> list:
    """(stage label, absolute path) for every validation figure on disk."""
    def key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    found, seen = [], set()
    for res in results:
        for raw in (res.get('plots') or []):
            if not isinstance(raw, str):
                continue
            path = os.path.normpath(
                raw if os.path.isabs(raw)
                else os.path.join(validation_dir, raw))
            if os.path.isfile(path) and key(path) not in seen:
                found.append((_stage_label(res.get('stage')), path))
                seen.add(key(path))
    # Pick up any figure in the validation folder that the metrics file did
    # not list (older runs, or a relative path that no longer resolves).
    for path in sorted(glob.glob(
            os.path.join(validation_dir, 'plots', 'validation', '*.png'))):
        path = os.path.normpath(path)
        if key(path) not in seen:
            found.append((None, path))
            seen.add(key(path))
    return found


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: list, columns: list) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if row.get(k) is None else row.get(k))
                             for k in columns})


def write_per_segment_mode_csvs(out_dir: str, rows: list) -> list:
    """Wide per-segment mode-share files for the SimWrapper bar panels."""
    written = []
    for seg_label in dict.fromkeys(row['segment'] for row in rows):
        subset = [r for r in rows if r['segment'] == seg_label]
        name = 'val_mode_%s.csv' % seg_label.lower().replace(' ', '_') \
            .replace('-', '_')
        write_csv(os.path.join(out_dir, name), subset,
                  ['mode', 'observed_pct', 'predicted_pct'])
        written.append((seg_label, name))
    return written


def _kpi_cards(kpis: list) -> str:
    cards = []
    for card in kpis:
        delta = card.get('delta')
        delta_html = ''
        if delta:
            tone = card.get('tone', 'neutral')
            delta_html = f'<div class="delta {tone}">{html.escape(delta)}</div>'
        cards.append(
            '<div class="card"><div class="label">%s</div>'
            '<div class="value">%s</div>%s<div class="note">%s</div></div>'
            % (html.escape(card['label']), html.escape(card['value']),
               delta_html, html.escape(card.get('note', ''))))
    return '<div class="cards">%s</div>' % ''.join(cards)


def _figure_block(stage: str, path: str) -> str:
    with open(path, 'rb') as fh:
        encoded = base64.b64encode(fh.read()).decode('ascii')
    caption = os.path.splitext(os.path.basename(path))[0]
    caption = caption.replace('_', ' ').replace('val ', '').title()
    return ('<figure><img alt="%s" src="data:image/png;base64,%s">'
            '<figcaption>%s</figcaption></figure>'
            % (html.escape(caption), encoded, html.escape(caption)))


def write_gallery(out_path: str, title: str, kpis: list, figures: list,
                  notes: str = '') -> str:
    """Self-contained (no CDN, base64 images) observed-vs-predicted gallery."""
    by_stage = []
    for stage, path in figures:
        if by_stage and by_stage[-1][0] == stage:
            by_stage[-1][1].append(path)
        else:
            by_stage.append((stage, [path]))

    sections = []
    for stage, paths in by_stage:
        heading = 'Observed versus predicted' if stage is None else stage
        body = ''.join(_figure_block(stage or '', p) for p in paths)
        sections.append('<h2>%s</h2><div class="grid">%s</div>'
                        % (html.escape(heading), body))

    doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --navy:#003B71; --blue:#0077C8; --ink:#1d2733; --line:#dde3ea; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px 32px 56px; font:15px/1.45 -apple-system,
          "Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
          background:#f7f9fb; }}
  h1 {{ margin:0 0 4px; font-size:24px; color:var(--navy); }}
  .sub {{ margin:0 0 22px; color:#5C6773; font-size:14px; }}
  h2 {{ margin:34px 0 12px; font-size:18px; color:var(--navy);
        border-bottom:2px solid var(--line); padding-bottom:6px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
            gap:14px; margin:18px 0 8px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:8px;
           padding:14px 16px; box-shadow:0 1px 2px rgba(16,42,67,.05); }}
  .label {{ font-size:12px; letter-spacing:.04em; text-transform:uppercase;
            color:#5C6773; }}
  .value {{ font-size:26px; font-weight:650; color:var(--navy); margin-top:4px; }}
  .delta {{ font-size:13px; margin-top:2px; }}
  .delta.good {{ color:#1e7b3c; }} .delta.warn {{ color:#a8560b; }}
  .delta.bad {{ color:#a3271b; }}
  .note {{ font-size:12px; color:#6b7684; margin-top:4px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
           gap:18px; }}
  figure {{ margin:0; background:#fff; border:1px solid var(--line);
            border-radius:8px; padding:10px; }}
  figure img {{ width:100%; height:auto; display:block; }}
  figcaption {{ font-size:12px; color:#5C6773; margin-top:8px; }}
  .notes {{ background:#fff; border-left:4px solid var(--blue); padding:12px 16px;
            border-radius:6px; font-size:13.5px; color:#33404f; margin:18px 0 0; }}
  footer {{ margin-top:36px; font-size:12px; color:#7a8593; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="sub">Fitted stages applied back to the HHTS persons, tours and trips
they were estimated on; predicted behaviour compared with observed.</p>
{cards}
{notes}
{sections}
<footer>Generated by build_validation_dashboard.py from the validation
artifacts of this run.</footer>
</body></html>
""".format(title=html.escape(title), cards=_kpi_cards(kpis) if kpis else '',
           notes=('<div class="notes">%s</div>' % notes) if notes else '',
           sections=''.join(sections) or
           '<p class="sub">No validation figures found.</p>')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    return out_path


def write_dashboard(out_dir: str, run_label: str, mode_segments: list,
                    has_purpose: bool, has_destination: bool,
                    has_gallery: bool, has_tod: bool = False,
                    has_length: bool = False) -> str:
    """SimWrapper tab mirroring the ActivitySim summary/validation layout."""
    rows = []

    if mode_segments:
        rows.append({'row1': [
            {'type': 'bar', 'title': 'Tour mode share — %s' % seg,
             'dataset': name, 'x': 'mode',
             'columns': ['observed_pct', 'predicted_pct'],
             'yAxisName': 'Share (%)', 'width': 1}
            for seg, name in mode_segments
        ]})

    row2 = []
    if has_purpose:
        row2.append({'type': 'bar', 'title': 'Activity frequency by purpose',
                     'dataset': 'val_purpose_means.csv', 'x': 'purpose',
                     'columns': ['observed', 'predicted'],
                     'yAxisName': 'Mean activities per person', 'width': 1})
    if has_destination:
        row2.append({'type': 'bar', 'title': 'Destination fit by segment',
                     'dataset': 'val_destination_fit.csv', 'x': 'segment',
                     'columns': ['mae_share', 'js_divergence'],
                     'yAxisName': 'Fit error', 'width': 1})
    if row2:
        rows.append({'row2': row2})

    if has_tod:
        rows.append({'row3': [
            {'type': 'bar', 'title': 'Trip departures by hour',
             'dataset': 'val_tod_profile.csv', 'x': 'hour',
             'columns': ['departures_obs', 'departures_pred'],
             'yAxisName': 'Share of departures (%)', 'width': 1},
            {'type': 'bar', 'title': 'First departure of the day',
             'dataset': 'val_tod_profile.csv', 'x': 'hour',
             'columns': ['first_obs', 'first_pred'],
             'yAxisName': 'Share of people (%)', 'width': 1},
        ]})
    if has_length:
        rows.append({'row4': [
            {'type': 'bar', 'title': 'Tour length distribution (km bands)',
             'dataset': 'val_trip_length.csv', 'x': 'bin',
             'columns': ['observed_pct', 'predicted_pct'],
             'yAxisName': 'Share of tours (%)', 'width': 2},
        ]})

    if has_gallery:
        rows.append({'row5': {'type': 'text',
                              'title': 'Observed versus predicted figures',
                              'content': (
                                  'Open [validation_gallery.html]'
                                  '(validation_gallery.html) for the full '
                                  'observed-versus-predicted gallery: one '
                                  'panel per stage, in-sample and held-out.')}})

    dash = {
        'header': {
            'tab': 'HHTS validation',
            'title': run_label,
            'description': 'Observed versus predicted behaviour on the HHTS '
                           're-prediction (ActivitySim-style summaries)',
        },
        'layout': {key: value for row in rows for key, value in row.items()},
    }
    path = os.path.join(out_dir, 'dashboard-2-validation.yaml')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dash, fh, indent=2)   # YAML is a superset of JSON
    return path


def write_topsheet(out_dir: str, rows_available: dict) -> str:
    """At-a-glance sheet: file list plus the counts SimWrapper can evaluate."""
    files, outputs = {}, []
    if rows_available.get('mode'):
        files['mode'] = {'file': 'val_mode_share.csv'}
        outputs.append({'title': 'Mode-share rows',
                        'value': '{@count(mode.mode)}'})
    if rows_available.get('kpis'):
        files['kpis'] = {'file': 'val_kpis.csv'}
        outputs.append({'title': 'Fit metrics recorded',
                        'value': '{@count(kpis.metric)}'})
    if rows_available.get('destination'):
        files['dest'] = {'file': 'val_destination_fit.csv'}
        outputs.append({'title': 'Destination segments',
                        'value': '{@count(dest.segment)}'})
    if rows_available.get('tod'):
        files['tod'] = {'file': 'val_tod_profile.csv'}
        outputs.append({'title': 'Departure-profile hours',
                        'value': '{@count(tod.hour)}'})
    if rows_available.get('length'):
        files['length'] = {'file': 'val_trip_length.csv'}
        outputs.append({'title': 'Distance bands',
                        'value': '{@count(length.bin)}'})
    sheet = {
        'title': 'HHTS re-prediction — at a glance',
        'files': files,
        'calculations': {},
        'outputs': outputs,
    }
    path = os.path.join(out_dir, 'topsheet-validation.yaml')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(sheet, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def tod_rows(results: list) -> list:
    """Hourly departure profile, widened so one file feeds both charts."""
    wide = {}
    for res in results:
        metrics = res.get('metrics') or {}
        for rec in (metrics.get('tod_profile') or []):
            if not isinstance(rec, dict):
                continue
            profile = str(rec.get('profile', '')).lower()
            key = 'departures' if profile.startswith('all') else 'first'
            hour = str(rec.get('hour', '')).strip()
            if not hour:
                continue
            row = wide.setdefault(hour, {'hour': hour})
            row[f'{key}_obs'] = _round(_num(rec.get('observed_pct')))
            row[f'{key}_pred'] = _round(_num(rec.get('predicted_pct')))
    return [wide[hour] for hour in sorted(wide)]


def trip_length_rows(results: list) -> list:
    """Tour-length distribution by distance band, observed vs predicted."""
    rows = []
    for res in results:
        metrics = res.get('metrics') or {}
        for rec in (metrics.get('trip_length_bins') or []):
            if not isinstance(rec, dict):
                continue
            rows.append({
                'bin': str(rec.get('bin', '')),
                'observed_pct': _round(_num(rec.get('observed_pct'))),
                'predicted_pct': _round(_num(rec.get('predicted_pct'))),
            })
    return rows


def _kpis(results: list, modes: list, dests: list) -> list:
    """Hand-picked fit indicators for the gallery header (ActivitySim style)."""
    def metric(stage, key):
        label = _stage_label(stage)
        for row in scalar_rows(results):
            if row['stage'] == label and row['metric'] == key:
                return row['value']
        return None

    cards = []
    r = metric(1, 'r')
    if r is not None:
        cards.append({'label': 'Frequency correlation',
                      'value': f'{r:.3f}',
                      'note': 'Observed vs predicted purpose means'})
    rmse = metric(1, 'mean_RMSE')
    if rmse is not None:
        cards.append({'label': 'Frequency RMSE',
                      'value': f'{rmse:.3f}',
                      'note': 'Mean activities per person'})
    gaps = [abs(row['delta_pp']) for row in modes
            if row.get('delta_pp') is not None]
    if gaps:
        worst = max(gaps)
        cards.append({'label': 'Worst mode-share gap',
                      'value': f'{worst:.2f} pp',
                      'note': 'All segments and modes',
                      'tone': 'good' if worst <= 2 else
                              ('warn' if worst <= 5 else 'bad')})
    maes = [row['mae_share'] for row in dests
            if row.get('mae_share') is not None]
    if maes:
        cards.append({'label': 'Destination MAE',
                      'value': f'{sum(maes) / len(maes):.4f}',
                      'note': 'Mean over segments'})
    tod_r = metric('profiles', 'tod_all_corr')
    if tod_r is not None:
        cards.append({'label': 'Departure-profile r',
                      'value': f'{tod_r:.3f}',
                      'note': 'Hourly shares, observed vs predicted',
                      'tone': 'good' if tod_r >= 0.90 else
                              ('warn' if tod_r >= 0.75 else 'bad')})
    len_obs = metric('profiles', 'length_mean_km_obs')
    len_pred = metric('profiles', 'length_mean_km_pred')
    if len_obs is not None:
        cards.append({'label': 'Mean tour length',
                      'value': (f'{len_pred:.1f} km' if len_pred is not None
                                else f'{len_obs:.1f} km'),
                      'note': f'Observed {len_obs:.1f} km'})
    if not cards:
        cards.append({'label': 'Validation metrics',
                      'value': str(len(results)),
                      'note': 'stages reported'})
    return cards


def build_validation_dashboard(run_dir: str | None = None,
                               validation_dir: str | None = None,
                               out_dir: str | None = None,
                               quiet: bool = False):
    """Build the validation dashboard; returns the written paths or None."""
    def say(msg):
        if not quiet:
            log(msg)

    if run_dir is None and validation_dir is None:
        cands = sorted(glob.glob(os.path.join(CODE_DIR, 'runs/ga_parallel/*')),
                       key=os.path.getmtime)
        run_dir = cands[-1] if cands else None
    run_dir = os.path.abspath(run_dir) if run_dir else None

    vdir = os.path.abspath(validation_dir) if validation_dir else \
        find_validation_dir(run_dir)
    if vdir is None or not os.path.isdir(vdir):
        say('  [validation] no HHTS validation artifacts found — skipped')
        return None

    results = load_metrics(vdir)
    figures = figure_paths(results, vdir)
    if not results and not figures:
        say('  [validation] validation directory is empty — skipped')
        return None

    if out_dir is None:
        root = run_dir or os.path.dirname(vdir)
        out_dir = os.path.join(root, 'summarize')
    os.makedirs(out_dir, exist_ok=True)

    modes = mode_share_rows(results, vdir)
    purposes = purpose_rows(results)
    dests = destination_rows(results)
    tods = tod_rows(results)
    lengths = trip_length_rows(results)
    kpis = scalar_rows(results)

    written = {}
    if kpis:
        written['kpis'] = os.path.join(out_dir, 'val_kpis.csv')
        write_csv(written['kpis'], kpis, ['stage', 'metric', 'value'])
    if modes:
        written['mode'] = os.path.join(out_dir, 'val_mode_share.csv')
        write_csv(written['mode'], modes,
                  ['segment', 'mode', 'observed_pct', 'predicted_pct',
                   'delta_pp'])
        segments = write_per_segment_mode_csvs(out_dir, modes)
    else:
        segments = []
    if purposes:
        written['purpose'] = os.path.join(out_dir, 'val_purpose_means.csv')
        write_csv(written['purpose'], purposes,
                  ['purpose', 'observed', 'predicted'])
    if dests:
        written['destination'] = os.path.join(out_dir,
                                             'val_destination_fit.csv')
        write_csv(written['destination'], dests,
                  ['kind', 'segment', 'mae_share', 'js_divergence',
                   'top_overlap_pct', 'n'])
    if tods:
        written['tod'] = os.path.join(out_dir, 'val_tod_profile.csv')
        write_csv(written['tod'], tods,
                  ['hour', 'departures_obs', 'departures_pred',
                   'first_obs', 'first_pred'])
    if lengths:
        written['length'] = os.path.join(out_dir, 'val_trip_length.csv')
        write_csv(written['length'], lengths,
                  ['bin', 'observed_pct', 'predicted_pct'])

    run_label = os.path.basename(run_dir) if run_dir else \
        os.path.basename(os.path.dirname(vdir))
    notes = ('Predicted values come from applying each fitted stage to the '
             'same HHTS records used for estimation (in-sample), plus the '
             'held-out person split where the run produced one. Mode shares '
             'are tour level; stop destinations are trip level.')
    written['gallery'] = write_gallery(
        os.path.join(out_dir, 'validation_gallery.html'),
        f'{run_label} — HHTS re-prediction validation',
        _kpis(results, modes, dests), figures, notes)
    written['dashboard'] = write_dashboard(
        out_dir, run_label, segments, bool(purposes), bool(dests), True,
        has_tod=bool(tods), has_length=bool(lengths))
    written['topsheet'] = write_topsheet(
        out_dir, {'mode': bool(modes), 'kpis': bool(kpis),
                  'destination': bool(dests),
                  'tod': bool(tods), 'length': bool(lengths)})

    say(f'  [validation] {len(figures)} figures, {len(results)} stages')
    say(f'  [validation] dashboard -> {written["dashboard"]}')
    return written


def self_test(tmp_dir: str | None = None) -> int:
    """Build the dashboard from a fixture so the writer can be exercised."""
    root = tmp_dir or tempfile.mkdtemp(prefix='valdash_')
    vdir = os.path.join(root, 'ga_hhts_validation')
    plots = os.path.join(vdir, 'plots', 'validation')
    os.makedirs(plots, exist_ok=True)

    fixture = [
        {'stage': 1,
         'plots': ['plots/validation/val_stage1_frequency.png'],
         'metrics': {'overall_rmse': 0.412, 'mean_RMSE': 0.412, 'r': 0.913,
                     'purpose_means': [
                         {'purpose': 'work', 'observed': 1.42, 'predicted': 1.39},
                         {'purpose': 'shopping', 'observed': 0.71,
                          'predicted': 0.75}]}},
        {'stage': 5, 'plots': [], 'metrics': {
            'work_mode_obs': {'1': 61.2, '2': 18.4, '3': 6.1, '4': 3.2, '5': 11.1},
            'work_mode_pred': {'1': 59.8, '2': 19.0, '3': 6.4, '4': 3.1, '5': 11.7},
            'edu_mode_obs': {'1': 22.0, '2': 12.0, '3': 30.0, '4': 4.0, '5': 32.0},
            'edu_mode_pred': {'1': 23.4, '2': 12.9, '3': 29.1, '4': 3.8, '5': 30.8},
            'nonmand_mode_obs': {'1': 58.0, '2': 24.0, '3': 9.0, '4': 2.0, '5': 7.0},
            'nonmand_mode_pred': {'1': 57.1, '2': 24.6, '3': 9.3, '4': 2.0, '5': 7.0},
            'mode_share_max_gap_pp': 1.6,
            'tour_level_destination_fit': [
                {'segment': 'work', 'mae_share': 0.021, 'js_divergence': 0.043,
                 'top25_overlap_pct': 72.0, 'n_tours': 4820},
                {'segment': 'edu', 'mae_share': 0.018, 'js_divergence': 0.038,
                 'top25_overlap_pct': 76.0, 'n_tours': 1310}],
        }},
        {'stage': 'profiles',
         'plots': ['plots/validation/val_profiles_time_of_day.png'],
         'metrics': {
             'tod_all_corr': 0.947,
             'tod_all_max_gap_pct_points': 1.8,
             'tod_all_peak_hour_obs': '08:00',
             'tod_all_peak_hour_pred': '08:00',
             'tod_first_corr': 0.962,
             'length_mean_km_obs': 9.4,
             'length_mean_km_pred': 9.9,
             'length_median_km_obs': 5.6,
             'length_median_km_pred': 5.9,
             'length_max_gap_pct_points': 1.1,
             'length_js': 0.004,
             'tod_profile': [
                 {'profile': 'All trip departures', 'hour': '07:00',
                  'observed_pct': 12.4, 'predicted_pct': 12.0},
                 {'profile': 'All trip departures', 'hour': '08:00',
                  'observed_pct': 18.1, 'predicted_pct': 17.6},
                 {'profile': 'First departure of the day', 'hour': '07:00',
                  'observed_pct': 21.5, 'predicted_pct': 21.0},
                 {'profile': 'First departure of the day', 'hour': '08:00',
                  'observed_pct': 24.8, 'predicted_pct': 24.1},
             ],
             'trip_length_bins': [
                 {'bin': '0-1 km', 'observed_pct': 18.2, 'predicted_pct': 17.4},
                 {'bin': '1-2 km', 'observed_pct': 14.1, 'predicted_pct': 13.9},
                 {'bin': '2-5 km', 'observed_pct': 26.3, 'predicted_pct': 26.8},
             ],
         }},
    ]
    with open(os.path.join(vdir, 'validation_metrics.json'), 'w',
              encoding='utf-8') as fh:
        json.dump({'stages': fixture}, fh, indent=2)

    # Two small figures; matplotlib is optional for the self-test.
    made = False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 2.6))
        ax.bar(['work', 'shopping'], [1.42, 0.71], width=.4, label='Observed')
        ax.bar(['work', 'shopping'], [1.39, 0.75], width=.2, label='Predicted')
        ax.set_ylabel('Activities / person')
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(plots, 'val_stage1_frequency.png'), dpi=110,
                    bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 2.6))
        ax.bar(['DA', 'SR', 'Walk', 'Bike', 'PT'], [61.2, 18.4, 6.1, 3.2, 11.1],
               width=.4, label='Observed')
        ax.bar(['DA', 'SR', 'Walk', 'Bike', 'PT'], [59.8, 19.0, 6.4, 3.1, 11.7],
               width=.2, label='Predicted')
        ax.set_ylabel('Share (%)')
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(plots, 'val_stage5_mode_shares.png'), dpi=110,
                    bbox_inches='tight')
        plt.close(fig)
        made = True
    except Exception as exc:                     # pragma: no cover - optional
        log(f'  [self-test] matplotlib unavailable ({exc}); '
            'dashboard built without figures')

    written = build_validation_dashboard(run_dir=root)
    if not written:
        log('  [self-test] FAILED: no dashboard written')
        return 1

    required = ['dashboard', 'gallery', 'topsheet']
    missing = [k for k in required if not os.path.isfile(written[k])]
    for key in ('mode', 'kpis', 'destination', 'purpose', 'tod', 'length'):
        if key in written and not os.path.isfile(written[key]):
            missing.append(key)
    if missing:
        log(f'  [self-test] FAILED: missing {missing}')
        return 1

    with open(written['dashboard'], 'r', encoding='utf-8') as fh:
        dash = json.load(fh)
    panels = sum(len(v) if isinstance(v, list) else 1
                 for v in dash.get('layout', {}).values())
    log(f'  [self-test] OK — {panels} panels, figures={made}, '
        f'outputs in {os.path.dirname(written["dashboard"])}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='ActivitySim-style validation dashboard for the HHTS '
                    're-prediction.')
    parser.add_argument('run_dir', nargs='?',
                        help='run directory (default: newest under '
                             'runs/ga_parallel)')
    parser.add_argument('--validation-dir',
                        help='directory holding validation_metrics.json')
    parser.add_argument('--out-dir',
                        help='output directory (default: <run>/summarize)')
    parser.add_argument('--self-test', action='store_true',
                        help='build from a fixture and verify the outputs')
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    written = build_validation_dashboard(args.run_dir, args.validation_dir,
                                         args.out_dir)
    if not written:
        return 2
    log('  view with:  simwrapper open asim %s'
        % os.path.dirname(written['dashboard']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
