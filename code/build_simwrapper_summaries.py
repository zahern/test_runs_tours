#!/usr/bin/env python3
"""Build a SimWrapper dashboard for a GA parallel run directory.

Turns any completed run dir (runs/ga_parallel/<TAG>) into a `summarize/`
subfolder that SimWrapper can render with `simwrapper open asim`:

  * consolidate per-stage coefficient CSVs (stage0..stage5) into one table
  * consolidate per-stage timings
  * aggregate saved_data/trips_for_aimsun.csv into mode-share / period /
    distance tables and zone-level origin/destination counts
  * build a SEQ SA2 geojson (cached in saved_data/sa2_seq.geojson) so the
    zone counts can be drawn as a choropleth map
  * write dashboard-1-summary.yaml (charts + map) and topsheet.yaml
    (at-a-glance KPI calculation table)

Usage:
  python build_simwrapper_summaries.py <run_dir> [--skip-map]

  <run_dir>   path to the GA run (e.g. runs/ga_parallel/destfix). If omitted,
              the newest subdirectory of runs/ga_parallel is used.
  --skip-map  don't build/use the geojson (faster, no map panel)

Writes everything under <run_dir>/summarize/. Existing outputs are replaced.
The job wrapper job_simwrapper_summaries.pbs calls this for RUN_ROOT/RUN_TAG.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import shutil

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DATA = os.path.join(CODE_DIR, 'saved_data')
SA2_SHP_DIR = os.path.join(SAVED_DATA, 'SA2_2021_AUST_SHP_GDA2020')
SA2_SHP = os.path.join(SA2_SHP_DIR, 'SA2_2016_AUST.shp')
ZONE_CSV = os.path.join(CODE_DIR, 'seq_zone_ids.csv')
TRIPS_CSV = os.path.join(SAVED_DATA, 'trips_for_aimsun.csv')
GEOJSON_CACHE = os.path.join(SAVED_DATA, 'sa2_seq.geojson')

COEF_FILES = [
    'stage0/stage0_cdap_coefficients.csv',
    'stage1/stage1_coefficients.csv',
    'stage2/stage2_coefficients.csv',
    'stage3/stage3_coefficients.csv',
    'stage4/stage4_coefficients.csv',
    'stage5/stage5_coefficients.csv',
    'stage5/stage5_models_coefficients.csv',
]


def log(msg):
    print(msg, flush=True)


def sanitize(name):
    s = re.sub(r'[^0-9A-Za-z_]+', '_', str(name))
    if not s or s[0].isdigit():
        s = 'v_' + s
    return s


def _first(row, *keys):
    """Return the first present, non-NaN value from row for the given keys.

    Unlike `row.get(k) or fallback`, this preserves legitimate 0.0 values
    (e.g. a holdfast parameter pinned at zero has std_err 0.0, which the
    `or`-style fallback wrongly converted to empty string -> NaN downstream).
    """
    for k in keys:
        v = row.get(k, None)
        if v is None or v == '':
            continue
        try:
            if float(v) != float(v):  # NaN
                continue
        except (TypeError, ValueError):
            continue
        return v
    return ''


def read_csv(path):
    if pd is not None:
        return pd.read_csv(path)
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def to_csv(rows, path):
    if not rows:
        return False
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    return True


def consolidate_coefficients(run_dir, out_path):
    """Merge every stage*_coefficients*.csv found in the run into one long table."""
    rows = []
    for rel in COEF_FILES:
        p = os.path.join(run_dir, rel)
        if not os.path.exists(p):
            continue
        try:
            data = read_csv(p)
        except Exception as e:
            log(f'  warn: cannot read {rel}: {e}')
            continue
        if pd is not None and isinstance(data, pd.DataFrame):
            records = data.to_dict('records')
        else:
            records = data
        for r in records:
            rows.append({
                'stage': r.get('stage') or rel.split('/')[0],
                'segment': r.get('segment', ''),
                'param': r.get('param') or r.get('parameter', ''),
                'value': r.get('value', ''),
                'std_err': _first(r, 'std_err', 'se'),
                't_stat': _first(r, 't_stat', 'tstat'),
                'p_value': _first(r, 'p_value', 'pval'),
            })
    if to_csv(rows, out_path):
        log(f'  wrote {out_path} ({len(rows)} coefficient rows)')


def consolidate_timings(run_dir, out_path):
    rows = []
    for stage_dir in sorted(glob.glob(os.path.join(run_dir, 'stage*'))):
        if not os.path.isdir(stage_dir):
            continue
        tpath = os.path.join(stage_dir, 'timings.csv')
        if not os.path.exists(tpath):
            continue
        try:
            data = read_csv(tpath)
        except Exception as e:
            log(f'  warn: cannot read {tpath}: {e}')
            continue
        if pd is not None and isinstance(data, pd.DataFrame):
            records = data.to_dict('records')
        else:
            records = data
        stage = os.path.basename(stage_dir)
        for r in records:
            rows.append({
                'stage': stage,
                'step': r.get('step', ''),
                'elapsed_seconds': r.get('elapsed_seconds', ''),
                'status': r.get('status', ''),
            })
    if to_csv(rows, out_path):
        log(f'  wrote {out_path} ({len(rows)} timing rows)')


def aggregate_trips(out_dir):
    """Aggregate saved_data/trips_for_aimsun.csv into summary CSVs."""
    if not os.path.exists(TRIPS_CSV):
        log(f'  skip trips aggregation: {TRIPS_CSV} missing')
        return False
    df = read_csv(TRIPS_CSV)
    if pd is None:
        log('  warn: pandas unavailable — trip tables not built')
        return False

    df['mode_name'] = df['mode_name'].fillna('Unknown').astype(str)
    mode = (df.groupby('mode_name').size().rename('n_trips').reset_index())
    mode['share'] = mode['n_trips'] / mode['n_trips'].sum()
    mode.to_csv(os.path.join(out_dir, 'trip_mode_share.csv'), index=False)
    # wide form (one row, one column per mode) — SimWrapper pie format
    wide = pd.DataFrame([{sanitize(m): n for m, n in zip(mode['mode_name'], mode['n_trips'])}])
    wide.to_csv(os.path.join(out_dir, 'trip_mode_share_wide.csv'), index=False)

    if 'time_period' in df.columns:
        period = df.groupby('time_period').size().rename('n_trips').reset_index()
        period.to_csv(os.path.join(out_dir, 'trips_by_period.csv'), index=False)

    if 'dist' in df.columns:
        bins = [0, 1, 2, 5, 10, 20, 50, 100, 1000]
        labels = ['<1', '1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+']
        d = pd.cut(df['dist'], bins=bins, labels=labels, right=False).fillna(labels[-1])
        dist = d.value_counts().reindex(labels).rename('n_trips').rename_axis('dist_km').reset_index()
        dist.to_csv(os.path.join(out_dir, 'trip_length_dist.csv'), index=False)

    if {'OTAZ', 'DTAZ'}.issubset(df.columns):
        zones = read_csv(ZONE_CSV) if os.path.exists(ZONE_CSV) else None
        agg = pd.DataFrame({
            'zone': pd.concat([df['OTAZ'], df['DTAZ']]),
            'kind': ['orig'] * len(df) + ['dest'] * len(df),
        }).groupby(['zone', 'kind']).size().unstack(fill_value=0).reset_index()
        agg.columns = [str(c) if str(c) != 'zone' else 'zone' for c in agg.columns]
        agg = agg.rename(columns={'orig': 'n_orig', 'dest': 'n_dest'})
        for c in ('n_orig', 'n_dest'):
            if c not in agg.columns:
                agg[c] = 0
        if zones is not None:
            zdf = read_csv(ZONE_CSV)
            zdf['ORDER'] = zdf['ORDER'].astype('int64')
            if {'longitude', 'latitude'}.issubset(zdf.columns):
                agg['ORDER'] = agg['zone'].astype('int64')
                agg = agg.merge(
                    zdf[['ORDER', 'longitude', 'latitude', 'SA2_MAIN16', 'SA2_NAME16']],
                    on='ORDER', how='left')
        agg.to_csv(os.path.join(out_dir, 'trips_by_zone.csv'), index=False)

    od = df.groupby(['OTAZ', 'DTAZ']).size().rename('n_trips').reset_index()
    od.to_csv(os.path.join(out_dir, 'trips_od.csv'), index=False)

    log(f'  wrote trip tables from {TRIPS_CSV} ({len(df):,} trips)')
    return True


def build_sa2_geojson(force=False):
    """SEQ SA2 polygon geojson (cached). Uses pyshp; source is EPSG 4283 lat/lon."""
    if os.path.exists(GEOJSON_CACHE) and not force:
        log(f'  reuse cached {GEOJSON_CACHE}')
        return GEOJSON_CACHE
    if not os.path.exists(SA2_SHP):
        log(f'  skip geojson: shapefile missing ({SA2_SHP})')
        return None
    try:
        import shapefile
    except Exception as e:
        log(f'  skip geojson: pyshp not installed ({e})')
        return None
    zones = {}
    if os.path.exists(ZONE_CSV):
        with open(ZONE_CSV, newline='', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                zones[str(r.get('SA2_MAIN16', '')).strip()] = {
                    'ORDER': r.get('ORDER', ''),
                    'NAME': r.get('SA2_NAME16', ''),
                }
    if not zones:
        log('  skip geojson: no zone crosswalk')
        return None
    sf = shapefile.Reader(SA2_SHP)
    fields = [f[0] for f in sf.fields[1:]]
    features = []
    kept = 0
    for srec, srec_rec in zip(sf.shapeRecords(), sf.records()):
        rec = dict(zip(fields, srec_rec))
        sa2 = str(rec.get('SA2_MAIN16', '')).strip()
        if sa2 not in zones:
            continue
        s = srec.shape
        if s.shapeType not in (shapefile.POLYGON, shapefile.POLYGONZ):
            continue
        coords = []
        for i in range(len(s.parts)):
            start = s.parts[i]
            end = s.parts[i + 1] if i + 1 < len(s.parts) else len(s.points)
            ring = [[float(pt[0]), float(pt[1])] for pt in s.points[start:end]]
            coords.append(ring)
        z = zones[sa2]
        features.append({
            'type': 'Feature',
            'properties': {
                'SA2_MAIN16': sa2,
                'ORDER': z['ORDER'],
                'SA2_NAME16': z['NAME'],
            },
            'geometry': {'type': 'Polygon', 'coordinates': coords},
        })
        kept += 1
    geojson = {'type': 'FeatureCollection', 'features': features}
    with open(GEOJSON_CACHE, 'w') as fh:
        json.dump(geojson, fh)
    log(f'  built {GEOJSON_CACHE} ({kept} SA2 polygons)')
    return GEOJSON_CACHE


def write_dashboard_yaml(out_dir, has_map, has_trips):
    rows = []

    if has_trips:
        rows.append({
            'row1': [
                {'type': 'bar', 'title': 'Trip mode share', 'dataset': 'trip_mode_share.csv',
                 'x': 'mode_name', 'columns': ['n_trips'], 'width': 1,
                 'yAxisName': 'Trips'},
                {'type': 'bar', 'title': 'Trips by time period', 'dataset': 'trips_by_period.csv',
                 'x': 'time_period', 'columns': ['n_trips'], 'width': 1,
                 'yAxisName': 'Trips'},
                {'type': 'bar', 'title': 'Trip length distribution', 'dataset': 'trip_length_dist.csv',
                 'x': 'dist_km', 'columns': ['n_trips'], 'width': 1,
                 'yAxisName': 'Trips'},
            ],
        })
        if has_map:
            rows.append({
                'row2': {
                    'type': 'map',
                    'title': 'Trip origins and destinations by SA2',
                    'height': 9,
                    'zoom': 8,
                    'mapIsIndependent': True,
                    'shapes': {'file': 'sa2_seq.geojson', 'join': 'ORDER'},
                    'datasets': {'zoneTrips': {'file': 'trips_by_zone.csv'}},
                    'display': {
                        'fill': {
                            'dataset': 'zoneTrips',
                            'join': 'ORDER',
                            'columnName': 'n_orig',
                            'colorRamp': {'ramp': 'Plasma', 'steps': 6},
                        },
                    },
                    'tooltip': ['shapes.SA2_NAME16', 'zoneTrips.n_orig', 'zoneTrips.n_dest'],
                },
            })

    # per-stage coefficient bars + one table of the consolidated file
    coef_panels = []
    stage_tables = sorted(glob.glob(os.path.join(out_dir, 'coefs_stage*.csv')))
    for i, sp in enumerate(stage_tables[:8]):
        stage = os.path.splitext(os.path.basename(sp))[0].replace('coefs_', '')
        coef_panels.append({
            'type': 'bar',
            'title': f'Coefficients — {stage}',
            'dataset': os.path.basename(sp),
            'x': 'param', 'columns': ['value'], 'width': 1,
            'height': 6,
        })
    if coef_panels:
        rows.append({'row3': coef_panels})

    if os.path.exists(os.path.join(out_dir, 'stage_timings.csv')):
        rows.append({
            'row4': {
                'type': 'bar',
                'title': 'Stage timings',
                'dataset': 'stage_timings.csv',
                'x': 'step', 'columns': ['elapsed_seconds'],
                'yAxisName': 'Seconds',
            },
        })

    if os.path.exists(os.path.join(out_dir, 'constraint_report.csv')):
        rows.append({
            'row5': {
                'type': 'table',
                'title': 'Constraint report',
                'description': 'View full CSV via the Details pane',
                'configFile': 'table-constraints.yaml',
            },
        })

    dash = {
        'header': {
            'tab': 'Summary',
            'title': os.path.basename(os.path.dirname(out_dir)),
            'description': 'GA estimation + prediction summaries',
        },
        'layout': {k: v for row in rows for k, v in row.items()},
    }
    with open(os.path.join(out_dir, 'dashboard-1-summary.yaml'), 'w') as fh:
        json.dump(dash, fh, indent=2)
    # YAML is a superset of JSON — SimWrapper reads this fine.
    log('  wrote dashboard-1-summary.yaml')


def write_topsheet(out_dir, has_trips):
    files = {}
    if has_trips:
        files['trips'] = {'file': 'trip_mode_share.csv'}
    if os.path.exists(os.path.join(out_dir, 'stage_timings.csv')):
        files['timings'] = {'file': 'stage_timings.csv'}
    calc = {
        'nModes': 'round({@count(trips.mode_name)})' if has_trips else '0',
    }
    outputs = []
    if has_trips:
        outputs = [
            {'title': 'Distinct trip modes', 'value': 'nModes'},
        ]
    top = {
        'title': 'Summary: Key Performance Indicators',
        'files': files,
        'calculations': calc,
        'outputs': outputs,
    }
    with open(os.path.join(out_dir, 'topsheet.yaml'), 'w') as fh:
        json.dump(top, fh, indent=2)
    log('  wrote topsheet.yaml')


def split_coefs_by_stage(coef_path, out_dir):
    """Slice the consolidated coefficient CSV into per-stage files for bar charts."""
    if pd is None or not os.path.exists(coef_path):
        return
    df = pd.read_csv(coef_path)
    for stage, g in df.groupby('stage'):
        fname = 'coefs_' + sanitize(str(stage))[:40] + '.csv'
        g2 = g.copy()
        g2['value'] = pd.to_numeric(g2['value'], errors='coerce')
        g2 = g2.dropna(subset=['value'])
        if g2.empty:
            continue
        g2 = g2.sort_values('value', key=lambda s: s.abs(), ascending=False).head(25)
        g2.to_csv(os.path.join(out_dir, fname), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', nargs='?')
    ap.add_argument('--skip-map', action='store_true')
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir:
        cands = sorted(glob.glob(os.path.join(CODE_DIR, 'runs/ga_parallel/*')), key=os.path.getmtime)
        run_dir = cands[-1] if cands else None
    if not run_dir or not os.path.isdir(run_dir):
        log(f'ERROR: run dir not found: {run_dir}')
        sys.exit(2)
    run_dir = os.path.abspath(run_dir)
    log(f'Building SimWrapper summaries for {run_dir}')

    out_dir = os.path.join(run_dir, 'summarize')
    os.makedirs(out_dir, exist_ok=True)

    coef_path = os.path.join(out_dir, 'stage_coefficients.csv')
    log('Consolidating stage coefficients...')
    consolidate_coefficients(run_dir, coef_path)
    split_coefs_by_stage(coef_path, out_dir)

    log('Consolidating stage timings...')
    consolidate_timings(run_dir, os.path.join(out_dir, 'stage_timings.csv'))

    if os.path.exists(os.path.join(run_dir, 'merged/constraint_report.csv')):
        shutil.copy2(os.path.join(run_dir, 'merged/constraint_report.csv'),
                     os.path.join(out_dir, 'constraint_report.csv'))
        log('  copied constraint_report.csv')
        with open(os.path.join(out_dir, 'table-constraints.yaml'), 'w') as fh:
            json.dump({
                'title': 'Constraint report — at a glance',
                'files': {'cr': {'file': 'constraint_report.csv'}},
                'calculations': {
                    'nRows': '{@count(cr.stage)}',
                    'nFree': '{@count(@filter(cr.status == free))}',
                },
                'outputs': [
                    {'title': 'Constraint rows', 'value': 'nRows'},
                    {'title': 'Free features', 'value': 'nFree',
                     'style': {'color': 'green'}},
                ],
            }, fh, indent=2)
        log('  wrote table-constraints.yaml')

    has_trips = aggregate_trips(out_dir)

    has_map = False
    if not args.skip_map:
        geojson = build_sa2_geojson()
        if geojson and os.path.exists(out_dir):
            shutil.copy2(geojson, os.path.join(out_dir, 'sa2_seq.geojson'))
            has_map = True

    write_dashboard_yaml(out_dir, has_map, has_trips)
    write_topsheet(out_dir, has_trips)

    log(f'Done. View with:  simwrapper open asim {out_dir}')


if __name__ == '__main__':
    main()
