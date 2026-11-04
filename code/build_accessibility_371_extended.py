"""Build extended 371-zone accessibility table with real crime data.

Reads (all in saved_data/ unless noted):
  - qps_division_offences.csv .... QPS reported offences by police division,
    monthly Jul-2001..Aug-2026, ~100 offence columns incl. Homicide (Murder).
    SOURCE: QLD open data portal (open-crime-data S3). NOTE on provenance:
    the published file has (a) a malformed header token '"Common Assault'"'
    (repaired to '"Common Assault"') and (b) a trailing row-number value on
    every data row with no header name (dropped). Both quirks verified:
    subtype columns sum exactly to their parent (Assault, Sexual Offences,
    Robbery, Drugs, Fraud, ...) on 20k rows, so alignment is exact.
  - QPS_DIVISIONS.zip ............ official police-division boundaries (KML).
  - ../seq_zone_ids.csv ........... legacy SA2 names + verified lon/lat
    (used for centroids; saved_data/sa2_centroids.csv is deliberately NOT
    used — its coordinates are wrong for many zones).
  - zone_landuse_overture.csv ..... ~150 land-use/transport columns per SA2.
  - accessibility_371.csv ......... existing 16 logsum access measures.
  - ../OD_Crashes.csv ............. crash OD table (ORDER zone space).
  - ../seq_zone_ids.csv ........... ORDER <-> SA2_MAIN16 crosswalk.

Method:
  1. Assign each SA2 centroid to its QPS division (point-in-polygon).
     SA2s outside QLD (e.g. Tweed) get no division -> NaN crime, flagged.
  2. Aggregate the most recent 12 months of offences per division, then
     allocate to SA2s by SA2 population share within the division
     (dasymetric; approximate — divisions don't follow SA2 boundaries).
  3. Crash attraction per SA2 from OD_Crashes via the ORDER crosswalk.
  4. Purpose attractiveness densities, transit intensity, walkability,
     and a composite visitability score per purpose.

Writes: saved_data/accessibility_371_extended.csv (one row per zone).

Usage:  python build_accessibility_371_extended.py [--months 12]
Only needs numpy/pandas (stdlib for KML/zip).
"""
import argparse
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(CODE_DIR, 'saved_data')

VIOLENT_TOP = ['Homicide (Murder)', 'Other Homicide', 'Attempted Murder',
               'Assault', 'Sexual Offences', 'Robbery',
               'Kidnapping & Abduction etc.', 'Extortion', 'Stalking',
               'Life Endangering Acts']
PROPERTY_TOP = ['Unlawful Entry', 'Arson', 'Other Property Damage',
                'Unlawful Use of Motor Vehicle',
                'Other Theft (excl. Unlawful Entry)', 'Fraud',
                'Handling Stolen Goods']
# subtype check (validates file alignment on the slice we use)
SUBTYPE_CHECK = {
    'Assault': ['Grievous Assault', 'Serious Assault',
                'Serious Assault (Other)', 'Common Assault'],
    'Sexual Offences': ['Rape and Attempted Rape', 'Other Sexual Offences'],
    'Robbery': ['Armed Robbery', 'Unarmed Robbery'],
}

MON = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
       'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def parse_month(s):
    s = str(s).strip().upper()
    return (2000 + int(s[3:5]), MON[s[:3]])


def load_division_polys():
    """{DIVNAME: [(lon,lat) arrays]} from the KML inside QPS_DIVISIONS.zip."""
    with zipfile.ZipFile(os.path.join(SAVED, 'QPS_DIVISIONS.zip')) as z:
        kml = z.read('QPS_DIVISIONS.kml').decode('utf-8', errors='replace')
    root = ET.fromstring(kml)
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    out = {}
    for pm in root.findall('.//k:Placemark', ns):
        name = pm.find('k:name', ns).text.strip().upper()
        polys = []
        for coords in pm.findall('.//k:LinearRing/k:coordinates', ns):
            pts = []
            for tok in coords.text.strip().split():
                lon, lat, *_ = tok.split(',')
                pts.append((float(lon), float(lat)))
            polys.append(np.asarray(pts, dtype=float))
        if polys:
            out[name] = polys
    return out


def point_in_poly(lon, lat, poly):
    """Vectorised ray casting for scalar point vs (N,2) polygon."""
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    j = np.arange(len(poly) - 1)
    xi, yi = x0[j], y0[j]
    xj, yj = x0[j + 1], y0[j + 1]
    cross = ((yi > lat) != (yj > lat)) & (
        lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-18) + xi)
    return bool(cross.sum() % 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=12)
    args = ap.parse_args()

    # ── zones ──────────────────────────────────────────────────────────
    acc = pd.read_csv(os.path.join(SAVED, 'accessibility_371.csv'))
    zones = pd.to_numeric(acc['zone_id'], errors='coerce').dropna().astype(int).tolist()
    print(f'zones: {len(zones)}', flush=True)
    lu = pd.read_csv(os.path.join(SAVED, 'zone_landuse_overture.csv'))
    lu['SA2'] = pd.to_numeric(lu['SA2_MAIN16'], errors='coerce').astype('Int64')
    lu = lu.drop_duplicates('SA2').set_index('SA2')
    # Centroids: seq_zone_ids.csv (legacy SA2s, verified lon/lat) joined by
    # SA2 name. saved_data/sa2_centroids.csv is NOT used: its coordinates
    # are wrong for many zones (e.g. Capalaba plotted near Ipswich,
    # Mermaid Waters plotted in inner Brisbane — verified Sep-2026).
    seq = pd.read_csv(os.path.join(CODE_DIR, 'seq_zone_ids.csv'))
    seq['namekey'] = seq['SA2_NAME16'].astype(str).str.strip().str.lower()
    seq = seq.drop_duplicates('namekey').set_index('namekey')
    lu['namekey'] = lu['sa2_name'].astype(str).str.strip().str.lower()
    import re
    from collections import Counter
    STOP = {'north', 'south', 'east', 'west', 'upper', 'lower', 'central',
            'new', 'mountain', 'mountains', 'mount', 'mt', 'port', 'lake',
            'river', 'bay', 'island', 'hills', 'hill', 'park', 'ridge',
            'vale', 'view', 'grove', 'plains', 'plain', 'heights', 'height',
            'middle', 'west', 'beach', 'point'}
    STOP = set(STOP)
    # token distinctiveness: in how many legacy names does each token occur?
    tokcount = Counter()
    for sn in seq.index.tolist():
        for t in set(re.split(r'[^a-z]+', sn)):
            if len(t) >= 5:
                tokcount[t] += 1
    cen_lon, cen_lat, cen_src, order_legacy = {}, {}, {}, {}
    alias_rows = []
    seq_names = seq.index.tolist()
    for z in zones:
        try:
            nm = str(lu.loc[z, 'namekey'])
        except KeyError:
            continue
        if nm in seq.index:
            cen_lon[z] = float(seq.loc[nm, 'longitude'])
            cen_lat[z] = float(seq.loc[nm, 'latitude'])
            cen_src[z] = 'seq-exact'
            # ORDER bridge: lets stage-5 (ORDER-indexed lu_emp) join these
            # SA2-keyed columns. NaN where the new-edition SA2 has no legacy
            # counterpart (growth areas, islands).
            try:
                order_legacy[z] = int(seq.loc[nm, 'ORDER'])
            except (KeyError, ValueError, TypeError):
                pass
            continue
        # alias: new-edition splits (e.g. 'Ormeau (East) - Stapylton' from
        # legacy 'ormeau - yatala'). Rank candidates by (#shared tokens,
        # longest shared token, name); accept only distinctive matches:
        # >=2 shared tokens, or 1 token that is long and rare. Generic
        # tokens ('heights', 'middle', 'mountain') must never alias —
        # they caused Toowoomba->Logan and coast->hinterland howlers.
        parts = [t for t in re.split(r'[^a-z]+', nm) if len(t) >= 5
                 and t not in STOP]
        scored = []
        for sn in seq_names:
            shared = [t for t in parts if t in sn]
            if shared:
                scored.append((len(shared), max(len(t) for t in shared),
                               min(tokcount[t] for t in shared), sn))
        if scored:
            scored.sort(key=lambda r: (-r[0], -r[1], r[2], r[3]))
            n_shared, max_len, min_tc, pick = scored[0]
            if n_shared >= 2 or (n_shared == 1 and max_len >= 5
                                 and min_tc <= 3):
                cen_lon[z] = float(seq.loc[pick, 'longitude'])
                cen_lat[z] = float(seq.loc[pick, 'latitude'])
                cen_src[z] = f'seq-alias:{pick}'
                try:
                    order_legacy[z] = int(seq.loc[pick, 'ORDER'])
                except (KeyError, ValueError, TypeError):
                    pass
                alias_rows.append({'zone_id': z,
                                   'sa2_name': lu.loc[z, 'sa2_name'],
                                   'alias_seq_name':
                                       seq.loc[pick, 'SA2_NAME16']})
    print(f'centroids from seq_zone_ids by name: {len(cen_lon)}/{len(zones)} '
          f'({sum(1 for v in cen_src.values() if v == "seq-exact")} exact, '
          f'{len(alias_rows)} alias)', flush=True)
    if alias_rows:
        ap = os.path.join(SAVED, 'sa2_centroid_aliases.csv')
        pd.DataFrame(alias_rows).to_csv(ap, index=False)
        print(f'  alias decisions logged for review: {ap}', flush=True)
    missing = [z for z in zones if z not in cen_lon]
    if missing:
        print(f'  no centroid e.g.: '
              f'{[str(lu.loc[z, "sa2_name"]) if z in lu.index else z for z in missing[:10]]}',
              flush=True)

    # ── SA2 -> division ────────────────────────────────────────────────
    polys = load_division_polys()
    print(f'divisions with polygons: {len(polys)}', flush=True)
    # QPS data covers Queensland only: Tweed (NSW) zones must stay NaN even
    # if an alias centroid lands them inside a QLD polygon.
    def _is_nsw(z):
        try:
            return str(lu.loc[z, 'sa4_name']) == 'Richmond - Tweed'
        except KeyError:
            return False

    sa2_div = {}
    for z in zones:
        if z not in cen_lon or _is_nsw(z):
            continue
        lon, lat = cen_lon[z], cen_lat[z]
        for dname, dp in polys.items():
            if any(point_in_poly(lon, lat, p) for p in dp):
                sa2_div[z] = dname
                break
    n_mapped = len(sa2_div)
    print(f'SA2s inside a QPS division: {n_mapped}/{len(zones)}', flush=True)
    unmapped = [z for z in zones if z not in sa2_div]
    if unmapped:
        names = []
        for z in unmapped[:12]:
            try:
                names.append(f'{z} ({lu.loc[z, "sa2_name"]})')
            except KeyError:
                names.append(str(z))
        print(f'  unmapped e.g.: {names}', flush=True)

    # ── offences, recent window ────────────────────────────────────────
    qps = pd.read_csv(os.path.join(SAVED, 'qps_division_offences.csv'))
    # alignment self-check on the slice we use
    for parent, kids in SUBTYPE_CHECK.items():
        lhs = pd.to_numeric(qps[parent], errors='coerce').fillna(0)
        rhs = sum(pd.to_numeric(qps[k], errors='coerce').fillna(0) for k in kids)
        bad = int((lhs != rhs).sum())
        assert bad == 0, f'alignment broken at {parent}: {bad} rows'
    print('offence file alignment OK (subtype sums exact)', flush=True)
    qps['_ym'] = qps['Month Year'].map(parse_month)
    win = sorted(qps['_ym'].unique())[-args.months:]
    print(f'window: {win[0]}..{win[-1]} ({len(win)} months)', flush=True)
    q = qps[qps['_ym'].isin(win)].copy()
    q['DIV'] = q['Division'].astype(str).str.strip().str.upper()
    for c in VIOLENT_TOP + PROPERTY_TOP + ['Homicide (Murder)', 'Assault',
                                           'Sexual Offences', 'Robbery',
                                           'Unlawful Entry']:
        q[c] = pd.to_numeric(q[c], errors='coerce').fillna(0)
    g = q.groupby('DIV')
    div_violent = g[VIOLENT_TOP].sum().sum(axis=1)
    div_property = g[PROPERTY_TOP].sum().sum(axis=1)
    div_murder = g['Homicide (Murder)'].sum()
    div_assault = g['Assault'].sum()
    div_total = g[VIOLENT_TOP + PROPERTY_TOP].sum().sum(axis=1)

    # ── allocate division -> SA2 by population share ───────────────────
    pop = pd.to_numeric(lu['TOTPOP'], errors='coerce').fillna(0)
    div_pop = {}
    for z, d in sa2_div.items():
        div_pop[d] = div_pop.get(d, 0.0) + float(pop.get(z, 0.0))
    rows = []
    for z in zones:
        d = sa2_div.get(z)
        p = float(pop.get(z, 0.0))
        share = (p / div_pop[d]) if (d and div_pop.get(d, 0)) else np.nan
        rows.append({
            'zone_id': z,
            'qps_division': d if d else None,
            'crime_violent_n': float(div_violent.get(d, np.nan)) * share if d else np.nan,
            'crime_property_n': float(div_property.get(d, np.nan)) * share if d else np.nan,
            'crime_murder_n': float(div_murder.get(d, np.nan)) * share if d else np.nan,
            'crime_assault_n': float(div_assault.get(d, np.nan)) * share if d else np.nan,
            'crime_total_n': float(div_total.get(d, np.nan)) * share if d else np.nan,
        })
    ext = pd.DataFrame(rows).set_index('zone_id')
    ext['crime_covered'] = ext['crime_total_n'].notna()
    ext['order_legacy'] = pd.Series(order_legacy)
    # rates
    jobs = pd.to_numeric(lu['TOTEMP'], errors='coerce').fillna(0)
    area = pd.to_numeric(lu['sa2_area_sqkm'], errors='coerce').fillna(np.nan)
    ext['crime_violent_per100k'] = ext['crime_violent_n'] / pop * 1e5
    ext['crime_property_per100k'] = ext['crime_property_n'] / pop * 1e5
    ext['crime_murder_per100k'] = ext['crime_murder_n'] / pop * 1e5
    ext['crime_total_per100k'] = ext['crime_total_n'] / pop * 1e5

    # ── crash attraction (ORDER space -> SA2) ──────────────────────────
    xw = pd.read_csv(os.path.join(CODE_DIR, 'seq_zone_ids.csv'),
                     usecols=['SA2_MAIN16', 'ORDER'])
    o2s = dict(zip(pd.to_numeric(xw['ORDER'], errors='coerce').astype(int),
                   pd.to_numeric(xw['SA2_MAIN16'], errors='coerce').astype(int)))
    crashes = pd.read_csv(os.path.join(CODE_DIR, 'OD_Crashes.csv'))
    crashes['DTAZ_SA2'] = pd.to_numeric(crashes['Destination_ID'],
                                        errors='coerce').map(o2s)
    crashes['Exp'] = pd.to_numeric(crashes['Exp'], errors='coerce').fillna(0)
    attract = crashes.dropna(subset=['DTAZ_SA2']).groupby('DTAZ_SA2')['Exp'].sum()
    ext['crash_attract'] = ext.index.map(attract).fillna(0.0)
    _jobs = jobs.reindex(ext.index)
    ext['crash_per1000jobs'] = ext['crash_attract'] / _jobs.replace(0, np.nan) * 1000
    # ── smoothed crash rate (usable safety term) ─────────────────────
    # Empirical-Bayes shrinkage toward the global mean (low-job zones get
    # extreme raw rates) + k-nearest-neighbour spatial blend so a single
    # black-spot OD pair can't dominate a small zone.
    _expo = (_jobs / 1000).fillna(0)
    _r0 = float(ext['crash_attract'].sum() / _expo.replace(0, np.nan).sum())
    _k = float(_expo.replace(0, np.nan).median())
    ext['crash_per1000jobs_shrunk'] = (
        (ext['crash_attract'] + _k * _r0) / (_expo + _k))
    # neighbour blend on seq centroids (zones without coords keep shrunk)
    try:
        _zs = [z for z in ext.index if z in cen_lon]
        _P = np.array([[cen_lon[z], cen_lat[z]] for z in _zs])
        _d2 = (( _P[:, None, :] - _P[None, :, :]) ** 2).sum(-1)
        _order = np.argsort(_d2, axis=1)[:, 1:9]  # 8 nearest, excl self
        _base = ext['crash_per1000jobs_shrunk'].reindex(_zs).fillna(_r0).values
        _nb = _base[_order].mean(axis=1)
        _smooth = pd.Series(0.5 * _base + 0.5 * _nb, index=_zs)
        ext['crash_per1000jobs_smooth'] = _smooth.reindex(ext.index)
        print(f'crash smoothing: global={_r0:.2f}/1000jobs, '
              f'prior_k={_k:.1f}, blended {len(_zs)} zones', flush=True)
    except Exception as _e:
        ext['crash_per1000jobs_smooth'] = ext['crash_per1000jobs_shrunk']
        print(f'crash neighbour blend skipped ({_e}); shrunk only',
              flush=True)

    # ── attractiveness densities (per km2) ─────────────────────────────
    dens = pd.DataFrame(index=ext.index)
    a = area.reindex(ext.index)
    dens['attract_shop'] = (pd.to_numeric(lu['retail_ovm'], errors='coerce').fillna(0) +
                            pd.to_numeric(lu['EATOUT_ovm'], errors='coerce').fillna(0) +
                            pd.to_numeric(lu['Shopping'], errors='coerce').fillna(0)).reindex(ext.index) / a
    dens['attract_food'] = (pd.to_numeric(lu['EATOUT_ovm'], errors='coerce').fillna(0) +
                            pd.to_numeric(lu['Entertainment_ovm'], errors='coerce').fillna(0)).reindex(ext.index) / a
    dens['attract_health'] = (pd.to_numeric(lu['Healthcare_ovm'], errors='coerce').fillna(0) +
                              pd.to_numeric(lu['Healthcare'], errors='coerce').fillna(0)).reindex(ext.index) / a
    dens['attract_edu'] = (pd.to_numeric(lu['Education_ovm'], errors='coerce').fillna(0) +
                           pd.to_numeric(lu['Education'], errors='coerce').fillna(0)).reindex(ext.index) / a
    dens['attract_rec'] = (pd.to_numeric(lu['n_parks'], errors='coerce').fillna(0) +
                           pd.to_numeric(lu['recreation_ground'], errors='coerce').fillna(0)).reindex(ext.index) / a
    dens['attract_work'] = jobs.reindex(ext.index) / a
    # transit intensity: stops x AM trips / headway
    stops = pd.to_numeric(lu['n_transit_stops'], errors='coerce').fillna(0).reindex(ext.index)
    trips = pd.to_numeric(lu['transit_trips_am'], errors='coerce').fillna(0).reindex(ext.index)
    head = pd.to_numeric(lu['transit_headway'], errors='coerce').fillna(np.nan).reindex(ext.index)
    dens['transit_intensity'] = stops * trips / head.replace(0, np.nan)
    # walkability: entropy + cycleway density + park share
    dens['lu_entropy'] = pd.to_numeric(lu['lu_entropy'], errors='coerce').reindex(ext.index)
    dens['cycleway_dens'] = pd.to_numeric(lu['cycleway_km'], errors='coerce').fillna(0).reindex(ext.index) / a
    dens['park_share'] = pd.to_numeric(lu['lu_area_park'], errors='coerce').fillna(0).reindex(ext.index) / a.replace(0, np.nan)
    dens['jobs_housing'] = pd.to_numeric(lu['jobs_housing_ratio'], errors='coerce').reindex(ext.index)

    # ── composite visitability per purpose ─────────────────────────────
    def z(s):
        s = pd.to_numeric(s, errors='coerce')
        return (np.log1p(s.fillna(0)) - np.log1p(s.fillna(0)).mean()) / \
            (np.log1p(s.fillna(0)).std() + 1e-9)

    safety_pen = (np.log1p(ext['crime_violent_per100k'].fillna(0)) +
                  np.log1p(ext['crash_per1000jobs_smooth'].fillna(0)))
    safety_pen = (safety_pen - safety_pen.mean()) / (safety_pen.std() + 1e-9)
    purpose_mix = {
        'work': ['attract_work', 'transit_intensity'],
        'edu': ['attract_edu', 'transit_intensity'],
        'shop': ['attract_shop', 'transit_intensity'],
        'eat': ['attract_food', 'transit_intensity'],
        'health': ['attract_health', 'transit_intensity'],
        'social': ['attract_rec', 'attract_food'],
    }
    for purp, cols in purpose_mix.items():
        # mean z-scored log-density of the purpose's attractors, penalised
        # by the safety z-score (violent crime + crash exposure).
        # lu_entropy stays a descriptor column, not part of the score.
        dens[f'visit_{purp}'] = sum(z(dens[c]) for c in cols) / len(cols) \
            - 0.5 * safety_pen

    out = pd.concat([ext, dens], axis=1)
    out_path = os.path.join(SAVED, 'accessibility_371_extended.csv')
    out.to_csv(out_path)
    print(f'wrote {out_path} ({out.shape})', flush=True)

    # ── summary ────────────────────────────────────────────────────────
    print('\n=== crime (allocated, last '
          f'{args.months} months) — top violent per 100k ===', flush=True)
    t = out.dropna(subset=['crime_violent_per100k']).sort_values(
        'crime_violent_per100k', ascending=False).head(10)
    for z, r in t.iterrows():
        try:
            nm = lu.loc[z, 'sa2_name']
        except KeyError:
            nm = '?'
        print(f'  {z} {nm}: violent={r.crime_violent_per100k:.0f}/100k '
              f'(murder={r.crime_murder_n:.1f}, assault={r.crime_assault_n:.0f}) '
              f'div={r.qps_division}', flush=True)
    print('\n=== murders (allocated counts, window) — top zones ===', flush=True)
    t = out.sort_values('crime_murder_n', ascending=False).head(10)
    for z, r in t.iterrows():
        try:
            nm = lu.loc[z, 'sa2_name']
        except KeyError:
            nm = '?'
        print(f'  {z} {nm}: murder_n={r.crime_murder_n:.2f} div={r.qps_division}',
              flush=True)
    print('\n=== visitability (work) — top/bottom 5 ===', flush=True)
    t = out.sort_values('visit_work', ascending=False)
    for z, r in list(t.head(5).iterrows()) + list(t.tail(5).iterrows()):
        try:
            nm = lu.loc[z, 'sa2_name']
        except KeyError:
            nm = '?'
        print(f'  {z} {nm}: {r.visit_work:+.2f}', flush=True)
    print(f'\nNaN crime rows (outside QLD): '
          f'{int(out["crime_violent_n"].isna().sum())}/{len(out)}', flush=True)

    # ── diagnostic plots ─────────────────────────────────────────────
    try:
        _make_plots(out, lu)
    except Exception as _e:
        print(f'plots skipped ({_e})', flush=True)


def _make_plots(out, lu):
    """Scatter maps (seq-centroid lon/lat) + top/bottom bars per index."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotdir = os.path.join(SAVED, 'plots_accessibility')
    os.makedirs(plotdir, exist_ok=True)

    seq = pd.read_csv(os.path.join(CODE_DIR, 'seq_zone_ids.csv'))
    seq['namekey'] = seq['SA2_NAME16'].astype(str).str.strip().str.lower()
    geo = {}
    for z in out.index:
        try:
            nm = str(lu.loc[z, 'sa2_name']).strip().lower()
        except KeyError:
            continue
        hit = seq[seq['namekey'] == nm]
        if len(hit):
            geo[z] = (float(hit.iloc[0]['longitude']),
                      float(hit.iloc[0]['latitude']))
    # alias zones reuse their parent centroid
    try:
        al = pd.read_csv(os.path.join(SAVED, 'sa2_centroid_aliases.csv'))
        for _, r in al.iterrows():
            z = int(r['zone_id'])
            if z not in geo:
                hit = seq[seq['SA2_NAME16'] == r['alias_seq_name']]
                if len(hit):
                    geo[z] = (float(hit.iloc[0]['longitude']),
                              float(hit.iloc[0]['latitude']))
    except Exception:
        pass
    gx = np.array([geo[z][0] for z in out.index if z in geo])
    gy = np.array([geo[z][1] for z in out.index if z in geo])
    keep = np.array([z for z in out.index if z in geo])
    print(f'plot geography: {len(keep)}/{len(out)} zones', flush=True)

    def _label(z):
        try:
            return str(lu.loc[z, 'sa2_name'])[:22]
        except KeyError:
            return str(z)

    maps = ['crime_violent_per100k', 'crime_murder_n',
            'crash_per1000jobs_smooth', 'transit_intensity',
            'visit_work', 'visit_shop', 'visit_social', 'attract_shop']
    for col in maps:
        v = pd.to_numeric(out.loc[keep, col], errors='coerce').values
        fin = np.isfinite(v)
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(gx[fin], gy[fin], c=np.log1p(np.maximum(v[fin], 0)),
                        s=28, cmap='inferno', edgecolors='none')
        ax.scatter(gx[~fin], gy[~fin], s=28, c='lightgrey',
                   label='no data', edgecolors='none')
        fig.colorbar(sc, ax=ax, label=f'log1p({col})')
        ax.set_title(f'SEQ 371 zones — {col} (n={int(fin.sum())})')
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_aspect('equal', adjustable='datalim')
        fig.tight_layout()
        fig.savefig(os.path.join(plotdir, f'map_{col}.png'), dpi=110)
        plt.close(fig)

    bars = {'crime_violent_per100k': 12, 'crime_murder_n': 12,
            'visit_work': 10, 'visit_social': 10}
    for col, n in bars.items():
        t = out.dropna(subset=[col]).sort_values(col, ascending=False)
        sel = list(t.head(n).index) + list(t.tail(n).index)
        vals = [float(out.loc[z, col]) for z in sel]
        labs = [_label(z) for z in sel]
        colors = ['crimson'] * n + ['steelblue'] * n
        fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * 2 * n)))
        ax.barh(range(2 * n - 1, -1, -1), vals, color=colors)
        ax.set_yticks(range(2 * n - 1, -1, -1))
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_title(f'{col} — top {n} (red) / bottom {n} (blue)')
        fig.tight_layout()
        fig.savefig(os.path.join(plotdir, f'bars_{col}.png'), dpi=110)
        plt.close(fig)
    print(f'plots written: {plotdir}', flush=True)


if __name__ == '__main__':
    main()
