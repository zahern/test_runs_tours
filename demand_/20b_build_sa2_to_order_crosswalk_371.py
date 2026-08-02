"""
20b_build_sa2_to_order_crosswalk_371.py
=======================================
Rebuild sa2_to_order_crosswalk.csv against the **371-zone SA2 system** so the
mapped-ORDER scope (NEW_POP_MAPPED=1) lines up with the 371 landuse + 371 skim.

The legacy demand_/sa2_to_order_crosswalk.csv maps SA2 -> the RETIRED 330 ORDER
(max ORDER 329, 324 rows). Using it while landuse/skim are 371 SA2 put the
population in a different zone id space (the 485 / '_altid_ {371,324}' errors).

This script targets code/saved_data/seq_zone_ids_371.csv (ORDER 1..371, one row
per SA2 in the 371 system, with centroids). Every SA2 is mapped to a 1..371
ORDER:
  * direct_371     — SA2 is one of the 371 zones -> its own ORDER (in_371=1)
  * via_nearest_371 — SA2 is outside the 371 set (fringe/external) -> the
                      geographically nearest 371 zone's ORDER (in_371=0)

Output schema is kept byte-compatible with the consumer
(build_synth_population.build_synth_population_mapped), which reads columns
`sa2_code`, `order` (int), and `in_324` (1 = in the modelled zone set):
    sa2_code, order, in_324, match_type, sa2_lat, sa2_lon
`in_324` here means "in the 371 modelled set" (kept under the old column name so
no consumer change is needed); `match_type` values gain the _371 suffix.

Writes demand_/sa2_to_order_crosswalk_371.csv (does NOT clobber the legacy file).
build_synth_population._DEFAULT_CROSSWALK prefers this 371 file when present.

Run (pandas/numpy only — no h5py/jax, works in the base env):
    cd Z:/test_runs_tours/demand_
    python 20b_build_sa2_to_order_crosswalk_371.py
"""
import os
import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
CODE_DIR  = os.path.join(HERE, '..', 'code')

ORDER_371_CSV = os.path.join(CODE_DIR, 'saved_data', 'seq_zone_ids_371.csv')
# SA2 lat/lon universe sources (unioned; first non-null centroid wins).
SA2_LATLON_SOURCES = [
    (os.path.join(HERE, 'sa2_crosswalk_507.csv'), 'sa2_code', 'sa2_lat', 'sa2_lon'),
    (os.path.join(HERE, 'sa2_to_order_crosswalk.csv'), 'sa2_code', 'sa2_lat', 'sa2_lon'),
]
OUT_CSV = os.path.join(HERE, 'sa2_to_order_crosswalk_371.csv')


def _haversine_nearest(lat0, lon0, lats, lons):
    """Index of the nearest (lats, lons) point to (lat0, lon0)."""
    lat0r, lats_r = np.radians(lat0), np.radians(lats)
    dlat = lats_r - lat0r
    dlon = np.radians(lons) - np.radians(lon0)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
    return int(np.argmin(a))


def main():
    # ── 1. 371 target zones (ORDER 1..371, SA2, centroid) ────────────────────
    z = pd.read_csv(ORDER_371_CSV)
    z['SA2_MAIN16'] = pd.to_numeric(z['SA2_MAIN16'], errors='coerce').astype('Int64')
    z['ORDER'] = pd.to_numeric(z['ORDER'], errors='coerce').astype(int)
    z = z.dropna(subset=['SA2_MAIN16', 'latitude', 'longitude'])
    sa2_to_order_371 = dict(zip(z['SA2_MAIN16'].astype(int), z['ORDER']))
    o_lat = z['latitude'].to_numpy(float)
    o_lon = z['longitude'].to_numpy(float)
    o_ord = z['ORDER'].to_numpy(int)
    set_371 = set(sa2_to_order_371)
    print(f"371 target zones: {len(set_371)} SA2  (ORDER {o_ord.min()}..{o_ord.max()})")

    # ── 2. SA2 universe + centroids (union of sources + the 371 set) ─────────
    latlon = {}   # sa2_code -> (lat, lon)
    for path, c_code, c_lat, c_lon in SA2_LATLON_SOURCES:
        if not os.path.exists(path):
            print(f"  (source missing, skipped: {os.path.basename(path)})")
            continue
        df = pd.read_csv(path, usecols=lambda c: c in (c_code, c_lat, c_lon))
        df = df.dropna(subset=[c_code, c_lat, c_lon])
        for code, la, lo in zip(pd.to_numeric(df[c_code], errors='coerce'),
                                df[c_lat], df[c_lon]):
            if pd.notna(code) and int(code) not in latlon:
                latlon[int(code)] = (float(la), float(lo))
        print(f"  + {os.path.basename(path)}: {len(df)} centroids "
              f"(universe now {len(latlon)})")
    # Ensure every 371 zone has a centroid (from the reference itself).
    for code, la, lo in zip(z['SA2_MAIN16'].astype(int), o_lat, o_lon):
        latlon.setdefault(int(code), (float(la), float(lo)))

    # ── 3. Map each SA2 -> a 1..371 ORDER ────────────────────────────────────
    rows = []
    n_direct = n_near = 0
    for sa2 in sorted(latlon):
        la, lo = latlon[sa2]
        if sa2 in set_371:
            rows.append((sa2, sa2_to_order_371[sa2], 1, 'direct_371', la, lo))
            n_direct += 1
        else:
            j = _haversine_nearest(la, lo, o_lat, o_lon)
            rows.append((sa2, int(o_ord[j]), 0, 'via_nearest_371', la, lo))
            n_near += 1

    out = pd.DataFrame(rows, columns=['sa2_code', 'order', 'in_324',
                                      'match_type', 'sa2_lat', 'sa2_lon'])
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(f"  {len(out)} SA2 rows -> {out['order'].nunique()} distinct ORDER "
          f"(min {out['order'].min()}, max {out['order'].max()})")
    print(f"  direct_371={n_direct}  via_nearest_371={n_near}")
    # Sanity: every ORDER target must be within 1..371.
    bad = out[(out['order'] < 1) | (out['order'] > len(set_371))]
    if len(bad):
        raise SystemExit(f"ERROR: {len(bad)} rows map outside 1..{len(set_371)} — "
                         "check seq_zone_ids_371.csv ORDER range.")
    print(f"  OK: all ORDER values within 1..{len(set_371)}")


if __name__ == '__main__':
    main()
