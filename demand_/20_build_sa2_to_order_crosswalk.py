"""
20_build_sa2_to_order_crosswalk.py
===================================
Build sa2_to_order_crosswalk.csv  — maps each of the 507 SA2 zones in the
new_pop synth pop to a 329-zone ORDER number.

This bridges two zone systems:
  - Population origins  : 507 SA2 zones (new_pop), 324 of which have complete
                          SA1/Lvl4 walk/bike skims (walk_bike_*_to_sa2_skims.h5)
  - Destination skims   : 329 ORDER zones (downcasted_matrices_base.omx)

Assignment rules, applied in order:
  1. Direct   — SA2_code == SA2_MAIN16 in seq_zone_ids_.csv  (307 zones)
  2. Nearest-ORDER — SA2 is in the 324 skim set but has no direct ORDER match
                     (17 Richmond-Tweed NSW zones); assigned to the geographically
                     nearest ORDER zone.
  3. Via-nearest-324 — SA2 is NOT in the 324 skim set (183 of 507 zones);
                        find the nearest 324-zone SA2 by lat/lon, then inherit
                        that zone's ORDER assignment.  This effectively merges
                        skim-uncovered population into the nearest covered zone.

Output: demand_/sa2_to_order_crosswalk.csv
  sa2_code    : 9-digit SA2 code (2021)
  order       : ORDER number (1-329/330) — OTAZ to use in pipeline
  in_324      : 1 if SA2 is in the 324-zone walk/bike skim set, else 0
  match_type  : 'direct' | 'nearest_order' | 'via_nearest_324'
  sa2_lat     : SA2 centroid latitude  (for reference / QA)
  sa2_lon     : SA2 centroid longitude

Usage
-----
    cd Z:/test_runs_tours/demand_
    python 20_build_sa2_to_order_crosswalk.py
"""
import os
import re
import h5py
import numpy as np
import pandas as pd

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
CODE_DIR  = os.path.join(OUT_DIR, '..', 'code')
ZONE_LIST = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"

SA2_XWALK_CSV   = os.path.join(OUT_DIR, 'sa2_crosswalk_507.csv')
ORDER_CSV        = os.path.join(CODE_DIR, 'seq_zone_ids_.csv')
SA1_SKIM_H5      = os.path.join(OUT_DIR, 'walk_bike_sa1_to_sa2_skims.h5')
CONCORDANCE_CSV  = os.path.join(OUT_DIR, 'sa2_2016_to_2021_concordance.csv')  # from 20b_*

OUT_CSV = os.path.join(OUT_DIR, 'sa2_to_order_crosswalk.csv')


def _haversine_min_idx(lat0, lon0, lats, lons):
    """Return index of nearest point in (lats, lons) to (lat0, lon0)."""
    lat0r = np.radians(lat0)
    lats_r = np.radians(lats)
    dlat = lats_r - lat0r
    dlon = np.radians(lons) - np.radians(lon0)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
    return int(np.argmin(a))


def main():
    # ── 1. Load 507 SA2 zone IDs ───────────────────────────────────────────
    with open(ZONE_LIST) as f:
        sa2_507 = sorted(int(x) for x in re.split(r'[\s,]+', f.read().strip()) if x)
    print(f"507-zone SA2 list: {len(sa2_507)} zones")

    # ── 2. Load lat/lon for 507 SA2 zones ─────────────────────────────────
    xwalk = pd.read_csv(SA2_XWALK_CSV, usecols=['sa2_code', 'sa2_lat', 'sa2_lon'])
    xwalk['sa2_code'] = xwalk['sa2_code'].astype(int)
    xwalk = xwalk.set_index('sa2_code')
    print(f"sa2_crosswalk_507: {len(xwalk)} SA2 lat/lon entries")

    # ── 3. Load 329 ORDER zones with lat/lon ──────────────────────────────
    order_df = pd.read_csv(ORDER_CSV)
    order_df = order_df[pd.to_numeric(order_df['SA2_MAIN16'], errors='coerce').fillna(0) > 0].copy()
    order_df['SA2_MAIN16'] = order_df['SA2_MAIN16'].astype(int)
    order_df['ORDER'] = order_df['ORDER'].astype(int)
    order_df = order_df.dropna(subset=['latitude', 'longitude'])
    sa2_to_order_direct = dict(zip(order_df['SA2_MAIN16'], order_df['ORDER']))
    order_lats = order_df['latitude'].values
    order_lons = order_df['longitude'].values
    order_ids  = order_df['ORDER'].values
    print(f"ORDER zones with lat/lon: {len(order_df)}")

    # ── 3b. Load 2016→2021 spatial concordance (from 20b_*.py) ───────────
    # Maps the 22 ORDER zones whose SA2_MAIN16 (2016 code) was split/renamed
    # in 2021 to their best-matching 2021 SA2 code (largest area overlap).
    # This lets us find those 2021 zones in the new_pop, rather than falling
    # back to haversine nearest-neighbour which ignores boundary geometry.
    sa2_2021_to_order_via_concordance: dict = {}  # 2021_code → ORDER
    n_concordance = 0
    if os.path.exists(CONCORDANCE_CSV):
        conc = pd.read_csv(CONCORDANCE_CSV)
        # For each unmatched 2016 ORDER code, look up its best 2021 equivalent
        order_sa2_map = dict(zip(order_df['SA2_MAIN16'], order_df['ORDER']))
        for _, row in conc.iterrows():
            sa2_16  = int(row['sa2_main16'])
            sa2_21  = int(row['sa2_code21'])
            if sa2_16 in order_sa2_map and sa2_21 not in sa2_to_order_direct:
                sa2_2021_to_order_via_concordance[sa2_21] = order_sa2_map[sa2_16]
                n_concordance += 1
        print(f"2016→2021 concordance loaded: {n_concordance} 2021 SA2 codes resolved "
              f"to ORDER via spatial overlap ({CONCORDANCE_CSV})")
    else:
        print(f"WARNING: concordance file not found ({CONCORDANCE_CSV}).\n"
              f"  Run 20b_build_sa2_2016_2021_concordance.py first for accurate\n"
              f"  2016→2021 mapping.  Falling back to haversine nearest-ORDER.")

    # ── 4. Load 324 SA2 codes from walk/bike skim ──────────────────────────
    with h5py.File(SA1_SKIM_H5, 'r') as f:
        skim_sa2 = set(int(c) for c in f['sa2_codes'][...])
    print(f"SA2 codes in 324-zone skim: {len(skim_sa2)}")

    # ── 5. Build crosswalk records ─────────────────────────────────────────
    records = []
    n_direct = n_nearest_order = n_via_324 = 0

    # Track 324-zone SA2s that have ORDER assignments (for via-nearest-324 step)
    sa2_to_order_324 = {}  # sa2_code → order (for the 324 set)

    # First pass: handle direct + nearest-ORDER (for SA2s in 324)
    for sa2_code in sorted(skim_sa2):
        row = xwalk.loc[sa2_code] if sa2_code in xwalk.index else None
        lat = float(row['sa2_lat']) if row is not None and not pd.isna(row['sa2_lat']) else np.nan
        lon = float(row['sa2_lon']) if row is not None and not pd.isna(row['sa2_lon']) else np.nan

        if sa2_code in sa2_to_order_direct:
            order = sa2_to_order_direct[sa2_code]
            match = 'direct'
            n_direct += 1
        elif sa2_code in sa2_2021_to_order_via_concordance:
            # 2021 SA2 is the spatial successor of a 2016 ORDER zone that changed codes
            order = sa2_2021_to_order_via_concordance[sa2_code]
            match = 'concordance'
            n_nearest_order += 1
        else:
            # Nearest ORDER zone by haversine (last resort — geographically closest)
            if not np.isnan(lat) and not np.isnan(lon):
                idx = _haversine_min_idx(lat, lon, order_lats, order_lons)
                order = int(order_ids[idx])
            else:
                order = int(order_ids[0])
            match = 'nearest_order'
            n_nearest_order += 1

        sa2_to_order_324[sa2_code] = order
        records.append({
            'sa2_code': sa2_code, 'order': order, 'in_324': 1,
            'match_type': match, 'sa2_lat': lat, 'sa2_lon': lon,
        })

    # For the 324-zone SA2s that have ORDER, build an array for nearest-324 lookup
    sa2_324_list  = sorted(sa2_to_order_324.keys())
    sa2_324_lats  = np.array([float(xwalk.loc[s]['sa2_lat'])
                               if s in xwalk.index else np.nan for s in sa2_324_list])
    sa2_324_lons  = np.array([float(xwalk.loc[s]['sa2_lon'])
                               if s in xwalk.index else np.nan for s in sa2_324_list])
    sa2_324_arr   = np.array(sa2_324_list)

    # Second pass: non-324 SA2 zones in the 507 set
    non_324 = [s for s in sa2_507 if s not in skim_sa2]
    print(f"Non-324 SA2 zones (will map via nearest-324): {len(non_324)}")

    for sa2_code in non_324:
        row = xwalk.loc[sa2_code] if sa2_code in xwalk.index else None
        lat = float(row['sa2_lat']) if row is not None and not pd.isna(row['sa2_lat']) else np.nan
        lon = float(row['sa2_lon']) if row is not None and not pd.isna(row['sa2_lon']) else np.nan

        # Find nearest 324-zone SA2 with valid coordinates
        valid = ~np.isnan(sa2_324_lats)
        if not np.isnan(lat) and not np.isnan(lon) and valid.any():
            idx = _haversine_min_idx(lat, lon, sa2_324_lats[valid], sa2_324_lons[valid])
            nearest_sa2 = int(sa2_324_arr[valid][idx])
        else:
            nearest_sa2 = int(sa2_324_arr[0])

        order = sa2_to_order_324[nearest_sa2]
        records.append({
            'sa2_code': sa2_code, 'order': order, 'in_324': 0,
            'match_type': 'via_nearest_324', 'sa2_lat': lat, 'sa2_lon': lon,
        })
        n_via_324 += 1

    out = pd.DataFrame(records).sort_values('sa2_code').reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\n=== CROSSWALK SUMMARY ===")
    print(f"  Total SA2 zones:           {len(out)}")
    print(f"  In 324 (direct match):     {n_direct}")
    print(f"  In 324 (nearest ORDER):    {n_nearest_order}")
    print(f"  Not in 324 (via nearest):  {n_via_324}")
    print(f"  Distinct ORDER zones used: {out['order'].nunique()}")
    print(f"  ORDER coverage:            {out['order'].nunique()}/330")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == '__main__':
    main()
