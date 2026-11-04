"""
run_accessibility_371.py
========================
Standalone driver to recompute zone-level accessibility over the 371-zone SA2
system, matching the 371 population/land use.

ZONE MIGRATION (371): the 507 skims (skims_371.omx) are keyed by
SA2 code, so accessibility runs in SA2-code space — land use zone ids must be
SA2 codes too. This driver loads the 371 land use (09b_build_landuse_357.py
output, keyed by SA2_MAIN16) and exposes SA2_MAIN16 as the `ORDER` column that
compute_accessibility keys on, so land use and skims align on SA2 code and the
OTAZ/DTAZ pivot/filter works.

Output: saved_data/accessibility_371.csv (one row per SA2 zone).

Note on walk/bike accessibility: the DISTWALK/DISTBIKE-based measures use the
skim file's own walk/bike matrices. Those are the placeholder (0.9×DIST) until
the real OSM walk/bike skim (21_build_osm_walk_bike_skims.py) is merged into the
507 skims; auto/transit accessibility is unaffected. Point --skims at a merged
skim to pick up the real active-mode distances.

Run (in zigenv; no internet needed):
  python run_accessibility_371.py \
      [--landuse ../new_pop/saved_data/zone_landuse_357.csv] \
      [--skims ../demand_/skims_371.omx] \
      [--out saved_data/accessibility_371.csv]
"""
import argparse
import os
import sys

import pandas as pd

DEMAND = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'demand_'))
NEWPOP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'new_pop'))
DEFAULT_LANDUSE = os.path.join(NEWPOP, 'saved_data', 'zone_landuse_357.csv')
DEFAULT_SKIMS = os.path.join(DEMAND, 'skims_371.omx')
DEFAULT_OUT = 'saved_data/accessibility_371.csv'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--landuse', default=os.environ.get('LANDUSE_371', DEFAULT_LANDUSE))
    ap.add_argument('--skims', default=os.environ.get('PIPELINE_SKIMS_PATH') or DEFAULT_SKIMS)
    ap.add_argument('--out', default=os.environ.get('ACCESS_OUT', DEFAULT_OUT))
    ap.add_argument('--plot-dir', default=None)
    args = ap.parse_args()

    for label, path in [('land use', args.landuse), ('skims', args.skims)]:
        if not os.path.exists(path):
            print(f"FATAL: {label} not found: {path}")
            if label == 'land use':
                print("       Build it first: qsub job_landuse_371.pbs "
                      "(demand_/09b_build_landuse_357.py)")
            sys.exit(2)

    from compute_accessibility import run_accessibility

    lu = pd.read_csv(args.landuse)
    # Zone-id column: 09b writes SA2_MAIN16. Expose it as ORDER so
    # compute_accessibility keys on SA2 code — the same id space as the skims.
    zid = next((c for c in ('SA2_MAIN16', 'sa2_code', 'SA2_CODE', 'ORDER')
                if c in lu.columns), None)
    if zid is None:
        print(f"FATAL: no zone-id column in {args.landuse} "
              f"(looked for SA2_MAIN16 / sa2_code / ORDER)")
        sys.exit(2)
    lu['ORDER'] = pd.to_numeric(lu[zid], errors='coerce').astype('Int64')
    lu = lu.dropna(subset=['ORDER']).copy()
    lu['ORDER'] = lu['ORDER'].astype(int)
    print(f"Land use: {len(lu)} zones (id column '{zid}' -> ORDER=SA2 code)")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    run_accessibility(lu, skims=args.skims, save_path=args.out, plot_dir=args.plot_dir)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
