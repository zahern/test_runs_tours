"""
build_synth_population.py
==========================
Assemble and enrich the full synthetic population from the per-zone CSV files
in saved_data/zone_ids_sa2/ and save the result to saved_data/synth_population.csv.

This script is a one-time (or re-run-on-demand) pre-processing step.  Once
synth_population.csv exists the pipeline can load it directly — much faster
than re-reading 329 zone CSVs every run.

What it does
------------
1. Reads every saved_data/zone_ids_sa2/order_N.csv  (one per SA2 zone).
2. Sets OTAZ = N from the filename.
3. Assigns a globally-unique person_id  (OTAZ * 1_000_000 + local_id)
   so IDs never collide across zones.
4. Calls enrich_synth_population() from synth_attributes.py to add:
       WORKSTAT, STUDYING, CARLICENCE, ptype, occp_group, HHSIZE
   and optionally attaches home-zone accessibility columns from landuse.
5. Saves the full enriched population to:
       saved_data/synth_population.csv
   and a compact persons-only version (no Census micro-columns) to:
       saved_data/synth_persons.csv

Usage
-----
    # Build from all 329 zones (default — takes ~2 min):
    python build_synth_population.py

    # Build from a subset of zones for testing:
    python build_synth_population.py --zones 40 124 200

    # Build without attaching accessibility (faster — no landuse/skims needed):
    python build_synth_population.py --no-accessibility

    # Specify a custom zone directory:
    python build_synth_population.py --synth-dir /path/to/zone_csvs

    # Specify output path:
    python build_synth_population.py --out saved_data/my_synth.csv

Outputs
-------
saved_data/synth_population.csv
    Full enriched population — all Census columns plus:
    person_id, OTAZ, WORKSTAT, STUDYING, CARLICENCE, ptype, occp_group,
    HHSIZE, AGEGROUP, vehrd (clipped), and optionally *_acc accessibility cols.

saved_data/synth_persons.csv
    Compact table — only the columns actually used by ABM stages:
    person_id, OTAZ, SA2_MAIN16, agep, age5p, sexp, WORKSTAT, STUDYING,
    CARLICENCE, ptype, occp_group, HHSIZE, vehrd, incp, bikes, carlicence
    and accessibility cols (if present).

saved_data/synth_population_summary.csv
    One row: total persons, zone count, ptype distribution, WORKSTAT distribution.
"""

import os
import re
import glob
import logging
import argparse
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SYNTH_DIR    = 'saved_data/zone_ids_sa2'
OUT_FULL     = 'saved_data/synth_population.csv'
OUT_PERSONS  = 'saved_data/synth_persons.csv'
OUT_SUMMARY  = 'saved_data/synth_population_summary.csv'

# Columns kept in the compact persons table (ABM-stage inputs only)
PERSONS_COLS = [
    'person_id', 'OTAZ', 'SA2_MAIN16',
    # age / sex
    'agep', 'age5p', 'AGEGROUP', 'sexp',
    # work / study attributes
    'WORKSTAT', 'STUDYING', 'CARLICENCE', 'carlicence',
    # person type (ABM segmentation)
    'ptype',
    # labour / occupation
    'occp_group', 'lfsp',
    # household
    'HHSIZE', 'vehrd', 'incp', 'nprd',
    # active transport
    'bikes',
    # household / person IDs (for join back)
    'abspid', 'abshid',
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_zone(filepath: str) -> pd.DataFrame:
    """Read one order_N.csv and set OTAZ from filename."""
    match = re.search(r'order_(\d+)', filepath)
    if not match:
        logger.warning(f"  Skipping {filepath} — cannot parse ORDER from filename")
        return None
    otaz = int(match.group(1))
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df['OTAZ'] = otaz
        return df
    except Exception as exc:
        logger.warning(f"  Failed to load {filepath}: {exc}")
        return None


def _assign_person_ids(synth: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a globally-unique person_id.

    Each zone CSV's abspid restarts from 0 so after concat they are duplicated
    across zones.  Build unique ids as OTAZ * 1_000_000 + local_id.
    """
    if 'person_id' in synth.columns and not synth['person_id'].duplicated().any():
        logger.info(f"  person_id already unique — keeping as-is")
        return synth

    id_src = next((c for c in ['abspid', 'PERSID'] if c in synth.columns), None)
    if id_src:
        synth['person_id'] = (
            synth['OTAZ'].astype(np.int64) * 1_000_000
            + synth[id_src].astype(np.int64)
        )
    else:
        synth['person_id'] = np.arange(len(synth), dtype=np.int64)

    n_unique = synth['person_id'].nunique()
    n_total  = len(synth)
    if n_unique < n_total:
        logger.warning(
            f"  {n_total - n_unique:,} duplicate person_ids after OTAZ * 1M + local_id — "
            f"falling back to row index"
        )
        synth['person_id'] = np.arange(n_total, dtype=np.int64)

    return synth


def _veh_bin(vehrd: pd.Series) -> pd.Series:
    """Household vehicles → 0/1/2+ bin used by mode choice frozen probs."""
    vr = vehrd.clip(upper=4).fillna(0).astype(int)
    bins = pd.cut(vr, bins=[-1, 0, 1, 99], labels=['0', '1', '2+'])
    return bins.astype(str)


def _income_band(incp: pd.Series) -> pd.Series:
    """
    ABS Census incp (1-19) → 3-band income label matching mode choice frozen probs.
    1-5  → low    (< $600/week)
    6-11 → medium ($600–$1799/week)
    12+  → high   (>= $1800/week)
    """
    ic = pd.to_numeric(incp, errors='coerce').fillna(0).astype(int)
    return pd.cut(ic, bins=[0, 5, 11, 99], labels=['low', 'medium', 'high']).astype(str)


def _age_group(age5p: pd.Series) -> pd.Series:
    """
    ABS age5p (5-year bands 1-18) → broad age group matching mode choice frozen probs.
    1-2  → child       (0-9)
    3-4  → youth       (10-19)
    5-10 → adult       (20-49)
    11-13 → middle_age  (50-64)
    14+  → senior       (65+)
    """
    ap = pd.to_numeric(age5p, errors='coerce').fillna(3).astype(int)
    conditions = [
        ap <= 2,
        (ap >= 3) & (ap <= 4),
        (ap >= 5) & (ap <= 10),
        (ap >= 11) & (ap <= 13),
        ap >= 14,
    ]
    choices = ['child', 'youth', 'adult', 'middle_age', 'senior']
    return np.select(conditions, choices, default='adult')


def build_synth_population(
    synth_dir:          str = SYNTH_DIR,
    zones:              list = None,
    attach_accessibility: bool = True,
    landuse:            pd.DataFrame = None,
    out_full:           str = OUT_FULL,
    out_persons:        str = OUT_PERSONS,
    out_summary:        str = OUT_SUMMARY,
    save:               bool = True,
    verbose:            bool = True,
) -> pd.DataFrame:
    """
    Assemble the full enriched synthetic population.

    Parameters
    ----------
    synth_dir            : folder containing order_N.csv files
    zones                : list of zone ORDER ints to load (None = all)
    attach_accessibility : join home-zone accessibility from landuse (requires
                          landuse with *_acc columns)
    landuse              : enriched landuse DataFrame — fetched automatically
                          if None and attach_accessibility is True
    out_full             : path for full CSV output
    out_persons          : path for compact persons CSV
    out_summary          : path for summary CSV
    save                 : write CSV files (True by default)
    verbose              : print progress

    Returns
    -------
    Enriched synth pop DataFrame.
    """
    from synth_attributes import enrich_synth_population

    # ── 1. Discover zone files ─────────────────────────────────────────────
    pattern = os.path.join(synth_dir, 'order_*.csv')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No order_*.csv files found in '{synth_dir}'.\n"
            f"  Expected: order_1.csv ... order_329.csv"
        )

    if zones is not None:
        zone_set = set(zones)
        files = [
            f for f in files
            if (m := re.search(r'order_(\d+)', f)) and int(m.group(1)) in zone_set
        ]
        logger.info(f"  Zone filter: {len(files)} files for zones {sorted(zone_set)[:10]}...")
    else:
        logger.info(f"  Found {len(files)} zone CSV files in {synth_dir}")

    # ── 2. Load and concatenate ────────────────────────────────────────────
    frames  = []
    missing = []
    for fp in files:
        df = _load_zone(fp)
        if df is not None:
            frames.append(df)
        else:
            m = re.search(r'order_(\d+)', fp)
            if m:
                missing.append(int(m.group(1)))

    if not frames:
        raise ValueError(f"No zone CSVs loaded from {synth_dir}")

    synth = pd.concat(frames, ignore_index=True)
    logger.info(f"  Raw concat: {len(synth):,} rows, {synth['OTAZ'].nunique()} zones")

    if missing:
        logger.warning(f"  Failed to load {len(missing)} zones: {missing[:20]}")

    # ── 3. Assign unique person_ids ────────────────────────────────────────
    synth = _assign_person_ids(synth)

    # ── 4. Load landuse if needed for accessibility ────────────────────────
    if attach_accessibility and landuse is None:
        try:
            from data_connect import Synth_Connect
            sc      = Synth_Connect()
            landuse = sc.get_land()
            logger.info("  Loaded landuse from Synth_Connect")
        except Exception as exc:
            logger.warning(
                f"  Could not load landuse ({exc}) — skipping accessibility attachment"
            )
            attach_accessibility = False

    # ── 5. Enrich population ───────────────────────────────────────────────
    synth = enrich_synth_population(
        synth,
        landuse = landuse if attach_accessibility else None,
    )

    # ── 6. Add mode-choice cell columns ───────────────────────────────────
    # These match the frozen-probability cell keys in stage 5 and stage 6
    if 'vehrd' in synth.columns:
        synth['veh_bin']     = _veh_bin(synth['vehrd'])
    if 'incp' in synth.columns:
        synth['income_band'] = _income_band(synth['incp'])
    if 'age5p' in synth.columns:
        synth['age_group']   = _age_group(synth['age5p'])

    # ── 7. Summary statistics ─────────────────────────────────────────────
    n_persons = len(synth)
    n_zones   = synth['OTAZ'].nunique()
    ptype_dist = synth['ptype'].value_counts().to_dict()
    ws_dist    = synth['WORKSTAT'].value_counts().to_dict()

    logger.info(
        f"  Enriched: {n_persons:,} persons | {n_zones} zones | "
        f"ptype: {ptype_dist}"
    )
    logger.info(f"  WORKSTAT: {ws_dist}")

    if verbose:
        print(f"\n  Synthetic population assembled:")
        print(f"    Persons     : {n_persons:,}")
        print(f"    Zones loaded: {n_zones}")
        print(f"    ptype distribution:")
        for pt, cnt in sorted(ptype_dist.items(), key=lambda x: -x[1]):
            print(f"        {pt:<20s}: {cnt:>8,}  ({cnt/n_persons*100:.1f}%)")
        print(f"    WORKSTAT (0=none, 1=FT, 2=PT): {ws_dist}")

    # ── 8. Save outputs ────────────────────────────────────────────────────
    if save:
        os.makedirs(os.path.dirname(out_full) or '.', exist_ok=True)

        # Full population
        synth.to_csv(out_full, index=False)
        logger.info(f"  Saved full population → {out_full}  ({os.path.getsize(out_full)/1e6:.1f} MB)")
        if verbose:
            print(f"\n  Saved: {out_full}  ({os.path.getsize(out_full)/1e6:.1f} MB)")

        # Compact persons table — only ABM-stage columns
        keep_cols = [c for c in PERSONS_COLS if c in synth.columns]
        # also keep any accessibility + mode-cell cols
        extra = [c for c in synth.columns
                 if c.endswith('_acc') or c in ('veh_bin', 'income_band', 'age_group')]
        keep_cols = list(dict.fromkeys(keep_cols + extra))  # preserve order, dedupe
        persons_df = synth[keep_cols]
        persons_df.to_csv(out_persons, index=False)
        logger.info(f"  Saved compact persons → {out_persons}")
        if verbose:
            print(f"  Saved: {out_persons}  ({len(keep_cols)} columns)")

        # Summary
        summary = {
            'n_persons':  n_persons,
            'n_zones':    n_zones,
            'n_files_loaded': len(frames),
            'n_files_missing': len(missing),
        }
        for pt, cnt in ptype_dist.items():
            summary[f'ptype_{pt}'] = cnt
        for ws, cnt in ws_dist.items():
            summary[f'WORKSTAT_{ws}'] = cnt
        pd.DataFrame([summary]).to_csv(out_summary, index=False)
        if verbose:
            print(f"  Saved: {out_summary}")

    return synth


# ---------------------------------------------------------------------------
# LOAD CACHED POPULATION
# ---------------------------------------------------------------------------

def load_synth_population(
    path: str = OUT_FULL,
    compact: bool = False,
) -> pd.DataFrame:
    """
    Load the pre-built enriched synthetic population from CSV.

    This is much faster than re-reading 329 zone files.
    Falls back to building from scratch if the file does not exist.

    Parameters
    ----------
    path    : path to synth_population.csv (or synth_persons.csv)
    compact : load the compact persons table instead of the full population
    """
    target = OUT_PERSONS if compact else path
    if not os.path.exists(target):
        logger.warning(
            f"  {target} not found — building from zone CSVs (first-time run)."
        )
        synth = build_synth_population(save=True)
        return synth[PERSONS_COLS] if compact else synth

    synth = pd.read_csv(target, low_memory=False)
    logger.info(f"  Loaded {len(synth):,} persons from {target}")
    return synth


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build and save the full enriched synthetic population.'
    )
    parser.add_argument(
        '--synth-dir',
        default=SYNTH_DIR,
        help=f'Folder with order_N.csv files (default: {SYNTH_DIR})',
    )
    parser.add_argument(
        '--zones',
        nargs='+',
        type=int,
        default=None,
        metavar='N',
        help='Restrict to specific zone ORDER numbers, e.g. --zones 40 124 200',
    )
    parser.add_argument(
        '--no-accessibility',
        action='store_true',
        help='Skip home-zone accessibility attachment (no landuse/skims needed)',
    )
    parser.add_argument(
        '--out',
        default=OUT_FULL,
        metavar='PATH',
        help=f'Output CSV path (default: {OUT_FULL})',
    )
    parser.add_argument(
        '--out-persons',
        default=OUT_PERSONS,
        metavar='PATH',
        help=f'Compact persons CSV path (default: {OUT_PERSONS})',
    )
    args = parser.parse_args()

    synth = build_synth_population(
        synth_dir            = args.synth_dir,
        zones                = args.zones,
        attach_accessibility = not args.no_accessibility,
        out_full             = args.out,
        out_persons          = args.out_persons,
        save                 = True,
        verbose              = True,
    )

    print(f"\nDone.  {len(synth):,} persons written.")
