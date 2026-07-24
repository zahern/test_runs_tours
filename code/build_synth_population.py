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

    # Build the new_pop 507-SA2-zone scope (zone_<sa2>.csv, OTAZ = SA2 code):
    python build_synth_population.py --new-pop

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

# new_pop / 507-SA2-zone scope (zone_<sa2>.csv files, OTAZ = SA2 code directly,
# no ORDER indirection). Override root with env var NEW_POP_DIR.
# Repo-relative default (a Z:\ literal only resolves on the Windows mount)
NEW_POP_DIR        = (os.environ.get('NEW_POP_DIR')
                      or os.path.normpath(os.path.join(
                          os.path.dirname(os.path.abspath(__file__)),
                          '..', 'new_pop')))
NEW_POP_SYNTH_DIR  = os.path.join(NEW_POP_DIR, 'saved_data', 'zone_ids_sa2')
NEW_POP_OUT_FULL    = os.path.join(NEW_POP_DIR, 'saved_data', 'synth_population.csv')
NEW_POP_OUT_PERSONS = os.path.join(NEW_POP_DIR, 'saved_data', 'synth_persons.csv')
NEW_POP_OUT_SUMMARY = os.path.join(NEW_POP_DIR, 'saved_data', 'synth_population_summary.csv')

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

def _parse_zone_id(filepath: str):
    """
    Parse the zone identifier from a zone CSV filename.

    Two conventions are supported:
      order_N.csv  -- legacy 329-zone scope, N is an arbitrary ORDER int.
      zone_S.csv   -- new_pop 507-SA2-zone scope, S is the SA2 code itself
                      (no ORDER indirection — see export_zone_sa2_files()
                      in synth_pop/run_pipeline.py).
    """
    match = re.search(r'order_(\d+)', filepath)
    if match:
        return int(match.group(1))
    match = re.search(r'zone_(\d+)', filepath)
    if match:
        return int(match.group(1))
    return None


def _load_zone(filepath: str) -> pd.DataFrame:
    """Read one zone CSV (order_N.csv or zone_<sa2>.csv) and set OTAZ from filename."""
    otaz = _parse_zone_id(filepath)
    if otaz is None:
        logger.warning(f"  Skipping {filepath} — cannot parse zone id from filename")
        return None
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
    # Supports both the legacy order_N.csv convention and the new_pop
    # zone_<sa2>.csv convention (see _parse_zone_id).
    files = sorted(glob.glob(os.path.join(synth_dir, 'order_*.csv'))
                    + glob.glob(os.path.join(synth_dir, 'zone_*.csv')))
    if not files:
        raise FileNotFoundError(
            f"No order_*.csv or zone_*.csv files found in '{synth_dir}'."
        )

    if zones is not None:
        zone_set = set(zones)
        files = [f for f in files if _parse_zone_id(f) in zone_set]
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
            zid = _parse_zone_id(fp)
            if zid is not None:
                missing.append(zid)

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
# 324-ZONE MAPPED POPULATION  (new_pop files → ORDER-based OTAZ)
# ---------------------------------------------------------------------------

#: Crosswalk produced by demand_/20_build_sa2_to_order_crosswalk.py
_DEFAULT_CROSSWALK = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', 'demand_', 'sa2_to_order_crosswalk.csv')
)

OUT_MAPPED_FULL    = 'saved_data/synth_population_mapped.csv'
OUT_MAPPED_PERSONS = 'saved_data/synth_persons_mapped.csv'
OUT_MAPPED_SUMMARY = 'saved_data/synth_population_mapped_summary.csv'


def build_synth_population_mapped(
    crosswalk_path: str = None,
    new_pop_dir:    str = None,
    out_full:       str = OUT_MAPPED_FULL,
    out_persons:    str = OUT_MAPPED_PERSONS,
    out_summary:    str = OUT_MAPPED_SUMMARY,
    attach_accessibility: bool = False,
    save:           bool = True,
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Load the new_pop 507-zone SA2 files, remap OTAZ from SA2 code → ORDER
    number using demand_/sa2_to_order_crosswalk.csv, and return an enriched
    synthetic population ready for the 329-zone pipeline.

    Non-324-zone SA2 people are merged into their nearest 324-zone neighbour's
    ORDER zone (crosswalk match_type='via_nearest_324').  Zones with zero
    persons in the new_pop CSV files are silently skipped.

    Run demand_/20_build_sa2_to_order_crosswalk.py first to generate the
    crosswalk.

    Parameters
    ----------
    crosswalk_path : path to sa2_to_order_crosswalk.csv
                     (default: ../demand_/sa2_to_order_crosswalk.csv)
    new_pop_dir    : root of new_pop scope
                     (default: NEW_POP_DIR env var or Z:/test_runs_tours/new_pop)
    out_full / out_persons / out_summary : output CSV paths
    attach_accessibility : join accessibility columns (requires skims; slow)
    save           : write output CSVs
    """
    from synth_attributes import enrich_synth_population

    # ── resolve paths ──────────────────────────────────────────────────────
    _xwalk_path = crosswalk_path or _DEFAULT_CROSSWALK
    if not os.path.exists(_xwalk_path):
        raise FileNotFoundError(
            f"Crosswalk not found: {_xwalk_path}\n"
            f"Run demand_/20_build_sa2_to_order_crosswalk.py first."
        )
    _np_dir  = new_pop_dir or os.environ.get('NEW_POP_DIR', NEW_POP_DIR)
    _np_zone = os.path.join(_np_dir, 'saved_data', 'zone_ids_sa2')
    if not os.path.isdir(_np_zone):
        raise FileNotFoundError(f"new_pop zone dir not found: {_np_zone}")

    # ── load crosswalk ─────────────────────────────────────────────────────
    xwalk = pd.read_csv(_xwalk_path)
    xwalk['sa2_code'] = xwalk['sa2_code'].astype(int)
    xwalk['order']    = xwalk['order'].astype(int)
    sa2_to_order  = dict(zip(xwalk['sa2_code'], xwalk['order']))
    sa2_in_324    = set(xwalk.loc[xwalk['in_324'] == 1, 'sa2_code'])
    logger.info(f"Crosswalk: {len(sa2_to_order)} SA2 zones → {xwalk['order'].nunique()} ORDER zones "
                f"({len(sa2_in_324)} in 324-zone skim set)")

    # ── discover zone_<sa2>.csv files ─────────────────────────────────────
    zone_files = sorted(glob.glob(os.path.join(_np_zone, 'zone_*.csv')))
    if not zone_files:
        raise FileNotFoundError(f"No zone_*.csv files in {_np_zone}")
    logger.info(f"Found {len(zone_files)} zone_<sa2>.csv files in {_np_zone}")

    # ── load and remap ─────────────────────────────────────────────────────
    frames  = []
    skipped_no_crosswalk = []
    skipped_empty        = []
    n_non324 = 0

    for fp in zone_files:
        sa2_code = _parse_zone_id(fp)   # reads the <sa2> part from filename
        if sa2_code is None:
            continue
        order = sa2_to_order.get(sa2_code)
        if order is None:
            skipped_no_crosswalk.append(sa2_code)
            continue
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as exc:
            logger.warning(f"  Failed to read {fp}: {exc}")
            continue
        if len(df) == 0:
            skipped_empty.append(sa2_code)
            continue

        df['OTAZ']     = order          # remap to ORDER number
        df['HOME_SA2'] = sa2_code       # keep original SA2 for reference
        if sa2_code not in sa2_in_324:
            df['_in324'] = 0
            n_non324 += len(df)
        else:
            df['_in324'] = 1
        frames.append(df)

    if not frames:
        raise ValueError("No zone CSVs loaded — check new_pop_dir and crosswalk.")

    # ── Fill empty SA2 zones from legacy 329-zone order_N.csv files ────────
    # Some SA2 zones in the new_pop have zero people in their CSV (skipped_empty).
    # For those ORDER zones, fall back to the legacy per-zone files so the
    # destination-choice model still sees people in every ORDER zone.
    if skipped_empty:
        for _empty_sa2 in skipped_empty:
            _empty_order = sa2_to_order.get(_empty_sa2)
            if _empty_order is None:
                continue
            _legacy_path = os.path.join(SYNTH_DIR, f'order_{_empty_order}.csv')
            if os.path.exists(_legacy_path):
                try:
                    _df = pd.read_csv(_legacy_path, low_memory=False)
                    if len(_df) > 0:
                        _df['OTAZ']     = _empty_order
                        _df['HOME_SA2'] = _empty_sa2
                        _df['_in324']   = 1 if _empty_sa2 in sa2_in_324 else 0
                        frames.append(_df)
                        logger.info(
                            f"  Filled empty SA2 {_empty_sa2} → ORDER {_empty_order} "
                            f"from legacy {_legacy_path} ({len(_df):,} persons)"
                        )
                except Exception as _leg_exc:
                    logger.warning(
                        f"  Legacy fallback failed for SA2 {_empty_sa2} "
                        f"ORDER {_empty_order}: {_leg_exc}"
                    )
            else:
                logger.warning(
                    f"  No legacy fallback for empty SA2 {_empty_sa2} "
                    f"(ORDER {_empty_order}) — {_legacy_path} not found; zone left blank"
                )

    synth = pd.concat(frames, ignore_index=True)

    # Assign person_id using HOME_SA2 (original SA2 code, unique per source file)
    # NOT OTAZ (ORDER), because multiple SA2s can map to the same ORDER zone which
    # causes OTAZ * 1M + abspid collisions for people whose abspid values overlap.
    id_src = next((c for c in ['abspid', 'PERSID'] if c in synth.columns), None)
    if id_src:
        synth['person_id'] = (
            synth['HOME_SA2'].astype(np.int64) * 1_000_000
            + synth[id_src].astype(np.int64)
        )
        if synth['person_id'].duplicated().any():
            n_dup = synth['person_id'].duplicated().sum()
            logger.warning(f"  {n_dup:,} duplicate person_ids even with HOME_SA2 — falling back to row index")
            synth['person_id'] = np.arange(len(synth), dtype=np.int64)
        else:
            logger.info(f"  person_id: {synth['person_id'].nunique():,} unique (HOME_SA2 × 1M + {id_src})")
    else:
        synth['person_id'] = np.arange(len(synth), dtype=np.int64)

    n_in324 = int((synth['_in324'] == 1).sum())
    logger.info(
        f"Loaded {len(synth):,} persons | {synth['OTAZ'].nunique()} ORDER zones\n"
        f"  In-324 persons      : {n_in324:,}\n"
        f"  Non-324 (remapped)  : {n_non324:,}\n"
        f"  Zones no crosswalk  : {len(skipped_no_crosswalk)}\n"
        f"  Zones empty (zero)  : {len(skipped_empty)}"
    )
    if skipped_no_crosswalk:
        logger.warning(f"  SA2 codes with no crosswalk entry (skipped): {skipped_no_crosswalk[:10]}")
    if skipped_empty:
        logger.info(f"  Empty SA2 zones (zero pop — left blank): {skipped_empty}")

    # drop internal flag column
    synth.drop(columns=['_in324'], inplace=True, errors='ignore')

    # ── enrich ────────────────────────────────────────────────────────────
    landuse = None
    if attach_accessibility:
        try:
            from data_connect import Synth_Connect
            landuse = Synth_Connect().get_land()
        except Exception as exc:
            logger.warning(f"Could not load landuse for accessibility: {exc}")

    synth = enrich_synth_population(synth, landuse=landuse)

    if 'vehrd' in synth.columns:
        synth['veh_bin']     = _veh_bin(synth['vehrd'])
    if 'incp' in synth.columns:
        synth['income_band'] = _income_band(synth['incp'])
    if 'age5p' in synth.columns:
        synth['age_group']   = _age_group(synth['age5p'])

    n_persons = len(synth)
    n_zones   = synth['OTAZ'].nunique()
    ptype_dist = synth['ptype'].value_counts().to_dict()

    if verbose:
        print(f"\n  324-zone mapped synthetic population:")
        print(f"    Persons       : {n_persons:,}")
        print(f"    ORDER zones   : {n_zones}")
        print(f"    In-324 zones  : {len(sa2_in_324)}")
        print(f"    Non-324 (merged) persons: {n_non324:,}")
        print(f"    Empty SA2 zones (zero pop): {len(skipped_empty)}")
        for pt, cnt in sorted(ptype_dist.items(), key=lambda x: -x[1]):
            print(f"      {pt:<20s}: {cnt:>8,}  ({cnt/n_persons*100:.1f}%)")

    # ── save ──────────────────────────────────────────────────────────────
    if save:
        os.makedirs(os.path.dirname(out_full) or '.', exist_ok=True)
        synth.to_csv(out_full, index=False)
        logger.info(f"Saved mapped population → {out_full}")

        keep_cols = [c for c in PERSONS_COLS + ['HOME_SA2'] if c in synth.columns]
        extra = [c for c in synth.columns
                 if c.endswith('_acc') or c in ('veh_bin', 'income_band', 'age_group')]
        keep_cols = list(dict.fromkeys(keep_cols + extra))
        synth[keep_cols].to_csv(out_persons, index=False)
        logger.info(f"Saved compact mapped persons → {out_persons}")

        pd.DataFrame([{
            'n_persons': n_persons, 'n_order_zones': n_zones,
            'n_in_324': len(sa2_in_324), 'n_non324_persons': n_non324,
            'n_empty_zones': len(skipped_empty),
            **{f'ptype_{k}': v for k, v in ptype_dist.items()}
        }]).to_csv(out_summary, index=False)

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
    parser.add_argument(
        '--new-pop',
        action='store_true',
        help=(f'Use the new_pop 507-SA2-zone scope: reads zone_<sa2>.csv from '
              f'{NEW_POP_SYNTH_DIR} and writes outputs under {NEW_POP_DIR}\\saved_data\\ '
              f'(overrides --synth-dir/--out/--out-persons unless explicitly given)'),
    )
    parser.add_argument(
        '--new-pop-mapped',
        action='store_true',
        help=(f'Load new_pop 507-zone CSVs and remap OTAZ from SA2 code → ORDER '
              f'number via demand_/sa2_to_order_crosswalk.csv.  Non-324-zone people '
              f'are merged into their nearest 324-zone ORDER neighbour.  Destination '
              f'skims stay on the 329-zone ORDER system.  Run '
              f'demand_/20_build_sa2_to_order_crosswalk.py first.'),
    )
    parser.add_argument(
        '--crosswalk',
        default=None,
        metavar='PATH',
        help='Path to sa2_to_order_crosswalk.csv (--new-pop-mapped only; '
             f'default: {_DEFAULT_CROSSWALK})',
    )
    args = parser.parse_args()

    if args.new_pop_mapped:
        synth = build_synth_population_mapped(
            crosswalk_path       = args.crosswalk,
            new_pop_dir          = os.environ.get('NEW_POP_DIR', NEW_POP_DIR),
            out_full             = args.out if args.out != OUT_FULL else OUT_MAPPED_FULL,
            out_persons          = args.out_persons if args.out_persons != OUT_PERSONS else OUT_MAPPED_PERSONS,
            attach_accessibility = not args.no_accessibility,
            save                 = True,
            verbose              = True,
        )
    else:
        synth_dir   = NEW_POP_SYNTH_DIR if args.new_pop and args.synth_dir == SYNTH_DIR else args.synth_dir
        out_full    = NEW_POP_OUT_FULL if args.new_pop and args.out == OUT_FULL else args.out
        out_persons = NEW_POP_OUT_PERSONS if args.new_pop and args.out_persons == OUT_PERSONS else args.out_persons
        out_summary = NEW_POP_OUT_SUMMARY if args.new_pop else OUT_SUMMARY

        synth = build_synth_population(
            synth_dir            = synth_dir,
            zones                = args.zones,
            attach_accessibility = not args.no_accessibility,
            out_full             = out_full,
            out_persons          = out_persons,
            out_summary          = out_summary,
            save                 = True,
            verbose              = True,
        )

    print(f"\nDone.  {len(synth):,} persons written.")
