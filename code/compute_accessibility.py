"""
compute_accessibility.py
=========================
Zone-level accessibility using the actual QTS/SEQ landuse columns.

Landuse columns available (from zone_order_and_landuse.csv):
  Employment:   RETEMPN, FPSEMPN, HEREMPN, AGREMPN, MWTEMPN, OTHEMP
  Education:    Primary, Secondary, Education (tertiary), HSENROLL, AGE0519
  Activities:   EATOUT, Shopping, Entertainment, ESCORT,
                OTHERMAINTENANCE, OTHERDISCRFacilities, Healthcare
  Land area:    residential, commercial, retail, industrial,
                recreation_ground, construction, sa2_area
  Population:   TOTPOP, TOTHH, POPULATION

Key design choices (informed by data analysis):
  - Employment cols are highly correlated (r≥0.88 with TOTEMP) so we use
    TOTEMP for generic work accessibility and add industry-specific bonuses
  - Activity counts (EATOUT, Shopping etc.) are facility counts, low-coverage
    (55-95%) but meaningful for non-mandatory destination choice
  - Land area columns (residential m², retail m²) are continuous measures
    of land use intensity — better for density-based measures than counts
  - Population density derived from residential/sa2_area correlates with
    non-mandatory activity generation
"""

import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ACCESSIBILITY SPECIFICATION
# Uses actual landuse columns from zone_order_and_landuse.csv
# =============================================================================

ACCESSIBILITY_SPEC = {

    # ── Auto work accessibility ───────────────────────────────────────────────
    # Attractor: total employment (TOTEMP = sum of all emp categories)
    # High correlation (r≥0.88) means TOTEMP is a valid aggregate
    'auto_work_acc': {
        'attractor_cols': {
            'TOTEMP':  1.0,   # total employment — computed as sum of below
            'FPSEMPN': 0.15,  # finance/professional bonus (white-collar)
            'HEREMPN': 0.10,  # healthcare bonus
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.05,
        'cap_tt':      120.0,
    },

    # ── Auto education accessibility ──────────────────────────────────────────
    # Uses actual enrolment counts from landuse (Primary, Secondary, Education)
    # Education = tertiary facilities count; Primary/Secondary = enrolment
    'auto_edu_acc': {
        'attractor_cols': {
            'Primary':   1.0,   # primary school enrolment
            'Secondary': 1.0,   # secondary school enrolment
            'Education': 50.0,  # tertiary facilities (scale up — count vs enrolment)
            'HSENROLL':  0.5,   # high school enrolment (supplement)
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.06,
        'cap_tt':      90.0,
    },

    # ── Auto shopping accessibility ───────────────────────────────────────────
    # Uses Shopping (facility count) and retail land area (m²)
    # retail area is continuous and highly correlated with shopping activity
    'auto_shop_acc': {
        'attractor_cols': {
            'Shopping': 10.0,   # facility count (scale up to match retail area)
            'retail':   0.001,  # retail floor area m² (normalise to comparable scale)
            'RETEMPN':  0.5,    # retail employment
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.08,
        'cap_tt':      60.0,
    },

    # ── Auto non-mandatory accessibility ─────────────────────────────────────
    # Composite of eat/entertainment/maintenance/discretionary facilities
    'auto_nonmand_acc': {
        'attractor_cols': {
            'EATOUT':               5.0,   # restaurant/cafe count
            'Entertainment':        3.0,   # entertainment venues
            'Shopping':             3.0,   # shopping facilities
            'OTHERMAINTENANCE':     2.0,   # maintenance services
            'OTHERDISCRFacilities': 2.0,  # discretionary facilities
            'Healthcare':           2.0,   # healthcare facilities
            'retail':              0.0005, # retail area contribution
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.07,
        'cap_tt':      60.0,
    },

    # ── Walk accessibility ─────────────────────────────────────────────────────
    # Walk focuses on local density: residential area + retail + EATOUT
    # residential m² is a strong indicator of walkable neighbourhood density
    'walk_acc': {
        'attractor_cols': {
            'EATOUT':      3.0,
            'Shopping':    3.0,
            'retail':     0.0003,
            'residential': 0.000005,  # residential floor area m²
            'TOTPOP':      0.01,
        },
        'skim_col':    'DISTWALK',
        'decay_type':  'logit',
        'decay_coeff': 1.5,
        'cap_tt':      7.5,
    },

    # ── Bike accessibility ────────────────────────────────────────────────────
    'bike_acc': {
        'attractor_cols': {
            'TOTEMP':  0.5,
            'retail':  0.0002,
            'EATOUT':  2.0,
        },
        'skim_col':    'DISTBIKE',
        'decay_type':  'logit',
        'decay_coeff': 0.4,
        'cap_tt':      25.0,
    },

    # ── Transit accessibility ─────────────────────────────────────────────────
    # Transit reaches employment centres — use TOTEMP with gentle decay
    'transit_acc': {
        'attractor_cols': {
            'TOTEMP':      1.0,
            'commercial': 0.000005,  # commercial area as CBD proxy
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.025,
        'cap_tt':      150.0,
    },

    # ── Recreation accessibility ──────────────────────────────────────────────
    # Parks, recreation grounds, sport — relevant for social/othdiscr trips
    'recreation_acc': {
        'attractor_cols': {
            'recreation_ground': 0.00001,
            'Entertainment':     3.0,
            'park':              0.001,
            'sport':             0.01,
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.06,
        'cap_tt':      45.0,
    },
}


# =============================================================================
# DERIVED LANDUSE COLUMNS
# Computed from raw columns before accessibility calculation
# =============================================================================

def enrich_landuse(landuse: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to landuse that improve accessibility measures.

    Derived:
        TOTEMP          total employment (sum of 6 categories)
        pop_density     population per km² (TOTPOP / sa2_area * 1e6)
        emp_density     employment per km² (TOTEMP / sa2_area * 1e6)
        res_density     residential floor area fraction (residential / sa2_area)
        school_enrol    combined primary + secondary enrolment
        activity_count  sum of all facility counts
    """
    lu = landuse.copy()

    # total employment — not in raw file, must be computed
    emp_cols = ['RETEMPN', 'FPSEMPN', 'HEREMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP']
    lu['TOTEMP'] = lu[[c for c in emp_cols if c in lu.columns]].sum(axis=1)

    # density measures (per km² where sa2_area is in m²)
    area = lu['sa2_area'].clip(lower=1e4)  # minimum 1 ha to avoid division noise
    lu['pop_density'] = lu['TOTPOP']       / area * 1e6
    lu['emp_density'] = lu['TOTEMP']       / area * 1e6
    lu['res_density'] = lu['residential']  / area  # fraction 0-1

    # combined school enrolment
    lu['school_enrol'] = (
        lu.get('Primary',   pd.Series(0, index=lu.index)).fillna(0) +
        lu.get('Secondary', pd.Series(0, index=lu.index)).fillna(0)
    )

    # total activity facility count
    act_cols = ['EATOUT','Shopping','Entertainment','ESCORT',
                'OTHERMAINTENANCE','OTHERDISCRFacilities','Healthcare']
    lu['activity_count'] = lu[[c for c in act_cols if c in lu.columns]].fillna(0).sum(axis=1)

    return lu


# =============================================================================
# DECAY FUNCTIONS
# =============================================================================

def _decay(tt: np.ndarray, decay_type: str, b: float) -> np.ndarray:
    tt = np.asarray(tt, dtype=np.float32)
    if decay_type == 'exp':
        return np.exp(-b * tt)
    elif decay_type == 'logit':
        return 1.0 / (1.0 + np.exp(b * tt))
    elif decay_type == 'power':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(tt > 0, tt ** (-b), 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_accessibility(
    landuse: pd.DataFrame,
    skim_df: pd.DataFrame,
    spec: dict = None,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Compute zone-level accessibility measures.

    ActivitySim pattern:
      access(o) = log1p( sum_d [ attractor(d) × decay(tt_od) ] )
    """
    if spec is None:
        spec = ACCESSIBILITY_SPEC

    # enrich landuse with derived columns
    lu = enrich_landuse(landuse)

    # zone index
    lu_col = 'ORDER' if 'ORDER' in lu.columns else lu.index.name
    if lu_col in lu.columns:
        zone_ids = lu[lu_col].astype(int).values
        lu = lu.set_index(lu_col)
    else:
        zone_ids = lu.index.astype(int).values

    zone_ids = zone_ids[zone_ids != 330]
    zone_ids = np.sort(zone_ids)
    n_zones  = len(zone_ids)
    zone_idx = {int(z): i for i, z in enumerate(zone_ids)}

    print(f"[Accessibility] {n_zones} zones × {len(spec)} measures")

    # filter skims to our zone universe
    sk = skim_df.copy()
    sk['OTAZ'] = sk['OTAZ'].astype(int)
    sk['DTAZ'] = sk['DTAZ'].astype(int)
    sk = sk[sk['OTAZ'].isin(zone_idx) & sk['DTAZ'].isin(zone_idx)]

    # pivot each needed skim column to (n_zones × n_zones) matrix
    needed_cols = {entry['skim_col'] for entry in spec.values()}
    skim_matrices = {}
    for col in needed_cols:
        if col not in sk.columns:
            logger.warning(f"  skim '{col}' missing — using zeros")
            skim_matrices[col] = np.zeros((n_zones, n_zones), dtype=np.float32)
            continue
        piv = (sk.pivot_table(index='OTAZ', columns='DTAZ',
                               values=col, aggfunc='first')
                 .reindex(index=zone_ids, columns=zone_ids)
                 .fillna(0).values.astype(np.float32))

        # unit detection
        nonzero = piv[piv > 0]
        if len(nonzero):
            median_val = float(np.median(nonzero))
            if median_val > 600:
                piv = piv / 60.0
                logger.info(f"  '{col}': converted seconds → minutes (median was {median_val:.0f})")
            elif median_val > 200:
                piv = piv / 10.0
                logger.info(f"  '{col}': converted tenths → minutes")
            print(f"  Skim '{col}': "
                  f"median={np.median(nonzero):.1f}  "
                  f"p95={float(np.percentile(nonzero,95)):.1f}  "
                  f"(units: {'min' if 'TIME' in col else 'km'})")

        skim_matrices[col] = piv

    # compute each measure
    results = {}
    for acc_name, entry in spec.items():
        att_cols   = entry['attractor_cols']
        skim_col   = entry['skim_col']
        decay_type = entry['decay_type']
        b          = entry['decay_coeff']
        cap_tt     = entry.get('cap_tt', 999.0)

        # build attractor vector (n_dest,)
        attractor = np.zeros(n_zones, dtype=np.float32)
        for col, weight in att_cols.items():
            if col in lu.columns:
                vals = (lu.reindex(zone_ids)[col]
                          .fillna(0).clip(lower=0).values
                          .astype(np.float32))
                attractor += weight * vals
            else:
                logger.debug(f"  '{acc_name}': attractor col '{col}' not in landuse")

        attractor = np.maximum(attractor, 0.0)

        tt_mat  = np.minimum(skim_matrices[skim_col].copy(), cap_tt)
        weights = _decay(tt_mat, decay_type, b)

        # log1p of weighted sum — ActivitySim _accumulate_accessibility
        acc_vals = np.log1p((weights * attractor[np.newaxis, :]).sum(axis=1))
        results[acc_name] = acc_vals

        print(f"  {acc_name:20s}: "
              f"min={acc_vals.min():.3f}  "
              f"mean={acc_vals.mean():.3f}  "
              f"max={acc_vals.max():.3f}")

    acc_df = pd.DataFrame(results, index=zone_ids)
    acc_df.index.name = 'zone_id'

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        acc_df.to_csv(save_path)
        print(f"[Accessibility] saved → {save_path}")

    return acc_df


def attach_to_landuse(landuse: pd.DataFrame,
                      acc_df: pd.DataFrame) -> pd.DataFrame:
    """Merge accessibility measures into the landuse DataFrame."""
    lu = enrich_landuse(landuse)  # also add TOTEMP etc.

    lu_col = 'ORDER' if 'ORDER' in lu.columns else None
    if lu_col:
        lu = lu.merge(
            acc_df.reset_index().rename(columns={'zone_id': lu_col}),
            on=lu_col, how='left')
    else:
        lu = lu.join(acc_df, how='left')

    for col in acc_df.columns:
        if col in lu.columns:
            med = lu[col].median()
            lu[col] = lu[col].fillna(med if not pd.isna(med) else 0.0)

    print(f"[Accessibility] attached {len(acc_df.columns)} accessibility + "
          f"{5} derived columns to landuse")
    return lu


def run_accessibility(
    landuse: pd.DataFrame,
    skims,
    spec: dict = None,
    save_path: str = 'saved_data/accessibility.csv',
) -> pd.DataFrame:
    from stage5_destination_choice import _skim_flat, _open_skims

    print("=" * 60)
    print("ACCESSIBILITY — Zone-level access measures")
    print("=" * 60)

    skims, close_it = _open_skims(skims)
    print("\nReading skims...")
    skim_df = _skim_flat(skims)
    print(f"  {len(skim_df)} OD pairs, {len(skim_df.columns)-2} skim columns")

    acc_df = compute_accessibility(landuse, skim_df, spec=spec, save_path=save_path)
    landuse_enriched = attach_to_landuse(landuse, acc_df)

    if close_it:
        skims.close()

    print("\nAccessibility complete.")
    return landuse_enriched
