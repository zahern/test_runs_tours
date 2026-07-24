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

    # ZONE MIGRATION (371): ORDER 330 was a dummy/external zone in the retired
    # 330-zone system; in the 371-zone SA2 system every ORDER is a real zone, so
    # the old hardcoded `!= 330` drop is gone. Any genuine external stations to
    # exclude can be listed in EXCLUDE_ZONES (comma-separated ORDER ids).
    _excl = os.environ.get('EXCLUDE_ZONES', '').strip()
    if _excl:
        _excl_ids = {int(x) for x in _excl.split(',') if x.strip()}
        zone_ids = zone_ids[~np.isin(zone_ids, list(_excl_ids))]
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

    # ── crash-exposure accessibility (safety as an upstream determinant) ──
    # ATRF reviewers: safety should enter decision layers beyond mode/
    # destination utilities. crash_acc summarises the expected crash
    # frequency a resident of zone i is exposed to across reachable
    # destinations, time-decayed like every other measure:
    #     crash_acc_i = log1p( Σ_j exp(-b·tt_ij) · CRASH_ij )
    # Unlike the spec-driven measures the "attractor" here is the OD-level
    # CRASH skim itself (crash exposure is a property of the OD movement,
    # not of the destination zone), hence the dedicated block. Joining this
    # onto persons via their home zone lets the GA test crash exposure in
    # CDAP, activity frequency, and duration models - not just in the
    # mode/destination utilities where the CRASH term already lives.
    if 'CRASH' in sk.columns:
        try:
            _crash_piv = np.zeros((n_zones, n_zones), dtype=np.float32)
            _oi = sk['OTAZ'].map(zone_idx).values
            _di = sk['DTAZ'].map(zone_idx).values
            _crash_piv[_oi, _di] = pd.to_numeric(
                sk['CRASH'], errors='coerce').fillna(0).clip(lower=0).values
            _tt_col = ('SOVTOLL_TIME__AM' if 'SOVTOLL_TIME__AM' in skim_matrices
                       else next(iter(skim_matrices)))
            _w = _decay(np.minimum(skim_matrices[_tt_col], 90.0), 'exp', 0.05)
            crash_vals = np.log1p((_w * _crash_piv).sum(axis=1))
            results['crash_acc'] = crash_vals
            print(f"  {'crash_acc':20s}: min={crash_vals.min():.3f}  "
                  f"mean={crash_vals.mean():.3f}  max={crash_vals.max():.3f}")
        except Exception as _ce:
            logger.warning(f"crash_acc computation failed (non-fatal): {_ce}")

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


# Accessibility measure → the landuse attractor that most directly drives it.
# Used by plot_accessibility to show the acc↔landuse linkage explicitly.
_ACC_LANDUSE_PAIRS = [
    ('auto_work_acc',   'TOTEMP',     'total employment'),
    ('auto_edu_acc',    'Education',  'education facilities'),
    ('auto_shop_acc',   'Shopping',   'shopping attractors'),
    ('walk_acc',        'retail',     'retail landuse area'),
    ('bike_acc',        'TOTEMP',     'total employment'),
    ('transit_acc',     'TOTEMP',     'total employment'),
    ('recreation_acc',  'park',       'park landuse area'),
    ('auto_nonmand_acc', 'TOTPOP',    'population'),
]


def plot_accessibility(landuse_enriched: pd.DataFrame,
                       acc_cols: list,
                       plot_dir: str) -> None:
    """
    Two diagnostic figures for the zone-level accessibility measures:

    accessibility_distributions.png — per-measure histogram across zones, so
        a degenerate measure (all zones alike → useless in a choice model)
        is visible at a glance.
    accessibility_vs_landuse.png — each measure scattered against the landuse
        attractor that feeds it (Pearson r annotated). Low r flags a measure
        whose decay/attractor wiring isn't actually transmitting the landuse
        signal.

    Non-fatal: any failure logs and returns (plots must never kill a run).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)
        cols = [c for c in acc_cols if c in landuse_enriched.columns]
        if not cols:
            return

        # ── 1. distributions ─────────────────────────────────────────────
        ncols = 4
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle('Zone accessibility measures — distribution across zones',
                     fontweight='bold')
        for i, c in enumerate(cols):
            v = pd.to_numeric(landuse_enriched[c], errors='coerce').dropna()
            axes[i].hist(v, bins=30, color='#2E86AB', alpha=0.85)
            axes[i].set_title(f'{c}\nmean={v.mean():.2f}  std={v.std():.2f}', fontsize=9)
            axes[i].grid(alpha=0.3)
        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        p1 = os.path.join(plot_dir, 'accessibility_distributions.png')
        fig.savefig(p1, dpi=110)
        plt.close(fig)
        print(f"[Accessibility] plot → {p1}")

        # ── 2. accessibility vs landuse attractors ───────────────────────
        pairs = [(a, lc, lbl) for a, lc, lbl in _ACC_LANDUSE_PAIRS
                 if a in landuse_enriched.columns and lc in landuse_enriched.columns]
        if pairs:
            nrows2 = int(np.ceil(len(pairs) / ncols))
            fig, axes = plt.subplots(nrows2, ncols, figsize=(4 * ncols, 3.2 * nrows2))
            axes = np.atleast_1d(axes).flatten()
            fig.suptitle('Accessibility vs landuse attractor (per zone)',
                         fontweight='bold')
            for i, (a, lc, lbl) in enumerate(pairs):
                x = pd.to_numeric(landuse_enriched[lc], errors='coerce')
                y = pd.to_numeric(landuse_enriched[a], errors='coerce')
                ok = x.notna() & y.notna()
                r = float(np.corrcoef(np.log1p(x[ok].clip(lower=0)), y[ok])[0, 1]) if ok.sum() > 2 else float('nan')
                axes[i].scatter(np.log1p(x[ok].clip(lower=0)), y[ok],
                                s=12, alpha=0.55, color='#1C7293')
                axes[i].set_xlabel(f'log1p({lc})', fontsize=8)
                axes[i].set_ylabel(a, fontsize=8)
                axes[i].set_title(f'{a} vs {lbl}\nr = {r:.2f}', fontsize=9)
                axes[i].grid(alpha=0.3)
            for j in range(len(pairs), len(axes)):
                axes[j].set_visible(False)
            fig.tight_layout()
            p2 = os.path.join(plot_dir, 'accessibility_vs_landuse.png')
            fig.savefig(p2, dpi=110)
            plt.close(fig)
            print(f"[Accessibility] plot → {p2}")
    except Exception as e:
        logger.warning(f"plot_accessibility failed (non-fatal): {e}")


def run_accessibility(
    landuse: pd.DataFrame,
    skims,
    spec: dict = None,
    save_path: str = 'saved_data/accessibility.csv',
    plot_dir: str = None,
) -> pd.DataFrame:
    from stage5_destination_choice import _skim_flat, _open_skims

    print("=" * 60)
    print("ACCESSIBILITY — Zone-level access measures")
    print("=" * 60)

    skims, close_it = _open_skims(skims)
    print("\nReading skims...")
    skim_df = _skim_flat(skims)
    print(f"  {len(skim_df)} OD pairs, {len(skim_df.columns)-2} skim columns")
    if len(skim_df) == 0 or 'OTAZ' not in skim_df.columns:
        # Fail HERE with the real cause instead of a KeyError('OTAZ') three
        # frames deeper. The usual culprit: an LZF-compressed raw OMX
        # (sa2_skims_507.omx) that PyTables cannot decode - "required filter
        # 'lzf' is not registered" in the warning above. The readable,
        # omx-spec-compliant conversion is produced by
        # demand_/11_convert_skims_to_openmatrix.py.
        raise RuntimeError(
            "Skims could not be read (0 OD pairs) - see the HDF5 warning "
            "above. If it mentions the 'lzf' filter, generate the readable "
            "conversion first:  cd ../demand_ && "
            "python 11_convert_skims_to_openmatrix.py  "
            "(writes sa2_skims_507_omxspec.omx, which pipeline_logger "
            "prefers automatically)."
        )

    acc_df = compute_accessibility(landuse, skim_df, spec=spec, save_path=save_path)
    landuse_enriched = attach_to_landuse(landuse, acc_df)

    if plot_dir:
        plot_accessibility(landuse_enriched, list(acc_df.columns), plot_dir)

    if close_it:
        skims.close()

    print("\nAccessibility complete.")
    return landuse_enriched
