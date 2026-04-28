"""
usual_location.py
==================
Usual Work Location and Usual School Location models.

ActivitySim equivalents: workplace_location, school_location

These run ONCE per person (not per tour) and assign each worker/student
a single fixed zone as their usual workplace or school. All subsequent
work/edu tours in Stage 5 destination choice use this as the KNOWN
destination rather than re-running MNL destination choice each time.

Why this matters
----------------
Without usual locations:
  - Every work tour runs destination MNL independently → same person
    goes to a different employment zone each day, which is unrealistic
  - Tour chains (home→work→home) have inconsistent DTAZ across days
  - Stage 5 destination model can't distinguish "worker in CBD" from
    "worker in suburb" — both look the same in the utility

With usual locations:
  - Each worker has one WORK_TAZ; each student has one SCHOOL_TAZ
  - Stage 5 skips destination choice for work/edu tours and uses these
  - Mode choice still runs (same person may drive vs PT on different days)
  - Non-mandatory destination choice is unaffected

Model structure
---------------
Simple weighted sampling from zone employment/enrolment attractors,
distance-decayed from the person's home zone.

  P(work zone z | home zone o) ∝ size_term(z, 'work') × exp(-b × tt_oz)
  P(school zone z | home zone o) ∝ size_term(z, 'edu') × exp(-b × tt_oz)

This is the ActivitySim workplace_location / school_location MNL pattern
but implemented as weighted sampling (faster, same expected distribution).

Outputs
-------
Adds to synth population DataFrame:
    WORK_TAZ    : usual work zone (int, 0 if not a worker)
    SCHOOL_TAZ  : usual school zone (int, 0 if not a student)

These columns are then used in Stage 5 to bypass destination choice
for work and edu segments.
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Optional

logger = logging.getLogger(__name__)


def _location_model_selector(name: str) -> str:
        """
        ActivitySim-style model selector normalisation.

        Mirrors activitysim.estimation.larch.location_choice.py:
            - strips common suffixes (_location, _destination, _subtour, _tour)
            - maps 'joint' to 'non_mandatory'
        """
        selector = str(name or '').strip().lower()
        selector = selector.replace('_location', '')
        selector = selector.replace('_destination', '')
        selector = selector.replace('_subtour', '')
        selector = selector.replace('_tour', '')
        if selector == 'joint':
                selector = 'non_mandatory'
        return selector


# =============================================================================
# SIZE TERM ATTRACTORS
# =============================================================================

WORK_ATTRACTOR_COLS = {
    'TOTEMP':  1.0,
    'FPSEMPN': 0.1,
    'HEREMPN': 0.05,
    'RETEMPN': 0.05,
}

# Industry-specific work attractor weights (indp codes from synth pop)
# Maps synth pop indp code → preferred landuse attractor column
INDUSTRY_ATTRACTOR = {
    # indp: {landuse_col: extra_weight}
    7:  {'RETEMPN': 0.5},          # retail
    8:  {'OTHEMP': 0.3},           # accommodation/food
    11: {'FPSEMPN': 0.5},          # financial
    12: {'FPSEMPN': 0.3},          # real estate
    13: {'FPSEMPN': 0.5},          # professional
    16: {'HEREMPN': 0.5},          # education
    17: {'HEREMPN': 0.5},          # health
}

EDU_ATTRACTOR_COLS = {
    # STUDYING code → columns
    1: {'Primary':  1.0},           # primary school
    2: {'Secondary': 1.0},          # secondary school
    3: {'Education': 0.5},          # tertiary PT
    4: {'Education': 1.0},          # tertiary FT
}

# Decay coefficients (exp(-b * tt_minutes))
WORK_DECAY_B   = 0.04   # ~30 min mean commute
SCHOOL_DECAY_B = 0.06   # ~20 min mean school trip


# =============================================================================
# ATTRACTOR VECTORS
# =============================================================================

def _work_attractor(landuse: pd.DataFrame,
                    zone_ids: np.ndarray,
                    indp: int = 0) -> np.ndarray:
    """
    Compute work attractor for each zone.
    indp: industry code from synth pop (adds industry-specific bonus).
    """
    lu = landuse.set_index('ORDER') if 'ORDER' in landuse.columns else landuse
    att = np.zeros(len(zone_ids), dtype=np.float32)

    for col, w in WORK_ATTRACTOR_COLS.items():
        if col in lu.columns:
            vals = lu.reindex(zone_ids)[col].fillna(0).clip(lower=0).values
            att += w * vals.astype(np.float32)

    # industry bonus
    extra = INDUSTRY_ATTRACTOR.get(int(indp), {})
    for col, w in extra.items():
        if col in lu.columns:
            vals = lu.reindex(zone_ids)[col].fillna(0).clip(lower=0).values
            att += w * vals.astype(np.float32)

    return np.maximum(att, 0.0)


def _edu_attractor(landuse: pd.DataFrame,
                   zone_ids: np.ndarray,
                   studying: int = 4) -> np.ndarray:
    """
    Compute education attractor for each zone.
    studying: STUDYING code (1=primary, 2=secondary, 3=PT, 4=FT).
    Falls back to TOTEMP-based attractor if education columns missing.
    """
    lu = landuse.set_index('ORDER') if 'ORDER' in landuse.columns else landuse
    cols = EDU_ATTRACTOR_COLS.get(int(studying), {'Education': 1.0})
    att = np.zeros(len(zone_ids), dtype=np.float32)

    found_any = False
    for col, w in cols.items():
        if col in lu.columns:
            vals = lu.reindex(zone_ids)[col].fillna(0).clip(lower=0).values
            if vals.sum() > 0:
                att += w * vals.astype(np.float32)
                found_any = True

    # Fallback: if education-specific columns are missing or all-zero,
    # use school_enrol (combined) or TOTEMP as a proxy attractor.
    # This prevents all students being sent to the same 1-4 zones.
    if not found_any:
        for fallback in ['school_enrol', 'HSENROLL', 'AGE0519', 'TOTEMP']:
            if fallback in lu.columns:
                vals = lu.reindex(zone_ids)[fallback].fillna(0).clip(lower=0).values
                if vals.sum() > 0:
                    att = vals.astype(np.float32)
                    logger.debug(f"_edu_attractor: using {fallback} as fallback for STUDYING={studying}")
                    break

    return np.maximum(att, 0.0)


# =============================================================================
# SKIM LOOKUP
# =============================================================================

def _get_tt_vector(skim_flat: pd.DataFrame,
                   otaz: int,
                   zone_ids: np.ndarray,
                   tt_col: str = 'SOVTOLL_TIME__AM') -> np.ndarray:
    """
    Get travel time from otaz to all zone_ids using the flat skim DataFrame.
    Returns array of travel times (minutes), 0 for missing OD pairs.
    """
    if skim_flat is None:
        return np.ones(len(zone_ids), dtype=np.float32) * 30.0

    row = skim_flat[skim_flat['OTAZ'] == otaz]
    if len(row) == 0:
        return np.ones(len(zone_ids), dtype=np.float32) * 30.0

    # build a zone→tt mapping
    zone_tt = row.set_index('DTAZ')[tt_col]
    tt = np.array([float(zone_tt.get(int(z), 30.0)) for z in zone_ids],
                  dtype=np.float32)

    # unit normalisation
    med = np.median(tt[tt > 0]) if (tt > 0).any() else 30.0
    if med > 600:
        tt = tt / 60.0
    elif med > 200:
        tt = tt / 10.0

    return tt


# =============================================================================
# SAMPLE ONE ZONE
# =============================================================================

def _sample_zone(attractor: np.ndarray,
                 tt: np.ndarray,
                 zone_ids: np.ndarray,
                 decay_b: float,
                 rng: np.random.Generator) -> int:
    """
    Sample one destination zone proportional to attractor × exp(-b×tt).
    """
    weights = attractor * np.exp(-decay_b * tt)
    total = weights.sum()
    if total <= 0:
        mask = attractor > 0
        if mask.any():
            return int(rng.choice(zone_ids[mask]))
        return int(zone_ids[rng.integers(0, len(zone_ids))])

    probs = (weights / total).astype(np.float64)
    # renormalise to fix floating-point drift (numpy requires sum == 1 exactly)
    probs = probs / probs.sum()
    return int(rng.choice(zone_ids, p=probs))


def _zone_probs(attractor: np.ndarray,
                tt: np.ndarray,
                decay_b: float) -> np.ndarray:
    """Return normalized destination probabilities for one chooser segment."""
    weights = attractor * np.exp(-decay_b * tt)
    total = weights.sum()
    if total <= 0:
        return np.array([], dtype=np.float64)
    probs = (weights / total).astype(np.float64)
    return probs / probs.sum()


# =============================================================================
# MAIN MODEL
# =============================================================================

def assign_usual_locations(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skim_flat: pd.DataFrame = None,
    seed: int = 42,
    save_path: str = 'saved_data/usual_locations.csv',
    tt_col: str = 'SOVTOLL_TIME__AM',
    hhts_tours: pd.DataFrame = None,
    plots_dir: str = None,
    assign_work: bool = True,
    assign_school: bool = True,
) -> pd.DataFrame:
    """
    Assign usual work zone (WORK_TAZ) and usual school zone (SCHOOL_TAZ)
    to every person in the synthetic population.

    Run BEFORE Stage 5. Stage 5 then uses these to skip destination choice
    for work and edu tours.

    Parameters
    ----------
    synth      : synthetic population DataFrame (after enrich_synth_population)
    landuse    : zone landuse DataFrame (from sc.get_land(), enriched)
    skim_flat  : flat OD skim DataFrame from _skim_flat() — if None, uses
                 distance-decay only based on zone order proximity
    seed       : random seed for reproducibility
    save_path  : CSV output path
    hhts_tours : HHTS tours DataFrame for validation plots (optional)
    plots_dir  : directory to save validation plots (defaults to save_path dir)

    Returns
    -------
    synth with WORK_TAZ and SCHOOL_TAZ columns added
    """
    print("=" * 55)
    print("USUAL LOCATIONS — Work & School Zone Assignment")
    print("=" * 55)

    rng = np.random.default_rng(seed)
    df  = synth.copy()
    df['WORK_TAZ']   = 0
    df['SCHOOL_TAZ'] = 0

    # zone universe (exclude catch-all zone 330)
    lu = landuse.copy()
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    zone_ids = lu[zone_col].astype(int).values
    zone_ids = zone_ids[(zone_ids > 0) & (zone_ids != 330)]
    zone_ids = np.sort(zone_ids)
    if len(zone_ids) == 0:
        raise ValueError("assign_usual_locations: no valid zone ids found in landuse")
    print(f"  Zone universe: {len(zone_ids)} zones")

    _work_cols_present = [c for c in WORK_ATTRACTOR_COLS if c in lu.columns]
    _edu_cols_present = sorted({c for cc in EDU_ATTRACTOR_COLS.values() for c in cc.keys() if c in lu.columns})
    if not _work_cols_present:
        logger.warning(
            "assign_usual_locations: no work attractor columns found in landuse "
            f"(expected one of {list(WORK_ATTRACTOR_COLS.keys())})"
        )
    if not _edu_cols_present:
        logger.warning(
            "assign_usual_locations: no school-specific attractor columns found in landuse "
            "(Education/Primary/Secondary). Fallback will be used."
        )
    logger.info(
        f"assign_usual_locations: landuse work cols={_work_cols_present} "
        f"school cols={_edu_cols_present}"
    )

    # ── identify workers and students ──────────────────────────────────
    is_worker  = df['WORKSTAT'].isin([1, 2]) if 'WORKSTAT' in df.columns \
                 else df.get('hrsp', pd.Series(1, index=df.index)).isin([2,3,4,5,6,7,8])
    is_student = df['STUDYING'].isin([1,2,3,4]) if 'STUDYING' in df.columns \
                 else df.get('stup', pd.Series(1, index=df.index)).isin([2,3,4])

    n_workers  = is_worker.sum()
    n_students = is_student.sum()
    print(f"  Workers: {n_workers} | Students: {n_students} | "
          f"assign_work={assign_work} assign_school={assign_school}")

    # ── pre-compute skim rows per home zone ────────────────────────────
    # group by OTAZ to avoid re-reading skim for every person
    otaz_col = 'OTAZ' if 'OTAZ' in df.columns else 'ORDER'
    skim_cache = {}   # otaz → tt array

    def get_tt(otaz):
        if otaz not in skim_cache:
            skim_cache[otaz] = _get_tt_vector(skim_flat, otaz, zone_ids, tt_col)
        return skim_cache[otaz]

    # ── assign WORK_TAZ ────────────────────────────────────────────────
    if assign_work:
        print("\n  Assigning work locations...")

        # Pre-compute one attractor vector per unique (indp, otaz) combination
        # rather than recomputing per-person — workers with same industry/home share it
        worker_df = df.loc[is_worker, [otaz_col, 'indp'] if 'indp' in df.columns else [otaz_col]].copy()
        if 'indp' not in worker_df.columns:
            worker_df['indp'] = 0

        work_probs = {}   # (otaz, indp) -> np.array probs over zone_ids
        unique_combos = worker_df[[otaz_col, 'indp']].drop_duplicates().values
        print(f"    {len(unique_combos)} unique (home_zone, industry) combinations")

        for otaz, indp in unique_combos:
            key = (int(otaz), int(indp))
            tt  = get_tt(int(otaz))
            att = _work_attractor(lu, zone_ids, indp=int(indp))
            work_probs[key] = _zone_probs(att, tt, WORK_DECAY_B)

        # Sample per person from cached combo probabilities.
        work_assign = np.zeros(len(worker_df), dtype=np.int32)
        for i, (otaz, indp) in enumerate(worker_df[[otaz_col, 'indp']].itertuples(index=False, name=None)):
            key = (int(otaz), int(indp))
            p = work_probs.get(key)
            if p is not None and len(p) == len(zone_ids) and np.isfinite(p).all() and p.sum() > 0:
                work_assign[i] = int(rng.choice(zone_ids, p=p))
            else:
                tt = get_tt(int(otaz))
                att = _work_attractor(lu, zone_ids, indp=int(indp))
                work_assign[i] = _sample_zone(att, tt, zone_ids, WORK_DECAY_B, rng)

        df.loc[is_worker, 'WORK_TAZ'] = work_assign

        missing_work = is_worker & df['WORK_TAZ'].fillna(0).astype(int).le(0)
        if missing_work.any():
            print(f"  WARNING: {missing_work.sum()} workers had no WORK_TAZ assigned; using home zone as fallback.")
            df.loc[missing_work, 'WORK_TAZ'] = df.loc[missing_work, otaz_col].fillna(0).astype(int)

        work_dist = df.loc[is_worker, 'WORK_TAZ']
        print(f"  Work TAZ: {work_dist.nunique()} unique zones | "
              f"mode={work_dist.mode().iloc[0]} | "
              f"median={int(work_dist.median())}")
    else:
        print("\n  Skipping work location assignment (assign_work=False)")

    # ── assign SCHOOL_TAZ ──────────────────────────────────────────────
    if assign_school:
        print("\n  Assigning school locations...")

        student_df = df.loc[is_student, [otaz_col, 'STUDYING'] if 'STUDYING' in df.columns else [otaz_col]].copy()
        if 'STUDYING' not in student_df.columns:
            student_df['STUDYING'] = 4

        school_probs = {}  # (otaz, studying) -> np.array probs over zone_ids
        unique_stu_combos = student_df[[otaz_col, 'STUDYING']].drop_duplicates().values
        print(f"    {len(unique_stu_combos)} unique (home_zone, study_type) combinations")

        for otaz, studying in unique_stu_combos:
            key = (int(otaz), int(studying))
            tt  = get_tt(int(otaz))
            att = _edu_attractor(lu, zone_ids, studying=int(studying))
            school_probs[key] = _zone_probs(att, tt, SCHOOL_DECAY_B)

        school_assign = np.zeros(len(student_df), dtype=np.int32)
        for i, (otaz, studying) in enumerate(student_df[[otaz_col, 'STUDYING']].itertuples(index=False, name=None)):
            key = (int(otaz), int(studying))
            p = school_probs.get(key)
            if p is not None and len(p) == len(zone_ids) and np.isfinite(p).all() and p.sum() > 0:
                school_assign[i] = int(rng.choice(zone_ids, p=p))
            else:
                tt = get_tt(int(otaz))
                att = _edu_attractor(lu, zone_ids, studying=int(studying))
                school_assign[i] = _sample_zone(att, tt, zone_ids, SCHOOL_DECAY_B, rng)

        df.loc[is_student, 'SCHOOL_TAZ'] = school_assign

        missing_school = is_student & df['SCHOOL_TAZ'].fillna(0).astype(int).le(0)
        if missing_school.any():
            print(f"  WARNING: {missing_school.sum()} students had no SCHOOL_TAZ assigned; using home zone as fallback.")
            df.loc[missing_school, 'SCHOOL_TAZ'] = df.loc[missing_school, otaz_col].fillna(0).astype(int)

        school_dist = df.loc[is_student, 'SCHOOL_TAZ']
        print(f"  School TAZ: {school_dist.nunique()} unique zones | "
              f"mode={school_dist.mode().iloc[0]} | "
              f"median={int(school_dist.median())}")
    else:
        print("\n  Skipping school location assignment (assign_school=False)")

    # ── save ───────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        out_cols = [c for c in [otaz_col, 'person_id', 'abspid',
                                'WORK_TAZ', 'SCHOOL_TAZ'] if c in df.columns]
        df[out_cols].to_csv(save_path, index=False)
        print(f"\n  Usual locations saved → {save_path}")

    # ── Validation: compare HHTS observed destinations to synth assignments ──
    _val_dir = plots_dir or (os.path.dirname(save_path) if save_path else 'saved_data')
    _validation = {}
    if hhts_tours is not None and not hhts_tours.empty:
        try:
            _validation = _validate_usual_locations(
                df, hhts_tours, is_worker, is_student, _val_dir)
        except Exception as _ve:
            logger.warning(f"  Usual location validation failed (non-fatal): {_ve}")

    _save_location_html(df, landuse, is_worker, is_student, otaz_col,
                        save_path.replace('.csv', '.html') if save_path else None,
                        validation=_validation)

    print("\nUsual location assignment complete.")
    return df


# =============================================================================
# HHTS VALIDATION
# =============================================================================

def _validate_usual_locations(
    df: pd.DataFrame,
    hhts_tours: pd.DataFrame,
    is_worker,
    is_student,
    save_dir: str,
) -> dict:
    """
    Compare synthetic WORK_TAZ / SCHOOL_TAZ assignments to observed HHTS
    work and school trip destinations.

    Returns a dict with:
        work_plots   : list of saved plot paths
        school_plots : list of saved plot paths
        work_metrics : {overlap, top10_coverage_obs, top10_coverage_syn}
        school_metrics: same
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    result = {'work_plots': [], 'school_plots': [], 'work_metrics': {}, 'school_metrics': {}}

    if hhts_tours is None or hhts_tours.empty:
        return result

    # ── map HHTS tour purpose column ───────────────────────────────────────
    # data_connect aggregates tours as 'tour_primary_purpose'; some paths
    # use 'primary_purpose', 'main_purpose', 'TOURPURP', or 'purpose'.
    purp_col = next((c for c in ['primary_purpose', 'tour_primary_purpose',
                                  'main_purpose', 'TOURPURP', 'purpose']
                     if c in hhts_tours.columns), None)
    # destination zone column: 'tour_destination' is the data_connect name;
    # 'DTAZ' / 'dtaz' / 'dest_zone' are used in other paths.
    dtaz_col = next((c for c in ['DTAZ', 'dtaz', 'tour_destination', 'dest_zone']
                     if c in hhts_tours.columns), None)

    if purp_col is None or dtaz_col is None:
        missing = []
        if purp_col is None: missing.append('purpose')
        if dtaz_col is None: missing.append('DTAZ/dest')
        logger.warning(
            f"_validate_usual_locations: HHTS tours missing {', '.join(missing)} col "
            f"(available: {list(hhts_tours.columns[:10])}) — skipping"
        )
        return result

    def _top_zones(series: pd.Series, n: int = 30):
        vc = series.dropna().astype(int).value_counts()
        return vc.head(n)

    def _plot_comparison(obs_vc: pd.Series, syn_vc: pd.Series,
                          title: str, save_path: str, top_n: int = 20):
        top_zones = obs_vc.head(top_n).index.union(syn_vc.head(top_n).index)
        top_zones = sorted(top_zones)

        obs_p = obs_vc.reindex(top_zones, fill_value=0) / max(obs_vc.sum(), 1) * 100
        syn_p = syn_vc.reindex(top_zones, fill_value=0) / max(syn_vc.sum(), 1) * 100

        x = np.arange(len(top_zones)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(10, len(top_zones) * 0.55), 5))
        ax.bar(x - w / 2, obs_p.values, w, label='HHTS observed', color='#2E86AB', alpha=0.85)
        ax.bar(x + w / 2, syn_p.values, w, label='Synth assigned', color='#E84855', alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([str(z) for z in top_zones], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Zone (SA2)')
        ax.set_ylabel('Share (%)')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _overlap_metric(obs_vc, syn_vc) -> float:
        """Bhattacharyya overlap between two zone distributions."""
        all_zones = obs_vc.index.union(syn_vc.index)
        p = obs_vc.reindex(all_zones, fill_value=0).values.astype(float)
        q = syn_vc.reindex(all_zones, fill_value=0).values.astype(float)
        p = p / max(p.sum(), 1e-9)
        q = q / max(q.sum(), 1e-9)
        return float(np.sum(np.sqrt(p * q)))

    # ── Work comparison ───────────────────────────────────────────────────
    work_mask_hhts = hhts_tours[purp_col].astype(str).str.lower().isin(
        ['work', 'full-time work', 'part-time work', 'workfulltime', 'workparttime'])
    obs_work_dtaz = hhts_tours.loc[work_mask_hhts, dtaz_col]
    syn_work_taz  = df.loc[is_worker, 'WORK_TAZ']

    if obs_work_dtaz.notna().any() and syn_work_taz.gt(0).any():
        obs_vc = _top_zones(obs_work_dtaz, 50)
        syn_vc = _top_zones(syn_work_taz, 50)
        p = _plot_comparison(
            obs_vc, syn_vc,
            title=f'Usual Work Location — HHTS Observed (n={obs_work_dtaz.notna().sum()}) vs Synth (n={syn_work_taz.gt(0).sum()})',
            save_path=os.path.join(save_dir, 'val_usual_work_location.png'),
        )
        result['work_plots'].append(p)
        ov = _overlap_metric(obs_vc, syn_vc)
        result['work_metrics'] = {
            'bhattacharyya_overlap': round(ov, 4),
            'n_obs': int(obs_work_dtaz.notna().sum()),
            'n_syn': int(syn_work_taz.gt(0).sum()),
            'n_obs_zones': int(obs_vc.shape[0]),
            'n_syn_zones': int(syn_vc.shape[0]),
        }
        logger.info(f"  Work location validation: overlap={ov:.3f} "
                    f"({int(obs_work_dtaz.notna().sum())} HHTS, "
                    f"{int(syn_work_taz.gt(0).sum())} synth)")

    # ── School comparison ─────────────────────────────────────────────────
    school_mask_hhts = hhts_tours[purp_col].astype(str).str.lower().isin(
        ['edu', 'education', 'primary school', 'secondary school',
         'full-time tafe/uni', 'part-time tafe/uni', 'school', 'university'])
    obs_school_dtaz = hhts_tours.loc[school_mask_hhts, dtaz_col]
    syn_school_taz  = df.loc[is_student, 'SCHOOL_TAZ']

    if obs_school_dtaz.notna().any() and syn_school_taz.gt(0).any():
        obs_vc = _top_zones(obs_school_dtaz, 50)
        syn_vc = _top_zones(syn_school_taz, 50)
        p = _plot_comparison(
            obs_vc, syn_vc,
            title=f'Usual School Location — HHTS Observed (n={obs_school_dtaz.notna().sum()}) vs Synth (n={syn_school_taz.gt(0).sum()})',
            save_path=os.path.join(save_dir, 'val_usual_school_location.png'),
        )
        result['school_plots'].append(p)
        ov = _overlap_metric(obs_vc, syn_vc)
        result['school_metrics'] = {
            'bhattacharyya_overlap': round(ov, 4),
            'n_obs': int(obs_school_dtaz.notna().sum()),
            'n_syn': int(syn_school_taz.gt(0).sum()),
            'n_obs_zones': int(obs_vc.shape[0]),
            'n_syn_zones': int(syn_vc.shape[0]),
        }
        logger.info(f"  School location validation: overlap={ov:.3f} "
                    f"({int(obs_school_dtaz.notna().sum())} HHTS, "
                    f"{int(syn_school_taz.gt(0).sum())} synth)")

    return result


# =============================================================================
# HTML REPORT
# =============================================================================

def _save_location_html(df, landuse, is_worker, is_student, otaz_col, path,
                         validation: dict = None):
    if path is None:
        return
    try:
        from collections import Counter

        work_zones = Counter(df.loc[is_worker, 'WORK_TAZ'].astype(int))
        school_zones = Counter(df.loc[is_student, 'SCHOOL_TAZ'].astype(int))
        top_work = sorted(work_zones.items(), key=lambda x: -x[1])[:10]
        top_school = sorted(school_zones.items(), key=lambda x: -x[1])[:10]

        zone_ids = (landuse['ORDER'] if 'ORDER' in landuse.columns else landuse.index).astype(int)
        work_attr = _work_attractor(landuse, zone_ids.values, indp=0)
        edu_attr = _edu_attractor(landuse, zone_ids.values, studying=4)
        work_attr_top = sorted(zip(zone_ids, work_attr), key=lambda x: -x[1])[:10]
        edu_attr_top = sorted(zip(zone_ids, edu_attr), key=lambda x: -x[1])[:10]

        def rows_html(items):
            return '\n'.join(
                f'<tr><td>{z}</td><td style="text-align:right">{n}</td></tr>'
                for z, n in items)

        def coeff_rows_html(items):
            return '\n'.join(
                f'<tr><td>{name}</td><td style="text-align:right">{value}</td></tr>'
                for name, value in items)

        work_coeffs = list(WORK_ATTRACTOR_COLS.items())
        industry_coeffs = [
            (f'indp={indp} +{col}', weight)
            for indp, extras in INDUSTRY_ATTRACTOR.items()
            for col, weight in extras.items()
        ]
        school_coeffs = [
            (f'STUDYING={studying}:{col}', weight)
            for studying, cols in EDU_ATTRACTOR_COLS.items()
            for col, weight in cols.items()
        ]
        decay_rows = [
            ('WORK_DECAY_B', WORK_DECAY_B),
            ('SCHOOL_DECAY_B', SCHOOL_DECAY_B),
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Usual Work &amp; School Location Model</title>
<style>
  body {{font-family: Arial,sans-serif; margin:40px;}}
  h1 {{border-bottom:2px solid #1565c0; padding-bottom:8px;}}
  h2 {{color:#1565c0; margin-top:28px;}}
  h3 {{color:#333; margin-top:22px;}}
  .stats {{background:#e3f2fd; padding:8px 12px; border-radius:4px; margin-bottom:12px;}}
  table {{border-collapse:collapse; width:100%; max-width:860px; margin-bottom:18px;}}
  th {{background:#1565c0; color:white; padding:8px 12px; text-align:left;}}
  td {{padding:6px 12px; border-bottom:1px solid #eee;}}
  .half {{display:inline-block; vertical-align:top; width:48%; margin-right:2%;}}
  .note {{color:#555; font-size:0.95em; margin-top:18px;}}
</style></head>
<body>
<h1>Usual Work &amp; School Location Model</h1>
<div class="stats">
  <b>Workers assigned:</b> {is_worker.sum()} &nbsp;|&nbsp;
  <b>Unique work zones:</b> {df.loc[is_worker, 'WORK_TAZ'].nunique()} &nbsp;|&nbsp;
  <b>Students assigned:</b> {is_student.sum()} &nbsp;|&nbsp;
  <b>Unique school zones:</b> {df.loc[is_student, 'SCHOOL_TAZ'].nunique()}
</div>

<h2>Model coefficients</h2>
<p class="note">These weights define the attractor functions used to sample usual work and school zones.</p>
<div class="half">
  <h3>Work attractor size terms</h3>
  <table><thead><tr><th>Attractor term</th><th>Weight</th></tr></thead>
  <tbody>{coeff_rows_html(work_coeffs)}</tbody></table>
</div>
<div class="half">
  <h3>School attractor size terms</h3>
  <table><thead><tr><th>Study type / term</th><th>Weight</th></tr></thead>
  <tbody>{coeff_rows_html(school_coeffs)}</tbody></table>
</div>

<h3>Industry-specific work bonuses</h3>
<table><thead><tr><th>Industry (indp)</th><th>Bonus term</th><th>Weight</th></tr></thead>
<tbody>"""
        + '\n'.join(
            f'<tr><td>{indp}</td><td>{col}</td><td style="text-align:right">{weight}</td></tr>'
            for indp, extras in INDUSTRY_ATTRACTOR.items() for col, weight in extras.items()
        )
        + "</tbody></table>"
        + f"\n<h3>Decay coefficients</h3>\n<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>{coeff_rows_html(decay_rows)}</tbody></table>"
        + "\n<h2>Attractor strength — generic top zones</h2>"
        + "\n<div class=\"half\">\n  <h3>Top work attractor zones</h3>\n  <table><thead><tr><th>Zone</th><th>Attractor</th></tr></thead>\n  <tbody>"
        + '\n'.join(
            f'<tr><td>{z}</td><td style="text-align:right">{val:.1f}</td></tr>'
            for z, val in work_attr_top)
        + "</tbody></table>\n</div>\n"
        + "\n<div class=\"half\">\n  <h3>Top school attractor zones</h3>\n  <table><thead><tr><th>Zone</th><th>Attractor</th></tr></thead>\n  <tbody>"
        + '\n'.join(
            f'<tr><td>{z}</td><td style="text-align:right">{val:.1f}</td></tr>'
            for z, val in edu_attr_top)
        + "</tbody></table>\n</div>\n"
        + "\n<h2>Assigned usual locations</h2>\n"
        + "\n<div class=\"half\">\n  <h3>Top assigned work zones</h3>\n  <table><thead><tr><th>Zone</th><th>Workers</th></tr></thead>\n  <tbody>" + rows_html(top_work) + "</tbody></table>\n</div>\n"
        + "\n<div class=\"half\">\n  <h3>Top assigned school zones</h3>\n  <table><thead><tr><th>Zone</th><th>Students</th></tr></thead>\n  <tbody>" + rows_html(top_school) + "</tbody></table>\n</div>\n"
        + _validation_html_section(validation)
        + "\n<p class=\"note\">Work and school locations are sampled from attractors that combine zone size and travel-time decay. Industry bonuses are applied for specific worker types.</p>\n"
        + "<p style=\"color:#888;font-size:0.85em;margin-top:24px;\">Generated by usual_location.py — QTS/HHTS ABM Pipeline</p>\n"
        + "</body></html>"

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',
                    exist_ok=True)
        with open(path, 'w') as f:
            f.write(html)
        print(f"  HTML report saved → {path}")
    except Exception as e:
        logger.debug(f"HTML report failed: {e}")


def _validation_html_section(validation: dict) -> str:
    """Render the HHTS validation section for the HTML report."""
    if not validation:
        return ''
    parts = ['\n<h2>Validation — HHTS Observed vs Synthetic Assigned</h2>\n']
    parts.append('<p class="note">Bhattacharyya overlap measures how closely the '
                 'synthetic assignment distribution matches the observed HHTS '
                 'trip destinations (1.0 = perfect match).</p>\n')

    def _metric_html(label: str, m: dict) -> str:
        if not m:
            return f'<p><em>{label}: no data available</em></p>'
        ov = m.get('bhattacharyya_overlap', float('nan'))
        color = '#155724' if ov > 0.7 else '#856404' if ov > 0.4 else '#721c24'
        return (
            f'<h3>{label}</h3>'
            f'<div class="stats">'
            f'<b>Overlap:</b> <span style="color:{color};font-weight:bold">{ov:.3f}</span>'
            f' &nbsp;|&nbsp; <b>HHTS trips:</b> {m.get("n_obs","?")} '
            f'&nbsp;|&nbsp; <b>Synth assigned:</b> {m.get("n_syn","?")}'
            f' &nbsp;|&nbsp; <b>HHTS zones:</b> {m.get("n_obs_zones","?")} '
            f'&nbsp;|&nbsp; <b>Synth zones:</b> {m.get("n_syn_zones","?")}'
            f'</div>'
        )

    parts.append(_metric_html('Work Location Validation', validation.get('work_metrics', {})))
    if validation.get('work_plots'):
        for p in validation['work_plots']:
            img = os.path.basename(p)
            parts.append(f'<p><img src="{img}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:8px" alt="Work validation"></p>\n')

    parts.append(_metric_html('School Location Validation', validation.get('school_metrics', {})))
    if validation.get('school_plots'):
        for p in validation['school_plots']:
            img = os.path.basename(p)
            parts.append(f'<p><img src="{img}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:8px" alt="School validation"></p>\n')

    return ''.join(parts)


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run_usual_locations(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skims=None,
    seed: int = 42,
    save_path: str = 'saved_data/usual_locations.csv',
    hhts_tours: pd.DataFrame = None,
    plots_dir: str = None,
    assign_work: bool = True,
    assign_school: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline entry point.  Called from pipeline_logger after CDAP,
    before Stage 1.

    If skims is provided (OMX path or handle), reads the flat OD skim table
    for travel-time-weighted zone sampling.  If None, falls back to
    proximity-only (zones sorted by ORDER distance from home zone).

    hhts_tours : optional HHTS tours table for HHTS vs synth validation plots
    plots_dir  : where to save validation plots; defaults to save_path directory

    Returns synth with WORK_TAZ and SCHOOL_TAZ added.
    """
    skim_flat = None
    close_skims = False

    if skims is not None:
        try:
            from stage5_destination_choice import _skim_flat, _open_skims
            skims_handle, close_skims = _open_skims(skims)
            print("  Reading skims for travel-time decay...")
            skim_flat = _skim_flat(skims_handle)
            print(f"  Skim loaded: {len(skim_flat)} OD pairs")
        except Exception as e:
            logger.warning(f"Could not load skims for usual locations: {e} "
                           "— using proximity fallback")

    result = assign_usual_locations(
        synth, landuse, skim_flat=skim_flat,
        seed=seed, save_path=save_path,
        hhts_tours=hhts_tours,
        plots_dir=plots_dir,
        assign_work=assign_work,
        assign_school=assign_school,
    )

    if close_skims:
        try:
            skims_handle.close()
        except Exception:
            pass

    return result


def workplace_location_model(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skims=None,
    seed: int = 42,
    save_path: str = 'saved_data/workplace_location.csv',
    hhts_tours: pd.DataFrame = None,
    plots_dir: str = None,
) -> pd.DataFrame:
    """ActivitySim-style workplace_location wrapper over usual-location assignment."""
    return run_usual_locations(
        synth=synth,
        landuse=landuse,
        skims=skims,
        seed=seed,
        save_path=save_path,
        hhts_tours=hhts_tours,
        plots_dir=plots_dir,
        assign_work=True,
        assign_school=False,
    )


def school_location_model(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skims=None,
    seed: int = 42,
    save_path: str = 'saved_data/school_location.csv',
    hhts_tours: pd.DataFrame = None,
    plots_dir: str = None,
) -> pd.DataFrame:
    """ActivitySim-style school_location wrapper over usual-location assignment."""
    return run_usual_locations(
        synth=synth,
        landuse=landuse,
        skims=skims,
        seed=seed,
        save_path=save_path,
        hhts_tours=hhts_tours,
        plots_dir=plots_dir,
        assign_work=False,
        assign_school=True,
    )


def location_choice_model(name: str, **kwargs):
    """
    ActivitySim-style location-choice entry point.

    Supported in this module:
      - workplace_location
      - school_location
    Tour- and trip-level destination wrappers are provided in stage5_destination_choice.py.
    """
    selector = _location_model_selector(name)
    if selector == 'workplace':
        return workplace_location_model(**kwargs)
    if selector == 'school':
        return school_location_model(**kwargs)
    raise ValueError(
        f"Unsupported location model '{name}' (selector='{selector}') in usual_location.py"
    )
