"""
ga_hypothesis_testing.py
=========================
Genetic Algorithm (GA) variable selection across all ABM pipeline stages.

Design principles
-----------------
1. GA is ESTIMATION ONLY   every fitness evaluation fits a model on HHTS
   and returns a log-likelihood / rho┬▓ / adj-R┬▓ score.  No prediction over
   the synthetic population is done during GA search.  Prediction runs ONCE
   at the end via apply_ga_results_to_pipeline().

2. Fitness CACHING   identical chromosomes (same binary gene pattern) are
   cached so they are never re-estimated. This removes the main redundancy
   in the original GA where crossover frequently produced duplicate individuals.

3. Adaptive mutation   when population diversity falls below a threshold
   (measured by mean hamming distance between individuals) the mutation rate
   temporarily increases to escape local optima.

4. Multi-restart   after the main run a configurable number of random-restart
   mini-runs (shorter n_generations) compete against the main result. This
   replaces the fragile "run again from scratch" pattern in the pipeline.

5. All-segment fitness aggregation   each stage's fitness is the mean over
   all valid (purpose, ptype) or (purpose, segment) combinations, not just
   the best-performing one.  This finds features that work broadly.

6. Collinearity repair ÔÇö after mutation, at most ONE variable from each
   collinearity group (see STAGE_COLLINEAR_GROUPS) may be active.  This
   prevents the GA from simultaneously selecting near-redundant variables
   (e.g. DIST and SOVTOLL_TIME__AM; TOTEMP and its component sub-totals)
   which inflate the parameter count and bias the BIC penalty upward.

7. Trip-level models ÔÇö Stage 5t (code 51) runs a separate GA for trip-level
   mode + destination choice.  Stage 5 (code 5) remains tour-level.
   This mirrors the ActivitySim distinction between tour_mode_choice and
   trip_mode_choice, giving independent estimated models for each level.

BIC minimisation ÔÇö CONFIRMED CORRECT
--------------------------------------
    The GA maximises fitness = ÔÇôBIC/n (always negative, closer to 0 = better).
    Maximising ÔÇôBIC/n is IDENTICAL to minimising BIC.

    BIC = ÔÇô2┬ÀLL + k┬Àln(n)   Higher BIC = worse model (more penalty for
    complexity than fit improvement).  Lower BIC = better model.

    ÔÇôBIC/n values:  very negative ÔåÆ bad model (high BIC);
                    close to 0    ÔåÆ good model (low BIC).

    The GA's best_so_far curve in convergence plots is shown as BIC/n
    (positive, with inverted y-axis so UP = better = lower BIC).

Fitness metrics (all stages)
----------------------------
    All stages use  **ÔÇôBIC/n**  (negative BIC normalised per observation).
    GA maximises fitness ÔåÆ minimises BIC.

    BIC = ÔÇô2┬ÀLL + k┬Àln(n)    where  k = number of free parameters
                                     n = number of observations

    ÔÇôBIC/n is always negative; values closer to 0 (less negative) are better.
    Convergence plots display BIC/n (positive, lower = better) for clarity.

    Stage 0  (CDAP MNL 3-alt)   : ÔÇôBIC/n  (higher = better)
    Stage -1 (usual location)   : mean log-P (LL per obs, higher = better)
    Stage 1  (ordered logit)    : ÔÇôBIC/n  (higher = better)
    Stage 2  (OLS lognormal LL) : ÔÇôBIC/n  (higher = better)
    Stage 3  (lognormal)        : ÔÇôBIC/n  (higher = better)
    Stage 4  (MNL timing)       : ÔÇôBIC/n  (higher = better)
    Stage 5  (Larch NL, TOUR)   : ÔÇôBIC/n  per-segment, averaged (TOUR-LEVEL)
    Stage 5b (Larch NL, TRIP)   : ÔÇôBIC/n  per-segment, averaged (TRIP-LEVEL) 
    Stage 6  (nested logit)     : ÔÇôBIC/n  (higher = better)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import pickle
import os
import glob
import copy
import html
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


def _plog(msg: str) -> None:
    """Write a timeline marker to pipeline.log via the pipeline logger."""
    logging.getLogger('pipeline').info(msg)


def _ga_save_stage_coef_csv(models: dict, stage_name: str, save_dir: str) -> None:
    """Write stage coefficient CSV/HTML for GA runs (parity with main path)."""
    rows = []

    def _add_rows(segment: str, obj: Any):
        pf = getattr(obj, 'pf', None)
        if pf is not None and hasattr(pf, 'iterrows'):
            for param, row in pf.iterrows():
                rows.append({
                    'stage': stage_name,
                    'segment': segment,
                    'param': str(param),
                    'value': float(row.get('value', float('nan'))),
                    'std_err': float(row.get('std_err', float('nan'))),
                    't_stat': float(row.get('t_stat', float('nan'))),
                    'p_value': float(row.get('pval', row.get('p_value', float('nan')))),
                    'source': 'larch_pf',
                })
            return

        if isinstance(obj, dict):
            if 'model' in obj:
                _add_rows(segment, obj['model'])
                if hasattr(obj.get('model', None), 'thresholds_'):
                    thr = getattr(obj['model'], 'thresholds_', None)
                    if thr is not None:
                        for i, v in enumerate(thr):
                            rows.append({
                                'stage': stage_name,
                                'segment': segment,
                                'param': f'threshold_{i}',
                                'value': float(v),
                                'std_err': float('nan'),
                                't_stat': float('nan'),
                                'p_value': float('nan'),
                                'source': 'ordered_logit_threshold',
                            })
                return
            coef = obj.get('coef', obj.get('coefficients', None))
            if isinstance(coef, dict):
                se = obj.get('std_err', obj.get('std_errors', {})) or {}
                ts = obj.get('t_stat', obj.get('t_stats', {})) or {}
                pv = obj.get('p_value', obj.get('pval', {})) or {}
                for k, v in coef.items():
                    rows.append({
                        'stage': stage_name,
                        'segment': segment,
                        'param': str(k),
                        'value': float(v),
                        'std_err': float(se.get(k, float('nan'))),
                        't_stat': float(ts.get(k, float('nan'))),
                        'p_value': float(pv.get(k, float('nan'))),
                        'source': 'dict',
                    })
                return

        coef_arr = getattr(obj, 'coef_export_values_', getattr(obj, 'coef_', None))
        if coef_arr is not None:
            coef_flat = np.asarray(coef_arr).ravel()
            feats = list(getattr(obj, 'coef_export_names_', getattr(obj, 'feature_names_', [])))
            if not feats:
                feats = [f'x{i}' for i in range(len(coef_flat))]
            se = np.asarray(getattr(obj, 'std_err_export_', getattr(obj, 'std_err_', [])), dtype=float).ravel()
            ts = np.asarray(getattr(obj, 't_stat_export_', getattr(obj, 't_stat_', [])), dtype=float).ravel()
            pv = np.asarray(getattr(obj, 'p_value_export_', getattr(obj, 'p_value_', [])), dtype=float).ravel()
            for i, (f, v) in enumerate(zip(feats, coef_flat)):
                rows.append({
                    'stage': stage_name,
                    'segment': segment,
                    'param': str(f),
                    'value': float(v),
                    'std_err': float(se[i]) if i < len(se) else float('nan'),
                    't_stat': float(ts[i]) if i < len(ts) else float('nan'),
                    'p_value': float(pv[i]) if i < len(pv) else float('nan'),
                    'source': 'model_object',
                })

    for key, obj in (models or {}).items():
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and isinstance(vals[0], dict) and 'model' in vals[0]:
                for sk, sv in obj.items():
                    _add_rows(f'{key}/{sk}', sv)
                continue
            # Stage 5: segment dict with mode_model / dest_model Larch objects
            _has_submodel = False
            for _sub_key in ('mode_model', 'dest_model'):
                if _sub_key in obj:
                    _add_rows(f'{key}/{_sub_key}', obj[_sub_key])
                    _has_submodel = True
            if _has_submodel:
                continue
        _add_rows(str(key), obj)

    if not rows:
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, f'{stage_name}_coefficients.csv')
    html_path = os.path.join(save_dir, f'{stage_name}_coefficients.html')
    df.to_csv(csv_path, index=False)

    html_rows = []
    for seg, seg_df in df.sort_values(['segment', 'param']).groupby('segment', dropna=False):
        html_rows.append(
            f"<h2>{html.escape(str(seg))}</h2>"
            "<table><tr><th>Parameter</th><th>Value</th><th>Std Err</th><th>t-stat</th><th>p-value</th><th>Source</th></tr>"
        )
        for _, r in seg_df.iterrows():
            html_rows.append(
                "<tr>"
                f"<td>{html.escape(str(r['param']))}</td>"
                f"<td style='text-align:right'>{float(r['value']):.6f}</td>"
                f"<td style='text-align:right'>{float(r.get('std_err', float('nan'))):.6f}</td>"
                f"<td style='text-align:right'>{float(r.get('t_stat', float('nan'))):.3f}</td>"
                f"<td style='text-align:right'>{float(r.get('p_value', float('nan'))):.4g}</td>"
                f"<td>{html.escape(str(r.get('source', '')))}</td>"
                "</tr>"
            )
        html_rows.append("</table>")
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{stage_name} coefficients</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;font-size:13px}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "th{background:#2c3e50;color:white;padding:6px 10px;text-align:left}"
        "td{border:1px solid #ddd;padding:5px 10px}"
        "tr:nth-child(even){background:#f9f9f9}"
        "</style></head><body>"
        f"<h1>{stage_name} coefficients</h1>"
        + "\n".join(html_rows)
        + "</body></html>"
    )
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)


def _ga_run_validation_hhts(
    out: dict,
    s1_models: dict,
    s2_models: dict,
    s3_models: dict,
    s4_models: dict,
    s5_models: dict,
    save_dir: str,
) -> None:
    """Run full HHTS validation to the same output location contract as main."""
    try:
        from validate_hhts import run_validation
        run_validation(
            out=out,
            s1_models=s1_models,
            s2_models=s2_models,
            s3_models=s3_models,
            s4_models=s4_models,
            s5_models=s5_models,
            out_dir=save_dir,
        )
    except Exception as exc:
        logger.warning(f"GA full HHTS validation failed: {exc}")


# ---------------------------------------------------------------------------
# COLLINEARITY GROUPS  (stage-specific)
# ---------------------------------------------------------------------------
# Each entry is a list of variable lists.  Within each inner list, at most
# ONE variable may be active at a time in any chromosome.  The GA's mutation
# and initialisation repair steps enforce this constraint to avoid selecting
# near-redundant variables that inflate k in BIC without new information.
#
# Guidelines:
#   • Time skims for the same mode at different periods are collinear.
#   • Distance is collinear with auto travel time (same corridor).
#   • Aggregate employment totals (TOTEMP) subsume component sub-categories.
#   • Log-transformed and untransformed versions of the same variable.
#   • ptype dummies and their underlying raw survey equivalents (WORKSTAT,
#     STUDYING) encode overlapping information.
#
STAGE_COLLINEAR_GROUPS: Dict[Any, List[List[str]]] = {
    # Stage 0 (CDAP)
    # WORKSTAT/STUDYING are no longer offered as raw ordinal/categorical codes —
    # _cdap_candidates() uses their per-category binaries (workstat_*, studying_*,
    # built in cdap.build_person_profiles) instead. Like the ptype dummies
    # (ptype_FullTimeWorker, ptype_SchoolChild, etc.), these are COMPLEMENTARY
    # binary indicators: a model may select several at once (FT workers ≠ PT
    # workers), so neither set is listed as collinear with itself or with ptype.
    0: [
        ['vehrd', 'no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle'],
        ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
         'income_midhi', 'income_hi', 'income_veryhi'],
    ],
    # Stage 1 (activity frequency)
    1: [
        ['vehrd', 'no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle'],
        ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
         'income_midhi', 'income_hi', 'income_veryhi'],
        ['age_child', 'age_teen', 'ptype_SchoolChild'],   # same population segment
        ['is_female', 'sexp'],
    ],
    # Stage 2 (day duration)
    2: [
        ['vehrd', 'no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle'],
        ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
         'income_midhi', 'income_hi', 'income_veryhi'],
        ['n_total', 'n_tours'],
    ],
    # Stage 3 (activity duration)
    3: [
        ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
         'income_midhi', 'income_hi', 'income_veryhi'],
        ['log_dist', 'B'],
        ['age_child', 'age_teen', 'ptype_SchoolChild'],
        ['is_female', 'sexp'],
    ],
    # Stage 4 (schedule placement)
    4: [
        ['time_consumed', 'day_position'],
        ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
         'income_midhi', 'income_hi', 'income_veryhi'],
    ],
    # Stage 5 mode (TOUR and TRIP)
    '5_mode': [
        ['SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM', 'HOV2TOLL_TIME__AM'],
        ['DISTWALK', 'DIST'],   # walk dist collinear with total dist
        ['DISTBIKE', 'DIST'],   # bike dist collinear with total dist
        ['vehrd', 'no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle'],
        ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
         'income_midhi', 'income_hi', 'income_veryhi'],
        ['age_child', 'age_teen', 'ptype_SchoolChild'],
        ['is_female', 'sexp'],
    ],
    # Stage 5 destination (TOUR and TRIP)
    '5_dest': [
        ['TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP'],
        ['Education', 'Primary', 'Secondary'],
        ['DIST', 'SOVTOLL_TIME__AM'],
        ['TOTPOP', 'TOTHH'],
        # Person-x-distance interaction terms - _dest_tree's runtime collinearity
        # check (stage5_destination_choice.py) flags every one of these pairs at
        # |r|=1.0 on every estimation, meaning the GA was routinely proposing
        # chromosomes that spend a free parameter (BIC penalty) on a coefficient
        # that gets holdfasted to 0 anyway. Grouped as one clique since they're
        # all pairwise collinear with each other, not just with log_DIST_1.
        ['ageNp_x_log_dist', 'incp_x_log_dist', 'nprd_x_log_dist',
         'vehrd_x_log_dist', 'log_DIST_1', 'auto_work_acc'],
        ['walk_feasible', 'intrazonal', 'commercial'],
        ['size_term', 'log_correction', 'log_TOTEMP_1'],
    ],
    # Stage 6 (trip mode)
    6: [
        ['tt_avg', 'tt_am'],
        ['log_dist', 'dist'],
        ['ldw', 'distwalk', 'log_dist'],
        ['ldb', 'distbike', 'log_dist'],
        ['has_vehicle', 'no_vehicle', 'one_vehicle', 'multi_vehicle', 'vehrd'],
        ['inc_hi', 'inc_lo', 'incp'],
        ['older', 'age5p'],
    ],
}

# ---------------------------------------------------------------------------
# REQUIRE-ONE GROUPS  (stage-specific)
# ---------------------------------------------------------------------------
# Companion to STAGE_COLLINEAR_GROUPS. Each entry is a list of variable
# lists; within each inner list, AT LEAST ONE variable must be active in any
# chromosome.
#
# IMPORTANT - only applies to variables the GA actually searches over.
# Stage 5 mode/destination choice (stage5_destination_choice.py) already
# hardcodes genuine per-mode impedance terms UNCONDITIONALLY in MODE_UEC /
# DEST_UEC, entirely outside the GA-selected chromosome:
#   MODE_UEC:  c_ivt_auto (DA/SR, SOVTOLL_TIME__AM), c_ivt_transit (Transit,
#              WLK_LOC_WLK_TOTIVT__AM), c_walk_time (Walk, DISTWALK),
#              c_bike_time (Bike, DISTBIKE) - one impedance term PER MODE,
#              classic nested-logit practice.
#   DEST_UEC:  c_log_dist / c_dist (universal distance deterrent), c_logsum,
#              c_size_term - all unconditional for every segment.
# _stage5_mode_candidates()/_stage5_dest_candidates() (the GA-searchable
# pools) only cover ENRICHMENT variables layered on top of that backbone
# (income, vehicle ownership, PM/HOV2 skim variants, land-use sub-splits) -
# 'SOVTOLL_TIME__AM'/'DIST' aren't even in those pools. A require-one group
# naming them doesn't enforce "the model has impedance" (already true by
# construction); it just forces one specific enrichment skim to always be
# selected, which has no theoretical basis. So: no require-one entries for
# '5_mode' / '5_dest' - the identification requirement is already met by the
# fixed UEC, not by anything the search controls.
#
# Stage 6 (trip-level mode choice) is genuinely different: it has no
# backbone UEC - tt_avg, tt_am, log_dist, dist, ldw, distwalk, ldb, distbike
# are ALL freely searchable (only 'crash' is hard-fixed via fixed_features).
# Without a require-one constraint the GA can legally converge on a
# specification with zero time/cost/distance variables, degenerating to pure
# ASCs + demographics - not identified as a meaningful mode choice model.
# The group below is the UNION of every impedance variable across all four
# STAGE_COLLINEAR_GROUPS[6] at-most-one subgroups (motorised time, general
# distance, walk-specific, bike-specific) - "the model must have SOME
# impedance measure," not "must have this specific skim variant." Which one
# satisfies it is left to the GA/data, consistent with classic mode choice
# theory (utility must be differentiated by cost/time somewhere, but theory
# doesn't dictate a single fixed skim column).
STAGE_REQUIRE_ONE_GROUPS: Dict[Any, List[List[str]]] = {
    6: [
        ['tt_avg', 'tt_am', 'log_dist', 'dist', 'ldw', 'distwalk', 'ldb', 'distbike'],
    ],
}


def _repair_collinearity(
    genes: np.ndarray,
    feature_names: List[str],
    collinear_groups: List[List[str]],
    rng: np.random.Generator,
    protected_idx: Optional[set[int]] = None,
) -> np.ndarray:
    """
    Repair a chromosome so at most ONE gene per collinearity group is active.

    For each group with more than one active gene, randomly zero all but one.
    This is called after every mutation and during population initialisation.
    """
    genes = genes.copy()
    protected_idx = protected_idx or set()
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    for group in collinear_groups:
        active_idx = [feat_idx[f] for f in group
                      if f in feat_idx and genes[feat_idx[f]] == 1]
        if len(active_idx) > 1:
            protected_active = [idx for idx in active_idx if idx in protected_idx]
            keep = (protected_active[0] if protected_active
                    else int(rng.choice(active_idx)))
            for idx in active_idx:
                if idx != keep:
                    genes[idx] = 0
    return genes


def _repair_require_one(
    genes: np.ndarray,
    feature_names: List[str],
    require_groups: List[List[str]],
    rng: np.random.Generator,
    protected_idx: Optional[set[int]] = None,
    collinear_groups: Optional[List[List[str]]] = None,
) -> np.ndarray:
    """
    Repair a chromosome so AT LEAST ONE gene per require-one group is active.

    Companion to _repair_collinearity. When a group has no active gene, one
    member is turned on - preferring a member that doesn't immediately
    conflict with an already-active partner in collinear_groups, so this
    activation doesn't just get undone by a later collinearity repair pass.
    Always called AFTER _repair_collinearity at every call site so the
    "at least one" guarantee is the final word.
    """
    genes = genes.copy()
    protected_idx = protected_idx or set()
    collinear_groups = collinear_groups or []
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    def _conflicts(idx: int) -> bool:
        for cg in collinear_groups:
            cg_idx = {feat_idx[f] for f in cg if f in feat_idx}
            if idx in cg_idx and any(genes[j] == 1 for j in cg_idx if j != idx):
                return True
        return False

    for group in require_groups:
        member_idx = [feat_idx[f] for f in group if f in feat_idx]
        if not member_idx or any(genes[i] == 1 for i in member_idx):
            continue
        safe = [i for i in member_idx if not _conflicts(i)]
        pick = int(rng.choice(safe if safe else member_idx))
        genes[pick] = 1
    return genes


def _enforce_feature_bounds(
    genes: np.ndarray,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
    feature_names: Optional[List[str]] = None,
    collinear_groups: Optional[List[List[str]]] = None,
    protected_idx: Optional[set[int]] = None,
    blocked_features: Optional[set[str]] = None,
) -> np.ndarray:
    """Repair active-feature count after any other chromosome repair."""
    genes = genes.copy()
    protected_idx = protected_idx or set()
    feature_names = feature_names or []
    collinear_groups = collinear_groups or []
    blocked_features = blocked_features or set()

    # If min==max, strict enforcement can collapse diversity. Keep a small
    # buffer so the GA can still explore nearby feature sets.
    flexible_max = max_features
    if max_features <= min_features:
        max_allowed = max(0, len(genes) - len(protected_idx))
        flexible_max = min(max_allowed, min_features + 3)
        flexible_max = max(flexible_max, min_features)

    genes[list(protected_idx)] = 1

    while int(genes.sum()) > flexible_max:
        removable = [idx for idx in np.where(genes == 1)[0] if idx not in protected_idx]
        if not removable:
            break
        genes[int(rng.choice(removable))] = 0

    max_attempts = 100
    attempts = 0
    while int(genes.sum()) < min_features and attempts < max_attempts:
        attempts += 1
        inactive = [idx for idx in np.where(genes == 0)[0]
                    if idx not in protected_idx
                    and (not feature_names or feature_names[idx] not in blocked_features)]
        if not inactive:
            break
        pick = int(rng.choice(inactive))
        genes[pick] = 1
        if collinear_groups and feature_names:
            genes = _repair_collinearity(
                genes, feature_names, collinear_groups, rng,
                protected_idx=protected_idx,
            )
            genes[list(protected_idx)] = 1
    return genes


def _combine_segment_feature_results(
    stage: int,
    segment_results: Dict[str, GAResult],
    feature_map: Dict[str, List[str]],
    fitness_metric: str,
    n_restarts: int,
) -> GAResult:
    """Bundle per-segment GA outputs when each segment uses its own feature space."""
    all_feature_names = sorted({
        feat
        for res in segment_results.values()
        for feat in getattr(res, 'feature_names', [])
    })
    best_fitness = (float(np.mean([r.best_fitness for r in segment_results.values()]))
                    if segment_results else -1e9)
    history = [{
        'generation': 0,
        'best_fitness': best_fitness,
        'best_so_far': best_fitness,
        'mean_fitness': best_fitness,
        'worst_fitness': best_fitness,
        'diversity': 0.0,
        'mutation_rate': 0.0,
        'n_features': int(sum(len(v) for v in feature_map.values())),
        'stagnation': 0,
    }]
    dummy = Individual(genes=np.zeros(len(all_feature_names), dtype=int), fitness=best_fitness)
    combined = GAResult(
        stage=stage,
        best_features=feature_map,
        best_fitness=best_fitness,
        best_individual=dummy,
        history=history,
        all_candidates=[r.best_individual.copy() for r in segment_results.values()],
        feature_names=all_feature_names,
        fitness_metric=fitness_metric,
        n_restarts=n_restarts,
        cache_hits=sum(r.cache_hits for r in segment_results.values()),
    )
    combined.metadata['segment_results'] = segment_results
    return combined


# ---------------------------------------------------------------------------
# 0.  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One chromosome = one feature mask."""
    genes:    np.ndarray
    fitness:  float = -np.inf
    metadata: dict  = field(default_factory=dict)

    def key(self) -> str:
        """Hashable representation for caching."""
        return self.genes.tobytes()

    def copy(self):
        return Individual(
            genes    = self.genes.copy(),
            fitness  = self.fitness,
            metadata = copy.deepcopy(self.metadata),
        )


@dataclass
class GAResult:
    """Output of one GA run."""
    stage:           int
    best_features:   Any          # List[str] or dict of lists
    best_fitness:    float
    best_individual: Individual
    history:         List[dict]
    all_candidates:  List[Individual]
    feature_names:   List[str]
    fitness_metric:  str
    n_restarts:      int = 0
    cache_hits:      int = 0
    metadata:        dict = field(default_factory=dict)

    def history_summary(self) -> pd.DataFrame:
        """Compact per-generation summary for reporting and debugging."""
        if not self.history:
            return pd.DataFrame(columns=[
                'generation', 'best_fitness', 'best_so_far', 'mean_fitness',
                'worst_fitness', 'diversity', 'mutation_rate', 'n_features',
                'stagnation'
            ])
        cols = [
            'generation', 'best_fitness', 'best_so_far', 'mean_fitness',
            'worst_fitness', 'diversity', 'mutation_rate', 'n_features',
            'stagnation'
        ]
        return pd.DataFrame(self.history).reindex(columns=cols)

    def plot_convergence(self, ax=None, save_path: Optional[str] = None,
                         segment_label: str = ''):
        """
        Plot GA convergence.

        For BIC-based metrics (fitness_metric contains 'BIC'):
          - Values are stored as -BIC/n (negative, maximised by GA)
          - Displayed as BIC/n (positive, decreasing = improving) for clarity
          - Y-axis label says "BIC/n  (lower = better)"

        For other metrics (rho┬▓, LL, etc.): displayed as-is, higher = better.
        """
        import matplotlib.pyplot as plt
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')

        gens        = [h['generation']                        for h in self.history]
        best        = [h['best_fitness']                      for h in self.history]
        best_so_far = [h.get('best_so_far', h['best_fitness']) for h in self.history]
        mean        = [h['mean_fitness']                      for h in self.history]
        worst       = [h['worst_fitness']                     for h in self.history]
        #todo why ia this not assesed and why 0 and np.nan
        div         = [h.get('diversity', 0)                  for h in self.history]
        mut         = [h.get('mutation_rate', np.nan)         for h in self.history]

        # For BIC metrics: negate to display BIC/n (positive, lower = better).
        # The curve naturally decreases over generations  no axis inversion needed.
        is_bic = 'BIC' in self.fitness_metric
        if is_bic:
            best        = [-v for v in best]
            best_so_far = [-v for v in best_so_far]
            mean        = [-v for v in mean]
            worst       = [-v for v in worst]
            ylabel = 'BIC / n  (lower = better)'
            fill_lo, fill_hi = worst, best_so_far   # worst=high BIC (top), best=low BIC (bottom)
        else:
            ylabel = self.fitness_metric
            fill_lo, fill_hi = worst, best_so_far

        # Proposed-solution range: band of ALL evaluated candidates,
        # including rejected moves.  Shows how broadly the search explores.
        has_proposed = any(h.get('proposed_min') is not None for h in self.history)
        if has_proposed:
            raw_pmn = [h.get('proposed_min') for h in self.history]
            raw_pmx = [h.get('proposed_max') for h in self.history]
            if is_bic:
                pmn = [-v if v is not None else np.nan for v in raw_pmx]
                pmx = [-v if v is not None else np.nan for v in raw_pmn]
            else:
                pmn = [v if v is not None else np.nan for v in raw_pmn]
                pmx = [v if v is not None else np.nan for v in raw_pmx]
            valid = [(g, lo, hi) for g, lo, hi in zip(gens, pmn, pmx)
                     if np.isfinite(lo) and np.isfinite(hi)]
            if valid:
                pg, plo, phi = zip(*valid)
                ax.fill_between(pg, plo, phi, alpha=0.08, color='#CC7733',
                                label='proposed candidates')

        if has_proposed:
            # SA/Bandit mode: only best-so-far, current, and proposed
            best_label = ('best-so-far (BIC/n)' if is_bic
                          else f'best-so-far ({self.fitness_metric})')
            ax.plot(gens, best_so_far, color='#534AB7', linewidth=2.2,
                    label=best_label, zorder=5)
            ax.plot(gens, best, color='#A59FF0', linewidth=1.2,
                    linestyle='-', label='current solution', zorder=4)
        else:
            # GA mode: full population visualisation
            ax.fill_between(gens, fill_lo, fill_hi, alpha=0.10, color='#534AB7',
                            label='population range')

            ax.plot(gens, mean, color='#888780', linewidth=1.2,
                    linestyle='--', label='population mean', zorder=3)
            ax.plot(gens, best, color='#A59FF0', linewidth=1.0,
                    linestyle=':', label='generation best', zorder=4)
            best_label = ('best-so-far (BIC/n)' if is_bic
                          else f'best-so-far ({self.fitness_metric})')
            ax.plot(gens, best_so_far, color='#534AB7', linewidth=2.2,
                    label=best_label, zorder=5)

        # Annotate final best value
        if gens:
            final_val = best_so_far[-1]
            ax.annotate(
                f'{final_val:.4f}',
                xy=(gens[-1], final_val),
                xytext=(-30, 10), textcoords='offset points',
                fontsize=8, color='#534AB7',
                arrowprops=dict(arrowstyle='->', color='#534AB7', lw=0.8),
            )

        # Stage label and improvement annotation
        stage_labels = {
             0: 'CDAP', -1: 'Usual Loc',
             1: 'Stage 1 • Activity Freq',
             2: 'Stage 2 • Day Duration',
             3: 'Stage 3 • Activity Duration',
             4: 'Stage 4 • Schedule Placement',
             5: 'Stage 5 • Dest + Mode',
            51: 'Stage 5.1 • Trip Mode',
             6: 'Stage 6 • Mode Choice',
        }
        slabel = stage_labels.get(self.stage, f'Stage {self.stage}')
        if segment_label:
            slabel += f'  [{segment_label}]'
        improvement = (best_so_far[-1] - best_so_far[0]) if len(best_so_far) >= 2 else 0.0
        impr_sign   = 'v' if is_bic else '^'
        ax.set_title(
            f'{slabel}\n'
            f'restarts={self.n_restarts}  cache_hits={self.cache_hits}  '
            f'delta={impr_sign}{abs(improvement):.4f}',
            fontsize=9, fontweight='normal', pad=4,
        )
        ax.set_xlabel('Generation', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        # Restart boundary markers ÔÇö vertical dashed lines at each restart start
        restart_marks = [h['generation'] for h in self.history
                         if h.get('is_restart_start', False)]
        for i, rg in enumerate(restart_marks):
            ax.axvline(rg, color='#CC5533', linewidth=1.0, linestyle='--',
                       alpha=0.55, zorder=2,
                       label='restart' if i == 0 else '_nolegend_')

        # Y-axis: centre tightly around the data so convergence is visible.
        # For BIC plots the full y-range is often tiny (< 0.1 BIC/n units) so
        # anchoring at 0 compresses all curves into an unreadable sliver at the top.
        _all_vals = [v for v in (best_so_far + mean + worst) if np.isfinite(v)]
        if _all_vals:
            _lo, _hi = min(_all_vals), max(_all_vals)
            _pad = max((_hi - _lo) * 0.25, 0.002)   # at least 0.002 padding
            ax.set_ylim(_lo - _pad, _hi + _pad)

        # BIC/n decreases as GA improves ÔåÆ converged end is at the bottom-right.
        # Place legend in upper right where there is room above the converged level.
        ax.legend(fontsize=8, frameon=False,
                  loc='upper right',
                  ncol=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(color='#e8e8e3', linewidth=0.5, zorder=0)
        ax.tick_params(labelsize=8)

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()

    def feature_importance(self) -> pd.DataFrame:
        """Frequency each feature appeared in top-20% of population."""
        top_n  = max(1, len(self.all_candidates) // 5)
        ranked = sorted(self.all_candidates,
                        key=lambda x: x.fitness, reverse=True)[:top_n]
        freq   = np.array([ind.genes for ind in ranked]).mean(axis=0)
        return (
            pd.DataFrame({'feature': self.feature_names, 'selection_freq': freq})
            .sort_values('selection_freq', ascending=False)
            .reset_index(drop=True)
        )

    def print_summary(self):
        hist = self.history_summary()
        improvement = 0.0
        if not hist.empty:
            improvement = float(hist['best_so_far'].iloc[-1] - hist['best_so_far'].iloc[0])
        is_bic = 'BIC' in self.fitness_metric
        print(f"\nStage {self.stage} GA result")
        print(f"  Fitness metric : {self.fitness_metric}")
        if is_bic:
            print(f"  Best BIC/n     : {-self.best_fitness:.4f}  (fitness={self.best_fitness:.4f})")
        else:
            print(f"  Best fitness   : {self.best_fitness:.4f}")
        print(f"  Improvement    : {improvement:.4f}")
        print(f"  Best features  : {self.best_features}")
        print(f"  Generations    : {len(self.history)}")
        print(f"  Restarts done  : {self.n_restarts}")
        print(f"  Cache hits     : {self.cache_hits}")


@dataclass
class AllStagesResult:
    """Container for GA results across all stages."""
    results: Dict[int, GAResult] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for stage, res in sorted(self.results.items(), key=lambda x: (x[0] < 0, x[0])):
            hist = res.history_summary()
            rows.append({
                'stage': stage,
                'fitness_metric': res.fitness_metric,
                'best_fitness': res.best_fitness,
                'final_best_so_far': (
                    float(hist['best_so_far'].iloc[-1]) if not hist.empty else np.nan
                ),
                'generations': len(res.history),
                'restarts': res.n_restarts,
                'cache_hits': res.cache_hits,
                'selected_feature_count': (
                    sum(len(v) for v in res.best_features.values())
                    if isinstance(res.best_features, dict)
                    else len(res.best_features)
                ),
                'selected_features': (
                    '; '.join(f'{k}={v}' for k, v in res.best_features.items())
                    if isinstance(res.best_features, dict)
                    else ', '.join(res.best_features)
                ),
            })
        return pd.DataFrame(rows)

    def export_reports(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        summary = self.summary_table()
        if not summary.empty:
            summary.to_csv(os.path.join(out_dir, 'ga_summary.csv'), index=False)
            summary.to_html(
                os.path.join(out_dir, 'ga_summary.html'),
                index=False,
                float_format=lambda x: f'{x:.4f}' if pd.notna(x) else '',
            )

        def _flatten_features(feats: Any) -> List[str]:
            if isinstance(feats, dict):
                items: List[str] = []
                for group, vals in feats.items():
                    if vals is None:
                        continue
                    if isinstance(vals, (list, tuple, set)):
                        for v in vals:
                            items.append(f"{group}:{v}")
                    else:
                        items.append(f"{group}:{vals}")
                return items
            if feats is None:
                return []
            if isinstance(feats, (list, tuple, set)):
                return [str(v) for v in feats]
            return [str(feats)]

        def _clean_segment_label(label: Any) -> str:
            txt = str(label)
            txt = txt.replace('ÔÇö', ' - ').replace('—', ' - ').replace('–', ' - ')
            txt = re.sub(r'\s+', ' ', txt).strip()
            return txt

        for stage, res in self.results.items():
            hist = res.history_summary()
            seg_results = res.metadata.get('segment_results', {}) if isinstance(res.metadata, dict) else {}
            if seg_results:
                seg_hist_frames = []
                for seg_label, seg_res in seg_results.items():
                    seg_hist = seg_res.history_summary()
                    if seg_hist.empty:
                        continue
                    seg_hist.insert(0, 'segment', _clean_segment_label(seg_label))
                    seg_hist_frames.append(seg_hist)
                if seg_hist_frames:
                    seg_hist_df = pd.concat(seg_hist_frames, ignore_index=True)
                    seg_hist_df.to_csv(
                        os.path.join(out_dir, f'ga_stage{stage}_segment_history.csv'),
                        index=False,
                    )
                    # Keep ga_stage*_history.csv meaningful by aggregating real
                    # per-segment search traces instead of a synthetic single row.
                    hist = (
                        seg_hist_df
                        .groupby('generation', as_index=False)
                        .agg({
                            'best_fitness': 'max',
                            'best_so_far': 'max',
                            'mean_fitness': 'mean',
                            'worst_fitness': 'min',
                            'diversity': 'mean',
                            'mutation_rate': 'mean',
                            'n_features': 'mean',
                            'stagnation': 'max',
                        })
                        .sort_values('generation')
                    )
                    hist['n_features'] = hist['n_features'].round().astype(int)
            if not hist.empty:
                hist.to_csv(os.path.join(out_dir, f'ga_stage{stage}_history.csv'), index=False)

            # Explicit per-stage best-model artifact (easy to inspect from run folders)
            best_txt = os.path.join(out_dir, f'ga_stage{stage}_best_model.txt')
            try:
                with open(best_txt, 'w', encoding='utf-8') as handle:
                    handle.write(f"stage={stage}\n")
                    handle.write(f"fitness_metric={res.fitness_metric}\n")
                    handle.write(f"best_fitness={res.best_fitness:.10f}\n")
                    if 'BIC' in res.fitness_metric and np.isfinite(res.best_fitness):
                        handle.write(f"best_bic_per_obs={-res.best_fitness:.10f}\n")
                    if isinstance(res.best_features, dict):
                        for k, vals in res.best_features.items():
                            handle.write(f"{k}={','.join(map(str, vals))}\n")
                    else:
                        handle.write(f"features={','.join(map(str, res.best_features))}\n")
            except Exception as exc:
                logger.warning(f"Could not write {best_txt}: {exc}")

            rows = []
            if isinstance(res.best_features, dict):
                for group, vals in res.best_features.items():
                    if not vals:
                        rows.append({'stage': stage, 'group': group, 'feature': ''})
                    else:
                        for feat in vals:
                            rows.append({'stage': stage, 'group': group, 'feature': feat})
            else:
                if not res.best_features:
                    rows.append({'stage': stage, 'group': 'all', 'feature': ''})
                else:
                    for feat in res.best_features:
                        rows.append({'stage': stage, 'group': 'all', 'feature': feat})
            try:
                pd.DataFrame(rows).to_csv(
                    os.path.join(out_dir, f'ga_stage{stage}_best_features.csv'),
                    index=False,
                )
            except Exception as exc:
                logger.warning(f"Could not write ga_stage{stage}_best_features.csv: {exc}")

            if seg_results:
                seg_rows = []
                for seg_label, seg_res in seg_results.items():
                    flat_feats = _flatten_features(seg_res.best_features)
                    seg_rows.append({
                        'stage': stage,
                        'segment': _clean_segment_label(seg_label),
                        'best_fitness': seg_res.best_fitness,
                        'best_bic_per_obs': -seg_res.best_fitness if 'BIC' in seg_res.fitness_metric else np.nan,
                        'n_features': len(flat_feats),
                        'best_features': ', '.join(flat_feats),
                    })
                try:
                    pd.DataFrame(seg_rows).to_csv(
                        os.path.join(out_dir, f'ga_stage{stage}_segment_best.csv'),
                        index=False,
                    )
                except Exception as exc:
                    logger.warning(f"Could not write ga_stage{stage}_segment_best.csv: {exc}")

    def summary(self):
        STAGE_LABELS = {
             0: 'CDAP (daily activity pattern)',
            -1: 'Usual location (decay + attractors)',
             1: 'Stage 1  activity frequency',
             2: 'Stage 2  day duration',
             3: 'Stage 3 activity duration',
             4: 'Stage 4 schedule placement',
             5: 'Stage 5 destination + mode',
             6: 'Stage 6 mode choice',
        }
        print("\n" + "="*60)
        print("GA HYPOTHESIS TESTING   ALL STAGES  (estimation only)")
        print("="*60)
        for stage, res in sorted(self.results.items(),
                                  key=lambda x: (x[0] < 0, x[0])):
            label = STAGE_LABELS.get(stage, f'Stage {stage}')
            print(f"\n{label}")
            print(f"  Metric    : {res.fitness_metric} = {res.best_fitness:.4f}")
            print(f"  Restarts  : {res.n_restarts}  |  Cache hits: {res.cache_hits}")
            if isinstance(res.best_features, dict):
                for k, v in res.best_features.items():
                    print(f"  {k:12}: {v}")
            else:
                print(f"  Selected  : {res.best_features}")

    def plot_all(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        # Collect main results + any per-segment sub-results stored in metadata
        plot_items = []  # list of (GAResult, segment_label)
        for stage, res in sorted(self.results.items(), key=lambda x: (x[0] < 0, x[0])):
            seg_results = res.metadata.get('segment_results', {})
            if seg_results:
                for seg_label, seg_res in seg_results.items():
                    plot_items.append((seg_res, seg_label))
            else:
                plot_items.append((res, ''))

        n    = len(plot_items)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(13, 4.5 * rows),
                                 facecolor='white', squeeze=False)
        axes = axes.flatten()
        for i, (res, seg_label) in enumerate(plot_items):
            if i < len(axes):
                res.plot_convergence(ax=axes[i], segment_label=seg_label)
        for j in range(len(plot_items), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('GA Variable Selection   All Stages', fontsize=13,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def save(self, path: str = 'saved_data/ga_results.pkl'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"GA results saved -> {path}")

    @staticmethod
    def load(path: str) -> 'AllStagesResult':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 1.  CORE GA ENGINE  (with caching, adaptive mutation, multi-restart)
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Binary-encoded GA with:
    - Fitness caching (no re-evaluation of seen gene patterns)
    - Adaptive mutation (increases when diversity drops)
    - Diversity tracking (mean pairwise Hamming distance)
    - Stagnation detection (triggers early restart / boosted mutation)
    """

    def __init__(
        self,
        n_features:          int,
        fitness_fn:          Callable[[np.ndarray], float],
        pop_size:            int   = 40,
        n_generations:       int   = 60,
        mutation_rate:       float = 0.18,
        mutation_rate_max:   float = 0.35,    # boosted rate when diversity low
        diversity_threshold: float = 0.30,    # fraction below which mutation boosts
        crossover_rate:      float = 0.75,
        elitism_k:           int   = 4,
        min_features:        int   = 3,
        max_features:        int   = None,    # None = no upper limit
        tournament_k:        int   = 4,
        stagnation_limit:    int   = 12,      # gens without improvement ÔåÆ mutation boost
        seed:                int   = 42,
        verbose:             bool  = True,
        feature_names:       List[str] = None,    # required for collinearity repair
        collinear_groups:    List[List[str]] = None,  # from STAGE_COLLINEAR_GROUPS
        require_one_groups:  List[List[str]] = None,  # from STAGE_REQUIRE_ONE_GROUPS
        fixed_features:      Optional[List[str]] = None,
        progress_label:      str = '',
        fitness_metric:      str = 'mean(-BIC/n)',
    ):
        self.n_features          = n_features
        self.fitness_fn          = fitness_fn
        self.pop_size            = pop_size
        self.n_generations       = n_generations
        self.mutation_rate       = mutation_rate
        self.mutation_rate_max   = mutation_rate_max
        self.diversity_threshold = diversity_threshold
        self.crossover_rate      = crossover_rate
        self.elitism_k           = elitism_k
        self.min_features        = min_features
        self.max_features        = max_features or n_features
        self.tournament_k        = tournament_k
        self.stagnation_limit    = stagnation_limit
        self.verbose             = verbose
        self.rng                 = np.random.default_rng(seed)
        self.feature_names       = feature_names or []
        self.collinear_groups    = collinear_groups or []
        self.require_one_groups  = require_one_groups or []
        self.fixed_features      = fixed_features or []
        self.progress_label      = progress_label
        self.fitness_metric      = fitness_metric
        self._fixed_idx          = {
            i for i, name in enumerate(self.feature_names)
            if name in set(self.fixed_features)
        }

        # fitness cache   shared across restarts
        self._cache: dict = {}
        self.cache_hits   = 0

    def _active_feature_names(self, genes: np.ndarray, limit: int = 8) -> str:
        if not self.feature_names:
            return f"n={int(genes.sum())}"
        active = [name for name, flag in zip(self.feature_names, genes) if flag == 1]
        if len(active) > limit:
            shown = active[:limit] + [f"... (+{len(active) - limit})"]
        else:
            shown = active
        return str(shown)

    # ÔöÇÔöÇ initialisation ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def _random_individual(self, n_active: int = None) -> Individual:
        min_active = max(self.min_features, len(self._fixed_idx))
        if n_active is None:
            hi = max(min_active + 1, self.max_features + 1)
            n_active = int(self.rng.integers(min_active, hi))
        n_active = max(min_active, min(n_active, self.max_features))
        genes    = np.zeros(self.n_features, dtype=int)
        if self._fixed_idx:
            genes[list(self._fixed_idx)] = 1
        extra_needed = max(0, n_active - int(genes.sum()))
        candidate_idx = [i for i in range(self.n_features) if i not in self._fixed_idx]
        if extra_needed > 0 and candidate_idx:
            active = self.rng.choice(candidate_idx, size=min(extra_needed, len(candidate_idx)), replace=False)
            genes[active] = 1
        if self.collinear_groups and self.feature_names:
            genes = _repair_collinearity(genes, self.feature_names,
                                         self.collinear_groups, self.rng,
                                         protected_idx=self._fixed_idx)
        genes = _enforce_feature_bounds(
            genes, min_active, self.max_features, self.rng,
            feature_names=self.feature_names,
            collinear_groups=self.collinear_groups,
            protected_idx=self._fixed_idx,
        )
        if self.require_one_groups and self.feature_names:
            genes = _repair_require_one(genes, self.feature_names,
                                        self.require_one_groups, self.rng,
                                        protected_idx=self._fixed_idx,
                                        collinear_groups=self.collinear_groups)
        return Individual(genes=genes)

    def _init_population(self, seed_individuals: list = None) -> List[Individual]:
        """
        Stratified initialisation: spread the population across four density
        levels so the GA doesn't converge from a single sparsity mode.

          25% sparse    (min_features  .. 25th percentile of range)
          25% medium    (25th  .. 50th percentile)
          25% rich      (50th  .. 75th percentile)
          25% dense     (75th percentile .. max_features)

        Seeded elites from a previous run are inserted first.  Stratification
        ensures the initial BIC landscape is well-explored even when the true
        best model is moderately large.
        """
        pop = []
        if seed_individuals:
            for ind in seed_individuals[:self.elitism_k]:
                pop.append(ind.copy())

        span  = max(self.max_features - self.min_features, 1)
        q1    = self.min_features + span // 4
        q2    = self.min_features + span // 2
        q3    = self.min_features + 3 * span // 4

        def _band_size(idx: int) -> int:
            """Number of individuals to generate for band idx (0-3)."""
            remaining = self.pop_size - len(pop)
            n_bands   = max(1, 4 - idx)
            return max(1, remaining // n_bands)

        for lo, hi, _idx in [
            (self.min_features, max(self.min_features, q1),  0),
            (max(self.min_features, q1), max(q1 + 1, q2),    1),
            (max(q1 + 1, q2),            max(q2 + 1, q3),    2),
            (max(q2 + 1, q3),            self.max_features,  3),
        ]:
            for _ in range(_band_size(_idx)):
                if len(pop) >= self.pop_size:
                    break
                n_act = int(self.rng.integers(lo, hi + 1)) if lo < hi else lo
                pop.append(self._random_individual(n_act))

        while len(pop) < self.pop_size:
            pop.append(self._random_individual())
        return pop

    # ÔöÇÔöÇ diversity ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def _diversity(self, pop: List[Individual]) -> float:
        """Mean pairwise Hamming distance, normalised to [0,1]."""
        if len(pop) < 2:
            return 0.0
        genes = np.array([ind.genes for ind in pop], dtype=float)
        n     = len(pop)
        dists = []
        for i in range(min(n, 20)):   # sample pairs for speed
            j = self.rng.integers(0, n)
            if i != j:
                dists.append(np.mean(genes[i] != genes[j]))
        return float(np.mean(dists)) if dists else 0.0

    # ÔöÇÔöÇ selection ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        idx      = self.rng.choice(len(pop), size=self.tournament_k, replace=False)
        best_idx = max(idx, key=lambda i: pop[i].fitness)
        return pop[best_idx].copy()

    # ÔöÇÔöÇ crossover ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        # uniform crossover   each gene independently inherited from either parent
        mask = self.rng.integers(0, 2, size=self.n_features).astype(bool)
        g1   = np.where(mask, p1.genes, p2.genes)
        g2   = np.where(mask, p2.genes, p1.genes)
        return Individual(genes=g1), Individual(genes=g2)

    # ÔöÇÔöÇ mutation ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def _mutate(self, ind: Individual, rate: float) -> Individual:
        genes = ind.genes.copy()
        mutable_idx = np.array([i for i in range(self.n_features) if i not in self._fixed_idx], dtype=int)
        flip = np.zeros(self.n_features, dtype=bool)
        if len(mutable_idx) > 0:
            flip[mutable_idx] = self.rng.random(len(mutable_idx)) < rate
        genes[flip] = 1 - genes[flip]
        if self._fixed_idx:
            genes[list(self._fixed_idx)] = 1

        # repair: enforce min/max features
        if self.collinear_groups and self.feature_names:
            genes = _repair_collinearity(genes, self.feature_names,
                                         self.collinear_groups, self.rng,
                                         protected_idx=self._fixed_idx)
        genes = _enforce_feature_bounds(
            genes,
            max(self.min_features, len(self._fixed_idx)),
            self.max_features,
            self.rng,
            feature_names=self.feature_names,
            collinear_groups=self.collinear_groups,
            protected_idx=self._fixed_idx,
        )
        if self.require_one_groups and self.feature_names:
            genes = _repair_require_one(genes, self.feature_names,
                                        self.require_one_groups, self.rng,
                                        protected_idx=self._fixed_idx,
                                        collinear_groups=self.collinear_groups)

        return Individual(genes=genes)

    # ÔöÇÔöÇ evaluation with caching ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def _evaluate_one(self, ind: Individual) -> Individual:
        key = ind.key()
        if key in self._cache:
            ind.fitness = self._cache[key]
            self.cache_hits += 1
        elif ind.fitness == -np.inf:
            try:
                ind.fitness = self.fitness_fn(ind.genes)
            except Exception as e:
                logger.warning(f"fitness_fn error: {e}")
                ind.fitness = -1e9
            self._cache[key] = ind.fitness
        return ind

    def _evaluate(self, pop: List[Individual]) -> List[Individual]:
        for ind in pop:
            self._evaluate_one(ind)
        return pop

    # ÔöÇÔöÇ main loop ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def run(self, seed_individuals: list = None) -> Tuple[Individual, List[dict], List[Individual]]:
        """
        Run the GA.

        Parameters
        ----------
        seed_individuals : optional list of Individual from a previous run
                           (used for multi-restart warm-starting)

        Returns
        -------
        best_ever : best Individual seen across all generations
        history   : list of per-generation stat dicts
        pop       : final population
        """
        pop      = self._init_population(seed_individuals)
        pop      = self._evaluate(pop)
        history  = []
        best_ever = max(pop, key=lambda x: x.fitness).copy()
        stagnation_count = 0
        invalid_warned = False

        # ── Hard early-stop on convergence ───────────────────────────────────
        # Most segments find their best solution within a handful of generations
        # (e.g. by gen ~11) and then run the full n_generations budget re-searching
        # a converged space — the dominant, avoidable cost of a staged run. Stop
        # once the best fitness has not improved for `patience` generations, after
        # a `min_gens` exploration floor (so the diversity-injection / mutation
        # boost above still get a chance to escape a local optimum first).
        # Disable with GA_EARLY_STOP=0; tune with GA_EARLY_STOP_PATIENCE /
        # GA_EARLY_STOP_MIN_GENS.
        import os as _os
        _early_stop = _os.environ.get('GA_EARLY_STOP', '1').strip() != '0'

        def _env_int(_name: str, _default: int) -> int:
            try:
                _v = int(_os.environ.get(_name, '0'))
                return _v if _v > 0 else _default
            except ValueError:
                return _default

        _es_patience = _env_int('GA_EARLY_STOP_PATIENCE', max(3 * self.stagnation_limit, 24))
        _es_min_gens = _env_int('GA_EARLY_STOP_MIN_GENS', max(2 * self.stagnation_limit, 20))

        for gen in range(self.n_generations):
            print(f"Gen {gen:3d} | evaluating population... ", end='', flush=True)
            fitnesses = [ind.fitness for ind in pop]
            gen_best  = max(pop, key=lambda x: x.fitness)
            div       = self._diversity(pop)

            if gen_best.fitness > best_ever.fitness:
                best_ever        = gen_best.copy()
                stagnation_count = 0
                if self.verbose:
                    label = f" [{self.progress_label}]" if self.progress_label else ""
                    metric = f"BIC/n={-best_ever.fitness:.4f}" if best_ever.fitness > -1e8 else f"fitness={best_ever.fitness:.4f}"
                    print(f"New best{label}: {metric}  feats={self._active_feature_names(best_ever.genes)}")
            else:
                stagnation_count += 1

            if not invalid_warned and gen_best.fitness <= -1e8:
                invalid_warned = True
                label = f" [{self.progress_label}]" if self.progress_label else ""
                print(
                    f"! No valid model yet{label}: all candidates are at the sentinel penalty. "
                    "Check estimator availability, data coverage, or model setup."
                )

            # adaptive mutation: boost when stagnating or diversity low
            if stagnation_count >= self.stagnation_limit or div < self.diversity_threshold:
                cur_rate = self.mutation_rate_max
            else:
                # linearly scale between base and max based on diversity
                t        = max(0.0, 1.0 - div / max(self.diversity_threshold, 1e-6))
                cur_rate = self.mutation_rate + t * (self.mutation_rate_max - self.mutation_rate)

            # Diversity injection: when population has fully collapsed (div==0)
            # for at least stagnation_limit gens, reinitialise all non-elite
            # members from scratch so the search can escape.
            if div == 0.0 and stagnation_count >= self.stagnation_limit:
                elites_now = sorted(pop, key=lambda x: x.fitness, reverse=True)[:self.elitism_k]
                fresh = [self._random_individual() for _ in range(self.pop_size - len(elites_now))]
                pop = self._evaluate(elites_now + fresh)
                if self.verbose:
                    label = f" [{self.progress_label}]" if self.progress_label else ""
                    print(f" ~ Diversity injection at gen {gen}{label}: reinitialised "
                          f"{len(fresh)} members (div was 0, stag={stagnation_count})")

            history.append({
                'generation':    gen,
                'best_fitness':  gen_best.fitness,
                'best_so_far':   best_ever.fitness,
                'mean_fitness':  float(np.mean(fitnesses)),
                'worst_fitness': float(np.min(fitnesses)),
                'n_features':    int(gen_best.genes.sum()),
                'diversity':     div,
                'mutation_rate': cur_rate,
                'stagnation':    stagnation_count,
            })

            if self.verbose and gen % 5 == 0:
                if 'BIC' in self.fitness_metric:
                    print(f"  Gen {gen:3d} | best_BIC/n={-gen_best.fitness:.4f} "
                          f"mean_BIC/n={-np.mean(fitnesses):.4f} "
                          f"n_feat={int(gen_best.genes.sum())} "
                          f"div={div:.2f} mut={cur_rate:.3f} "
                          f"stag={stagnation_count}")
                else:
                    print(f"  Gen {gen:3d} | best={gen_best.fitness:.4f} "
                          f"mean={np.mean(fitnesses):.4f} "
                          f"n_feat={int(gen_best.genes.sum())} "
                          f"div={div:.2f} mut={cur_rate:.3f} "
                          f"stag={stagnation_count}")

            # ── Early-stop: best converged, no improvement for `patience` gens ──
            if (_early_stop and gen >= _es_min_gens
                    and stagnation_count >= _es_patience):
                label = f" [{self.progress_label}]" if self.progress_label else ""
                _metric = (f"BIC/n={-best_ever.fitness:.4f}"
                           if best_ever.fitness > -1e8 else f"fitness={best_ever.fitness:.4f}")
                print(f"  Early stop{label} at gen {gen}/{self.n_generations}: "
                      f"best unchanged for {stagnation_count} gens "
                      f"(patience={_es_patience}); {_metric}")
                break

            # elitism
            elites  = sorted(pop, key=lambda x: x.fitness, reverse=True)[:self.elitism_k]
            new_pop = [e.copy() for e in elites]

            while len(new_pop) < self.pop_size:
                p1     = self._tournament_select(pop)
                p2     = self._tournament_select(pop)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1, cur_rate))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2, cur_rate))

            pop = self._evaluate(new_pop)
            # Track range of non-elite offspring for plot proposed-candidates band
            _prop = [ind.fitness for ind in pop[self.elitism_k:]
                     if np.isfinite(ind.fitness) and ind.fitness > -1e8]
            if _prop and history:
                history[-1]['proposed_min'] = float(min(_prop))
                history[-1]['proposed_max'] = float(max(_prop))

        return best_ever, history, pop

    def run_with_restarts(
        self,
        n_restarts:       int  = 1,
        restart_gens:     int  = None,
        seed_individuals: list = None,
    ) -> Tuple[Individual, List[dict], List[Individual]]:
        """
        Run main GA then n_restarts shorter re-runs, keeping the global best.

        Parameters
        ----------
        n_restarts       : number of random-restart runs after the main run
        restart_gens     : generations per restart (default: n_generations // 3)
        seed_individuals : optional list of Individual to warm-start the MAIN run

        The fitness cache is shared across all runs so restarts benefit from
        all evaluations already done in the main run.

        Restart history entries are offset so generation counts are continuous
        across the full run (no reset to 0 on restart).  Each restart's first
        entry is marked with is_restart_start=True so plot_convergence draws
        a vertical boundary marker at the restart point.

        Returns
        -------
        Same as run() but best_ever is the global best across all restarts.
        """
        restart_gens = restart_gens or max(10, self.n_generations // 3)

        print(f"  Main run: {self.n_generations} generations, pop={self.pop_size}")
        best_ever, history, pop = self.run(seed_individuals=seed_individuals)
        print(f"  Main run best: {best_ever.fitness:.4f}")

        orig_gens          = self.n_generations
        self.n_generations = restart_gens
        gen_offset         = len(history)   # keep x-axis continuous across restarts

        for r in range(n_restarts):
            print(f"  Restart {r+1}/{n_restarts}: {restart_gens} generations "
                  f"(cache size={len(self._cache)})")
            # warm-start each restart with the current global elites
            best_r, hist_r, pop_r = self.run(seed_individuals=[best_ever])
            # Offset generation numbers so they never reset to 0
            for i, h in enumerate(hist_r):
                h['generation'] += gen_offset
                if i == 0:
                    h['is_restart_start'] = True   # flag for vertical marker in plot
            gen_offset += len(hist_r)
            if best_r.fitness > best_ever.fitness:
                best_ever = best_r.copy()
                print(f"    ^ New global best: {best_ever.fitness:.4f}")
            history.extend(hist_r)

        self.n_generations = orig_gens
        return best_ever, history, pop


# ---------------------------------------------------------------------------
# 1b.  SIMULATED ANNEALING ENGINE  (alternative searcher for external estimators)
# ---------------------------------------------------------------------------

class SimulatedAnnealing:
    """
    Simulated annealing over the binary feature-selection space.

    Same interface as GeneticAlgorithm: construct with the same kwargs, then
    call run() or run_with_restarts().  Total evaluation budget equals
    pop_size * n_generations steps (the same as the GA), grouped into
    n_generations reporting buckets so history length is identical.

    GA-only parameters (crossover_rate, elitism_k, tournament_k,
    stagnation_limit, mutation_rate_max, diversity_threshold) are accepted
    but ignored so the constructor signature is drop-in compatible.

    Activated when GA_ESTIMATOR is metacount/searchlibrium/both/hybrid; the
    Boltzmann acceptance criterion avoids premature convergence on the rough
    fitness landscapes produced by model-based evaluators.
    """

    def __init__(
        self,
        n_features:          int,
        fitness_fn:          Callable[[np.ndarray], float],
        pop_size:            int   = 40,
        n_generations:       int   = 60,
        mutation_rate:       float = 0.08,   # ignored (SA uses bit-flip step)
        mutation_rate_max:   float = 0.35,   # ignored
        diversity_threshold: float = 0.30,   # ignored
        crossover_rate:      float = 0.75,   # ignored
        elitism_k:           int   = 4,      # ignored
        min_features:        int   = 3,
        max_features:        int   = None,
        tournament_k:        int   = 4,      # ignored
        stagnation_limit:    int   = 12,     # ignored
        seed:                int   = 42,
        verbose:             bool  = True,
        feature_names:       List[str] = None,
        collinear_groups:    List[List[str]] = None,
        require_one_groups:  List[List[str]] = None,
        fixed_features:      Optional[List[str]] = None,
        progress_label:      str  = '',
        fitness_metric:      str  = 'mean(-BIC/n)',
        t_initial:           float = None,   # None = auto-calibrate from first moves
        n_neighbors:         int   = 1,      # bits to flip per SA step
    ):
        self.n_features     = n_features
        self.fitness_fn     = fitness_fn
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.min_features   = min_features
        self.max_features   = max_features or n_features
        self.seed           = seed
        self.verbose        = verbose
        self.rng            = np.random.default_rng(seed)
        self.feature_names  = feature_names or []
        self.collinear_groups = collinear_groups or []
        self.require_one_groups = require_one_groups or []
        self.fixed_features = fixed_features or []
        self.progress_label = progress_label
        self.fitness_metric = fitness_metric
        self.t_initial      = t_initial
        self.n_neighbors    = max(1, n_neighbors)
        self._fixed_idx     = {
            i for i, name in enumerate(self.feature_names)
            if name in set(self.fixed_features)
        }
        self._cache: dict = {}
        self.cache_hits = 0

    def _random_genes(self, n_active: int = None) -> np.ndarray:
        min_active = max(self.min_features, len(self._fixed_idx))
        if n_active is None:
            hi = max(min_active + 1, self.max_features + 1)
            n_active = int(self.rng.integers(min_active, hi))
        n_active = max(min_active, min(n_active, self.max_features))
        genes = np.zeros(self.n_features, dtype=int)
        if self._fixed_idx:
            genes[list(self._fixed_idx)] = 1
        extra = max(0, n_active - int(genes.sum()))
        cands = [i for i in range(self.n_features) if i not in self._fixed_idx]
        if extra > 0 and cands:
            chosen = self.rng.choice(cands, size=min(extra, len(cands)), replace=False)
            genes[chosen] = 1
        if self.collinear_groups and self.feature_names:
            genes = _repair_collinearity(genes, self.feature_names,
                                         self.collinear_groups, self.rng,
                                         protected_idx=self._fixed_idx)
        genes = _enforce_feature_bounds(
            genes, min_active, self.max_features, self.rng,
            feature_names=self.feature_names,
            collinear_groups=self.collinear_groups,
            protected_idx=self._fixed_idx,
        )
        if self.require_one_groups and self.feature_names:
            genes = _repair_require_one(genes, self.feature_names,
                                        self.require_one_groups, self.rng,
                                        protected_idx=self._fixed_idx,
                                        collinear_groups=self.collinear_groups)
        return genes

    def _neighbor(self, genes: np.ndarray) -> np.ndarray:
        """
        Flip n_neighbors mutable bits then repair constraints.

        Bit-flip probabilities are biased by per-feature weakness scores from
        ga_external_estimators: features that repeatedly come out with near-zero
        coefficients are more likely to be removed and less likely to be added,
        steering the search toward leaner, more significant specifications.
        """
        g = genes.copy()
        mutable = [i for i in range(self.n_features) if i not in self._fixed_idx]
        if not mutable:
            return g
        n_flip = min(self.n_neighbors, len(mutable))

        # --- Adaptive flip weights ------------------------------------------
        # Start uniform (1.0 for each mutable index), then apply weakness bias.
        # • Feature is ON  + weak  → weight += w * 0.4   (favour flipping OUT)
        # • Feature is OFF + weak  → weight  = max(0.05, 1 - w * 0.15)  (avoid adding)
        # • No weakness data       → unchanged (weight stays 1.0)
        weights = np.ones(len(mutable), dtype=float)
        if self.feature_names:
            try:
                from ga_external_estimators import get_feature_weakness, is_feature_blacklisted
                weakness = get_feature_weakness()
                if weakness:
                    for wi, feat_idx in enumerate(mutable):
                        if feat_idx >= len(self.feature_names):
                            continue
                        w = weakness.get(self.feature_names[feat_idx], 0.0)
                        if w <= 0.0:
                            continue
                        if g[feat_idx] == 1:
                            weights[wi] += w * 0.4
                        else:
                            weights[wi] = max(0.05, 1.0 - w * 0.15)
                # Permanently blacklisted features: prevent flipping ON
                for wi, feat_idx in enumerate(mutable):
                    if feat_idx >= len(self.feature_names):
                        continue
                    if is_feature_blacklisted(self.feature_names[feat_idx]):
                        if g[feat_idx] == 0:
                            weights[wi] = 0.0  # never flip ON a blacklisted feature
                        else:
                            weights[wi] += 50.0  # strongly favour flipping OFF
            except ImportError:
                pass
        weights = weights / weights.sum()

        idxs = self.rng.choice(mutable, size=n_flip, replace=False, p=weights)
        g[idxs] = 1 - g[idxs]
        if self.collinear_groups and self.feature_names:
            g = _repair_collinearity(g, self.feature_names,
                                     self.collinear_groups, self.rng,
                                     protected_idx=self._fixed_idx)
        g = _enforce_feature_bounds(
            g, max(self.min_features, len(self._fixed_idx)), self.max_features, self.rng,
            feature_names=self.feature_names,
            collinear_groups=self.collinear_groups,
            protected_idx=self._fixed_idx,
        )
        if self.require_one_groups and self.feature_names:
            g = _repair_require_one(g, self.feature_names,
                                    self.require_one_groups, self.rng,
                                    protected_idx=self._fixed_idx,
                                    collinear_groups=self.collinear_groups)
        return g

    def _evaluate(self, genes: np.ndarray) -> float:
        key = tuple(genes.tolist())
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        try:
            val = self.fitness_fn(genes)
        except Exception as e:
            logger.warning("SA fitness_fn error: %s", e)
            val = -1e9
        self._cache[key] = val
        return val

    def _active_names(self, genes: np.ndarray, limit: int = 8) -> str:
        if not self.feature_names:
            return f"n={int(genes.sum())}"
        active = [n for n, g in zip(self.feature_names, genes) if g == 1]
        if len(active) > limit:
            active = active[:limit] + [f'...+{len(active) - limit}']
        return str(active)

    def run(self, seed_individuals: list = None) -> Tuple[Individual, List[dict], List[Individual]]:
        """
        Run SA.  Returns (best_ever, history, [best_individual]) —
        same signature as GeneticAlgorithm.run().
        """
        n_steps_total = self.pop_size * self.n_generations

        cur_genes = (seed_individuals[0].genes.copy()
                     if seed_individuals else self._random_genes())
        cur_fit   = self._evaluate(cur_genes)
        best_genes = cur_genes.copy()
        best_fit   = cur_fit

        # Auto-calibrate T_0: sample ~20 neighbour deltas so ~60% of
        # equal-magnitude worsening moves are accepted at the start.
        if self.t_initial is None:
            deltas = []
            for _ in range(min(20, max(n_steps_total, 1))):
                g = self._neighbor(cur_genes)
                f = self._evaluate(g)
                if np.isfinite(f) and f > -1e8:
                    deltas.append(abs(f - cur_fit))
            t0 = max(float(np.mean(deltas)) if deltas else 0.1, 0.02) * 1.5
        else:
            t0 = float(self.t_initial)

        # Geometric cooling: T_final = T_0 * 1e-3
        t_final = t0 * 1e-3
        alpha   = (t_final / max(t0, 1e-15)) ** (1.0 / max(n_steps_total, 1))
        T       = t0

        history: List[dict] = []

        for gen in range(self.n_generations):
            print(f"SA gen {gen:3d} | T={T:.5f} | evaluating neighbors... ", end='', flush=True)
            bucket_fits: List[float] = []      # accepted-state fitness after each step
            proposed_fits: List[float] = []    # ALL proposed (before accept/reject)
            n_accepted = 0

            for _ in range(self.pop_size):
                cand_genes = self._neighbor(cur_genes)
                cand_fit   = self._evaluate(cand_genes)
                proposed_fits.append(cand_fit)   # record regardless of accept/reject
                delta      = cand_fit - cur_fit

                if delta >= 0 or (T > 1e-15 and self.rng.random() < np.exp(delta / T)):
                    cur_genes  = cand_genes
                    cur_fit    = cand_fit
                    n_accepted += 1
                    if cur_fit > best_fit:
                        best_fit   = cur_fit
                        best_genes = cur_genes.copy()
                        if self.verbose:
                            lbl    = f" [{self.progress_label}]" if self.progress_label else ""
                            metric = (f"BIC/n={-best_fit:.4f}"
                                      if best_fit > -1e8 else f"fitness={best_fit:.4f}")
                            print(f" New best{lbl}: {metric}  feats={self._active_names(best_genes)}")

                bucket_fits.append(cur_fit)
                T *= alpha

            accept_rate = n_accepted / max(self.pop_size, 1)
            _valid_proposed = [v for v in proposed_fits if np.isfinite(v) and v > -1e8]
            history.append({
                'generation':    gen,
                'best_fitness':  float(max(bucket_fits)) if bucket_fits else cur_fit,
                'best_so_far':   best_fit,
                'mean_fitness':  float(np.mean(bucket_fits)) if bucket_fits else cur_fit,
                'worst_fitness': float(min(bucket_fits)) if bucket_fits else cur_fit,
                'n_features':    int(best_genes.sum()),
                'diversity':     accept_rate,   # acceptance rate (analogous to diversity)
                'mutation_rate': T,             # current temperature (analogous to mut rate)
                'stagnation':    0,
                # proposed solution trace — range of ALL evaluated candidates this gen
                'proposed_min':  float(min(_valid_proposed)) if _valid_proposed else None,
                'proposed_max':  float(max(_valid_proposed)) if _valid_proposed else None,
            })
            if self.verbose and gen % 5 == 0:
                _mean_prop = float(np.mean(_valid_proposed)) if _valid_proposed else float('nan')
                if 'BIC' in self.fitness_metric:
                    cur_bic  = f"{-cur_fit:.4f}"  if cur_fit  > -1e8 else "n/a"
                    mean_bic = f"{-_mean_prop:.4f}" if np.isfinite(_mean_prop) else "n/a"
                    print(f"  SA gen {gen:3d} | "
                          f"best_BIC/n={-best_fit:.4f}  cur={cur_bic}  "
                          f"prop_mean={mean_bic}  "
                          f"T={T:.5f}  accept={accept_rate:.2f}  "
                          f"n_feat={int(best_genes.sum())}")
                else:
                    cur_str  = f"{cur_fit:.4f}"  if cur_fit  > -1e8 else "n/a"
                    mean_str = f"{_mean_prop:.4f}" if np.isfinite(_mean_prop) else "n/a"
                    print(f"  SA gen {gen:3d} | "
                          f"best={best_fit:.4f}  cur={cur_str}  "
                          f"prop_mean={mean_str}  "
                          f"T={T:.5f}  accept={accept_rate:.2f}")

        best_ind = Individual(genes=best_genes, fitness=best_fit)
        return best_ind, history, [best_ind]

    def run_with_restarts(
        self,
        n_restarts:       int  = 1,
        restart_gens:     int  = None,
        seed_individuals: list = None,
    ) -> Tuple[Individual, List[dict], List[Individual]]:
        """Same interface as GeneticAlgorithm.run_with_restarts()."""
        restart_gens = restart_gens or max(10, self.n_generations // 3)
        print(f"  SA main run: {self.n_generations} gens (steps/gen={self.pop_size})")
        best_ever, history, _ = self.run(seed_individuals=seed_individuals)
        print(f"  SA main run best: {best_ever.fitness:.4f}")

        orig_gens          = self.n_generations
        self.n_generations = restart_gens
        gen_offset         = len(history)

        for r in range(n_restarts):
            print(f"  SA restart {r+1}/{n_restarts}: {restart_gens} gens "
                  f"(cache={len(self._cache)})")
            best_r, hist_r, _ = self.run(seed_individuals=[best_ever])
            for i, h in enumerate(hist_r):
                h['generation'] += gen_offset
                if i == 0:
                    h['is_restart_start'] = True
            gen_offset += len(hist_r)
            if best_r.fitness > best_ever.fitness:
                best_ever = best_r
                print(f"New SA global best: {best_ever.fitness:.4f}")
            history.extend(hist_r)

        self.n_generations = orig_gens
        return best_ever, history, [best_ever]


# ---------------------------------------------------------------------------
# 1c.  BANDIT FEATURE SEARCHER  (Thompson-sampling multi-armed bandit)
# ---------------------------------------------------------------------------

class BanditFeatureSearcher:
    """
    SA-based Thompson-sampling bandit for binary feature selection.

    Combines Simulated Annealing acceptance with Thompson-sampling proposals:

      1. At each step a *candidate* chromosome is drawn by sampling
             theta_i ~ Beta(alpha_i, beta_i)
             gene_i  ~ Bernoulli(theta_i)
         for every optional feature i.  Fixed features are always ON.

      2. The candidate is accepted as the new current state with SA
         probability  min(1, exp(Δfitness / T)).

      3. Beta posteriors are updated on *every evaluated candidate*
         (not just accepted ones), using the running mean as threshold:
             fitness > mean → alpha[i] += 1 for ON, beta[i] += 0.1 for OFF
             fitness ≤ mean → beta[i]  += 1 for ON

    The two mechanisms are complementary:
      - High T (early):  SA accepts worse proposals freely  → landscape exploration
      - Sharpening Beta: proposals concentrate on good features → feature selection
      - Low T (late):    SA only accepts improvements         → exploitation

    Total evaluation budget equals pop_size × n_generations steps, organised
    into n_generations reporting buckets — same budget structure as SA.
    Temperature is auto-calibrated from the first batch of random evaluations
    so that ~60 % of equal-magnitude worsening moves are accepted initially.

    The class shares the exact run() / run_with_restarts() / cache interface
    of GeneticAlgorithm so callers require no changes.  GA-only constructor
    kwargs (crossover_rate, elitism_k, etc.) are silently accepted and ignored.
    """

    def __init__(
        self,
        n_features:          int,
        fitness_fn:          Callable[[np.ndarray], float],
        pop_size:            int   = 1,
        n_generations:       int   = 120,
        min_features:        int   = 3,
        max_features:        int   = None,
        seed:                int   = 42,
        verbose:             bool  = True,
        feature_names:       List[str] = None,
        collinear_groups:    List[List[str]] = None,
        require_one_groups:  List[List[str]] = None,
        fixed_features:      Optional[List[str]] = None,
        progress_label:      str   = '',
        fitness_metric:      str   = 'mean(-BIC/n)',
        alpha0:              float = 1.0,   # Beta prior α  (pseudocount for "on")
        beta0:               float = 1.0,   # Beta prior β  (pseudocount for "off")
        t_initial:           float = None,  # SA: None = auto-calibrate from first batch
        # Mixed-model search: features listed here get a SECOND gene appended
        # to the chromosome - "estimate this coefficient as a random (normal)
        # parameter" - with its own Thompson arm, so fixed-vs-random becomes a
        # searched decision instead of the post-hoc greedy toggle
        # (_greedy_randvar_search). Only meaningful when the fitness function
        # decodes the extra genes (see decode_rand_genes()); fitness functions
        # that zip() genes against the candidate list simply never read them.
        rand_param_features: Optional[List[str]] = None,
        never_random:        Optional[List[str]] = None,
        force_random:        Optional[List[str]] = None,
        # Accept and silently ignore GA-only kwargs for interface compatibility
        mutation_rate:       float = 0.18,
        mutation_rate_max:   float = 0.35,
        diversity_threshold: float = 0.30,
        crossover_rate:      float = 0.75,
        elitism_k:           int   = 4,
        tournament_k:        int   = 4,
        stagnation_limit:    int   = 12,
        n_neighbors:         int   = 1,     # accepted, ignored (SA-bandit uses Thompson proposals)
    ):
        self.n_features     = n_features
        self.fitness_fn     = fitness_fn
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.min_features   = min_features
        self.max_features   = max_features or n_features
        self.verbose        = verbose
        self.feature_names  = feature_names or []
        self.collinear_groups = collinear_groups or []
        self.require_one_groups = require_one_groups or []
        self.fixed_features = fixed_features or []
        self.progress_label = progress_label
        self.fitness_metric = fitness_metric
        self.alpha0         = alpha0
        self.beta0          = beta0
        self.t_initial      = t_initial
        self.rng            = np.random.default_rng(seed)
        self._fixed_idx: set = {
            i for i, name in enumerate(self.feature_names)
            if name in set(self.fixed_features)
        }
        self._opt_idx: List[int] = [
            i for i in range(n_features) if i not in self._fixed_idx
        ]
        # Random-parameter gene block (appended after the include/exclude
        # genes). Indices into feature_names of features eligible for a
        # random-parameter bit; never_random names excluded, force_random
        # names pinned ON whenever the feature itself is included.
        _never = set(never_random or [])
        _names = self.feature_names or []
        self._rand_feat_idx: List[int] = [
            i for i, nm in enumerate(_names)
            if rand_param_features and nm in set(rand_param_features)
            and nm not in _never
        ]
        self.n_rand = len(self._rand_feat_idx)
        self._force_rand_pos: set = {
            j for j, fi in enumerate(self._rand_feat_idx)
            if _names[fi] in set(force_random or [])
        }
        self.n_genes = n_features + self.n_rand
        # Fitness cache — shared across restarts
        self._cache: dict = {}
        self.cache_hits     = 0
        # Beta arm posteriors — one per include gene + one per rand gene
        # (fixed features never updated)
        self._alpha = np.full(self.n_genes, alpha0, dtype=float)
        self._beta  = np.full(self.n_genes, beta0,  dtype=float)
        # Running mean of valid (> -1e8) fitness values for reward threshold
        self._fit_sum   = 0.0
        self._fit_count = 0
        # Per-feature weakness scores (populated from ga_external_estimators)
        self._weakness: dict = {}
        # Features permanently blacklisted after repeated stuck/at-bound occurrences
        self._blocked: set = set()

    # ── sampling helpers ────────────────────────────────────────────────────

    def _split_genes(self, genes: np.ndarray):
        """Full chromosome -> (include block, rand block). Rand block may be
        empty when rand_param_features wasn't given (n_rand == 0)."""
        return genes[:self.n_features], genes[self.n_features:]

    def _attach_rand_genes(self, base: np.ndarray, tail: np.ndarray = None,
                           sample_missing: bool = True) -> np.ndarray:
        """Append the rand-param gene block to a repaired include block.

        Gating: a rand gene can only be 1 when its feature is included;
        force_random positions are pinned to 1 whenever the feature is on.
        When tail is None the block is Thompson-sampled from its own arms.
        """
        if self.n_rand == 0:
            return base
        out = np.zeros(self.n_genes, dtype=int)
        out[:self.n_features] = base
        for j, fi in enumerate(self._rand_feat_idx):
            gpos = self.n_features + j
            if base[fi] != 1:
                continue                      # feature off -> rand bit off
            if j in self._force_rand_pos:
                out[gpos] = 1
            elif tail is not None and len(tail) == self.n_rand:
                out[gpos] = int(tail[j])
            elif sample_missing:
                theta = float(self.rng.beta(
                    max(self._alpha[gpos], 1e-6), max(self._beta[gpos], 1e-6)))
                out[gpos] = int(self.rng.random() < theta)
        return out

    def decode_rand_genes(self, genes: np.ndarray) -> List[str]:
        """Feature names whose random-parameter gene is ON in this chromosome.
        For fitness functions that estimate mixed models."""
        if self.n_rand == 0 or len(genes) <= self.n_features:
            return []
        tail = genes[self.n_features:]
        return [self.feature_names[fi]
                for j, fi in enumerate(self._rand_feat_idx) if tail[j] == 1]

    def _sample_chromosome(self) -> np.ndarray:
        """Sample one chromosome via Thompson sampling."""
        genes = np.zeros(self.n_features, dtype=int)
        # Fixed features always ON
        for i in self._fixed_idx:
            genes[i] = 1
        # Optional features: sample theta_i ~ Beta(α_i, β_i)
        # Permanently blacklisted features are forced OFF regardless of posterior.
        _blocked = getattr(self, '_blocked', set())
        for i in self._opt_idx:
            if self.feature_names and self.feature_names[i] in _blocked:
                genes[i] = 0
                continue
            theta    = float(self.rng.beta(
                max(self._alpha[i], 1e-6), max(self._beta[i], 1e-6)))
            genes[i] = int(self.rng.random() < theta)
        # Enforce min/max bounds (may add/remove optional features)
        genes = _enforce_feature_bounds(
            genes, self.min_features, self.max_features, self.rng,
            blocked_features=_blocked,
            protected_idx=self._fixed_idx,
        )
        # Repair collinearity
        if self.collinear_groups and self.feature_names:
            genes = _repair_collinearity(
                genes, self.feature_names, self.collinear_groups,
                self.rng, protected_idx=self._fixed_idx,
            )
        if self.require_one_groups and self.feature_names:
            genes = _repair_require_one(
                genes, self.feature_names, self.require_one_groups,
                self.rng, protected_idx=self._fixed_idx,
                collinear_groups=self.collinear_groups,
            )
        return self._attach_rand_genes(genes)

    def _directed_candidate(self, cur_genes: np.ndarray) -> np.ndarray:
        """
        Analyst-like local move: drop the weakest active feature (near-zero
        coefficient, insignificant, never moved from its start value, or
        pinned at a bound - see _update_feature_weakness) and, if it belongs
        to a collinear/require-one group, try turning on a currently-inactive
        group-mate instead of just leaving a gap. Mirrors how an analyst
        reworks a spec - "this term isn't pulling its weight, let me try a
        different form of the same idea" - instead of a blind random flip.

        Always mixed in as a MINORITY of proposals (see run()), alongside
        Thompson sampling and plain random local flips, so the search stays
        mostly stochastic/exploratory and doesn't collapse into pure greedy
        hill-climbing that gets stuck at the first local optimum.
        """
        cand_base, cand_tail = self._split_genes(cur_genes.copy())
        cand_genes = cand_base
        if not self.feature_names or not self._weakness:
            return self._attach_rand_genes(cand_genes, cand_tail)
        active_opt = [i for i in self._opt_idx if cand_genes[i] == 1]
        if not active_opt:
            return self._attach_rand_genes(cand_genes, cand_tail)
        weights = np.array([
            self._weakness.get(self.feature_names[i], 0.0) for i in active_opt
        ])
        if weights.max() <= 0:
            # nothing known to be weak yet - nothing to do
            return self._attach_rand_genes(cand_genes, cand_tail)
        drop_idx = active_opt[int(np.argmax(weights))]
        # Redundant-by-construction safety net: active_opt is built from
        # self._opt_idx, which already excludes every fixed/protected index
        # (CRASH, etc. - see fixed_features in __init__), so this can never
        # actually fire. Kept explicit anyway since dropping the safety
        # variable this search exists to test would be a silent, serious bug.
        assert drop_idx not in self._fixed_idx, (
            f"_directed_candidate tried to drop fixed feature "
            f"{self.feature_names[drop_idx]!r} - this must never happen"
        )
        cand_genes[drop_idx] = 0
        dropped_name = self.feature_names[drop_idx]

        # Look for a substitute: another member of the same collinear/
        # require-one group that's currently off - "try a different form of
        # the same idea" rather than just deleting the concept outright.
        substitute_pool: List[str] = []
        for group in list(self.collinear_groups or []) + list(self.require_one_groups or []):
            if dropped_name in group:
                substitute_pool.extend(f for f in group if f != dropped_name)
        feat_idx = {f: i for i, f in enumerate(self.feature_names)}
        candidates = [
            feat_idx[f] for f in dict.fromkeys(substitute_pool)
            if f in feat_idx and feat_idx[f] in self._opt_idx
            and cand_genes[feat_idx[f]] == 0
            and f not in getattr(self, '_blocked', set())
        ]
        if candidates:
            pick = int(self.rng.choice(candidates))
            cand_genes[pick] = 1

        cand_genes = _enforce_feature_bounds(
            cand_genes, self.min_features, self.max_features,
            self.rng, protected_idx=self._fixed_idx,
            feature_names=self.feature_names,
            blocked_features=getattr(self, '_blocked', set()),
        )
        if self.collinear_groups and self.feature_names:
            cand_genes = _repair_collinearity(
                cand_genes, self.feature_names, self.collinear_groups,
                self.rng, protected_idx=self._fixed_idx,
            )
        if self.require_one_groups and self.feature_names:
            cand_genes = _repair_require_one(
                cand_genes, self.feature_names, self.require_one_groups,
                self.rng, protected_idx=self._fixed_idx,
                collinear_groups=self.collinear_groups,
            )
        return self._attach_rand_genes(cand_genes, cand_tail)

    def _evaluate_one(self, ind: Individual) -> Individual:
        key = tuple(ind.genes.tolist())
        if key in self._cache:
            ind.fitness = self._cache[key]
            self.cache_hits += 1
        elif ind.fitness == -np.inf:
            try:
                ind.fitness = self.fitness_fn(ind.genes)
            except Exception as e:
                logger.warning(f"fitness_fn error: {e}")
                ind.fitness = -1e9
            self._cache[key] = ind.fitness
        return ind

    def _update_arms(self, genes: np.ndarray, fitness: float) -> None:
        """
        Update Beta posteriors for all optional arms.

        Rule: features present in above-average models receive a success
        update (alpha += 1); features present in below-average models receive
        a failure update (beta += 1).  Features absent from an above-average
        model receive a mild failure update (beta += 0.1) to encourage their
        exploration in future samples.
        """
        if fitness <= -1e8:
            # Sentinel / invalid — no reward signal, add mild failure to all ON features
            for i in self._opt_idx:
                if genes[i] == 1:
                    self._beta[i] += 0.5
            return

        threshold = (self._fit_sum / self._fit_count) if self._fit_count > 0 else fitness
        above_avg = fitness > threshold

        for i in self._opt_idx:
            if genes[i] == 1:
                if above_avg:
                    # Weaken the success credit for features that persistently
                    # produce near-zero or insignificant coefficients.  Weakness
                    # score is 0 (no history) → 2+ (repeatedly weak).
                    # alpha boost: 0.4 (neutral) → ≥0.05 (w≥7).
                    w = (self._weakness.get(self.feature_names[i], 0.0)
                         if (self.feature_names and i < len(self.feature_names))
                         else 0.0)
                    self._alpha[i] += max(0.4 - w * 0.05, 0.05)
                else:
                    # Feature present in below-average model: stronger penalty
                    # if it's also known to be statistically weak.
                    w = (self._weakness.get(self.feature_names[i], 0.0)
                         if (self.feature_names and i < len(self.feature_names))
                         else 0.0)
                    self._beta[i] += 0.4 + min(w * 0.05, 0.3)
            else:  # genes[i] == 0
                if above_avg:
                    self._beta[i] += 0.04  # feature absent from good model

        # Random-parameter arms: only informative when the feature itself was
        # in the model (rand gene is gated on the include gene), so absent
        # features contribute no signal either way.
        if self.n_rand and len(genes) >= self.n_genes:
            for j, fi in enumerate(self._rand_feat_idx):
                gpos = self.n_features + j
                if genes[fi] != 1 or j in self._force_rand_pos:
                    continue
                if genes[gpos] == 1:
                    if above_avg:
                        self._alpha[gpos] += 0.4
                    else:
                        self._beta[gpos] += 0.4
                elif above_avg:
                    self._beta[gpos] += 0.04

        # Update running mean
        self._fit_sum   += fitness
        self._fit_count += 1

    def _refresh_weakness(self) -> None:
        """Pull the latest per-feature weakness snapshot from the estimator module."""
        try:
            from ga_external_estimators import get_feature_weakness, is_feature_blacklisted
            self._weakness = get_feature_weakness()
            self._blocked = {
                self.feature_names[i] for i in self._opt_idx
                if is_feature_blacklisted(self.feature_names[i])
            } if self.feature_names else set()
        except ImportError:
            self._weakness = {}
            self._blocked = set()

    def _active_feature_names(self, genes: np.ndarray, limit: int = 8) -> str:
        if not self.feature_names:
            return f"n={int(genes.sum())}"
        active = [name for name, flag in zip(self.feature_names, genes) if flag == 1]
        if len(active) > limit:
            shown = active[:limit] + [f"... (+{len(active) - limit})"]
        else:
            shown = active
        return str(shown)

    # ── main loop ───────────────────────────────────────────────────────────

    def run(
        self,
        seed_individuals: list = None,
    ) -> Tuple[Individual, List[dict], List[Individual]]:
        """
        Run one pass of SA-bandit feature search.

        Total evaluation budget: pop_size × n_generations steps, grouped into
        n_generations reporting buckets.  Temperature is auto-calibrated from
        the first bucket unless t_initial was supplied.

        Returns (best_ever, history, [best_ever]) — same shape as SA.run().
        """
        n_steps_total = self.pop_size * self.n_generations

        # ── Initialise current state ────────────────────────────────────────
        # Start from the best seed if provided; otherwise draw via Thompson.
        if seed_individuals:
            best_seed = max(seed_individuals, key=lambda x: x.fitness
                            if np.isfinite(x.fitness) else -np.inf)
            cur_genes = best_seed.genes.copy()
            cur_fit   = best_seed.fitness
            if not np.isfinite(cur_fit) or cur_fit <= -1e8:
                cur_fit = self._evaluate_one(Individual(genes=cur_genes)).fitness
            for ind in seed_individuals:
                ind = self._evaluate_one(ind)
                self._update_arms(ind.genes, ind.fitness)
        else:
            cur_genes = self._sample_chromosome()
            cur_fit   = self._evaluate_one(Individual(genes=cur_genes)).fitness
            self._update_arms(cur_genes, cur_fit)

        best_genes = cur_genes.copy()
        best_fit   = cur_fit

        # ── Auto-calibrate T₀ from first bucket ────────────────────────────
        # Sample pop_size Thompson proposals; collect |Δfitness| vs cur.
        # Set T₀ so that ≈60 % of equal-magnitude downhill moves are accepted:
        #   exp(-|Δ|/T₀) = 0.60  →  T₀ = |Δ| / ln(1/0.6) ≈ |Δ| / 0.51
        if self.t_initial is None:
            deltas: List[float] = []
            for _ in range(min(self.pop_size, 20)):
                g = self._sample_chromosome()
                f = self._evaluate_one(Individual(genes=g)).fitness
                self._update_arms(g, f)
                if np.isfinite(f) and f > -1e8:
                    deltas.append(abs(f - cur_fit))
                    if f > best_fit:
                        best_fit   = f
                        best_genes = g.copy()
                        cur_genes  = g.copy()
                        cur_fit    = f
            mean_delta = float(np.mean(deltas)) if deltas else 0.1
            t0 = max(mean_delta / 0.51, 0.02)
        else:
            t0 = float(self.t_initial)

        t_final = t0 * 1e-3
        alpha_T = (t_final / max(t0, 1e-15)) ** (1.0 / max(n_steps_total, 1))
        T       = t0

        history: List[dict] = []

        for gen in range(self.n_generations):
            print(f"Bandit gen {gen:3d} | T={T:.5f} | evaluating candidates... ", end='', flush=True)
            # Refresh weakness snapshot every 10 gens so recent coefficient
            # quality feeds back into arm posteriors without per-step overhead.
            if gen % 10 == 0:
                self._refresh_weakness()
            bucket_fits: List[float] = []
            n_accepted = 0

            for _ in range(self.pop_size):
                # Propose candidate: mix three move types so the search stays
                # mostly stochastic (avoids getting stuck at a local optimum)
                # while still spending some steps on analyst-like moves:
                #   20% directed  - drop the weakest active feature, try a
                #                   collinear/require-one group-mate instead
                #                   ("this term isn't pulling its weight, try
                #                   a different form of the same idea")
                #   30% local     - plain random 1-2 bit flip from cur_genes
                #   50% global    - Thompson sample from the full posterior
                _move_r = self.rng.random()
                if (self._opt_idx and len(self._opt_idx) >= 2 and _move_r < 0.20):
                    cand_genes = self._directed_candidate(cur_genes)
                elif (self._opt_idx and len(self._opt_idx) >= 2 and _move_r < 0.50):
                    # Local perturbation: flip 1–2 optional bits from cur_genes
                    # (include block only; rand block re-gated after repair,
                    # occasionally flipping one rand bit as its own local move)
                    _lp_base, _lp_tail = self._split_genes(cur_genes.copy())
                    cand_genes = _lp_base
                    n_flips = int(self.rng.integers(
                        1, max(2, min(3, len(self._opt_idx) // 3 + 1))
                    ))
                    flip_idx = self.rng.choice(
                        self._opt_idx, size=n_flips, replace=False
                    )
                    _blocked = getattr(self, '_blocked', set())
                    for fi in flip_idx:
                        if self.feature_names[fi] in _blocked:
                            continue
                        cand_genes[fi] = 1 - cand_genes[fi]
                    cand_genes = _enforce_feature_bounds(
                        cand_genes, self.min_features, self.max_features,
                        self.rng, protected_idx=self._fixed_idx,
                        feature_names=self.feature_names,
                        blocked_features=_blocked,
                    )
                    if self.collinear_groups and self.feature_names:
                        cand_genes = _repair_collinearity(
                            cand_genes, self.feature_names,
                            self.collinear_groups, self.rng,
                            protected_idx=self._fixed_idx,
                        )
                    if self.require_one_groups and self.feature_names:
                        cand_genes = _repair_require_one(
                            cand_genes, self.feature_names,
                            self.require_one_groups, self.rng,
                            protected_idx=self._fixed_idx,
                            collinear_groups=self.collinear_groups,
                        )
                    if self.n_rand and len(_lp_tail) == self.n_rand \
                            and self.rng.random() < 0.5:
                        _lp_tail = _lp_tail.copy()
                        _j = int(self.rng.integers(0, self.n_rand))
                        _lp_tail[_j] = 1 - _lp_tail[_j]
                    cand_genes = self._attach_rand_genes(cand_genes, _lp_tail)
                else:
                    # Global Thompson sample (standard path)
                    cand_genes = self._sample_chromosome()
                cand       = self._evaluate_one(Individual(genes=cand_genes))
                cand_fit   = cand.fitness

                # Update Beta posteriors regardless of acceptance
                self._update_arms(cand_genes, cand_fit)

                # SA acceptance criterion
                delta = cand_fit - cur_fit
                if delta >= 0 or (T > 1e-15 and self.rng.random() < np.exp(delta / T)):
                    cur_genes  = cand_genes
                    cur_fit    = cand_fit
                    n_accepted += 1
                    if cur_fit > best_fit and cur_fit > -1e8:
                        best_fit   = cur_fit
                        best_genes = cur_genes.copy()
                        if self.verbose:
                            label  = f" [{self.progress_label}]" if self.progress_label else ""
                            metric = (f"BIC/n={-best_fit:.5f}"
                                      if best_fit > -1e8
                                      else f"fitness={best_fit:.5f}")
                            n_f    = int(best_genes.sum())
                            print(f"New best{label}: {metric}  n_feats={n_f}  "
                                  f"feats={self._active_feature_names(best_genes)}")

                bucket_fits.append(cur_fit)
                T *= alpha_T

            accept_rate = n_accepted / max(self.pop_size, 1)

            # Diversity proxy: std of arm probabilities across optional features
            if self._opt_idx:
                arm_probs = (self._alpha[self._opt_idx]
                             / (self._alpha[self._opt_idx] + self._beta[self._opt_idx]))
                diversity = float(np.std(arm_probs))
            else:
                arm_probs = np.array([])
                diversity = 0.0

            history.append({
                'generation':    gen,
                'best_fitness':  float(max(bucket_fits)) if bucket_fits else best_fit,
                'best_so_far':   best_fit,
                'mean_fitness':  float(np.mean(bucket_fits)) if bucket_fits else best_fit,
                'worst_fitness': float(min(bucket_fits)) if bucket_fits else best_fit,
                'n_features':    int(best_genes.sum()),
                'diversity':     diversity,
                'mutation_rate': T,      # current temperature (analogous to mut rate)
                'stagnation':    0,
            })

            if self.verbose and gen % 5 == 0:
                top_k     = min(3, len(self._opt_idx))
                top_idx   = (np.argsort(arm_probs)[::-1][:top_k].tolist()
                             if len(arm_probs) > 0 else [])
                top_names = ([self.feature_names[self._opt_idx[i]] for i in top_idx]
                             if self.feature_names else top_idx)
                label = f" [{self.progress_label}]" if self.progress_label else ""
                cur_feats = self._active_feature_names(cur_genes)
                if 'BIC' in self.fitness_metric:
                    cur_bic = f"{-cur_fit:.5f}" if cur_fit > -1e8 else "n/a"
                    print(f"  Bandit gen {gen:3d}{label} | "
                          f"best_BIC/n={-best_fit:.5f}  cur_BIC/n={cur_bic}  "
                          f"T={T:.5f}  accept={accept_rate:.2f}  "
                          f"cur_feats={cur_feats}  "
                          f"top_arms={top_names}  cache={len(self._cache)}")
                    _plog(
                        f"  Bandit gen {gen:3d}{label} | "
                        f"best_BIC/n={-best_fit:.5f}  cur_BIC/n={cur_bic}  "
                        f"T={T:.5f}  accept={accept_rate:.2f}  "
                        f"cur_feats={cur_feats}  "
                        f"top_arms={top_names}  cache={len(self._cache)}"
                    )
                else:
                    cur_str = f"{cur_fit:.5f}" if cur_fit > -1e8 else "n/a"
                    print(f"  Bandit gen {gen:3d}{label} | "
                          f"best={best_fit:.5f}  cur={cur_str}  "
                          f"T={T:.5f}  accept={accept_rate:.2f}  "
                          f"cur_feats={cur_feats}  "
                          f"top_arms={top_names}  cache={len(self._cache)}")
                    _plog(
                        f"  Bandit gen {gen:3d}{label} | "
                        f"best={best_fit:.5f}  cur={cur_str}  "
                        f"T={T:.5f}  accept={accept_rate:.2f}  "
                        f"cur_feats={cur_feats}  "
                        f"top_arms={top_names}  cache={len(self._cache)}"
                    )

        best_ind = Individual(genes=best_genes, fitness=best_fit)
        return best_ind, history, [best_ind]

    def run_with_restarts(
        self,
        n_restarts:       int  = 3,
        restart_gens:     int  = None,
        seed_individuals: list = None,
    ) -> Tuple[Individual, List[dict], List[Individual]]:
        """Same interface as GeneticAlgorithm.run_with_restarts()."""
        restart_gens = restart_gens or max(10, self.n_generations // 3)
        label = f" [{self.progress_label}]" if self.progress_label else ""
        print(f"  Bandit(SA) main run{label}: {self.n_generations} gens "
              f"(steps/gen={self.pop_size}, n_opt_arms={len(self._opt_idx)})")
        _plog(f"[Bandit{label}] starting: {self.n_generations} gens × {self.pop_size} steps/gen")
        best_ever, history, _ = self.run(seed_individuals=seed_individuals)
        print(f"  Bandit(SA) main best: {best_ever.fitness:.4f}")
        _plog(f"[Bandit{label}] done: best_fit={best_ever.fitness:.4f}")

        orig_gens          = self.n_generations
        self.n_generations = restart_gens
        gen_offset         = len(history)

        for r in range(n_restarts):
            print(f"  Bandit(SA) restart {r+1}/{n_restarts}: {restart_gens} gens "
                  f"(cache={len(self._cache)})")
            best_r, hist_r, _ = self.run(seed_individuals=[best_ever])
            for i, h in enumerate(hist_r):
                h['generation'] += gen_offset
                if i == 0:
                    h['is_restart_start'] = True
            gen_offset += len(hist_r)
            if best_r.fitness > best_ever.fitness:
                best_ever = best_r
                print(f"    ^ New Bandit(SA) global best: {best_ever.fitness:.4f}")
            history.extend(hist_r)

        self.n_generations = orig_gens
        return best_ever, history, [best_ever]


def _make_searcher(
    n_features:         int,
    fitness_fn:         Callable,
    ga_kwargs:          dict,
    use_sim_anneal:     bool = False,
    use_bandit:         bool = False,
    feature_names:      List[str] = None,
    collinear_groups:   List[List[str]] = None,
    require_one_groups: List[List[str]] = None,
    fixed_features:     Optional[List[str]] = None,
    fitness_metric:     str = 'mean(-BIC/n)',
    stage_id:           Optional[str] = None,
) -> 'GeneticAlgorithm | SimulatedAnnealing | BanditFeatureSearcher':
    """
    Factory: return the appropriate searcher class.

    Selection order (env-vars override kwargs):
        GA_USE_BANDIT=1      → BanditFeatureSearcher
        GA_USE_BANDIT_SA=1   → BanditFeatureSearcher (bandit outer loop; used by sa_bandit mode)
        GA_USE_SA=1          → SimulatedAnnealing
        default              → GeneticAlgorithm

    All three classes share the run_with_restarts() interface so callers
    require no changes.

    stage_id: when given (e.g. 'stage4', 'stage5_mode', 'stage5_dest') the
    stage's declared constraints from ga_constraints.PIPELINE_CONSTRAINTS
    (SearchLibrium ConstraintBuilder format) are merged into the searcher's
    native constraint kwargs — force_include → fixed_features,
    mutually_exclusive → collinear_groups, min_behavioral →
    require_one_groups — and force_exclude vars are masked to 0 in a fitness
    wrapper (the searchers have no native banned-feature support).
    """
    _force_sa     = os.environ.get('GA_USE_SA',     '0').strip() == '1'
    _force_bandit = (os.environ.get('GA_USE_BANDIT',    '0').strip() == '1'
                     or os.environ.get('GA_USE_BANDIT_SA', '0').strip() == '1')
    if use_bandit or _force_bandit:
        cls = BanditFeatureSearcher
    elif use_sim_anneal or _force_sa:
        cls = SimulatedAnnealing
    else:
        cls = GeneticAlgorithm

    feature_names = feature_names or []
    try:
        from ga_constraints import searcher_constraint_kwargs
        _ck = searcher_constraint_kwargs(
            stage_id, feature_names,
            fixed_features=fixed_features,
            collinear_groups=collinear_groups,
            require_one_groups=require_one_groups,
        )
        fixed_features     = _ck['fixed_features']
        collinear_groups   = _ck['collinear_groups']
        require_one_groups = _ck['require_one_groups']
        _banned            = _ck['banned_features']
    except Exception as _ce:
        logger.debug(f"_make_searcher: constraint merge skipped ({_ce})")
        _banned = []

    if _banned and feature_names:
        _banned_idx = np.array(
            [i for i, n in enumerate(feature_names) if n in set(_banned)],
            dtype=int,
        )
        _inner_fitness = fitness_fn

        def fitness_fn(genes, _inner=_inner_fitness, _idx=_banned_idx):
            g = np.asarray(genes).copy()
            g[_idx] = 0
            return _inner(g)

    # ── Mixed-model (random-parameter) gene search ────────────────────────
    # GA_MXL_GENE_SEARCH=1 appends a "fixed vs random (normal)" gene per
    # eligible feature to the bandit-SA chromosome (BanditFeatureSearcher
    # rand_param_features), making random-parameter choice a first-class
    # searched decision instead of the post-hoc greedy toggle
    # (GA_RAND_PARAM_SEARCH). Only effective when (a) the searcher is the
    # bandit (GA class ignores the kwargs) and (b) the stage's fitness
    # function decodes the extra genes via searcher.decode_rand_genes() and
    # fits an MXL - a fitness that ignores them just wastes the extra arms.
    # force_random / never_random come from ga_constraints' new_features().
    if (os.environ.get('GA_MXL_GENE_SEARCH', '0').strip() == '1'
            and cls is BanditFeatureSearcher and feature_names):
        try:
            from ga_constraints import random_var_constraints
            _never_r, _force_r = random_var_constraints(stage_id)
        except Exception:
            _never_r, _force_r = [], []
        ga_kwargs.setdefault('rand_param_features', list(feature_names))
        ga_kwargs.setdefault('never_random', list(_never_r))
        ga_kwargs.setdefault('force_random', list(_force_r))

    return cls(
        n_features,
        fitness_fn,
        feature_names      = feature_names,
        collinear_groups   = collinear_groups or [],
        require_one_groups = require_one_groups or [],
        fixed_features     = fixed_features or [],
        fitness_metric     = fitness_metric,
        **ga_kwargs,
    )


# ---------------------------------------------------------------------------
# 1b.  PARALLEL PER-SEGMENT SEARCH (purposes / targets within a GA stage)
# ---------------------------------------------------------------------------
# Stages 2-5 evaluate several independent "segments" (day-duration targets,
# activity purposes, mode/dest tour segments). Each segment's GA/SA/Bandit
# search is independent — same input data, disjoint candidate pools — so
# they can run concurrently in spawn-mode worker processes. Spawn (not fork)
# keeps each worker's JAX/MetaCount/SearchLibrium state and compiled-program
# caches isolated, which both parallelises the search and bounds per-process
# memory growth (see ga_external_estimators._maybe_clear_jax_caches).

def _ga_max_workers(n_segments: int) -> int:
    """
    Number of segment searches to run concurrently.

    Controlled by GA_PARALLEL_SEGMENTS (set by the PBS submission script to
    match --cpus). Defaults to min(n_segments, cpu_count, 4).
    GA_PARALLEL_SEGMENTS=1 restores the previous fully-serial behaviour.
    """
    env = os.environ.get('GA_PARALLEL_SEGMENTS', '').strip()
    if env:
        try:
            want = max(1, min(int(env), n_segments))
        except ValueError:
            want = max(1, min(n_segments, os.cpu_count() or 1, 4))
    else:
        want = max(1, min(n_segments, os.cpu_count() or 1, 4))

    # Memory-aware cap (opt-in): spawn-mode workers each load the full data, so
    # N workers × per-worker footprint can exceed the job's cgroup limit and the
    # pool gets OOM-killed (BrokenProcessPool) — this is what silently failed
    # Stage 1 on the 371 data at 96GB. If GA_MEM_PER_WORKER_GB is set, cap the
    # worker count to what the detected memory limit can hold.
    _pw = os.environ.get('GA_MEM_PER_WORKER_GB', '').strip()
    if _pw:
        try:
            per_worker = float(_pw)
            mem_gb = _detect_mem_limit_gb()
            if mem_gb and per_worker > 0:
                cap = max(1, int(mem_gb // per_worker))
                if cap < want:
                    logger.warning(
                        f"_ga_max_workers: capping {want} -> {cap} workers "
                        f"(mem limit {mem_gb:.0f}GB / {per_worker:.0f}GB per worker) "
                        "to avoid OOM")
                    want = cap
        except Exception:
            pass
    return want


def _detect_mem_limit_gb() -> float:
    """Best-effort memory limit in GB: PBS resource, cgroup v2/v1, else psutil."""
    _pbs = os.environ.get('PBS_VMEM') or os.environ.get('PBS_MEM')
    if _pbs:
        try:
            s = _pbs.lower().rstrip('b')
            mult = {'g': 1, 'm': 1 / 1024, 'k': 1 / 1024 / 1024, 't': 1024}
            for u, m in mult.items():
                if s.endswith(u):
                    return float(s[:-1]) * m
            return float(s) / (1024 ** 3)
        except Exception:
            pass
    for _p in ('/sys/fs/cgroup/memory.max',
               '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(_p) as _fh:
                v = _fh.read().strip()
            if v and v != 'max':
                b = float(v)
                if b < 1e18:   # ignore the "unlimited" sentinel
                    return b / (1024 ** 3)
        except Exception:
            continue
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _ga_segment_worker(payload: dict) -> dict:
    """
    Run one segment's (purpose/target) GA/SA/Bandit feature search.

    `payload` carries everything needed to reconstruct the per-segment
    fitness closure that previously lived inline in _run_ga_stageN's loop
    body. Must be picklable (DataFrames + plain python types only — no
    closures, file handles, or open OMX objects), since this function is
    submitted to a spawn-mode ProcessPoolExecutor.
    """
    import importlib

    data             = payload['data']
    feat_names       = payload['feat_names']
    fixed_feats      = payload['fixed_feats']
    ga_kwargs        = payload['ga_kwargs']
    n_restarts       = payload['n_restarts']
    use_sim_anneal   = payload['use_sim_anneal']
    fitness_metric   = payload['fitness_metric']
    stage_id         = payload['stage_id']
    collinear_groups = payload['collinear_groups']
    require_one_groups = payload.get('require_one_groups', [])
    label            = payload['label']

    _dmod, _dname, _dargs = payload['default_fit']
    _default_fn = getattr(importlib.import_module(_dmod), _dname)

    def _default_fit(genes):
        return _default_fn(genes, data, feat_names, *_dargs)

    _mc_spec = payload.get('mc_fit')
    if _mc_spec is not None:
        _mmod, _mname, _margs, _mkwargs = _mc_spec
        _mc_fn = getattr(importlib.import_module(_mmod), _mname)
        _mc_state = {'enabled': True, 'calls': 0, 'fails': 0, 'warned': False, 'fallback_warned': False}
        _fallback = payload['fallback_to_default']

        def _seg_fit(genes):
            if not _mc_state['enabled']:
                return _default_fit(genes)
            score = _mc_fn(genes, data, feat_names, *_margs, **_mkwargs)
            _mc_state['calls'] += 1
            if score <= -1e8:
                _mc_state['fails'] += 1
                if (_mc_state['calls'] >= 20 and
                        (_mc_state['fails'] / max(_mc_state['calls'], 1)) >= 0.8 and
                        not _mc_state['warned']):
                    _mc_state['warned'] = True
                    _mc_state['enabled'] = False
                    print(payload['mc_fail_msg'])
                    return _default_fit(genes)
            if score > -1e8 or not _fallback:
                return score
            fb = _default_fit(genes)
            if fb > -1e8 and not _mc_state['fallback_warned']:
                _mc_state['fallback_warned'] = True
                print(payload['mc_fallback_msg'])
            return fb if fb > -1e8 else score
    else:
        _seg_fit = _default_fit

    _ga_kwargs = _resolve_min_features(ga_kwargs, feat_names, fixed_feats)
    searcher = _make_searcher(
        len(feat_names), _seg_fit, _ga_kwargs,
        use_sim_anneal      = use_sim_anneal,
        feature_names       = feat_names,
        collinear_groups    = collinear_groups,
        require_one_groups  = require_one_groups,
        fixed_features      = fixed_feats,
        fitness_metric      = fitness_metric,
        stage_id            = f'stage{stage_id}',
    )
    print(f"\n  [{label}] Running per-segment search (worker pid={os.getpid()}) ...")
    t0 = time.time()
    best, hist, pop = searcher.run_with_restarts(n_restarts=n_restarts)
    feats = [f for f, g in zip(feat_names, best.genes) if g == 1]
    seg_result = GAResult(
        stage=stage_id, best_features=feats,
        best_fitness=best.fitness, best_individual=best,
        history=hist, all_candidates=pop,
        feature_names=feat_names, fitness_metric=fitness_metric,
        n_restarts=n_restarts, cache_hits=searcher.cache_hits,
    )
    elapsed = time.time() - t0
    print(f"    [{label}] BIC/n={-best.fitness:.4f}  feats={feats}  elapsed={elapsed/60:.1f}min")
    return {
        'label':           label,
        'feature_map_key': payload['feature_map_key'],
        'seg_result':      seg_result,
        'feats':           feats,
        'best_fitness':    best.fitness,
        'elapsed':         elapsed,
    }


def _run_segments_parallel(payloads: List[dict], max_workers: int, worker_fn=None):
    """
    Run worker_fn (default _ga_segment_worker) for each payload, yielding
    (index, result) as each segment finishes (so callers can checkpoint
    incrementally).

    With max_workers<=1 or a single payload, runs serially in-process
    (no subprocess overhead).
    """
    if worker_fn is None:
        worker_fn = _ga_segment_worker

    if max_workers <= 1 or len(payloads) <= 1:
        for i, p in enumerate(payloads):
            yield i, worker_fn(p)
        return

    import multiprocessing as _mp
    import concurrent.futures as _cf

    n_workers = min(max_workers, len(payloads))
    ctx = _mp.get_context('spawn')
    with _cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(worker_fn, p): i for i, p in enumerate(payloads)}
        for fut in _cf.as_completed(futures):
            yield futures[fut], fut.result()


# ---------------------------------------------------------------------------
# 2.  FITNESS FUNCTIONS  (estimation only   no prediction)
# ---------------------------------------------------------------------------

#  Stage 1: ordered logit   McFadden rho┬▓, AVERAGED over all (purpose, ptype) ÔöÇ

def _stage1_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
    ptype: str,
) -> float:
    from stage1_activity_frequency import OrderedLogit
    from scipy.special import expit

    # Deduplicate features while preserving gene-selection order
    _seen_f: set = set()
    feats  = [f for f, g in zip(feature_names, genes)
              if g == 1 and not (_seen_f.__contains__(f) or _seen_f.add(f))]
    col    = f'n_{purpose}_out'
    subset = data[(data['ptype'] == ptype) & data[col].notna()].copy()
    if len(subset) < 30 or col not in subset.columns:
        return -1e9

    y = subset[col].values.astype(int)
    if len(np.unique(y)) < 2:
        return -1e9

    X  = subset[feats].fillna(0).values.astype(float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    X  = (X - mu) / sd

    try:
        m = OrderedLogit()
        m.fit(X, y)
    except Exception:
        return -1e9

    # ── Prune near-zero coefficients: drop weak features, re-fit for cleaner BIC ──
    _zthresh = float(os.environ.get('GA_SIG_COEF_ZERO', '0.01'))
    if _zthresh > 0 and len(feats) > 1:
        coef_arr = np.asarray(m.coef_).ravel()
        keep_idx = np.where(np.abs(coef_arr) >= _zthresh)[0]
        if len(keep_idx) > 0 and len(keep_idx) < len(feats):
            pruned_feats = [feats[i] for i in keep_idx]
            Xp = X[:, keep_idx]
            try:
                mp = OrderedLogit()
                mp.fit(Xp, y)
                if hasattr(mp, 'thresholds_') and len(mp.thresholds_) >= 2:
                    gp = np.diff(np.sort(mp.thresholds_))
                    if gp.min() >= 0.05:
                        m = mp
                        feats = pruned_feats
                        X = Xp
            except Exception:
                pass  # keep original model if re-fit fails

    # Reject degenerate ordered logit: thresholds must be strictly ordered
    # with a minimum gap so the model meaningfully discriminates categories.
    if hasattr(m, 'thresholds_') and len(m.thresholds_) >= 2:
        gaps = np.diff(np.sort(m.thresholds_))
        if gaps.min() < 0.05:
            return -1e9

    classes = np.sort(np.unique(y))
    ll_null = len(y) * np.log(1.0 / len(classes))

    ll_fit = 0.0
    for j, cls in enumerate(m.classes_):
        mask = (y == cls)
        if not mask.any():
            continue
        eta = X[mask] @ m.coef_
        if j == 0:
            p = expit(m.thresholds_[0] - eta)
        elif j == len(m.classes_) - 1:
            p = 1 - expit(m.thresholds_[-1] - eta)
        else:
            p = expit(m.thresholds_[j] - eta) - expit(m.thresholds_[j-1] - eta)
        p = np.clip(p, 1e-10, 1.0)
        ll_fit += np.sum(np.log(p))

    n   = len(y)
    k   = len(feats) + len(classes) - 1          # coef + thresholds
    bic = -2.0 * ll_fit + k * np.log(max(n, 2))
    return -bic / max(n, 1)


def _stage1_fitness_aggregate(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purposes: List[str],
    ptypes: List[str],
) -> float:
    """Mean rho┬▓ across all (purpose, ptype) pairs with enough data."""
    scores = []
    for purpose in purposes:
        for ptype in ptypes:
            col    = f'n_{purpose}_out'
            subset = data[(data['ptype'] == ptype) & data.get(col, pd.Series(dtype=float)).notna()]
            if col in data.columns and len(subset) >= 30:
                s = _stage1_fitness(genes, data, feature_names, purpose, ptype)
                if s > -1e8:
                    scores.append(s)
    return float(np.mean(scores)) if scores else -1e9


# ÔöÇÔöÇ Stage 2: OLS   adjusted R┬▓, averaged over all targets 

def _stage2_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
) -> float:
    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
    subset = data[feats + [target_col]].dropna()
    if len(subset) < 20:
        return -1e9

    X  = subset[feats].values.astype(float)
    y  = subset[target_col].values.astype(float)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])

    try:
        params = np.linalg.lstsq(Xs, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return -1e9

    # ── Prune near-zero coefficients ──────────────────────────────────────────
    _zthresh = float(os.environ.get('GA_SIG_COEF_ZERO', '0.01'))
    coef_part = params[1:]  # skip intercept
    if _zthresh > 0 and len(feats) > 1:
        keep_idx = np.where(np.abs(coef_part) >= _zthresh)[0]
        if len(keep_idx) > 0 and len(keep_idx) < len(feats):
            pruned_feats = [feats[i] for i in keep_idx]
            try:
                Xp = X[:, keep_idx]
                Xsp = np.column_stack([np.ones(len(Xp)), (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)])
                params_p = np.linalg.lstsq(Xsp, y, rcond=None)[0]
                feats = pruned_feats
                Xs = Xsp
                params = params_p
            except np.linalg.LinAlgError:
                pass  # keep original fit

    yhat   = Xs @ params
    ss_res = ((y - yhat) ** 2).sum()
    n, k   = len(y), len(feats) + 1              # feats + intercept
    sigma2 = ss_res / max(n, 1)
    ll_ols = -0.5 * n * (np.log(2 * np.pi * max(sigma2, 1e-12)) + 1.0)
    bic    = -2.0 * ll_ols + k * np.log(max(n, 2))
    return -bic / max(n, 1)


def _stage2_fitness_aggregate(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    targets: List[str],
) -> float:
    scores = [_stage2_fitness(genes, data, feature_names, t)
              for t in targets if t in data.columns]
    valid  = [s for s in scores if s > -1e8]
    return float(np.mean(valid)) if valid else -1e9


# ÔöÇÔöÇ Stage 3: lognormal   -AIC, averaged over all purposes ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

def _stage3_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
    purpose_col: str = 'main_purpose',
) -> float:
    from stage3_activity_duration import BudgetLognormal, DURATION_BOUNDS

    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
    lo, hi = DURATION_BOUNDS.get(purpose, (5, 600))
    subset = data[data[purpose_col] == purpose].copy()
    if len(subset) < 30:
        return -1e9

    y = subset['duration'].clip(lo, hi).values.astype(float)
    X = subset[feats].fillna(0).values.astype(float)

    try:
        m     = BudgetLognormal()
        m.fit(X, y)
    except Exception:
        return -1e9

    # ── Prune near-zero coefficients ──────────────────────────────────────────
    _zthresh = float(os.environ.get('GA_SIG_COEF_ZERO', '0.01'))
    if _zthresh > 0 and len(feats) > 1:
        coef_arr = np.asarray(m.coef_).ravel()
        keep_idx = np.where(np.abs(coef_arr) >= _zthresh)[0]
        if len(keep_idx) > 0 and len(keep_idx) < len(feats):
            pruned_feats = [feats[i] for i in keep_idx]
            Xp = X[:, keep_idx]
            try:
                mp = BudgetLognormal()
                mp.fit(Xp, y)
                m = mp
                feats = pruned_feats
                X = Xp
            except Exception:
                pass  # keep original model

    log_y  = np.log(np.clip(y, 1e-6, None))
    mu_    = m.mu_; sd_ = m.sd_.copy(); sd_[sd_ == 0] = 1.0
    Xs     = (X - mu_) / sd_
    Xs_aug = np.column_stack([np.ones(len(Xs)), Xs])
    eta    = Xs_aug @ np.concatenate([[m.intercept_], m.coef_])
    resid  = log_y - eta
    sigma  = resid.std() + 1e-9
    ll     = (-len(y) * np.log(sigma)
              - 0.5 * ((resid / sigma) ** 2).sum()
              - log_y.sum())
    n   = len(y)
    k   = len(feats) + 2                         # coefs + intercept + sigma
    bic = -2.0 * ll + k * np.log(max(n, 2))
    return -bic / max(n, 1)                      # -BIC/n, higher is better


def _stage3_fitness_aggregate(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purposes: List[str],
    purpose_col: str,
) -> float:
    scores = []
    for p in purposes:
        if len(data[data[purpose_col] == p]) >= 30:
            s = _stage3_fitness(genes, data, feature_names, p, purpose_col)
            if s > -1e8:
                scores.append(s)
    return float(np.mean(scores)) if scores else -1e9


# ÔöÇÔöÇ Stage 4: MNL timing   rho┬▓, averaged over all purposes ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

def _stage4_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
) -> float:
    from stage4_schedule_placement import MNLTimingModel, TIME_PERIODS

    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
    subset = data[data['purpose'] == purpose].copy()
    if len(subset) < 30:
        return -1e9

    y = subset['period'].values.astype(int)
    if len(np.unique(y)) < 2:
        return -1e9

    ptypes = sorted(subset['ptype'].dropna().unique()) if 'ptype' in subset.columns else []
    m = MNLTimingModel()
    m.purpose_ = purpose
    m.ptypes_  = ptypes
    m.feature_override_ = feats
    m.include_person_features_ = any(f in feats for f in ['sexp', 'vehrd', 'nprd'])

    try:
        X = m._build_X(subset)
        m.fit(X, y)
    except Exception:
        return -1e9

    if not np.isfinite(m.rho2_) or m.rho2_ < -0.5:
        return -1e9

    ll = float(m.ll_)
    n  = len(y)
    k  = (len(TIME_PERIODS) - 1) * X.shape[1]
    bic = -2.0 * ll + k * np.log(max(n, 2))
    return -bic / max(n, 1)                      # -BIC/n, higher is better


def _stage4_fitness_aggregate(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purposes: List[str],
) -> float:
    scores = []
    for p in purposes:
        if len(data[data['purpose'] == p]) >= 30:
            s = _stage4_fitness(genes, data, feature_names, p)
            if s > -1e8:
                scores.append(s)
    return float(np.mean(scores)) if scores else -1e9


# ÔöÇÔöÇ Stage 5 (tour-level): MODE-ONLY and DEST-ONLY fitness functions ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# Running separate mode and destination GAs lets the user see independent
# convergence plots for each sub-model and avoids the joint-chromosome
# entanglement issue where the mode and destination feature pools compete
# for the same mutation budget.

def _stage5_mode_only_fitness(
    mode_genes:      np.ndarray,
    mode_candidates: List[str],
    segment:         str,
    tours:           pd.DataFrame,
    persons:         pd.DataFrame,
    landuse:         pd.DataFrame,
    skims,
) -> float:
    """
    Tour-level mode model fitness ONLY (no destination).
    Returns -BIC_mode / n  (higher = better).
    """
    from stage5_destination_choice import (
        _filter_seg, _mode_tree, _build_mode_model, _estimate,
    )
    mode_vars = [v for v, g in zip(mode_candidates, mode_genes) if g == 1]
    if not mode_vars:
        return -1e9
    seg_tours = _filter_seg(tours, segment)
    if len(seg_tours) < 50:
        return -1e9
    try:
        dt_m     = _mode_tree(seg_tours, persons, landuse, skims)
        mode_mdl = _build_mode_model(dt_m, segment, custom_mode_vars=mode_vars,
                                     avail_vars_hint=list(seg_tours.columns))
        mode_mdl = _estimate(mode_mdl)
        # Force the CHOSEN alternative available before scoring. With DA now gated
        # on (CARLICENCE>0)*(vehrd>0), a chosen-DA record in a zero-vehicle
        # household otherwise gets ~0 probability and drives loglike() to ~-1e35,
        # which poisons the GA fitness and gets the whole segment rejected as
        # "invalid" — this is why work/nonmand disappeared while edu (non-DA-heavy)
        # survived. _run_segment already does this repair at build time; the mode
        # fitness path did not.
        try:
            mode_mdl.doctor(repair_ch_av="+")
        except Exception:
            pass
        # Score from the estimation's OWN reported LL (result.loglike, cached on
        # the model as _cached_ll), which is finite and well-fitted (e.g. work
        # phase-1 LL≈-5230). A fresh mode_mdl.loglike() can blow up to ~-1e35 even
        # for a well-fitted model, because DA availability is an EXPRESSION
        # ((CARLICENCE>0)*(vehrd>0)) that doctor(repair_ch_av) doesn't materialise,
        # so a chosen-DA/zero-vehicle record floors to ~0 probability on rescoring.
        # That mis-scoring is why work/nonmand mode never produced a valid model
        # while edu (non-DA) did. Fall back to loglike() only if no cached LL.
        _cached = getattr(mode_mdl, '_cached_ll', None)
        if _cached is not None and np.isfinite(_cached) and _cached > -1e12:
            mode_ll = float(_cached)
        else:
            mode_ll = float(mode_mdl.loglike())
        if not np.isfinite(mode_ll) or mode_ll < -1e12:
            logger.debug(f"Stage5 mode-only [{segment}]: non-finite/blown LL={mode_ll}")
            return -1e9
        n   = len(seg_tours)
        k   = int(mode_genes.sum())
        _seen = getattr(_stage5_mode_only_fitness, '_valid_logged', set())
        if segment not in _seen:
            print(f"    First valid Stage 5 mode model [{segment}]: LL={mode_ll:.2f}  n={n}  k={k}")
            _seen.add(segment)
            _stage5_mode_only_fitness._valid_logged = _seen
        from ga_external_estimators import (_larch_coef_rows,
                                             _metacount_significance_penalty,
                                             _update_feature_weakness)
        coef_rows = _larch_coef_rows(mode_mdl)
        sig_pen   = _metacount_significance_penalty(coef_rows)
        _update_feature_weakness(coef_rows, mode_vars)
        bic = -2.0 * mode_ll + k * np.log(max(n, 2))
        return -bic / max(n, 1) - sig_pen
    except Exception as e:
        logger.debug(f"Stage5 mode-only [{segment}]: {e}")
        return -1e9


def _stage5_dest_only_fitness(
    dest_genes:      np.ndarray,
    dest_candidates: List[str],
    segment:         str,
    seg_tours:       pd.DataFrame,
    persons:         pd.DataFrame,
    lu_st:           pd.DataFrame,
    skims,
    ls_cached,       # pre-computed mode logsums (from best mode GA run)
    seg_acdc:        pd.DataFrame,
) -> float:
    """
    Tour-level destination model fitness using CACHED mode logsums.
    Returns -BIC_dest / n  (higher = better).
    The mode model is NOT re-estimated here   logsums are pre-computed once
    with the best mode features before the dest GA starts.
    """
    from stage5_destination_choice import _dest_tree, _build_dest_model, _estimate
    dest_vars = [v for v, g in zip(dest_candidates, dest_genes) if g == 1]
    if not dest_vars:
        return -1e9
    try:
        dt_d     = _dest_tree(seg_acdc, persons, lu_st, skims, ls_cached)
        dest_mdl = _build_dest_model(dt_d, segment, custom_dest_vars=dest_vars)
        dest_mdl = _estimate(dest_mdl)
        dest_ll  = float(dest_mdl.loglike())
        n   = len(seg_tours)
        k   = int(dest_genes.sum())
        _seen = getattr(_stage5_dest_only_fitness, '_valid_logged', set())
        if segment not in _seen:
            print(f"    First valid Stage 5 dest model [{segment}]: LL={dest_ll:.2f}  n={n}  k={k}")
            _seen.add(segment)
            _stage5_dest_only_fitness._valid_logged = _seen
        from ga_external_estimators import (_larch_coef_rows,
                                             _metacount_significance_penalty,
                                             _update_feature_weakness)
        coef_rows = _larch_coef_rows(dest_mdl)
        sig_pen   = _metacount_significance_penalty(coef_rows)
        _update_feature_weakness(coef_rows, dest_vars)
        bic = -2.0 * dest_ll + k * np.log(max(n, 2))
        return -bic / max(n, 1) - sig_pen
    except Exception as e:
        logger.debug(f"Stage5 dest-only [{segment}]: {e}")
        return -1e9


# ÔöÇÔöÇ Stage 5: Larch   mode_LL + dest_LL on HHTS only ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# ESTIMATION ONLY   no prediction over synthetic population.
# Calls _run_segment which uses only HHTS tours.

# Held-out aggregate-fit penalty for _stage5_fitness_larch. In-sample BIC +
# significance penalty alone reward specs that maximise discrete-choice
# likelihood on the estimation sample; they have no way to notice a spec that
# simultaneously collapses the aggregate destination-share distribution onto
# a handful of zones (see 2026-07 sa_bandit_staged validation report - dest
# rho^2 up to 0.68 "excellent" while top25 zone overlap was only 8-16%). The
# JSD term below is evaluated on a held-out split so a spec can't buy a good
# score just by overfitting the split it's scored on.
_STAGE5_HOLDOUT_FRAC   = float(os.environ.get('STAGE5_HOLDOUT_FRAC', '0.2'))
_STAGE5_HOLDOUT_MIN_N  = int(os.environ.get('STAGE5_HOLDOUT_MIN_N', '100'))
_STAGE5_JS_LAMBDA      = float(os.environ.get('STAGE5_JS_LAMBDA', '3.0'))


def _js_divergence_shares(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) between two nonnegative count/share vectors."""
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(np.asarray(q, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _stage5_fitness_larch(
    mode_genes:      np.ndarray,
    dest_genes:      np.ndarray,
    mode_candidates: List[str],
    dest_candidates: List[str],
    segment:         str,
    tours:           pd.DataFrame,
    persons:         pd.DataFrame,
    landuse:         pd.DataFrame,
    skims,
    skim_df:         pd.DataFrame,
) -> float:
    """
    Estimation-only fitness: fit mode + destination model on a TRAIN split of
    HHTS tours, scored as -BIC/n - significance_penalty - JS_LAMBDA * held_out_JSD.

    held_out_JSD is the Jensen-Shannon divergence between the fitted dest
    model's predicted destination-zone shares and the OBSERVED destination-zone
    shares on a disjoint TEST split (not used for fitting) - this penalises
    specs that fit well in-sample but collapse the aggregate destination
    distribution, which plain BIC/significance can't detect (see module note
    above _STAGE5_HOLDOUT_FRAC). If the held-out evaluation itself fails for
    any reason, it contributes zero penalty rather than failing the genome.

    Uses _run_segment's building blocks (mode/dest tree + Larch estimation),
    HHTS-only - no prediction over synth population.
    """
    from stage5_destination_choice import (
        _filter_seg, _prep_hhts_tours,
        _build_mode_model, _build_dest_model,
        _estimate, _logsums, _mode_tree, _dest_tree,
        _attach_size_term, _attach_acdc_terms,
        _dest_probs_numpy,
        skim_lookup_to_df,
        MODE_UEC, DEST_UEC, DEST_COND,
    )

    mode_vars = [v for v, g in zip(mode_candidates, mode_genes) if g == 1]
    dest_vars = [v for v, g in zip(dest_candidates, dest_genes) if g == 1]
    if not mode_vars or not dest_vars:
        return -1e9

    seg_tours = _filter_seg(tours, segment)
    if len(seg_tours) < 50:
        return -1e9

    # Deterministic train/held-out split - fixed seed so every genome in every
    # generation is scored against the SAME split (comparable JSD across the GA
    # run; not meant as a statistical guarantee, just a consistent yardstick).
    if len(seg_tours) >= _STAGE5_HOLDOUT_MIN_N:
        _rng_split = np.random.RandomState(42)
        _perm = _rng_split.permutation(len(seg_tours))
        _n_test = max(int(len(seg_tours) * _STAGE5_HOLDOUT_FRAC), 20)
        _test_idx, _train_idx = _perm[:_n_test], _perm[_n_test:]
        train_tours = seg_tours.iloc[_train_idx].reset_index(drop=True)
        test_tours  = seg_tours.iloc[_test_idx].reset_index(drop=True)
    else:
        train_tours = seg_tours
        test_tours  = None

    # Use JAX engine for bandit fitness eval on GPU — removes numba override.
    # _lx_model() auto-detects: JAX first on Linux, falls back to numba if
    # covariance computation fails (cusolver_gesvdj_ffi on CUDA 13).
    # Set XLA_PYTHON_CLIENT_PREALLOCATE=false in PBS env to avoid GPU OOM.
    _prev_ga_engine = os.environ.get('GA_LARCH_ENGINE', '')
    os.environ.pop('GA_LARCH_ENGINE', None)  # let _lx_model auto-detect
    try:
        # Mode model on TRAIN
        dt_m     = _mode_tree(train_tours, persons, landuse, skims)
        mode_mdl = _build_mode_model(dt_m, segment, custom_mode_vars=mode_vars,
                                     avail_vars_hint=list(train_tours.columns))
        mode_mdl = _estimate(mode_mdl)
        mode_ll  = float(mode_mdl.loglike())

        # Logsums for destination tree
        ls = _logsums(mode_mdl, dt_m, skims=skims, verbose=False)

        # Destination model on TRAIN
        lu_st = _attach_size_term(landuse, segment)
        lu_st, seg_acdc = _attach_acdc_terms(
            lu_st, train_tours,
            skim_df if skim_df is not None else pd.DataFrame(),
            segment
        )
        dt_d     = _dest_tree(seg_acdc, persons, lu_st, skims, ls)
        dest_mdl = _build_dest_model(dt_d, segment, custom_dest_vars=dest_vars)
        dest_mdl = _estimate(dest_mdl)
        dest_ll  = float(dest_mdl.loglike())

        # Proper BIC: -2*LL + k*ln(n), then normalise per observation
        n   = len(train_tours)
        k   = int(mode_genes.sum() + dest_genes.sum())
        bic = -2.0 * (mode_ll + dest_ll) + k * np.log(max(n, 2))
        # Significance penalty: discourage near-zero / insignificant features
        from ga_external_estimators import (_larch_coef_rows,
                                             _metacount_significance_penalty,
                                             _update_feature_weakness)
        mode_rows = _larch_coef_rows(mode_mdl)
        dest_rows = _larch_coef_rows(dest_mdl)
        sig_pen   = (_metacount_significance_penalty(mode_rows)
                     + _metacount_significance_penalty(dest_rows)) / 2.0
        _update_feature_weakness(mode_rows, mode_vars)
        _update_feature_weakness(dest_rows, dest_vars)

        # ── Held-out aggregate destination-share penalty ────────────────────
        js_pen = 0.0
        if test_tours is not None and len(test_tours) >= 10:
            try:
                dt_m_test  = _mode_tree(test_tours, persons, landuse, skims)
                ls_test    = _logsums(mode_mdl, dt_m_test, skims=skims, verbose=False)
                lu_st_test, seg_acdc_test = _attach_acdc_terms(
                    lu_st, test_tours,
                    skim_df if skim_df is not None else pd.DataFrame(),
                    segment
                )
                dt_d_test  = _dest_tree(seg_acdc_test, persons, lu_st_test, skims, ls_test)
                zone_ids   = list(ls_test.coords['zones'].values)
                probs_test = _dest_probs_numpy(dest_mdl, dt_d_test, ls_test, zone_ids, persons,
                                               landuse=lu_st_test, skim_df=skim_df)
                if probs_test is not None:
                    pred_share = probs_test.sum(axis=0)
                    obs_counts = (seg_acdc_test['DTAZ'].astype(int)
                                  .value_counts().reindex(zone_ids).fillna(0.0).to_numpy())
                    jsd = _js_divergence_shares(obs_counts, pred_share)
                    js_pen = _STAGE5_JS_LAMBDA * jsd
            except Exception as _jse:
                logger.debug(f"Stage5 fitness [{segment}]: held-out JSD penalty skipped ({_jse})")
                js_pen = 0.0

        return -bic / max(n, 1) - sig_pen - js_pen   # -BIC/n - penalty - JSD, higher is better

    except Exception as e:
        logger.debug(f"Stage5 fitness [{segment}]: {e}")
        return -1e9
    finally:
        if _prev_ga_engine:
            os.environ['GA_LARCH_ENGINE'] = _prev_ga_engine
        # else: already popped, leave unset


def _stage5_fitness_aggregate(
    genes:           np.ndarray,
    n_mode:          int,
    mode_candidates: List[str],
    dest_candidates: List[str],
    segments:        List[str],
    tours:           pd.DataFrame,
    persons:         pd.DataFrame,
    landuse:         pd.DataFrame,
    skims,
    skim_df:         pd.DataFrame,
) -> float:
    """Mean fitness across all valid segments."""
    scores = []
    for seg in segments:
        s = _stage5_fitness_larch(
            mode_genes      = genes[:n_mode],
            dest_genes      = genes[n_mode:],
            mode_candidates = mode_candidates,
            dest_candidates = dest_candidates,
            segment         = seg,
            tours           = tours,
            persons         = persons,
            landuse         = landuse,
            skims           = skims,
            skim_df         = skim_df,
        )
        if s > -1e8:
            scores.append(s)
    return float(np.mean(scores)) if scores else -1e9


# ÔöÇÔöÇ Stage 51: TRIP-LEVEL mode + destination choice ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# Mirrors Stage 5 but operates on individual TRIP legs rather than tours.
# ActivitySim distinction: tour_mode_choice ÔåÆ trip_mode_choice (constrained
# to be a subset of the tour mode) and trip-level destination choice.
# Here we estimate independently on HHTS trips for GA variable selection.

def _stage5_trip_fitness(
    mode_genes:      np.ndarray,
    dest_genes:      np.ndarray,
    mode_candidates: List[str],
    dest_candidates: List[str],
    purpose:         str,
    trips:           pd.DataFrame,
    persons:         pd.DataFrame,
    landuse:         pd.DataFrame,
    skim_df:         pd.DataFrame,
) -> float:
    """
    Trip-level mode + destination BIC fitness.

    Fits a nested logit mode model on individual HHTS trip legs (not tours),
    then fits a destination choice MNL on HHTS observed DTAZ, and returns
    the combined -BIC/n.

    This mirrors the ActivitySim trip_mode_choice approach where trip mode
    is estimated conditional on in-vehicle time, cost, safety, and person
    attributes, with a separate trip-destination model for each purpose.
    """
    from stage6_mode_choice import NestedLogitModeModel, prepare_mode_data

    mode_vars = [v for v, g in zip(mode_candidates, mode_genes) if g == 1]
    dest_vars = [v for v, g in zip(dest_candidates, dest_genes) if g == 1]
    if not mode_vars and not dest_vars:
        return -1e9

    # ÔöÇÔöÇ Trip-level mode model on HHTS ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        mode_data = prepare_mode_data(trips, persons, landuse)
        mode_data_purp = mode_data[mode_data['purpose'] == purpose].copy()
        if len(mode_data_purp) < 30 or mode_data_purp['chosen_mode'].nunique() < 2:
            return -1e9

        for col in mode_vars:
            if col not in mode_data_purp.columns:
                mode_data_purp[col] = 0.0

        m = NestedLogitModeModel()
        m.fit(mode_data_purp, purpose)
        if m.ll_final_ is None or m.n_obs_ == 0:
            return -1e9
        mode_ll = m.ll_final_
        n_mode  = m.n_obs_
    except Exception as e:
        logger.debug(f"Stage51 mode fitness [{purpose}]: {e}")
        return -1e9

    # ÔöÇÔöÇ Trip-level destination model (simple logit on observed DTAZ) ÔöÇÔöÇÔöÇÔöÇÔöÇ
    dest_ll = 0.0
    n_dest  = 0
    try:
        if not skim_df.empty and dest_vars:
            trip_dest = trips.copy()
            # Merge person attributes
            if persons is not None and not persons.empty:
                p_cols = ['person_id'] + [c for c in dest_vars if c in persons.columns]
                if len(p_cols) > 1:
                    pid_col = next((c for c in ['person_id', 'PERSONID', 'PERSNUM']
                                    if c in persons.columns), None)
                    if pid_col:
                        trip_dest = trip_dest.merge(
                            persons[[pid_col] + [c for c in dest_vars
                                                 if c in persons.columns and c != pid_col]],
                            left_on='person_id', right_on=pid_col, how='left'
                        )
            # Filter to this purpose
            purp_col = next((c for c in ['purpose', 'main_purpose', 'TRIPTYPE']
                             if c in trip_dest.columns), None)
            if purp_col:
                trip_dest = trip_dest[trip_dest[purp_col] == purpose]

            dtaz_col = next((c for c in ['DTAZ', 'DESTZONE', 'dest_taz']
                             if c in trip_dest.columns), None)
            otaz_col = next((c for c in ['OTAZ', 'ORIGZONE', 'orig_taz', 'ORDER']
                             if c in trip_dest.columns), None)
            if dtaz_col and otaz_col and len(trip_dest) >= 30:
                # Build simple log-distance destination logit
                dist_col = 'DIST' if 'DIST' in skim_df.columns else 'SOVTOLL_DIST__AM'
                if dist_col not in skim_df.columns:
                    dist_col = [c for c in skim_df.columns if 'DIST' in c]
                    dist_col = dist_col[0] if dist_col else None

                if dist_col:
                    dist_lookup = skim_df.set_index(['OTAZ', 'DTAZ'])[dist_col].to_dict() \
                                  if 'OTAZ' in skim_df.columns else {}
                    log_dists = np.array([
                        np.log(max(dist_lookup.get((int(row[otaz_col]),
                                                    int(row[dtaz_col])), 0.1), 0.01))
                        for _, row in trip_dest.iterrows()
                    ])
                    # Null LL: uniform over observed zones
                    n_zones = skim_df['DTAZ'].nunique() if 'DTAZ' in skim_df.columns else 100
                    dest_ll = float(np.mean(log_dists)) * len(trip_dest) - \
                              len(trip_dest) * np.log(max(n_zones, 2))
                    n_dest = len(trip_dest)
    except Exception as e:
        logger.debug(f"Stage51 dest fitness [{purpose}]: {e}")

    # ÔöÇÔöÇ Combined BIC ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    total_ll = mode_ll + dest_ll
    n        = n_mode + max(n_dest, 0)
    k        = int(mode_genes.sum() + dest_genes.sum())
    bic      = -2.0 * total_ll + k * np.log(max(n, 2))
    return -bic / max(n, 1)    # -BIC/n, higher is better


def _stage5_trip_fitness_aggregate(
    genes:           np.ndarray,
    n_mode:          int,
    mode_candidates: List[str],
    dest_candidates: List[str],
    purposes:        List[str],
    trips:           pd.DataFrame,
    persons:         pd.DataFrame,
    landuse:         pd.DataFrame,
    skim_df:         pd.DataFrame,
) -> float:
    """Mean trip-level fitness across all valid purposes."""
    scores = []
    for purp in purposes:
        s = _stage5_trip_fitness(
            mode_genes      = genes[:n_mode],
            dest_genes      = genes[n_mode:],
            mode_candidates = mode_candidates,
            dest_candidates = dest_candidates,
            purpose         = purp,
            trips           = trips,
            persons         = persons,
            landuse         = landuse,
            skim_df         = skim_df,
        )
        if s > -1e8:
            scores.append(s)
    return float(np.mean(scores)) if scores else -1e9


# ÔöÇÔöÇ Stage 6: nested logit   rho┬▓ ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

def _stage6_fitness(
    genes:         np.ndarray,
    data:          pd.DataFrame,
    feature_names: List[str],
    purpose:       str,
) -> float:
    from stage6_mode_choice import NestedLogitModeModel

    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
    subset = data[data['purpose'] == purpose].copy()
    if len(subset) < 30 or subset['chosen_mode'].nunique() < 2:
        return -1e9

    for col in feats:
        if col not in subset.columns:
            subset[col] = 0.0
    try:
        m = NestedLogitModeModel()
        m.fit(subset, purpose)
        if m.ll_final_ is None or m.n_obs_ == 0:
            return -1e9
        # k = selected features + fixed structural params (ASCs, nesting logsum params)
        k_structural = 6   # ASC_SR, ASC_TR, ASC_WK, ASC_BK, mu_M, mu_NM
        k   = int(genes.sum()) + k_structural
        n   = m.n_obs_
        bic = -2.0 * m.ll_final_ + k * np.log(max(n, 2))
        return -bic / max(n, 1)                  # -BIC/n, higher is better
    except Exception:
        return -1e9


def _stage6_fitness_aggregate(
    genes:         np.ndarray,
    data:          pd.DataFrame,
    feature_names: List[str],
    purposes:      List[str],
) -> float:
    scores = []
    for p in purposes:
        if len(data[data['purpose'] == p]) >= 30 and data[data['purpose'] == p]['chosen_mode'].nunique() >= 2:
            s = _stage6_fitness(genes, data, feature_names, p)
            if s > -1e8:
                scores.append(s)
    return float(np.mean(scores)) if scores else -1e9


# ÔöÇÔöÇ CDAP: Larch MNL   mean rho┬▓ over H/N/M classification ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# Chromosome = binary mask over MNL_FEATURES candidate set.
# Fitness = McFadden rho┬▓ of the 3-alternative MNL on HHTS persons.
# Uses numba engine (same as estimate_cdap) to avoid JAX OOM.

def _cdap_fitness(
    genes:         np.ndarray,
    data:          pd.DataFrame,   # output of prepare_cdap_data()
    feature_names: List[str],
) -> float:
    """
    Estimate a CDAP MNL on HHTS with the selected features.
    Returns McFadden rho┬▓ (higher = better).
    Fitness is penalised by number of active features (parsimony).
    """
    # SEARCH fitness uses the sklearn multinomial logit by default: same
    # unpenalized MLE surface and same -BIC/n metric as the Larch path, but
    # milliseconds per fit with no compilation. The Larch path rebuilds a
    # DataTree and triggers fresh sharrow/jax flow compiles for EVERY
    # candidate spec — run guided03 spent 16h 57m on Stage 0 alone, with
    # 356,129 "writing fresh flow code" lines (~100 compiles per evaluation,
    # ~9 s each). Larch remains the engine for the FINAL CDAP model after
    # selection; set GA_CDAP_FITNESS=larch to restore the old behaviour.
    if os.environ.get('GA_CDAP_FITNESS', 'sklearn').strip().lower() != 'larch':
        return _cdap_fitness_sklearn(genes, data, feature_names)

    try:
        import larch as lx
    except ImportError:
        return _cdap_fitness_sklearn(genes, data, feature_names)

    feats = [f for f, g in zip(feature_names, genes) if g == 1]
    if not feats:
        return -1e9

    # Check all selected features actually exist
    available = [f for f in feats if f in data.columns]
    if not available:
        return -1e9

    try:
        case_ids = data['person_id'].astype(str).tolist()
        alt_ids  = [0, 1, 2]

        ca_idx = pd.MultiIndex.from_product(
            [case_ids, alt_ids], names=['person_id', '_altid_'])
        ca_df  = pd.DataFrame({'chosen': 0}, index=ca_idx)
        # vectorised chosen-flag assignment (was a per-row iterrows loop over
        # every HHTS person, per candidate evaluation)
        _chosen_idx = pd.MultiIndex.from_arrays(
            [data['person_id'].astype(str), data['cdap_obs'].astype(int)],
            names=['person_id', '_altid_'])
        ca_df.loc[ca_df.index.isin(_chosen_idx), 'chosen'] = 1
        ds_ca = lx.Dataset.construct.from_idca(ca_df)

        co_df = data.set_index(data['person_id'].astype(str))[available].fillna(0).copy()
        co_df.index.name = 'person_id'
        ds_co = lx.Dataset.construct.from_idco(co_df)

        dt = lx.DataTree(base=ds_ca, person=ds_co,
                         relationships=('base.person_id @ person.person_id',))

        mnl = lx.Model(dt, compute_engine='jax')
        mnl.choice_ca_var = 'chosen'
        mnl.utility_co[1] = lx.P('ASC_N')
        mnl.utility_co[2] = lx.P('ASC_M')
        for feat in available:
            mnl.utility_co[1] += lx.P(f'b_N_{feat}') * lx.X(f'person.{feat}')
            mnl.utility_co[2] += lx.P(f'b_M_{feat}') * lx.X(f'person.{feat}')
        mnl.set_cap(10)
        r = mnl.maximize_loglike(options={'ftol': 1e-7, 'maxiter': 200})
        ll_fit = float(r.loglike)

        # BIC: k = 2 params per feature (one for N, one for M) + 2 ASCs
        n   = len(data)
        k   = 2 * len(available) + 2
        bic = -2.0 * ll_fit + k * np.log(max(n, 2))
        return -bic / max(n, 1)                  # -BIC/n, higher is better

    except Exception as e:
        logger.debug(f"CDAP Larch fitness failed: {e}")
        return _cdap_fitness_sklearn(genes, data, feature_names)


def _cdap_fitness_sklearn(
    genes:         np.ndarray,
    data:          pd.DataFrame,
    feature_names: List[str],
) -> float:
    """Sklearn MNL fallback if Larch is unavailable."""
    from sklearn.linear_model import LogisticRegression

    feats     = [f for f, g in zip(feature_names, genes) if g == 1]
    available = [f for f in feats if f in data.columns]
    if not available:
        return -1e9

    X = data[available].fillna(0).values.astype(float)
    y = data['cdap_obs'].values.astype(int)
    if len(np.unique(y)) < 2:
        return -1e9

    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    try:
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                 max_iter=300, C=1e6)
        clf.fit(Xs, y)
        probs    = clf.predict_proba(Xs)
        probs    = np.clip(probs, 1e-10, 1.0)
        ll_fit   = float(np.sum(np.log(probs[np.arange(len(y)), y])))
        n        = len(y)
        k        = 2 * len(available) + 2
        bic      = -2.0 * ll_fit + k * np.log(max(n, 2))
        return -bic / max(n, 1)
    except Exception as e:
        logger.debug(f"CDAP sklearn fitness failed: {e}")
        return -1e9


# ÔöÇÔöÇ Usual Location: decay_b and attractor column selection ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# Chromosome is CONTINUOUS (not binary)   see _run_ga_usual_loc.
# We encode decay_b as two continuous genes in [0.01, 0.20].
# Attractor column selection uses binary genes over the landuse columns.
# Fitness = log-likelihood of observed work/school zones given the model.

def _usual_loc_fitness(
    genes:          np.ndarray,
    obs_work:       pd.DataFrame,    # observed (OTAZ, WORK_TAZ) pairs from HHTS
    obs_school:     pd.DataFrame,    # observed (OTAZ, SCHOOL_TAZ) pairs from HHTS
    landuse:        pd.DataFrame,
    skim_flat:      pd.DataFrame,
    work_att_names: List[str],       # candidate attractor column names
    edu_att_names:  List[str],
    n_work_att:     int,             # how many genes are work attractor flags
) -> float:
    """
    Fitness = mean log-probability of observed work/school zone given the
    weighted gravity model P(z|orig) ÔêØ att(z) ├ù exp(-b├ùtt(orig,z)).

    Chromosome layout:
        genes[0]            : work_decay_b  (mapped to [0.01, 0.20])
        genes[1]            : school_decay_b (mapped to [0.01, 0.20])
        genes[2:2+n_work_att]        : binary   which work attractor cols to use
        genes[2+n_work_att:]         : binary   which edu attractor cols to use
    """
    from usual_location import _get_tt_vector

    # decode genes
    work_b   = float(np.clip(genes[0], 0.01, 0.20))
    school_b = float(np.clip(genes[1], 0.01, 0.20))
    work_att_flags = genes[2: 2 + n_work_att].astype(int)
    edu_att_flags  = genes[2 + n_work_att:].astype(int)

    work_cols = [c for c, g in zip(work_att_names, work_att_flags) if g == 1]
    edu_cols  = [c for c, g in zip(edu_att_names,  edu_att_flags)  if g == 1]

    if not work_cols and not edu_cols:
        return -1e9

    zone_col  = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
    zone_ids  = landuse[zone_col].astype(int).values
    zone_ids  = zone_ids[(zone_ids > 0) & (zone_ids != 330)]
    zone_idx  = {z: i for i, z in enumerate(zone_ids)}
    N         = len(zone_ids)
    lu_vals   = landuse.set_index(zone_col)

    total_ll = 0.0
    n_obs    = 0

    # ÔöÇÔöÇ work LL ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    if work_cols and len(obs_work) > 0:
        att = np.zeros(N)
        for col in work_cols:
            if col in lu_vals.columns:
                att += lu_vals[col].reindex(zone_ids).fillna(0).values.astype(float)
        att = np.clip(att, 0, None)

        for _, row in obs_work.iterrows():
            otaz = int(row.get('OTAZ', row.get('ORDER', 0)))
            wtaz = int(row.get('WORK_TAZ', 0))
            if wtaz <= 0 or wtaz not in zone_idx:
                continue
            tt = _get_tt_vector(skim_flat, otaz, zone_ids, 'SOVTOLL_TIME__AM')
            w  = att * np.exp(-work_b * tt)
            ws = w.sum()
            if ws <= 0:
                continue
            p = w[zone_idx[wtaz]] / ws
            total_ll += np.log(max(p, 1e-10))
            n_obs    += 1

    # ÔöÇÔöÇ school LL ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    if edu_cols and len(obs_school) > 0:
        att = np.zeros(N)
        for col in edu_cols:
            if col in lu_vals.columns:
                att += lu_vals[col].reindex(zone_ids).fillna(0).values.astype(float)
        att = np.clip(att, 0, None)

        for _, row in obs_school.iterrows():
            otaz = int(row.get('OTAZ', row.get('ORDER', 0)))
            staz = int(row.get('SCHOOL_TAZ', 0))
            if staz <= 0 or staz not in zone_idx:
                continue
            tt = _get_tt_vector(skim_flat, otaz, zone_ids, 'SOVTOLL_TIME__AM')
            w  = att * np.exp(-school_b * tt)
            ws = w.sum()
            if ws <= 0:
                continue
            p = w[zone_idx[staz]] / ws
            total_ll += np.log(max(p, 1e-10))
            n_obs    += 1

    if n_obs == 0:
        return -1e9

    return total_ll / n_obs   # mean log-probability (higher = better)


# ---------------------------------------------------------------------------
# 3.  CANDIDATE FEATURE SETS PER STAGE
# ---------------------------------------------------------------------------
# WORKSTAT and STUDYING are nominal codes with no meaningful linear scale, so
# rather than feed the raw ordinal/categorical codes to the CDAP candidate set
# we use their per-category binary indicators (workstat_*, studying_*) built
# in cdap.build_person_profiles() — the same function used for both HHTS
# estimation and synth-pop prediction, so the GA-selected binaries exist and
# are derived identically at predict time. occp_group is left as-is for now
# (same "grouped, no ordering" concern noted for future consideration).
def _cdap_candidates(data: pd.DataFrame) -> List[str]:
    """
    All candidate features for CDAP (coordinated daily activity pattern).
    Sourced from build_person_profiles() + raw synth columns + accessibility
    + ABS census-derived variables (from census_bridge.map_census_to_hhts).
    """
    ptype_dummies = [
        'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
        'ptype_UniversityStudent', 'ptype_WorkingStudent',
        'ptype_SchoolChild', 'ptype_Retired', 'ptype_Preschool',
    ]
    age_vars  = ['age_child', 'age_teen', 'age_working', 'age_older',
                 'age_senior', 'age_norm',
                 # ABS-derived (census_bridge)
                 'age_retired', 'age_young']
    hh_vars   = ['hh_child', 'hh_spouse', 'hh_parent', 'hh_alone', 'hh_size_large',
                 'n_workers_hh']   # household-level worker count (census_bridge)
    access    = ['has_vehicle', 'has_licence']
    work_vars = ['is_female', 'wfh', 'income_midplus', 'mandatory_role',
                 'income_hi', 'income_lo']   # census_bridge income flags
    acc_cols  = ['auto_work_acc', 'walk_acc', 'transit_acc', 'auto_nonmand_acc',
                 'crash_acc']   # home-zone crash exposure - lets the search test
                                # safety in CDAP/generation, not only mode/dest
                                # utilities (ATRF reviewer request)
    raw_synth = ['vehrd', 'nprd', 'incp', 'sexp',
                 'workstat_fulltime', 'workstat_parttime', 'workstat_notworking',
                 'studying_none', 'studying_primary', 'studying_secondary',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'CARLICENCE', 'occp_group']
    abs_cols  = ['empp', 'occp', 'rntrd', 'mstp',  # ABS direct (census_bridge)
                 'mtwp',          # travel-to-work mode   strong CDAP predictor
                 'preferred_mode_DA', 'preferred_mode_Transit',
                 'preferred_mode_Walk', 'preferred_mode_Bike']
    all_cands = (ptype_dummies + age_vars + hh_vars + access +
                 work_vars + acc_cols + raw_synth + abs_cols)
    return [c for c in all_cands if c in data.columns]


def _usual_loc_work_attractor_candidates() -> List[str]:
    """Landuse columns eligible as work zone attractors."""
    return ['TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN',
            'AGREMPN', 'MWTEMPN', 'OTHEMP', 'commercial', 'industrial']


def _usual_loc_edu_attractor_candidates() -> List[str]:
    """Landuse columns eligible as school/university zone attractors."""
    return ['Education', 'Primary', 'Secondary', 'education',
            'HEREMPN', 'HSENROLL', 'AGE0519', 'TOTEMP']


def _profile_vars(data: pd.DataFrame) -> List[str]:
    """CDAP-style profile variables available in the data."""
    profile = ['has_vehicle', 'has_licence', 'is_female', 'wfh', 'income_midplus',
               'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
               'age_norm', 'hh_child', 'hh_spouse', 'hh_parent', 'hh_alone',
               'hh_size_large', 'mandatory_role',
               # ABS-derived (census_bridge)
               'income_hi', 'income_lo', 'age_retired', 'age_young', 'n_workers_hh']
    return [c for c in profile if c in data.columns]


def _abs_person_vars(data: pd.DataFrame) -> List[str]:
    """
    ABS census columns available after map_census_to_hhts() has been run.
    These are the extra attributes that synth_attributes + census_bridge add
    but the HHTS estimation data may not have   included only when present.
    """
    abs_cols = [
        'empp',           # employment status (ABS direct)
        'occp',           # occupation 1-digit (ABS direct)
        'rntrd',          # rent/mortgage status
        'mstp',           # marital status
        'mtwp',           # method of travel to work   mode preference proxy
        'preferred_mode', # string mode from mtwp mapping
        'n_workers_hh',   # household-level worker count
        'income_hi',      # incp >= 10
        'income_lo',      # incp <= 4
        'age_working',    # agep 20-64
        'age_retired',    # agep >= 60
        'age_young',      # agep <= 20
        'bikes',          # bicycles in household (imputed by census_bridge)
        'WFHWEEK',        # WFH days per week (imputed)
        'INDUSTRY',       # industry text (derived from indp)
    ]
    return [c for c in abs_cols if c in data.columns]


def _taste_interaction_candidates(
    data: pd.DataFrame,
    continuous: List[str],
    segments: List[str],
    prefix: str = 'ix',
    max_pairs: int = 30,
) -> List[str]:
    """
    Build taste-variation interaction columns in `data` and return their names.

    For each (cont, seg) pair where both columns exist and have variance, compute
    cont * seg and store as '<prefix>_<seg>_x_<cont>'.  These proxies capture
    heterogeneous preferences without requiring a full mixed-logit random-coefficient
    sampler, which is not supported by the current scikit/statsmodels estimators.

    Only pairs where the interaction column has non-trivial variance are kept.
    `max_pairs` caps the number added to avoid blowing up the chromosome size.
    """
    added: List[str] = []
    for seg in segments:
        if seg not in data.columns:
            continue
        for cont in continuous:
            if cont not in data.columns or len(added) >= max_pairs:
                break
            col_name = f'{prefix}_{seg}_x_{cont}'
            if col_name not in data.columns:
                try:
                    vals = data[seg] * data[cont]
                    if vals.std() > 1e-9:
                        data[col_name] = vals
                except Exception:
                    continue
            if col_name in data.columns:
                added.append(col_name)
    return added


def _meaningful_interaction_candidates(
    data: pd.DataFrame,
    specs: List[Tuple[str, str, str]],
    min_nonzero_share: float = 0.05,
    max_abs_corr: float = 0.97,
) -> List[str]:
    """Create only interactions with usable coverage and without parent duplication."""
    added: List[str] = []
    for left, right, name in specs:
        if left not in data.columns or right not in data.columns:
            continue
        lhs = pd.to_numeric(data[left], errors='coerce').fillna(0.0).astype(float)
        rhs = pd.to_numeric(data[right], errors='coerce').fillna(0.0).astype(float)
        vals = lhs * rhs
        nonzero_share = float((np.abs(vals) > 1e-9).mean())
        if vals.std() <= 1e-9 or nonzero_share < min_nonzero_share:
            continue
        keep = True
        for parent in (lhs, rhs):
            if parent.std() <= 1e-9:
                continue
            corr = np.corrcoef(vals, parent)[0, 1]
            if np.isfinite(corr) and abs(float(corr)) >= max_abs_corr:
                keep = False
                break
        if not keep:
            continue
        data[name] = vals
        added.append(name)
    return added


def _available_feature_list(data: pd.DataFrame, names: List[str]) -> List[str]:
    return [c for c in dict.fromkeys(names) if c in data.columns]


_S1_CORE_BY_PURPOSE = {
    'work': ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'ptype_WorkingStudent',
             'mandatory_role', 'incp', 'vehrd', 'CARLICENCE', 'auto_work_acc'],
    'edu':  ['ptype_SchoolChild', 'ptype_UniversityStudent', 'ptype_WorkingStudent',
             'age_child', 'age_teen', 'transit_acc'],
    'escort':   ['hh_child', 'nprd', 'vehrd', 'has_vehicle', 'age_working'],
    'shopping': ['incp', 'vehrd', 'has_vehicle', 'auto_shop_acc', 'nprd'],
    'social':   ['nprd', 'vehrd', 'walk_acc', 'transit_acc', 'is_female'],
    'othmaint': ['nprd', 'vehrd', 'has_vehicle', 'auto_nonmand_acc'],
    'othdiscr': ['incp', 'vehrd', 'auto_nonmand_acc', 'age_older'],
    'eat':      ['incp', 'nprd', 'walk_acc', 'transit_acc'],
    # Combined non-mandatory: union of all six nonmand core specs (work/edu/nonmand structure).
    'nonmand':  ['hh_child', 'nprd', 'vehrd', 'has_vehicle', 'age_working',
                 'incp', 'auto_shop_acc', 'walk_acc', 'transit_acc', 'is_female',
                 'auto_nonmand_acc', 'age_older', 'mandatory_role',
                 'ptype_FullTimeWorker', 'ptype_PartTimeWorker'],
}

_S1_OPTIONAL_BY_PURPOSE = {
    # ── Work frequency (NB2) ─────────────────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on work tour count.
    'work':     ['nprd', 'HHSIZE', 'is_female', 'age_working', 'wfh', 'WFHWEEK',
                 'n_workers_hh', 'income_hi', 'income_lo', 'occp_group', 'empp',
                 'income_midplus', 'sexp',
                 'workstat_fulltime', 'workstat_parttime',  # binary, not raw WORKSTAT ordinal
                 'mandatory_role',
                 'has_vehicle', 'no_vehicle', 'hh_child',
                 'transit_acc', 'walk_acc', 'bike_acc',
                 'ptype_Retired', 'age_senior',            # retirement suppresses work tours
                 'rntrd', 'mstp', 'preferred_mode',        # housing tenure, marital status, mode pref
                 'INDUSTRY', 'rlhp'],                       # industry sector, residential tenure type
    # ── Education frequency (NB2) ────────────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on edu tour count.
    'edu':      ['vehrd', 'has_vehicle', 'is_female', 'income_lo', 'mandatory_role',
                 'auto_work_acc', 'walk_acc', 'HHSIZE',
                 'income_hi', 'income_midplus',
                 'studying_tertiary_pt', 'studying_tertiary_ft',  # binary, not raw STUDYING
                 'studying_secondary', 'studying_primary',
                 'nprd',
                 'age_working', 'sexp', 'no_vehicle', 'transit_acc',
                 'ptype_Retired', 'CARLICENCE', 'age_senior',  # driving licence for school access
                 'ptype_Preschool',                             # preschool enrolments (childcare as edu)
                 'rntrd', 'hh_spouse', 'hh_parent'],            # housing + family structure → school choice
    # ── Escort frequency (NB2) ───────────────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on escort tour count.
    'escort':   ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'is_female',
                 'auto_nonmand_acc', 'no_vehicle', 'income_midplus',
                 'income_hi', 'income_lo', 'incp', 'sexp',
                 'workstat_fulltime', 'workstat_parttime', 'mandatory_role',
                 'studying_tertiary_pt', 'studying_tertiary_ft', 'age_working',
                 'transit_acc', 'walk_acc',
                 'ptype_Retired', 'age_teen',              # retired grandparents escort; teens escort siblings
                 'hh_alone', 'hh_size_large',               # solo HH never escorts; large HH generates more escort
                 'ptype_Preschool', 'ptype_WorkingStudent', # preschool needs escort; working students escort too
                 'hh_spouse', 'hh_parent',                  # spouse/parent roles affect escort duties
                 'mstp'],                                    # marital status (single parents escort more)
    # ── Shopping frequency (NB2) ─────────────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on shopping tour count.
    'shopping': ['hh_child', 'age_older', 'wfh', 'income_hi', 'is_female',
                 'no_vehicle', 'bike_acc', 'walk_acc',
                 'income_lo', 'income_midplus', 'sexp', 'mandatory_role',
                 'workstat_fulltime', 'workstat_parttime', 'age_working', 'transit_acc', 'auto_shop_acc',
                 'ptype_Retired', 'age_senior',            # retirees shop more; elderly have distinct patterns
                 'hh_alone', 'hh_size_large',              # solo HH shops for one; large HH shops more
                 'CARLICENCE', 'bikes',                    # car/bike access to shops
                 'ptype_Preschool', 'rntrd',               # preschool needs family shopping; owners vs renters
                 'mstp'],                                   # marital status influences shopping patterns
    # ── Social frequency (NB2) ───────────────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on social tour count.
    'social':   ['incp', 'age_teen', 'age_older', 'hh_child', 'bikes', 'no_vehicle',
                 'income_hi', 'income_lo', 'income_midplus', 'is_female', 'sexp',
                 'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'age_working',
                 'transit_acc', 'walk_acc', 'bike_acc',
                 'ptype_Retired', 'age_senior',            # retirees have more social time
                 'hh_alone', 'hh_size_large', 'CARLICENCE',  # solo HH seeks social contact; car for outings
                 'ptype_UniversityStudent', 'ptype_WorkingStudent',  # students socialise more
                 'hh_spouse', 'hh_parent',                 # family roles affect social patterns
                 'mstp', 'rntrd'],                          # marital/housing status → social engagement
    # ── Other-maintenance frequency (NB2) ────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on othmaint tour count.
    'othmaint': ['ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                 'hh_child', 'is_female', 'income_lo', 'walk_acc',
                 'income_hi', 'income_midplus', 'incp', 'sexp',
                 'workstat_fulltime', 'workstat_parttime',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'mandatory_role', 'age_older',
                 'auto_nonmand_acc', 'transit_acc', 'no_vehicle',
                 'ptype_Retired', 'age_senior',            # retirees have more time for maintenance errands
                 'hh_alone', 'hh_size_large', 'CARLICENCE',  # car access for errand trips
                 'ptype_UniversityStudent', 'rntrd',        # students errand patterns; renters vs owners
                 'mstp'],                                    # marital status affects maintenance burden
    # ── Other-discretionary frequency (NB2) ──────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on othdiscr tour count.
    'othdiscr': ['walk_acc', 'transit_acc', 'is_female', 'nprd', 'bikes',
                 'income_hi', 'income_lo', 'income_midplus', 'incp', 'sexp',
                 'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'age_working', 'hh_child',
                 'bike_acc', 'auto_nonmand_acc', 'no_vehicle',
                 'ptype_Retired', 'age_senior', 'age_teen',   # retired/teens have most discretionary time
                 'hh_alone', 'hh_size_large', 'CARLICENCE',   # solo HH/car access for leisure
                 'ptype_UniversityStudent', 'ptype_WorkingStudent',  # students have flexible leisure time
                 'hh_spouse', 'mstp'],                         # spouse/marital status → joint vs solo leisure
    # ── Eat-out frequency (NB2) ──────────────────────────────────────────────
    # Add to this list to test extra socio-demographic shifters on eat tour count.
    'eat':      ['is_female', 'income_hi', 'age_working', 'bike_acc', 'auto_nonmand_acc',
                 'income_lo', 'income_midplus', 'sexp', 'mandatory_role',
                 'workstat_fulltime', 'workstat_parttime', 'age_older', 'hh_child', 'no_vehicle',
                 'walk_acc', 'transit_acc',
                 'ptype_Retired', 'age_senior', 'age_teen',  # retirees/teens eat out frequently
                 'hh_alone', 'hh_size_large',                # solo HH eats out often; large HH group meals
                 'CARLICENCE', 'bikes',                      # car/bike access to restaurants
                 'ptype_UniversityStudent', 'ptype_WorkingStudent',  # students eat out more
                 'mstp', 'rntrd'],                            # marital/housing → dining patterns
    # Combined non-mandatory: union of all six nonmand optional pools.
    'nonmand':  ['incp', 'age_teen', 'age_older', 'hh_child', 'bikes', 'no_vehicle',
                 'income_hi', 'income_lo', 'income_midplus', 'is_female', 'sexp',
                 'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'age_working',
                 'transit_acc', 'walk_acc', 'bike_acc', 'auto_shop_acc', 'auto_nonmand_acc',
                 'ptype_Retired', 'age_senior',
                 'hh_alone', 'hh_size_large', 'CARLICENCE',
                 'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                 'ptype_UniversityStudent', 'ptype_WorkingStudent', 'ptype_Preschool',
                 'hh_spouse', 'hh_parent',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'mstp', 'rntrd', 'wfh'],
}

# Augment stage 1 optional pools with granular income / vehicle / household-size bands.
# The collinear groups ensure at most one income band and at most one vehicle band is
# active per chromosome; the BIC + significance penalties handle the rest.
_S1_OPTIONAL_EXTRA = [
    'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',  # finer income splits
    'one_vehicle', 'multi_vehicle',                                   # 1-car vs 2+ car HH
    'hh_couple', 'hh_medium',                                         # 2-person couple; 3-person family
    # ── Extended categorical / socio-demographic bands ─────────────────────────
    'hh_spouse',        'hh_parent',         # household relationship roles
    'age_retired',      'age_young',         # finer age brackets (ABS-derived)
    'rlhp',                                   # rent/live-here-payment status
    'WFHDAY',                                 # work-from-home day dummy
    'bikes',                                  # bicycles in household
    'empp',                                   # employment status (ABS code)
    'occp_group',                             # occupation group
    'rntrd',                                  # rent/mortgage status (ABS)
    'mstp',                                   # marital status (ABS)
    'preferred_mode',                         # mode preference derived from mtwp
    'INDUSTRY',                               # industry category (ABS-derived)
]
for _p1k in list(_S1_OPTIONAL_BY_PURPOSE):
    for _v1 in _S1_OPTIONAL_EXTRA:
        if _v1 not in _S1_OPTIONAL_BY_PURPOSE[_p1k]:
            _S1_OPTIONAL_BY_PURPOSE[_p1k].append(_v1)

# No interaction terms — use binary ptype dummies to capture heterogeneity directly.
_S1_INTERACTIONS: dict = {}

_S2_CORE_BY_TARGET = {
    'start_of_day': ['n_work_out', 'n_edu_out',
                     'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                     'ptype_SchoolChild', 'ptype_UniversityStudent',
                     'vehrd', 'wfh', 'WFHWEEK'],
    'end_of_day':   ['n_work_out', 'n_edu_out', 'n_tours',
                     'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                     'ptype_SchoolChild', 'ptype_UniversityStudent',
                     'vehrd'],
    'B':            ['n_total', 'n_tours', 'n_work_out', 'n_edu_out',
                     'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                     'ptype_SchoolChild', 'ptype_UniversityStudent',
                     'incp', 'vehrd'],
}

_S2_OPTIONAL_BY_TARGET = {
    # ── Start-of-day departure time (tobit/lognormal) ─────────────────────────
    # Add to this list to test extra variables that shift when a person first
    # leaves home. Mandatory-tour counts are in core; discretionary counts here.
    'start_of_day': ['n_social_out', 'n_escort_out', 'sexp', 'nprd', 'income_hi', 'has_vehicle',
                     'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out',
                     'is_female', 'income_lo', 'income_midplus', 'incp',
                     'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'age_working', 'age_older',
                     'walk_acc', 'transit_acc',
                     'ptype_Retired',                       # retired depart later (no peak-hour pressure)
                     'n_eat_out',                           # eating tours extend the activity window
                     'hh_alone', 'hh_size_large',          # solo vs large-HH scheduling constraints
                     'age_senior', 'n_workers_hh'],         # very old depart later; multi-worker HH stagger
    # ── End-of-day return time (tobit/lognormal) ──────────────────────────────
    # Add to this list to test extra variables that shift when a person last
    # returns home.
    'end_of_day':   ['n_social_out', 'n_shopping_out', 'sexp', 'nprd', 'income_hi', 'has_vehicle',
                     'n_escort_out', 'n_othmaint_out', 'n_othdiscr_out',
                     'is_female', 'income_lo', 'income_midplus', 'incp',
                     'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'age_working', 'age_older',
                     'walk_acc', 'transit_acc',
                     'ptype_Retired',                       # retired return later (flexible schedule)
                     'n_eat_out',                           # eating tours delay return
                     'hh_alone', 'hh_size_large',          # solo vs large-HH timing
                     'age_senior', 'n_workers_hh',         # very old return earlier; multi-worker stagger
                     'CARLICENCE'],                         # car licence enables later return trips
    # ── Total out-of-home time budget B (tobit/lognormal) ────────────────────
    # Add to this list to test extra variables that shift total time away from home.
    'B':            ['n_social_out', 'n_escort_out', 'n_shopping_out', 'n_othmaint_out',
                     'n_othdiscr_out', 'sexp', 'nprd', 'income_hi', 'income_lo', 'has_vehicle',
                     'is_female', 'income_midplus', 'mandatory_role',
                     'workstat_fulltime', 'workstat_parttime',
                     'age_working', 'age_older', 'hh_child', 'no_vehicle',
                     'walk_acc', 'transit_acc',
                     'ptype_Retired',                       # retired have longer out-of-home time
                     'n_eat_out',                           # eating tours add to total budget
                     'hh_alone', 'hh_size_large',          # solo HH flexible; large HH constrained
                     'age_senior', 'n_workers_hh',         # very old shorter budget; multi-worker households
                     'CARLICENCE'],                         # car licence expands accessible range -> longer trips
}

# Augment stage 2 optional pools with granular income / vehicle / household-size bands.
_S2_OPTIONAL_EXTRA = [
    'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
    'one_vehicle', 'multi_vehicle',
    'hh_couple', 'hh_medium',
    'hh_spouse', 'hh_parent', 'age_retired', 'age_young',
    'rlhp', 'WFHDAY', 'bikes', 'empp', 'occp_group',
    'rntrd', 'mstp', 'preferred_mode', 'INDUSTRY',
]
for _p2k in list(_S2_OPTIONAL_BY_TARGET):
    for _v2 in _S2_OPTIONAL_EXTRA:
        if _v2 not in _S2_OPTIONAL_BY_TARGET[_p2k]:
            _S2_OPTIONAL_BY_TARGET[_p2k].append(_v2)

_S3_PURPOSE_PERSON_CORE = {
    'work':     ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'ptype_WorkingStudent',
                 'incp', 'vehrd', 'CARLICENCE'],
    'edu':      ['ptype_SchoolChild', 'ptype_UniversityStudent',
                 'age_child', 'age_teen'],
    'escort':   ['hh_child', 'nprd', 'vehrd', 'has_vehicle'],
    'shopping': ['incp', 'vehrd', 'nprd', 'has_vehicle'],
    'social':   ['nprd', 'is_female', 'age_teen', 'age_older'],
    'othmaint': ['nprd', 'has_vehicle',
                 'ptype_FullTimeWorker', 'ptype_PartTimeWorker'],
    'othdiscr': ['incp', 'vehrd', 'age_older'],
    'eat':      ['incp', 'nprd', 'age_working'],
    # Combined non-mandatory: union of all six nonmand person cores.
    'nonmand':  ['hh_child', 'nprd', 'vehrd', 'has_vehicle', 'incp',
                 'is_female', 'age_teen', 'age_older', 'ptype_FullTimeWorker',
                 'ptype_PartTimeWorker', 'age_working'],
}

_S3_PURPOSE_OPTIONAL = {
    # ── Work activity duration (lognormal/survival) ───────────────────────────
    # Add to this list to test extra variables that shift work episode duration.
    'work':     ['income_hi', 'income_lo', 'income_midplus', 'is_female', 'mandatory_role',
                 'nprd', 'sexp',
                 'workstat_fulltime', 'workstat_parttime',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'n_workers_hh', 'wfh', 'WFHWEEK', 'CARLICENCE', 'has_vehicle',
                 'auto_work_acc', 'age_working', 'hh_child', 'no_vehicle',
                 'transit_acc', 'walk_acc', 'age_older',
                 'empp',                                    # employment participation intensity → longer work durations
                 'hh_alone'],                               # solo HH has fewer time-coordination constraints
    # ── Education activity duration (lognormal/survival) ─────────────────────
    # Add to this list to test extra variables that shift education episode duration.
    'edu':      ['income_lo', 'income_midplus', 'income_hi', 'is_female',
                 'vehrd', 'mandatory_role', 'sexp',
                 'workstat_fulltime', 'workstat_parttime',
                 'studying_secondary', 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'transit_acc', 'walk_acc', 'nprd', 'has_vehicle',
                 'age_working', 'hh_child', 'no_vehicle', 'bike_acc',
                 'ptype_Retired',                           # rare retired learners; evening classes
                 'hh_alone', 'CARLICENCE'],                 # car access to school/university
    # ── Escort activity duration (lognormal/survival) ────────────────────────
    # Add to this list to test extra variables that shift escort episode duration.
    'escort':   ['ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                 'is_female', 'income_midplus', 'income_hi', 'income_lo',
                 'mandatory_role', 'workstat_fulltime', 'workstat_parttime',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'sexp', 'hh_child',
                 'age_working', 'age_older', 'no_vehicle', 'incp',
                 'transit_acc', 'walk_acc',
                 'ptype_Retired',                           # retired grandparents spend more time escorting
                 'age_teen',                                # teens escorting siblings have longer durations
                 'hh_alone'],                               # solo HH occasionally escorts (less common)
    # ── Shopping activity duration (lognormal/survival) ──────────────────────
    # Add to this list to test extra variables that shift shopping episode duration.
    'shopping': ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'nprd', 'sexp', 'mandatory_role',
                 'workstat_fulltime', 'workstat_parttime',
                 'age_older', 'has_vehicle', 'walk_acc', 'auto_shop_acc',
                 'age_working', 'hh_child', 'no_vehicle', 'bikes',
                 'transit_acc', 'bike_acc',
                 'ptype_Retired', 'age_senior',             # retirees browse longer; elderly have more time
                 'hh_alone', 'hh_size_large'],              # solo HH shops carefully; large HH buys in bulk
    # ── Social activity duration (lognormal/survival) ────────────────────────
    # Add to this list to test extra variables that shift social episode duration.
    'social':   ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'sexp', 'mandatory_role', 'nprd',
                 'workstat_fulltime', 'workstat_parttime',
                 'transit_acc', 'walk_acc', 'age_working', 'hh_child',
                 'no_vehicle', 'bikes', 'bike_acc', 'auto_nonmand_acc',
                 'ptype_Retired', 'age_senior',             # retired have long social visits
                 'hh_alone', 'CARLICENCE'],                 # solo HH seeks longer social contact; car for outings
    # ── Other-maintenance activity duration (lognormal/survival) ─────────────
    # Add to this list to test extra variables that shift othmaint episode duration.
    'othmaint': ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'sexp', 'workstat_fulltime', 'workstat_parttime',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'mandatory_role',
                 'hh_child', 'has_vehicle', 'auto_nonmand_acc',
                 'age_working', 'age_older', 'no_vehicle', 'bikes',
                 'transit_acc', 'walk_acc', 'bike_acc', 'incp',
                 'ptype_Retired', 'age_senior',             # retirees spend more time on maintenance tasks
                 'hh_alone'],                               # solo HH handles all maintenance personally → longer
    # ── Other-discretionary activity duration (lognormal/survival) ───────────
    # Add to this list to test extra variables that shift othdiscr episode duration.
    'othdiscr': ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'nprd', 'sexp', 'workstat_fulltime', 'workstat_parttime', 'mandatory_role',
                 'walk_acc', 'bike_acc', 'age_working', 'hh_child',
                 'no_vehicle', 'bikes', 'transit_acc', 'auto_nonmand_acc',
                 'ptype_Retired', 'age_senior', 'age_teen', # retirees/teens engage in longer leisure
                 'hh_alone', 'CARLICENCE'],                 # solo HH has more time; car enables longer outings
    # ── Eat activity duration (lognormal/survival) ────────────────────────────
    # Add to this list to test extra variables that shift eat episode duration.
    'eat':      ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'nprd', 'sexp', 'mandatory_role',
                 'workstat_fulltime', 'workstat_parttime',
                 'walk_acc', 'transit_acc', 'age_older', 'hh_child',
                 'no_vehicle', 'bike_acc', 'auto_nonmand_acc', 'has_vehicle',
                 'ptype_Retired', 'age_senior', 'age_teen',  # retirees/teens have longer meal durations
                 'hh_alone', 'hh_size_large',                # solo HH lingers; group meals take longer
                 'CARLICENCE'],                              # car access enables more distant, longer dining
    # Combined non-mandatory: union of all six nonmand optional pools.
    'nonmand':  ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'nprd', 'sexp', 'mandatory_role',
                 'workstat_fulltime', 'workstat_parttime',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'hh_child', 'has_vehicle', 'auto_nonmand_acc', 'auto_shop_acc',
                 'age_working', 'age_older', 'age_teen', 'no_vehicle', 'bikes',
                 'transit_acc', 'walk_acc', 'bike_acc', 'incp',
                 'ptype_Retired', 'age_senior',
                 'hh_alone', 'hh_size_large', 'CARLICENCE',
                 'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                 'hh_spouse', 'hh_parent', 'mstp', 'rntrd'],
}

# Augment stage 3 optional pools with granular income / vehicle / household-size bands.
_S3_OPTIONAL_EXTRA = [
    'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
    'one_vehicle', 'multi_vehicle',
    'hh_couple', 'hh_medium',
    'hh_spouse', 'hh_parent', 'age_retired', 'age_young',
    'rlhp', 'WFHDAY', 'bikes', 'empp', 'occp_group',
    'rntrd', 'mstp', 'preferred_mode', 'INDUSTRY',
]
for _p3k in list(_S3_PURPOSE_OPTIONAL):
    for _v3 in _S3_OPTIONAL_EXTRA:
        if _v3 not in _S3_PURPOSE_OPTIONAL[_p3k]:
            _S3_PURPOSE_OPTIONAL[_p3k].append(_v3)

# No interaction terms — binary ptype dummies capture heterogeneity directly.
_S3_INTERACTIONS: dict = {}

_S4_PURPOSE_PERSON_CORE = {
    'work':     ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'ptype_WorkingStudent',
                 'CARLICENCE', 'vehrd'],
    'edu':      ['ptype_SchoolChild', 'ptype_UniversityStudent',
                 'age_child', 'age_teen'],
    'escort':   ['hh_child', 'vehrd', 'has_vehicle', 'nprd'],
    'shopping': ['incp', 'vehrd', 'has_vehicle', 'nprd'],
    'social':   ['is_female', 'nprd', 'age_teen', 'age_older'],
    'othmaint': ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'vehrd', 'nprd'],
    'othdiscr': ['incp', 'vehrd', 'age_older'],
    'eat':      ['incp', 'nprd', 'walk_acc'],
}

_S4_PURPOSE_OPTIONAL = {
    # ── Work tour schedule placement (time-of-day logit) ─────────────────────
    # Add to this list to test extra variables that shift work-tour start time.
    'work':     ['is_female', 'income_hi', 'income_lo', 'income_midplus',
                 'sexp', 'mandatory_role', 'has_vehicle', 'no_vehicle',
                 'CARLICENCE', 'wfh', 'WFHWEEK', 'n_workers_hh', 'incp',
                 'age_working', 'hh_child', 'transit_acc', 'walk_acc',
                 'auto_work_acc', 'workstat_fulltime', 'workstat_parttime',
                 'ptype_Retired',                           # rare non-standard work tours by retired
                 'hh_alone'],                               # solo worker has flexible scheduling
    # ── Education tour schedule placement (time-of-day logit) ────────────────
    # Add to this list to test extra variables that shift edu-tour start time.
    'edu':      ['is_female', 'income_lo', 'income_midplus', 'income_hi',
                 'vehrd', 'mandatory_role', 'sexp',
                 'workstat_fulltime', 'workstat_parttime',
                 'studying_secondary', 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'has_vehicle', 'transit_acc', 'walk_acc',
                 'age_working', 'hh_child', 'no_vehicle', 'bike_acc', 'incp',
                 'ptype_Retired',                           # retired adult learners attend at non-standard times
                 'hh_alone'],                               # solo household learners have flexible timing
    # ── Escort tour schedule placement (time-of-day logit) ───────────────────
    # Add to this list to test extra variables that shift escort-tour start time.
    'escort':   ['ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                 'is_female', 'income_midplus', 'income_hi', 'income_lo',
                 'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'sexp', 'hh_child',
                 'age_working', 'age_older', 'no_vehicle', 'incp',
                 'transit_acc', 'walk_acc',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'ptype_Retired',                           # retired grandparents escort at mid-day
                 'hh_alone', 'hh_size_large', 'n_workers_hh'],  # HH structure constrains escort timing
    # ── Shopping tour schedule placement (time-of-day logit) ─────────────────
    # Add to this list to test extra variables that shift shopping-tour start time.
    'shopping': ['is_female', 'income_hi', 'income_lo', 'income_midplus',
                 'sexp', 'mandatory_role', 'workstat_fulltime', 'workstat_parttime', 'has_vehicle',
                 'walk_acc', 'bike_acc', 'age_older',
                 'age_working', 'hh_child', 'no_vehicle', 'bikes',
                 'transit_acc', 'auto_shop_acc', 'auto_nonmand_acc',
                 'ptype_Retired',                           # retirees shop mid-day, avoiding peak hours
                 'hh_alone', 'hh_size_large', 'CARLICENCE'],  # solo HH shops anytime; large HH evenings
    # ── Social tour schedule placement (time-of-day logit) ───────────────────
    # Add to this list to test extra variables that shift social-tour start time.
    'social':   ['income_hi', 'income_lo', 'income_midplus', 'is_female',
                 'mandatory_role', 'sexp', 'workstat_fulltime', 'workstat_parttime', 'has_vehicle',
                 'transit_acc', 'walk_acc', 'age_working', 'hh_child',
                 'no_vehicle', 'bikes', 'bike_acc', 'incp',
                 'ptype_Retired',                           # retired social visits concentrate mid-morning
                 'hh_alone', 'hh_size_large', 'CARLICENCE'],  # solo HH flexible; CARLICENCE for evening visits
    # ── Other-maintenance tour schedule placement (time-of-day logit) ────────
    # Add to this list to test extra variables that shift othmaint-tour start time.
    'othmaint': ['is_female', 'income_hi', 'income_lo', 'income_midplus',
                 'sexp', 'workstat_fulltime', 'workstat_parttime',
                 'studying_tertiary_pt', 'studying_tertiary_ft',
                 'mandatory_role',
                 'hh_child', 'has_vehicle', 'auto_nonmand_acc',
                 'age_older', 'no_vehicle', 'bikes', 'incp',
                 'transit_acc', 'walk_acc', 'bike_acc',
                 'ptype_Retired',                           # retired do maintenance mid-day
                 'hh_alone', 'hh_size_large', 'CARLICENCE'],  # household structure affects errand timing
    # ── Other-discretionary tour schedule placement (time-of-day logit) ──────
    # Add to this list to test extra variables that shift othdiscr-tour start time.
    'othdiscr': ['is_female', 'income_hi', 'income_lo', 'income_midplus',
                 'nprd', 'sexp', 'workstat_fulltime', 'workstat_parttime', 'mandatory_role',
                 'walk_acc', 'bike_acc', 'age_working', 'hh_child',
                 'no_vehicle', 'bikes', 'transit_acc', 'auto_nonmand_acc', 'incp',
                 'ptype_Retired',                           # retired leisure at non-peak times
                 'hh_alone', 'hh_size_large', 'CARLICENCE'],  # solo HH most flexible; car for late outings
    # ── Eat tour schedule placement (time-of-day logit) ──────────────────────
    # Add to this list to test extra variables that shift eat-tour start time.
    'eat':      ['is_female', 'income_hi', 'income_lo', 'income_midplus',
                 'sexp', 'mandatory_role', 'vehrd',
                 'workstat_fulltime', 'workstat_parttime', 'has_vehicle',
                 'transit_acc', 'age_older', 'age_working', 'hh_child',
                 'no_vehicle', 'bike_acc', 'auto_nonmand_acc', 'incp', 'nprd',
                 'ptype_Retired',                           # retirees eat at mid-day, not lunch-rush
                 'hh_alone', 'hh_size_large', 'CARLICENCE'],  # solo HH dines anytime; CARLICENCE for evening
}

# Augment stage 4 optional pools with granular income / vehicle / household-size bands.
_S4_OPTIONAL_EXTRA = [
    'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
    'one_vehicle', 'multi_vehicle',
    'hh_couple', 'hh_medium',
    'hh_spouse', 'hh_parent', 'age_retired', 'age_young',
    'rlhp', 'WFHDAY', 'bikes', 'empp', 'occp_group',
    'rntrd', 'mstp', 'preferred_mode', 'INDUSTRY',
]
for _p4k in list(_S4_PURPOSE_OPTIONAL):
    for _v4 in _S4_OPTIONAL_EXTRA:
        if _v4 not in _S4_PURPOSE_OPTIONAL[_p4k]:
            _S4_PURPOSE_OPTIONAL[_p4k].append(_v4)

# No interaction terms — binary ptype dummies capture heterogeneity directly.
_S4_INTERACTIONS: dict = {}


def _prune_zero_coef_features(
    feats: List[str],
    coefs: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = 0,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Iteratively drop features whose |coefficient| < threshold, then re-fit.

    Returns (pruned_feats, pruned_coefs, pruned_X) or original if no pruning.
    The caller must re-compute BIC/AIC with the returned values.

    Threshold is read from GA_SIG_COEF_ZERO env var (default 0.01), matching
    the convention in ga_external_estimators._metacount_significance_penalty.
    """
    threshold = float(os.environ.get('GA_SIG_COEF_ZERO', '0.01'))
    if threshold <= 0 or len(feats) <= 1:
        return feats, coefs, X

    coef_arr = np.asarray(coefs).ravel()
    feat_mask = np.abs(coef_arr) >= threshold
    if feat_mask.all():
        return feats, coefs, X

    pruned_feats = [f for f, ok in zip(feats, feat_mask) if ok]
    if not pruned_feats:
        return feats, coefs, X

    pruned_X = X[:, feat_mask]
    return pruned_feats, coef_arr[feat_mask], pruned_X


def _stage1_feature_spec(data: pd.DataFrame, purpose: str) -> Tuple[List[str], List[str]]:
    core = _available_feature_list(data, _S1_CORE_BY_PURPOSE.get(purpose, []))
    optional = _available_feature_list(data, _S1_OPTIONAL_BY_PURPOSE.get(purpose, []))
    return list(dict.fromkeys(core + optional)), core


def _filter_candidates_by_signal(
    data: pd.DataFrame,
    candidates: List[str],
    min_unique: int = 2,
    min_binary_share: float = 0.01,
    max_binary_share: float = 0.99,
) -> List[str]:
    """Drop near-constant or all-missing candidates that destabilize GA selection."""
    keep: List[str] = []
    for c in candidates:
        if c not in data.columns:
            continue
        s = pd.to_numeric(data[c], errors='coerce') if data[c].dtype == object else data[c]
        s = pd.Series(s).dropna()
        if s.empty:
            continue
        nunq = int(s.nunique(dropna=True))
        if nunq < min_unique:
            continue
        # Binary columns with extreme prevalence have little identification power.
        if nunq == 2:
            p = float((s > 0).mean())
            if p < min_binary_share or p > max_binary_share:
                continue
        keep.append(c)
    return list(dict.fromkeys(keep))


# Structural (activity-count / budget) candidates shared by every Stage 3 purpose.
_S3_STRUCTURAL = [
    'start_hr', 'log_dist', 'B', 'budget_per_activity_hr', 'day_span_hr',
    'n_mandatory', 'n_discretionary',
    'n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
    'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
]


def _stage2_feature_spec(data: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    core = _available_feature_list(data, _S2_CORE_BY_TARGET.get(target, []))
    optional = _available_feature_list(data, _S2_OPTIONAL_BY_TARGET.get(target, []))
    return list(dict.fromkeys(core + optional)), core


def _stage3_feature_spec(data: pd.DataFrame, purpose: str) -> Tuple[List[str], List[str]]:
    structural = _available_feature_list(data, _S3_STRUCTURAL)
    core = structural + _available_feature_list(data, _S3_PURPOSE_PERSON_CORE.get(purpose, []))
    optional = _available_feature_list(data, _S3_PURPOSE_OPTIONAL.get(purpose, []))
    return list(dict.fromkeys(core + optional)), list(dict.fromkeys(core))


def _raw_stage_candidates(stage: int, key: str) -> List[str]:
    """Full candidate list for a Stage 2 target / Stage 3 purpose BEFORE any
    data-column or signal filtering — the denominator for availability."""
    if stage == 2:
        return list(dict.fromkeys(list(_S2_CORE_BY_TARGET.get(key, []))
                                  + list(_S2_OPTIONAL_BY_TARGET.get(key, []))))
    if stage == 3:
        return list(dict.fromkeys(list(_S3_STRUCTURAL)
                                  + list(_S3_PURPOSE_PERSON_CORE.get(key, []))
                                  + list(_S3_PURPOSE_OPTIONAL.get(key, []))))
    return []


def _record_candidate_availability(stage, label, key, kept, data):
    """Log + persist which candidate predictors were kept vs dropped (and why)
    for one Stage 2/3 target/purpose. Lets a flat search be diagnosed as
    BIC-parsimony (terms present but penalised out — lower GA_COMPLEXITY_PENALTY)
    vs missing data columns (fix the synth-pop attributes). Appends
    candidate_availability_stage<N>.csv into GA_CHECKPOINT_DIR (or CWD)."""
    raw = _raw_stage_candidates(stage, key)
    if not raw:
        return
    cols = set(getattr(data, 'columns', []))
    keptset = set(kept)
    not_in_data = [c for c in raw if c not in cols]
    near_const  = [c for c in raw if c in cols and c not in keptset]
    n_search = len(raw) - len(not_in_data) - len(near_const)
    print(f"    [{label}] candidate availability: {len(raw)} defined -> "
          f"{n_search} searchable | {len(not_in_data)} not-in-data | "
          f"{len(near_const)} near-constant/filtered")
    if not_in_data:
        print(f"        not-in-data: {not_in_data}")
    if near_const:
        print(f"        near-constant/filtered: {near_const}")
    _dir = os.environ.get('GA_CHECKPOINT_DIR') or '.'
    try:
        import csv as _csv
        os.makedirs(_dir, exist_ok=True)
        _path = os.path.join(_dir, f'candidate_availability_stage{stage}.csv')
        _new = not os.path.exists(_path)
        with open(_path, 'a', newline='', encoding='utf-8') as _fh:
            _w = _csv.writer(_fh)
            if _new:
                _w.writerow(['stage', 'target', 'term', 'status'])
            for c in raw:
                st = ('searchable' if c in keptset
                      else 'not_in_data' if c not in cols
                      else 'near_constant')
                _w.writerow([stage, label, c, st])
    except Exception:
        pass


def _stage4_feature_spec(data: pd.DataFrame, purpose: str) -> Tuple[List[str], List[str]]:
    structural = _available_feature_list(
        data,
        ['day_position', 'time_consumed', 'duration_min', 'time_remaining',
         'time_of_day_sin', 'time_of_day_cos'],
    )
    core = structural + _available_feature_list(data, _S4_PURPOSE_PERSON_CORE.get(purpose, []))
    optional = _available_feature_list(data, _S4_PURPOSE_OPTIONAL.get(purpose, []))
    return list(dict.fromkeys(core + optional)), list(dict.fromkeys(core))


# ActivitySim-style interaction ("perturbation character") candidate genes,
# materialised by _ensure_stage_interactions(). Listed in every stage's
# candidate pool; the `c in data.columns` filter keeps only those whose column
# actually exists for that stage (e.g. timing interactions only in stage 4).
_ACCESS_INTERACTIONS = [
    f'{pfx}_x_{acc}'
    for acc in ('walk_acc', 'transit_acc', 'auto_work_acc', 'auto_shop_acc',
                'auto_nonmand_acc', 'bike_acc')
    for pfx in ('noveh', 'incp')
]
_LIFECYCLE_INTERACTIONS = ['child_x_working', 'child_x_female', 'older_x_noveh',
                           'senior_x_veh', 'retired_x_veh']
_OCC_WFH_INTERACTIONS = ['wfh_flag', 'wfh_x_worker', 'wfh_x_inchi',
                         'occ_whitecollar', 'occ_bluecollar', 'occ_x_incp']
_COUNT_TIMING_INTERACTIONS = ['ntotal_x_veh', 'ntours_x_worker',
                              'timeremain_x_worker', 'daypos_x_child']
# Extra search options: mobility / affordability / lifestage interactions.
_MOBILITY_INTERACTIONS = ['worker_x_autoacc', 'worker_x_transitacc',
                          'child_x_walkacc', 'incp_x_veh', 'incp_per_person',
                          'workers_x_veh', 'senior_x_transitacc']
_STAGE_INTERACTION_CANDIDATES = (_ACCESS_INTERACTIONS + _LIFECYCLE_INTERACTIONS
                                 + _OCC_WFH_INTERACTIONS + _COUNT_TIMING_INTERACTIONS
                                 + _MOBILITY_INTERACTIONS)


def _stage1_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 1 (activity frequency) — binary-first candidate set.

    WORKSTAT and STUDYING are ordinal codes; the GA uses their ptype binary
    equivalents (ptype_FullTimeWorker, ptype_SchoolChild, …) instead.
    Age groups are binary (age_child, age_teen, …), not AGEGROUP ordinal.
    No interaction terms — binary dummies capture heterogeneity directly.
    """
    base    = ['sexp', 'vehrd', 'nprd', 'incp', 'CARLICENCE', 'occp_group', 'HHSIZE']
    age_grp = ['age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
               'age_retired', 'age_young']
    extra   = ['rlhp', 'bikes', 'WFHWEEK', 'WFHDAY',
               'income_hi', 'income_lo', 'income_midplus',
               'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
               'has_vehicle', 'no_vehicle', 'one_vehicle', 'multi_vehicle',
               'hh_couple', 'hh_medium', 'is_female']
    profile = _profile_vars(data)
    abs_    = _abs_person_vars(data)
    ptype   = [c for c in data.columns if c.startswith('ptype_')]
    acc     = ['auto_work_acc', 'auto_shop_acc', 'walk_acc', 'transit_acc',
               'auto_nonmand_acc', 'bike_acc']
    return list(dict.fromkeys(
        c for c in base + age_grp + extra + profile + abs_ + acc + ptype
        + _STAGE_INTERACTION_CANDIDATES
        if c in data.columns
    ))


def _stage2_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 2 (day duration) — activity counts + binary person attributes."""
    counts  = ['n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
               'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
               'n_total', 'n_tours']
    base    = ['sexp', 'vehrd', 'nprd', 'incp', 'CARLICENCE', 'occp_group',
               'HHSIZE', 'WFHWEEK', 'WFHDAY',
               'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
               'income_hi', 'income_lo', 'income_midplus',
               'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
               'has_vehicle', 'no_vehicle', 'one_vehicle', 'multi_vehicle',
               'hh_couple', 'hh_medium']
    profile = _profile_vars(data)
    abs_    = _abs_person_vars(data)
    acc     = ['auto_work_acc', 'auto_nonmand_acc', 'walk_acc', 'transit_acc',
               'auto_shop_acc', 'bike_acc']
    ptype   = [c for c in data.columns if c.startswith('ptype_')]
    return list(dict.fromkeys(
        c for c in counts + base + profile + abs_ + acc + ptype
        + _STAGE_INTERACTION_CANDIDATES
        if c in data.columns
    ))


def _stage3_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 3 (activity duration) — activity context + binary person attributes."""
    act_vars = ['start_hr', 'log_dist', 'B', 'seq_num', 'is_last_act']
    counts   = ['n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
                'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
                'n_total']
    base     = ['sexp', 'vehrd', 'nprd', 'incp', 'CARLICENCE', 'occp_group',
                'HHSIZE', 'age_child', 'age_teen', 'age_older', 'age_senior',
                'income_hi', 'income_lo', 'income_midplus',
                'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
                'has_vehicle', 'no_vehicle', 'one_vehicle', 'multi_vehicle',
                'hh_couple', 'hh_medium']
    profile  = _profile_vars(data)
    abs_     = _abs_person_vars(data)
    ptype    = [c for c in data.columns if c.startswith('ptype_')]
    return list(dict.fromkeys(
        c for c in act_vars + counts + base + profile + abs_ + ptype
        + _STAGE_INTERACTION_CANDIDATES
        if c in data.columns
    ))


def _stage4_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 4 (schedule placement) — timing context + binary person attributes."""
    timing  = ['day_position', 'time_consumed', 'duration_min', 'time_remaining',
               'time_of_day_sin', 'time_of_day_cos']
    base    = ['sexp', 'vehrd', 'nprd', 'CARLICENCE', 'incp', 'occp_group',
               'HHSIZE', 'WFHWEEK',
               'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
               'income_hi', 'income_lo', 'income_midplus',
               'income_verylo', 'income_mid', 'income_midhi', 'income_veryhi',
               'has_vehicle', 'no_vehicle', 'one_vehicle', 'multi_vehicle',
               'hh_couple', 'hh_medium']
    profile = _profile_vars(data)
    abs_    = _abs_person_vars(data)
    purp    = [c for c in data.columns if c.startswith('purp_')]
    ptype   = [c for c in data.columns if c.startswith('ptype_')]
    return list(dict.fromkeys(
        c for c in timing + base + profile + abs_ + purp + ptype
        + _STAGE_INTERACTION_CANDIDATES
        if c in data.columns
    ))


def _stage5_mode_candidates() -> List[str]:
    """
    Candidate variables for Stage 5 (tour-level) and Stage 51 (trip-level) mode choice.

    Split into two sub-pools:
    ALWAYS FIXED (backbone)   kept regardless of GA selection:
        DIST, SOVTOLL_TIME__AM, WLK_LOC_WLK_TOTIVT__AM, DISTWALK, DISTBIKE, CRASH

    TESTED BY GA (additional enrichments around the backbone):
        ÔÇó Alternative time periods (PM, HOV)
        ÔÇó Socio-demographic person attributes (income, age, gender, household,
          vehicle ownership, licence, person-type flags)
        ÔÇó ABS census-derived attributes (mtwp, empp, n_workers_hh)

    ActivitySim tour_mode_choice uses: IVT ├ù purpose, age ├ù mode, hhsize ├ù SR,
    vehicle availability, licence, gender.  Trip_mode_choice additionally uses
    position-in-tour flags (outbound/inbound).  Both are reproduced here.
    """
    # ÔöÇÔöÇ Skim variables (GA tests these enrichments) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    skim_extra = [
        'SOVTOLL_TIME__PM',       # PM period IVT   peak shoulder
        'HOV2TOLL_TIME__AM',      # HOV2 IVT   for shared-ride estimation
        'WLK_LOC_WLK_XWAIT__AM',  # transit excess wait
        'CRASH',                  # safety term (kept available for mandatory inclusion)
    ]
    # ÔöÇÔöÇ Vehicle / licence (key mode-choice predictors) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    veh_vars = ['vehrd', 'no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle',
                'CARLICENCE']
    # ÔöÇÔöÇ Income (affects car and transit share) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    inc_vars = ['incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
                'income_midhi', 'income_hi', 'income_veryhi']
    # ÔöÇÔöÇ Age / gender ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    age_gender = [
        'sexp', 'is_female',
        'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
    ]
    # ÔöÇÔöÇ Household composition ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    hh_vars = ['nprd', 'hh_child', 'hh_alone', 'hh_couple', 'hh_medium',
               'hh_size_large', 'n_workers_hh']
    # ÔöÇÔöÇ Person type (mode captivity) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    ptype_vars = [
        'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
        'ptype_UniversityStudent', 'ptype_WorkingStudent',
        'ptype_SchoolChild', 'ptype_Retired',
    ]
    # ÔöÇÔöÇ ABS census (method of travel to work, employment status) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    abs_vars = ['mtwp', 'empp', 'bikes', 'WFHWEEK', 'wfh',
                'ebikes', 'escooter']    # micromobility ownership (HHTS/synth cols)
    # ÔöÇÔöÇ Home-zone accessibility (attached from landuse at the person's home
    # SA2) ÔöÇ strong mode-captivity signals: high walk/transit accessibility at
    # home shifts mode share away from car regardless of the specific OD.
    acc_origin = ['walk_acc', 'bike_acc', 'transit_acc', 'auto_work_acc',
                  'crash_acc']   # home-zone crash exposure (reviewer: safety upstream of mode utility)
    # WORKSTAT / STUDYING as per-category binary dummies — complementary indicators
    # (a model may select workstat_fulltime AND workstat_parttime simultaneously
    # since FT workers differ from PT workers in mode choice, not just more of
    # the same effect). Built in cdap.build_person_profiles and
    # stage5._ensure_person_behavioural_cols from the raw WORKSTAT/STUDYING codes.
    # mandatory_role (WORKSTAT >= 1, any worker) kept as a broader single binary.
    workstat_vars = ['workstat_fulltime', 'workstat_parttime', 'workstat_notworking']
    studying_vars = ['studying_tertiary_pt', 'studying_tertiary_ft',
                     'studying_secondary', 'studying_primary']
    misc = ['mandatory_role']
    dest_lu_vars = ['is_cbd', 'transit_walk_acc']

    # ── ActivitySim-inspired GA genes (gated in MODE_UEC / _COEFF_GATE) ──────
    # Generic-vs-alternative-specific toggles (ga_constraints.generic_groups):
    #   selecting the *_generic gene applies ONE shared effect across modes
    #   instead of the per-mode block (mutex-guarded against the ASV genes).
    generic_genes = ['incp_generic', 'veh_generic', 'crash_generic',
                     'age_active_generic']
    # Cost / tolls & extra time-of-day IVT periods (mutex: one auto IVT period)
    cost_tod = ['SOVTOLL_TOLL__AM', 'auto_toll', 'auto_cost',
                'SOVTOLL_TIME__MD', 'SOVTOLL_TIME__EV',
                'WLK_LOC_DRV_DTIM__AM']
    # Auto-sufficiency / short-distance active-mode dummies
    captivity_dist = ['auto_deficient', 'short_dist_active']
    # Occupation / job-type ("job_tup") × mode
    occ_genes = ['occp_group', 'occp_mgr', 'occp_trades', 'occp_prof']

    return (skim_extra + veh_vars + inc_vars + age_gender +
            hh_vars + ptype_vars + abs_vars + acc_origin +
            workstat_vars + studying_vars + misc + dest_lu_vars +
            generic_genes + cost_tod + captivity_dist + occ_genes)


def _stage5_dest_candidates() -> List[str]:
    """
    Candidate variables for Stage 5 destination choice (tour and trip level).

    Split into sub-pools:
    ALWAYS FIXED (backbone)   kept regardless of GA selection:
        c_logsum, c_log_dist, c_dist, c_size_term, c_aggl, c_comp, c_crash,
        c_incp_dist, c_veh_dist, key accessibility interactions

    TESTED BY GA (additional enrichments):
        ÔÇó Land-use employment categories by sector
        ÔÇó Education enrollment variables
        ÔÇó Non-mandatory attraction variables (retail, recreation)
        ÔÇó Population and housing density
        ÔÇó Person-by-destination interaction terms (income ├ù employment,
          age ├ù education, gender ├ù recreation)
        ÔÇó ABS demographic context at destination
        ÔÇó Accessibility at destination

    ActivitySim destination choice (non_mandatory_tour_destination) uses:
    size terms (employment, enrollment, retail), distance, logsums,
    demographics at destination.  Same structure reproduced here.
    """
    # sku deterrents (beyond the fixed backbone)
    skim_extra = ['SOVTOLL_TIME__AM']
    # ÔöÇÔöÇ Employment by sector (work tours) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    emp_work = [
        'FPSEMPN', 'HEREMPN', 'RETEMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP',
        'TOTEMP', 'commercial', 'industrial', 'government', 'governmental',
    ]
    # ÔöÇÔöÇ Education enrollment (edu tours) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    emp_edu = [
        'Education', 'Primary', 'Secondary', 'education',
        'HSENROLL', 'AGE0519',
        'Primary_total_priv', 'Secondary_total_priv', 'HEREMPN',
    ]
    # ÔöÇÔöÇ Non-mandatory attraction ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    emp_nonmand = [
        'Shopping', 'ESCORT', 'Entertainment', 'OTHERMAINTENANCE',
        'OTHERDISCRFacilities', 'EATOUT', 'OTHEMP',
        'recreation_ground', 'sport', 'park', 'village_green',
        'community', 'civic', 'civic_admin',
        'Healthcare', 'Public_Services', 'childcare',
    ]
    # ÔöÇÔöÇ Population / area ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    pop_area = ['TOTPOP', 'TOTHH', 'residential', 'retail',
                'construction', 'sa2_area',
                # further OSM/zoning landuse columns present in
                # zone_landuse_507.csv but previously not searchable:
                'institutional', 'logistics', 'communications',
                'airport', 'harbour', 'trees', 'utility']
    # ÔöÇÔöÇ Accessibility at destination (zone-level, not person-level) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    # full set of computed accessibility measures (pipeline_logger
    # accessibility step) - previously only 4 of the 8 were searchable.
    acc_dest = ['auto_work_acc', 'auto_edu_acc', 'auto_shop_acc',
                'auto_nonmand_acc', 'walk_acc', 'bike_acc', 'transit_acc',
                'recreation_acc', 'crash_acc']
    # ÔöÇÔöÇ Person attributes tested as interaction terms with destination ÔöÇÔöÇÔöÇÔöÇ
    # (person-by-destination interactions already in fixed backbone for income,
    # vehicles; GA tests additional demographics)
    person_dest_int = [
        'has_vehicle',    # vehicle ├ù destination   for car-dependent zones
        'incp',           # income ├ù distance (in addition to fixed c_incp_dist)
        'age_working',    # working-age ├ù employment density
        'age_older',      # elderly ├ù healthcare/services
        'hh_child',       # family ├ù education / escort destinations
        'is_female',      # gender ├ù recreation / safety
        'n_workers_hh',   # household workers ├ù employment concentration
        'ptype_UniversityStudent',   # university student x HEREMPN
        'ptype_FullTimeWorker',      # FT worker x TOTEMP
        'income_hi',      # high-income x finance / professional zones
        'income_veryhi',  # very high income x premium destinations
        'income_lo',      # low-income x affordable / community destinations
        'income_mid',     # middle-income x mixed-use zones
    ]
    # -- ACDC themed agglomeration variants (mutex with c_aggl) --
    acdc_themed = ['acdc_aggl_retail', 'acdc_aggl_office',
                   'acdc_aggl_edu', 'acdc_aggl_leisure']
    # -- Home-pull gravity (mutex: home_pull vs home_pull_late) --
    home_pull_vars = ['home_pull', 'home_pull_late']
    # -- Mode-relevant landuse columns (CBD flag, transit walk access) --
    mode_lu_vars = ['is_cbd', 'transit_walk_acc']
    # -- ActivitySim-inspired dest genes (coeff-name gated in DEST_UEC_FLAT) --
    #    piecewise distance spline (replaces c_dist) + CBD/urban-core attractor.
    dist_spline = ['c_dist_0_2', 'c_dist_2_5', 'c_dist_5_10', 'c_cbd_dest']
    return list(dict.fromkeys(
        skim_extra + emp_work + emp_edu + emp_nonmand + pop_area +
        acc_dest + person_dest_int + acdc_themed + home_pull_vars +
        mode_lu_vars + dist_spline + _overlay_dest_interaction_genes()
    ))


def _overlay_dest_interaction_genes(segment: str = None) -> List[str]:
    """ptype×land interaction coefficient names declared in the user constraint
    overlay (constraint_gui.py). Added to the dest candidate pool so the GA can
    select them; the coefficients are built by stage5_destination_choice._dest_tree.
    Filtered to `segment` when the affinity names one."""
    try:
        from stage5_destination_choice import _overlay_dest_interactions
        out = []
        for it in _overlay_dest_interactions():
            segs = it.get('segments')
            if segment is None or not segs or segment in segs:
                out.append(it['coeff'])
        return out
    except Exception:
        return []


def _stage5_mode_candidates_for_segment(segment: str) -> List[str]:
    """Purpose-aware stage 5 mode candidates by segment."""
    base = {
        'SOVTOLL_TIME__PM', 'HOV2TOLL_TIME__AM', 'WLK_LOC_WLK_XWAIT__AM',
        'CRASH',
        'vehrd', 'no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle', 'CARLICENCE',
        'sexp', 'is_female',
        'incp', 'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
        'income_midhi', 'income_hi', 'income_veryhi',
        'age_working', 'age_older', 'nprd', 'hh_child', 'n_workers_hh',
        'hh_couple', 'hh_medium',
        'mandatory_role',
        'is_cbd', 'transit_walk_acc',
        # ActivitySim-inspired genes (all gated in MODE_UEC / _COEFF_GATE)
        'incp_generic', 'veh_generic', 'crash_generic', 'age_active_generic',
        'auto_cost', 'SOVTOLL_TOLL__AM', 'SOVTOLL_TIME__MD', 'SOVTOLL_TIME__EV',
        'WLK_LOC_DRV_DTIM__AM',
        'auto_deficient', 'short_dist_active',
        'occp_group', 'occp_mgr', 'occp_trades', 'occp_prof',
        'safety_sensitivity',   # per-agent crash-aversion x crash (active modes)
    }
    by_seg = {
        # ── Work tours: who you are, household structure, WFH patterns ───────
        # Add to 'work' to test extra mode-utility shifters for commute trips.
        'work': {
            'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
            'workstat_fulltime', 'workstat_parttime',  # finer split within workers
            'mtwp', 'empp', 'wfh', 'WFHWEEK',
            'hh_alone',       # solo worker has full vehicle without sharing
            'hh_size_large',  # large HH may share vehicle → different mode pressure
        },
        # ── Education tours: student type, age, active-travel propensity ─────
        # Add to 'edu' to test extra mode-utility shifters for school/uni trips.
        'edu': {
            'ptype_UniversityStudent', 'ptype_WorkingStudent', 'ptype_SchoolChild',
            'studying_secondary', 'studying_tertiary_pt', 'studying_tertiary_ft',
            'age_child', 'age_teen', 'bikes',
            'hh_alone',       # sole-adult student household; full bike/car availability
        },
        # ── Non-mandatory tours: retirement, bikes, household composition ─────
        # Add to 'nonmand' to test extra mode-utility shifters for leisure trips.
        'nonmand': {
            'ptype_Retired', 'bikes', 'hh_size_large',
            'workstat_notworking',  # non-workers are more transit/walk-captive
            'age_senior',
            'empp',           # employment status affects discretionary vehicle availability
            'hh_alone',       # solo HH has full vehicle access; no sharing constraints
        },
    }
    return list(dict.fromkeys(sorted(base | by_seg.get(segment, set()))))


def _stage5_dest_candidates_for_segment(segment: str) -> List[str]:
    """Purpose-aware stage 5 destination candidates by segment."""
    common = {
        'SOVTOLL_TIME__AM',
        'c_crash',
        'TOTPOP', 'TOTHH', 'sa2_area',
        'auto_work_acc', 'auto_shop_acc', 'walk_acc', 'transit_acc',
        'incp', 'has_vehicle', 'n_workers_hh', 'is_female',
        'acdc_aggl_retail', 'acdc_aggl_office',
        'acdc_aggl_edu', 'acdc_aggl_leisure',
        # ActivitySim-inspired dest genes (gated by coeff-name in DEST_UEC_FLAT):
        #   piecewise distance spline (replaces c_dist) + CBD/urban-core pull.
        'c_dist_0_2', 'c_dist_2_5', 'c_dist_5_10', 'c_cbd_dest', 'is_cbd',
        'c_dist_decay', 'c_dist_decay_slow',   # non-linear gravity distance-decay
        'c_crash_safety_dest',   # per-agent safety utility x destination crash
    }
    by_seg = {
        'work': {
            'TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN', 'AGREMPN', 'MWTEMPN',
            'commercial', 'industrial', 'government', 'governmental',
            'ptype_FullTimeWorker', 'income_hi',
            # additional industry-sector attractor coefficients (coeff-name gated)
            'c_indp_agri', 'c_indp_mfg', 'c_indp_construction', 'c_indp_govt',
            'c_indp_industrial', 'c_indp_education',
        },
        'edu': {
            'Education', 'Primary', 'Secondary', 'HSENROLL', 'AGE0519',
            'Primary_total_priv', 'Secondary_total_priv',
            'ptype_UniversityStudent', 'hh_child', 'age_working',
        },
        'nonmand': {
            'Shopping', 'Entertainment', 'OTHERMAINTENANCE', 'OTHERDISCRFacilities',
            'EATOUT', 'OTHEMP', 'Healthcare', 'Public_Services', 'childcare',
            'recreation_ground', 'sport', 'park', 'community', 'civic',
            'residential', 'retail', 'construction',
            'age_older',
            'home_pull', 'home_pull_late',
        },
    }
    return list(dict.fromkeys(sorted(common | by_seg.get(segment, set()))
                              + _overlay_dest_interaction_genes(segment)))


def _stage6_candidates(data: pd.DataFrame) -> List[str]:
    # Core skim + cost variables
    base_skim = ['tt_avg', 'tt_am', 'log_dist', 'dist', 'distwalk', 'distbike',
                 'ldw', 'ldb', 'crash', 'cost_proxy', 'accessibility']
    # Vehicle / licence
    base_veh  = ['has_vehicle', 'no_vehicle', 'vehrd', 'CARLICENCE']
    # Person demographics   richer heterogeneity per ATRF2022 / ActivitySim
    base_pers = ['sexp', 'female', 'incp', 'inc_hi', 'inc_lo',
                 'age5p', 'young', 'older',
                 'worker', 'student',
                 'start_hr', 'nprd']
    # Legacy columns from previous spec
    base_leg  = ['occp_group', 'is_female', 'income_midplus', 'age_norm']

    abs_      = _abs_person_vars(data)
    all_cands = base_skim + base_veh + base_pers + base_leg + abs_
    return [c for c in dict.fromkeys(all_cands) if c in data.columns]

# ---------------------------------------------------------------------------
# 4.  PER-STAGE GA RUNNERS  (estimation only)
# ---------------------------------------------------------------------------

def _run_ga_cdap(persons, ga_kwargs, n_restarts=2) -> GAResult:
    """
    GA / SA / Bandit for CDAP feature selection.

    Fitness = -BIC/n of the 3-alt MNL (H/N/M) on HHTS persons.
    Uses _make_searcher so the bandit is active when GA_USE_BANDIT_SA=1.
    Estimation only — no prediction over synth pop.
    """
    from cdap import prepare_cdap_data, build_person_profiles

    print("\n[CDAP] Preparing estimation data from HHTS persons...")
    data       = prepare_cdap_data(persons)
    feat_names = _cdap_candidates(data)
    print(f"  {len(data)} HHTS persons | {len(feat_names)} candidate features")
    print(f"  CDAP dist: {data['cdap_obs'].value_counts().sort_index().to_dict()}")
    print(f"  Candidates: {feat_names}")

    if not feat_names:
        raise RuntimeError("CDAP GA: no candidate features found in data.")

    def fitness(genes):
        return _cdap_fitness(genes, data, feat_names)

    searcher = _make_searcher(
        len(feat_names), fitness, ga_kwargs,
        feature_names       = feat_names,
        collinear_groups    = STAGE_COLLINEAR_GROUPS.get(0, []),
        require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get(0, []),
        fitness_metric      = 'mean(-BIC/n) [Larch MNL]',
    )
    best, history, pop = searcher.run_with_restarts(n_restarts=n_restarts)
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=0, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='mean(-BIC/n) [Larch MNL]',
        n_restarts=n_restarts, cache_hits=searcher.cache_hits,
    )


def _run_ga_usual_loc(persons, landuse, skims, ga_kwargs, n_restarts=2) -> GAResult:
    """
    GA for usual location model parameter selection.

    Chromosome layout (mixed binary + continuous):
        gene[0]       : work_decay_b    continuous in [0.01, 0.20]
        gene[1]       : school_decay_b   continuous in [0.01, 0.20]
        genes[2:]     : binary   which attractor columns to include
                        [work_att_cols... | edu_att_cols...]

    Fitness = mean log-probability of observed work/school zone.

    Observed work/school zones come from the HHTS person table.
    We look for WORK_TAZ / SCHOOL_TAZ columns, or derive from WORKZONE
    and SCHOOLZONE if present.
    """
    import openmatrix as omx
    from stage5_destination_choice import _skim_flat

    print("\n[UsualLoc] Preparing HHTS work/school observed zones...")

    # Build skim flat table
    close_skims = False
    if isinstance(skims, str):
        skims       = omx.open_file(skims, 'r')
        close_skims = True
    try:
        skim_flat = _skim_flat(skims)
    except Exception:
        skim_flat = pd.DataFrame()
    if close_skims:
        skims.close()

    # Extract observed usual work / school zones from HHTS
    obs_work   = pd.DataFrame()
    obs_school = pd.DataFrame()

    # Common column names across different HHTS formats
    for wzone_col in ['WORK_TAZ', 'WORKZONE', 'WORK_ZONE', 'wtaz']:
        if wzone_col in persons.columns:
            otaz_col = 'OTAZ' if 'OTAZ' in persons.columns else 'ORDER'
            obs_work = persons[persons[wzone_col].fillna(0).astype(int) > 0][
                [otaz_col, wzone_col]].rename(columns={wzone_col: 'WORK_TAZ',
                                                        otaz_col: 'OTAZ'})
            break

    for szone_col in ['SCHOOL_TAZ', 'SCHOOLZONE', 'SCHOOL_ZONE', 'staz']:
        if szone_col in persons.columns:
            otaz_col = 'OTAZ' if 'OTAZ' in persons.columns else 'ORDER'
            obs_school = persons[persons[szone_col].fillna(0).astype(int) > 0][
                [otaz_col, szone_col]].rename(columns={szone_col: 'SCHOOL_TAZ',
                                                         otaz_col: 'OTAZ'})
            break

    if obs_work.empty and obs_school.empty:
        raise RuntimeError(
            "UsualLoc GA: no observed work/school zones found in persons table.\n"
            "  Need WORK_TAZ/WORKZONE or SCHOOL_TAZ/SCHOOLZONE columns in HHTS persons."
        )

    print(f"  Observed work zones:   {len(obs_work)}")
    print(f"  Observed school zones: {len(obs_school)}")

    work_att_names = _usual_loc_work_attractor_candidates()
    edu_att_names  = _usual_loc_edu_attractor_candidates()
    n_work_att     = len(work_att_names)
    n_edu_att      = len(edu_att_names)
    n_total        = 2 + n_work_att + n_edu_att   # 2 continuous decay + binary flags

    feat_names = (['work_decay_b', 'school_decay_b'] +
                  [f'work_att:{c}' for c in work_att_names] +
                  [f'edu_att:{c}' for c in edu_att_names])

    def fitness(genes):
        # Map binary genes [0,1] to continuous [0.01, 0.20] for decay params
        g = genes.copy().astype(float)
        g[0] = 0.01 + g[0] * 0.19    # work_decay_b
        g[1] = 0.01 + g[1] * 0.19    # school_decay_b
        return _usual_loc_fitness(
            g, obs_work, obs_school, landuse, skim_flat,
            work_att_names, edu_att_names, n_work_att
        )

    # Use min_features=3 (2 decay + at least 1 attractor)
    kw = {**ga_kwargs, 'min_features': 3}
    ga  = GeneticAlgorithm(
        n_total, fitness,
        feature_names       = feat_names,
        collinear_groups    = STAGE_COLLINEAR_GROUPS.get(-1, []),
        require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get(-1, []),
        **kw,
    )
    best, history, pop = ga.run_with_restarts(n_restarts=n_restarts)

    # Decode best solution
    g         = best.genes.astype(float)
    best_work_b   = float(0.01 + g[0] * 0.19)
    best_school_b = float(0.01 + g[1] * 0.19)
    best_work_att = [c for c, f in zip(work_att_names, g[2:2+n_work_att]) if f >= 0.5]
    best_edu_att  = [c for c, f in zip(edu_att_names,  g[2+n_work_att:])  if f >= 0.5]

    best_features = {
        'work_decay_b':   best_work_b,
        'school_decay_b': best_school_b,
        'work_attractors': best_work_att,
        'edu_attractors':  best_edu_att,
    }
    print(f"\n  Best work_decay_b={best_work_b:.3f}  school_decay_b={best_school_b:.3f}")
    print(f"  Best work attractors: {best_work_att}")
    print(f"  Best edu  attractors: {best_edu_att}")

    return GAResult(
        stage=-1, best_features=best_features,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='mean-log-P(zone)',
        n_restarts=n_restarts, cache_hits=ga.cache_hits,
    )


def _majority_vote_result(segment_results, purp_best_genes, feat_names,
                          agg_fn, stage, fitness_metric, n_restarts) -> GAResult:
    """
    Compute majority-vote consensus chromosome from per-purpose GA results,
    re-evaluate aggregate fitness, and build a combined GAResult whose history
    uses a single synthetic entry for the aggregate (so convergence plots of
    the combined result don't show inflated per-segment values).
    """
    if purp_best_genes:
        vote_matrix = np.array(list(purp_best_genes.values()))
        vote_frac   = vote_matrix.mean(axis=0)
        consensus   = (vote_frac >= 0.5).astype(int)
        consensus_ind         = Individual(genes=consensus)
        consensus_ind.fitness = agg_fn(consensus)
        best_feats  = [f for f, g in zip(feat_names, consensus) if g == 1]
        agg_cache   = sum(r.cache_hits for r in segment_results.values())
    else:
        consensus_ind = Individual(genes=np.zeros(len(feat_names), dtype=int))
        consensus_ind.fitness = -1e9
        best_feats    = []
        agg_cache     = 0

    # Synthetic single-entry history so the combined plot is not misleading
    agg_history = [{
        'generation':    0,
        'best_fitness':  consensus_ind.fitness,
        'best_so_far':   consensus_ind.fitness,
        'mean_fitness':  consensus_ind.fitness,
        'worst_fitness': consensus_ind.fitness,
        'diversity':     0.0,
        'mutation_rate': 0.0,
        'n_features':    int(consensus_ind.genes.sum()),
        'stagnation':    0,
    }]
    best_seg    = max(segment_results.values(), key=lambda r: r.best_fitness)
    agg_pop     = best_seg.all_candidates

    combined = GAResult(
        stage          = stage,
        best_features  = best_feats,
        best_fitness   = consensus_ind.fitness,
        best_individual= consensus_ind,
        history        = agg_history,
        all_candidates = agg_pop,
        feature_names  = feat_names,
        fitness_metric = fitness_metric,
        n_restarts     = n_restarts,
        cache_hits     = agg_cache,
    )
    combined.metadata['segment_results'] = segment_results
    return combined


# ÔöÇÔöÇ Purpose label map used by all per-purpose runners 
_PURP_LABELS = {
    'work':     'Work',
    'edu':      'School/Edu',
    'social':   'Social',
    'escort':   'Escort',
    'shopping': 'Shopping',
    'othmaint': 'Oth-Maint',
    'othdiscr': 'Oth-Discr',
    'eat':      'Eat Out',
    'nonmand':  'Non-Mandatory (combined)',
}


def _resolve_min_features(ga_kwargs: dict, feat_names: list, fixed_feats: list) -> dict:
    """
    Return a copy of ga_kwargs with min_features adjusted so the GA always has
    at least 2 optional (non-fixed) features it can toggle on/off.

    Without this guard, very small feature spaces (e.g. shopping with 5-6
    total features and 4-5 fixed core features) collapse to 1-2 valid
    chromosomes → population diversity = 0 → no improvement possible.
    """
    kw = dict(ga_kwargs)
    n_total   = len(feat_names)
    n_fixed   = len(fixed_feats)
    n_optional = max(0, n_total - n_fixed)
    user_min  = int(kw.get('min_features', 3))

    if n_total > n_fixed:
        # Ensure at least 2 optional features can freely toggle: cap at n_total - 2
        # but never below n_fixed (all core features must be active).
        effective_min = min(
            max(user_min, n_fixed),
            max(n_fixed, n_total - max(2, n_optional // 2)),
        )
        kw['min_features'] = min(effective_min, n_total - 1)
    else:
        kw['min_features'] = n_total
    return kw


def _enrich_ga_data(data: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    """Merge additional person-level columns into GA estimation data.

    Stages 1-4 use a narrow PERSON_FEATURES list.  This helper adds columns
    that the GA optional feature specs reference but are not merged by default,
    so the search has a meaningful number of toggleable features.

    Always added (derived from PERSON_FEATURES that are already in data):
      * has_vehicle / no_vehicle / one_vehicle / multi_vehicle  (from vehrd, if present)
      * hh_couple / hh_medium  (from nprd, alongside hh_alone / hh_size_large)

    Merged from `persons` if not already in data (one pass, drop_duplicates):
      * CARLICENCE, hh_child, n_workers_hh, AGEGROUP, wfh, WFHWEEK, incp

    Derived from newly-merged columns:
      * income_hi / income_lo / income_midplus / income_verylo / income_mid
        / income_midhi / income_veryhi  (from incp, if newly added)
      * age_child / age_teen / age_working / age_older / age_senior  (from AGEGROUP)
    """
    # ── derived from vehrd (already in data) ──────────────────────────────
    if 'vehrd' in data.columns:
        _need_copy = (
            'has_vehicle'   not in data.columns or 'no_vehicle'    not in data.columns or
            'one_vehicle'   not in data.columns or 'multi_vehicle' not in data.columns
        )
        if _need_copy:
            data = data.copy()
        if 'has_vehicle' not in data.columns:
            data['has_vehicle']   = (data['vehrd'] > 0).astype(int)
        if 'no_vehicle' not in data.columns:
            data['no_vehicle']    = (data['vehrd'] == 0).astype(int)
        if 'one_vehicle' not in data.columns:
            data['one_vehicle']   = (data['vehrd'] == 1).astype(int)
        if 'multi_vehicle' not in data.columns:
            data['multi_vehicle'] = (data['vehrd'] >= 2).astype(int)

    # ── merge extra columns from persons ──────────────────────────────────
    _extra_cols = [c for c in ['CARLICENCE', 'hh_child', 'n_workers_hh',
                                'AGEGROUP', 'wfh', 'WFHWEEK', 'incp']
                   if c in persons.columns and c not in data.columns]
    if _extra_cols and 'person_id' in data.columns and 'person_id' in persons.columns:
        _pers_extra = (
            persons[['person_id'] + _extra_cols]
            .drop_duplicates('person_id')
        )
        data = data.merge(_pers_extra, on='person_id', how='left')

    # ── derive income flags from incp (may have been just merged) ─────────
    if 'incp' in data.columns:
        if 'income_hi'      not in data.columns:
            data['income_hi']      = (data['incp'] >= 12).astype(int)
        if 'income_lo'      not in data.columns:
            data['income_lo']      = (data['incp'] <= 4).astype(int)
        if 'income_midplus' not in data.columns:
            data['income_midplus'] = (data['incp'] >= 8).astype(int)
        # Granular income bands (ABS HHTS weekly income codes; 1=nil, 14=$3000+/wk)
        if 'income_verylo'  not in data.columns:
            data['income_verylo']  = (data['incp'] <= 2).astype(int)          # <$8k pa
        if 'income_mid'     not in data.columns:
            data['income_mid']     = data['incp'].between(5, 7).astype(int)   # $21-42k pa
        if 'income_midhi'   not in data.columns:
            data['income_midhi']   = data['incp'].between(8, 11).astype(int)  # $42-91k pa
        if 'income_veryhi'  not in data.columns:
            data['income_veryhi']  = (data['incp'] >= 13).astype(int)         # $104k+ pa

    # ── derive age-group dummies from AGEGROUP ────────────────────────────
    if 'AGEGROUP' in data.columns:
        if 'age_child'   not in data.columns:
            data['age_child']   = (data['AGEGROUP'] <= 2).astype(int)
        if 'age_teen'    not in data.columns:
            data['age_teen']    = (data['AGEGROUP'] == 3).astype(int)
        if 'age_working' not in data.columns:
            data['age_working'] = data['AGEGROUP'].between(4, 7).astype(int)
        if 'age_older'   not in data.columns:
            data['age_older']   = (data['AGEGROUP'] >= 8).astype(int)
        if 'age_senior'  not in data.columns:
            data['age_senior']  = (data['AGEGROUP'] >= 10).astype(int)

    # ── derive household size flags from nprd ─────────────────────────────
    # hh_alone    : single-person household (sole adult, fewer time constraints)
    # hh_couple   : two-person household (couple without young children)
    # hh_medium   : three-person household (small family)
    # hh_size_large: 4+ person household (shared schedules, more escort demand)
    if 'nprd' in data.columns:
        if 'hh_alone' not in data.columns:
            data['hh_alone']      = (data['nprd'] == 1).astype(int)
        if 'hh_size_large' not in data.columns:
            data['hh_size_large'] = (data['nprd'] >= 4).astype(int)
        if 'hh_couple' not in data.columns:
            data['hh_couple']     = (data['nprd'] == 2).astype(int)
        if 'hh_medium' not in data.columns:
            data['hh_medium']     = (data['nprd'] == 3).astype(int)

    data = _ensure_stage_interactions(data)
    return data


def _ensure_stage_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add ActivitySim-style interaction ("perturbation character") columns for the
    stage 1-4 GA candidate pools, mirroring the alternative-specific enrichment
    added to stage 5. Each interaction is created ONLY when its base columns are
    already present, so the same helper is safe across every stage's dataframe
    (a candidate gene whose column is absent is silently dropped by the
    `c in data.columns` filter in the _stageN_candidates functions).

    Families (matching the user-selected enrichment set):
      A. accessibility/density x person   (captivity-conditioned accessibility)
      B. person-type / lifecycle          (children x age, retired x vehicles, ...)
      C. occupation / industry & WFH      (job-type and work-from-home effects)
      D. activity-count / timing          (added where those base columns exist)
    """
    d = data.copy()   # never mutate a caller-shared frame

    def _set(col, series):
        if col not in d.columns:
            d[col] = np.asarray(series).astype(np.float32)

    def has(*cols):
        return all(c in d.columns for c in cols)

    # ── A. Accessibility x captivity / income ────────────────────────────────
    # Home-zone accessibility matters more for car-less / lower-income persons.
    _noveh = (d['vehrd'] <= 0).astype(float) if 'vehrd' in d.columns else None
    for _acc in ('walk_acc', 'transit_acc', 'auto_work_acc', 'auto_shop_acc',
                 'auto_nonmand_acc', 'bike_acc'):
        if _acc not in d.columns:
            continue
        if _noveh is not None:
            _set(f'noveh_x_{_acc}', _noveh * d[_acc].astype(float))
        if 'incp' in d.columns:
            _set(f'incp_x_{_acc}', d['incp'].astype(float) * d[_acc].astype(float))

    # ── B. Person-type / lifecycle interactions ──────────────────────────────
    if has('hh_child', 'age_working'):
        _set('child_x_working', d['hh_child'].astype(float) * d['age_working'].astype(float))
    if has('hh_child') and 'is_female' in d.columns:
        _set('child_x_female', d['hh_child'].astype(float) * d['is_female'].astype(float))
    if has('age_older', 'no_vehicle'):
        _set('older_x_noveh', d['age_older'].astype(float) * d['no_vehicle'].astype(float))
    if has('age_senior', 'has_vehicle'):
        _set('senior_x_veh', d['age_senior'].astype(float) * d['has_vehicle'].astype(float))
    if 'ptype_Retired' in d.columns and 'has_vehicle' in d.columns:
        _set('retired_x_veh', d['ptype_Retired'].astype(float) * d['has_vehicle'].astype(float))

    # ── C. Occupation / industry & work-from-home ────────────────────────────
    _wfh = None
    for _wc in ('wfh', 'WFHWEEK', 'WFHDAY'):
        if _wc in d.columns:
            _wfh = (d[_wc].astype(float) > 0).astype(float)
            break
    if _wfh is not None:
        _set('wfh_flag', _wfh)
        if 'mandatory_role' in d.columns:
            _set('wfh_x_worker', _wfh * d['mandatory_role'].astype(float))
        if 'income_hi' in d.columns:
            _set('wfh_x_inchi', _wfh * d['income_hi'].astype(float))
    if 'occp_group' in d.columns:
        _oc = d['occp_group'].astype(float)
        _set('occ_whitecollar', (_oc == 1).astype(float))
        _set('occ_bluecollar',  (_oc >= 4).astype(float))
        if 'incp' in d.columns:
            _set('occ_x_incp', _oc * d['incp'].astype(float))

    # ── D. Activity-count / timing interactions (stage 2/3/4) ────────────────
    # Only fire where the stage actually carries these base columns.
    if 'n_total' in d.columns and 'has_vehicle' in d.columns:
        _set('ntotal_x_veh', d['n_total'].astype(float) * d['has_vehicle'].astype(float))
    if 'n_tours' in d.columns and 'mandatory_role' in d.columns:
        _set('ntours_x_worker', d['n_tours'].astype(float) * d['mandatory_role'].astype(float))
    if 'time_remaining' in d.columns and 'mandatory_role' in d.columns:
        _set('timeremain_x_worker', d['time_remaining'].astype(float) * d['mandatory_role'].astype(float))
    if 'day_position' in d.columns and 'hh_child' in d.columns:
        _set('daypos_x_child', d['day_position'].astype(float) * d['hh_child'].astype(float))

    # ── E. Extra search options: mobility / affordability / lifestage ────────
    if has('age_working', 'auto_work_acc'):
        _set('worker_x_autoacc', d['age_working'].astype(float) * d['auto_work_acc'].astype(float))
    if has('age_working', 'transit_acc'):
        _set('worker_x_transitacc', d['age_working'].astype(float) * d['transit_acc'].astype(float))
    if has('hh_child', 'walk_acc'):
        _set('child_x_walkacc', d['hh_child'].astype(float) * d['walk_acc'].astype(float))
    if 'incp' in d.columns and 'has_vehicle' in d.columns:
        _set('incp_x_veh', d['incp'].astype(float) * d['has_vehicle'].astype(float))
    if 'incp' in d.columns and 'nprd' in d.columns:
        _set('incp_per_person', d['incp'].astype(float) / (d['nprd'].astype(float) + 1.0))
    if has('n_workers_hh', 'has_vehicle'):
        _set('workers_x_veh', d['n_workers_hh'].astype(float) * d['has_vehicle'].astype(float))
    if 'age_senior' in d.columns and 'transit_acc' in d.columns:
        _set('senior_x_transitacc', d['age_senior'].astype(float) * d['transit_acc'].astype(float))

    return d


def _run_ga_stage1(tours, persons, ga_kwargs, n_restarts=2, landuse=None) -> GAResult:
    from stage1_activity_frequency import prepare_frequency_data, PURPOSES, assign_ptype_column
    from ga_external_estimators import ga_estimator, stage1_fitness_metacount, reset_feature_weakness
    reset_feature_weakness()   # clear weakness scores from any prior stage

    print("\n[Stage 1] Preparing estimation data...")
    data       = prepare_frequency_data(tours, persons, landuse=landuse)
    # Feature engineering: create binary columns referenced by optional feature specs
    if 'ptype' in data.columns:
        _dummies = pd.get_dummies(data['ptype'], prefix='ptype')
        for _c in _dummies.columns:
            if _c not in data.columns:
                data[_c] = _dummies[_c].astype(int)
    if 'sexp' in data.columns and 'is_female' not in data.columns:
        data['is_female'] = (data['sexp'] == 2).astype(int)
    if 'incp' in data.columns:
        if 'income_hi'      not in data.columns: data['income_hi']      = (data['incp'] >= 12).astype(int)
        if 'income_lo'      not in data.columns: data['income_lo']      = (data['incp'] <= 4).astype(int)
        if 'income_midplus' not in data.columns: data['income_midplus'] = (data['incp'] >= 8).astype(int)
    if 'WORKSTAT' in data.columns and 'mandatory_role' not in data.columns:
        data['mandatory_role'] = (data['WORKSTAT'] >= 1).astype(int)
        #breakdown workstat by _binary
    data = _enrich_ga_data(data, persons)

    # Build combined non-mandatory count for the consolidated model.
    # Uses work/edu/nonmand structure matching Stage 5 — all six nonmand purposes
    # (escort, shopping, social, othmaint, othdiscr, eat) pooled into one count.
    from stage1_activity_frequency import NONMAND_PURPOSES as _NONMAND_PURPOSES
    _nonmand_cols = [f'n_{p}_out' for p in _NONMAND_PURPOSES if f'n_{p}_out' in data.columns]
    if _nonmand_cols and 'n_nonmand_out' not in data.columns:
        data['n_nonmand_out'] = data[_nonmand_cols].sum(axis=1).clip(upper=18)

    from stage1_activity_frequency import VALID_PTYPES_FOR_PURPOSE as _VALID_PTYPES
    _all_ptypes = data['ptype'].dropna().unique().tolist()
    _all_purps  = [p for p in PURPOSES if f'n_{p}_out' in data.columns]
    _nonmand_set = set(_NONMAND_PURPOSES)
    # Replace all six nonmand purposes with one combined 'nonmand' purpose → work/edu/nonmand
    purposes   = ([p for p in _all_purps if p not in _nonmand_set]
                  + (['nonmand'] if 'n_nonmand_out' in data.columns else list(_nonmand_set & set(_all_purps))))
    est        = ga_estimator()
    use_sim_anneal = est in ('metacount', 'both', 'hybrid', 'sa_bandit')

    # Purpose-specific ptype lists for the GA fitness function — mirrors the
    # filtering applied in estimate_frequency_models().
    def _ptypes_for(purpose: str):
        valid = _VALID_PTYPES.get(purpose)
        if valid is None:
            return _all_ptypes
        return [p for p in _all_ptypes if p in valid]

    ptypes = _all_ptypes  # fallback for any code that still references the flat list

    print(f"  {len(data)} persons | purpose-specific candidate sets | "
          f"{len(purposes)} purposes (work/edu/nonmand)  {len(_all_ptypes)} ptypes | estimator={est}")

    # Never fall back to statsmodels when a proper estimator is selected:
    # mixing model families (NB2 vs ordered-logit) mid-search would make BIC
    # values non-comparable across iterations.
    fallback_to_default = (
        os.environ.get('GA_ESTIMATOR_FALLBACK', '0') == '1'
        and est not in ('metacount', 'searchlibrium', 'both', 'hybrid')
    )
    fallback_warned = set()
    mc_state = {'enabled': est in ('metacount', 'both', 'hybrid', 'sa_bandit'), 'calls': 0, 'fails': 0, 'warned': False}

    fitness_metric = (
        'mean(-BIC/n) [MetaCount NB2]'
        if est in ('metacount', 'both', 'hybrid', 'sa_bandit')
        else 'mean(-BIC/n)'
    )

    segment_results: dict = {}
    purp_best_feats: dict = {}

    all_feat_names: set = set()

    # ── Per-purpose checkpoint (save after each purpose, reload on resume) ──
    _ckpt_dir  = os.environ.get('GA_CHECKPOINT_DIR', None)
    _ckpt_file = os.path.join(_ckpt_dir, 'ga_stage1_purposes.pkl') if _ckpt_dir else None
    _completed_purposes: dict = {}
    if _ckpt_file and os.path.exists(_ckpt_file):
        try:
            with open(_ckpt_file, 'rb') as _f:
                _completed_purposes = pickle.load(_f)
            print(f"  Resuming Stage 1: found {len(_completed_purposes)} completed purpose(s) in checkpoint")
        except Exception as _e:
            print(f"  Warning: could not load Stage 1 purpose checkpoint: {_e}")
            _completed_purposes = {}

    _pending = []
    for purpose in purposes:
        purp_label = f"Stage1   {_PURP_LABELS.get(purpose, purpose)}"

        # Resume: skip already-completed purposes
        if purp_label in _completed_purposes:
            print(f"\n  [{purp_label}] Skipping (loaded from per-purpose checkpoint)")
            _ck = _completed_purposes[purp_label]
            segment_results[purp_label] = _ck['seg_result']
            purp_best_feats[purpose]    = _ck['feats']
            all_feat_names.update(_ck.get('feat_names', []))
            continue

        feat_names, fixed_feats = _stage1_feature_spec(data, purpose)
        feat_names = _filter_candidates_by_signal(data, feat_names)
        fixed_feats = [f for f in fixed_feats if f in feat_names]
        if not feat_names:
            print(f"    ! {purp_label}: no usable candidates after filtering, skipping")
            continue
        all_feat_names.update(feat_names)

        _purpose_ptypes = _ptypes_for(purpose)
        _pending.append((purpose, purp_label, {
            'label':               purp_label,
            'stage_id':            1,
            'data':                data,
            'feat_names':          feat_names,
            'fixed_feats':         fixed_feats,
            'ga_kwargs':           ga_kwargs,
            'n_restarts':          n_restarts,
            'use_sim_anneal':      use_sim_anneal,
            'fitness_metric':      fitness_metric,
            'collinear_groups':    STAGE_COLLINEAR_GROUPS.get(1, []),
            'require_one_groups':  STAGE_REQUIRE_ONE_GROUPS.get(1, []),
            'fallback_to_default': fallback_to_default,
            'feature_map_key':     purpose,
            'default_fit':         ('ga_hypothesis_testing', '_stage1_fitness_aggregate', ([purpose], _purpose_ptypes)),
            'mc_fit': (
                ('ga_external_estimators', 'stage1_fitness_metacount', ([purpose],), {'use_sim_anneal': use_sim_anneal})
                if mc_state['enabled'] else None
            ),
            'mc_fail_msg':     "    ! Stage1 MetaCount repeatedly failing; disabling MetaCount for remaining evaluations",
            'mc_fallback_msg': f"    ! Stage1 MetaCount failed for {purpose}; using default ordered-logit fallback",
        }))

    if _pending:
        _n_workers = _ga_max_workers(len(_pending))
        print(f"\n  Stage 1: running {len(_pending)} purpose(s) "
              f"({'parallel, workers=' + str(_n_workers) if _n_workers > 1 else 'serial'}) ...")
        _plog(f"[Stage 1] launching {len(_pending)} purpose(s), workers={_n_workers}")
        for i, result in _run_segments_parallel([p for *_, p in _pending], _n_workers):
            purpose, purp_label, _ = _pending[i]
            seg_result = result['seg_result']
            purp_feats = result['feats']
            segment_results[purp_label] = seg_result
            purp_best_feats[purpose]    = purp_feats
            _plog(f"[Stage 1] Done: {purp_label}  BIC/n={-result['best_fitness']:.4f}  "
                  f"feats={len(purp_feats)}  elapsed={result['elapsed']/60:.1f}min")

            # Save per-purpose checkpoint so a walltime-killed job can resume here
            if _ckpt_file:
                _feat_names_ck = (list(seg_result.feature_names)
                                  if hasattr(seg_result, 'feature_names') else [])
                _completed_purposes[purp_label] = {
                    'seg_result': seg_result,
                    'feats':      purp_feats,
                    'feat_names': _feat_names_ck,
                }
                try:
                    with open(_ckpt_file, 'wb') as _f:
                        pickle.dump(_completed_purposes, _f)
                    print(f"    Stage 1 purpose checkpoint saved "
                          f"({len(_completed_purposes)}/{len(purposes)} purposes)")
                    _plog(f"[Stage 1] checkpoint: {len(_completed_purposes)}/{len(purposes)} purposes done")
                except Exception as _e:
                    print(f"    Warning: could not save Stage 1 purpose checkpoint: {_e}")

    # Expand the combined 'discr' result back to the four individual purpose keys
    # Expand 'nonmand' features back to all six individual nonmand purposes so
    # downstream code (estimate_frequency_models, pipeline_logger) can apply
    # the GA-selected features to each purpose's final model.
    if 'nonmand' in purp_best_feats:
        _nonmand_feats = purp_best_feats.pop('nonmand')
        for _np in _NONMAND_PURPOSES:
            if _np in _all_purps:
                purp_best_feats[_np] = _nonmand_feats

    return _combine_segment_feature_results(
        stage=1,
        segment_results=segment_results,
        feature_map=purp_best_feats,
        fitness_metric=fitness_metric,
        n_restarts=n_restarts,
    )


def _run_ga_stage2(tours, persons, ga_kwargs, n_restarts=2) -> GAResult:
    from stage2_day_duration import aggregate_person_day
    from ga_external_estimators import ga_estimator, reset_feature_weakness
    reset_feature_weakness()   # clear weakness scores from any prior stage

    print("\n[Stage 2] Preparing estimation data...")
    data       = aggregate_person_day(tours, persons)
    # Feature engineering: create binary columns referenced by optional feature specs
    if 'ptype' in data.columns:
        _dummies = pd.get_dummies(data['ptype'], prefix='ptype')
        for _c in _dummies.columns:
            if _c not in data.columns:
                data[_c] = _dummies[_c].astype(int)
    if 'sexp' in data.columns and 'is_female' not in data.columns:
        data['is_female'] = (data['sexp'] == 2).astype(int)
    if 'incp' in data.columns:
        if 'income_hi'      not in data.columns: data['income_hi']      = (data['incp'] >= 12).astype(int)
        if 'income_lo'      not in data.columns: data['income_lo']      = (data['incp'] <= 4).astype(int)
        if 'income_midplus' not in data.columns: data['income_midplus'] = (data['incp'] >= 8).astype(int)
    if 'WORKSTAT' in data.columns and 'mandatory_role' not in data.columns:
        data['mandatory_role'] = (data['WORKSTAT'] >= 1).astype(int)
    data = _enrich_ga_data(data, persons)
    targets    = [t for t in ['start_of_day', 'end_of_day', 'B'] if t in data.columns]
    est        = ga_estimator()
    use_sim_anneal = est in ('metacount', 'both', 'hybrid', 'sa_bandit')

    print(f"  {len(data)} person-days | target-specific feature sets | "
          f"targets: {targets} | estimator={est}")

    _TARGET_LABELS = {
        'start_of_day': 'Stage2   Start-of-Day',
        'end_of_day':   'Stage2   End-of-Day',
        'B':            'Stage2   Day Duration (B)',
    }

    fallback_to_default = (
        os.environ.get('GA_ESTIMATOR_FALLBACK', '0') == '1'
        and est not in ('metacount', 'searchlibrium', 'both', 'hybrid')
    )
    mc_state = {'enabled': est in ('metacount', 'both', 'hybrid', 'sa_bandit')}

    if est in ('metacount', 'both', 'hybrid', 'sa_bandit'):
        fitness_metric = 'mean(-BIC/n) [MetaCount Tobit/lognormal]'
    else:
        fitness_metric = 'mean(-BIC/n)'

    segment_results: dict = {}
    tgt_best_feats:  dict = {}

    _target_key_map = {
        'start_of_day': 'start',
        'end_of_day': 'end',
        'B': 'budget',
    }

    # ── Per-target checkpoint ──────────────────────────────────────────────
    _ckpt_dir  = os.environ.get('GA_CHECKPOINT_DIR', None)
    _ckpt_file = os.path.join(_ckpt_dir, 'ga_stage2_targets.pkl') if _ckpt_dir else None
    _completed_targets: dict = {}
    if _ckpt_file and os.path.exists(_ckpt_file):
        try:
            with open(_ckpt_file, 'rb') as _f:
                _completed_targets = pickle.load(_f)
            print(f"  Resuming Stage 2: found {len(_completed_targets)} completed target(s) in checkpoint")
        except Exception as _e:
            print(f"  Warning: could not load Stage 2 target checkpoint: {_e}")
            _completed_targets = {}

    _pending: list = []   # (target, tgt_label, payload)
    for target in targets:
        tgt_label = _TARGET_LABELS.get(target, f'Stage2   {target}')

        # Resume: skip already-completed targets
        if tgt_label in _completed_targets:
            print(f"\n  [{tgt_label}] Skipping (loaded from per-target checkpoint)")
            _ck = _completed_targets[tgt_label]
            segment_results[tgt_label] = _ck['seg_result']
            tgt_best_feats[_target_key_map.get(target, target)] = _ck['feats']
            continue

        feat_names, fixed_feats = _stage2_feature_spec(data, target)
        feat_names = _filter_candidates_by_signal(data, feat_names)
        fixed_feats = [f for f in fixed_feats if f in feat_names]
        _record_candidate_availability(2, tgt_label, target, feat_names, data)
        if not feat_names:
            print(f"    ! {tgt_label}: no usable candidates after filtering, skipping")
            continue

        _pending.append((target, tgt_label, {
            'label':              tgt_label,
            'stage_id':           2,
            'data':               data,
            'feat_names':         feat_names,
            'fixed_feats':        fixed_feats,
            'ga_kwargs':          ga_kwargs,
            'n_restarts':         n_restarts,
            'use_sim_anneal':     use_sim_anneal,
            'fitness_metric':     fitness_metric,
            'collinear_groups':   STAGE_COLLINEAR_GROUPS.get(2, []),
            'require_one_groups': STAGE_REQUIRE_ONE_GROUPS.get(2, []),
            'fallback_to_default': fallback_to_default,
            'feature_map_key':    _target_key_map.get(target, target),
            # Stage 2 GA search uses fast OLS-BIC regardless of estimator mode.
            # MetaCount Tobit/lognormal is only used for the *final* model fit
            # (estimate_stage2), not as the GA fitness oracle.  This reduces
            # each evaluation from ~10-90 s to <1 ms (~1000x speed-up).
            'default_fit': ('ga_external_estimators', 'stage2_fitness_fast', ([target],)),
            'mc_fit':      None,
            'mc_fail_msg':     "",
            'mc_fallback_msg': "",
        }))

    if _pending:
        _n_workers = _ga_max_workers(len(_pending))
        print(f"\n  Stage 2: running {len(_pending)} target(s) "
              f"({'parallel, workers=' + str(_n_workers) if _n_workers > 1 else 'serial'}) ...")
        _plog(f"[Stage 2] launching {len(_pending)} target(s), workers={_n_workers}")
        for i, result in _run_segments_parallel([p for *_, p in _pending], _n_workers):
            target, tgt_label, _ = _pending[i]
            seg_result = result['seg_result']
            tgt_feats  = result['feats']
            segment_results[tgt_label] = seg_result
            tgt_best_feats[result['feature_map_key']] = tgt_feats
            _plog(f"[Stage 2] Done: {tgt_label}  BIC/n={-result['best_fitness']:.4f}  "
                  f"feats={len(tgt_feats)}  elapsed={result['elapsed']/60:.1f}min")

            # Save per-target checkpoint
            if _ckpt_file:
                _completed_targets[tgt_label] = {'seg_result': seg_result, 'feats': tgt_feats}
                try:
                    with open(_ckpt_file, 'wb') as _f:
                        pickle.dump(_completed_targets, _f)
                    print(f"    Stage 2 target checkpoint saved "
                          f"({len(_completed_targets)}/{len(targets)} targets)")
                    _plog(f"[Stage 2] checkpoint: {len(_completed_targets)}/{len(targets)} targets done")
                except Exception as _e:
                    print(f"    Warning: could not save Stage 2 target checkpoint: {_e}")

    return _combine_segment_feature_results(
        stage=2,
        segment_results=segment_results,
        feature_map=tgt_best_feats,
        fitness_metric=fitness_metric,
        n_restarts=n_restarts,
    )


def _run_ga_stage3(tours, persons, ga_kwargs, trips=None, n_restarts=2) -> GAResult:
    from stage3_activity_duration import prepare_duration_data, PURPOSES
    from ga_external_estimators import ga_estimator, reset_feature_weakness
    reset_feature_weakness()   # clear weakness scores from any prior stage

    print("\n[Stage 3] Preparing estimation data...")
    data = prepare_duration_data(tours, persons, trips=trips)
    # Feature engineering: create binary columns referenced by optional feature specs
    if 'ptype' in data.columns:
        _dummies = pd.get_dummies(data['ptype'], prefix='ptype')
        for _c in _dummies.columns:
            if _c not in data.columns:
                data[_c] = _dummies[_c].astype(int)
    if 'sexp' in data.columns and 'is_female' not in data.columns:
        data['is_female'] = (data['sexp'] == 2).astype(int)
    if 'incp' in data.columns:
        if 'income_hi'      not in data.columns: data['income_hi']      = (data['incp'] >= 12).astype(int)
        if 'income_lo'      not in data.columns: data['income_lo']      = (data['incp'] <= 4).astype(int)
        if 'income_midplus' not in data.columns: data['income_midplus'] = (data['incp'] >= 8).astype(int)
    if 'WORKSTAT' in data.columns and 'mandatory_role' not in data.columns:
        data['mandatory_role'] = (data['WORKSTAT'] >= 1).astype(int)
    data = _enrich_ga_data(data, persons)
    purp_col = next(
        (c for c in ['main_purpose', '_purpose_coded', 'purpose'] if c in data.columns),
        None
    )
    if purp_col is None:
        raise RuntimeError("Stage 3 GA: no purpose column in prepared data.")

    # Use work/edu/nonmand structure matching Stage 5.
    # All six non-mandatory purposes are pooled into a 'nonmand' segment.
    _S3_NONMAND_PURPS = ['escort', 'shopping', 'social', 'othmaint', 'othdiscr', 'eat']
    _s3_nonmand_set   = set(_S3_NONMAND_PURPS)

    # Build per-segment data subsets with 'nonmand' relabeling
    _data_by_seg: dict = {}
    for _p in ['work', 'edu']:
        _sub = data[data[purp_col] == _p]
        if len(_sub) >= 30:
            _data_by_seg[_p] = data  # pass full data; fitness filters by purpose
    # For nonmand: relabel all six non-mandatory rows so fitness can filter on 'nonmand'
    _nm_rows = data[data[purp_col].isin(_s3_nonmand_set)].copy()
    _nm_rows[purp_col] = 'nonmand'
    if len(_nm_rows) >= 30:
        _data_by_seg['nonmand'] = _nm_rows

    purposes = list(_data_by_seg.keys())
    est      = ga_estimator()
    use_sim_anneal = est in ('metacount', 'both', 'hybrid', 'sa_bandit')
    dur_col  = next((c for c in ['duration_min', 'duration', 'B'] if c in data.columns), None)
    print(f"  {len(data)} activities | work/edu/nonmand structure | "
          f"{len(purposes)} segments | estimator={est}")

    fallback_to_default = (
        os.environ.get('GA_ESTIMATOR_FALLBACK', '0') == '1'
        and est not in ('metacount', 'searchlibrium', 'both', 'hybrid')
    )

    if est in ('metacount', 'both', 'hybrid', 'sa_bandit'):
        fitness_metric = 'mean(-BIC/n) [MetaCount lognormal]'
    else:
        fitness_metric = 'mean(-BIC/n)'

    mc_state = {'enabled': est in ('metacount', 'both', 'hybrid', 'sa_bandit') and dur_col is not None}
    segment_results: dict = {}
    purp_best_feats: dict = {}

    # ── Per-purpose checkpoint ──────────────────────────────────────────────
    _ckpt_dir  = os.environ.get('GA_CHECKPOINT_DIR', None)
    _ckpt_file = os.path.join(_ckpt_dir, 'ga_stage3_purposes.pkl') if _ckpt_dir else None
    _completed_purposes: dict = {}
    if _ckpt_file and os.path.exists(_ckpt_file):
        try:
            with open(_ckpt_file, 'rb') as _f:
                _completed_purposes = pickle.load(_f)
            print(f"  Resuming Stage 3: found {len(_completed_purposes)} completed purpose(s) in checkpoint")
        except Exception as _e:
            print(f"  Warning: could not load Stage 3 purpose checkpoint: {_e}")
            _completed_purposes = {}

    _pending: list = []   # (purpose, purp_label, payload)
    for purpose in purposes:
        purp_label = f"Stage3   {_PURP_LABELS.get(purpose, purpose)}"

        if purp_label in _completed_purposes:
            print(f"\n  [{purp_label}] Skipping (loaded from per-purpose checkpoint)")
            _ck = _completed_purposes[purp_label]
            segment_results[purp_label] = _ck['seg_result']
            purp_best_feats[purpose]    = _ck['feats']
            continue

        _seg_data = _data_by_seg.get(purpose, data)
        feat_names, fixed_feats = _stage3_feature_spec(_seg_data, purpose)
        _record_candidate_availability(3, purp_label, purpose, feat_names, _seg_data)
        if not feat_names:
            continue

        _pending.append((purpose, purp_label, {
            'label':              purp_label,
            'stage_id':           3,
            'data':               _seg_data,
            'feat_names':         feat_names,
            'fixed_feats':        fixed_feats,
            'ga_kwargs':          ga_kwargs,
            'n_restarts':         n_restarts,
            'use_sim_anneal':     use_sim_anneal,
            'fitness_metric':     fitness_metric,
            'collinear_groups':   STAGE_COLLINEAR_GROUPS.get(3, []),
            'require_one_groups': STAGE_REQUIRE_ONE_GROUPS.get(3, []),
            'fallback_to_default': fallback_to_default,
            'feature_map_key':    purpose,
            # Stage 3 GA search uses fast OLS-on-log(duration) BIC.
            # MetaCount lognormal MLE is only used in the *final* estimate_stage3
            # call, not as the GA fitness oracle (~1000x speed-up).
            'default_fit': ('ga_external_estimators', 'stage3_fitness_fast',
                            ([purpose], purp_col, dur_col)),
            'mc_fit':      None,
            'mc_fail_msg':     "",
            'mc_fallback_msg': "",
        }))

    if _pending:
        _n_workers = _ga_max_workers(len(_pending))
        print(f"\n  Stage 3: running {len(_pending)} purpose(s) "
              f"({'parallel, workers=' + str(_n_workers) if _n_workers > 1 else 'serial'}) ...")
        _plog(f"[Stage 3] launching {len(_pending)} purpose(s), workers={_n_workers}")
        for i, result in _run_segments_parallel([p for *_, p in _pending], _n_workers):
            purpose, purp_label, _ = _pending[i]
            seg_result = result['seg_result']
            purp_feats = result['feats']
            segment_results[purp_label] = seg_result
            purp_best_feats[result['feature_map_key']] = purp_feats
            _plog(f"[Stage 3] Done: {purp_label}  BIC/n={-result['best_fitness']:.4f}  "
                  f"feats={len(purp_feats)}  elapsed={result['elapsed']/60:.1f}min")

            # Save per-purpose checkpoint
            if _ckpt_file:
                _completed_purposes[purp_label] = {'seg_result': seg_result, 'feats': purp_feats}
                try:
                    with open(_ckpt_file, 'wb') as _f:
                        pickle.dump(_completed_purposes, _f)
                    print(f"    Stage 3 purpose checkpoint saved "
                          f"({len(_completed_purposes)}/{len(purposes)} purposes)")
                    _plog(f"[Stage 3] checkpoint: {len(_completed_purposes)}/{len(purposes)} purposes done")
                except Exception as _e:
                    print(f"    Warning: could not save Stage 3 purpose checkpoint: {_e}")

    # Expand 'nonmand' features back to all six individual nonmand purposes so
    # downstream estimation uses the GA-selected features for each purpose.
    if 'nonmand' in purp_best_feats:
        _nm_feats = purp_best_feats.pop('nonmand')
        _s3_all_purps = [p for p in PURPOSES if len(data[data[purp_col] == p]) >= 30]
        for _np in _S3_NONMAND_PURPS:
            if _np in _s3_all_purps:
                purp_best_feats[_np] = _nm_feats

    return _combine_segment_feature_results(
        stage=3,
        segment_results=segment_results,
        feature_map=purp_best_feats,
        fitness_metric=fitness_metric,
        n_restarts=n_restarts,
    )


def _run_ga_stage23_joint(tours, persons, ga_kwargs, trips=None, n_restarts=2) -> GAResult:
    """
    Joint Stage 2+3 SA-Bandit search with a shared feature pool.

    Instead of running separate SA-Bandit searches for Stage 2 (3 targets) and
    Stage 3 (3 purpose segments), this runs a **single** SA-Bandit search over
    a unified candidate feature pool.  The fitness oracle
    `stage23_fitness_joint` evaluates each chromosome against both sub-models
    simultaneously using fast OLS-BIC proxies (~1 ms per eval).

    After the SA-Bandit converges, `fit_joint_stage23` is called **once** with
    the winning feature set to produce the final JAX-optimised joint model
    (budget-constrained lognormal/Weibull).  This replaces the combined ~754 min
    (634+120) that separate MetaCount runs would require.

    Activated via STAGE23_JOINT=1.
    """
    from stage2_day_duration import aggregate_person_day
    from stage3_activity_duration import prepare_duration_data, PURPOSES
    from ga_external_estimators import ga_estimator, reset_feature_weakness, stage23_fitness_joint
    from stage23_joint import fit_joint_stage23, joint_model_summary
    reset_feature_weakness()

    print("\n[Stage 2+3 Joint] Preparing estimation data...")

    # ── Stage 2 data ─────────────────────────────────────────────────────────
    person_day = aggregate_person_day(tours, persons)
    for _col, _val in [
        ('ptype', lambda d: pd.get_dummies(d['ptype'], prefix='ptype')),
    ]:
        if _col in person_day.columns:
            _dummies = pd.get_dummies(person_day[_col], prefix=_col)
            for _c in _dummies.columns:
                if _c not in person_day.columns:
                    person_day[_c] = _dummies[_c].astype(int)
    if 'sexp' in person_day.columns and 'is_female' not in person_day.columns:
        person_day['is_female'] = (person_day['sexp'] == 2).astype(int)
    if 'incp' in person_day.columns:
        if 'income_hi' not in person_day.columns:
            person_day['income_hi'] = (person_day['incp'] >= 12).astype(int)
        if 'income_lo' not in person_day.columns:
            person_day['income_lo'] = (person_day['incp'] <= 4).astype(int)
        if 'income_midplus' not in person_day.columns:
            person_day['income_midplus'] = (person_day['incp'] >= 8).astype(int)
    person_day = _enrich_ga_data(person_day, persons)

    # ── Stage 3 data ─────────────────────────────────────────────────────────
    tours3 = prepare_duration_data(tours, persons, trips=trips)
    if 'ptype' in tours3.columns:
        _dummies = pd.get_dummies(tours3['ptype'], prefix='ptype')
        for _c in _dummies.columns:
            if _c not in tours3.columns:
                tours3[_c] = _dummies[_c].astype(int)
    if 'sexp' in tours3.columns and 'is_female' not in tours3.columns:
        tours3['is_female'] = (tours3['sexp'] == 2).astype(int)
    if 'incp' in tours3.columns:
        if 'income_hi' not in tours3.columns:
            tours3['income_hi'] = (tours3['incp'] >= 12).astype(int)
        if 'income_lo' not in tours3.columns:
            tours3['income_lo'] = (tours3['incp'] <= 4).astype(int)
        if 'income_midplus' not in tours3.columns:
            tours3['income_midplus'] = (tours3['incp'] >= 8).astype(int)
    tours3 = _enrich_ga_data(tours3, persons)

    purp_col = next(
        (c for c in ['main_purpose', '_purpose_coded', 'purpose'] if c in tours3.columns),
        None,
    )
    if purp_col is None:
        raise RuntimeError("Stage 2+3 Joint GA: no purpose column found in tour data.")
    dur_col = next((c for c in ['duration_min', 'duration', 'B'] if c in tours3.columns), None)

    # Pool non-mandatory purposes as 'nonmand' (consistent with Stage 3 structure)
    _S3_NONMAND = {'escort', 'shopping', 'social', 'othmaint', 'othdiscr', 'eat'}
    tours3_nm = tours3.copy()
    tours3_nm.loc[tours3_nm[purp_col].isin(_S3_NONMAND), purp_col] = 'nonmand'
    active_purposes = [p for p in ['work', 'edu', 'nonmand']
                       if (tours3_nm[purp_col] == p).sum() >= 30]

    # ── Joint candidate feature pool ─────────────────────────────────────────
    # Union of Stage 2 and Stage 3 candidate features, filtered to columns
    # present in both DataFrames (so the oracle can evaluate both sub-models).
    s2_cands = _stage2_candidates(person_day)
    s3_cands = _stage3_candidates(tours3)
    # Shared pool: features available in both datasets
    both_cols = set(person_day.columns) & set(tours3.columns)
    all_cands = list(dict.fromkeys(
        c for c in s2_cands + s3_cands if c in both_cols
    ))
    # Stage-3-only activity-context vars (not in person_day): add as optional
    s3_only = [c for c in s3_cands if c not in both_cols and c in tours3.columns]
    all_cands = list(dict.fromkeys(all_cands + s3_only))
    all_cands = _filter_candidates_by_signal(person_day, all_cands)

    # Fixed features: intercept candidates always included
    fixed_feats = [c for c in ['vehrd', 'sexp', 'nprd'] if c in all_cands]

    print(f"  {len(person_day)} person-days | {len(tours3)} activity episodes | "
          f"purposes={active_purposes} | candidates={len(all_cands)} | "
          f"estimator={ga_estimator()}")

    # ── Data bundle for the fitness oracle ───────────────────────────────────
    data_bundle = {
        'person_day': person_day,
        'tours':      tours3_nm,
        'purp_col':   purp_col,
        'dur_col':    dur_col or 'duration_min',
    }

    est = ga_estimator()
    use_sim_anneal = est in ('metacount', 'both', 'hybrid', 'sa_bandit')
    fitness_metric = 'mean(-BIC/n) [joint OLS-BIC Stage2+3]'

    # ── Single-segment SA-Bandit search ──────────────────────────────────────
    label = 'Stage2+3   Joint (budget-constrained)'
    payload = {
        'label':               label,
        'stage_id':            23,
        'data':                data_bundle,
        'feat_names':          all_cands,
        'fixed_feats':         fixed_feats,
        'ga_kwargs':           ga_kwargs,
        'n_restarts':          n_restarts,
        'use_sim_anneal':      use_sim_anneal,
        'fitness_metric':      fitness_metric,
        'collinear_groups':    STAGE_COLLINEAR_GROUPS.get(2, []),
        'require_one_groups':  STAGE_REQUIRE_ONE_GROUPS.get(2, []),
        'fallback_to_default': False,
        'feature_map_key':     'joint',
        # Fast OLS-BIC proxy: <2 ms per eval, so SA-Bandit can run 3000+ evals.
        # fit_joint_stage23 (JAX) is called ONCE on the winning feature set below.
        'default_fit': ('ga_external_estimators', 'stage23_fitness_joint',
                        (active_purposes, 5.0)),
        'mc_fit':      None,
        'mc_fail_msg':     '',
        'mc_fallback_msg': '',
    }

    result = _run_segments_parallel([payload], max_workers=1)
    result = dict(list(result))   # consume generator → {0: result_dict}
    seg_result = result[0]['seg_result']
    best_feats = result[0]['feats']

    print(f"\n  [Stage 2+3 Joint] SA-Bandit converged → {len(best_feats)} features: {best_feats}")

    # ── Final joint estimation with winning feature set ───────────────────────
    print("  [Stage 2+3 Joint] Running final JAX joint estimation...")
    final_models = fit_joint_stage23(
        person_day=person_day,
        tours=tours3,
        persons=persons,
        purposes=PURPOSES,
        feature_override={
            'B':     [f for f in best_feats if f in person_day.columns],
            'start': [f for f in best_feats if f in person_day.columns],
            'end':   [f for f in best_feats if f in person_day.columns],
            'purp':  [f for f in best_feats if f in tours3.columns],
        },
    )
    print(f"  {joint_model_summary(final_models)}")

    # Attach final models to the GAResult metadata so _run_joint_stage23 and
    # _write_timing_comparison can access them.
    combined = _combine_segment_feature_results(
        stage=23,
        segment_results={label: seg_result},
        feature_map={'joint': best_feats},
        fitness_metric=fitness_metric,
        n_restarts=n_restarts,
    )
    if combined.metadata is None:
        combined.metadata = {}
    combined.metadata['stage23_joint_models'] = final_models
    combined.metadata['best_features_joint'] = best_feats
    return combined


def _run_ga_stage4(trips, tours, persons, ga_kwargs, n_restarts=2) -> GAResult:
    from stage4_schedule_placement import prepare_timing_data, PURPOSES
    from ga_external_estimators import ga_estimator, reset_feature_weakness
    reset_feature_weakness()   # clear weakness scores from any prior stage

    print("\n[Stage 4] Preparing estimation data...")
    data       = prepare_timing_data(trips, tours, persons)
    # Feature engineering: add binary columns expected by the GA feature spec
    if 'ptype' in data.columns:
        _dummies = pd.get_dummies(data['ptype'], prefix='ptype')
        for _c in _dummies.columns:
            if _c not in data.columns:
                data[_c] = _dummies[_c].astype(int)
    if 'sexp' in data.columns and 'is_female' not in data.columns:
        data['is_female'] = (data['sexp'] == 2).astype(int)
    if 'WORKSTAT' in data.columns and 'mandatory_role' not in data.columns:
        data['mandatory_role'] = (data['WORKSTAT'] >= 1).astype(int)
    data = _enrich_ga_data(data, persons)
    purposes   = [p for p in PURPOSES if len(data[data['purpose'] == p]) >= 30]
    est            = ga_estimator()
    use_sim_anneal = (os.environ.get('GA_STAGE4_USE_SA', '1').strip() == '1'
                      or est in ('searchlibrium', 'both', 'hybrid', 'sa_bandit'))
    n_alts         = int(data['period'].max() + 1) if 'period' in data.columns else 6
    print(f"  {len(data)} trips | purpose-specific feature sets | "
          f"{len(purposes)} purposes | estimator={est}")

    fallback_to_default = (
        os.environ.get('GA_ESTIMATOR_FALLBACK', '0') == '1'
        and est not in ('metacount', 'searchlibrium', 'both', 'hybrid')
    )

    if est in ('searchlibrium', 'both', 'hybrid', 'sa_bandit'):
        fitness_metric = 'mean(-BIC/n) [SearchLibrium MNL]'
    else:
        fitness_metric = 'mean(-BIC/n)'

    sl_state = {'enabled': est in ('searchlibrium', 'both', 'hybrid', 'sa_bandit')}
    segment_results: dict = {}
    purp_best_feats: dict = {}

    # ── Per-purpose checkpoint ──────────────────────────────────────────────
    _ckpt_dir4  = os.environ.get('GA_CHECKPOINT_DIR', None)
    _ckpt_file4 = os.path.join(_ckpt_dir4, 'ga_stage4_purposes.pkl') if _ckpt_dir4 else None
    _completed_purposes4: dict = {}
    if _ckpt_file4 and os.path.exists(_ckpt_file4):
        try:
            with open(_ckpt_file4, 'rb') as _f:
                _completed_purposes4 = pickle.load(_f)
            print(f"  Resuming Stage 4: found {len(_completed_purposes4)} completed purpose(s) in checkpoint")
        except Exception as _e:
            print(f"  Warning: could not load Stage 4 purpose checkpoint: {_e}")
            _completed_purposes4 = {}

    _pending: list = []   # (purpose, purp_label, payload)
    for purpose in purposes:
        purp_label = f"Stage4   {_PURP_LABELS.get(purpose, purpose)}"

        if purp_label in _completed_purposes4:
            print(f"\n  [{purp_label}] Skipping (loaded from per-purpose checkpoint)")
            _ck4 = _completed_purposes4[purp_label]
            segment_results[purp_label] = _ck4['seg_result']
            purp_best_feats[purpose]    = _ck4['feats']
            continue

        feat_names, fixed_feats = _stage4_feature_spec(data, purpose)
        if not feat_names:
            continue

        _pending.append((purpose, purp_label, {
            'label':              purp_label,
            'stage_id':           4,
            'data':               data,
            'feat_names':         feat_names,
            'fixed_feats':        fixed_feats,
            'ga_kwargs':          ga_kwargs,
            'n_restarts':         n_restarts,
            'use_sim_anneal':     use_sim_anneal,
            'fitness_metric':     fitness_metric,
            'collinear_groups':   STAGE_COLLINEAR_GROUPS.get(4, []),
            'require_one_groups': STAGE_REQUIRE_ONE_GROUPS.get(4, []),
            'fallback_to_default': fallback_to_default,
            'feature_map_key':    purpose,
            'default_fit':        ('ga_hypothesis_testing', '_stage4_fitness_aggregate', ([purpose],)),
            'mc_fit': (
                ('ga_external_estimators', 'stage4_fitness_searchlibrium', ([purpose],), {'n_alts': n_alts})
                if sl_state['enabled'] else None
            ),
            'mc_fail_msg':     "    ! Stage4 SearchLibrium repeatedly failing; disabling SearchLibrium for remaining evaluations",
            'mc_fallback_msg': f"    ! Stage4 SearchLibrium failed for {purpose}; using default MNL fallback",
        }))

    if _pending:
        _n_workers = _ga_max_workers(len(_pending))
        print(f"\n  Stage 4: running {len(_pending)} purpose(s) "
              f"({'parallel, workers=' + str(_n_workers) if _n_workers > 1 else 'serial'}) ...")
        _plog(f"[Stage 4] launching {len(_pending)} purpose(s), workers={_n_workers}")
        for i, result in _run_segments_parallel([p for *_, p in _pending], _n_workers):
            purpose, purp_label, _ = _pending[i]
            seg_result = result['seg_result']
            purp_feats = result['feats']
            segment_results[purp_label] = seg_result
            purp_best_feats[result['feature_map_key']] = purp_feats
            _plog(f"[Stage 4] Done: {purp_label}  BIC/n={-result['best_fitness']:.4f}  "
                  f"feats={len(purp_feats)}  elapsed={result['elapsed']/60:.1f}min")

            # Save per-purpose checkpoint
            if _ckpt_file4:
                _completed_purposes4[purp_label] = {'seg_result': seg_result, 'feats': purp_feats}
                try:
                    with open(_ckpt_file4, 'wb') as _f:
                        pickle.dump(_completed_purposes4, _f)
                    print(f"    Stage 4 purpose checkpoint saved "
                          f"({len(_completed_purposes4)}/{len(purposes)} purposes)")
                    _plog(f"[Stage 4] checkpoint: {len(_completed_purposes4)}/{len(purposes)} purposes done")
                except Exception as _e:
                    print(f"    Warning: could not save Stage 4 purpose checkpoint: {_e}")

    return _combine_segment_feature_results(
        stage=4,
        segment_results=segment_results,
        feature_map=purp_best_feats,
        fitness_metric=fitness_metric,
        n_restarts=n_restarts,
    )


def _ga_stage5_segment_worker(payload: dict) -> dict:
    """
    Run mode GA then dest GA for one Stage 5 segment (work/edu/nonmand) in an
    isolated worker process.

    Each worker opens its own OMX skim handle via payload['skims_path'] (open
    OMX/HDF5 handles aren't picklable). If skims_path is None, payload['skims']
    carries an already-open handle — only valid when running serially
    in-process (max_workers<=1, no pickling involved).
    """
    import openmatrix as omx
    from stage5_destination_choice import (
        _mode_tree, _build_mode_model, _estimate, _logsums,
        _attach_size_term, _attach_acdc_terms,
    )

    seg            = payload['seg']
    seg_label      = payload['seg_label']
    tours          = payload['tours']
    persons        = payload['persons']
    landuse        = payload['landuse']
    skim_df        = payload['skim_df']
    seg_tours      = payload['seg_tours']
    seg_mode_cands = payload['seg_mode_cands']
    seg_dest_cands = payload['seg_dest_cands']
    ga_kwargs      = payload['ga_kwargs']
    n_restarts     = payload['n_restarts']

    skims_path = payload.get('skims_path')
    if skims_path:
        skims = omx.open_file(skims_path, 'r')
        _close_skims = True
    else:
        skims = payload['skims']
        _close_skims = False

    try:
        _seg_t0 = time.time()
        print(f"\n  [Stage 5   {seg_label}] Mode SA/GA (worker pid={os.getpid()}) ...")
        print(f"    Candidate pools: mode={len(seg_mode_cands)}  dest={len(seg_dest_cands)}")

        # ── Mode search (chromosome = mode candidates only)
        # Warm-start: evaluate full-feature baseline first, then SA/GA perturbs
        # downward from that optimum.  SA is active when GA_USE_SA=1 (--sa-cases).
        def _mode_fit(genes):
            return _stage5_mode_only_fitness(
                genes, seg_mode_cands, seg, tours, persons, landuse, skims)

        _bl_mode_genes   = np.ones(len(seg_mode_cands), dtype=int)
        _bl_mode_fitness = _mode_fit(_bl_mode_genes)
        print(f"    Mode baseline (all features): BIC/n={-_bl_mode_fitness:.4f}")

        _mode_fixed = [f for f in ['CRASH'] if f in seg_mode_cands]
        _ga_kwargs_mode = dict(ga_kwargs)
        if len(seg_mode_cands) > len(_mode_fixed):
            _user_min_mode = int(_ga_kwargs_mode.get('min_features', 3))
            _ga_kwargs_mode['min_features'] = min(
                max(_user_min_mode, len(_mode_fixed)),
                len(seg_mode_cands) - 1,
            )
        else:
            _ga_kwargs_mode['min_features'] = len(seg_mode_cands)
        mode_searcher = _make_searcher(
            len(seg_mode_cands), _mode_fit, _ga_kwargs_mode,
            use_sim_anneal   = False,
            use_bandit       = True,
            feature_names       = seg_mode_cands,
            collinear_groups    = STAGE_COLLINEAR_GROUPS.get('5_mode', []),
            require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get('5_mode', []),
            fixed_features   = _mode_fixed,
            fitness_metric   = 'mean(-BIC/n) [Larch NL]',
            stage_id         = 'stage5_mode',
        )
        _bl_mode_key = tuple(int(x) for x in _bl_mode_genes.tolist())
        mode_searcher._cache[_bl_mode_key] = _bl_mode_fitness
        _bl_mode_seed = Individual(genes=_bl_mode_genes, fitness=_bl_mode_fitness)
        mode_best, mode_hist, mode_pop = mode_searcher.run_with_restarts(
            n_restarts=n_restarts, seed_individuals=[_bl_mode_seed])
        best_mode_vars = [v for v, g in zip(seg_mode_cands, mode_best.genes) if g == 1]
        print(f"    Mode best vars: {best_mode_vars}  BIC/n={-mode_best.fitness:.4f}")

        mode_seg_result = GAResult(
            stage=5, best_features={'mode': best_mode_vars, 'dest': []},
            best_fitness=mode_best.fitness, best_individual=mode_best,
            history=mode_hist, all_candidates=mode_pop,
            feature_names=seg_mode_cands, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=mode_searcher.cache_hits,
        )
        mode_seg_result.metadata['sub_model'] = 'mode'

        # ── Pre-compute mode logsums with best mode vars
        ls_cached    = None
        lu_st_cached = None
        seg_acdc     = None
        logsums_ok   = False
        try:
            dt_m          = _mode_tree(seg_tours, persons, landuse, skims)
            mode_mdl_best = _build_mode_model(dt_m, seg, custom_mode_vars=best_mode_vars,
                                              avail_vars_hint=list(seg_tours.columns))
            mode_mdl_best = _estimate(mode_mdl_best)
            ls_cached     = _logsums(mode_mdl_best, dt_m, skims=skims, verbose=False)
            lu_st_cached  = _attach_size_term(landuse, seg)
            lu_st_cached, seg_acdc = _attach_acdc_terms(
                lu_st_cached, seg_tours, skim_df, seg)
            logsums_ok    = True
            print(f"    Mode logsums cached for Dest GA.")
        except Exception as e:
            logger.warning(f"Stage 5 [{seg}] could not pre-compute mode logsums: {e}")

        # ── Dest search (chromosome = dest candidates only)
        print(f"\n  [Stage 5   {seg_label}] Dest SA/GA ...")
        if logsums_ok:
            def _dest_fit(genes):
                return _stage5_dest_only_fitness(
                    genes, seg_dest_cands, seg, seg_tours, persons, lu_st_cached, skims, ls_cached, seg_acdc)

            # Warm-start: fit full dest model first, SA/GA perturbs from there.
            _bl_dest_genes   = np.ones(len(seg_dest_cands), dtype=int)
            _bl_dest_fitness = _dest_fit(_bl_dest_genes)
            print(f"    Dest baseline (all features): BIC/n={-_bl_dest_fitness:.4f}")

            _dest_fixed = [f for f in ['c_crash'] if f in seg_dest_cands]
            _ga_kwargs_dest = dict(ga_kwargs)
            if len(seg_dest_cands) > len(_dest_fixed):
                _user_min_dest = int(_ga_kwargs_dest.get('min_features', 3))
                _ga_kwargs_dest['min_features'] = min(
                    max(_user_min_dest, len(_dest_fixed)),
                    len(seg_dest_cands) - 1,
                )
            else:
                _ga_kwargs_dest['min_features'] = len(seg_dest_cands)
            dest_searcher = _make_searcher(
                len(seg_dest_cands), _dest_fit, _ga_kwargs_dest,
                use_sim_anneal   = False,
                use_bandit       = True,
                feature_names       = seg_dest_cands,
                collinear_groups    = STAGE_COLLINEAR_GROUPS.get('5_dest', []),
                require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get('5_dest', []),
                fixed_features   = _dest_fixed,
                fitness_metric   = 'mean(-BIC/n) [Larch NL]',
                stage_id         = 'stage5_dest',
            )
            _bl_dest_key = tuple(int(x) for x in _bl_dest_genes.tolist())
            dest_searcher._cache[_bl_dest_key] = _bl_dest_fitness
            _bl_dest_seed = Individual(genes=_bl_dest_genes, fitness=_bl_dest_fitness)
            dest_best, dest_hist, dest_pop = dest_searcher.run_with_restarts(
                n_restarts=n_restarts, seed_individuals=[_bl_dest_seed])
            dest_cache = dest_searcher.cache_hits
        else:
            # Fallback: joint GA if logsums could not be computed
            def _joint_fit(genes):
                return _stage5_fitness_larch(
                    genes[:len(seg_mode_cands)], genes[len(seg_mode_cands):], seg_mode_cands, seg_dest_cands,
                    seg, tours, persons, landuse, skims, skim_df)
            fallback_genes = np.zeros(len(seg_dest_cands), dtype=int)
            dest_best  = Individual(genes=fallback_genes, fitness=-1e9)
            dest_hist  = []
            dest_pop   = [dest_best]
            dest_cache = 0

        best_dest_vars = [v for v, g in zip(seg_dest_cands, dest_best.genes) if g == 1]
        print(f"    Dest best vars: {best_dest_vars}  BIC/n={-dest_best.fitness:.4f}")

        dest_seg_result = GAResult(
            stage=5, best_features={'mode': best_mode_vars, 'dest': best_dest_vars},
            best_fitness=dest_best.fitness, best_individual=dest_best,
            history=dest_hist, all_candidates=dest_pop,
            feature_names=seg_dest_cands, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=dest_cache,
        )
        dest_seg_result.metadata['sub_model'] = 'dest'

        return {
            'seg':             seg,
            'seg_label':       seg_label,
            'mode_seg_result': mode_seg_result,
            'dest_seg_result': dest_seg_result,
            'best_mode_vars':  best_mode_vars,
            'best_dest_vars':  best_dest_vars,
            'mode_genes':      mode_best.genes.copy(),
            'elapsed':         time.time() - _seg_t0,
        }
    finally:
        if _close_skims:
            skims.close()


def _run_ga_stage5(trips, persons, skims, landuse, ga_kwargs,
                   n_restarts=2, tours_src=None) -> GAResult:
    """
    GA for Stage 5 mode + destination variable selection.
    ESTIMATION ONLY   fits Larch models on HHTS, no prediction over synth pop.
    Aggregates fitness across all valid segments (work, edu, nonmand).

    Parameters
    ----------
    trips      : HHTS trips table (individual legs)
    persons    : HHTS persons table
    tours_src  : HHTS tours table with tour_mode column. If provided,
                 used as the source for _prep_hhts_tours so mode choice
                 is estimated on real observed modes. Falls back to trips
                 with dominant-mode derivation if None.
    """
    import openmatrix as omx
    from stage5_destination_choice import _prep_hhts_tours, _skim_flat, _filter_seg

    print("\n[Stage 5] Preparing HHTS estimation data...")
    # Use the HHTS tours table (has tour_mode) rather than trips (has no tour_mode).
    # Passing trips directly causes all tours to default to DA  degenerate mode model.
    _tours_input = tours_src if tours_src is not None else trips
    if tours_src is not None and 'tour_mode' in tours_src.columns:
        pass   # good   tours table already has observed modes
    elif tours_src is None:
        # Last-resort fallback: derive dominant trip mode onto the trips table
        _mode_col = next((c for c in ['mode', 'MAINMODE'] if c in trips.columns), None)
        if _mode_col and 'tour_id' in trips.columns:
            _dom = (trips.groupby(['person_id', 'tour_id'])[_mode_col]
                    .agg(lambda x: x.value_counts().idxmax() if len(x) else 'DA')
                    .reset_index().rename(columns={_mode_col: 'tour_mode'}))
            _trips_with_mode = trips.copy()
            if 'tour_mode' in _trips_with_mode.columns:
                _trips_with_mode = _trips_with_mode.drop(columns=['tour_mode'])
            _tours_input = _trips_with_mode.merge(
                _dom, on=['person_id', 'tour_id'], how='left')
    tours    = _prep_hhts_tours(_tours_input, persons)
    # Feature engineering: create binary columns referenced by Stage 5 candidate pools
    if 'ptype' in tours.columns:
        _dummies = pd.get_dummies(tours['ptype'], prefix='ptype')
        for _c in _dummies.columns:
            if _c not in tours.columns:
                tours[_c] = _dummies[_c].astype(int)
    if 'sexp' in tours.columns and 'is_female' not in tours.columns:
        tours['is_female'] = (tours['sexp'] == 2).astype(int)
    if 'incp' in tours.columns:
        if 'income_hi'      not in tours.columns: tours['income_hi']      = (tours['incp'] >= 12).astype(int)
        if 'income_lo'      not in tours.columns: tours['income_lo']      = (tours['incp'] <= 4).astype(int)
        if 'income_midplus' not in tours.columns: tours['income_midplus'] = (tours['incp'] >= 8).astype(int)
    if 'WORKSTAT' in tours.columns and 'mandatory_role' not in tours.columns:
        tours['mandatory_role'] = (tours['WORKSTAT'] >= 1).astype(int)
    skim_df  = pd.DataFrame()

    # Path to the OMX skim file, used by parallel segment workers (each worker
    # opens its own handle — open OMX/HDF5 handles aren't picklable).
    skims_path = skims if isinstance(skims, str) else getattr(skims, 'filename', None)

    close_skims = False
    if isinstance(skims, str):
        skims       = omx.open_file(skims, 'r')
        close_skims = True

    try:
        skim_df = _skim_flat(skims)
        print(f"  Skim flat table: {len(skim_df)} OD pairs")
    except Exception as e:
        logger.warning(f"Stage 5 GA: could not build skim_df: {e}")

    mode_cands = _stage5_mode_candidates()
    dest_cands = _stage5_dest_candidates()
    n_mode     = len(mode_cands)

    # ÔöÇÔöÇ Segment mapping ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    # Tour-level segments: work, edu (school), nonmand (non-mandatory).
    # "mandatory" = work + edu combined; also evaluated for summary.
    SEG_LABELS = {
        'work':    'Work (tour)',
        'edu':     'School/Edu (tour)',
        'nonmand': 'Non-Mandatory (tour)',
    }

    segments   = ['work', 'edu', 'nonmand']
    valid_segs = [s for s in segments if len(_filter_seg(tours, s)) >= 50]
    print(f"  {len(tours)} HHTS tours | mode_cands={len(mode_cands)}  dest_cands={len(dest_cands)} | "
          f"valid segments: {valid_segs}")

    # ÔöÇÔöÇ Per-segment SEPARATE mode GA then dest GA ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    # Separate GAs avoid joint-chromosome entanglement between mode and
    # destination pools.  Mode GA runs first; its best vars are used to
    # pre-compute logsums for the subsequent destination GA.
    segment_results = {}
    seg_best_mode_vars = {}
    seg_best_mode_genes = {}
    seg_best_dest_vars = {}

    # -- Per-segment checkpoint --------------------------------------------
    _ckpt_dir5  = os.environ.get('GA_CHECKPOINT_DIR', None)
    _ckpt_file5 = os.path.join(_ckpt_dir5, 'ga_stage5_segments.pkl') if _ckpt_dir5 else None
    _completed_segs: dict = {}
    if _ckpt_file5 and os.path.exists(_ckpt_file5):
        try:
            with open(_ckpt_file5, 'rb') as _f:
                _completed_segs = pickle.load(_f)
            print(f"  Resuming Stage 5: found {len(_completed_segs)} completed segment(s) in checkpoint")
        except Exception as _e:
            print(f"  Warning: could not load Stage 5 segment checkpoint: {_e}")
            _completed_segs = {}

    _pending: list = []   # (seg, seg_label, payload)
    for seg in valid_segs:
        seg_label = SEG_LABELS.get(seg, seg)

        if seg in _completed_segs:
            print(f"\n  [Stage 5   {seg_label}] Skipping (loaded from per-segment checkpoint)")
            _ck = _completed_segs[seg]
            segment_results[f'{seg_label}   Mode'] = _ck['mode_seg_result']
            segment_results[f'{seg_label}   Dest'] = _ck['dest_seg_result']
            seg_best_mode_vars[seg]  = _ck['best_mode_vars']
            seg_best_dest_vars[seg]  = _ck['best_dest_vars']
            seg_best_mode_genes[seg] = _ck['mode_genes']
            continue

        seg_tours = _filter_seg(tours, seg)
        seg_mode_cands = _filter_candidates_by_signal(seg_tours, _stage5_mode_candidates_for_segment(seg))
        seg_dest_cands = _filter_candidates_by_signal(seg_tours, _stage5_dest_candidates_for_segment(seg))
        seg_mode_cands = [c for c in seg_mode_cands if c in mode_cands]
        seg_dest_cands = [c for c in seg_dest_cands if c in dest_cands]

        # CRASH is the safety-testing variable this run exists to evaluate
        # (see safety_baseline/safety_nosafety experiment modes). Force it
        # back in even if _filter_candidates_by_signal dropped it for looking
        # near-constant in this segment's data, so it's never silently
        # missing from the mode model. It's marked fixed_features below
        # (see _mode_fixed), which makes it structurally impossible for the
        # GA/SA/Bandit search - including the directed move - to remove it:
        # _opt_idx (the only index set any perturbation touches) excludes
        # every fixed index by construction.
        if 'CRASH' in mode_cands and 'CRASH' not in seg_mode_cands:
            logger.warning(
                f"[{seg}] CRASH failed the candidate-signal filter (looks "
                "near-constant in this segment's data) - force-included "
                "anyway so it's never silently absent from the mode model."
            )
            seg_mode_cands = seg_mode_cands + ['CRASH']

        if not seg_mode_cands:
            seg_mode_cands = mode_cands[:]
        if not seg_dest_cands:
            seg_dest_cands = dest_cands[:]

        _pending.append((seg, seg_label, {
            'seg':            seg,
            'seg_label':      seg_label,
            'tours':          tours,
            'persons':        persons,
            'landuse':        landuse,
            'skim_df':        skim_df,
            'seg_tours':      seg_tours,
            'seg_mode_cands': seg_mode_cands,
            'seg_dest_cands': seg_dest_cands,
            'ga_kwargs':      ga_kwargs,
            'n_restarts':     n_restarts,
            'skims_path':     skims_path,
            'skims':          skims if skims_path is None else None,
        }))

    if _pending:
        _n_workers = _ga_max_workers(len(_pending))
        if skims_path is None and _n_workers > 1:
            print("  ! Stage 5: skim handle has no resolvable file path; "
                  "running segments serially (cannot share OMX handle across processes)")
            _n_workers = 1
        print(f"\n  Stage 5: running {len(_pending)} segment(s) "
              f"({'parallel, workers=' + str(_n_workers) if _n_workers > 1 else 'serial'}) ...")
        _plog(f"[Stage 5] launching {len(_pending)} segment(s), workers={_n_workers}")
        for i, result in _run_segments_parallel(
                [p for *_, p in _pending], _n_workers, worker_fn=_ga_stage5_segment_worker):
            seg, seg_label, _ = _pending[i]
            mode_seg_result = result['mode_seg_result']
            dest_seg_result = result['dest_seg_result']
            segment_results[f'{seg_label}   Mode'] = mode_seg_result
            segment_results[f'{seg_label}   Dest'] = dest_seg_result
            seg_best_mode_vars[seg]  = result['best_mode_vars']
            seg_best_dest_vars[seg]  = result['best_dest_vars']
            seg_best_mode_genes[seg] = result['mode_genes']
            _plog(f"[Stage 5] Done: {seg_label}  "
                  f"mode_BIC/n={-mode_seg_result.best_fitness:.4f} feats={len(result['best_mode_vars'])}  "
                  f"dest_BIC/n={-dest_seg_result.best_fitness:.4f} feats={len(result['best_dest_vars'])}  "
                  f"elapsed={result['elapsed']/60:.1f}min")

            # Save per-segment checkpoint
            if _ckpt_file5:
                _completed_segs[seg] = {
                    'mode_seg_result': mode_seg_result,
                    'dest_seg_result': dest_seg_result,
                    'best_mode_vars':  result['best_mode_vars'],
                    'best_dest_vars':  result['best_dest_vars'],
                    'mode_genes':      result['mode_genes'],
                }
                try:
                    with open(_ckpt_file5, 'wb') as _f:
                        pickle.dump(_completed_segs, _f)
                    print(f"    Stage 5 segment checkpoint saved "
                          f"({len(_completed_segs)}/{len(valid_segs)} segments)")
                    _plog(f"[Stage 5] checkpoint: {len(_completed_segs)}/{len(valid_segs)} segments done")
                except Exception as _e:
                    print(f"    Warning: could not save Stage 5 segment checkpoint: {_e}")

    # ÔöÇÔöÇ Aggregate: majority vote across segments per sub-model 
    def _mode_agg(g):
        scores = [_stage5_mode_only_fitness(g, mode_cands, s, tours, persons, landuse, skims)
                  for s in valid_segs]
        scores = [s for s in scores if s > -1e8]
        return float(np.mean(scores)) if scores else -1e9

    def _dest_agg(g):
        scores = []
        for s in valid_segs:
            f = seg_best_mode_genes.get(s)
            if f is None: continue
            best_mv = [v for v, gg in zip(mode_cands, f) if gg == 1]
            try:
                from stage5_destination_choice import (
                    _filter_seg as _fs, _mode_tree as _mt, _build_mode_model as _bm,
                    _estimate as _est, _logsums as _lgs, _attach_size_term as _ast,
                    _attach_acdc_terms as _aac
                )
                _st   = _fs(tours, s)
                _dt_m = _mt(_st, persons, landuse, skims)
                _mdl  = _bm(_dt_m, s, custom_mode_vars=best_mv)
                _mdl  = _est(_mdl)
                _ls   = _lgs(_mdl, _dt_m, skims=skims, verbose=False)
                _lu   = _ast(landuse, s)
                _lu, _acdc = _aac(_lu, _st, skim_df, s)
                sv = _stage5_dest_only_fitness(g, dest_cands, s, _st, persons, _lu, skims, _ls, _acdc)
                if sv > -1e8:
                    scores.append(sv)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else -1e9

    mode_counts: Dict[str, int] = {}
    for vals in seg_best_mode_vars.values():
        for v in vals:
            mode_counts[v] = mode_counts.get(v, 0) + 1
    dest_counts: Dict[str, int] = {}
    for vals in seg_best_dest_vars.values():
        for v in vals:
            dest_counts[v] = dest_counts.get(v, 0) + 1

    vote_n = max(len(valid_segs), 1)
    best_mode_feats = sorted([v for v, k in mode_counts.items() if k / vote_n >= 0.5])
    best_dest_feats = sorted([v for v, k in dest_counts.items() if k / vote_n >= 0.5])

    # Guardrail: if majority vote is too sparse, keep top-supported candidates.
    if len(best_mode_feats) < 3 and mode_counts:
        best_mode_feats = [k for k, _ in sorted(mode_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
    if len(best_dest_feats) < 4 and dest_counts:
        best_dest_feats = [k for k, _ in sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[:4]]

    if 'CRASH' in mode_cands and 'CRASH' not in best_mode_feats:
        best_mode_feats.append('CRASH')
    if 'c_crash' in dest_cands and 'c_crash' not in best_dest_feats:
        best_dest_feats.append('c_crash')

    consensus_mode = np.array([1 if v in set(best_mode_feats) else 0 for v in mode_cands], dtype=int)
    consensus_dest = np.array([1 if v in set(best_dest_feats) else 0 for v in dest_cands], dtype=int)

    consensus_genes = np.concatenate([consensus_mode, consensus_dest])
    try:
        agg_fit = _stage5_fitness_aggregate(
            consensus_genes, n_mode, mode_cands, dest_cands,
            valid_segs, tours, persons, landuse, skims, skim_df)
    except Exception:
        agg_fit = -1e9

    if close_skims:
        skims.close()

    consensus_ind         = Individual(genes=consensus_genes)
    consensus_ind.fitness = agg_fit
    agg_cache             = sum(r.cache_hits for r in segment_results.values())

    combined = GAResult(
        stage          = 5,
        best_features  = {'mode': best_mode_feats, 'dest': best_dest_feats},
        best_fitness   = consensus_ind.fitness,
        best_individual= consensus_ind,
        history        = [{
            'generation': 0, 'best_fitness': agg_fit, 'best_so_far': agg_fit,
            'mean_fitness': agg_fit, 'worst_fitness': agg_fit,
            'diversity': 0.0, 'mutation_rate': 0.0,
            'n_features': int(consensus_genes.sum()), 'stagnation': 0,
        }],
        all_candidates = [],
        feature_names  = mode_cands + dest_cands,
        fitness_metric = 'mean(-BIC/n)',
        n_restarts     = n_restarts,
        cache_hits     = agg_cache,
    )
    combined.metadata['segment_results'] = segment_results
    print(f"\n  Stage 5 aggregate: mode={best_mode_feats}  dest={best_dest_feats}")
    return combined


def _run_ga_stage5_trip(trips, persons, skims, landuse, ga_kwargs,
                        n_restarts=2) -> GAResult:
    """
    GA for Stage 51   TRIP-LEVEL mode + destination variable selection.

    Runs SEPARATE mode and destination GAs per trip purpose, mirroring the
    ActivitySim distinction between trip_mode_choice and trip destination
    choice.  Separate GAs produce independent convergence plots for each
    sub-model and each purpose.

    For each valid purpose:
      1. Mode GA  : chromosome = mode_candidates only ÔåÆ -BIC_mode/n
      2. Dest GA  : chromosome = dest_candidates only ÔåÆ -BIC_dest/n
                    (trip-level destination logit using observed HHTS DTAZ)

    Results are labelled  "<Purpose>   Mode (trip)"  and  "<Purpose>   Dest (trip)".
    Majority-vote consensus across purposes selects the final feature sets.
    """
    import openmatrix as omx
    from stage6_mode_choice import prepare_mode_data, PURPOSES as TRIP_PURPOSES
    from stage5_destination_choice import _skim_flat

    print("\n[Stage 51   TRIP LEVEL] Preparing HHTS trip estimation data...")

    close_skims = False
    if isinstance(skims, str):
        skims       = omx.open_file(skims, 'r')
        close_skims = True

    try:
        skim_df = _skim_flat(skims)
        print(f"  Skim flat table: {len(skim_df)} OD pairs")
    except Exception as e:
        logger.warning(f"Stage 51 GA: could not build skim_df: {e}")
        skim_df = pd.DataFrame()

    mode_cands = _stage5_mode_candidates()
    dest_cands = _stage5_dest_candidates()
    n_mode     = len(mode_cands)

    # valid purposes = those with ÔëÑ50 trip-mode observations in HHTS
    try:
        trip_mode_data = prepare_mode_data(trips, persons, landuse)
    except Exception as e:
        logger.warning(f"Stage 51 GA: prepare_mode_data failed: {e}")
        trip_mode_data = trips.copy()

    purpose_col = 'purpose' if 'purpose' in trip_mode_data.columns else None
    valid_purposes = [
        p for p in TRIP_PURPOSES
        if purpose_col and len(trip_mode_data[trip_mode_data[purpose_col] == p]) >= 50
    ]
    if not valid_purposes:
        valid_purposes = [p for p in TRIP_PURPOSES
                          if len(trips.get('purpose', pd.Series(dtype=str))
                                 [trips.get('purpose', pd.Series(dtype=str)) == p]) >= 30]
    print(f"  Trip-level GA valid purposes: {valid_purposes}")

    _TRIP_PURP_LABELS = {
        'work':     'Work (trip)',
        'edu':      'School/Edu (trip)',
        'social':   'Social (trip)',
        'escort':   'Escort (trip)',
        'shopping': 'Shopping (trip)',
        'othmaint': 'Oth-Maint (trip)',
        'othdiscr': 'Oth-Discr (trip)',
        'eat':      'Eat Out (trip)',
    }

    segment_results:     dict = {}
    purp_best_mode_genes: dict = {}
    purp_best_dest_genes: dict = {}

    for purp in valid_purposes:
        purp_label = _TRIP_PURP_LABELS.get(purp, f'{purp} (trip)')

        #  Mode GA
        print(f"\n  [Stage 51   {purp_label}] Mode GA ...")

        def _trip_mode_fit(genes, _p=purp):
            """Trip-level mode model BIC/n for one purpose."""
            from stage6_mode_choice import NestedLogitModeModel, prepare_mode_data as _pmd
            mode_vars = [v for v, g in zip(mode_cands, genes) if g == 1]
            if not mode_vars:
                return -1e9
            try:
                md = prepare_mode_data(trips, persons, landuse)
                md = md[md['purpose'] == _p].copy() if 'purpose' in md.columns else md
                if len(md) < 30 or md.get('chosen_mode', pd.Series()).nunique() < 2:
                    return -1e9
                for col in mode_vars:
                    if col not in md.columns:
                        md[col] = 0.0
                m = NestedLogitModeModel()
                m.fit(md, _p)
                if m.ll_final_ is None or m.n_obs_ == 0:
                    return -1e9
                n   = m.n_obs_
                k   = int(genes.sum())
                bic = -2.0 * m.ll_final_ + k * np.log(max(n, 2))
                return -bic / max(n, 1)
            except Exception as e:
                logger.debug(f"Stage51 trip mode [{_p}]: {e}")
                return -1e9

        mode_ga  = GeneticAlgorithm(
            n_mode, _trip_mode_fit,
            feature_names       = mode_cands,
            collinear_groups    = STAGE_COLLINEAR_GROUPS.get('5_mode', []),
            require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get('5_mode', []),
            fixed_features   = [f for f in ['CRASH'] if f in mode_cands],
            **ga_kwargs,
        )
        mode_best, mode_hist, mode_pop = mode_ga.run_with_restarts(n_restarts=n_restarts)
        best_mode_vars = [v for v, g in zip(mode_cands, mode_best.genes) if g == 1]
        print(f"    Mode best vars: {best_mode_vars}  BIC/n={-mode_best.fitness:.4f}")

        mode_seg_result = GAResult(
            stage=51, best_features={'mode': best_mode_vars, 'dest': []},
            best_fitness=mode_best.fitness, best_individual=mode_best,
            history=mode_hist, all_candidates=mode_pop,
            feature_names=mode_cands, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=mode_ga.cache_hits,
        )
        mode_seg_result.metadata['sub_model'] = 'mode'
        segment_results[f'{purp_label}   Mode'] = mode_seg_result
        purp_best_mode_genes[purp] = mode_best.genes.copy()

        #  Dest GA
        print(f"\n  [Stage 51   {purp_label}] Dest GA ...")

        def _trip_dest_fit(genes, _p=purp):
            return _stage5_trip_fitness(
                mode_genes      = purp_best_mode_genes[_p],  # fixed from mode GA
                dest_genes      = genes,
                mode_candidates = mode_cands,
                dest_candidates = dest_cands,
                purpose         = _p,
                trips           = trips,
                persons         = persons,
                landuse         = landuse,
                skim_df         = skim_df,
            )

        dest_ga  = GeneticAlgorithm(
            len(dest_cands), _trip_dest_fit,
            feature_names       = dest_cands,
            collinear_groups    = STAGE_COLLINEAR_GROUPS.get('5_dest', []),
            require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get('5_dest', []),
            fixed_features   = [f for f in ['c_crash'] if f in dest_cands],
            **ga_kwargs,
        )
        dest_best, dest_hist, dest_pop = dest_ga.run_with_restarts(n_restarts=n_restarts)
        best_dest_vars = [v for v, g in zip(dest_cands, dest_best.genes) if g == 1]
        print(f"    Dest best vars: {best_dest_vars}  BIC/n={-dest_best.fitness:.4f}")

        dest_seg_result = GAResult(
            stage=51, best_features={'mode': best_mode_vars, 'dest': best_dest_vars},
            best_fitness=dest_best.fitness, best_individual=dest_best,
            history=dest_hist, all_candidates=dest_pop,
            feature_names=dest_cands, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=dest_ga.cache_hits,
        )
        dest_seg_result.metadata['sub_model'] = 'dest'
        segment_results[f'{purp_label}   Dest'] = dest_seg_result
        purp_best_dest_genes[purp] = dest_best.genes.copy()

    #  Majority-vote aggregate
    if purp_best_mode_genes:
        mode_vote  = np.array(list(purp_best_mode_genes.values())).mean(axis=0)
        consensus_mode = (mode_vote >= 0.5).astype(int)
        best_mode_feats = [v for v, g in zip(mode_cands, consensus_mode) if g == 1]
    else:
        consensus_mode  = np.zeros(n_mode, dtype=int)
        best_mode_feats = []

    if purp_best_dest_genes:
        dest_vote  = np.array(list(purp_best_dest_genes.values())).mean(axis=0)
        consensus_dest = (dest_vote >= 0.5).astype(int)
        best_dest_feats = [v for v, g in zip(dest_cands, consensus_dest) if g == 1]
    else:
        consensus_dest  = np.zeros(len(dest_cands), dtype=int)
        best_dest_feats = []

    if 'CRASH' in mode_cands and 'CRASH' not in best_mode_feats:
        best_mode_feats.append('CRASH')
        consensus_mode[mode_cands.index('CRASH')] = 1
    if 'c_crash' in dest_cands and 'c_crash' not in best_dest_feats:
        best_dest_feats.append('c_crash')
        consensus_dest[dest_cands.index('c_crash')] = 1

    consensus_genes = np.concatenate([consensus_mode, consensus_dest])
    try:
        agg_fit = _stage5_trip_fitness_aggregate(
            consensus_genes, n_mode, mode_cands, dest_cands,
            valid_purposes, trips, persons, landuse, skim_df)
    except Exception:
        agg_fit = -1e9

    if close_skims:
        skims.close()

    agg_cache = sum(r.cache_hits for r in segment_results.values())
    combined = GAResult(
        stage          = 51,
        best_features  = {'mode': best_mode_feats, 'dest': best_dest_feats},
        best_fitness   = agg_fit,
        best_individual= Individual(genes=consensus_genes, fitness=agg_fit),
        history        = [{
            'generation': 0, 'best_fitness': agg_fit, 'best_so_far': agg_fit,
            'mean_fitness': agg_fit, 'worst_fitness': agg_fit,
            'diversity': 0.0, 'mutation_rate': 0.0,
            'n_features': int(consensus_genes.sum()), 'stagnation': 0,
        }],
        all_candidates = [],
        feature_names  = mode_cands + dest_cands,
        fitness_metric = 'mean(-BIC/n)',
        n_restarts     = n_restarts,
        cache_hits     = agg_cache,
    )
    combined.metadata['segment_results'] = segment_results
    print(f"\n  Stage 51 (trip-level) aggregate: mode={best_mode_feats}  dest={best_dest_feats}")
    return combined


def _run_ga_stage6(trips, persons, landuse, ga_kwargs, n_restarts=2) -> GAResult:
    """
    GA for Stage 6 trip-level mode choice variable selection.
    Runs one GA per trip-purpose (work, edu, social, ÔÇª) then aggregates via
    majority-vote feature selection so the user gets per-purpose convergence plots
    (trip-level: work, school, mandatory/non-mandatory).
    """
    from stage6_mode_choice import prepare_mode_data, PURPOSES

    print("\n[Stage 6] Preparing estimation data...")
    data       = prepare_mode_data(trips, persons, landuse)
    feat_names = _stage6_candidates(data)
    purposes   = [p for p in PURPOSES
                  if len(data[data['purpose'] == p]) >= 30
                  and data[data['purpose'] == p]['chosen_mode'].nunique() >= 2]

    mode_dist = data['chosen_mode'].value_counts().to_dict()
    print(f"  {len(data)} trips | {len(feat_names)} features | "
          f"{len(purposes)} purposes: {purposes}")
    print(f"  HHTS mode distribution: {mode_dist}")

    #  Calibrated seed individuals 
    _core = {'tt_avg', 'log_dist', 'ldw', 'ldb', 'crash',
             'has_vehicle', 'no_vehicle', 'vehrd', 'accessibility'}
    _demo = _core | {'female', 'older', 'young', 'worker', 'student', 'incp', 'inc_lo'}
    _full = _demo | {'CARLICENCE', 'cost_proxy', 'age5p', 'inc_hi',
                     'distwalk', 'distbike'}

    def _make_seed(feature_set):
        return Individual(genes=np.array(
            [1 if f in feature_set else 0 for f in feat_names], dtype=int))

    #  Purpose group labels (trip-level) 
    PURP_LABELS = {
        'work':      'Work (trip)',
        'edu':       'School/Edu (trip)',
        'social':    'Social (trip)',
        'escort':    'Escort (trip)',
        'shopping':  'Shopping (trip)',
        'othmaint':  'Oth-Maint (trip)',
        'othdiscr':  'Oth-Discr (trip)',
        'eat':       'Eat Out (trip)',
    }

    #  Per-purpose GA runs (trip-level) 
    segment_results: dict = {}
    purp_best_genes: dict  = {}

    for purp in purposes:
        purp_label = PURP_LABELS.get(purp, f'{purp} (trip)')
        purp_data  = data[data['purpose'] == purp].copy()
        print(f"\n  [Stage 6   {purp_label}] n={len(purp_data)}  Running GA ...")

        def _purp_fitness(genes, _p=purp):
            return _stage6_fitness(genes, data, feat_names, _p)

        seeds = [_make_seed(s) for s in [_core, _demo, _full]]
        for s in seeds:
            try:   s.fitness = _purp_fitness(s.genes)
            except Exception: s.fitness = -1e9
        seeds.sort(key=lambda x: x.fitness, reverse=True)

        purp_ga   = GeneticAlgorithm(
            len(feat_names), _purp_fitness,
            feature_names       = feat_names,
            collinear_groups    = STAGE_COLLINEAR_GROUPS.get(6, []),
            require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get(6, []),
            fixed_features   = [f for f in ['crash'] if f in feat_names],
            **ga_kwargs,
        )
        purp_best, purp_hist, purp_pop = purp_ga.run_with_restarts(
            n_restarts=n_restarts, seed_individuals=seeds)

        purp_feats = [f for f, g in zip(feat_names, purp_best.genes) if g == 1]

        seg_result = GAResult(
            stage          = 6,
            best_features  = purp_feats,
            best_fitness   = purp_best.fitness,
            best_individual= purp_best,
            history        = purp_hist,
            all_candidates = purp_pop,
            feature_names  = feat_names,
            fitness_metric = 'mean(-BIC/n)',
            n_restarts     = n_restarts,
            cache_hits     = purp_ga.cache_hits,
        )
        segment_results[purp_label] = seg_result
        purp_best_genes[purp]       = purp_best.genes.copy()
        print(f"    {purp_label}: BIC/n={-purp_best.fitness:.4f}  feats={purp_feats}")

    #  Majority-vote aggregate
    if purp_best_genes:
        vote_matrix = np.array(list(purp_best_genes.values()))
        vote_frac   = vote_matrix.mean(axis=0)
        consensus   = (vote_frac >= 0.5).astype(int)

        agg_fitness_fn  = lambda g: _stage6_fitness_aggregate(g, data, feat_names, purposes)
        consensus_ind   = Individual(genes=consensus)
        consensus_ind.fitness = agg_fitness_fn(consensus)
        best_feats      = [f for f, g in zip(feat_names, consensus) if g == 1]
        best_result     = max(segment_results.values(), key=lambda r: r.best_fitness)
        agg_hist        = best_result.history
        agg_pop         = best_result.all_candidates
        agg_cache       = sum(r.cache_hits for r in segment_results.values())
    else:
        # Fallback: single aggregate GA
        def fitness(genes):
            return _stage6_fitness_aggregate(genes, data, feat_names, purposes)
        seeds = [_make_seed(s) for s in [_core, _demo, _full]]
        for s in seeds:
            try: s.fitness = fitness(s.genes)
            except Exception: s.fitness = -1e9
        seeds.sort(key=lambda x: x.fitness, reverse=True)
        ga  = GeneticAlgorithm(
            len(feat_names), fitness,
            feature_names       = feat_names,
            collinear_groups    = STAGE_COLLINEAR_GROUPS.get(6, []),
            require_one_groups  = STAGE_REQUIRE_ONE_GROUPS.get(6, []),
            fixed_features   = [f for f in ['crash'] if f in feat_names],
            **ga_kwargs,
        )
        agg_best, agg_hist, agg_pop = ga.run_with_restarts(
            n_restarts=n_restarts, seed_individuals=seeds)
        consensus_ind = agg_best
        best_feats    = [f for f, g in zip(feat_names, agg_best.genes) if g == 1]
        agg_cache     = ga.cache_hits

    combined = GAResult(
        stage          = 6,
        best_features  = best_feats,
        best_fitness   = consensus_ind.fitness,
        best_individual= consensus_ind,
        history        = agg_hist,
        all_candidates = agg_pop,
        feature_names  = feat_names,
        fitness_metric = 'mean(-BIC/n)',
        n_restarts     = n_restarts,
        cache_hits     = agg_cache,
    )
    combined.metadata['segment_results'] = segment_results
    print(f"\n  Stage 6 aggregate (majority vote): {best_feats}")
    return combined


# ---------------------------------------------------------------------------
# 5.  APPLY GA RESULTS TO PIPELINE  (prediction   run ONCE after GA)
# ---------------------------------------------------------------------------

def apply_ga_results_to_pipeline(
    ga_results:   AllStagesResult,
    out:          dict,
    synth:        pd.DataFrame,
    landuse:      pd.DataFrame,
    skims_path:   str,
    save_dir:     str = 'saved_data',
    seed:         int = 42,
) -> tuple:
    """
    Apply GA-selected features to run the full pipeline prediction once.

    This is the ONLY place prediction over the synthetic population happens.
    The GA only does estimation; once the best feature sets are found,
    this function re-estimates each model with those features and then
    runs the prediction over synth.

    Parameters
    ----------
    ga_results   : AllStagesResult from run_ga_all_stages()
    out          : from Synth_Connect().process_survey()
    synth        : enriched synthetic population
    landuse      : from sc.get_land()
    skims_path   : path to .omx skim file
    save_dir     : where to write outputs and saved models
    seed         : random seed

    Returns
    -------
    plan         : validated trip chain with mode + destination assigned
    models       : fitted Stage 5 model dict
    """
    def _invalid_stage5_segments(models: Any) -> List[str]:
        expected = ('work', 'edu', 'nonmand')
        bad: List[str] = []
        if not isinstance(models, dict):
            return list(expected)
        for seg in expected:
            seg_res = models.get(seg)
            if not isinstance(seg_res, dict):
                bad.append(seg)
                continue
            if seg_res.get('mode_model') is None or seg_res.get('dest_model') is None:
                bad.append(seg)
        return bad

    os.makedirs(save_dir, exist_ok=True)
    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    from stage1_activity_frequency import run_stage1
    from stage2_day_duration        import run_stage2
    from stage3_activity_duration   import run_stage3
    from stage4_schedule_placement  import run_stage4
    from stage5_destination_choice  import run_stage5
    from stage6_plan_finalisation   import run_stage6

    print("\n" + "="*60)
    print("APPLYING GA RESULTS   FULL PREDICTION PIPELINE")
    print("="*60)

    s1_feats = ga_results.results.get(1, None)
    s2_feats = ga_results.results.get(2, None)
    s3_feats = ga_results.results.get(3, None)
    s4_feats = ga_results.results.get(4, None)

    # ── Persist the final proposed feature sets ──────────────────────────
    try:
        import json as _json
        _feat_summary = {}
        for _sid, _res in sorted(ga_results.results.items()):
            if _res is None:
                continue
            _bf = getattr(_res, 'best_features', None)
            _ff = getattr(_res, 'best_fitness', float('nan'))
            _fm = getattr(_res, 'fitness_metric', '')
            if isinstance(_bf, dict):
                _feat_summary[f'stage{_sid}'] = {
                    'features': {str(k): list(v) if isinstance(v, (list, tuple, set)) else v
                                 for k, v in _bf.items()},
                    'best_fitness': float(_ff),
                    'fitness_metric': str(_fm),
                }
            elif isinstance(_bf, (list, tuple)):
                _feat_summary[f'stage{_sid}'] = {
                    'features': list(_bf),
                    'best_fitness': float(_ff),
                    'fitness_metric': str(_fm),
                }
        _feat_path = os.path.join(save_dir, 'selected_features_summary.json')
        with open(_feat_path, 'w', encoding='utf-8') as _fh:
            _json.dump(_feat_summary, _fh, indent=2, default=str)
        print(f"  Final selected features → {_feat_path}")
    except Exception as _fe:
        logger.warning(f"  selected_features_summary.json write failed: {_fe}")

    print("\n[1/5] Stage 1   activity frequency prediction...")
    synth_s1, s1_models = run_stage1(
        out["tours"], out["persons"], synth,
        feature_override=s1_feats.best_features if s1_feats else None,
    )
    synth_s1.to_csv(os.path.join(save_dir, 'stage1_output.csv'), index=False)
    _save_model_bundle(s1_models, save_dir, models_dir, 'stage1_models.pkl')
    _ga_save_stage_coef_csv(s1_models, 'stage1', save_dir)

    # ── Stage 2 + 3: joint Weibull AFT model or independent ──────────────────
    # When STAGE23_JOINT=1, the GA searched a shared feature pool and produced a
    # single JAX-fitted Weibull AFT budget-constrained model (Cho et al. 2023).
    # That model is used directly for prediction; independent run_stage2/run_stage3
    # are used as fallback when the joint model is unavailable.
    _s23_joint_res = ga_results.results.get(23)
    _s23_models = None
    if _s23_joint_res is not None:
        _s23_models = (getattr(_s23_joint_res, 'metadata', None) or {}).get('stage23_joint_models')
        if _s23_models is None:
            # Try loading from saved pkl (e.g. if resumed from checkpoint)
            import pickle as _pk
            _pkl_path = os.path.join(save_dir, 'stage23_joint_models.pkl')
            if os.path.exists(_pkl_path):
                with open(_pkl_path, 'rb') as _f:
                    _s23_models = _pk.load(_f)
                logger.info(f"Loaded joint Stage 2+3 model from {_pkl_path}")

    if _s23_models is not None and _s23_models.get('success'):
        print("\n[2+3/5] Stage 2+3   joint Weibull AFT budget-constrained prediction...")
        try:
            from stage23_joint import predict_joint_stage23
            synth_s2 = predict_joint_stage23(synth_s1, out["persons"], _s23_models, noise=True)
            # predict_joint_stage23 adds: start_of_day, end_of_day, B, <purp>_duration_min
            # Stage 3 activities DataFrame — expand synth-level duration columns to tour rows
            activities = out["tours"].copy()
            for _purp_col in [c for c in synth_s2.columns if c.endswith('_duration_min')]:
                _purp = _purp_col.replace('_duration_min', '')
                if 'person_id' in activities.columns and 'person_id' in synth_s2.columns:
                    _dur_map = synth_s2.set_index('person_id')[_purp_col]
                    _purp_mask = activities.get('main_purpose', activities.get('purpose', pd.Series(dtype=str))) == _purp
                    activities.loc[_purp_mask, 'duration_min'] = (
                        activities.loc[_purp_mask, 'person_id'].map(_dur_map)
                    )
            synth_s2.to_csv(os.path.join(save_dir, 'stage2_output.csv'), index=False)
            activities.to_csv(os.path.join(save_dir, 'stage3_output.csv'), index=False)
            # Save joint model bundle
            import pickle as _pk
            _s23_pkl = os.path.join(models_dir, 'stage23_joint_models.pkl')
            with open(_s23_pkl, 'wb') as _f:
                _pk.dump(_s23_models, _f)
            _ga_save_stage_coef_csv({'joint': _s23_models}, 'stage23_joint', save_dir)
            logger.info(f"Joint Stage 2+3 prediction complete — {len(synth_s2):,} persons")
        except Exception as _j23e:
            logger.error(f"Joint Stage 2+3 prediction failed: {_j23e}; falling back to independent", exc_info=True)
            _s23_models = None   # trigger fallback below

    if _s23_models is None or not _s23_models.get('success'):
        print("\n[2/5] Stage 2   day duration prediction...")
        synth_s2, s2_models = run_stage2(
            out["tours"], out["persons"], synth_s1,
            feature_override=s2_feats.best_features if s2_feats else None,
        )
        synth_s2.to_csv(os.path.join(save_dir, 'stage2_output.csv'), index=False)
        _save_model_bundle(s2_models, save_dir, models_dir, 'stage2_models.pkl')
        _ga_save_stage_coef_csv(s2_models, 'stage2', save_dir)

        print("\n[3/5] Stage 3   activity duration prediction...")
        activities, s3_models = run_stage3(
            out["tours"], out["persons"], synth_s2, trips=out["trips"],
            feature_override=s3_feats.best_features if s3_feats else None,
        )
        activities.to_csv(os.path.join(save_dir, 'stage3_output.csv'), index=False)
        _save_model_bundle(s3_models, save_dir, models_dir, 'stage3_models.pkl')
        _ga_save_stage_coef_csv(s3_models, 'stage3', save_dir)

    print("\n[4/5] Stage 4   schedule placement prediction...")
    scheduled, s4_models = run_stage4(
        out["trips"], out["tours"], out["persons"], activities,
        feature_override=s4_feats.best_features if s4_feats else None,
    )
    scheduled.to_csv(os.path.join(save_dir, 'stage4_output.csv'), index=False)
    _save_model_bundle(s4_models, save_dir, models_dir, 'stage4_models.pkl')
    _ga_save_stage_coef_csv(s4_models, 'stage4', save_dir)

    print("\n[5/5] Stage 5   destination + mode choice (zone-by-zone)...")
    s5_result = ga_results.results.get(5, None)
    _s5_bf = getattr(s5_result, 'best_features', None) if s5_result else None
    if isinstance(_s5_bf, dict):
        s5_mode_feats = _s5_bf.get('mode') or _s5_bf.get('work_mode')
        s5_dest_feats = _s5_bf.get('dest') or _s5_bf.get('work_dest')
    else:
        s5_mode_feats = None
        s5_dest_feats = None

    # Baseline-first gate: require an unperturbed Stage 5 fit before applying
    # GA feature perturbations. This prevents searching on top of fallback models.
    _prefit_baseline = os.environ.get('GA_STAGE5_PREFIT_BASELINE', '1').strip() != '0'
    _baseline_coef_warm = {}
    if _prefit_baseline:
        logger.info(
            "Stage 5 prefit: estimating baseline model (no GA overrides) "
            "before applying GA-selected perturbations."
        )
        _base_scheduled_dest, _base_s5_models = run_stage5(
            out["trips"], synth, scheduled,
            skims=skims_path,
            landuse=landuse,
            tours=out["tours"],
            seed=seed,
            mode_feature_override=None,
            dest_feature_override=None,
            save_path=os.path.join(save_dir, 'stage5_models_prefit.pkl'),
        )
        _base_bad_s5 = _invalid_stage5_segments(_base_s5_models)
        if _base_bad_s5:
            raise RuntimeError(
                "Stage 5 baseline prefit failed (incomplete models for segments: "
                f"{_base_bad_s5}). "
                "Fix baseline estimation before GA perturbations."
            )
        for _seg_res in _base_s5_models.values():
            if not isinstance(_seg_res, dict):
                continue
            _mc = _seg_res.get('mode_coef')
            _dc = _seg_res.get('dest_coef')
            if isinstance(_mc, dict):
                _baseline_coef_warm.update(_mc)
            if isinstance(_dc, dict):
                _baseline_coef_warm.update(_dc)
        logger.info(
            "Stage 5 prefit passed; applying GA overrides with baseline warm-start "
            f"({len(_baseline_coef_warm)} coefficients)."
        )

    scheduled_dest, s5_models = run_stage5(
        out["trips"], synth, scheduled,
        skims          = skims_path,
        landuse        = landuse,
        tours          = out["tours"],
        seed           = seed,
        mode_feature_override = s5_mode_feats,
        dest_feature_override = s5_dest_feats,
        coef_warm_start = _baseline_coef_warm,
        save_path       = os.path.join(save_dir, 'stage5_models.pkl'),
    )
    _bad_s5 = _invalid_stage5_segments(s5_models)
    if _bad_s5:
        _msg = (
            "GA Stage 5 produced incomplete models "
            f"(missing mode/dest submodels for segments: {_bad_s5})."
        )
        logger.warning(_msg)
        _allow_fallback = os.environ.get('GA_STAGE5_FALLBACK_TO_MAIN', '0').strip() != '0'
        if _allow_fallback:
            logger.warning(
                "Retrying Stage 5 with full baseline specification (no GA overrides) "
                "to avoid invalid downstream validation outputs."
            )
            scheduled_dest, s5_models = run_stage5(
                out["trips"], synth, scheduled,
                skims=skims_path,
                landuse=landuse,
                tours=out["tours"],
                seed=seed,
                mode_feature_override=None,
                dest_feature_override=None,
                save_path=os.path.join(save_dir, 'stage5_models.pkl'),
            )
            _bad_s5 = _invalid_stage5_segments(s5_models)
        if _bad_s5:
            raise RuntimeError(
                "Stage 5 models are incomplete after GA apply/retry; "
                f"missing mode/dest submodels for {_bad_s5}. "
                "Set GA_STAGE5_FALLBACK_TO_MAIN=1 to auto-retry baseline spec."
            )
    scheduled_dest.to_csv(os.path.join(save_dir, 'stage5_output.csv'), index=False)
    scheduled_dest.to_csv(os.path.join(save_dir, 'stage6_output.csv'), index=False)
    _save_model_bundle(s5_models, save_dir, models_dir, 'stage5_models.pkl')

    #  Save Stage 5 tour-model coefficient HTML (same as main pipeline) 
    try:
        from stage5_destination_choice import _save_coefficients as _s5_save_coeff
        _coef_csv = os.path.join(save_dir, 'stage5_models_coefficients.csv')
        _s5_save_coeff(s5_models, _coef_csv)
        print(f"  GA Stage 5 tour coefficients → {_coef_csv.replace('.csv', '.html')}")
    except Exception as _sce:
        logger.warning(f"  GA Stage 5 coefficient save failed: {_sce}")
        try:
            from pipeline_logger import _write_smoke_models_html
            _write_smoke_models_html(
                s5_models, save_dir,
                title='Stage 5 Final Model Coefficients',
            )
            print(f"  GA Stage 5 tour coefficient report (fallback HTML) → {save_dir}/stage5_models_coefficients.html")
        except Exception as _fbe:
            logger.warning(f"  GA Stage 5 fallback HTML also failed: {_fbe}")
    # Also save via unified GA coefficient writer (stage5_coefficients.html)
    _ga_save_stage_coef_csv(s5_models, 'stage5', save_dir)

    #  Location choice summary / validation 
    try:
        from stage5_destination_choice import save_location_choice_summary
        save_location_choice_summary(
            models   = s5_models,
            obs_tours = out.get('tours', pd.DataFrame()),
            landuse  = landuse,
            save_dir = save_dir,
            persons  = out.get('persons'),
        )
    except Exception as _lce:
        logger.warning(f"  GA location choice summary failed: {_lce}")

    #  Full HHTS validation (parity with main pipeline)
    _ga_run_validation_hhts(
        out=out,
        s1_models=s1_models,
        s2_models=s2_models,
        s3_models=s3_models,
        s4_models=s4_models,
        s5_models=s5_models,
        save_dir=save_dir,
    )

    #  Trip-level mode + destination models 
    # Estimate separate trip-level models so the GA run produces the same
    # artefacts as the main pipeline.
    try:
        from stage6_mode_choice import (
            estimate_mode_models as _est_trip_mode,
            save_models as _s6_save,
            _save_s6_coefficients,
            _write_s6_html_report,
        )
        from stage5_destination_choice import (
            _skim_flat as _s5_skim_flat,
            _run_segment as _s5_run_seg,
            save_models as _s5_save_trip,
            _save_coefficients as _s5_save_trip_coeff,
        )
        import openmatrix as omx

        # Trip mode models
        _trip_mode_mdls = _est_trip_mode(out["trips"], out.get("persons", pd.DataFrame()), landuse)
        _save_model_bundle(_trip_mode_mdls, save_dir, models_dir, 'stage5_trip_mode_models.pkl')
        _save_s6_coefficients(_trip_mode_mdls,
                              os.path.join(save_dir, 'stage5_trip_mode_coefficients.csv'))
        _write_s6_html_report(
            _trip_mode_mdls,
            os.path.join(save_dir, 'stage5_trip_mode_coefficients.html'),
        )
        print(f"  GA trip mode models ÔåÆ {save_dir}/stage5_trip_mode_models.pkl")

        # Trip destination models
        _skims_ga = omx.open_file(skims_path, 'r')
        _skim_df_ga = _s5_skim_flat(_skims_ga)
        _trip_like = _build_trip_like_table(out["trips"], out.get("persons", pd.DataFrame()))
        _trip_dest_mdls = {}
        for _seg in ['work', 'edu', 'nonmand']:
            try:
                _r = _s5_run_seg(_seg, _trip_like,
                                 out.get("persons", pd.DataFrame()), landuse,
                                 _skims_ga, _skim_df_ga, seed=seed)
                if _r:
                    _trip_dest_mdls[_seg] = _r
            except Exception as _se:
                logger.debug(f"  GA trip dest [{_seg}]: {_se}")
        if _trip_dest_mdls:
            _save_model_bundle(_trip_dest_mdls, save_dir, models_dir, 'stage5_trip_dest_models.pkl')
            _s5_save_trip_coeff(_trip_dest_mdls,
                                os.path.join(save_dir, 'stage5_trip_dest_coefficients.csv'))
            print(f"  GA trip dest models ÔåÆ {save_dir}/stage5_trip_dest_models.pkl")
        if _trip_mode_mdls or _trip_dest_mdls:
            _save_model_bundle(
                {
                    'trip_mode_models': _trip_mode_mdls,
                    'trip_dest_models': _trip_dest_mdls,
                },
                save_dir,
                models_dir,
                'stage5b_models.pkl',
            )
        _skims_ga.close()
    except Exception as _tte:
        logger.warning(f"  GA trip model estimation failed: {_tte}")

    #  Predictions vs HHTS validation 
    try:
        _ga_validate_vs_hhts(s5_models, out, scheduled_dest, landuse, save_dir)
    except Exception as _ve:
        logger.warning(f"  GA HHTS validation failed: {_ve}")

    print(f"\n{'='*60}")
    print(f"GA pipeline prediction complete.")
    print(f"  Persons: {synth['person_id'].nunique() if 'person_id' in synth.columns else len(synth):,}")
    print(f"  Trips:   {(scheduled_dest['row_type']=='trip').sum() if 'row_type' in scheduled_dest.columns else len(scheduled_dest):,}")
    print(f"  Models saved ÔåÆ {models_dir}/")
    print(f"  Coefficients ÔåÆ {save_dir}/stage5_models_coefficients.html")

    _write_ga_vs_main_tex(ga_results, save_dir)

    # Metacount mixed (random-coefficient) FINAL models for stages 1-3.
    # Re-run the best GA-selected features through the mixed metacount estimator
    # at full draw count → NB2 (stage 1) / lognormal (stages 2-3) with RANDOM
    # coefficients, saved as coefficient HTML.
    #
    # Runs when the base estimator is a metacount variant OR whenever random-
    # coefficient search is enabled (default-on, see ga_constraints.
    # rand_search_enabled) — so a plain `default` staged run still produces the
    # mixed random-coefficient finals for stages 1-3. Requires metacountregressor
    # to be importable; falls back silently (each stage is wrapped in try/except)
    # when it is not, and the primary OrderedLogit/Tobit prediction models are
    # untouched.
    _mc_est = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    try:
        from ga_constraints import rand_search_enabled as _rse_gate
        from ga_external_estimators import _METACOUNT_OK as _mc_ok_gate
        _run_mixed_finals = (
            (_mc_est in ('metacount', 'both', 'hybrid', 'sa_bandit') or _rse_gate())
            and _mc_ok_gate
        )
    except Exception:
        _run_mixed_finals = _mc_est in ('metacount', 'both', 'hybrid', 'sa_bandit')
    if _run_mixed_finals:
        try:
            from ga_external_estimators import (
                stage1_final_mixed_metacount  as _s1_mc,
                stage23_final_mixed_metacount as _s23_mc,
                _write_metacount_html,
            )
            from stage1_activity_frequency import prepare_estimation_data as _prep_s1
            from stage2_day_duration       import prepare_data            as _prep_s2
            from stage3_activity_duration  import prepare_data            as _prep_s3

            _R_final = int(os.environ.get('MCR_R_FINAL', '500'))
            print(f"\n  Metacount final-model estimation (R={_R_final}) ...")

            # Stage 1
            try:
                _s1_data = _prep_s1(out['tours'], out['persons'])
                _s1_feats = ga_results.results.get(1, None)
                _s1_vars  = _s1_feats.best_features if _s1_feats else []
                if _s1_vars:
                    _purps = [p for p in ['work', 'edu', 'nonmand', 'social', 'shopping']
                              if f'n_{p}_out' in _s1_data.columns]
                    _s1_scores, _s1_coefs = _s1_mc(_s1_data, _s1_vars, _purps, R=_R_final)
                    _write_metacount_html(
                        _s1_coefs,
                        os.path.join(save_dir, 'stage1_metacount_coefficients.html'),
                        'Stage 1   MetaCount NB2 (mixed) coefficients',
                    )
                    print(f"  Stage 1 metacount HTML ÔåÆ {save_dir}/stage1_metacount_coefficients.html")
            except Exception as _e1:
                logger.warning(f"  Stage 1 metacount final fit failed: {_e1}")

            # Stage 2
            try:
                _s2_data  = _prep_s2(out['tours'], out['persons'])
                _s2_feats = ga_results.results.get(2, None)
                _s2_vars  = _s2_feats.best_features if _s2_feats else []
                if _s2_vars:
                    _s2_coef_map = {}
                    for _ycol in ['start_of_day', 'end_of_day']:
                        if _ycol in _s2_data.columns:
                            _sc, _cr = _s23_mc(_s2_data, _s2_vars, _ycol, 'Stage2', R=_R_final)
                            _s2_coef_map[_ycol] = _cr
                    _write_metacount_html(
                        _s2_coef_map,
                        os.path.join(save_dir, 'stage2_metacount_coefficients.html'),
                        'Stage 2   MetaCount lognormal (mixed) coefficients',
                    )
                    print(f"  Stage 2 metacount HTML ÔåÆ {save_dir}/stage2_metacount_coefficients.html")
            except Exception as _e2:
                logger.warning(f"  Stage 2 metacount final fit failed: {_e2}")

            # Stage 3
            try:
                _s3_data  = _prep_s3(out['tours'], out['persons'], trips=out.get('trips'))
                _s3_feats = ga_results.results.get(3, None)
                _s3_vars  = _s3_feats.best_features if _s3_feats else []
                if _s3_vars:
                    _dur_col = next((c for c in ['duration', 'dur_min', 'act_duration'] if c in _s3_data.columns), None)
                    _s3_coef_map = {}
                    if _dur_col:
                        _sc, _cr = _s23_mc(_s3_data, _s3_vars, _dur_col, 'Stage3', R=_R_final)
                        _s3_coef_map[_dur_col] = _cr
                    _write_metacount_html(
                        _s3_coef_map,
                        os.path.join(save_dir, 'stage3_metacount_coefficients.html'),
                        'Stage 3   MetaCount lognormal (mixed) coefficients',
                    )
                    print(f"  Stage 3 metacount HTML ÔåÆ {save_dir}/stage3_metacount_coefficients.html")
            except Exception as _e3:
                logger.warning(f"  Stage 3 metacount final fit failed: {_e3}")

        except Exception as _mc_html_e:
            logger.warning(f"  Metacount HTML report generation failed: {_mc_html_e}")

    return scheduled_dest, s5_models


def _build_trip_like_table(trips: pd.DataFrame,
                           persons: pd.DataFrame) -> pd.DataFrame:
    """
    Convert HHTS trip legs to a tours-like DataFrame for _run_segment.
    Mirrors the pipeline_logger._trips_to_seg_table helper.
    """
    if trips.empty:
        return pd.DataFrame()
    t = trips.copy()

    mode_col = next((c for c in ['mode', 'MAINMODE', 'TRAVMODE']
                     if c in t.columns), None)
    _mode_map = {'DA': 1, 'SR': 2, 'Walk': 3, 'Bike': 4, 'Transit': 5}
    if mode_col and 'tour_mode' not in t.columns:
        t['tour_mode'] = t[mode_col].map(_mode_map).fillna(2).astype(int)
    elif 'tour_mode' not in t.columns:
        t['tour_mode'] = 1

    purp_col = next((c for c in ['purpose', 'main_purpose', 'TRIPTYPE']
                     if c in t.columns), None)
    if purp_col and 'primary_purpose' not in t.columns:
        _purp_remap = {
            'work': 'work', 'work_related': 'work',
            'edu': 'edu', 'school': 'edu', 'education': 'edu',
            'social': 'nonmand', 'escort': 'nonmand', 'shopping': 'nonmand',
            'othmaint': 'nonmand', 'othdiscr': 'nonmand', 'eat': 'nonmand',
            'recreation': 'nonmand', 'personal': 'nonmand', 'other': 'nonmand',
        }
        t['primary_purpose'] = (t[purp_col].astype(str).str.lower().str.strip()
                                 .map(lambda p: _purp_remap.get(p, 'nonmand')))

    for zcol, alts in [('OTAZ', ['ORIGZONE', 'orig_taz']),
                       ('DTAZ', ['DESTZONE', 'dest_taz'])]:
        if zcol not in t.columns:
            for a in alts:
                if a in t.columns:
                    t[zcol] = t[a]
                    break
    for zcol in ['OTAZ', 'DTAZ']:
        if zcol in t.columns:
            t[zcol] = pd.to_numeric(t[zcol], errors='coerce').fillna(0).astype(int)

    # Merge person attributes
    if persons is not None and not persons.empty:
        pid_col = next((c for c in ['person_id', 'PERSONID'] if c in persons.columns), None)
        t_pid   = next((c for c in ['person_id', 'PERSONID'] if c in t.columns), None)
        if pid_col and t_pid:
            _pcols = [pid_col] + [c for c in persons.columns
                                  if c != pid_col and c not in t.columns
                                  and c in ['ptype', 'WORKSTAT', 'STUDYING', 'vehrd',
                                             'has_vehicle', 'has_licence', 'indp']]
            t = t.merge(persons[_pcols], left_on=t_pid, right_on=pid_col,
                        how='left', suffixes=('', '_person'))

    if 'weight' not in t.columns:
        t['weight'] = 1.0
    return t.reset_index(drop=True)


def _ga_validate_vs_hhts(s5_models: dict, out: dict,
                          predicted_plan: pd.DataFrame,
                          landuse: pd.DataFrame,
                          save_dir: str) -> None:
    """
    Compare GA-model predictions to HHTS observed trips/tours.

    Writes  ga_hhts_validation.html  and  ga_hhts_validation.csv
    with mode share, destination distance distribution, and zone-level
    assignment metrics so the GA model quality can be assessed.
    """
    rows = []

    #  Mode share comparison 
    try:
        _hhts_trips = out.get('trips', pd.DataFrame())
        _mode_col = next((c for c in ['mode', 'MAINMODE', 'tour_mode']
                          if c in _hhts_trips.columns), None)
        if _mode_col and not _hhts_trips.empty:
            obs_share = (_hhts_trips[_mode_col]
                         .value_counts(normalize=True) * 100)
            for mode, share in obs_share.items():
                rows.append({
                    'category': 'mode_share',
                    'dimension': str(mode),
                    'observed': round(float(share), 2),
                    'predicted': None,
                    'unit': '%',
                })

        # Predicted mode share from plan
        if 'mode_name' in predicted_plan.columns:
            trip_rows = (predicted_plan[predicted_plan['row_type'] == 'trip']
                         if 'row_type' in predicted_plan.columns else predicted_plan)
            pred_share = trip_rows['mode_name'].value_counts(normalize=True) * 100
            for mode, share in pred_share.items():
                existing = next((r for r in rows
                                 if r['category'] == 'mode_share'
                                 and r['dimension'] == str(mode)), None)
                if existing:
                    existing['predicted'] = round(float(share), 2)
                else:
                    rows.append({
                        'category': 'mode_share',
                        'dimension': str(mode),
                        'observed': None,
                        'predicted': round(float(share), 2),
                        'unit': '%',
                    })
    except Exception as _me:
        logger.debug(f"  GA mode share comparison: {_me}")

    #  Trip distance distribution 
    try:
        _dist_col = next((c for c in ['dist', 'SOVTOLL_DIST__AM', 'travel_dist']
                          if c in predicted_plan.columns), None)
        if _dist_col:
            trip_rows = (predicted_plan[predicted_plan['row_type'] == 'trip']
                         if 'row_type' in predicted_plan.columns else predicted_plan)
            dist_vals = pd.to_numeric(trip_rows[_dist_col], errors='coerce').dropna()
            for stat_name, stat_fn in [
                ('dist_mean_km',   lambda x: x.mean()),
                ('dist_median_km', lambda x: x.median()),
                ('dist_p90_km',    lambda x: x.quantile(0.90)),
            ]:
                rows.append({
                    'category': 'trip_distance',
                    'dimension': stat_name,
                    'observed': None,
                    'predicted': round(float(stat_fn(dist_vals)), 3),
                    'unit': 'km',
                })
    except Exception as _de:
        logger.debug(f"  GA distance distribution: {_de}")

    #  Tour-level LL (how well GA models fit observed HHTS tours) 
    try:
        for seg, res in s5_models.items():
            for mtype in ('mode', 'dest'):
                ll = res.get(f'{mtype}_ll', None)
                rho2 = float('nan')
                mdl = res.get(f'{mtype}_model')
                if mdl is not None:
                    try:
                        rho2 = float(mdl.rho_sq_null())
                    except Exception:
                        pass
                if ll is not None:
                    rows.append({
                        'category': f'tour_{mtype}_LL',
                        'dimension': seg,
                        'observed': round(float(ll), 2),
                        'predicted': round(rho2, 4) if np.isfinite(rho2) else None,
                        'unit': f'LL / rho2',
                    })
    except Exception as _le:
        logger.debug(f"  GA LL extraction: {_le}")

    #  Write outputs 
    if rows:
        df_val = pd.DataFrame(rows)
        csv_path  = os.path.join(save_dir, 'ga_hhts_validation.csv')
        html_path = os.path.join(save_dir, 'ga_hhts_validation.html')
        df_val.to_csv(csv_path, index=False)

        _mode_rows  = df_val[df_val['category'] == 'mode_share']
        _dist_rows  = df_val[df_val['category'] == 'trip_distance']
        _ll_rows    = df_val[df_val['category'].str.contains('_LL')]

        def _tbl(sub_df, headers, row_fn):
            rows_html = '\n'.join(row_fn(r) for _, r in sub_df.iterrows())
            hdrs = ''.join(f'<th>{h}</th>' for h in headers)
            return f"<table><tr>{hdrs}</tr>{rows_html}</table>"

        mode_tbl = _tbl(
            _mode_rows, ['Mode', 'Observed %', 'Predicted %'],
            lambda r: (f"<tr><td>{r['dimension']}</td>"
                       f"<td style='text-align:right'>{r['observed'] if r['observed'] is not None else ' ':.1f}</td>"
                       f"<td style='text-align:right'>{r['predicted'] if r['predicted'] is not None else ' ':.1f}</td></tr>")
        ) if not _mode_rows.empty else '<p>No mode data</p>'

        dist_tbl = _tbl(
            _dist_rows, ['Metric', 'Predicted'],
            lambda r: f"<tr><td>{r['dimension']}</td><td>{r['predicted']:.2f} {r['unit']}</td></tr>"
        ) if not _dist_rows.empty else '<p>No distance data</p>'

        ll_tbl = _tbl(
            _ll_rows, ['Metric', 'Segment', 'LL', 'rho┬▓'],
            lambda r: (f"<tr><td>{r['category']}</td><td>{r['dimension']}</td>"
                       f"<td style='text-align:right'>{r['observed']:.1f}</td>"
                       f"<td style='text-align:right'>{r['predicted'] if r['predicted'] is not None else 'n/a'}</td></tr>")
        ) if not _ll_rows.empty else '<p>No LL data</p>'

        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>GA HHTS Validation</title>
<style>body{{font-family:Arial,sans-serif;margin:20px;font-size:13px}}
table{{border-collapse:collapse;width:80%;margin-bottom:24px}}
th{{background:#2c3e50;color:white;padding:6px 10px;text-align:left}}
td{{border:1px solid #ddd;padding:5px 10px}}
tr:nth-child(even){{background:#f9f9f9}}
h1,h2{{color:#2c3e50}}</style></head><body>
<h1>GA Final Model   Predictions vs HHTS</h1>
<h2>Mode Share (observed HHTS vs predicted plan)</h2>{mode_tbl}
<h2>Trip Distance Distribution (predicted plan)</h2>{dist_tbl}
<h2>Model Fit (tour-level LL and ¤ü┬▓)</h2>{ll_tbl}
</body></html>"""

        with open(html_path, 'w', encoding='utf-8') as _hf:
            _hf.write(html)
        print(f"  GA HHTS validation ÔåÆ {html_path}")
        logger.info(f"  GA HHTS validation CSV ÔåÆ {csv_path}")


def _save_model(model, path: str) -> None:
    """Silently save a model object (dict, fitted estimator, etc.) to pickle.

    Uses cloudpickle, not stdlib pickle: Larch model objects (e.g. Stage 5's
    mode/dest models) carry lambdas (DictOfAlts.<lambda> in larch.model.linear)
    that stdlib pickle cannot serialize ("Can't pickle <function ...>: attribute
    lookup ... failed"), silently producing an empty/missing file. This mirrors
    the save path stage5_destination_choice.py already uses successfully.
    """
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            try:
                import cloudpickle
                cloudpickle.dump(model, f)
            except ImportError:
                pickle.dump(model, f)
        print(f"    Model saved ÔåÆ {path}")
    except Exception as e:
        logger.warning(f"Could not save model to {path}: {e}")


def _save_model_bundle(model, save_dir: str, models_dir: str, *filenames: str) -> None:
    """Save a pickle both at the run root and under models/."""
    seen = set()
    for filename in filenames:
        for folder in (save_dir, models_dir):
            path = os.path.join(folder, filename)
            if path in seen:
                continue
            seen.add(path)
            _save_model(model, path)


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _latest_main_run_dir(save_dir: str) -> Optional[str]:
    """Pick the most recent main run directory with comparable outputs."""
    run_root = os.environ.get('PIPELINE_RUN_ROOT', 'runs')
    candidate_roots = [
        os.path.dirname(os.path.abspath(save_dir)),
        os.path.join(os.getcwd(), run_root),
    ]
    candidates = []
    for base in candidate_roots:
        if not os.path.isdir(base):
            continue
        candidates.extend(glob.glob(os.path.join(base, 'main__*')))
        candidates.extend(glob.glob(os.path.join(base, 'main_*')))

    valid = []
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if any(
            os.path.exists(os.path.join(path, filename))
            for filename in ('stage6_output.csv', 'stage5_output.csv', 'validation_report.html')
        ):
            valid.append(path)
    if not valid:
        return None
    valid.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return valid[0]


def _load_plan_for_compare(run_dir: Optional[str]) -> pd.DataFrame:
    if not run_dir:
        return pd.DataFrame()
    for filename in ('stage6_output.csv', 'stage5_output.csv'):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as exc:
                logger.warning(f"Could not read comparison plan {path}: {exc}")
    return pd.DataFrame()


def _plan_shape_metrics(plan: pd.DataFrame) -> dict:
    if plan.empty:
        return {'persons': None, 'trip_rows': None, 'activity_rows': None, 'rows': 0}
    row_type = plan['row_type'] if 'row_type' in plan.columns else pd.Series('', index=plan.index)
    trip_rows = int((row_type == 'trip').sum()) if 'row_type' in plan.columns else None
    activity_rows = int((row_type == 'activity').sum()) if 'row_type' in plan.columns else None
    persons = int(plan['person_id'].nunique()) if 'person_id' in plan.columns else None
    return {
        'persons': persons,
        'trip_rows': trip_rows,
        'activity_rows': activity_rows,
        'rows': int(len(plan)),
    }


def _write_ga_vs_main_tex(
    ga_results: 'AllStagesResult',
    save_dir: str,
    main_dir: Optional[str] = None,
) -> Optional[str]:
    """Write a LaTeX summary comparing the current GA run to the latest main run."""
    main_dir = main_dir or _latest_main_run_dir(save_dir)
    tex_path = os.path.join(save_dir, 'ga_vs_main_model.tex')

    ga_plan = _load_plan_for_compare(save_dir)
    main_plan = _load_plan_for_compare(main_dir)
    ga_metrics = _plan_shape_metrics(ga_plan)
    main_metrics = _plan_shape_metrics(main_plan)

    estimator_rows = []
    summary = ga_results.summary_table()
    if not summary.empty:
        for _, row in summary.iterrows():
            estimator_rows.append(
                f"{_tex_escape(row.get('stage_label', row.get('stage', 'stage')))} & "
                f"{_tex_escape(row.get('metric', ''))} & "
                f"{row.get('best_fitness', float('nan')):.4f} \\\\"
            )

    artifact_names = [
        'stage1_models.pkl', 'stage2_models.pkl', 'stage3_models.pkl',
        'stage4_models.pkl', 'stage5_models.pkl', 'stage5_trip_mode_models.pkl',
        'stage5_trip_dest_models.pkl', 'stage5b_models.pkl',
    ]
    artifact_rows = []
    for name in artifact_names:
        ga_exists = os.path.exists(os.path.join(save_dir, name))
        main_exists = os.path.exists(os.path.join(main_dir, name)) if main_dir else False
        artifact_rows.append(
            f"{_tex_escape(name)} & {'yes' if ga_exists else 'no'} & {'yes' if main_exists else 'no'} & "
            f"{'yes' if ga_exists == main_exists else 'no'} \\\\"
        )

    comparison_rows = []
    for label, key in [
        ('Persons', 'persons'),
        ('Trip rows', 'trip_rows'),
        ('Activity rows', 'activity_rows'),
        ('Total rows', 'rows'),
    ]:
        ga_val = ga_metrics.get(key)
        main_val = main_metrics.get(key)
        match = (ga_val == main_val) if ga_val is not None and main_val is not None else False
        comparison_rows.append(
            f"{label} & {_tex_escape(ga_val if ga_val is not None else 'n/a')} & "
            f"{_tex_escape(main_val if main_val is not None else 'n/a')} & "
            f"{'yes' if match else 'no'} \\\\"
        )

    estimator_table = '\n'.join(estimator_rows) if estimator_rows else 'No stage metrics were available.\\'
    artifact_table = '\n'.join(artifact_rows)
    comparison_table = '\n'.join(comparison_rows)
    baseline_text = _tex_escape(main_dir) if main_dir else 'No main run found'
    ga_dir_text = _tex_escape(save_dir)
    estimator_name = _tex_escape(os.environ.get('GA_ESTIMATOR', 'default'))

    tex = f"""% Auto-generated GA vs main comparison summary
\\section*{{GA Estimator Comparison Summary}}

Current GA estimator override: \\texttt{{{estimator_name}}}.\\
Current GA validation directory: \\texttt{{{ga_dir_text}}}.\\
Main-model comparison directory: \\texttt{{{baseline_text}}}.\\

\\subsection*{{Estimator-Specific GA Stages}}
\\begin{{tabular}}{{lll}}
Stage & Fitness metric & Best fitness \\\\
\\hline
{estimator_table}
\\end{{tabular}}

\\subsection*{{Prediction Shape Check}}
These counts should usually remain aligned with the main-model run if the same synthetic population and schedule skeleton are being propagated.\\
\\begin{{tabular}}{{lrrl}}
Measure & GA run & Main run & Match \\\\
\\hline
{comparison_table}
\\end{{tabular}}

\\subsection*{{Pickle Availability}}
The GA validation path now saves root-level pickles so downstream tooling can treat the GA output like a normal run directory.\\
\\begin{{tabular}}{{llll}}
Artifact & GA run & Main run & Same availability \\\\
\\hline
{artifact_table}
\\end{{tabular}}
"""

    try:
        with open(tex_path, 'w', encoding='utf-8') as handle:
            handle.write(tex.strip() + '\n')
        print(f"GA vs main TeX summary ÔåÆ {tex_path}")
        return tex_path
    except Exception as exc:
        logger.warning(f"Could not write GA vs main TeX summary: {exc}")
        return None


def generate_ga_validation(
    ga_results:   AllStagesResult,
    out:          dict,
    synth:        pd.DataFrame,
    landuse:      pd.DataFrame,
    skims_path:   str,
    save_dir:     str = 'saved_data/validation',
) -> pd.DataFrame:
    """
    Run the full pipeline with GA-selected features, save all models, and
    generate validation plots identical to a main pipeline run.

    Steps
    -----
    1. Re-estimate every stage model with GA best features, predict over synth.
    2. Save all models to  save_dir/models/*.pkl
    3. Generate activity plan plots (population summary + sample plans) via
       plot_activity_plans.plot_all()
    4. Save GA convergence overview to  save_dir/plots/ga_convergence.png
    5. Write GA feature-selection summary CSV to  save_dir/ga_summary.csv

    Parameters
    ----------
    ga_results   : AllStagesResult from run_ga_all_stages()
    out          : from Synth_Connect().process_survey()
    synth        : enriched synthetic population
    landuse      : from sc.get_land()
    skims_path   : path to .omx skim file
    save_dir     : root output directory

    Returns
    -------
    plan         : final trip-chain DataFrame (same as main pipeline output)
    """
    from plot_activity_plans import plot_all as _plot_activity_plans

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print("\nApplying GA results to full pipeline (validation run)...")
    plan, models = apply_ga_results_to_pipeline(
        ga_results, out, synth, landuse, skims_path,
        save_dir=save_dir, seed=42,
    )

    # ÔöÇÔöÇ Activity plan validation plots (same as main pipeline) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    print(f"\nGenerating activity plan plots ÔåÆ {plot_dir}/")
    try:
        _plot_activity_plans(plan, n_sample=12, save_dir=plot_dir)
    except Exception as e:
        logger.warning(f"Activity plan plots failed: {e}")

    # ÔöÇÔöÇ GA convergence overview ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    conv_path = os.path.join(plot_dir, 'ga_convergence.png')
    print(f"Saving GA convergence overview ÔåÆ {conv_path}")
    try:
        ga_results.plot_all(save_path=conv_path)
    except Exception as e:
        logger.warning(f"GA convergence plot failed: {e}")

    # ÔöÇÔöÇ Per-stage BIC/n summary bar chart ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    _plot_bic_summary(ga_results, save_dir=plot_dir)

    # ÔöÇÔöÇ GA feature-selection summary CSV ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    summary = ga_results.summary_table()
    summary_path = os.path.join(save_dir, 'ga_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"GA summary ÔåÆ {summary_path}")
    print(summary.to_string(index=False))

    print(f"\nValidation complete.  All outputs ÔåÆ {save_dir}/")
    return plan


def _plot_bic_summary(ga_results: 'AllStagesResult', save_dir: str = '.') -> None:
    """Bar chart of final BIC/n per stage (and per sub-model when available)."""
    import matplotlib.pyplot as plt

    rows = []
    for stage, res in sorted(ga_results.results.items(),
                              key=lambda x: (x[0] < 0, x[0])):
        segs = res.metadata.get('segment_results', {})
        if segs:
            for label, sr in segs.items():
                if sr.best_fitness > -1e8 and 'BIC' in sr.fitness_metric:
                    rows.append({'label': label, 'bic_n': -sr.best_fitness})
        elif res.best_fitness > -1e8 and 'BIC' in res.fitness_metric:
            stage_labels = {0: 'CDAP', -1: 'Usual Loc', 1: 'Stage1',
                            2: 'Stage2', 3: 'Stage3', 4: 'Stage4',
                            5: 'Stage5', 51: 'Stage51', 6: 'Stage6'}
            rows.append({'label': stage_labels.get(stage, str(stage)),
                         'bic_n': -res.best_fitness})

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.55), 5), facecolor='white')
    ax.barh(df['label'], df['bic_n'], color='#534AB7', alpha=0.8)
    ax.set_xlabel('Final BIC / n  (lower = better)', fontsize=10)
    ax.set_title('GA Variable Selection   Final BIC/n per Sub-Model', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(df['bic_n']):
        ax.text(v + 0.01 * max(df['bic_n']), i, f'{v:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bic_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"BIC summary chart ÔåÆ {path}")


# ---------------------------------------------------------------------------
# 6.  PUBLIC API
# ---------------------------------------------------------------------------

def run_ga_stage(
    stage:            int,
    tours:            Optional[pd.DataFrame] = None,
    trips:            Optional[pd.DataFrame] = None,
    persons:          Optional[pd.DataFrame] = None,
    skims                                    = None,
    landuse:          Optional[pd.DataFrame] = None,
    pop_size:         int   = 40,
    n_generations:    int   = 60,
    mutation_rate:    float = 0.08,
    crossover_rate:   float = 0.75,
    elitism_k:        int   = 4,
    min_features:     int   = 5,
    max_features:     int   = None,
    tournament_k:     int   = 4,
    stagnation_limit: int   = 15,
    n_restarts:       int   = 2,
    seed:             int   = 42,
    verbose:          bool  = True,
    stage_overrides:  Optional[Dict[int, Dict[str, Any]]] = None,
) -> GAResult:
    """
    Run GA variable selection for one pipeline stage (estimation only).

    Stage codes
    -----------
     0  : CDAP   daily activity pattern feature selection (H/N/M MNL)
     -1  : Usual location   decay_b + attractor column selection
     1  : Stage 1   activity frequency (ordered logit)
     2  : Stage 2   day duration (OLS)
     3  : Stage 3   activity duration (lognormal)
     4  : Stage 4   schedule placement (MNL timing)
     5  : Stage 5   destination + mode choice (Larch NL, TOUR level)
     51  : Stage 5b   trip-level mode + destination choice (HHTS trip legs)
     6  : Stage 6   mode choice (nested logit, TRIP level, no destination)
    """
    logging.basicConfig(level=logging.WARNING)

    ga_kwargs = dict(
        pop_size          = pop_size,
        n_generations     = n_generations,
        mutation_rate     = mutation_rate,
        crossover_rate    = crossover_rate,
        elitism_k         = elitism_k,
        min_features      = min_features,
        max_features      = max_features,
        tournament_k      = tournament_k,
        stagnation_limit  = stagnation_limit,
        seed              = seed,
        verbose           = verbose,
    )

    # Apply stage-specific overrides if provided
    if stage_overrides and stage in stage_overrides:
        override = stage_overrides[stage]
        ga_kwargs.update(override)
        if verbose:
            print(f"  Stage {stage} overrides applied: {override}")

    stage_labels = {
         0: 'CDAP   daily activity pattern',
        -1: 'Usual location   decay + attractors',
         1: 'Stage 1   activity frequency',
         2: 'Stage 2   day duration',
         3: 'Stage 3   activity duration',
         4: 'Stage 4   schedule placement',
         5: 'Stage 5   tour-level destination + mode',
        51: 'Stage 51   trip-level mode + destination',
         6: 'Stage 6   mode choice (trip, no destination)',
    }
    label = stage_labels.get(stage, f'Stage {stage}')
    # ── detect active algorithm so the header reflects the real mode ──────────
    _est_env   = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    _is_bandit_sa = (os.environ.get('GA_USE_BANDIT_SA', '0') == '1'
                     or _est_env == 'sa_bandit')
    _is_bandit    = os.environ.get('GA_USE_BANDIT', '0') == '1' or _is_bandit_sa
    _is_sa        = (os.environ.get('GA_USE_SA', '0') == '1'
                     or _est_env in ('metacount', 'both', 'hybrid'))
    if _is_bandit_sa:
        _algo_tag = 'SA-Bandit (Thompson sampling + SA acceptance)'
    elif _is_bandit:
        _algo_tag = 'Bandit (Thompson sampling)'
    elif _is_sa:
        _algo_tag = f'SA  [{_est_env}]'
    else:
        _algo_tag = 'GA (genetic algorithm)'
    print(f"\n{'='*65}")
    print(f"FEATURE SELECTION  [{_algo_tag}]")
    print(f"  Stage: {label}  (estimation only)")
    print(f"  pop={pop_size}  gens={n_generations}  restarts={n_restarts}  "
          f"budget={pop_size * n_generations} steps")
    if not (_is_bandit or _is_sa):
        print(f"  mut_base={mutation_rate}->0.25(adaptive)  tournament_k={tournament_k}")
    print(f"{'='*65}")

    dispatch = {
         0: lambda: _run_ga_cdap(persons, ga_kwargs, n_restarts),
        -1: lambda: _run_ga_usual_loc(persons, landuse, skims, ga_kwargs, n_restarts),
         1: lambda: _run_ga_stage1(tours, persons, ga_kwargs, n_restarts, landuse=landuse),
         2: lambda: _run_ga_stage2(tours, persons, ga_kwargs, n_restarts),
         3: lambda: _run_ga_stage3(tours, persons, ga_kwargs,
                                   trips=trips, n_restarts=n_restarts),
         4: lambda: _run_ga_stage4(trips, tours, persons, ga_kwargs, n_restarts),
         5: lambda: _run_ga_stage5(trips, persons, skims, landuse,
                                   ga_kwargs, n_restarts,
                                   tours_src=tours),
        51: lambda: _run_ga_stage5_trip(trips, persons, skims, landuse,
                                       ga_kwargs, n_restarts),
         6: lambda: _run_ga_stage6(trips, persons, landuse, ga_kwargs, n_restarts),
    }

    if stage not in dispatch:
        raise ValueError(
            f"stage must be 0 (CDAP), -1 (usual_loc), 1ÔÇô6, or 51 (trip-level), got {stage}")

    result = dispatch[stage]()
    result.print_summary()
    return result


def run_ga_all_stages(
    tours:             pd.DataFrame,
    trips:             pd.DataFrame,
    persons:           pd.DataFrame,
    skims                           = None,
    landuse:           Optional[pd.DataFrame] = None,
    stages:            Optional[List[int]]    = None,
    pop_size:          int   = 40,
    n_generations:     int   = 60,
    mutation_rate:     float = 0.08,
    n_restarts:        int   = 2,
    seed:              int   = 42,
    include_cdap:          bool  = True,
    include_usual_loc:     bool  = False,
    include_trip_mode_dest: bool = False,
    save_path:         Optional[str] = 'saved_data/ga_results.pkl',
    stage_overrides:   Optional[Dict[int, Dict[str, Any]]] = None,
) -> AllStagesResult:
    """
    Run GA variable selection (estimation only) across all pipeline stages.

    Includes CDAP (stage 0) by default and optionally usual location (stage -1).
    Prediction over the synthetic population is NOT done here.
    Call apply_ga_results_to_pipeline(results, ...) afterwards.

    Stage codes
    -----------
     0  : CDAP feature selection             (include_cdap=True by default)
    -1  : Usual location decay + attractors  (include_usual_loc=False by default;
                                              requires WORK_TAZ in persons)
     1ÔÇô6: Pipeline stages 1 to 6

    Parameters
    ----------
    tours, trips, persons : HHTS from process_survey()
    skims, landuse        : required for stages 5, 6, and usual_loc (-1)
    stages                : explicit list of stage codes; if None, built automatically
    include_cdap          : add stage 0 (CDAP) to the run list
    include_usual_loc     : add stage -1 (usual location)   needs WORK_TAZ in persons
    include_trip_mode_dest: add stage 51 (trip-level mode + destination) in addition
                            to stage 5 (tour-level). Useful for validating whether
                            trip-level model produces consistent variable selections.
    save_path             : checkpoint file saved after each stage
    stage_overrides       : optional dict mapping stage code ÔåÆ GA kwargs overrides.
                            e.g. {1: {'pop_size': 60, 'n_restarts': 3}}
                            Built-in defaults for stages 1-4 encourage more
                            exploration (larger pop, more gens, more restarts).
    """
    from ga_external_estimators import report_availability, ga_estimator

    # ── Estimator-aware budget defaults ────────────────────────────────────
    # MetaCount / SearchLibrium evaluations are expensive (5-30 sec each),
    # whereas default statsmodels evaluations are cheap (~1 sec).  Heavy
    # budgets that work fine for statsmodels cause multi-day runs with
    # MetaCount, so we scale down when the estimator is metacount/hybrid/sa.
    _est = ga_estimator()
    _is_slow_estimator = _est in ('metacount', 'searchlibrium', 'both', 'hybrid', 'sa_bandit')

    if _is_slow_estimator:
        # ── MetaCount / SA budget (lightweight) ────────────────────────────
        # pop_size × n_generations = total per-purpose budget.
        # n_restarts × restart_gens × pop_size = restart budget.
        # With SA: pop_size=20, gens=40 → 800 evals main + 2×(20×13)=520 = 1320
        default_early_overrides = {
            1: {'pop_size': 20, 'n_generations': 40, 'n_restarts': 2,
                'mutation_rate': 0.12, 'stagnation_limit': 8,
                'tournament_k': 3, 'crossover_rate': 0.70},
            2: {'pop_size': 20, 'n_generations': 40, 'n_restarts': 2,
                'mutation_rate': 0.12, 'stagnation_limit': 8,
                'tournament_k': 3, 'crossover_rate': 0.70},
            3: {'pop_size': 20, 'n_generations': 40, 'n_restarts': 2,
                'mutation_rate': 0.12, 'stagnation_limit': 8,
                'tournament_k': 3, 'crossover_rate': 0.70},
            4: {'pop_size': 20, 'n_generations': 40, 'n_restarts': 2,
                'mutation_rate': 0.12, 'stagnation_limit': 8,
                'tournament_k': 3, 'crossover_rate': 0.70},
        }
    else:
        # Default exploration-boosting overrides for early stages (1-4).
        # These stages are fast (seconds/eval) so larger budgets are cheap.
        default_early_overrides = {
            1: {'pop_size': 80, 'n_generations': 150, 'n_restarts': 4,
                'mutation_rate': 0.12, 'stagnation_limit': 12,
                'tournament_k': 5, 'crossover_rate': 0.80},
            2: {'pop_size': 80, 'n_generations': 150, 'n_restarts': 4,
                'mutation_rate': 0.12, 'stagnation_limit': 12,
                'tournament_k': 5, 'crossover_rate': 0.80},
            3: {'pop_size': 80, 'n_generations': 150, 'n_restarts': 4,
                'mutation_rate': 0.12, 'stagnation_limit': 12,
                'tournament_k': 5, 'crossover_rate': 0.80},
            4: {'pop_size': 80, 'n_generations': 150, 'n_restarts': 4,
                'mutation_rate': 0.12, 'stagnation_limit': 12,
                'tournament_k': 5, 'crossover_rate': 0.80},
        }

    # Merge user-provided overrides on top of defaults
    merged_overrides: Dict[int, Dict[str, Any]] = {}
    for stg, ov in default_early_overrides.items():
        merged_overrides[stg] = dict(ov)
    if stage_overrides:
        for stg, ov in stage_overrides.items():
            if stg in merged_overrides:
                merged_overrides[stg].update(ov)
            else:
                merged_overrides[stg] = dict(ov)

    report_availability()

    if stages is None:
        stages = []
        if include_cdap:
            stages.append(0)
        stages.extend([1, 2, 3, 4])
        if skims   is not None: stages.append(5)
        if skims is not None and include_trip_mode_dest: stages.append(51)
        if landuse is not None: stages.append(6)
        if include_usual_loc and skims is not None and landuse is not None:
            stages.append(-1)

    all_results = AllStagesResult()

    for stage in stages:
        try:
            result = run_ga_stage(
                stage           = stage,
                tours           = tours,
                trips           = trips,
                persons         = persons,
                skims           = skims,
                landuse         = landuse,
                pop_size        = pop_size,
                n_generations   = n_generations,
                mutation_rate   = mutation_rate,
                n_restarts      = n_restarts,
                seed            = seed,
                stage_overrides = merged_overrides,
            )
            all_results.results[stage] = result

            if save_path:
                all_results.save(save_path)
                all_results.export_reports(os.path.dirname(save_path) or '.')
                print(f"  Checkpoint saved ÔåÆ {save_path}")

        except Exception as e:
            logger.error(f"Stage {stage} GA failed: {e}", exc_info=True)
            print(f"  Stage {stage} skipped: {e}")

    all_results.summary()
    return all_results
