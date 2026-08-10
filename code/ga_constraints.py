"""
ga_constraints.py
=================
Single source of truth for search-space constraints across the staged
GA/SA/Bandit pipeline, expressed in SearchLibrium's ConstraintBuilder dict
format so the same structure drives:

  1. the in-house searchers (GeneticAlgorithm / SimulatedAnnealing /
     BanditFeatureSearcher in ga_hypothesis_testing.py) — force_include maps
     to fixed_features, mutually_exclusive to collinear_groups,
     min_behavioral to require_one_groups, force_exclude to a fitness-level
     gene mask (the searchers have no native "banned" support);

  2. the external estimator fits (SearchLibrium MNL/MixedLogit for stage 4,
     MetaCount for stages 1–3) — force_random/never_random decide which
     selected variables get random coefficients and with what distribution;

  3. the stage-5 Larch models — never_random/force_random filter/extend the
     MXL mixture list in _estimate_mixed_logit.

Constraint dict keys (SearchLibrium ConstraintBuilder.dict() format):
    force_include : list[str]           always in the model
    force_exclude : list[str]           never in the model
    mutually_exclusive : list[list[str]] at most one of each group
    min_behavioral : (n, [vars])        at least n of these
    force_random  : dict[str, str]      var -> distribution ('n', 'ln', 'tn')
    never_random  : list[str]           never given a random coefficient

Distributions follow SearchLibrium: 'n' Normal, 'ln' LogNormal, 'tn'
truncated Normal.

Env overrides:
    GA_RAND_PARAM_SEARCH=1   enable greedy random-vs-fixed toggling after
                             stage-4 variable selection (see
                             ga_external_estimators.stage4_final_mixed_logit)
    GA_STAGE5_RAND=a,b,c     extra stage-5 mode-MXL random params (Larch
                             param names, e.g. c_ivt)
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Use SearchLibrium's builder when the installed version has it (>= 0.0.138)
# so constraint dicts stay format-compatible with its own search/harmony
# modules; degrade silently to plain dicts otherwise.
try:
    from SearchLibrium.constraints_builder import create_constraints  # noqa: F401
    SEARCHLIB_CONSTRAINTS_OK = True
except Exception:                                     # pragma: no cover
    create_constraints = None
    SEARCHLIB_CONSTRAINTS_OK = False


# ---------------------------------------------------------------------------
# Per-stage constraint declarations
# ---------------------------------------------------------------------------
# Names must match the stage's candidate feature names (the feat_names list
# each searcher is built with). Missing names are ignored at apply time, so
# these are safe across candidate-pool variations.

# ---------------------------------------------------------------------------
# Shared "one encoding per attribute" mutex groups (stages 1-4)
# ---------------------------------------------------------------------------
# RULE (per user): binary CATEGORY dummies are a partition — they do NOT
# overlap, so they must stay FREE to co-occur (a normal dummy-variable set, one
# as reference). They are NEVER made mutually exclusive with each other. The
# only exclusivity is:
#   • a continuous encoding of the SAME attribute is mutex with each band
#     (prevents double-encoding continuous + categorical), expressed PAIRWISE so
#     the bands themselves remain free; and
#   • the continuous encodings of one attribute are mutex with each other.
# Genuinely REDUNDANT alternative encodings of one concept (not a partition,
# e.g. the several WFH encodings, or sexp vs is_female) do stay fully mutex.
#
# Family = (continuous/redundant forms, category-band dummies).
_ENCODING_FAMILIES = {
    'income':    (['incp', 'incp_generic'],
                  ['income_verylo', 'income_lo', 'income_mid', 'income_midplus',
                   'income_midhi', 'income_hi', 'income_veryhi']),
    'vehicle':   (['vehrd', 'veh_generic', 'hh_vehicles_pred'],
                  ['no_vehicle', 'one_vehicle', 'has_vehicle', 'multi_vehicle']),
    'household': (['nprd', 'HHSIZE'],
                  ['hh_alone', 'hh_couple', 'hh_medium', 'hh_size_large']),
    'age':       (['age5p'],
                  ['age_child', 'age_teen', 'age_working', 'age_older',
                   'age_senior', 'age_retired', 'age_young']),
}
# Fully-redundant alternative encodings of ONE concept (not a partition) — these
# genuinely overlap, so at most one may be active.
_ALTERNATIVE_ENCODINGS = [
    ['wfh', 'wfh_flag', 'WFHWEEK', 'WFHDAY'],
    ['sexp', 'is_female'],
    # Modeled household bike count (mobility_models) vs the observed 'bikes'
    # count — the prediction is a smooth function of the same attributes, so the
    # two are collinear; at most one in a spec.
    ['bikes', 'hh_bikes_pred'],
    # Tour-sequence length (tour_sequence) vs the observed tour count — both are
    # counts of tours per person, mechanically near-identical.
    ['n_tours', 'n_tours_seq'],
]


def _build_encoding_mutex() -> List[list]:
    groups: List[list] = []
    for _cont, _bands in _ENCODING_FAMILIES.values():
        if len(_cont) >= 2:
            groups.append(list(_cont))                 # continuous forms vs each other
        for _c in _cont:
            for _b in _bands:
                groups.append([_c, _b])                # continuous vs each band (bands free)
    for _grp in _ALTERNATIVE_ENCODINGS:
        groups.append(list(_grp))
    return groups


_ENCODING_MUTEX = _build_encoding_mutex()

# Continuous behavioural drivers that make good RANDOM (mixed) coefficients when
# selected — taste variation on these is what a mixed model captures. Binary
# dummies are deliberately excluded (a random coef on a 0/1 is weakly
# identified). Applied as force_random for stages 1-4.
_CONTINUOUS_RANDOM = {
    'incp': 'n', 'vehrd': 'n', 'nprd': 'n', 'age5p': 'n',
    'walk_acc': 'n', 'transit_acc': 'n', 'auto_work_acc': 'n',
    'auto_shop_acc': 'n', 'auto_nonmand_acc': 'n', 'bike_acc': 'n',
    'n_total': 'n', 'n_tours': 'n',
}

# Structural / weakly-identified terms that must NEVER be random (stages 1-4):
# model structure plus the 0/1 person-type & band dummies.
_NEVER_RANDOM_14 = [
    'const', 'offset', 'threshold_0', 'threshold_1', 'threshold_2',
    'is_female', 'CARLICENCE', 'mandatory_role',
    'has_vehicle', 'no_vehicle', 'one_vehicle', 'multi_vehicle',
    'income_verylo', 'income_lo', 'income_mid', 'income_midplus',
    'income_midhi', 'income_hi', 'income_veryhi',
    'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
    'hh_alone', 'hh_couple', 'hh_medium', 'hh_size_large',
]

PIPELINE_CONSTRAINTS: Dict[str, dict] = {
    # Stage 1 — activity generation (MetaCount count models)
    'stage1': {
        # Random-coefficient search default: taste variation on continuous
        # behavioural drivers (income, vehicles, accessibility) is where a
        # mixed count model earns its keep; the structural const/offset and
        # 0/1 binary dummies are kept fixed (a random coef on a binary is
        # weakly identified). never_random wins over force_random.
        'mutually_exclusive': [list(g) for g in _ENCODING_MUTEX],
        'force_random': dict(_CONTINUOUS_RANDOM),
        'never_random': list(_NEVER_RANDOM_14),
    },
    # Stage 2 — day-pattern / duration targets
    'stage2': {
        'mutually_exclusive': [list(g) for g in _ENCODING_MUTEX],
        'force_random': dict(_CONTINUOUS_RANDOM),
        'never_random': list(_NEVER_RANDOM_14),
    },
    # Stage 3 — activity duration
    'stage3': {
        'mutually_exclusive': [list(g) for g in _ENCODING_MUTEX],
        'force_random': dict(_CONTINUOUS_RANDOM),
        'never_random': list(_NEVER_RANDOM_14),
    },
    # Stage 4 — schedule placement (SearchLibrium MNL/MXL over time periods)
    'stage4': {
        # Peak-period pressure variables are the behavioural core of the
        # timing model; taste variation on them is exactly what the MXL is
        # for. Keep interaction dummies fixed — a random coefficient on a
        # 0/1 segment dummy is weakly identified at 50 draws.
        'mutually_exclusive': [list(g) for g in _ENCODING_MUTEX],
        'force_random': dict(_CONTINUOUS_RANDOM),
        'never_random': list(_NEVER_RANDOM_14) + ['is_weekend', 'is_school_term'],
    },
    # Stage 5 — tour mode choice (Larch NL/MXL)
    'stage5_mode': {
        'force_include': ['CRASH'],
        # Encoding validity: never two encodings of the same person attribute
        # in one spec (continuous age x bike vs the young-adult bin x bike;
        # continuous vehicles x DA vs the zero-vehicle dummy on the same
        # nest). Names absent from a run's candidate pool are ignored.
        'mutually_exclusive': [
            ['c_age_bike', 'c_young_bike'],
            ['c_veh_da', 'c_noveh_da'],
            ['c_incp_da', 'c_incp_band_da'],
            # Purpose × mode dummies: at most ONE encoding of purpose behaviour
            # per mode (e.g. work_sr, edu_sr are separate; but work_walk vs
            # age_walk are alternative ways to capture the same pattern).
            ['c_work_sr', 'c_ptwork_sr'],
            ['c_work_walk', 'c_age_walk'],
            ['c_work_bike', 'c_age_bike'],
            ['c_work_transit', 'c_notwork_transit'],
            # Worker-status encodings (GENE names actually in the mode pool): the
            # WORKSTAT-derived binary and the equivalent ptype_* dummy encode the
            # SAME construct and both gate the SAME c_*work_* coefficient, so
            # co-selecting them just wastes a gene and muddies identification of
            # the (already fragile) worker×mode captivity term. At most one each.
            # (workstat_notworking vs ptype_Retired are NOT paired — retired is a
            # subset of not-working, a legitimately distinct captivity signal.)
            ['workstat_fulltime', 'ptype_FullTimeWorker'],
            ['workstat_parttime', 'ptype_PartTimeWorker'],
            # CBD proximity: at most one of {c_cbd_transit, c_cbd_da, c_cbd_sr}
            # is independently identifiable (all reference emp.is_cbd).
            ['c_cbd_transit', 'c_cbd_da', 'c_cbd_sr'],
            # Micromobility: continuous bikes vs binary ebikes/escooter ownership
            ['c_bikes_bike', 'c_ebike_bike', 'c_escooter_bike'],
            # ── Generic-vs-alternative-specific encoding (see generic_groups) ──
            # The GA picks EITHER one mode-generic coefficient (same effect on
            # every non-DA alternative) OR the per-mode alternative-specific
            # block — never both. This lets the search literally swap between
            # "one income slope for all modes" and "a slope per mode".
            ['incp_generic', 'incp', 'income_verylo', 'income_lo', 'income_mid',
             'income_midplus', 'income_midhi', 'income_hi', 'income_veryhi'],
            ['veh_generic', 'vehrd', 'no_vehicle', 'one_vehicle',
             'has_vehicle', 'multi_vehicle'],
            ['crash_generic', 'CRASH'],
            ['age_active_generic', 'age_child', 'age_teen', 'age_working',
             'age_older', 'age_senior'],
            # In-vehicle-time period: at most ONE auto IVT period in a model
            # (AM is the backbone; PM/MD/EV are alternative representations).
            ['SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM',
             'SOVTOLL_TIME__MD', 'SOVTOLL_TIME__EV'],
            # Auto money cost: monetised toll OR the composite cost proxy, not both.
            ['SOVTOLL_TOLL__AM', 'cost_proxy'],
        ],
        # NOTE: no min_behavioral for mode — the level-of-service backbone
        # (c_ivt_auto, c_walk_time, c_bike_time) is structurally always-on via
        # _ALWAYS_INCLUDE in _mode_spec_custom, so "at least one impedance term"
        # is guaranteed without a require-one gene. (A require-one over GA genes
        # would wrongly collapse to forcing whichever single impedance gene is
        # in the candidate pool, e.g. SOVTOLL_TIME__PM.)
        # ASCs and nest params must stay fixed; level-of-service terms are
        # the sanctioned random set (mirrors _random_params in
        # _estimate_mixed_logit).
        'force_random': {'c_ivt': 'n', 'c_walk': 'n',
                         'c_bike': 'n', 'c_da_aff': 'n'},
        'never_random': ['ASC_SR', 'ASC_Walk', 'ASC_Bike', 'ASC_Transit',
                         'Mu_Motor', 'Mu_Root'],
        # ── Generic-vs-alternative-specific collapse (refit substitution) ────
        # Each entry: generic_gene -> the mode-generic replacement + the ASV
        # coefficient block it substitutes for. Read by BOTH the GA candidate
        # gating in stage5_destination_choice._mode_spec_custom AND the
        # post-estimation _collapse_generic_groups refit, so the search space
        # and the refit never disagree about what "generic" means.
        'generic_groups': {
            'incp_generic': {
                'coeff': 'c_incp_gen',
                'expr':  'person.incp',
                'modes': ['SR', 'Transit', 'Walk', 'Bike'],
                'asv':   ['c_incp_da', 'c_incp_sr', 'c_incp_transit', 'c_incp_active'],
            },
            'veh_generic': {
                'coeff': 'c_veh_gen',
                'expr':  'person.vehrd',
                'modes': ['DA', 'SR'],
                'asv':   ['c_veh_da', 'c_veh_sr'],
            },
            'crash_generic': {
                'coeff': 'c_crash_gen',
                'expr':  'log(od.CRASH+1)',
                'modes': ['DA', 'SR', 'Transit'],
                'asv':   ['c_crash_veh'],
            },
            'age_active_generic': {
                'coeff': 'c_age_active_gen',
                'expr':  'person.age5p',
                'modes': ['Walk', 'Bike'],
                'asv':   ['c_age_walk', 'c_age_bike'],
            },
        },
        # ── Sign constraints (theoretical priors, enforced at build + refit) ─
        # 'neg' → coefficient bounded ≤ 0; 'pos' → bounded ≥ 0. Replaces the
        # scattered ad-hoc _pf_set(..., 'maximum'/'minimum') calls in
        # _build_mode_model with one declarative list.
        'sign_constraints': {
            'c_ivt_auto': 'neg', 'c_ivt_transit': 'neg',
            'c_ivt_auto_pm': 'neg', 'c_ivt_auto_md': 'neg', 'c_ivt_auto_ev': 'neg',
            'c_ivt_hov2_am': 'neg',
            'c_walk_time': 'neg', 'c_bike_time': 'neg',
            'c_cost_auto': 'neg', 'c_toll_auto': 'neg',
            'c_xwait_transit': 'neg', 'c_drv_transit': 'neg',
            'c_crash_safety': 'neg',
            'ASC_Walk': 'neg',
            # Habitual mode-usage captivity: more habitual use of a mode raises
            # its utility (≥ 0). Also blocks a perverse sign from any residual
            # collinearity with ownership terms.
            'c_transit_use_transit': 'pos', 'c_car_use_da': 'pos',
            'c_bike_use_bike': 'pos',
        },
        # ── Protected backbone (never removed by _prune_and_refit) ───────────
        'protected_backbone': [
            'ASC_SR', 'ASC_Walk', 'ASC_Bike', 'ASC_Transit',
            'Mu_Motor', 'Mu_NonMotor', 'Mu_Root',
            'c_ivt_auto', 'c_ivt_transit', 'c_walk_time', 'c_bike_time',
            # Zone-varying accessibility: breaks logsum-distance collinearity
            'c_transit_access',
        ],
    },
    # Stage 5 — destination choice (Larch MNL, flat CA)
    'stage5_dest': {
        'force_include': ['c_crash'],
        # Encoding validity: one encoding per person attribute (continuous
        # income x distance vs banded income x distance; continuous age vs
        # age-band interactions). Absent names are ignored.
        'mutually_exclusive': [
            ['c_incp_dist', 'c_incp_band_dist'],
            ['c_age_dist', 'c_age_band_dist'],
            # Agglomeration: the blended composite (c_aggl), its collinear twin
            # (c_comp, |r|≈0.9998) AND all 8 per-type themed variants
            # (retail/office/edu/leisure/parks/health/food/civic) share the SAME
            # spatial decay kernel, so they are mutually collinear. Letting them
            # co-occur made the dest Hessian singular — every agglomeration term
            # ended AT A BOUND with a degenerate proxy SE (fake p-values), and the
            # GA penalty then couldn't tell them apart. Keep the WHOLE family a
            # single at-most-one mutex; combined with the require-one below this
            # forces EXACTLY ONE agglomeration term, so the search still identifies
            # WHICH land-use component drives each segment ("mapping to
            # components") but with a real, significant, identified coefficient.
            ['c_aggl', 'c_comp', 'c_aggl_retail', 'c_aggl_office', 'c_aggl_edu',
             'c_aggl_leisure', 'c_aggl_parks', 'c_aggl_health', 'c_aggl_food',
             'c_aggl_civic'],
            # Home-pull: at most one of the two time-scaling variants.
            ['c_home_pull', 'c_home_pull_late'],
            # Accessibility: walk/transit accessibility is either conditioned
            # on vehicle ownership (noveh) or unconditional, not both.
            ['c_walk_acc', 'c_noveh_walk_acc'],
            ['c_transit_acc', 'c_noveh_transit_acc'],
            # (competition c_comp* is folded into the agglomeration mutex above,
            #  since acdc_aggl and acdc_comp are collinear.)
            # Distance form: EITHER the smooth linear penalty c_dist OR the
            # piecewise distance-bin spline, never both (they span the same
            # distance axis and are collinear together).
            ['c_dist', 'c_dist_0_2', 'c_dist_2_5', 'c_dist_5_10'],
            # Accessibility form: the mode-choice logsum and the raw log-distance
            # term both encode OD separation and are collinear, so at most ONE is
            # active. Combined with the require-one group below this means EXACTLY
            # one of {c_logsum, c_log_dist} — which stops the logsum coefficient
            # being holdfast-at-0 / pruned for collinearity with c_log_dist.
            ['c_logsum', 'c_log_dist'],
        ],
        # Semantic pairing: when occupation-type person terms steer work
        # destinations, at least one matching industry attractor must be
        # present so job types can actually point at same-industry employment
        # on the map (occupation x distance alone has nothing to aim at).
        'min_behavioral': [
            (1, ['c_indp_finance', 'c_indp_health', 'c_indp_retail']),
            # At least one accessibility/distance term must survive; with the
            # c_logsum/c_log_dist mutex above this guarantees EXACTLY one is
            # present, so the destination model never loses distance signal.
            (1, ['c_logsum', 'c_log_dist']),
            # At least one agglomeration term must survive so spatial-structure
            # signal is never dropped entirely. With c_aggl no longer forced,
            # this require-one lets the search satisfy it with the blended term
            # OR (preferably) one/more of the per-type themed components.
            (1, ['c_aggl', 'c_aggl_retail', 'c_aggl_office', 'c_aggl_edu',
                 'c_aggl_leisure', 'c_aggl_parks', 'c_aggl_health',
                 'c_aggl_food', 'c_aggl_civic']),
        ],
        # size_term and log_correction are structural (fixed priors /
        # sampling correction) — never random, never searched out.
        'never_random': ['c_size_term', 'log_correction'],
        # Distance sensitivity heterogeneity is handled post-estimation via
        # _estimate_dest_dist_heterogeneity (s_log_dist), not in-model.
        # ── Sign constraints ─────────────────────────────────────────────────
        # Distance/crash deter (≤0); size/accessibility/agglomeration attract (≥0).
        'sign_constraints': {
            'c_log_dist': 'neg', 'c_dist': 'neg', 'c_crash': 'neg',
            'c_crash_safety_dest': 'neg',
            'c_dist_0_2': 'neg', 'c_dist_2_5': 'neg', 'c_dist_5_10': 'neg',
            'c_size_term': 'pos',
            'c_dist_decay': 'pos', 'c_dist_decay_slow': 'pos',
            'c_aggl': 'pos', 'c_aggl_retail': 'pos', 'c_aggl_office': 'pos',
            'c_aggl_edu': 'pos', 'c_aggl_leisure': 'pos',
            'c_aggl_parks': 'pos', 'c_aggl_health': 'pos',
            'c_aggl_food': 'pos', 'c_aggl_civic': 'pos',
            'c_walk_acc': 'pos', 'c_transit_acc': 'pos', 'c_auto_work_acc': 'pos',
            # Mode-choice logsum: better accessibility -> more attractive dest.
            'c_logsum': 'pos',
        },
        # ── Protected backbone (never removed by _prune_and_refit) ───────────
        'protected_backbone': [
            # c_log_dist is intentionally NOT forced here anymore: it now shares an
            # exactly-one accessibility group with c_logsum (mutex + require-one).
            # Forcing it would permanently exclude the logsum; the require-one
            # group guarantees one of the two is always present regardless.
            'c_size_term', 'log_correction', 'c_crash',
        ],
    },
}


# ---------------------------------------------------------------------------
# User constraint overlay (authored by the Streamlit GUI, constraint_gui.py)
# ---------------------------------------------------------------------------
# A JSON file the user edits/generates to ADD constraints on top of the
# hard-coded PIPELINE_CONSTRAINTS above — without touching Python. Read once and
# merged into every get_constraints()/accessor call, so a run picks up whatever
# the GUI last generated.
#
# Path resolution (first hit wins):
#   1. $GA_CONSTRAINTS_FILE            (explicit override)
#   2. <this dir>/ga_constraints_user.json
#   3. <this dir>/runs/constraints/ga_constraints_user.json
#
# Overlay JSON shape (all keys optional):
#   {
#     "stage5_dest": {
#        "force_include": ["c_crash"],           # must-have coefficients/features
#        "force_exclude": ["c_comp"],
#        "mutually_exclusive": [["c_aggl","c_aggl_retail"]],
#        "min_behavioral": [[1, ["c_indp_finance","c_indp_health"]]],
#        "sign_constraints": {"c_dist": "neg"},
#        "generic_groups": { ... }               # advanced (mode only)
#     },
#     "ptype_land": [                              # ptype ↔ land-attribute affinity
#        {"ptype": "ptype_UniversityStudent", "land": "Education",
#         "segments": ["edu"], "pairing": true, "interaction": true}
#     ]
#   }
# List-valued keys are APPENDED to the base (deduped); dict-valued keys are
# merged (overlay wins per key). ptype_land is expanded into pairing rules here
# and into interaction genes by the candidate/tree code (see ptype_land_pairs).

import json as _json
import pathlib as _pathlib

_OVERLAY_CACHE: Optional[dict] = None
_OVERLAY_MTIME: Optional[float] = None


def constraint_overlay_path() -> Optional[str]:
    """Resolve the user overlay JSON path, or None if none exists."""
    env = os.environ.get('GA_CONSTRAINTS_FILE', '').strip()
    if env:
        return env if os.path.exists(env) else None
    here = _pathlib.Path(__file__).resolve().parent
    for cand in (here / 'ga_constraints_user.json',
                 here / 'runs' / 'constraints' / 'ga_constraints_user.json'):
        if cand.exists():
            return str(cand)
    return None


def load_constraint_overlay(force: bool = False) -> dict:
    """
    Load (and cache) the user constraint overlay JSON. Re-reads automatically
    when the file's mtime changes so a freshly-generated overlay is picked up
    without restarting a long process. Returns {} when no overlay file exists
    or it fails to parse (a malformed overlay must never break a run).
    """
    global _OVERLAY_CACHE, _OVERLAY_MTIME
    path = constraint_overlay_path()
    if not path:
        _OVERLAY_CACHE, _OVERLAY_MTIME = {}, None
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _OVERLAY_CACHE or {}
    if (not force) and _OVERLAY_CACHE is not None and _OVERLAY_MTIME == mtime:
        return _OVERLAY_CACHE
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = _json.load(fh)
        if not isinstance(data, dict):
            raise ValueError('overlay root must be a JSON object')
        _OVERLAY_CACHE, _OVERLAY_MTIME = data, mtime
        logger.info(f"[constraints] loaded user overlay: {path} "
                    f"(stages={[k for k in data if k != 'ptype_land']}, "
                    f"ptype_land={len(data.get('ptype_land', []))})")
        return data
    except Exception as exc:
        logger.warning(f"[constraints] failed to read overlay {path}: {exc!r} — ignoring")
        _OVERLAY_CACHE, _OVERLAY_MTIME = {}, mtime
        return {}


def normalize_min_behavioral(mb) -> List[tuple]:
    """
    Normalise a min_behavioral spec to a list of (n, [vars]) groups.

    Accepts a single (n, [vars]) tuple/list (the legacy PIPELINE_CONSTRAINTS
    form) or a list of such groups (what the overlay/GUI appends). Empty list
    for anything unrecognised.
    """
    if not mb:
        return []
    # single group: (n, [vars])
    if (isinstance(mb, (list, tuple)) and len(mb) == 2
            and isinstance(mb[0], (int, float))
            and isinstance(mb[1], (list, tuple))):
        return [(int(mb[0]), list(mb[1]))]
    # list of groups
    out = []
    if isinstance(mb, (list, tuple)):
        for g in mb:
            if (isinstance(g, (list, tuple)) and len(g) == 2
                    and isinstance(g[1], (list, tuple))):
                out.append((int(g[0]), list(g[1])))
    return out


def _merge_constraint_dicts(base: dict, over: dict) -> dict:
    """
    Merge one overlay stage-dict onto a base stage-dict.

    * min_behavioral → normalise both sides to a list of (n, [vars]) groups and
                       concatenate (so the overlay ADDS requirements)
    * list values    → append overlay items not already present (order-preserving)
    * dict values    → shallow-update (overlay entries win)
    * scalars        → overlay replaces base
    """
    out = dict(base)
    for key, oval in over.items():
        if key == 'min_behavioral':
            groups = normalize_min_behavioral(out.get('min_behavioral'))
            for g in normalize_min_behavioral(oval):
                if g not in groups:
                    groups.append(g)
            out['min_behavioral'] = groups
            continue
        bval = out.get(key)
        if isinstance(bval, list) and isinstance(oval, list):
            merged = list(bval)
            for item in oval:
                if item not in merged:
                    merged.append(item)
            out[key] = merged
        elif isinstance(bval, dict) and isinstance(oval, dict):
            m = dict(bval)
            m.update(oval)
            out[key] = m
        else:
            out[key] = oval
    return out


def ptype_land_pairs() -> List[dict]:
    """
    ptype ↔ land-attribute affinities from the overlay. Each item:
        {'ptype': str, 'land': str, 'segments': [str],
         'pairing': bool, 'interaction': bool}
    Consumed by the candidate/tree code to (a) require the land attractor when
    the ptype driver is active (pairing) and (b) materialise a ptype×land
    interaction gene (interaction). Empty when no overlay / no pairs.
    """
    ov = load_constraint_overlay()
    pairs = []
    for p in ov.get('ptype_land', []) or []:
        if not isinstance(p, dict) or 'ptype' not in p or 'land' not in p:
            continue
        pairs.append({
            'ptype':       str(p['ptype']),
            'land':        str(p['land']),
            'segments':    list(p.get('segments', []) or []),
            'pairing':     bool(p.get('pairing', True)),
            'interaction': bool(p.get('interaction', True)),
        })
    return pairs


def get_constraints(stage: str) -> dict:
    """
    Return the constraint dict for a pipeline stage, with the user overlay
    merged on top of the hard-coded PIPELINE_CONSTRAINTS ('' keys tolerated).

    ptype↔land *pairing* rules are authored by the GUI as ordinary
    min_behavioral overlay entries on the target stage, so they flow through the
    generic merge here; the ptype_land section itself only drives interaction
    genes (see ptype_land_pairs / the candidate + _dest_tree code).
    """
    base = dict(PIPELINE_CONSTRAINTS.get(stage, {}))
    ov = load_constraint_overlay()
    if stage and stage in ov and isinstance(ov[stage], dict):
        base = _merge_constraint_dicts(base, ov[stage])
    return base


def generic_groups(stage: Optional[str]) -> Dict[str, dict]:
    """
    Generic-vs-alternative-specific collapse groups for a stage.

    Maps a 'generic' gene name to the mode-generic replacement it stands for::

        {generic_gene: {'coeff': str, 'expr': str, 'modes': [...], 'asv': [...]}}

    Consumed by stage5_destination_choice._mode_spec_custom (to swap the ASV
    block for one shared coefficient when the generic gene is selected) and by
    _collapse_generic_groups (to try the same swap post-estimation and keep the
    better-BIC form). Empty dict when the stage declares none.
    """
    c = get_constraints(stage) if stage else {}
    return {k: dict(v) for k, v in c.get('generic_groups', {}).items()}


def sign_constraints(stage: Optional[str]) -> Dict[str, str]:
    """
    Theoretical sign priors for a stage: {param_name: 'neg' | 'pos'}.

    'neg' → the coefficient is bounded ≤ 0 (deterrent: cost, time, crash);
    'pos' → bounded ≥ 0 (attractor: size, accessibility, agglomeration).
    Applied at model-build time (bounds) and re-enforced after refit.
    """
    c = get_constraints(stage) if stage else {}
    return dict(c.get('sign_constraints', {}))


def protected_backbone(stage: Optional[str]) -> List[str]:
    """
    Parameters _prune_and_refit must never drop, even if they fit to ~0 or are
    statistically insignificant. Includes the stage's declared backbone plus
    its force_include list (structural identification terms).
    """
    c = get_constraints(stage) if stage else {}
    names = list(c.get('protected_backbone', []))
    for v in c.get('force_include', []):
        if v not in names:
            names.append(v)
    return names


# ---------------------------------------------------------------------------
# Mapping onto the in-house searchers
# ---------------------------------------------------------------------------

def searcher_constraint_kwargs(
    stage: Optional[str],
    feat_names: Sequence[str],
    fixed_features: Optional[List[str]] = None,
    collinear_groups: Optional[List[List[str]]] = None,
    require_one_groups: Optional[List[List[str]]] = None,
) -> dict:
    """
    Merge the stage's declared constraints into the kwargs the in-house
    searchers already understand. Returns a dict with keys
    fixed_features / collinear_groups / require_one_groups / banned_features
    (the last for the fitness-level mask in _make_searcher).
    """
    fixed = list(fixed_features or [])
    coll = [list(g) for g in (collinear_groups or [])]
    req1 = [list(g) for g in (require_one_groups or [])]
    banned: List[str] = []

    if stage:
        c = get_constraints(stage)
        feat_set = set(feat_names)
        for v in c.get('force_include', []):
            if v in feat_set and v not in fixed:
                fixed.append(v)
        for v in c.get('force_exclude', []):
            if v in feat_set:
                banned.append(v)
        for grp in c.get('mutually_exclusive', []):
            g = [v for v in grp if v in feat_set]
            if len(g) >= 2 and g not in coll:
                coll.append(g)
        for _n, grp in normalize_min_behavioral(c.get('min_behavioral')):
            g = [v for v in grp if v in feat_set]
            if g and g not in req1:
                req1.append(g)
        if fixed or banned or coll or req1:
            logger.info(
                f"[constraints:{stage}] fixed={fixed} banned={banned} "
                f"mutex_groups={len(coll)} require_one_groups={len(req1)}"
            )
    return {
        'fixed_features': fixed,
        'collinear_groups': coll,
        'require_one_groups': req1,
        'banned_features': banned,
    }


# ---------------------------------------------------------------------------
# Random-coefficient policy
# ---------------------------------------------------------------------------

def random_var_policy(
    stage: Optional[str],
    selected_vars: Sequence[str],
    default_dist: str = 'n',
    default_random: bool = True,
) -> Dict[str, str]:
    """
    Decide which of the selected variables get random coefficients.

    force_random entries always win (with their declared distribution);
    never_random entries are always excluded; the rest follow
    ``default_random`` (True = current pipeline behaviour of randomising
    every selected variable).
    """
    c = get_constraints(stage) if stage else {}
    force = dict(c.get('force_random', {}))
    never = set(c.get('never_random', []))

    rv: Dict[str, str] = {}
    for v in selected_vars:
        if v in never:
            continue
        if v in force:
            rv[v] = force[v] or default_dist
        elif default_random:
            rv[v] = default_dist
    # force_random vars that were not selected are NOT injected — selection
    # decides presence, this policy only decides randomness of what's present.
    return rv


def rand_search_enabled(stage: Optional[str] = None) -> bool:
    """
    Whether random-coefficient search (greedy fixed-vs-random toggling after
    feature selection) runs for a stage.

    Default is ON for every stage — making random parameters a first-class part
    of the search, driven by each stage's force_random / never_random policy.
    Disable globally with GA_RAND_SEARCH=0 (or the legacy GA_RAND_PARAM_SEARCH=0);
    disable one stage with GA_RAND_SEARCH_<STAGE>=0 (e.g. GA_RAND_SEARCH_STAGE1=0).
    """
    _global = os.environ.get('GA_RAND_SEARCH',
                             os.environ.get('GA_RAND_PARAM_SEARCH', '1')).strip()
    if _global == '0':
        return False
    if stage:
        _key = f'GA_RAND_SEARCH_{str(stage).upper()}'
        if os.environ.get(_key, '1').strip() == '0':
            return False
    return True


def rand_param_search_enabled() -> bool:
    """Backward-compatible alias: greedy random-vs-fixed toggling (stage 4).
    Now defaults ON (see rand_search_enabled)."""
    return rand_search_enabled('stage4')


def random_var_constraints(stage: Optional[str]):
    """(never_random, force_random_names) for the bandit-SA rand-param genes
    (GA_MXL_GENE_SEARCH=1 chromosome extension in _make_searcher). The gene
    block is binary fixed-vs-random; distributions still come from
    random_var_policy at estimation time."""
    c = get_constraints(stage) if stage else {}
    return list(c.get('never_random', [])), list(c.get('force_random', {}).keys())


def stage5_extra_random_params() -> List[str]:
    """Extra Larch param names to randomise in the stage-5 mode MXL."""
    raw = os.environ.get('GA_STAGE5_RAND', '').strip()
    return [p.strip() for p in raw.split(',') if p.strip()] if raw else []
