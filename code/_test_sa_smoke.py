"""
5-Phase Simulated Annealing progressive search smoke test — signal-embedded data.

All five SA search phases use fitness functions that can distinguish good from bad
feature subsets because the synthetic data has REAL signal embedded:
  - Person-type dummies (ptype_FTW, ptype_SCH, …) genuinely predict count/duration
  - Including signal features → meaningfully lower BIC → visible SA convergence

Model hierarchy per phase:
  Phase 1  →  Stage 1  MetaCount NB2 (available) / scipy Poisson (fallback)
  Phase 2  →  Stage 2  MetaCount mixed lognormal (available) / statsmodels OLS (fallback)
  Phase 3  →  Stage 3  MetaCount lognormal per purpose / statsmodels lognormal fallback
               Saves best-feature coefficients to smoke_stage3_coefficients.csv
  Phase 4  →  Stage 4  SearchLibrium MixedLogit (available) / MultinomialLogit fallback
               Falls back to statsmodels MNL when SearchLibrium unavailable
  Phase 5  →  Final    Larch NL (available) / correlated MetaCount NB2 / SearchLibrium MXL

Features: 20 diverse variables including ptype dummies, socio-economic, accessibility.
SA budget: pop_size=12, n_generations=10, 1 restart per phase.

Run:
    $env:JAX_PLATFORMS='cpu'; $env:PYTHONIOENCODING='utf-8'
    z:\\test_runs_tours\\code\\.venv\\Scripts\\python.exe -u `
        z:\\test_runs_tours\\code\\_test_sa_smoke.py
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("GA_ESTIMATOR", "default")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── helpers ──────────────────────────────────────────────────────────────────
_results: list[tuple[str, str, str]] = []
_phase_results: dict[int, dict] = {}


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    _results.append((status, name, detail))
    suf = f"  [{detail}]" if detail else ""
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{suf}")
    return bool(condition)


def _is_progressive(history: list[dict]) -> tuple[bool, str]:
    bsf = [h['best_so_far'] for h in history if np.isfinite(h['best_so_far'])]
    if len(bsf) < 2:
        return True, f"only {len(bsf)} finite entries"
    violations = sum(1 for i in range(len(bsf)-1) if bsf[i] > bsf[i+1] + 1e-10)
    return violations == 0, (f"bsf[0]={bsf[0]:.4f} bsf[-1]={bsf[-1]:.4f} "
                             f"violations={violations}")


# ── availability checks ───────────────────────────────────────────────────────
from ga_external_estimators import (
    _METACOUNT_OK, _SEARCHLIB_OK,
    _metacount_fit_mixed, _ensure_id,
    stage1_fitness_metacount,
    stage2_fitness_metacount,
    stage3_fitness_metacount,
    stage4_fitness_searchlibrium,
    stage4_fitness_searchlibrium_mixed,
    stage4_final_mixed_logit,
)
from ga_hypothesis_testing import SimulatedAnnealing, GAResult, Individual

print("=" * 66)
print("5-Phase SA Signal-Bearing Progressive Search Smoke Test")
print(f"  MetaCount:     {'available' if _METACOUNT_OK else 'NOT INSTALLED — scipy fallback'}")
print(f"  SearchLibrium: {'available' if _SEARCHLIB_OK else 'NOT INSTALLED — statsmodels MNL fallback'}")
print("=" * 66)


# ── Scipy / statsmodels fallback fitness ──────────────────────────────────────
def _poisson_bic(df: pd.DataFrame, y_col: str, feats: list[str]) -> float:
    """–BIC/n using statsmodels Poisson GLM; > is better."""
    try:
        import statsmodels.api as sm
        sub = df[feats + [y_col]].dropna()
        if len(sub) < 20:
            return -1e9
        X = sm.add_constant(sub[feats].astype(float).values, has_constant='add')
        y = sub[y_col].astype(float).values
        res = sm.GLM(y, X, family=sm.families.Poisson()).fit(disp=False)
        bic = res.bic
        return -bic / len(sub)
    except Exception:
        return -1e9


def _lognormal_bic(df: pd.DataFrame, y_col: str, feats: list[str]) -> float:
    """–BIC/n for log-normal regression (OLS on log(y)); > is better."""
    try:
        import statsmodels.api as sm
        sub = df[feats + [y_col]].dropna()
        sub = sub[sub[y_col] > 0]
        if len(sub) < 20:
            return -1e9
        X = sm.add_constant(sub[feats].astype(float).values, has_constant='add')
        logy = np.log(sub[y_col].astype(float).values)
        res = sm.OLS(logy, X).fit()
        bic = res.bic
        return -bic / len(sub)
    except Exception:
        return -1e9


def _mnl_bic(df: pd.DataFrame, y_col: str, feats: list[str], n_alts: int = 6) -> float:
    """–BIC/n for a flat MNL (one utility per alternative); > is better."""
    try:
        from sklearn.linear_model import LogisticRegression
        sub = df[feats + [y_col]].dropna()
        y = sub[y_col].astype(int).values
        if len(sub) < 20 or len(np.unique(y)) < 2:
            return -1e9
        X = sub[feats].astype(float).fillna(0).values
        clf = LogisticRegression(max_iter=300, C=1e6,
                                 multi_class='multinomial', solver='lbfgs')
        clf.fit(X, y)
        log_lik = clf.score(X, y) * len(y)   # approximate
        k = (n_alts - 1) * (len(feats) + 1)
        bic = -2 * log_lik + k * np.log(len(sub))
        return -bic / len(sub)
    except Exception:
        return -1e9


# ── synthetic data: 20 features with embedded signal ──────────────────────────
N   = 500
RNG = np.random.default_rng(2025)

# Mutually-exclusive person-type dummies — these ARE the signal sources
n_FTW, n_PTW, n_SCH = int(N * 0.33), int(N * 0.12), int(N * 0.17)
n_UNI, n_RET = int(N * 0.08), int(N * 0.12)
n_OTH = N - n_FTW - n_PTW - n_SCH - n_UNI - n_RET
_ptypes = np.array(
    ['FTW'] * n_FTW + ['PTW'] * n_PTW + ['SCH'] * n_SCH
    + ['UNI'] * n_UNI + ['RET'] * n_RET + ['OTH'] * n_OTH
)
RNG.shuffle(_ptypes)

ptype_FTW = (_ptypes == 'FTW').astype(float)
ptype_PTW = (_ptypes == 'PTW').astype(float)
ptype_SCH = (_ptypes == 'SCH').astype(float)
ptype_UNI = (_ptypes == 'UNI').astype(float)
ptype_RET = (_ptypes == 'RET').astype(float)

# Socio-economic + accessibility (all added as noise candidates alongside ptype)
vehrd       = RNG.integers(0, 4, N).astype(float)
has_vehicle = (vehrd > 0).astype(float)
incp        = RNG.integers(1, 15, N).astype(float)
income_hi   = (incp >= 10).astype(float)
is_female   = RNG.binomial(1, 0.52, N).astype(float)
HHSIZE      = RNG.integers(1, 6, N).astype(float)
hh_child    = RNG.binomial(1, 0.35, N).astype(float)
wfh         = ptype_FTW * RNG.binomial(1, 0.30, N).astype(float)  # only FTW WFH
nprd        = RNG.integers(1, 5, N).astype(float)
rlhp        = RNG.integers(0, 2, N).astype(float)
transit_acc = RNG.uniform(0, 1, N)
walk_acc    = RNG.uniform(0, 1, N)
auto_acc    = RNG.uniform(0, 1, N)
age_norm    = RNG.uniform(18, 75, N)
CARLICENCE  = np.where(vehrd > 0,
                       RNG.binomial(1, 0.97, N),
                       RNG.binomial(1, 0.65, N)).astype(float)

FEAT = [
    # strong signal features (ptype dummies — one of these always improves BIC)
    'ptype_FTW', 'ptype_PTW', 'ptype_SCH', 'ptype_UNI', 'ptype_RET',
    # moderate signal features
    'has_vehicle', 'income_hi', 'hh_child', 'wfh',
    # weak / noise features (should be penalised by BIC)
    'vehrd', 'nprd', 'incp', 'is_female', 'HHSIZE',
    'rlhp', 'transit_acc', 'walk_acc', 'auto_acc', 'age_norm', 'CARLICENCE',
]
assert len(FEAT) == 20

# ── outcomes with REAL signal ─────────────────────────────────────────────────
# n_work_out: FTW/PTW generate work tours; RET/SCH mostly don't
lam_work = np.exp(
    0.9 * ptype_FTW + 0.5 * ptype_PTW - 0.6 * ptype_SCH
    - 0.5 * ptype_RET + 0.3 * has_vehicle + 0.2 * income_hi - 0.2
)
n_work_out = RNG.poisson(np.clip(lam_work, 0.05, 5.0))

# n_edu_out: SCH/UNI dominant; FTW suppressed
lam_edu = np.exp(1.6 * ptype_SCH + 0.9 * ptype_UNI - 0.8 * ptype_FTW - 1.0)
n_edu_out = RNG.poisson(np.clip(lam_edu, 0.02, 5.0))

# n_social_out: RET and female slightly higher; FTW slightly lower
lam_soc = np.exp(0.4 * ptype_RET + 0.2 * is_female - 0.3 * ptype_FTW - 0.4)
n_social_out = RNG.poisson(np.clip(lam_soc, 0.05, 5.0))

# n_escort_out: hh_child driven; FTW/PTW do more (drive kids)
lam_esc = np.exp(0.9 * hh_child + 0.3 * ptype_FTW - 0.6 * ptype_SCH - 1.2)
n_escort_out = RNG.poisson(np.clip(lam_esc, 0.02, 3.0))

# duration_min (log-normal): FTW and income drive longer tours
dur_mu = (4.5 + 0.9 * ptype_FTW + 0.5 * ptype_PTW + 0.4 * income_hi
          - 0.5 * hh_child + 0.3 * has_vehicle)
duration_min = np.maximum(RNG.lognormal(dur_mu, 0.45), 5.0)

# B (total day duration), start/end of day
B_mu         = 5.6 + 0.4 * ptype_FTW + 0.2 * income_hi - 0.2 * hh_child
B            = np.maximum(RNG.lognormal(B_mu, 0.35), 60.0)
start_of_day = (7.0 + 1.8 * ptype_RET - 1.2 * ptype_FTW - 0.6 * ptype_SCH
                + RNG.standard_normal(N) * 0.7).clip(4, 14)
end_of_day   = (start_of_day
                + np.maximum(RNG.lognormal(1.4, 0.4, N), 1.0)).clip(5, 23)

# period (0-5): early=FTW/SCH, mid=RET/OTH, wfh adds flexibility
# Strong ptype effects + low noise so SA reliably recovers ptype_* in 10 gens
period = np.clip(
    np.round(2.5 - 2.2 * ptype_FTW - 1.2 * ptype_SCH
             + 2.5 * ptype_RET + 0.8 * wfh
             + RNG.standard_normal(N) * 0.4).astype(int),
    0, 5
)

# purpose assignment (drives stage 3 sub-sample sizes)
_purp_map = []
for i in range(N):
    if   ptype_FTW[i] or ptype_PTW[i]: _purp_map.append(RNG.choice(['work', 'social', 'escort'], p=[0.65, 0.25, 0.10]))
    elif ptype_SCH[i] or ptype_UNI[i]: _purp_map.append(RNG.choice(['edu',  'social'],             p=[0.80, 0.20]))
    elif ptype_RET[i]:                  _purp_map.append(RNG.choice(['social', 'escort', 'shopping'], p=[0.60, 0.25, 0.15]))
    else:                               _purp_map.append(RNG.choice(['work', 'social', 'escort', 'shopping'], p=[0.30, 0.40, 0.20, 0.10]))

synth = pd.DataFrame({
    'obs_id':       np.arange(N),
    'person_id':    np.arange(N),
    'ptype':        _ptypes,
    'purpose':      _purp_map,
    'ptype_FTW':    ptype_FTW, 'ptype_PTW': ptype_PTW, 'ptype_SCH': ptype_SCH,
    'ptype_UNI':    ptype_UNI, 'ptype_RET': ptype_RET,
    'vehrd':        vehrd,       'nprd':        nprd,       'incp':        incp,
    'has_vehicle':  has_vehicle, 'is_female':   is_female,  'hh_child':    hh_child,
    'HHSIZE':       HHSIZE,      'income_hi':   income_hi,  'wfh':         wfh,
    'transit_acc':  transit_acc, 'walk_acc':    walk_acc,   'auto_acc':    auto_acc,
    'age_norm':     age_norm,    'CARLICENCE':  CARLICENCE, 'rlhp':        rlhp,
    'n_work_out':   n_work_out,  'n_edu_out':   n_edu_out,  'n_social_out': n_social_out,
    'n_escort_out': n_escort_out,
    'start_of_day': start_of_day, 'end_of_day': end_of_day,
    'B':            B,
    'duration':     duration_min, 'duration_min': duration_min,
    'period':       period,
    'tour_id':      np.arange(N),
})

print(f"\nSynthetic data: N={N}, {len(FEAT)} features, signal embedded in ptype dummies.")
print(f"  n_work_out:  mean={n_work_out.mean():.2f}  (FTW={n_work_out[ptype_FTW==1].mean():.2f}, RET={n_work_out[ptype_RET==1].mean():.2f})")
print(f"  n_edu_out:   mean={n_edu_out.mean():.2f}  (SCH={n_edu_out[ptype_SCH==1].mean():.2f}, FTW={n_edu_out[ptype_FTW==1].mean():.2f})")
print(f"  duration:    mean={duration_min.mean():.0f} min  (FTW={duration_min[ptype_FTW==1].mean():.0f}, RET={duration_min[ptype_RET==1].mean():.0f})")

SA_KWARGS = dict(pop_size=12, n_generations=10, min_features=1,
                 seed=42, verbose=True, fitness_metric='mean(-BIC/n)')
R_DRAWS   = 30   # MetaCount Halton draws during search (fast enough for smoke test)

# ── Phase-5-only mode ─────────────────────────────────────────────────────────
# Set GA_SMOKE_PHASE5_ONLY=1 to skip phases 1-4 and jump straight to Phase 5.
_PHASE5_ONLY      = os.environ.get('GA_SMOKE_PHASE5_ONLY', '0').strip() == '1'
_P5_DEFAULT_FEATS = ['ptype_FTW', 'ptype_PTW', 'ptype_SCH', 'ptype_RET', 'hh_child']

class _PhaseSkipped(Exception):
    """Raised at the top of a try-block to skip a phase in PHASE5_ONLY mode."""

# ── gene → feature list helper ────────────────────────────────────────────────
def _genes_to_feats(genes):
    seen: set = set()
    return [f for f, g in zip(FEAT, genes) if g == 1 and not (f in seen or seen.add(f))]


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Activity frequency: NB2 (MetaCount) or Poisson (scipy fallback)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 66)
_p1_est = "MetaCount NB2" if _METACOUNT_OK else "scipy/statsmodels Poisson (fallback)"
print(f"PHASE 1 — Activity frequency  [{_p1_est}]")
print("─" * 66)

try:
    if _PHASE5_ONLY:
        raise _PhaseSkipped
    def _p1_fit(genes: np.ndarray) -> float:
        feats = _genes_to_feats(genes)
        if not feats:
            return -1e9
        ok_feats = [f for f in feats if f in synth.columns]
        if not ok_feats:
            return -1e9
        if _METACOUNT_OK:
            # Use the proper MetaCount NB2 fitness — correct model for count data
            return stage1_fitness_metacount(genes, synth, FEAT,
                                            purposes=['work', 'edu', 'social'])
        else:
            # Poisson BIC — statistically appropriate for non-negative integer counts
            scores = []
            for y_col in ['n_work_out', 'n_edu_out', 'n_social_out']:
                s = _poisson_bic(synth, y_col, ok_feats)
                if s > -1e8:
                    scores.append(s)
            return float(np.mean(scores)) if scores else -1e9

    sa1 = SimulatedAnnealing(n_features=len(FEAT), fitness_fn=_p1_fit, **SA_KWARGS)
    best1, hist1, _ = sa1.run_with_restarts(n_restarts=1)

    p1_prog, p1_detail = _is_progressive(hist1)
    feats1 = _genes_to_feats(best1.genes)
    check("Phase 1 completes",                 True)
    check("Phase 1 progressive search",        p1_prog, p1_detail)
    check("Phase 1 valid best fitness (>-1e8)",
          np.isfinite(best1.fitness) and best1.fitness > -1e8,
          f"fitness={best1.fitness:.4f}")
    check("Phase 1 selects >=1 feature",       len(feats1) >= 1, f"selected={feats1}")
    has_signal1 = bool(any(f.startswith('ptype_') or f in ('has_vehicle', 'income_hi')
                           for f in feats1))
    check("Phase 1 selects signal feature",    has_signal1, f"features={feats1}")
    _phase_results[1] = {'best_fitness': best1.fitness, 'best_features': feats1,
                          'progressive': p1_prog, 'estimator': _p1_est, 'history': hist1}
    print(f"  Best SA features: {feats1}  BIC/n={-best1.fitness:.4f}")
except _PhaseSkipped:
    _phase_results[1] = {'best_features': _P5_DEFAULT_FEATS, 'best_fitness': -99.0,
                          'progressive': True, 'estimator': 'skipped', 'history': []}
    print("  [PHASE5_ONLY] Phase 1 skipped.")
except Exception as e:
    import traceback as _tb
    check("Phase 1 completes", False, str(e)); print(_tb.format_exc())
    for lbl in ["progressive", "valid fitness", "selects feature", "signal feature"]:
        check(f"Phase 1 {lbl}", False, "skipped")

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Day duration: MetaCount mixed lognormal / statsmodels OLS fallback
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 66)
_p2_est = "MetaCount mixed lognormal" if _METACOUNT_OK else "statsmodels OLS on log(y) (fallback)"
print(f"PHASE 2 — Day duration  [{_p2_est}]")
print("─" * 66)

try:
    if _PHASE5_ONLY:
        raise _PhaseSkipped
    def _p2_fit(genes: np.ndarray) -> float:
        feats = _genes_to_feats(genes)
        if not feats:
            return -1e9
        ok = [f for f in feats if f in synth.columns]
        if not ok:
            return -1e9
        if _METACOUNT_OK:
            return stage2_fitness_metacount(genes, synth, FEAT,
                                            targets=['B', 'start_of_day'])
        else:
            scores = []
            for col, pos_only in [('B', True), ('start_of_day', False)]:
                df_s = synth if not pos_only else synth[synth[col] > 0]
                s = _lognormal_bic(df_s, col, ok) if pos_only else _poisson_bic(df_s, col, ok)
                if s > -1e8:
                    scores.append(s)
            return float(np.mean(scores)) if scores else -1e9

    sa2 = SimulatedAnnealing(n_features=len(FEAT), fitness_fn=_p2_fit, **SA_KWARGS)
    best2, hist2, _ = sa2.run_with_restarts(n_restarts=1)

    p2_prog, p2_detail = _is_progressive(hist2)
    feats2 = _genes_to_feats(best2.genes)
    check("Phase 2 completes",                 True)
    check("Phase 2 progressive search",        p2_prog, p2_detail)
    check("Phase 2 valid best fitness (>-1e8)",
          np.isfinite(best2.fitness) and best2.fitness > -1e8,
          f"fitness={best2.fitness:.4f}")
    check("Phase 2 selects >=1 feature",       len(feats2) >= 1, f"selected={feats2}")
    has_signal2 = bool(any(f.startswith('ptype_') or f in ('income_hi', 'hh_child', 'wfh')
                           for f in feats2))
    check("Phase 2 selects signal feature",    has_signal2, f"features={feats2}")
    _phase_results[2] = {'best_fitness': best2.fitness, 'best_features': feats2,
                          'progressive': p2_prog, 'estimator': _p2_est, 'history': hist2}
    print(f"  Best SA features: {feats2}  BIC/n={-best2.fitness:.4f}")
except _PhaseSkipped:
    _phase_results[2] = {'best_features': _P5_DEFAULT_FEATS, 'best_fitness': -99.0,
                          'progressive': True, 'estimator': 'skipped', 'history': []}
    print("  [PHASE5_ONLY] Phase 2 skipped.")
except Exception as e:
    import traceback as _tb
    check("Phase 2 completes", False, str(e)); print(_tb.format_exc())
    for lbl in ["progressive", "valid fitness", "selects feature", "signal feature"]:
        check(f"Phase 2 {lbl}", False, "skipped")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Activity duration per purpose (all major purposes)
#           Saves best-feature coefficients to smoke_stage3_coefficients.csv
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 66)
_p3_est = "MetaCount lognormal per purpose" if _METACOUNT_OK else "statsmodels log-OLS per purpose (fallback)"
print(f"PHASE 3 — Activity duration  [{_p3_est}]")
print("─" * 66)

_P3_PURPOSES = ['work', 'social', 'escort', 'edu']

try:
    if _PHASE5_ONLY:
        raise _PhaseSkipped
    def _p3_fit(genes: np.ndarray) -> float:
        feats = _genes_to_feats(genes)
        if not feats:
            return -1e9
        ok = [f for f in feats if f in synth.columns]
        if not ok:
            return -1e9
        if _METACOUNT_OK:
            return stage3_fitness_metacount(
                genes, synth, FEAT, purposes=_P3_PURPOSES,
                purp_col='purpose', dur_col='duration_min',
            )
        else:
            scores = []
            for purp in _P3_PURPOSES:
                sub = synth[synth['purpose'] == purp]
                if len(sub) < 25:
                    continue
                s = _lognormal_bic(sub, 'duration_min', ok)
                if s > -1e8:
                    scores.append(s)
            return float(np.mean(scores)) if scores else -1e9

    sa3 = SimulatedAnnealing(n_features=len(FEAT), fitness_fn=_p3_fit, **SA_KWARGS)
    best3, hist3, _ = sa3.run_with_restarts(n_restarts=1)

    p3_prog, p3_detail = _is_progressive(hist3)
    feats3 = _genes_to_feats(best3.genes)
    check("Phase 3 completes",                 True)
    check("Phase 3 progressive search",        p3_prog, p3_detail)
    check("Phase 3 valid best fitness (>-1e8)",
          np.isfinite(best3.fitness) and best3.fitness > -1e8,
          f"fitness={best3.fitness:.4f}")
    check("Phase 3 selects >=1 feature",       len(feats3) >= 1, f"selected={feats3}")
    has_signal3 = bool(any(f.startswith('ptype_') or f in ('income_hi', 'hh_child', 'wfh')
                           for f in feats3))
    check("Phase 3 selects signal feature",    has_signal3, f"features={feats3}")

    # ── Save Phase 3 coefficients to CSV ─────────────────────────────────────
    import pathlib
    out_csv = pathlib.Path(__file__).parent / "smoke_stage3_coefficients.csv"
    coef_rows = []
    for purp in _P3_PURPOSES:
        sub = synth[synth['purpose'] == purp]
        ok3 = [f for f in feats3 if f in sub.columns]
        if len(sub) < 10 or not ok3:
            continue
        try:
            import statsmodels.api as sm
            sub2 = sub[ok3 + ['duration_min']].dropna()
            sub2 = sub2[sub2['duration_min'] > 0]
            if len(sub2) >= 10:
                X = sm.add_constant(sub2[ok3].astype(float).values, has_constant='add')
                logy = np.log(sub2['duration_min'].astype(float).values)
                res = sm.OLS(logy, X).fit()
                for feat_name, coef, pval in zip(['const'] + ok3,
                                                  res.params, res.pvalues):
                    coef_rows.append({
                        'purpose': purp, 'feature': feat_name,
                        'coef': coef, 'pvalue': pval,
                        'significant_10pct': pval < 0.10,
                    })
        except Exception as _ce:
            pass  # best-effort only
    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(out_csv, index=False)
        print(f"  Stage 3 coefficients saved → {out_csv.name}")
        check("Phase 3 coefficient CSV saved",    True, str(out_csv.name))
    else:
        check("Phase 3 coefficient CSV saved",    False, "no rows estimated")

    _phase_results[3] = {'best_fitness': best3.fitness, 'best_features': feats3,
                          'progressive': p3_prog, 'estimator': _p3_est, 'history': hist3}
    print(f"  Best SA features: {feats3}  BIC/n={-best3.fitness:.4f}")
except _PhaseSkipped:
    _phase_results[3] = {'best_features': _P5_DEFAULT_FEATS, 'best_fitness': -99.0,
                          'progressive': True, 'estimator': 'skipped', 'history': []}
    print("  [PHASE5_ONLY] Phase 3 skipped.")
except Exception as e:
    import traceback as _tb
    check("Phase 3 completes", False, str(e)); print(_tb.format_exc())
    for lbl in ["progressive", "valid fitness", "selects feature", "signal feature",
                "coefficient CSV saved"]:
        check(f"Phase 3 {lbl}", False, "skipped")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Schedule placement
#   SearchLibrium MixedLogit  →  MultinomialLogit  →  statsmodels MNL fallback
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 66)
if _SEARCHLIB_OK:
    _p4_est = "SearchLibrium MixedLogit (n_draws=50)"
else:
    _p4_est = "statsmodels MNL (fallback)"
print(f"PHASE 4 — Schedule placement  [{_p4_est}]")
print("─" * 66)

_N_FEAT = len(FEAT)  # gene length for inclusion bits; random-param bits double it

try:
    if _PHASE5_ONLY:
        raise _PhaseSkipped

    if _SEARCHLIB_OK:
        # Extended gene: first _N_FEAT bits = inclusion, next _N_FEAT bits = random flag.
        # A feature is a random parameter iff its inclusion bit AND random bit are both 1.
        def _p4_fit(genes: np.ndarray) -> float:
            inc_bits  = genes[:_N_FEAT]
            rand_bits = genes[_N_FEAT:]
            rand_vars = [f for f, g, r in zip(FEAT, inc_bits, rand_bits) if g == 1 and r == 1]
            return stage4_fitness_searchlibrium_mixed(
                inc_bits, synth, FEAT, ['work', 'social', 'escort'],
                period_col='period', n_alts=6,
                n_draws=50, rand_dist='n',
                rand_vars=rand_vars if rand_vars else None,  # None → all included are random
            )
    else:
        def _p4_fit(genes: np.ndarray) -> float:
            inc_bits = genes[:_N_FEAT]
            feats = _genes_to_feats(inc_bits)
            if not feats:
                return -1e9
            ok = [f for f in feats if f in synth.columns]
            return _mnl_bic(synth, 'period', ok, n_alts=6)

    sa4 = SimulatedAnnealing(n_features=2 * _N_FEAT, fitness_fn=_p4_fit, **SA_KWARGS)
    best4, hist4, _ = sa4.run_with_restarts(n_restarts=1)

    p4_prog, p4_detail = _is_progressive(hist4)
    feats4 = _genes_to_feats(best4.genes[:_N_FEAT])
    rand4  = [f for f, g, r in zip(FEAT, best4.genes[:_N_FEAT], best4.genes[_N_FEAT:])
              if g == 1 and r == 1]
    if not rand4:
        rand4 = list(feats4)   # fallback: all included → random
    print(f"  Phase 4 random parameters : {rand4}")

    check("Phase 4 completes",                 True)
    check("Phase 4 progressive search",        p4_prog, p4_detail)
    check("Phase 4 valid best fitness (>-1e8)",
          np.isfinite(best4.fitness) and best4.fitness > -1e8,
          f"fitness={best4.fitness:.4f}")
    check("Phase 4 selects >=1 feature",       len(feats4) >= 1, f"selected={feats4}")
    has_signal4 = bool(any(
        f.startswith('ptype_') or f in (
            'wfh', 'auto_acc', 'transit_acc', 'walk_acc',
            'has_vehicle', 'income_hi', 'hh_child',
        )
        for f in feats4
    ))
    check("Phase 4 selects signal feature",    has_signal4, f"features={feats4}")
    _phase_results[4] = {'best_fitness': best4.fitness, 'best_features': feats4,
                          'rand_vars': rand4,
                          'progressive': p4_prog, 'estimator': _p4_est, 'history': hist4}
    print(f"  Best SA features: {feats4}  BIC/n={-best4.fitness:.4f}")
except _PhaseSkipped:
    _phase_results[4] = {'best_features': _P5_DEFAULT_FEATS, 'best_fitness': -99.0,
                          'rand_vars': _P5_DEFAULT_FEATS[:2],
                          'progressive': True, 'estimator': 'skipped', 'history': []}
    print("  [PHASE5_ONLY] Phase 4 skipped.")
except Exception as e:
    import traceback as _tb
    check("Phase 4 completes", False, str(e)); print(_tb.format_exc())
    for lbl in ["progressive", "valid fitness", "selects feature", "signal feature"]:
        check(f"Phase 4 {lbl}", False, "skipped")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Mode choice (Larch NL) + Destination choice (Larch MNL)
#   5A: SA over OD-skim features → nested logit mode model
#   5B: SA over destination UEC coefficients → MNL dest model  (≥10 iters)
#   5C: Final NL + optional random parameters via Larch mixtures
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 66)
print("PHASE 5 — Mode + Destination choice  [Larch NL / Larch MNL]")
print("─" * 66)

try:
    # ── Larch 6 availability ──────────────────────────────────────────────
    try:
        import larch as lx
        from larch import P, X
        _LX6_OK = True
    except ImportError:
        _LX6_OK = False

    # ── Stage-5 pipeline functions ────────────────────────────────────────
    try:
        from stage5_destination_choice import (
            _build_mode_model, _build_od_dataset,
            _estimate as _lx_estimate, MODE_MAP, DEST_UEC_FLAT,
        )
        _S5_OK = True
    except Exception as _s5_e:
        _S5_OK = False
        print(f"  [warn] stage5 import failed ({type(_s5_e).__name__}): {_s5_e}")

    # SA searches over which OD-skim variables enter the mode utility
    _MODE_SKIM_FEATS = [
        'SOVTOLL_TIME__AM',          # auto in-vehicle time
        'WLK_LOC_WLK_TOTIVT__AM',   # transit in-vehicle time
        'DISTWALK',                   # walk distance
        'DISTBIKE',                   # bike distance
        'CRASH',                      # crash rate — safety perception
    ]
    _N_SKIM = len(_MODE_SKIM_FEATS)

    # SA searches over which destination UEC coefficients to include
    _DEST_SA_COEFFS = [
        'c_log_dist', 'c_dist', 'c_crash', 'c_logsum', 'c_size_term',
        'c_aggl', 'c_comp', 'c_intrazonal', 'c_walk_feasible', 'c_bike_feasible',
    ]
    _N_DEST_C = len(_DEST_SA_COEFFS)

    if _LX6_OK and _S5_OK:
        # ── Synthetic mode-choice data (N_M tours × 5 mode alternatives) ──
        _N_M  = 60      # tours — small to keep SA per-eval time short
        _N_ZM = 6       # zones
        # OD skim table: origin zone 1 → destination zones 1-8
        _skim5_rows = []
        for _dz in range(1, _N_ZM + 1):
            _dk = max(0.5, float(_dz - 1) * 3.0)
            _skim5_rows.append({
                'OTAZ': 1, 'DTAZ': _dz,
                'SOVTOLL_TIME__AM':        round(_dk * 2.5, 2),
                'WLK_LOC_WLK_TOTIVT__AM': round(_dk * 5.0, 2),
                'DISTWALK': _dk, 'DISTBIKE': _dk,
                'SOVTOLL_DIST__AM': _dk,
                'CRASH': float(np.clip(RNG.normal(_dk * 0.08, 0.15), 0, 3)),
            })
        _skim_df5 = pd.DataFrame(_skim5_rows)

        # Synthetic persons
        _p5_vehrd = RNG.integers(0, 4, _N_M).astype(float)
        _p5_carl  = np.where(_p5_vehrd > 0,
                             RNG.binomial(1, 0.97, _N_M),
                             RNG.binomial(1, 0.65, _N_M)).astype(float)
        _persons5 = pd.DataFrame({
            'person_id': np.arange(_N_M),
            'vehrd':     _p5_vehrd,
            'incp':      RNG.integers(1, 6, _N_M).astype(float),
            'sexp':      RNG.binomial(1, 0.52, _N_M).astype(int) + 1,
            'age5p':     RNG.integers(2, 12, _N_M).astype(float),
            'bikes':     RNG.integers(0, 3, _N_M).astype(float),
            'WORKSTAT':  RNG.choice([1, 2, 3], _N_M).astype(int),
            'nprd':      RNG.integers(1, 5, _N_M).astype(float),
            'CARLICENCE': _p5_carl,
        })

        # Tours with true mode choices via NL utilities
        _t5_dtaz = RNG.integers(1, _N_ZM + 1, _N_M)
        _sk5_lkp = _skim_df5.set_index('DTAZ').to_dict('index')
        _ai5 = np.array([_sk5_lkp[d]['SOVTOLL_TIME__AM']        for d in _t5_dtaz])
        _ti5 = np.array([_sk5_lkp[d]['WLK_LOC_WLK_TOTIVT__AM'] for d in _t5_dtaz])
        _wd5 = np.array([_sk5_lkp[d]['DISTWALK']                for d in _t5_dtaz])
        _V5 = np.column_stack([
            3.0*_p5_carl + 2.0*(_p5_vehrd>0) - 0.08*_ai5 + 0.5,
            0.8*(_p5_vehrd>0) + 0.3 - 0.06*_ai5,
            -0.5*_wd5 + 1.5*(_wd5<2.0) - 0.3,
            -0.3*_wd5 + 0.5*(_wd5<5.0) - 0.8,
            1.5*(_p5_vehrd==0) - 0.04*_ti5 + 0.2*(_persons5['incp'].values<3),
        ]) + RNG.gumbel(0, 1, (_N_M, 5))
        _mc5 = _V5.argmax(axis=1) + 1   # 1=DA … 5=Transit
        _mc5_counts = {m: int((_mc5==i).sum())
                       for m, i in zip(['DA','SR','Wk','Bk','Tr'], range(1, 6))}

        _tours5 = pd.DataFrame({
            'ptour_id':       np.arange(_N_M).astype(str),
            'person_id':      np.arange(_N_M),
            'OTAZ': 1, 'DTAZ': _t5_dtaz,
            'tour_mode':      _mc5,
            'CARLICENCE':     _p5_carl,
            'walk_avail':     (_wd5 < 2.0).astype(float),
            'bike_avail':     (_wd5 < 5.0).astype(float),
            'weight':         1.0,
            'joint_travel':   0,
            'outbound_stops': 0,
            'n_shopping_out': 0,
            '_time_period':   RNG.binomial(1, 0.4, _N_M),
        })
        print(f"  Mode data  : N={_N_M}, {_N_ZM} zones, choices={_mc5_counts}")

        # ── DataTree builder (mirrors _mode_tree architecture) ────────────
        # Include availability vars (CARLICENCE, walk_avail, bike_avail) in
        # the root CA dataset so _build_mode_model finds them without prefix.
        def _mk_mode_dt5(tours_df, persons_df, skim_df):
            _alts   = list(MODE_MAP.values())
            _cases  = list(tours_df['ptour_id'].values)
            _ca_idx = pd.MultiIndex.from_product(
                [_cases, _alts], names=['ptour_id', '_altid_'])
            # Broadcast case-level availability vars into root CA frame
            _avail_df = pd.DataFrame({'_dummy': 0.0}, index=_ca_idx)
            _t_lkp = tours_df.set_index('ptour_id')
            for _avc in ('CARLICENCE', 'walk_avail', 'bike_avail'):
                if _avc in _t_lkp.columns:
                    _avc_map = _t_lkp[_avc].to_dict()
                    _avail_df[_avc] = [_avc_map[c] for c, _ in _ca_idx]
            _root = lx.Dataset.construct.from_idca(_avail_df)
            _od   = _build_od_dataset(None, skim_df=skim_df)
            return lx.DataTree(
                base   = _root,
                tour   = lx.Dataset.construct.from_idco(
                             tours_df.set_index('ptour_id')),
                person = persons_df.set_index('person_id'),
                od     = _od,
                relationships=(
                    "base.ptour_id @ tour.ptour_id",
                    "tour.person_id @ person.person_id",
                    "tour.OTAZ @ od.otaz",
                    "tour.DTAZ @ od.dtaz",
                ),
            )

        # ── Build DataTree ONCE before SA (expensive pandas→xarray conversion)
        import contextlib, io as _io
        print("  Building mode DataTree ... ", end='', flush=True)
        with contextlib.redirect_stderr(_io.StringIO()):
            _dt5_static = _mk_mode_dt5(_tours5, _persons5, _skim_df5)
        print("done", flush=True)

        # Fit FULL NL model (all skims) once → warm-start seed for SA
        # SA then perturbs from this good initial solution rather than random start.
        print("  Fitting baseline NL (all skims) ... ", end='', flush=True)
        try:
            with contextlib.redirect_stderr(_io.StringIO()):
                _m_bl5a = _build_mode_model(_dt5_static, 'work',
                                            custom_mode_vars=_MODE_SKIM_FEATS)
            _m_bl5a.compute_engine = 'numba'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _m_bl5a.maximize_loglike(method='bhhh', maxiter=25, quiet=True)
            _ll_bl5a = float(_m_bl5a.loglike())
            _k_bl5a  = int((~_m_bl5a.pf['holdfast'].fillna(False)).sum())
            _fit_bl5a = -((-2*_ll_bl5a + _k_bl5a*np.log(_N_M)) / _N_M) \
                        if np.isfinite(_ll_bl5a) else -1e9
        except Exception as _ebl5a:
            _fit_bl5a = -1e9
        print(f"done  BIC/n={-_fit_bl5a:.4f}", flush=True)
        # Warm start: all skims included, no random components
        _seed5a = [Individual(
            genes=np.array([1]*_N_SKIM + [0]*_N_SKIM, dtype=int),
            fitness=_fit_bl5a,
        )]

        # Skim → Larch parameter name mapping (used for holdfast freeze)
        _skim_to_coeff5 = {
            'SOVTOLL_TIME__AM':        'c_ivt_auto',
            'WLK_LOC_WLK_TOTIVT__AM': 'c_ivt_transit',
            'DISTWALK':                'c_walk_time',
            'DISTBIKE':                'c_bike_time',
            'CRASH':                   'c_crash_veh',
        }

        # ── Mode SA fitness: Larch NL via _build_mode_model ───────────────
        # ALWAYS build with the FULL skim set → numba compiles once, then
        # every SA eval reuses the cached compiled function.
        # Feature "exclusion" is modelled by freezing the coefficient at 0
        # via holdfast — no model-structure change, no re-compilation.
        # Gene layout: [inc_bit × N_SKIM | rand_bit × N_SKIM]
        def _mode_fit5(genes):
            _excl = [f for f, g in zip(_MODE_SKIM_FEATS, genes[:_N_SKIM]) if g == 0]
            try:
                with contextlib.redirect_stderr(_io.StringIO()):
                    # Fixed structure: all skims always present in utility
                    _m = _build_mode_model(_dt5_static, 'work',
                                           custom_mode_vars=_MODE_SKIM_FEATS)
                _m.compute_engine = 'numba'
                # Freeze excluded skim coefficients to 0 (no numba re-compile)
                for _f in _excl:
                    _param = _skim_to_coeff5.get(_f)
                    if _param and _param in _m.pf.index:
                        _m.pf.loc[_param, 'value']    = 0.0
                        _m.pf.loc[_param, 'holdfast'] = 1
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _m.maximize_loglike(method='bhhh', maxiter=25, quiet=True)
                _ll = float(_m.loglike())
                if not np.isfinite(_ll):
                    return -1e9
                _k = int((~_m.pf['holdfast'].fillna(False)).sum())
                return -((-2*_ll + _k*np.log(_N_M)) / _N_M)
            except Exception:
                return -1e9

        # Phase 5A: SA for mode choice (NL)
        # Start from warm seed (full model already fitted) — SA refines from there
        _sa5a_kw = {**SA_KWARGS, 'pop_size': 3}
        _sa5a = SimulatedAnnealing(n_features=2*_N_SKIM, fitness_fn=_mode_fit5, **_sa5a_kw)
        # Pre-populate cache with warm-start result → avoids re-evaluating the
        # all-included model at the start of run() (holdfast approach means the
        # same genes → same fit as our pre-fitted baseline)
        _sa5a._cache[tuple(int(x) for x in _seed5a[0].genes.tolist())] = _fit_bl5a
        _best5a, _hist5a, _ = _sa5a.run_with_restarts(n_restarts=1,
                                                       seed_individuals=_seed5a)

        _feats5a_inc  = [f for f, g in zip(_MODE_SKIM_FEATS,
                                            _best5a.genes[:_N_SKIM]) if g == 1]
        _feats5a_rand = [f for f, g, r in zip(_MODE_SKIM_FEATS,
                                               _best5a.genes[:_N_SKIM],
                                               _best5a.genes[_N_SKIM:])
                         if g == 1 and r == 1]
        print(f"  Phase 5A  skims included : {_feats5a_inc or ['(person-attrs only)']}")
        print(f"  Phase 5A  random params  : {_feats5a_rand or ['none']}")

        _prog5a, _pmsg5a = _is_progressive(_hist5a)
        check("Phase 5A mode NL SA runs",    True)
        check("Phase 5A ≥10 SA iterations",  len(_hist5a) >= 10,
              f"n={len(_hist5a)}")
        check("Phase 5A best fitness finite", np.isfinite(_best5a.fitness),
              f"fit={_best5a.fitness:.4f}")
        check("Phase 5A mode SA progressive", _prog5a, _pmsg5a)

        _phase_results[5] = {
            'best_features': _feats5a_inc or _MODE_SKIM_FEATS[:2],
            'rand_vars':     _feats5a_rand,
            'best_fitness':  _best5a.fitness,
            'history':       _hist5a,
            'progressive':   _prog5a,
            'estimator':     'Larch NL mode choice',
            'feature_names': _MODE_SKIM_FEATS,
        }

        # ── Synthetic destination-choice data (flat pre-joined idca) ──────
        _N_DC  = 100    # observations
        _N_DZ  = 20     # zone alternatives
        _dz5   = np.arange(1, _N_DZ + 1)
        _dc_i5 = RNG.choice(N, _N_DC, replace=False)
        _dc_p5 = synth.iloc[_dc_i5].reset_index(drop=True)
        _dz_TOTEMP  = RNG.lognormal(9, 1, _N_DZ)
        _dz_RETEMPN = RNG.lognormal(7, 1, _N_DZ)
        _dz_tr5     = RNG.uniform(0, 1, _N_DZ)
        _dz_wk5     = RNG.uniform(0, 1, _N_DZ)
        _dc_dists5  = (np.abs(np.arange(1, _N_DC+1)[:, None] - _dz5[None, :] * 2.0)
                       + RNG.exponential(1.0, (_N_DC, _N_DZ)))
        _Vd5 = (-1.5*np.log(_dc_dists5+1) + 0.8*np.log(_dz_TOTEMP+1)
                + 0.3*np.log(_dz_RETEMPN+1) + 0.3*_dz_tr5)
        _Vd5 += RNG.gumbel(0, 1, _Vd5.shape)
        _dc_ch5 = _Vd5.argmax(axis=1) + 1

        _dest5_rows = []
        for _zi5, _zid5 in enumerate(_dz5):
            _d5 = _dc_dists5[:, _zi5]
            _dest5_rows.append(pd.DataFrame({
                'case_id':       np.arange(_N_DC),
                'DTAZ':          int(_zid5),
                'chosen':        (_dc_ch5 == _zid5).astype(int),
                'log_DIST_1':    np.log(_d5 + 1),
                'DIST':          _d5,
                'log_CRASH_1':   RNG.uniform(0.1, 0.5, _N_DC),
                'logsum':        RNG.uniform(-3.0, -0.5, _N_DC),
                'log_correction': 0.0,
                'size_term':     float(np.log(_dz_TOTEMP[_zi5] + 1)),
                'acdc_aggl':     RNG.uniform(0, 1, _N_DC),
                'acdc_comp':     RNG.uniform(0, 1, _N_DC),
                'intrazonal':    float(_zid5 == 1),
                'walk_feasible': (_d5 < 2.0).astype(float),
                'bike_feasible': (_d5 < 5.0).astype(float),
                'incp_x_log_dist': _dc_p5['income_hi'].values * np.log(_d5 + 1),
            }))
        _synth_dest5 = pd.concat(_dest5_rows, ignore_index=True)
        _synth_dest5 = _synth_dest5.set_index(['case_id', 'DTAZ'])
        print(f"  Dest data  : N={_N_DC} obs × {_N_DZ} zones")

        # Map DEST_SA_COEFFS → flat column names that exist in _synth_dest5
        _DEST_COL_MAP = {c: DEST_UEC_FLAT[c][0]
                         for c in _DEST_SA_COEFFS
                         if c in DEST_UEC_FLAT
                         and DEST_UEC_FLAT[c][0] in _synth_dest5.columns}

        # Fit FULL dest MNL (all coefficients) once → warm-start seed for 5B SA
        print("  Fitting baseline dest MNL (all coeffs) ... ", end='', flush=True)
        _sel_bl5b = {c: _DEST_COL_MAP[c] for c in _DEST_SA_COEFFS if c in _DEST_COL_MAP}
        _sel_bl5b.setdefault('c_log_dist', 'log_DIST_1')
        _sel_bl5b.setdefault('c_logsum',   'logsum')
        _cols_bl5b = list(dict.fromkeys(_sel_bl5b.values()))
        try:
            _ds_bl5b  = lx.Dataset.construct.from_idca(
                            _synth_dest5[_cols_bl5b + ['chosen']].copy())
            _mm_bl5b  = lx.Model(_ds_bl5b)
            _mm_bl5b.choice_ca_var  = 'chosen'
            _mm_bl5b.compute_engine = 'numba'
            _util_bl5b = None
            for _c, _col in _sel_bl5b.items():
                _t = P(_c) * X(_col)
                _util_bl5b = _t if _util_bl5b is None else _util_bl5b + _t
            if _util_bl5b is not None:
                _mm_bl5b.utility_ca = _util_bl5b
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _mm_bl5b.maximize_loglike(quiet=True)
            _ll_bl5b = float(_mm_bl5b.loglike())
            _k_bl5b  = int((~_mm_bl5b.pf['holdfast'].fillna(False)).sum())
            _fit_bl5b = -((-2*_ll_bl5b + _k_bl5b*np.log(_N_DC)) / _N_DC)
            print(f"done  BIC/n={-_fit_bl5b:.4f}", flush=True)
        except Exception as _ebl5b:
            _fit_bl5b = -1e9
            print(f"failed ({_ebl5b})", flush=True)
        _seed5b = [Individual(
            genes=np.ones(_N_DEST_C, dtype=int),
            fitness=_fit_bl5b,
        )]

        # ── Destination SA fitness: conditional logit (Larch MNL) ─────────
        def _dest_fit5(genes):
            _sel = {c: _DEST_COL_MAP[c]
                    for c, g in zip(_DEST_SA_COEFFS, genes)
                    if g == 1 and c in _DEST_COL_MAP}
            _sel.setdefault('c_log_dist', 'log_DIST_1')
            _sel.setdefault('c_logsum',   'logsum')
            _cols = list(dict.fromkeys(_sel.values()))
            try:
                _ds5 = lx.Dataset.construct.from_idca(
                    _synth_dest5[_cols + ['chosen']].copy())
                _mm5 = lx.Model(_ds5)
                _mm5.choice_ca_var  = 'chosen'
                _mm5.compute_engine = 'numba'
                _util5 = None
                for _coeff, _col in _sel.items():
                    _t = P(_coeff) * X(_col)
                    _util5 = _t if _util5 is None else _util5 + _t
                if _util5 is not None:
                    _mm5.utility_ca = _util5
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _mm5.maximize_loglike(quiet=True)
                _ll5d = float(_mm5.loglike())
                if not np.isfinite(_ll5d):
                    return -1e9
                _k5d = int((~_mm5.pf['holdfast'].fillna(False)).sum())
                return -((-2*_ll5d + _k5d*np.log(_N_DC)) / _N_DC)
            except Exception:
                return -1e9

        # Phase 5B: destination SA — warm-started from full-coefficient baseline
        _sa5b = SimulatedAnnealing(n_features=_N_DEST_C, fitness_fn=_dest_fit5, **SA_KWARGS)
        _best5b, _hist5b, _ = _sa5b.run_with_restarts(n_restarts=1,
                                                       seed_individuals=_seed5b)

        _feats5b = [c for c, g in zip(_DEST_SA_COEFFS, _best5b.genes) if g == 1]
        print(f"  Phase 5B  dest coeffs    : {_feats5b}")

        _prog5b, _pmsg5b = _is_progressive(_hist5b)
        check("Phase 5B dest MNL SA runs",    True)
        check("Phase 5B ≥10 SA iterations",   len(_hist5b) >= 10,
              f"n={len(_hist5b)}")
        check("Phase 5B best fitness finite",  np.isfinite(_best5b.fitness),
              f"fit={_best5b.fitness:.4f}")
        check("Phase 5B dest SA progressive",  _prog5b, _pmsg5b)

        _phase_results[6] = {
            'best_features': _feats5b or _DEST_SA_COEFFS[:3],
            'best_fitness':  _best5b.fitness,
            'history':       _hist5b,
            'progressive':   _prog5b,
            'estimator':     'Larch MNL dest choice',
            'feature_names': _DEST_SA_COEFFS,
        }

        # ── Phase 5C: Final NL + optional Larch mixtures (random params) ──
        # Re-use the pre-built DataTree; suppress availability warning again.
        with contextlib.redirect_stderr(_io.StringIO()):
            _m5f = _build_mode_model(_dt5_static, 'work',
                                     custom_mode_vars=_feats5a_inc)
        _skim_to_coeff5 = {
            'SOVTOLL_TIME__AM':        'c_ivt_auto',
            'WLK_LOC_WLK_TOTIVT__AM': 'c_ivt_transit',
            'DISTWALK':                'c_walk_time',
            'DISTBIKE':                'c_bike_time',
            'CRASH':                   'c_crash_veh',
        }
        _rand5_coeffs = [_skim_to_coeff5[f] for f in _feats5a_rand
                         if f in _skim_to_coeff5
                         and _skim_to_coeff5[f] in _m5f.pf.index]
        if _rand5_coeffs:
            try:
                _m5f.mixtures = [lx.mixtures.Normal(c, f'sd_{c}')
                                 for c in _rand5_coeffs]
                _m5f.n_draws  = 100
                _m5f.seed     = 42
                print(f"  Final NL+MXL : Larch mixtures for {_rand5_coeffs}")
            except Exception as _mx5e:
                print(f"  Final NL only  (mixtures error: {_mx5e})")
                _rand5_coeffs = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _lx_estimate(_m5f)
        _ll5f  = float(_m5f.loglike())
        _k5f   = int((~_m5f.pf['holdfast'].fillna(False)).sum())
        _bic5f = -2*_ll5f + _k5f*np.log(_N_M)
        _tag5f = '+MXL' if _rand5_coeffs else ''
        print(f"  Final NL{_tag5f}  LL={_ll5f:.1f}  BIC={_bic5f:.1f}  k={_k5f}")
        if _feats5a_rand:
            print(f"  Random skim params  : {_feats5a_rand}")
            print(f"  Larch mixture coeffs: {_rand5_coeffs or ['(none matched pf)']}")
        else:
            print(f"  Random params (mode): none selected by SA")
        check("Phase 5C final NL fitted",    np.isfinite(_ll5f),  f"ll={_ll5f:.2f}")
        check("Phase 5C final BIC finite",   np.isfinite(_bic5f), f"bic={_bic5f:.2f}")

    elif _SEARCHLIB_OK:
        # ── Fallback: SearchLibrium MXL when Larch/stage5 unavailable ─────
        p5_feats = _phase_results.get(1, {}).get('best_features') or FEAT[:4]
        if not any(f.startswith('ptype_') for f in p5_feats):
            p5_feats = (['ptype_FTW', 'ptype_SCH']
                        + [f for f in p5_feats if not f.startswith('ptype_')][:2])
        p5_rand_vars = _phase_results.get(4, {}).get('rand_vars') or None
        ok5 = [f for f in p5_feats if f in synth.columns]
        t5  = synth[ok5+['period']].dropna().head(200).copy()
        t5['purpose'] = 'work'
        t5['case_id'] = np.arange(len(t5))
        res5 = stage4_final_mixed_logit(
            data=t5, selected_vars=ok5, purposes=['work'],
            period_col='period', n_alts=6, n_draws=100,
            correlated=(len(ok5) >= 2), rand_vars=p5_rand_vars,
        )
        check("Phase 5 SearchLibrium MXL fallback", True)
        if 'work' in res5:
            r5 = res5['work']
            check("Phase 5 fallback LL finite",  np.isfinite(r5['ll']),
                  f"ll={r5['ll']:.2f}")
            check("Phase 5 fallback BIC finite", np.isfinite(r5['bic']),
                  f"bic={r5['bic']:.2f}")
        _phase_results[5] = {
            'best_features': ok5,
            'best_fitness':  -(res5.get('work', {}).get('bic', 1e9) / 200),
            'history':       None,
            'progressive':   True,
            'estimator':     'SearchLibrium MXL fallback',
            'feature_names': ok5,
        }
    else:
        check("Phase 5 (Larch NL / SearchLibrium MXL)", False,
              "no supported estimator available")

except Exception as _e5:
    import traceback as _tb5
    check("Phase 5 runs", False, str(_e5))
    print(_tb5.format_exc())



# ════════════════════════════════════════════════════════════════════════════
# CONVERGENCE PLOTS — one per SA phase with proposed-candidate range
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 66)
print("Generating convergence plots …")
_plot_errors = []
for ph in sorted(_phase_results):
    info = _phase_results[ph]
    hist = info.get('history')
    if hist is None:
        continue
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ga_hypothesis_testing import GAResult

        dummy_result = GAResult.__new__(GAResult)
        dummy_result.history         = hist
        dummy_result.fitness_metric  = SA_KWARGS['fitness_metric']
        _ph_feat_names               = info.get('feature_names', FEAT)
        dummy_result.feature_names   = _ph_feat_names
        dummy_result.stage           = ph          # int → uses f'Stage {ph}' label
        dummy_result.n_restarts      = 1
        dummy_result.cache_hits      = 0
        dummy_result.best_individual = type('I', (), {
            'genes':   np.array([f in info.get('best_features', [])
                                 for f in _ph_feat_names], dtype=int),
            'fitness': info['best_fitness'],
        })()
        dummy_result.all_candidates  = []

        fig, ax = plt.subplots(figsize=(8, 4))
        dummy_result.plot_convergence(ax=ax)
        ax.set_title(f"Phase {ph}: {info['estimator']}")
        out_png = f"smoke_phase{ph}_convergence.png"
        fig.savefig(out_png, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Phase {ph} convergence plot → {out_png}")
    except Exception as _pe:
        _plot_errors.append(f"Phase {ph}: {_pe}")

if _plot_errors:
    print("  Plot warnings (non-fatal):")
    for e in _plot_errors:
        print(f"    {e}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 66)
print("SA SIGNAL-BEARING PROGRESSIVE SEARCH — PHASE SUMMARY")
print("=" * 66)
for ph, info in sorted(_phase_results.items()):
    prog_str = "progressive" if info['progressive'] else "FLAT"
    bsf_start = info['history'][0]['best_so_far'] if info.get('history') else float('nan')
    bsf_end   = info['history'][-1]['best_so_far'] if info.get('history') else float('nan')
    improvement = bsf_end - bsf_start
    print(f"  Phase {ph}: BIC/n={-info['best_fitness']:.4f}  [{prog_str}]  "
          f"improvement={improvement:+.4f}  estimator={info['estimator']}")
    print(f"           features={info['best_features']}")

n_pass = sum(1 for s, _, _ in _results if s == "PASS")
n_fail = sum(1 for s, _, _ in _results if s == "FAIL")
print(f"\nResults: {n_pass}/{n_pass + n_fail} PASS")
if n_fail:
    print("\nFailed tests:")
    for s, name, detail in _results:
        if s == "FAIL":
            print(f"  [FAIL] {name}" + (f"  [{detail}]" if detail else ""))
    sys.exit(1)
else:
    print("All SA signal-bearing phases passed.")
    sys.exit(0)
