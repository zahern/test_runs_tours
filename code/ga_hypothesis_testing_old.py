"""
ga_hypothesis_testing.py
=========================
Genetic Algorithm (GA) variable selection across all 6 ABM pipeline stages.

For each stage the GA evolves a population of "chromosomes" — binary masks
over the candidate feature set — and selects for best model fit.

Fitness metrics by stage
-------------------------
    Stage 1  (ordered logit)    : McFadden rho²  (higher = better)
    Stage 2  (OLS / lognormal)  : adjusted R²    (higher = better)
    Stage 3  (lognormal)        : AIC             (lower  = better, negated)
    Stage 4  (MNL timing)       : McFadden rho²  (higher = better)
    Stage 5  (MNL destination)  : McFadden rho²  (higher = better)
    Stage 6  (nested logit)     : McFadden rho²  (higher = better)

GA parameters (configurable per stage)
---------------------------------------
    pop_size        : population size (default 30)
    n_generations   : number of generations (default 40)
    mutation_rate   : probability of flipping each gene (default 0.1)
    crossover_rate  : probability of crossover (default 0.7)
    elitism_k       : top-k individuals carried to next generation (default 3)
    min_features    : minimum features that must be active (default 2)
    tournament_k    : tournament selection size (default 3)

Usage — single stage
---------------------
    from ga_hypothesis_testing import run_ga_stage

    out    = Synth_Connect().process_survey(save=False)
    result = run_ga_stage(
        stage      = 1,
        tours      = out["tours"],
        persons    = out["persons"],
        pop_size   = 30,
        n_generations = 40,
    )
    print(result.best_features)
    print(result.best_fitness)
    result.plot_convergence()

Usage — all stages
-------------------
    from ga_hypothesis_testing import run_ga_all_stages

    out     = Synth_Connect().process_survey(save=False)
    skims   = pd.read_csv('OD_skims.csv')
    landuse = sc.get_land()

    results = run_ga_all_stages(
        tours=out["tours"], trips=out["trips"],
        persons=out["persons"], skims=skims, landuse=landuse,
    )
    results.summary()          # print table of best features per stage
    results.save('saved_data/ga_results.pkl')
"""

import numpy as np
import pandas as pd
import logging
import pickle
import os
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0.  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One chromosome = one feature mask."""
    genes:      np.ndarray          # binary array, length = n_features
    fitness:    float = -np.inf
    metadata:   dict  = field(default_factory=dict)

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
    best_features:   List[str]
    best_fitness:    float
    best_individual: Individual
    history:         List[dict]          # per-generation stats
    all_candidates:  List[Individual]    # final population
    feature_names:   List[str]
    fitness_metric:  str

    def plot_convergence(self, ax=None, save_path: Optional[str] = None):
        """Plot best and mean fitness over generations."""
        import matplotlib.pyplot as plt
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')

        gens  = [h['generation'] for h in self.history]
        best  = [h['best_fitness']  for h in self.history]
        mean  = [h['mean_fitness']  for h in self.history]
        worst = [h['worst_fitness'] for h in self.history]

        ax.fill_between(gens, worst, best, alpha=0.12, color='#534AB7')
        ax.plot(gens, mean, color='#888780', linewidth=1.2,
                linestyle='--', label='mean')
        ax.plot(gens, best, color='#534AB7', linewidth=2,
                label=f'best ({self.fitness_metric})')

        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel(self.fitness_metric, fontsize=10)
        ax.set_title(f'Stage {self.stage} GA convergence', fontsize=11,
                     fontweight='normal')
        ax.legend(fontsize=9, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(color='#e8e8e3', linewidth=0.5)

        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()

    def feature_importance(self) -> pd.DataFrame:
        """
        Frequency each feature appeared in top-20% of population.
        Higher = more consistently selected by the GA.
        """
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
        print(f"\nStage {self.stage} GA result")
        print(f"  Fitness metric : {self.fitness_metric}")
        print(f"  Best fitness   : {self.best_fitness:.4f}")
        print(f"  Best features  : {self.best_features}")
        print(f"  Generations    : {len(self.history)}")


@dataclass
class AllStagesResult:
    """Container for GA results across all stages."""
    results: Dict[int, GAResult] = field(default_factory=dict)

    def summary(self):
        print("\n" + "="*60)
        print("GA HYPOTHESIS TESTING — ALL STAGES")
        print("="*60)
        for stage, res in sorted(self.results.items()):
            print(f"\nStage {stage}  |  {res.fitness_metric} = {res.best_fitness:.4f}")
            print(f"  Selected: {res.best_features}")

    def plot_all(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        n = len(self.results)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(12, 4 * rows),
                                 facecolor='white')
        axes = np.array(axes).flatten()
        for i, (stage, res) in enumerate(sorted(self.results.items())):
            res.plot_convergence(ax=axes[i])
        for j in range(len(self.results), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def save(self, path: str = 'saved_data/ga_results.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"GA results saved → {path}")

    @staticmethod
    def load(path: str) -> 'AllStagesResult':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 1.  CORE GA ENGINE
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Binary-encoded GA for feature selection.

    Each gene = 1 (feature included) or 0 (excluded).
    Fitness function is supplied externally per stage.
    """

    def __init__(
        self,
        n_features:     int,
        fitness_fn:     Callable[[np.ndarray], float],
        pop_size:       int   = 30,
        n_generations:  int   = 40,
        mutation_rate:  float = 0.10,
        crossover_rate: float = 0.70,
        elitism_k:      int   = 3,
        min_features:   int   = 2,
        tournament_k:   int   = 3,
        seed:           int   = 42,
        verbose:        bool  = True,
    ):
        self.n_features     = n_features
        self.fitness_fn     = fitness_fn
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.mutation_rate  = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_k      = elitism_k
        self.min_features   = min_features
        self.tournament_k   = tournament_k
        self.verbose        = verbose
        self.rng            = np.random.default_rng(seed)

    # ── initialisation ────────────────────────────────────────────────

    def _init_population(self) -> List[Individual]:
        pop = []
        for _ in range(self.pop_size):
            # ensure at least min_features active
            genes = np.zeros(self.n_features, dtype=int)
            active = self.rng.choice(
                self.n_features,
                size=max(self.min_features,
                         self.rng.integers(self.min_features,
                                           self.n_features + 1)),
                replace=False,
            )
            genes[active] = 1
            pop.append(Individual(genes=genes))
        return pop

    # ── selection ─────────────────────────────────────────────────────

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        competitors = self.rng.choice(len(pop),
                                      size=self.tournament_k,
                                      replace=False)
        best = max(competitors, key=lambda i: pop[i].fitness)
        return pop[best].copy()

    # ── crossover ─────────────────────────────────────────────────────

    def _crossover(self, p1: Individual,
                   p2: Individual) -> tuple:
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        # uniform crossover
        mask  = self.rng.integers(0, 2, size=self.n_features).astype(bool)
        g1    = np.where(mask, p1.genes, p2.genes)
        g2    = np.where(mask, p2.genes, p1.genes)
        return Individual(genes=g1), Individual(genes=g2)

    # ── mutation ──────────────────────────────────────────────────────

    def _mutate(self, ind: Individual) -> Individual:
        genes = ind.genes.copy()
        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                genes[i] = 1 - genes[i]
        # repair: enforce minimum features
        if genes.sum() < self.min_features:
            inactive = np.where(genes == 0)[0]
            extra = self.rng.choice(
                inactive,
                size=self.min_features - genes.sum(),
                replace=False,
            )
            genes[extra] = 1
        return Individual(genes=genes)

    # ── evaluation ────────────────────────────────────────────────────

    def _evaluate(self, pop: List[Individual]) -> List[Individual]:
        for ind in pop:
            if ind.fitness == -np.inf:   # not yet evaluated
                try:
                    ind.fitness = self.fitness_fn(ind.genes)
                except Exception as e:
                    logger.warning(f"fitness_fn error: {e}")
                    ind.fitness = -1e9
        return pop

    # ── main loop ─────────────────────────────────────────────────────

    def run(self) -> tuple:
        pop     = self._init_population()
        pop     = self._evaluate(pop)
        history = []

        best_ever = max(pop, key=lambda x: x.fitness).copy()

        for gen in range(self.n_generations):
            fitnesses = [ind.fitness for ind in pop]
            gen_best  = max(pop, key=lambda x: x.fitness)

            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best.copy()

            history.append({
                'generation':   gen,
                'best_fitness': gen_best.fitness,
                'mean_fitness': float(np.mean(fitnesses)),
                'worst_fitness':float(np.min(fitnesses)),
                'n_features':   int(gen_best.genes.sum()),
            })

            if self.verbose and gen % 5 == 0:
                print(f"  Gen {gen:3d} | best={gen_best.fitness:.4f} "
                      f"mean={np.mean(fitnesses):.4f} "
                      f"n_feat={int(gen_best.genes.sum())}")

            # elitism — carry best individuals forward
            elites = sorted(pop, key=lambda x: x.fitness,
                            reverse=True)[:self.elitism_k]

            # generate new population
            new_pop = [e.copy() for e in elites]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop)
                p2 = self._tournament_select(pop)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            pop = self._evaluate(new_pop)

        return best_ever, history, pop


# ---------------------------------------------------------------------------
# 2.  FITNESS FUNCTIONS PER STAGE
# ---------------------------------------------------------------------------

# ── Stage 1: ordered logit — McFadden rho² ───────────────────────────────

def _stage1_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
    ptype: str,
) -> float:
    from stage1_activity_frequency import OrderedLogit
    from scipy.special import logit as logit_fn

    feats   = [f for f, g in zip(feature_names, genes) if g == 1]
    col     = f'n_{purpose}_out'
    subset  = data[(data['ptype'] == ptype) & data[col].notna()].copy()

    if len(subset) < 30 or col not in subset.columns:
        return -1e9

    y = subset[col].values.astype(int)
    if len(np.unique(y)) < 2:
        return -1e9

    X = subset[feats].fillna(0).values.astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # null LL
    classes = np.sort(np.unique(y))
    probs   = np.array([(y <= c).mean() for c in classes[:-1]])
    probs   = np.clip(probs, 0.01, 0.99)

    m = OrderedLogit()
    m.fit(X, y)

    # compute LL of fitted model
    from scipy.special import expit
    ll_fitted = 0.0
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
        ll_fitted += np.sum(np.log(p))

    ll_null = len(y) * np.log(1.0 / len(classes))
    rho2    = 1 - (ll_fitted / ll_null) if ll_null != 0 else 0.0
    # AIC penalty for number of parameters
    k    = len(feats) + len(classes) - 1
    aic  = -2 * ll_fitted + 2 * k
    return rho2 - 0.001 * k   # small parsimony penalty


# ── Stage 2: OLS — adjusted R² ───────────────────────────────────────────

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

    X = subset[feats].values.astype(float)
    y = subset[target_col].values.astype(float)

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs = np.column_stack([np.ones(len(Xs)), Xs])

    try:
        params = np.linalg.lstsq(Xs, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return -1e9

    yhat  = Xs @ params
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2    = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n, k  = len(y), len(feats)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
    return float(adj_r2)


# ── Stage 3: lognormal — AIC (negated so higher = better) ────────────────

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

    m = BudgetLognormal()
    m.fit(X, y)

    log_y  = np.log(np.clip(y, 1e-6, None))
    mu_    = m.mu_; sd_ = m.sd_.copy(); sd_[sd_==0]=1.0
    Xs     = (X - mu_) / sd_
    Xs_aug = np.column_stack([np.ones(len(Xs)), Xs])
    eta    = Xs_aug @ np.concatenate([[m.intercept_], m.coef_])
    resid  = log_y - eta
    sigma  = resid.std() + 1e-9
    ll     = -len(y)*np.log(sigma) - 0.5*((resid/sigma)**2).sum() - log_y.sum()

    k   = len(feats) + 2   # betas + intercept + sigma
    aic = -2 * ll + 2 * k
    return -aic   # negate so GA maximises


# ── Stage 4: MNL timing — McFadden rho² ──────────────────────────────────

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

    # build feature matrix for selected features
    subset['purpose']  = purpose   # ensure column exists
    feat_subset        = subset.copy()
    missing            = [f for f in feats if f not in feat_subset.columns]
    for col in missing:
        feat_subset[col] = 0.0

    m = MNLTimingModel()
    m.ptypes_ = []
    X = np.column_stack([
        feat_subset[f].fillna(0).values.astype(float) for f in feats
    ]) if feats else np.ones((len(subset), 1))

    mu, sd = X.mean(0), X.std(0)
    sd[sd==0] = 1.0
    m.mu_ = mu; m.sd_ = sd

    K = len(TIME_PERIODS)
    F = X.shape[1]
    Xs = (X - mu) / sd
    init = np.zeros((K-1)*F)

    from scipy.optimize import minimize
    res = minimize(m._neg_ll, init, args=(Xs, y),
                   jac=m._neg_ll_grad, method='L-BFGS-B',
                   options={'maxiter':150,'ftol':1e-7})

    ll_fitted = -res.fun
    ll_null   = len(y) * np.log(1.0 / K)
    rho2      = 1 - ll_fitted / ll_null if ll_null != 0 else 0.0
    k_params  = (K-1) * F
    return rho2 - 0.0005 * k_params


# ── Stage 5: Larch mode+destination — log-likelihood ─────────────────────
#
# The GA searches over WHICH variables to include in the Larch utility specs.
# Candidate variables are split into two pools:
#   MODE_CANDIDATES  — skim vars that can appear in mode utility
#   DEST_CANDIDATES  — skim + landuse vars for destination utility
#
# For each chromosome (binary mask), we build restricted specs, estimate
# both models on HHTS, and return the sum of log-likelihoods as fitness.
# This replaces the old numpy MNL rho² approximation.

def _stage5_fitness_larch(
    mode_genes: np.ndarray,
    dest_genes: np.ndarray,
    mode_candidates: List[str],
    dest_candidates: List[str],
    segment: str,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skims,
) -> float:
    """
    Fitness = mode_LL + dest_LL for the given variable subsets.
    Lower (more negative) LL = worse. Returns sum (higher = better).
    """
    from stage5_destination_choice import (
        _filter_seg, _prep_hhts_tours,
        _build_mode_model, _build_dest_model,
        _estimate, _logsums, _mode_tree, _dest_tree,
        _mode_spec_custom, _dest_spec_custom,
    )

    mode_vars = [v for v, g in zip(mode_candidates, mode_genes) if g == 1]
    dest_vars = [v for v, g in zip(dest_candidates, dest_genes) if g == 1]

    if len(mode_vars) < 1 or len(dest_vars) < 1:
        return -1e9

    seg_tours = _filter_seg(tours, segment)
    if len(seg_tours) < 50:
        return -1e9

    try:
        dt_m  = _mode_tree(seg_tours, persons, landuse, skims)
        mode_mdl = _build_mode_model(dt_m, segment,
                                     custom_mode_vars=mode_vars)
        mode_mdl = _estimate(mode_mdl)
        mode_ll  = float(mode_mdl.loglike())

        ls    = _logsums(mode_mdl, dt_m, verbose=False)
        dt_d  = _dest_tree(seg_tours, persons, landuse, skims, ls)
        dest_mdl = _build_dest_model(dt_d, segment,
                                     custom_dest_vars=dest_vars)
        dest_mdl = _estimate(dest_mdl)
        dest_ll  = float(dest_mdl.loglike())

        # penalise complexity (BIC-style penalty per active variable)
        n_params = mode_genes.sum() + dest_genes.sum()
        penalty  = 0.5 * n_params * np.log(max(len(seg_tours), 2))

        return mode_ll + dest_ll - penalty

    except Exception as e:
        logger.debug(f"Stage 5 Larch fitness failed ({segment}): {e}")
        return -1e9


def _stage5_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
    skim_lookup: dict,
    lu_df: pd.DataFrame,
    lu_size_cols: dict,
) -> float:
    """Legacy fallback — only used if Larch is unavailable."""
    return -1e9  # replaced by _stage5_fitness_larch


# ── Stage 6: nested logit — McFadden rho² ────────────────────────────────

def _stage6_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
) -> float:
    from stage6_mode_choice import NestedLogitModeModel, MODES, MODE_IDX

    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
    subset = data[data['purpose'] == purpose].copy()
    if len(subset) < 30 or subset['chosen_mode'].nunique() < 2:
        return -1e9

    m = NestedLogitModeModel()

    # ensure columns exist
    for col in feats:
        if col not in subset.columns:
            subset[col] = 0.0

    try:
        m.fit(subset, purpose)
    except Exception:
        return -1e9

    return float(np.clip(m.rho2_, -1, 1)) if m.rho2_ is not None else -1e9


# ---------------------------------------------------------------------------
# 3.  CANDIDATE FEATURE SETS PER STAGE
# ---------------------------------------------------------------------------

def _stage1_candidates(data: pd.DataFrame) -> List[str]:
    base  = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp']
    extra = ['rlhp', 'AGEGROUP', 'bikes', 'WFHWEEK']
    # accessibility columns — present if run_accessibility() was called
    acc   = ['auto_work_acc', 'auto_shop_acc', 'walk_acc', 'transit_acc']
    ptype_dummies = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in base + extra + acc + ptype_dummies if c in data.columns]


def _stage2_candidates(data: pd.DataFrame) -> List[str]:
    base = [
        'n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
        'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
        'n_total', 'n_tours',
        'sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp',
    ]
    # accessibility — people in accessible zones have longer days (longer B)
    acc  = ['auto_work_acc', 'auto_nonmand_acc', 'walk_acc']
    ptype_dummies = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in base + acc + ptype_dummies if c in data.columns]


def _stage3_candidates(data: pd.DataFrame) -> List[str]:
    base = [
        'start_hr', 'log_dist', 'B',
        'n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
        'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
        'sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp',
    ]
    ptype_dummies = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in base + ptype_dummies if c in data.columns]


def _stage4_candidates(data: pd.DataFrame) -> List[str]:
    base = [
        'day_position', 'time_consumed', 'duration_min', 'time_remaining',
        'sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING',
    ]
    purp_dummies  = [c for c in data.columns if c.startswith('purp_')]
    ptype_dummies = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in base + purp_dummies + ptype_dummies if c in data.columns]


def _stage5_mode_candidates() -> List[str]:
    """Skim variables eligible for inclusion in the mode utility."""
    return [
        'SOVTOLL_TIME__AM',   # auto in-vehicle time AM peak
        'SOVTOLL_TIME__PM',   # auto in-vehicle time PM peak
        'DISTWALK',           # walk distance
        'DISTBIKE',           # bike distance
        'CRASH',              # crash risk index
        'DIST',               # total distance
    ]


def _stage5_dest_candidates() -> List[str]:
    """Skim + landuse variables eligible for destination utility."""
    return [
        # skim deterrents
        'DIST', 'CRASH',
        # employment attractors — work
        'FPSEMPN', 'HEREMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP', 'RETEMPN',
        'TOTEMP',
        # education attractors
        'Education', 'Primary', 'Secondary',
        'education',
        # non-mandatory attractors
        'Shopping', 'ESCORT', 'Entertainment',
        'OTHERMAINTENANCE', 'OTHERDISCRFacilities', 'EATOUT',
        # area-type attractors
        'industrial', 'commercial', 'residential', 'retail',
        'construction',
        # accessibility
        'TOTPOP', 'TOTHH',
    ]


def _stage5_candidates() -> List[str]:
    """Combined list for backwards compatibility."""
    return _stage5_mode_candidates() + _stage5_dest_candidates()


def _stage6_candidates(data: pd.DataFrame) -> List[str]:
    return [
        'tt_avg', 'tt_am', 'log_dist', 'dist',
        'distwalk', 'distbike', 'crash',
        'cost_proxy', 'has_vehicle', 'no_vehicle',
        'accessibility', 'sexp', 'vehrd', 'incp',
        'start_hr',
    ]


# ---------------------------------------------------------------------------
# 4.  PER-STAGE GA RUNNERS
# ---------------------------------------------------------------------------

def _run_ga_stage1(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    ga_kwargs: dict,
) -> GAResult:
    from stage1_activity_frequency import (
        prepare_frequency_data, PURPOSES, assign_ptype_column
    )
    print("\n[Stage 1] Preparing data...")
    data      = prepare_frequency_data(tours, persons)
    ptypes    = data['ptype'].dropna().unique()
    feat_names = _stage1_candidates(data)

    # run GA over the most common (purpose, ptype) pair — largest sample
    # you can extend to iterate over all pairs
    best_results = {}
    for purpose in PURPOSES:
        for ptype in ptypes:
            subset = data[(data['ptype'] == ptype)]
            col    = f'n_{purpose}_out'
            if col not in subset.columns or len(subset) < 50:
                continue

            print(f"\n  GA: {purpose} / {ptype}  (n={len(subset)})")

            def fitness(genes):
                return _stage1_fitness(genes, data, feat_names, purpose, ptype)

            ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
            best, history, pop = ga.run()
            best_results[(purpose, ptype)] = (best, history, pop)

    # aggregate: pick the (purpose, ptype) with best fitness
    if not best_results:
        raise RuntimeError("Stage 1 GA: no valid (purpose, ptype) pairs found.")

    best_key  = max(best_results, key=lambda k: best_results[k][0].fitness)
    best, history, pop = best_results[best_key]
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=1, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='rho²',
    )


def _run_ga_stage2(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    ga_kwargs: dict,
) -> GAResult:
    from stage2_day_duration import aggregate_person_day
    print("\n[Stage 2] Preparing data...")
    data      = aggregate_person_day(tours, persons)
    feat_names = _stage2_candidates(data)

    results = {}
    for target in ['start_of_day', 'end_of_day', 'B']:
        print(f"\n  GA: target={target}  (n={len(data)})")

        def fitness(genes, t=target):
            return _stage2_fitness(genes, data, feat_names, t)

        ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
        best, history, pop = ga.run()
        results[target] = (best, history, pop)

    best_key  = max(results, key=lambda k: results[k][0].fitness)
    best, history, pop = results[best_key]
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=2, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='adj-R²',
    )


def _run_ga_stage3(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    ga_kwargs: dict,
    trips: Optional[pd.DataFrame] = None,
) -> GAResult:
    from stage3_activity_duration import prepare_duration_data, PURPOSES
    print("\n[Stage 3] Preparing data...")

    # Pass trips so prepare_duration_data gets individual activity rows
    # for all purposes (eat, social, othdiscr etc.) not just tour primaries.
    data = prepare_duration_data(tours, persons, trips=trips)

    # purpose column is 'main_purpose' after prepare_duration_data
    purp_col = next(
        (c for c in ['main_purpose', '_purpose_coded', 'purpose']
         if c in data.columns),
        None
    )
    if purp_col is None:
        raise RuntimeError("Stage 3 GA: no purpose column in prepared data.")

    print(f"  Purpose column: '{purp_col}'  "
          f"Values: {data[purp_col].dropna().unique().tolist()}")

    feat_names = _stage3_candidates(data)
    print(f"  Features: {feat_names}")

    results = {}
    for purpose in PURPOSES:
        subset = data[data[purp_col] == purpose]
        if len(subset) < 30:   # relaxed from 50 — eat/othdiscr may be sparse
            print(f"  Skipping {purpose}: only {len(subset)} obs")
            continue
        print(f"\n  GA: {purpose}  (n={len(subset)})")

        def fitness(genes, p=purpose, pc=purp_col):
            return _stage3_fitness(genes, data, feat_names, p, purpose_col=pc)

        ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
        best, history, pop = ga.run()
        results[purpose] = (best, history, pop)

    if not results:
        # print diagnostics before raising
        print(f"  Data shape: {data.shape}")
        print(f"  Purpose counts:\n{data[purp_col].value_counts()}")
        raise RuntimeError(
            "Stage 3 GA: no valid purposes found. "
            "Check that trips is passed and DESTPURP/purpose column exists."
        )

    best_key  = max(results, key=lambda k: results[k][0].fitness)
    best, history, pop = results[best_key]
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=3, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='-AIC',
    )


def _run_ga_stage4(
    trips: pd.DataFrame,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    ga_kwargs: dict,
) -> GAResult:
    from stage4_schedule_placement import prepare_timing_data, PURPOSES
    print("\n[Stage 4] Preparing data...")
    data      = prepare_timing_data(trips, tours, persons)
    feat_names = _stage4_candidates(data)

    results = {}
    for purpose in PURPOSES:
        subset = data[data['purpose'] == purpose]
        if len(subset) < 50:
            continue
        print(f"\n  GA: {purpose}  (n={len(subset)})")

        def fitness(genes, p=purpose):
            return _stage4_fitness(genes, data, feat_names, p)

        ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
        best, history, pop = ga.run()
        results[purpose] = (best, history, pop)

    if not results:
        raise RuntimeError("Stage 4 GA: no valid purposes found.")

    best_key  = max(results, key=lambda k: results[k][0].fitness)
    best, history, pop = results[best_key]
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=4, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='rho²',
    )


def _run_ga_stage5(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    skims,
    landuse: pd.DataFrame,
    ga_kwargs: dict,
) -> GAResult:
    """
    GA for Stage 5 — searches over which skim and landuse variables
    to include in the Larch mode and destination utility functions.

    Runs one GA per segment (work, edu, nonmand) using a JOINT chromosome:
        [mode_genes | dest_genes]
    where mode_genes selects from _stage5_mode_candidates() and
    dest_genes selects from _stage5_dest_candidates().

    Fitness = mode_LL + dest_LL (Larch log-likelihood, higher = better).
    """
    from stage5_destination_choice import _prep_hhts_tours

    print("\n[Stage 5] Preparing HHTS tours...")
    tours = _prep_hhts_tours(trips, persons)

    mode_cands = _stage5_mode_candidates()
    dest_cands = _stage5_dest_candidates()
    n_mode     = len(mode_cands)
    n_dest     = len(dest_cands)
    n_total    = n_mode + n_dest
    feat_names = mode_cands + dest_cands

    segments   = ['work', 'edu', 'nonmand']
    results    = {}

    # open skims once
    import openmatrix as omx
    close_skims = False
    if isinstance(skims, str):
        skims = omx.open_file(skims, 'r')
        close_skims = True

    for seg in segments:
        from stage5_destination_choice import _filter_seg
        seg_tours = _filter_seg(tours, seg)
        if len(seg_tours) < 50:
            print(f"  Skipping {seg} — only {len(seg_tours)} tours")
            continue

        print(f"\n  GA: segment={seg}  (n={len(seg_tours)})")

        def fitness(genes, s=seg):
            return _stage5_fitness_larch(
                mode_genes      = genes[:n_mode],
                dest_genes      = genes[n_mode:],
                mode_candidates = mode_cands,
                dest_candidates = dest_cands,
                segment         = s,
                tours           = tours,
                persons         = persons,
                landuse         = landuse,
                skims           = skims,
            )

        ga = GeneticAlgorithm(n_total, fitness, **ga_kwargs)
        best, history, pop = ga.run()
        results[seg] = (best, history, pop)

    if close_skims:
        skims.close()

    if not results:
        raise RuntimeError("Stage 5 GA: no valid segments found.")

    best_key  = max(results, key=lambda k: results[k][0].fitness)
    best, history, pop = results[best_key]
    best_mode_feats = [f for f, g in zip(mode_cands, best.genes[:n_mode]) if g == 1]
    best_dest_feats = [f for f, g in zip(dest_cands, best.genes[n_mode:]) if g == 1]
    best_feats      = best_mode_feats + best_dest_feats

    print(f"\n  Best mode vars: {best_mode_feats}")
    print(f"  Best dest vars: {best_dest_feats}")
    print(f"  Best fitness: {best.fitness:.4f}")

    return GAResult(
        stage=5,
        best_features={'mode': best_mode_feats, 'dest': best_dest_feats},
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='loglikelihood',
    )


def _run_ga_stage6(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: Optional[pd.DataFrame],
    ga_kwargs: dict,
) -> GAResult:
    from stage6_mode_choice import prepare_mode_data, PURPOSES
    print("\n[Stage 6] Preparing data...")
    data      = prepare_mode_data(trips, persons, landuse)
    feat_names = _stage6_candidates(data)

    results = {}
    for purpose in PURPOSES:
        subset = data[data['purpose'] == purpose]
        if len(subset) < 50 or subset['chosen_mode'].nunique() < 2:
            continue
        print(f"\n  GA: {purpose}  (n={len(subset)})")

        def fitness(genes, p=purpose):
            return _stage6_fitness(genes, data, feat_names, p)

        ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
        best, history, pop = ga.run()
        results[purpose] = (best, history, pop)

    if not results:
        raise RuntimeError("Stage 6 GA: no valid purposes found.")

    best_key  = max(results, key=lambda k: results[k][0].fitness)
    best, history, pop = results[best_key]
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=6, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='rho²',
    )


# ---------------------------------------------------------------------------
# 5.  PUBLIC API
# ---------------------------------------------------------------------------

def run_ga_stage(
    stage: int,
    tours:   Optional[pd.DataFrame] = None,
    trips:   Optional[pd.DataFrame] = None,
    persons: Optional[pd.DataFrame] = None,
    skims:   Optional[pd.DataFrame] = None,
    landuse: Optional[pd.DataFrame] = None,
    pop_size:      int   = 30,
    n_generations: int   = 40,
    mutation_rate: float = 0.10,
    crossover_rate:float = 0.70,
    elitism_k:     int   = 3,
    min_features:  int   = 2,
    tournament_k:  int   = 3,
    seed:          int   = 42,
    verbose:       bool  = True,
) -> GAResult:
    """
    Run GA variable selection for a single pipeline stage.

    Parameters
    ----------
    stage         : 1–6
    tours, trips, persons, skims, landuse : DataFrames from process_survey()
    pop_size      : GA population size
    n_generations : number of GA generations
    mutation_rate : per-gene mutation probability
    crossover_rate: crossover probability
    elitism_k     : elites carried forward each generation
    min_features  : minimum features that must be active
    tournament_k  : tournament selection size
    seed          : random seed
    verbose       : print progress

    Returns
    -------
    GAResult with best_features, best_fitness, history, plot_convergence()

    Example
    -------
        result = run_ga_stage(
            stage=1, tours=out["tours"], persons=out["persons"],
            pop_size=30, n_generations=40,
        )
        print(result.best_features)
        result.plot_convergence()
        print(result.feature_importance())
    """
    logging.basicConfig(level=logging.WARNING)

    ga_kwargs = dict(
        pop_size=pop_size, n_generations=n_generations,
        mutation_rate=mutation_rate, crossover_rate=crossover_rate,
        elitism_k=elitism_k, min_features=min_features,
        tournament_k=tournament_k, seed=seed, verbose=verbose,
    )

    print(f"\n{'='*55}")
    print(f"GA VARIABLE SELECTION — STAGE {stage}")
    print(f"  pop={pop_size}  gens={n_generations}  mut={mutation_rate}")
    print(f"{'='*55}")

    dispatch = {
        1: lambda: _run_ga_stage1(tours, persons, ga_kwargs),
        2: lambda: _run_ga_stage2(tours, persons, ga_kwargs),
        3: lambda: _run_ga_stage3(tours, persons, ga_kwargs, trips=trips),
        4: lambda: _run_ga_stage4(trips, tours, persons, ga_kwargs),
        5: lambda: _run_ga_stage5(trips, persons, skims, landuse, ga_kwargs),
        6: lambda: _run_ga_stage6(trips, persons, landuse, ga_kwargs),
    }

    if stage not in dispatch:
        raise ValueError(f"stage must be 1–6, got {stage}")

    result = dispatch[stage]()
    result.print_summary()
    return result


def run_ga_all_stages(
    tours:   pd.DataFrame,
    trips:   pd.DataFrame,
    persons: pd.DataFrame,
    skims:   Optional[pd.DataFrame] = None,
    landuse: Optional[pd.DataFrame] = None,
    stages:  Optional[List[int]]    = None,
    pop_size:      int   = 30,
    n_generations: int   = 40,
    mutation_rate: float = 0.10,
    seed:          int   = 42,
    save_path:     Optional[str] = 'saved_data/ga_results.pkl',
) -> AllStagesResult:
    """
    Run GA variable selection across all (or specified) pipeline stages.

    Parameters
    ----------
    tours, trips, persons : from process_survey()
    skims, landuse        : required for stages 5 and 6
    stages                : list of stages to run, default [1,2,3,4,5,6]
                            (skip 5/6 if skims/landuse not provided)
    pop_size, n_generations, mutation_rate, seed : GA settings
    save_path             : save results to pickle after each stage

    Returns
    -------
    AllStagesResult with .summary(), .plot_all(), .save()

    Example
    -------
        sc      = Synth_Connect()
        out     = sc.process_survey(save=False)
        skims   = pd.read_csv('OD_skims.csv')
        landuse = sc.get_land()

        results = run_ga_all_stages(
            tours=out["tours"], trips=out["trips"],
            persons=out["persons"], skims=skims, landuse=landuse,
            pop_size=30, n_generations=40,
        )
        results.summary()
        results.plot_all(save_path='saved_data/plots/ga_convergence.png')
        results.save()
    """
    if stages is None:
        stages = [1, 2, 3, 4]
        if skims   is not None: stages.append(5)
        if landuse is not None: stages.append(6)

    all_results = AllStagesResult()

    for stage in stages:
        try:
            result = run_ga_stage(
                stage=stage,
                tours=tours, trips=trips, persons=persons,
                skims=skims, landuse=landuse,
                pop_size=pop_size, n_generations=n_generations,
                mutation_rate=mutation_rate, seed=seed,
            )
            all_results.results[stage] = result

            if save_path:
                all_results.save(save_path)

        except Exception as e:
            logger.error(f"Stage {stage} GA failed: {e}")
            print(f"  Stage {stage} skipped: {e}")

    all_results.summary()
    return all_results
