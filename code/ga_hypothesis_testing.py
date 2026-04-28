"""
ga_hypothesis_testing.py
=========================
Genetic Algorithm (GA) variable selection across all ABM pipeline stages.

Design principles
-----------------
1. GA is ESTIMATION ONLY — every fitness evaluation fits a model on HHTS
   and returns a log-likelihood / rho² / adj-R² score.  No prediction over
   the synthetic population is done during GA search.  Prediction runs ONCE
   at the end via apply_ga_results_to_pipeline().

2. Fitness CACHING — identical chromosomes (same binary gene pattern) are
   cached so they are never re-estimated. This removes the main redundancy
   in the original GA where crossover frequently produced duplicate individuals.

3. Adaptive mutation — when population diversity falls below a threshold
   (measured by mean hamming distance between individuals) the mutation rate
   temporarily increases to escape local optima.

4. Multi-restart — after the main run a configurable number of random-restart
   mini-runs (shorter n_generations) compete against the main result. This
   replaces the fragile "run again from scratch" pattern in the pipeline.

5. All-segment fitness aggregation — each stage's fitness is the mean over
   all valid (purpose, ptype) or (purpose, segment) combinations, not just
   the best-performing one.  This finds features that work broadly.

6. Collinearity repair — after mutation, at most ONE variable from each
   collinearity group (see STAGE_COLLINEAR_GROUPS) may be active.  This
   prevents the GA from simultaneously selecting near-redundant variables
   (e.g. DIST and SOVTOLL_TIME__AM; TOTEMP and its component sub-totals)
   which inflate the parameter count and bias the BIC penalty upward.

7. Trip-level models — Stage 5t (code 51) runs a separate GA for trip-level
   mode + destination choice.  Stage 5 (code 5) remains tour-level.
   This mirrors the ActivitySim distinction between tour_mode_choice and
   trip_mode_choice, giving independent estimated models for each level.

BIC minimisation — CONFIRMED CORRECT
--------------------------------------
    The GA maximises fitness = –BIC/n (always negative, closer to 0 = better).
    Maximising –BIC/n is IDENTICAL to minimising BIC.

    BIC = –2·LL + k·ln(n)   Higher BIC = worse model (more penalty for
    complexity than fit improvement).  Lower BIC = better model.

    –BIC/n values:  very negative → bad model (high BIC);
                    close to 0    → good model (low BIC).

    The GA's best_so_far curve in convergence plots is shown as BIC/n
    (positive, with inverted y-axis so UP = better = lower BIC).

Fitness metrics (all stages)
----------------------------
    All stages use  **–BIC/n**  (negative BIC normalised per observation).
    GA maximises fitness → minimises BIC.

    BIC = –2·LL + k·ln(n)    where  k = number of free parameters
                                     n = number of observations

    –BIC/n is always negative; values closer to 0 (less negative) are better.
    Convergence plots display BIC/n (positive, lower = better) for clarity.

    Stage 0  (CDAP MNL 3-alt)   : –BIC/n  (higher = better)
    Stage -1 (usual location)   : mean log-P (LL per obs, higher = better)
    Stage 1  (ordered logit)    : –BIC/n  (higher = better)
    Stage 2  (OLS lognormal LL) : –BIC/n  (higher = better)
    Stage 3  (lognormal)        : –BIC/n  (higher = better)
    Stage 4  (MNL timing)       : –BIC/n  (higher = better)
    Stage 5  (Larch NL, TOUR)   : –BIC/n  per-segment, averaged (TOUR-LEVEL)
    Stage 51 (Larch NL, TRIP)   : –BIC/n  per-segment, averaged (TRIP-LEVEL)
    Stage 6  (nested logit)     : –BIC/n  (higher = better)
"""

import numpy as np
import pandas as pd
import logging
import pickle
import os
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


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
    # ── Stage 0 (CDAP) ──────────────────────────────────────────────────
    0: [
        ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'WORKSTAT'],
        ['ptype_UniversityStudent', 'ptype_SchoolChild', 'STUDYING'],
        ['vehrd', 'has_vehicle'],
        ['incp', 'income_hi', 'income_lo', 'income_midplus'],
    ],
    # ── Stage 1 (activity frequency) ────────────────────────────────────
    1: [
        ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'WORKSTAT'],
        ['ptype_UniversityStudent', 'ptype_SchoolChild', 'STUDYING'],
        ['vehrd', 'has_vehicle'],
        ['incp', 'income_hi', 'income_lo', 'income_midplus'],
    ],
    # ── Stage 2 (day duration) ───────────────────────────────────────────
    2: [
        ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'WORKSTAT'],
        ['ptype_UniversityStudent', 'ptype_SchoolChild', 'STUDYING'],
        ['vehrd', 'has_vehicle'],
        ['incp', 'income_hi', 'income_lo', 'income_midplus'],
        ['n_total', 'n_tours'],   # n_total ≈ sum of purpose counts
    ],
    # ── Stage 3 (activity duration) ─────────────────────────────────────
    3: [
        ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'WORKSTAT'],
        ['ptype_UniversityStudent', 'ptype_SchoolChild', 'STUDYING'],
        ['incp', 'income_hi', 'income_lo', 'income_midplus'],
        ['log_dist', 'B'],   # highly correlated in many purpose segments
    ],
    # ── Stage 4 (schedule placement) ────────────────────────────────────
    4: [
        ['time_consumed', 'day_position'],   # both encode position in day
        ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'WORKSTAT'],
        ['ptype_UniversityStudent', 'ptype_SchoolChild', 'STUDYING'],
        ['incp', 'income_hi', 'income_lo', 'income_midplus'],
    ],
    # ── Stage 5 mode (TOUR and TRIP) ────────────────────────────────────
    '5_mode': [
        ['SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM', 'HOV2TOLL_TIME__AM'],
        ['DISTWALK', 'DIST'],   # walk dist collinear with total dist
        ['DISTBIKE', 'DIST'],   # bike dist collinear with total dist
        ['vehrd', 'has_vehicle', 'no_vehicle'],
        ['incp', 'income_hi', 'income_lo', 'income_midplus'],
        ['ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'WORKSTAT'],
        ['ptype_UniversityStudent', 'ptype_SchoolChild', 'STUDYING'],
        ['age_child', 'age_teen', 'ptype_SchoolChild'],
        ['is_female', 'sexp'],
    ],
    # ── Stage 5 destination (TOUR and TRIP) ─────────────────────────────
    '5_dest': [
        ['TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP'],
        ['Education', 'Primary', 'Secondary'],
        ['DIST', 'SOVTOLL_TIME__AM'],
        ['TOTPOP', 'TOTHH'],
    ],
    # ── Stage 6 (trip mode) ──────────────────────────────────────────────
    6: [
        ['tt_avg', 'tt_am'],
        ['log_dist', 'dist'],
        ['ldw', 'distwalk', 'log_dist'],
        ['ldb', 'distbike', 'log_dist'],
        ['has_vehicle', 'no_vehicle', 'vehrd'],
        ['inc_hi', 'inc_lo', 'incp'],
        ['older', 'age5p'],
        ['worker', 'WORKSTAT'],
        ['student', 'STUDYING'],
    ],
}


def _repair_collinearity(
    genes: np.ndarray,
    feature_names: List[str],
    collinear_groups: List[List[str]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Repair a chromosome so at most ONE gene per collinearity group is active.

    For each group with more than one active gene, randomly zero all but one.
    This is called after every mutation and during population initialisation.
    """
    genes = genes.copy()
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    for group in collinear_groups:
        active_idx = [feat_idx[f] for f in group
                      if f in feat_idx and genes[feat_idx[f]] == 1]
        if len(active_idx) > 1:
            keep = rng.choice(active_idx)
            for idx in active_idx:
                if idx != keep:
                    genes[idx] = 0
    return genes


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

        For other metrics (rho², LL, etc.): displayed as-is, higher = better.
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
        div         = [h.get('diversity', 0)                  for h in self.history]
        mut         = [h.get('mutation_rate', np.nan)         for h in self.history]

        # For BIC metrics: negate to display BIC/n (positive, lower = better).
        # The curve naturally decreases over generations — no axis inversion needed.
        is_bic = 'BIC' in self.fitness_metric
        if is_bic:
            best        = [-v for v in best]
            best_so_far = [-v for v in best_so_far]
            mean        = [-v for v in mean]
            worst       = [-v for v in worst]
            ylabel = 'BIC / n  (↓ lower = better)'
            fill_lo, fill_hi = worst, best_so_far   # worst=high BIC (top), best=low BIC (bottom)
        else:
            ylabel = self.fitness_metric
            fill_lo, fill_hi = worst, best_so_far

        # ── main convergence lines ─────────────────────────────────────────
        ax.fill_between(gens, fill_lo, fill_hi, alpha=0.10, color='#534AB7',
                        label='population range')
        ax.plot(gens, mean, color='#888780', linewidth=1.2,
                linestyle='--', label='population mean', zorder=3)
        ax.plot(gens, best, color='#A59FF0', linewidth=1.0,
                linestyle=':', label='generation best', zorder=4)
        best_label = 'best-so-far (BIC/n)' if is_bic else f'best-so-far ({self.fitness_metric})'
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

        # ── diversity / mutation on secondary axis ─────────────────────────
        ax2 = ax.twinx()
        ax2.plot(gens, div, color='#E07030', linewidth=1.0,
                 linestyle=':', alpha=0.7, label='diversity')
        if not np.all(np.isnan(mut)):
            ax2.plot(gens, mut, color='#2C8C69', linewidth=1.0,
                     linestyle='-.', alpha=0.8, label='mutation rate')
        ax2.set_ylabel('Diversity / mutation rate', fontsize=8, color='#E07030')
        ax2.tick_params(axis='y', colors='#E07030', labelsize=8)
        ax2.set_ylim(0, max(max(div) * 1.5, 0.5) if div else (0, 1))
        ax2.spines[['top']].set_visible(False)

        # Stage label and improvement annotation
        stage_labels = {
             0: 'CDAP', -1: 'Usual Loc',
             1: 'Stage 1 — Activity Freq',
             2: 'Stage 2 — Day Duration',
             3: 'Stage 3 — Activity Duration',
             4: 'Stage 4 — Schedule Placement',
             5: 'Stage 5 — Dest + Mode',
            51: 'Stage 5.1 — Trip Mode',
             6: 'Stage 6 — Mode Choice',
        }
        slabel = stage_labels.get(self.stage, f'Stage {self.stage}')
        if segment_label:
            slabel += f'  [{segment_label}]'
        improvement = (best_so_far[-1] - best_so_far[0]) if len(best_so_far) >= 2 else 0.0
        impr_sign   = '↓' if is_bic else '↑'
        ax.set_title(
            f'{slabel}\n'
            f'restarts={self.n_restarts}  cache_hits={self.cache_hits}  '
            f'Δ={impr_sign}{abs(improvement):.4f}',
            fontsize=9, fontweight='normal', pad=4,
        )
        ax.set_xlabel('Generation', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        # Restart boundary markers — vertical dashed lines at each restart start
        restart_marks = [h['generation'] for h in self.history
                         if h.get('is_restart_start', False)]
        for i, rg in enumerate(restart_marks):
            ax.axvline(rg, color='#CC5533', linewidth=1.0, linestyle='--',
                       alpha=0.55, zorder=2,
                       label='restart' if i == 0 else '_nolegend_')

        # For BIC plots anchor the y-axis at 0 so the scale is always clear
        if is_bic:
            ax.set_ylim(bottom=0)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        # BIC/n decreases as GA improves → converged end is at the bottom-right.
        # Place legend in upper right where there is room above the converged level.
        ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False,
                  loc='upper right',
                  ncol=2)
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

        for stage, res in self.results.items():
            hist = res.history_summary()
            if not hist.empty:
                hist.to_csv(os.path.join(out_dir, f'ga_stage{stage}_history.csv'), index=False)

    def summary(self):
        STAGE_LABELS = {
             0: 'CDAP (daily activity pattern)',
            -1: 'Usual location (decay + attractors)',
             1: 'Stage 1 — activity frequency',
             2: 'Stage 2 — day duration',
             3: 'Stage 3 — activity duration',
             4: 'Stage 4 — schedule placement',
             5: 'Stage 5 — destination + mode',
             6: 'Stage 6 — mode choice',
        }
        print("\n" + "="*60)
        print("GA HYPOTHESIS TESTING — ALL STAGES  (estimation only)")
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
        fig.suptitle('GA Variable Selection — All Stages', fontsize=13,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def save(self, path: str = 'saved_data/ga_results.pkl'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"GA results saved → {path}")

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
        mutation_rate:       float = 0.08,
        mutation_rate_max:   float = 0.25,    # boosted rate when diversity low
        diversity_threshold: float = 0.20,    # fraction below which mutation boosts
        crossover_rate:      float = 0.75,
        elitism_k:           int   = 4,
        min_features:        int   = 2,
        max_features:        int   = None,    # None = no upper limit
        tournament_k:        int   = 4,
        stagnation_limit:    int   = 12,      # gens without improvement → mutation boost
        seed:                int   = 42,
        verbose:             bool  = True,
        feature_names:       List[str] = None,    # required for collinearity repair
        collinear_groups:    List[List[str]] = None,  # from STAGE_COLLINEAR_GROUPS
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

        # fitness cache — shared across restarts
        self._cache: dict = {}
        self.cache_hits   = 0

    # ── initialisation ────────────────────────────────────────────────────

    def _random_individual(self) -> Individual:
        n_active = self.rng.integers(self.min_features, self.max_features + 1)
        genes    = np.zeros(self.n_features, dtype=int)
        active   = self.rng.choice(self.n_features, size=n_active, replace=False)
        genes[active] = 1
        if self.collinear_groups and self.feature_names:
            genes = _repair_collinearity(genes, self.feature_names,
                                         self.collinear_groups, self.rng)
        return Individual(genes=genes)

    def _init_population(self, seed_individuals: list = None) -> List[Individual]:
        """
        Initialise population with balanced sparsity + biased-toward-active seeding.
        If seed_individuals (elites from a previous run) are supplied they are
        included directly and the rest of the population is random.
        """
        pop = []
        if seed_individuals:
            for ind in seed_individuals[:self.elitism_k]:
                pop.append(ind.copy())

        while len(pop) < self.pop_size:
            pop.append(self._random_individual())
        return pop

    # ── diversity ─────────────────────────────────────────────────────────

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

    # ── selection ─────────────────────────────────────────────────────────

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        idx      = self.rng.choice(len(pop), size=self.tournament_k, replace=False)
        best_idx = max(idx, key=lambda i: pop[i].fitness)
        return pop[best_idx].copy()

    # ── crossover ─────────────────────────────────────────────────────────

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        # uniform crossover — each gene independently inherited from either parent
        mask = self.rng.integers(0, 2, size=self.n_features).astype(bool)
        g1   = np.where(mask, p1.genes, p2.genes)
        g2   = np.where(mask, p2.genes, p1.genes)
        return Individual(genes=g1), Individual(genes=g2)

    # ── mutation ──────────────────────────────────────────────────────────

    def _mutate(self, ind: Individual, rate: float) -> Individual:
        genes = ind.genes.copy()
        flip  = self.rng.random(self.n_features) < rate
        genes[flip] = 1 - genes[flip]

        # repair: enforce min/max features
        n_active = genes.sum()
        if n_active < self.min_features:
            inactive = np.where(genes == 0)[0]
            extra    = self.rng.choice(inactive,
                                       size=self.min_features - n_active,
                                       replace=False)
            genes[extra] = 1
        if n_active > self.max_features:
            active   = np.where(genes == 1)[0]
            remove   = self.rng.choice(active,
                                       size=n_active - self.max_features,
                                       replace=False)
            genes[remove] = 0

        # repair: collinearity — keep at most one per correlated group
        if self.collinear_groups and self.feature_names:
            genes = _repair_collinearity(genes, self.feature_names,
                                         self.collinear_groups, self.rng)

        return Individual(genes=genes)

    # ── evaluation with caching ───────────────────────────────────────────

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

    # ── main loop ─────────────────────────────────────────────────────────

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

        for gen in range(self.n_generations):
            fitnesses = [ind.fitness for ind in pop]
            gen_best  = max(pop, key=lambda x: x.fitness)
            div       = self._diversity(pop)

            if gen_best.fitness > best_ever.fitness:
                best_ever        = gen_best.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # adaptive mutation: boost when stagnating or diversity low
            if stagnation_count >= self.stagnation_limit or div < self.diversity_threshold:
                cur_rate = self.mutation_rate_max
            else:
                # linearly scale between base and max based on diversity
                t        = max(0.0, 1.0 - div / max(self.diversity_threshold, 1e-6))
                cur_rate = self.mutation_rate + t * (self.mutation_rate_max - self.mutation_rate)

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
                print(f"  Gen {gen:3d} | best={gen_best.fitness:.4f} "
                      f"mean={np.mean(fitnesses):.4f} "
                      f"n_feat={int(gen_best.genes.sum())} "
                      f"div={div:.2f} mut={cur_rate:.3f} "
                      f"stag={stagnation_count}")

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

        return best_ever, history, pop

    def run_with_restarts(
        self,
        n_restarts:       int  = 3,
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
                print(f"    ↑ New global best: {best_ever.fitness:.4f}")
            history.extend(hist_r)

        self.n_generations = orig_gens
        return best_ever, history, pop


# ---------------------------------------------------------------------------
# 2.  FITNESS FUNCTIONS  (estimation only — no prediction)
# ---------------------------------------------------------------------------

# ── Stage 1: ordered logit — McFadden rho², AVERAGED over all (purpose, ptype) ─

def _stage1_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
    ptype: str,
) -> float:
    from stage1_activity_frequency import OrderedLogit
    from scipy.special import expit

    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
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
    """Mean rho² across all (purpose, ptype) pairs with enough data."""
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


# ── Stage 2: OLS — adjusted R², averaged over all targets ────────────────

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


# ── Stage 3: lognormal — -AIC, averaged over all purposes ────────────────

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


# ── Stage 4: MNL timing — rho², averaged over all purposes ───────────────

def _stage4_fitness(
    genes: np.ndarray,
    data: pd.DataFrame,
    feature_names: List[str],
    purpose: str,
) -> float:
    from stage4_schedule_placement import MNLTimingModel, TIME_PERIODS
    from scipy.optimize import minimize as _minimize

    feats  = [f for f, g in zip(feature_names, genes) if g == 1]
    subset = data[data['purpose'] == purpose].copy()
    if len(subset) < 30:
        return -1e9

    y = subset['period'].values.astype(int)
    if len(np.unique(y)) < 2:
        return -1e9

    for col in feats:
        if col not in subset.columns:
            subset[col] = 0.0

    X  = np.column_stack([subset[f].fillna(0).values.astype(float)
                          for f in feats]) if feats else np.ones((len(subset), 1))
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    m = MNLTimingModel()
    m.ptypes_ = []
    m.mu_ = mu; m.sd_ = sd

    K    = len(TIME_PERIODS)
    init = np.zeros((K - 1) * X.shape[1])
    try:
        res  = _minimize(m._neg_ll, init, args=(Xs, y),
                         jac=m._neg_ll_grad, method='L-BFGS-B',
                         options={'maxiter': 150, 'ftol': 1e-7})
        ll_f = -res.fun
    except Exception:
        return -1e9

    n   = len(y)
    k   = (K - 1) * X.shape[1]                  # MNL utility coefficients
    bic = -2.0 * ll_f + k * np.log(max(n, 2))
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


# ── Stage 5 (tour-level): MODE-ONLY and DEST-ONLY fitness functions ──────
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
        mode_mdl = _build_mode_model(dt_m, segment, custom_mode_vars=mode_vars)
        mode_mdl = _estimate(mode_mdl)
        mode_ll  = float(mode_mdl.loglike())
        n   = len(seg_tours)
        k   = int(mode_genes.sum())
        bic = -2.0 * mode_ll + k * np.log(max(n, 2))
        return -bic / max(n, 1)
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
    The mode model is NOT re-estimated here — logsums are pre-computed once
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
        bic = -2.0 * dest_ll + k * np.log(max(n, 2))
        return -bic / max(n, 1)
    except Exception as e:
        logger.debug(f"Stage5 dest-only [{segment}]: {e}")
        return -1e9


# ── Stage 5: Larch — mode_LL + dest_LL on HHTS only ─────────────────────
# ESTIMATION ONLY — no prediction over synthetic population.
# Calls _run_segment which uses only HHTS tours.

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
    Estimation-only fitness: fit mode + destination model on HHTS and
    return mode_LL + dest_LL.  No prediction over synth population.

    Uses _run_segment which is already the HHTS-only estimation path.
    """
    from stage5_destination_choice import (
        _filter_seg, _prep_hhts_tours,
        _build_mode_model, _build_dest_model,
        _estimate, _logsums, _mode_tree, _dest_tree,
        _attach_size_term, _attach_acdc_terms,
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

    try:
        # Mode model on HHTS
        dt_m     = _mode_tree(seg_tours, persons, landuse, skims)
        mode_mdl = _build_mode_model(dt_m, segment, custom_mode_vars=mode_vars)
        mode_mdl = _estimate(mode_mdl)
        mode_ll  = float(mode_mdl.loglike())

        # Logsums for destination tree
        ls = _logsums(mode_mdl, dt_m, skims=skims, verbose=False)

        # Destination model on HHTS
        lu_st = _attach_size_term(landuse, segment)
        lu_st, seg_acdc = _attach_acdc_terms(
            lu_st, seg_tours,
            skim_df if skim_df is not None else pd.DataFrame(),
            segment
        )
        dt_d     = _dest_tree(seg_acdc, persons, lu_st, skims, ls)
        dest_mdl = _build_dest_model(dt_d, segment, custom_dest_vars=dest_vars)
        dest_mdl = _estimate(dest_mdl)
        dest_ll  = float(dest_mdl.loglike())

        # Proper BIC: –2·LL + k·ln(n), then normalise per observation
        n   = len(seg_tours)
        k   = int(mode_genes.sum() + dest_genes.sum())
        bic = -2.0 * (mode_ll + dest_ll) + k * np.log(max(n, 2))
        return -bic / max(n, 1)                  # -BIC/n, higher is better

    except Exception as e:
        logger.debug(f"Stage5 fitness [{segment}]: {e}")
        return -1e9


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


# ── Stage 51: TRIP-LEVEL mode + destination choice ────────────────────────
# Mirrors Stage 5 but operates on individual TRIP legs rather than tours.
# ActivitySim distinction: tour_mode_choice → trip_mode_choice (constrained
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

    # ── Trip-level mode model on HHTS ────────────────────────────────────
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

    # ── Trip-level destination model (simple logit on observed DTAZ) ─────
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

    # ── Combined BIC ─────────────────────────────────────────────────────
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


# ── Stage 6: nested logit — rho² ─────────────────────────────────────────

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


# ── CDAP: Larch MNL — mean rho² over H/N/M classification ───────────────
# Chromosome = binary mask over MNL_FEATURES candidate set.
# Fitness = McFadden rho² of the 3-alternative MNL on HHTS persons.
# Uses numba engine (same as estimate_cdap) to avoid JAX OOM.

def _cdap_fitness(
    genes:         np.ndarray,
    data:          pd.DataFrame,   # output of prepare_cdap_data()
    feature_names: List[str],
) -> float:
    """
    Estimate a CDAP MNL on HHTS with the selected features.
    Returns McFadden rho² (higher = better).
    Fitness is penalised by number of active features (parsimony).
    """
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
        for _, row in data.iterrows():
            ca_df.loc[(str(row['person_id']), int(row['cdap_obs'])), 'chosen'] = 1
        ds_ca = lx.Dataset.construct.from_idca(ca_df)

        co_df = data.set_index(data['person_id'].astype(str))[available].fillna(0).copy()
        co_df.index.name = 'person_id'
        ds_co = lx.Dataset.construct.from_idco(co_df)

        dt = lx.DataTree(base=ds_ca, person=ds_co,
                         relationships=('base.person_id @ person.person_id',))

        mnl = lx.Model(dt, compute_engine='numba')
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


# ── Usual Location: decay_b and attractor column selection ───────────────
# Chromosome is CONTINUOUS (not binary) — see _run_ga_usual_loc.
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
    weighted gravity model P(z|orig) ∝ att(z) × exp(-b×tt(orig,z)).

    Chromosome layout:
        genes[0]            : work_decay_b  (mapped to [0.01, 0.20])
        genes[1]            : school_decay_b (mapped to [0.01, 0.20])
        genes[2:2+n_work_att]        : binary — which work attractor cols to use
        genes[2+n_work_att:]         : binary — which edu attractor cols to use
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

    # ── work LL ────────────────────────────────────────────────────────
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

    # ── school LL ──────────────────────────────────────────────────────
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
    acc_cols  = ['auto_work_acc', 'walk_acc', 'transit_acc', 'auto_nonmand_acc']
    raw_synth = ['vehrd', 'nprd', 'incp', 'sexp', 'WORKSTAT', 'STUDYING',
                 'CARLICENCE', 'occp_group']
    abs_cols  = ['empp', 'occp', 'rntrd', 'mstp',  # ABS direct (census_bridge)
                 'mtwp',          # travel-to-work mode — strong CDAP predictor
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
    but the HHTS estimation data may not have — included only when present.
    """
    abs_cols = [
        'empp',           # employment status (ABS direct)
        'occp',           # occupation 1-digit (ABS direct)
        'rntrd',          # rent/mortgage status
        'mstp',           # marital status
        'mtwp',           # method of travel to work — mode preference proxy
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


def _stage1_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 1 (activity frequency / ordered logit) — full candidate set."""
    base    = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp',
               'CARLICENCE', 'occp_group', 'HHSIZE']
    extra   = ['rlhp', 'AGEGROUP', 'bikes', 'WFHWEEK', 'WFHDAY']
    profile = _profile_vars(data)
    abs_    = _abs_person_vars(data)
    ptype   = [c for c in data.columns if c.startswith('ptype_')]
    acc     = ['auto_work_acc', 'auto_shop_acc', 'walk_acc', 'transit_acc',
               'auto_nonmand_acc']
    return [c for c in base + extra + profile + abs_ + acc + ptype
            if c in data.columns]


def _stage2_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 2 (day duration) — activity counts + all person attrs."""
    counts  = ['n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
               'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
               'n_total', 'n_tours']
    base    = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp',
               'CARLICENCE', 'occp_group']
    profile = _profile_vars(data)
    abs_    = _abs_person_vars(data)
    acc     = ['auto_work_acc', 'auto_nonmand_acc', 'walk_acc', 'transit_acc']
    ptype   = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in counts + base + profile + abs_ + acc + ptype
            if c in data.columns]


def _stage3_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 3 (activity duration / lognormal) — activity context + person attrs."""
    act_vars = ['start_hr', 'log_dist', 'B']
    counts   = ['n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
                'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out']
    base     = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp',
                'CARLICENCE', 'occp_group']
    profile  = _profile_vars(data)
    abs_     = _abs_person_vars(data)
    ptype    = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in act_vars + counts + base + profile + abs_ + ptype
            if c in data.columns]


def _stage4_candidates(data: pd.DataFrame) -> List[str]:
    """Stage 4 (schedule placement / MNL timing) — timing context + person attrs."""
    timing  = ['day_position', 'time_consumed', 'duration_min', 'time_remaining']
    base    = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'CARLICENCE',
               'incp', 'occp_group']
    profile = _profile_vars(data)
    abs_    = _abs_person_vars(data)
    purp    = [c for c in data.columns if c.startswith('purp_')]
    ptype   = [c for c in data.columns if c.startswith('ptype_')]
    return [c for c in timing + base + profile + abs_ + purp + ptype
            if c in data.columns]


def _stage5_mode_candidates() -> List[str]:
    """
    Candidate variables for Stage 5 (tour-level) and Stage 51 (trip-level) mode choice.

    Split into two sub-pools:
    ALWAYS FIXED (backbone) — kept regardless of GA selection:
        DIST, SOVTOLL_TIME__AM, WLK_LOC_WLK_TOTIVT__AM, DISTWALK, DISTBIKE, CRASH

    TESTED BY GA (additional enrichments around the backbone):
        • Alternative time periods (PM, HOV)
        • Socio-demographic person attributes (income, age, gender, household,
          vehicle ownership, licence, person-type flags)
        • ABS census-derived attributes (mtwp, empp, n_workers_hh)

    ActivitySim tour_mode_choice uses: IVT × purpose, age × mode, hhsize × SR,
    vehicle availability, licence, gender.  Trip_mode_choice additionally uses
    position-in-tour flags (outbound/inbound).  Both are reproduced here.
    """
    # ── Skim variables (GA tests these enrichments) ──────────────────────
    skim_extra = [
        'SOVTOLL_TIME__PM',       # PM period IVT — peak shoulder
        'HOV2TOLL_TIME__AM',      # HOV2 IVT — for shared-ride estimation
        'WLK_LOC_WLK_XWAIT__AM',  # transit excess wait
    ]
    # ── Vehicle / licence (key mode-choice predictors) ───────────────────
    veh_vars = ['vehrd', 'has_vehicle', 'no_vehicle', 'CARLICENCE']
    # ── Income (affects car and transit share) ───────────────────────────
    inc_vars = ['incp', 'income_hi', 'income_lo', 'income_midplus']
    # ── Age / gender ──────────────────────────────────────────────────────
    age_gender = [
        'sexp', 'is_female',
        'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
    ]
    # ── Household composition ─────────────────────────────────────────────
    hh_vars = ['nprd', 'hh_child', 'hh_alone', 'hh_size_large', 'n_workers_hh']
    # ── Person type (mode captivity) ─────────────────────────────────────
    ptype_vars = [
        'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
        'ptype_UniversityStudent', 'ptype_WorkingStudent',
        'ptype_SchoolChild', 'ptype_Retired',
    ]
    # ── ABS census (method of travel to work, employment status) ─────────
    abs_vars = ['mtwp', 'empp', 'bikes', 'WFHWEEK', 'wfh']
    # ── Work/study status ─────────────────────────────────────────────────
    work_study = ['WORKSTAT', 'STUDYING', 'mandatory_role']

    return (skim_extra + veh_vars + inc_vars + age_gender +
            hh_vars + ptype_vars + abs_vars + work_study)


def _stage5_dest_candidates() -> List[str]:
    """
    Candidate variables for Stage 5 destination choice (tour and trip level).

    Split into sub-pools:
    ALWAYS FIXED (backbone) — kept regardless of GA selection:
        c_logsum, c_log_dist, c_dist, c_size_term, c_aggl, c_comp, c_crash,
        c_incp_dist, c_veh_dist, key accessibility interactions

    TESTED BY GA (additional enrichments):
        • Land-use employment categories by sector
        • Education enrollment variables
        • Non-mandatory attraction variables (retail, recreation)
        • Population and housing density
        • Person-by-destination interaction terms (income × employment,
          age × education, gender × recreation)
        • ABS demographic context at destination
        • Accessibility at destination

    ActivitySim destination choice (non_mandatory_tour_destination) uses:
    size terms (employment, enrollment, retail), distance, logsums,
    demographics at destination.  Same structure reproduced here.
    """
    # sku deterrents (beyond the fixed backbone)
    skim_extra = ['SOVTOLL_TIME__AM']
    # ── Employment by sector (work tours) ────────────────────────────────
    emp_work = [
        'FPSEMPN', 'HEREMPN', 'RETEMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP',
        'TOTEMP', 'commercial', 'industrial', 'government', 'governmental',
    ]
    # ── Education enrollment (edu tours) ─────────────────────────────────
    emp_edu = [
        'Education', 'Primary', 'Secondary', 'education',
        'HSENROLL', 'AGE0519',
        'Primary_total_priv', 'Secondary_total_priv', 'HEREMPN',
    ]
    # ── Non-mandatory attraction ──────────────────────────────────────────
    emp_nonmand = [
        'Shopping', 'ESCORT', 'Entertainment', 'OTHERMAINTENANCE',
        'OTHERDISCRFacilities', 'EATOUT', 'OTHEMP',
        'recreation_ground', 'sport', 'park', 'village_green',
        'community', 'civic', 'civic_admin',
        'Healthcare', 'Public_Services', 'childcare',
    ]
    # ── Population / area ────────────────────────────────────────────────
    pop_area = ['TOTPOP', 'TOTHH', 'residential', 'retail',
                'construction', 'sa2_area']
    # ── Accessibility at destination (zone-level, not person-level) ───────
    acc_dest = ['auto_work_acc', 'auto_shop_acc', 'walk_acc', 'transit_acc']
    # ── Person attributes tested as interaction terms with destination ────
    # (person-by-destination interactions already in fixed backbone for income,
    # vehicles; GA tests additional demographics)
    person_dest_int = [
        'has_vehicle',    # vehicle × destination — for car-dependent zones
        'incp',           # income × distance (in addition to fixed c_incp_dist)
        'age_working',    # working-age × employment density
        'age_older',      # elderly × healthcare/services
        'hh_child',       # family × education / escort destinations
        'is_female',      # gender × recreation / safety
        'n_workers_hh',   # household workers × employment concentration
        'ptype_UniversityStudent',   # university student × HEREMPN
        'ptype_FullTimeWorker',      # FT worker × TOTEMP
        'income_hi',      # high-income × finance / professional zones
    ]
    return list(dict.fromkeys(
        skim_extra + emp_work + emp_edu + emp_nonmand + pop_area +
        acc_dest + person_dest_int
    ))


def _stage6_candidates(data: pd.DataFrame) -> List[str]:
    # Core skim + cost variables
    base_skim = ['tt_avg', 'tt_am', 'log_dist', 'dist', 'distwalk', 'distbike',
                 'ldw', 'ldb', 'crash', 'cost_proxy', 'accessibility']
    # Vehicle / licence
    base_veh  = ['has_vehicle', 'no_vehicle', 'vehrd', 'CARLICENCE']
    # Person demographics — richer heterogeneity per ATRF2022 / ActivitySim
    base_pers = ['sexp', 'female', 'incp', 'inc_hi', 'inc_lo',
                 'age5p', 'young', 'older',
                 'WORKSTAT', 'worker', 'STUDYING', 'student',
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
    GA for CDAP feature selection.
    Fitness = McFadden rho² of the 3-alt MNL (H/N/M) on HHTS persons.
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

    ga  = GeneticAlgorithm(
        len(feat_names), fitness,
        feature_names    = feat_names,
        collinear_groups = STAGE_COLLINEAR_GROUPS.get(0, []),
        **ga_kwargs,
    )
    best, history, pop = ga.run_with_restarts(n_restarts=n_restarts)
    best_feats = [f for f, g in zip(feat_names, best.genes) if g == 1]

    return GAResult(
        stage=0, best_features=best_feats,
        best_fitness=best.fitness, best_individual=best,
        history=history, all_candidates=pop,
        feature_names=feat_names, fitness_metric='mean(-BIC/n)',
        n_restarts=n_restarts, cache_hits=ga.cache_hits,
    )


def _run_ga_usual_loc(persons, landuse, skims, ga_kwargs, n_restarts=2) -> GAResult:
    """
    GA for usual location model parameter selection.

    Chromosome layout (mixed binary + continuous):
        gene[0]       : work_decay_b  — continuous in [0.01, 0.20]
        gene[1]       : school_decay_b — continuous in [0.01, 0.20]
        genes[2:]     : binary — which attractor columns to include
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
        feature_names    = feat_names,
        collinear_groups = STAGE_COLLINEAR_GROUPS.get(-1, []),
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


# ── Purpose label map used by all per-purpose runners ────────────────────
_PURP_LABELS = {
    'work':     'Work',
    'edu':      'School/Edu',
    'social':   'Social',
    'escort':   'Escort',
    'shopping': 'Shopping',
    'othmaint': 'Oth-Maint',
    'othdiscr': 'Oth-Discr',
    'eat':      'Eat Out',
}


def _run_ga_stage1(tours, persons, ga_kwargs, n_restarts=2) -> GAResult:
    from stage1_activity_frequency import prepare_frequency_data, PURPOSES, assign_ptype_column

    print("\n[Stage 1] Preparing estimation data...")
    data       = prepare_frequency_data(tours, persons)
    ptypes     = data['ptype'].dropna().unique().tolist()
    feat_names = _stage1_candidates(data)
    purposes   = [p for p in PURPOSES if f'n_{p}_out' in data.columns]

    print(f"  {len(data)} persons | {len(feat_names)} candidate features | "
          f"{len(purposes)} purposes × {len(ptypes)} ptypes")

    segment_results: dict = {}
    purp_best_genes: dict = {}

    for purpose in purposes:
        purp_label = f"Stage1 — {_PURP_LABELS.get(purpose, purpose)}"
        print(f"\n  [{purp_label}] Running per-purpose GA ...")

        def _purp_fit(genes, _p=purpose, _pt=ptypes):
            return _stage1_fitness_aggregate(genes, data, feat_names, [_p], _pt)

        purp_ga  = GeneticAlgorithm(
            len(feat_names), _purp_fit,
            feature_names    = feat_names,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get(1, []),
            **ga_kwargs,
        )
        purp_best, purp_hist, purp_pop = purp_ga.run_with_restarts(n_restarts=n_restarts)
        purp_feats = [f for f, g in zip(feat_names, purp_best.genes) if g == 1]
        seg_result = GAResult(
            stage=1, best_features=purp_feats,
            best_fitness=purp_best.fitness, best_individual=purp_best,
            history=purp_hist, all_candidates=purp_pop,
            feature_names=feat_names, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=purp_ga.cache_hits,
        )
        segment_results[purp_label] = seg_result
        purp_best_genes[purpose]    = purp_best.genes.copy()
        print(f"    BIC/n={-purp_best.fitness:.4f}  feats={purp_feats}")

    agg_fn = lambda g: _stage1_fitness_aggregate(g, data, feat_names, purposes, ptypes)
    return _majority_vote_result(segment_results, purp_best_genes, feat_names,
                                 agg_fn, stage=1, fitness_metric='mean(-BIC/n)',
                                 n_restarts=n_restarts)


def _run_ga_stage2(tours, persons, ga_kwargs, n_restarts=2) -> GAResult:
    from stage2_day_duration import aggregate_person_day

    print("\n[Stage 2] Preparing estimation data...")
    data       = aggregate_person_day(tours, persons)
    feat_names = _stage2_candidates(data)
    targets    = [t for t in ['start_of_day', 'end_of_day', 'B'] if t in data.columns]

    print(f"  {len(data)} person-days | {len(feat_names)} features | targets: {targets}")

    _TARGET_LABELS = {
        'start_of_day': 'Stage2 — Start-of-Day',
        'end_of_day':   'Stage2 — End-of-Day',
        'B':            'Stage2 — Day Duration (B)',
    }

    segment_results: dict = {}
    tgt_best_genes:  dict = {}

    for target in targets:
        tgt_label = _TARGET_LABELS.get(target, f'Stage2 — {target}')
        print(f"\n  [{tgt_label}] Running per-target GA ...")

        def _tgt_fit(genes, _t=target):
            return _stage2_fitness_aggregate(genes, data, feat_names, [_t])

        tgt_ga   = GeneticAlgorithm(
            len(feat_names), _tgt_fit,
            feature_names    = feat_names,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get(2, []),
            **ga_kwargs,
        )
        tgt_best, tgt_hist, tgt_pop = tgt_ga.run_with_restarts(n_restarts=n_restarts)
        tgt_feats = [f for f, g in zip(feat_names, tgt_best.genes) if g == 1]
        seg_result = GAResult(
            stage=2, best_features=tgt_feats,
            best_fitness=tgt_best.fitness, best_individual=tgt_best,
            history=tgt_hist, all_candidates=tgt_pop,
            feature_names=feat_names, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=tgt_ga.cache_hits,
        )
        segment_results[tgt_label] = seg_result
        tgt_best_genes[target]     = tgt_best.genes.copy()
        print(f"    BIC/n={-tgt_best.fitness:.4f}  feats={tgt_feats}")

    agg_fn = lambda g: _stage2_fitness_aggregate(g, data, feat_names, targets)
    return _majority_vote_result(segment_results, tgt_best_genes, feat_names,
                                 agg_fn, stage=2, fitness_metric='mean(-BIC/n)',
                                 n_restarts=n_restarts)


def _run_ga_stage3(tours, persons, ga_kwargs, trips=None, n_restarts=2) -> GAResult:
    from stage3_activity_duration import prepare_duration_data, PURPOSES

    print("\n[Stage 3] Preparing estimation data...")
    data = prepare_duration_data(tours, persons, trips=trips)
    purp_col = next(
        (c for c in ['main_purpose', '_purpose_coded', 'purpose'] if c in data.columns),
        None
    )
    if purp_col is None:
        raise RuntimeError("Stage 3 GA: no purpose column in prepared data.")

    feat_names = _stage3_candidates(data)
    purposes   = [p for p in PURPOSES if len(data[data[purp_col] == p]) >= 30]
    print(f"  {len(data)} activities | {len(feat_names)} features | "
          f"{len(purposes)} purposes with n\u226530")

    segment_results: dict = {}
    purp_best_genes: dict = {}

    for purpose in purposes:
        purp_label = f"Stage3 — {_PURP_LABELS.get(purpose, purpose)}"
        print(f"\n  [{purp_label}] Running per-purpose GA ...")

        def _purp_fit(genes, _p=purpose, _pc=purp_col):
            return _stage3_fitness_aggregate(genes, data, feat_names, [_p], _pc)

        purp_ga  = GeneticAlgorithm(
            len(feat_names), _purp_fit,
            feature_names    = feat_names,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get(3, []),
            **ga_kwargs,
        )
        purp_best, purp_hist, purp_pop = purp_ga.run_with_restarts(n_restarts=n_restarts)
        purp_feats = [f for f, g in zip(feat_names, purp_best.genes) if g == 1]
        seg_result = GAResult(
            stage=3, best_features=purp_feats,
            best_fitness=purp_best.fitness, best_individual=purp_best,
            history=purp_hist, all_candidates=purp_pop,
            feature_names=feat_names, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=purp_ga.cache_hits,
        )
        segment_results[purp_label] = seg_result
        purp_best_genes[purpose]    = purp_best.genes.copy()
        print(f"    BIC/n={-purp_best.fitness:.4f}  feats={purp_feats}")

    agg_fn = lambda g: _stage3_fitness_aggregate(g, data, feat_names, purposes, purp_col)
    return _majority_vote_result(segment_results, purp_best_genes, feat_names,
                                 agg_fn, stage=3, fitness_metric='mean(-BIC/n)',
                                 n_restarts=n_restarts)


def _run_ga_stage4(trips, tours, persons, ga_kwargs, n_restarts=2) -> GAResult:
    from stage4_schedule_placement import prepare_timing_data, PURPOSES

    print("\n[Stage 4] Preparing estimation data...")
    data       = prepare_timing_data(trips, tours, persons)
    feat_names = _stage4_candidates(data)
    purposes   = [p for p in PURPOSES if len(data[data['purpose'] == p]) >= 30]
    print(f"  {len(data)} trips | {len(feat_names)} features | "
          f"{len(purposes)} purposes")

    segment_results: dict = {}
    purp_best_genes: dict = {}

    for purpose in purposes:
        purp_label = f"Stage4 — {_PURP_LABELS.get(purpose, purpose)}"
        print(f"\n  [{purp_label}] Running per-purpose GA ...")

        def _purp_fit(genes, _p=purpose):
            return _stage4_fitness_aggregate(genes, data, feat_names, [_p])

        purp_ga  = GeneticAlgorithm(
            len(feat_names), _purp_fit,
            feature_names    = feat_names,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get(4, []),
            **ga_kwargs,
        )
        purp_best, purp_hist, purp_pop = purp_ga.run_with_restarts(n_restarts=n_restarts)
        purp_feats = [f for f, g in zip(feat_names, purp_best.genes) if g == 1]
        seg_result = GAResult(
            stage=4, best_features=purp_feats,
            best_fitness=purp_best.fitness, best_individual=purp_best,
            history=purp_hist, all_candidates=purp_pop,
            feature_names=feat_names, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=purp_ga.cache_hits,
        )
        segment_results[purp_label] = seg_result
        purp_best_genes[purpose]    = purp_best.genes.copy()
        print(f"    BIC/n={-purp_best.fitness:.4f}  feats={purp_feats}")

    agg_fn = lambda g: _stage4_fitness_aggregate(g, data, feat_names, purposes)
    return _majority_vote_result(segment_results, purp_best_genes, feat_names,
                                 agg_fn, stage=4, fitness_metric='mean(-BIC/n)',
                                 n_restarts=n_restarts)


def _run_ga_stage5(trips, persons, skims, landuse, ga_kwargs,
                   n_restarts=2, tours_src=None) -> GAResult:
    """
    GA for Stage 5 mode + destination variable selection.
    ESTIMATION ONLY — fits Larch models on HHTS, no prediction over synth pop.
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
    # Passing trips directly causes all tours to default to DA → degenerate mode model.
    _tours_input = tours_src if tours_src is not None else trips
    if tours_src is not None and 'tour_mode' in tours_src.columns:
        pass   # good — tours table already has observed modes
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
    skim_df  = pd.DataFrame()

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
    n_total    = n_mode + len(dest_cands)
    feat_names = mode_cands + dest_cands

    # ── Segment mapping ────────────────────────────────────────────────────
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

    # ── Per-segment SEPARATE mode GA then dest GA ──────────────────────────
    # Separate GAs avoid joint-chromosome entanglement between mode and
    # destination pools.  Mode GA runs first; its best vars are used to
    # pre-compute logsums for the subsequent destination GA.
    segment_results: dict = {}
    seg_best_mode_genes: dict = {}
    seg_best_dest_genes: dict = {}

    from stage5_destination_choice import (
        _filter_seg, _mode_tree, _build_mode_model, _estimate, _logsums,
        _attach_size_term, _attach_acdc_terms, _dest_tree,
    )

    for seg in valid_segs:
        seg_label = SEG_LABELS.get(seg, seg)
        seg_tours = _filter_seg(tours, seg)
        print(f"\n  [Stage 5 — {seg_label}] Mode GA ...")

        # ── Mode GA (chromosome = mode candidates only) ────────────────────
        def _mode_fit(genes, _s=seg):
            return _stage5_mode_only_fitness(
                genes, mode_cands, _s, tours, persons, landuse, skims)

        mode_ga  = GeneticAlgorithm(
            n_mode, _mode_fit,
            feature_names    = mode_cands,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get('5_mode', []),
            **ga_kwargs,
        )
        mode_best, mode_hist, mode_pop = mode_ga.run_with_restarts(n_restarts=n_restarts)
        best_mode_vars = [v for v, g in zip(mode_cands, mode_best.genes) if g == 1]
        print(f"    Mode best vars: {best_mode_vars}  BIC/n={-mode_best.fitness:.4f}")

        mode_seg_result = GAResult(
            stage=5, best_features={'mode': best_mode_vars, 'dest': []},
            best_fitness=mode_best.fitness, best_individual=mode_best,
            history=mode_hist, all_candidates=mode_pop,
            feature_names=mode_cands, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=mode_ga.cache_hits,
        )
        mode_seg_result.metadata['sub_model'] = 'mode'
        segment_results[f'{seg_label} — Mode'] = mode_seg_result
        seg_best_mode_genes[seg] = mode_best.genes.copy()

        # ── Pre-compute mode logsums with best mode vars ───────────────────
        ls_cached    = None
        lu_st_cached = None
        seg_acdc     = None
        logsums_ok   = False
        try:
            dt_m          = _mode_tree(seg_tours, persons, landuse, skims)
            mode_mdl_best = _build_mode_model(dt_m, seg, custom_mode_vars=best_mode_vars)
            mode_mdl_best = _estimate(mode_mdl_best)
            ls_cached     = _logsums(mode_mdl_best, dt_m, skims=skims, verbose=False)
            lu_st_cached  = _attach_size_term(landuse, seg)
            lu_st_cached, seg_acdc = _attach_acdc_terms(
                lu_st_cached, seg_tours, skim_df, seg)
            logsums_ok    = True
            print(f"    Mode logsums cached for Dest GA.")
        except Exception as e:
            logger.warning(f"Stage 5 [{seg}] could not pre-compute mode logsums: {e}")

        # ── Dest GA (chromosome = dest candidates only) ────────────────────
        print(f"\n  [Stage 5 — {seg_label}] Dest GA ...")
        if logsums_ok:
            def _dest_fit(genes, _s=seg, _ls=ls_cached, _acdc=seg_acdc,
                          _lu=lu_st_cached, _st=seg_tours):
                return _stage5_dest_only_fitness(
                    genes, dest_cands, _s, _st, persons, _lu, skims, _ls, _acdc)

            dest_ga = GeneticAlgorithm(
                len(dest_cands), _dest_fit,
                feature_names    = dest_cands,
                collinear_groups = STAGE_COLLINEAR_GROUPS.get('5_dest', []),
                **ga_kwargs,
            )
            dest_best, dest_hist, dest_pop = dest_ga.run_with_restarts(n_restarts=n_restarts)
            dest_cache = dest_ga.cache_hits
        else:
            # Fallback: joint GA if logsums could not be computed
            def _joint_fit(genes, _s=seg):
                return _stage5_fitness_larch(
                    genes[:n_mode], genes[n_mode:], mode_cands, dest_cands,
                    _s, tours, persons, landuse, skims, skim_df)
            fallback_genes = np.zeros(len(dest_cands), dtype=int)
            dest_best  = Individual(genes=fallback_genes, fitness=-1e9)
            dest_hist  = []
            dest_pop   = [dest_best]
            dest_cache = 0

        best_dest_vars = [v for v, g in zip(dest_cands, dest_best.genes) if g == 1]
        print(f"    Dest best vars: {best_dest_vars}  BIC/n={-dest_best.fitness:.4f}")

        dest_seg_result = GAResult(
            stage=5, best_features={'mode': best_mode_vars, 'dest': best_dest_vars},
            best_fitness=dest_best.fitness, best_individual=dest_best,
            history=dest_hist, all_candidates=dest_pop,
            feature_names=dest_cands, fitness_metric='mean(-BIC/n)',
            n_restarts=n_restarts, cache_hits=dest_cache,
        )
        dest_seg_result.metadata['sub_model'] = 'dest'
        segment_results[f'{seg_label} — Dest'] = dest_seg_result
        seg_best_dest_genes[seg] = dest_best.genes.copy()

    # ── Aggregate: majority vote across segments per sub-model ────────────
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

    if seg_best_mode_genes:
        mode_vote = np.array(list(seg_best_mode_genes.values())).mean(axis=0)
        consensus_mode = (mode_vote >= 0.5).astype(int)
        best_mode_feats = [v for v, g in zip(mode_cands, consensus_mode) if g == 1]
    else:
        consensus_mode  = np.zeros(n_mode, dtype=int)
        best_mode_feats = []

    if seg_best_dest_genes:
        dest_vote = np.array(list(seg_best_dest_genes.values())).mean(axis=0)
        consensus_dest = (dest_vote >= 0.5).astype(int)
        best_dest_feats = [v for v, g in zip(dest_cands, consensus_dest) if g == 1]
    else:
        consensus_dest  = np.zeros(len(dest_cands), dtype=int)
        best_dest_feats = []

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
    GA for Stage 51 — TRIP-LEVEL mode + destination variable selection.

    Runs SEPARATE mode and destination GAs per trip purpose, mirroring the
    ActivitySim distinction between trip_mode_choice and trip destination
    choice.  Separate GAs produce independent convergence plots for each
    sub-model and each purpose.

    For each valid purpose:
      1. Mode GA  : chromosome = mode_candidates only → -BIC_mode/n
      2. Dest GA  : chromosome = dest_candidates only → -BIC_dest/n
                    (trip-level destination logit using observed HHTS DTAZ)

    Results are labelled  "<Purpose> — Mode (trip)"  and  "<Purpose> — Dest (trip)".
    Majority-vote consensus across purposes selects the final feature sets.
    """
    import openmatrix as omx
    from stage6_mode_choice import prepare_mode_data, PURPOSES as TRIP_PURPOSES
    from stage5_destination_choice import _skim_flat

    print("\n[Stage 51 — TRIP LEVEL] Preparing HHTS trip estimation data...")

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

    # valid purposes = those with ≥50 trip-mode observations in HHTS
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

        # ── Mode GA ────────────────────────────────────────────────────────
        print(f"\n  [Stage 51 — {purp_label}] Mode GA ...")

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
            feature_names    = mode_cands,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get('5_mode', []),
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
        segment_results[f'{purp_label} — Mode'] = mode_seg_result
        purp_best_mode_genes[purp] = mode_best.genes.copy()

        # ── Dest GA ────────────────────────────────────────────────────────
        print(f"\n  [Stage 51 — {purp_label}] Dest GA ...")

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
            feature_names    = dest_cands,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get('5_dest', []),
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
        segment_results[f'{purp_label} — Dest'] = dest_seg_result
        purp_best_dest_genes[purp] = dest_best.genes.copy()

    # ── Majority-vote aggregate ────────────────────────────────────────────
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
    Runs one GA per trip-purpose (work, edu, social, …) then aggregates via
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

    # ── Calibrated seed individuals ────────────────────────────────────────
    _core = {'tt_avg', 'log_dist', 'ldw', 'ldb', 'crash',
             'has_vehicle', 'no_vehicle', 'vehrd', 'accessibility'}
    _demo = _core | {'female', 'older', 'young', 'worker', 'student', 'incp', 'inc_lo'}
    _full = _demo | {'CARLICENCE', 'cost_proxy', 'age5p', 'inc_hi', 'WORKSTAT',
                     'distwalk', 'distbike'}

    def _make_seed(feature_set):
        return Individual(genes=np.array(
            [1 if f in feature_set else 0 for f in feat_names], dtype=int))

    # ── Purpose group labels (trip-level) ─────────────────────────────────
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

    # ── Per-purpose GA runs (trip-level) ───────────────────────────────────
    segment_results: dict = {}
    purp_best_genes: dict  = {}

    for purp in purposes:
        purp_label = PURP_LABELS.get(purp, f'{purp} (trip)')
        purp_data  = data[data['purpose'] == purp].copy()
        print(f"\n  [Stage 6 — {purp_label}] n={len(purp_data)}  Running GA ...")

        def _purp_fitness(genes, _p=purp):
            return _stage6_fitness(genes, data, feat_names, _p)

        seeds = [_make_seed(s) for s in [_core, _demo, _full]]
        for s in seeds:
            try:   s.fitness = _purp_fitness(s.genes)
            except Exception: s.fitness = -1e9
        seeds.sort(key=lambda x: x.fitness, reverse=True)

        purp_ga   = GeneticAlgorithm(
            len(feat_names), _purp_fitness,
            feature_names    = feat_names,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get(6, []),
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

    # ── Majority-vote aggregate ────────────────────────────────────────────
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
            feature_names    = feat_names,
            collinear_groups = STAGE_COLLINEAR_GROUPS.get(6, []),
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
# 5.  APPLY GA RESULTS TO PIPELINE  (prediction — run ONCE after GA)
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
    print("APPLYING GA RESULTS — FULL PREDICTION PIPELINE")
    print("="*60)

    s1_feats = ga_results.results.get(1, None)
    s2_feats = ga_results.results.get(2, None)
    s3_feats = ga_results.results.get(3, None)
    s4_feats = ga_results.results.get(4, None)

    print("\n[1/5] Stage 1 — activity frequency prediction...")
    synth_s1, s1_models = run_stage1(
        out["tours"], out["persons"], synth,
        feature_override=s1_feats.best_features if s1_feats else None,
    )
    synth_s1.to_csv(os.path.join(save_dir, 'stage1_output.csv'), index=False)
    _save_model(s1_models, os.path.join(models_dir, 'stage1_models.pkl'))

    print("\n[2/5] Stage 2 — day duration prediction...")
    synth_s2, s2_models = run_stage2(
        out["tours"], out["persons"], synth_s1,
        feature_override=s2_feats.best_features if s2_feats else None,
    )
    synth_s2.to_csv(os.path.join(save_dir, 'stage2_output.csv'), index=False)
    _save_model(s2_models, os.path.join(models_dir, 'stage2_models.pkl'))

    print("\n[3/5] Stage 3 — activity duration prediction...")
    activities, s3_models = run_stage3(
        out["tours"], out["persons"], synth_s2, trips=out["trips"],
        feature_override=s3_feats.best_features if s3_feats else None,
    )
    activities.to_csv(os.path.join(save_dir, 'stage3_output.csv'), index=False)
    _save_model(s3_models, os.path.join(models_dir, 'stage3_models.pkl'))

    print("\n[4/5] Stage 4 — schedule placement prediction...")
    scheduled, s4_models = run_stage4(
        out["trips"], out["tours"], out["persons"], activities,
        feature_override=s4_feats.best_features if s4_feats else None,
    )
    scheduled.to_csv(os.path.join(save_dir, 'stage4_output.csv'), index=False)
    _save_model(s4_models, os.path.join(models_dir, 'stage4_models.pkl'))

    print("\n[5/5] Stage 5 — destination + mode choice (zone-by-zone)...")
    s5_result = ga_results.results.get(5, None)
    s5_mode_feats = s5_result.best_features.get('mode') if s5_result else None
    s5_dest_feats = s5_result.best_features.get('dest') if s5_result else None

    scheduled_dest, s5_models = run_stage5(
        out["trips"], synth, scheduled,
        skims          = skims_path,
        landuse        = landuse,
        tours          = out["tours"],
        seed           = seed,
        mode_feature_override = s5_mode_feats,
        dest_feature_override = s5_dest_feats,
    )
    scheduled_dest.to_csv(os.path.join(save_dir, 'stage5_output.csv'), index=False)
    _save_model(s5_models, os.path.join(models_dir, 'stage5_models.pkl'))

    print(f"\n{'='*60}")
    print(f"GA pipeline prediction complete.")
    print(f"  Persons: {synth['person_id'].nunique() if 'person_id' in synth.columns else len(synth):,}")
    print(f"  Trips:   {(scheduled_dest['row_type']=='trip').sum():,}")
    print(f"  Models saved → {models_dir}/")

    return scheduled_dest, s5_models


def _save_model(model, path: str) -> None:
    """Silently save a model object (dict, fitted estimator, etc.) to pickle."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"    Model saved → {path}")
    except Exception as e:
        logger.warning(f"Could not save model to {path}: {e}")


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

    # ── Activity plan validation plots (same as main pipeline) ──────────
    print(f"\nGenerating activity plan plots → {plot_dir}/")
    try:
        _plot_activity_plans(plan, n_sample=12, save_dir=plot_dir)
    except Exception as e:
        logger.warning(f"Activity plan plots failed: {e}")

    # ── GA convergence overview ──────────────────────────────────────────
    conv_path = os.path.join(plot_dir, 'ga_convergence.png')
    print(f"Saving GA convergence overview → {conv_path}")
    try:
        ga_results.plot_all(save_path=conv_path)
    except Exception as e:
        logger.warning(f"GA convergence plot failed: {e}")

    # ── Per-stage BIC/n summary bar chart ────────────────────────────────
    _plot_bic_summary(ga_results, save_dir=plot_dir)

    # ── GA feature-selection summary CSV ────────────────────────────────
    summary = ga_results.summary_table()
    summary_path = os.path.join(save_dir, 'ga_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"GA summary → {summary_path}")
    print(summary.to_string(index=False))

    print(f"\nValidation complete.  All outputs → {save_dir}/")
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
    ax.set_title('GA Variable Selection — Final BIC/n per Sub-Model', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(df['bic_n']):
        ax.text(v + 0.01 * max(df['bic_n']), i, f'{v:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bic_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"BIC summary chart → {path}")


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
    min_features:     int   = 2,
    max_features:     int   = None,
    tournament_k:     int   = 4,
    stagnation_limit: int   = 12,
    n_restarts:       int   = 2,
    seed:             int   = 42,
    verbose:          bool  = True,
) -> GAResult:
    """
    Run GA variable selection for one pipeline stage (estimation only).

    Stage codes
    -----------
     0  : CDAP — daily activity pattern feature selection (H/N/M MNL)
    -1  : Usual location — decay_b + attractor column selection
     1  : Stage 1 — activity frequency (ordered logit)
     2  : Stage 2 — day duration (OLS)
     3  : Stage 3 — activity duration (lognormal)
     4  : Stage 4 — schedule placement (MNL timing)
     5  : Stage 5 — destination + mode choice (Larch NL, TOUR level)
    51  : Stage 51 — trip-level mode + destination choice (HHTS trip legs)
     6  : Stage 6 — mode choice (nested logit, TRIP level, no destination)
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

    stage_labels = {
         0: 'CDAP — daily activity pattern',
        -1: 'Usual location — decay + attractors',
         1: 'Stage 1 — activity frequency',
         2: 'Stage 2 — day duration',
         3: 'Stage 3 — activity duration',
         4: 'Stage 4 — schedule placement',
         5: 'Stage 5 — tour-level destination + mode',
        51: 'Stage 51 — trip-level mode + destination',
         6: 'Stage 6 — mode choice (trip, no destination)',
    }
    label = stage_labels.get(stage, f'Stage {stage}')
    print(f"\n{'='*60}")
    print(f"GA VARIABLE SELECTION — {label}  (estimation only)")
    print(f"  pop={pop_size}  gens={n_generations}  restarts={n_restarts}")
    print(f"  mut_base={mutation_rate}→0.25(adaptive)  tournament_k={tournament_k}")
    print(f"{'='*60}")

    dispatch = {
         0: lambda: _run_ga_cdap(persons, ga_kwargs, n_restarts),
        -1: lambda: _run_ga_usual_loc(persons, landuse, skims, ga_kwargs, n_restarts),
         1: lambda: _run_ga_stage1(tours, persons, ga_kwargs, n_restarts),
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
            f"stage must be 0 (CDAP), -1 (usual_loc), 1–6, or 51 (trip-level), got {stage}")

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
     1–6: Pipeline stages 1 to 6

    Parameters
    ----------
    tours, trips, persons : HHTS from process_survey()
    skims, landuse        : required for stages 5, 6, and usual_loc (-1)
    stages                : explicit list of stage codes; if None, built automatically
    include_cdap          : add stage 0 (CDAP) to the run list
    include_usual_loc     : add stage -1 (usual location) — needs WORK_TAZ in persons
    include_trip_mode_dest: add stage 51 (trip-level mode + destination) in addition
                            to stage 5 (tour-level). Useful for validating whether
                            trip-level model produces consistent variable selections.
    save_path             : checkpoint file saved after each stage
    """
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
                stage         = stage,
                tours         = tours,
                trips         = trips,
                persons       = persons,
                skims         = skims,
                landuse       = landuse,
                pop_size      = pop_size,
                n_generations = n_generations,
                mutation_rate = mutation_rate,
                n_restarts    = n_restarts,
                seed          = seed,
            )
            all_results.results[stage] = result

            if save_path:
                all_results.save(save_path)
                all_results.export_reports(os.path.dirname(save_path) or '.')
                print(f"  Checkpoint saved → {save_path}")

        except Exception as e:
            logger.error(f"Stage {stage} GA failed: {e}", exc_info=True)
            print(f"  Stage {stage} skipped: {e}")

    all_results.summary()
    return all_results
