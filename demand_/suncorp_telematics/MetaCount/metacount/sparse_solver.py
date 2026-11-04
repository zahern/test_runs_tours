"""
SparseSolver: Fast coordinate descent with warm-starts for random parameter sweeps.

Useful for:
- Hyperparameter sensitivity analysis (varying sparsity prior, noise prior)
- Random feature sampling (dropout-style)
- Bootstrap / posterior sampling with different seeds
- Quick refitting when design matrix changes slightly
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from joblib import Parallel, delayed
from .sparse_bayes import SparseBayesianRegressor


class SparseSolver:
    """
    Manages a fleet of SparseBayesianRegressor instances for rapid parameter sweeps.
    
    Each solver instance can be warm-started from a previous fit, enabling
    fast exploration of:
    - Different (a0, b0) sparsity hyperparameters
    - Different noise priors (c0, d0)
    - Subsampled data (bootstrap)
    - Different feature subsets
    """
    
    def __init__(
        self,
        base_config: Dict,
        n_features: int
    ):
        """
        Parameters
        ----------
        base_config : dict with keys like max_iter, tol, verbose
        n_features : dimensionality of design matrix
        """
        self.base_config = base_config
        self.n_features = n_features
        self.results = []
    
    def solve_one(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_overrides: Optional[Dict] = None,
        warm_start_from: Optional[SparseBayesianRegressor] = None,
        seed: Optional[int] = None
    ) -> SparseBayesianRegressor:
        """
        Fit a single model with given parameter overrides.
        
        Returns fitted SparseBayesianRegressor.
        """
        config = self.base_config.copy()
        if param_overrides:
            config.update(param_overrides)
        
        if seed is not None:
            np.random.seed(seed)
        
        model = SparseBayesianRegressor(
            n_features=self.n_features,
            **config
        )
        
        if warm_start_from is not None:
            # Transfer variational parameters
            model.m = warm_start_from.m.copy()
            model.S = warm_start_from.S.copy()
            model.a = warm_start_from.a.copy()
            model.b = warm_start_from.b.copy()
            model.c = warm_start_from.c
            model.d = warm_start_from.d
            model.fitted_ = True
        
        model.fit(X, y, warm_start=warm_start_from is not None)
        return model
    
    def sweep_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List],
        n_jobs: int = 1,
        base_warm_start: Optional[SparseBayesianRegressor] = None
    ) -> List[Dict]:
        """
        Grid search over hyperparameters with warm starts.
        
        param_grid: dict mapping param_name -> list of values
        e.g., {'a0': [1e-6, 1e-4, 1e-2], 'max_iter': [200, 500]}
        
        Returns list of result dicts with model, params, metrics.
        """
        # Generate all combinations
        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))
        
        def fit_combo(combo):
            overrides = dict(zip(keys, combo))
            model = self.solve_one(X, y, overrides, warm_start_from=base_warm_start)
            report = model.get_sparsity_report()
            report.update({"params": overrides})
            return report
        
        if n_jobs == 1:
            results = [fit_combo(c) for c in combos]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_combo)(c) for c in combos
            )
        
        self.results.extend(results)
        return results
    
    def bootstrap_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 50,
        sample_frac: float = 0.8,
        feature_frac: float = 1.0,
        n_jobs: int = 1,
        base_warm_start: Optional[SparseBayesianRegressor] = None
    ) -> List[SparseBayesianRegressor]:
        """
        Fit bootstrap ensemble with optional feature subsampling.
        
        Returns list of fitted models.
        """
        N, D = X.shape
        n_sample = int(N * sample_frac)
        n_feat = max(1, int(D * feature_frac))
        
        def fit_bootstrap(seed):
            np.random.seed(seed)
            idx = np.random.choice(N, n_sample, replace=True)
            feat_idx = np.random.choice(D, n_feat, replace=False) if feature_frac < 1.0 else np.arange(D)
            X_sub = X[idx][:, feat_idx]
            y_sub = y[idx]
            
            # Need model with correct n_features
            config = self.base_config.copy()
            config['n_features'] = len(feat_idx)
            model = SparseBayesianRegressor(**config)
            
            # Can't warm-start with different feature set easily; start fresh
            model.fit(X_sub, y_sub)
            return model, feat_idx
        
        if n_jobs == 1:
            results = [fit_bootstrap(i) for i in range(n_bootstrap)]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_bootstrap)(i) for i in range(n_bootstrap)
            )
        
        models, feat_indices = zip(*results)
        return list(models)
    
    def random_feature_perturbation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_perturbations: int = 20,
        noise_scale: float = 0.1,
        n_jobs: int = 1,
        base_model: Optional[SparseBayesianRegressor] = None
    ) -> List[Dict]:
        """
        Perturb design matrix with noise and refit to assess stability.
        
        Returns list of {model, perturbation_seed, feature_stability} dicts.
        """
        N, D = X.shape
        
        def fit_perturbed(seed):
            np.random.seed(seed)
            X_pert = X + np.random.randn(N, D) * noise_scale * np.std(X, axis=0, keepdims=True)
            model = self.solve_one(X_pert, y, warm_start_from=base_model)
            return {"model": model, "seed": seed}
        
        if n_jobs == 1:
            results = [fit_perturbed(i) for i in range(n_perturbations)]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_perturbed)(i) for i in range(n_perturbations)
            )
        
        # Compute feature selection stability
        active_matrix = np.zeros((n_perturbations, D), dtype=bool)
        for i, r in enumerate(results):
            active_matrix[i, r["model"].get_active_features()] = True
        
        stability = active_matrix.mean(axis=0)
        for i, r in enumerate(results):
            r["feature_stability"] = stability
        
        return results
    
    def leave_one_out_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = 1
    ) -> Dict:
        """
        Leave-one-out cross-validation (exact, not approximate).
        Returns predictions and metrics for each left-out point.
        """
        N = X.shape[0]
        preds = np.zeros(N)
        pred_stds = np.zeros(N)
        
        def fit_loo(i):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            model = SparseBayesianRegressor(n_features=self.n_features, **self.base_config)
            model.fit(X[mask], y[mask])
            mean, std = model.predict(X[i:i+1], return_std=True)
            return i, mean[0], std[0] if std is not None else 0.0
        
        if n_jobs == 1:
            for i in range(N):
                idx, p, s = fit_loo(i)
                preds[idx] = p
                pred_stds[idx] = s
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(fit_loo)(i) for i in range(N))
            for idx, p, s in results:
                preds[idx] = p
                pred_stds[idx] = s
        
        mse = np.mean((y - preds)**2)
        mae = np.mean(np.abs(y - preds))
        coverage = np.mean(np.abs(y - preds) < 1.96 * pred_stds) if np.any(pred_stds > 0) else np.nan
        
        return {
            "predictions": preds,
            "pred_stds": pred_stds,
            "mse": mse,
            "mae": mae,
            "coverage_95": coverage,
            "r2": 1 - np.sum((y - preds)**2) / np.sum((y - np.mean(y))**2)
        }


def quick_sparsity_path(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: np.ndarray = None,
    base_config: Dict = None
) -> List[Dict]:
    """
    Compute sparsity path: fit models with increasing sparsity penalty.
    Equivalent to Lasso path but with ARD Bayesian formulation.
    
    Returns list of {alpha_scale, model, n_active, elbo} dicts.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-6, 2, 20)
    if base_config is None:
        base_config = {"max_iter": 300, "tol": 1e-5}
    
    N, D = X.shape
    results = []
    
    prev_model = None
    for a0 in alpha_grid:
        config = base_config.copy()
        config['a0'] = a0
        config['b0'] = a0
        
        model = SparseBayesianRegressor(n_features=D, **config)
        if prev_model is not None:
            model.m = prev_model.m.copy()
            model.S = prev_model.S.copy()
            model.a = prev_model.a.copy()
            model.b = prev_model.b.copy()
            model.c = prev_model.c
            model.d = prev_model.d
            model.fitted_ = True
        
        model.fit(X, y, warm_start=prev_model is not None)
        report = model.get_sparsity_report()
        report['alpha_scale'] = float(a0)
        results.append(report)
        prev_model = model
    
    return results