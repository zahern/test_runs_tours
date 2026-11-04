"""
RiskPredictor: Predict per-post, per-bucket hard-braking risk for a future fixture
using attendance, time-of-day, fixture-type, day-of-week, seasonality, and historical panel data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .sparse_bayes import SparseBayesianRegressor, SparseBayesianRegressorBatch
from .sparse_solver import SparseSolver, quick_sparsity_path


class RiskPredictor:
    """
    Hierarchical risk model:
    
    For each post j and bucket h:
      r_{j,h}^{(g)} = f(x^{(g)}) + epsilon
    
    where x^{(g)} includes:
      - log(attendance) 
      - time-of-day (sin/cos of kickoff hour)
      - fixture type (NRL, Super Rugby, A-League) one-hot
      - historical mean risk for that post/bucket (base rate)
      - interaction terms
    
    Uses SparseBayesianRegressor for automatic feature selection and uncertainty.
    """
    
    def __init__(
        self,
        posts_df: pd.DataFrame,
        n_buckets: int = 28,
        fixture_types: Optional[List[str]] = None,
        sparsity_prior: float = 1e-6,
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        posts_df : DataFrame with columns ['node_id', 'lat', 'lon', 'rank']
        n_buckets : number of 15-min time buckets around kickoff
        fixture_types : list of fixture type strings for one-hot encoding
        sparsity_prior : a0,b0 for ARD prior (smaller = sparser)
        verbose : print fitting progress
        """
        self.posts_df = posts_df.reset_index(drop=True)
        self.node_ids = posts_df['node_id'].to_numpy()
        self.post_ranks = posts_df['rank'].to_numpy() if 'rank' in posts_df.columns else np.arange(len(posts_df))
        self.n_posts = len(posts_df)
        self.n_buckets = n_buckets
        self.n_panels = self.n_posts * n_buckets
        self.fixture_types = fixture_types or ["NRL", "SuperRugby", "ALeague"]
        self.sparsity_prior = sparsity_prior
        self.verbose = verbose
        
        # Model: one sparse regressor per (post, bucket) panel
        # But we use shared design matrix with post/bucket specific coefficients
        self.models = None
        self.fitted_ = False
        self.feature_names = None
    
    def _build_design_matrix(
        self,
        fixtures_df: pd.DataFrame,
        historical_means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Build design matrix X for fixtures.
        
        Expected columns: fixture_id, attendance, kickoff_hour, fixture_type
        Optional columns: kickoff_iso (for date features), weather features
        
        Returns X: (n_fixtures, n_features)
        """
        n_fixtures = len(fixtures_df)
        
        # Core features
        attendance = fixtures_df['attendance'].to_numpy(dtype=float)
        kickoff_hour = fixtures_df['kickoff_hour'].to_numpy(dtype=float)
        
        # Log attendance (with small constant)
        log_att = np.log(attendance + 1.0)
        
        # Time of day: cyclic encoding
        tod_sin = np.sin(2 * np.pi * kickoff_hour / 24)
        tod_cos = np.cos(2 * np.pi * kickoff_hour / 24)
        
        # Day of week features (from kickoff_iso if available, else infer from fixture_id)
        if 'kickoff_iso' in fixtures_df.columns:
            dt = pd.to_datetime(fixtures_df['kickoff_iso'])
        else:
            # Try to parse from fixture_id if it contains date
            dt = pd.to_datetime(fixtures_df['fixture_id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0], 
                                errors='coerce')
            # Fallback: assume all same day
            dt = dt.fillna(pd.Timestamp('2026-01-01'))
        
        dow = dt.dt.dayofweek.to_numpy()  # 0=Mon, 6=Sun
        month = dt.dt.month.to_numpy()
        day_of_year = dt.dt.dayofyear.to_numpy()
        
        # Day of week: cyclic
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        # Weekend indicator
        is_weekend = (dow >= 5).astype(float)
        
        # Month: cyclic (seasonal)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Day of year: cyclic (annual pattern)
        doy_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Fixture type one-hot
        type_dummies = pd.get_dummies(fixtures_df['fixture_type'], prefix='type')
        for t in self.fixture_types:
            col = f'type_{t}'
            if col not in type_dummies.columns:
                type_dummies[col] = 0
        type_dummies = type_dummies[[f'type_{t}' for t in self.fixture_types]]
        
        # Base historical mean (per post-bucket)
        if historical_means is not None:
            overall_mean = np.mean(historical_means)
        else:
            overall_mean = 0.0
        
        # Build feature matrix
        features = {
            'intercept': np.ones(n_fixtures),
            'log_attendance': log_att,
            'tod_sin': tod_sin,
            'tod_cos': tod_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
            'is_weekend': is_weekend,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'doy_sin': doy_sin,
            'doy_cos': doy_cos,
            'historical_mean': overall_mean * np.ones(n_fixtures),
        }
        for t in self.fixture_types:
            features[f'type_{t}'] = type_dummies[f'type_{t}'].to_numpy()
        
        # Interaction terms
        features['log_att_x_tod_sin'] = log_att * tod_sin
        features['log_att_x_tod_cos'] = log_att * tod_cos
        features['log_att_x_dow_sin'] = log_att * dow_sin
        features['log_att_x_dow_cos'] = log_att * dow_cos
        features['log_att_x_is_weekend'] = log_att * is_weekend
        features['log_att_x_month_sin'] = log_att * month_sin
        features['log_att_x_month_cos'] = log_att * month_cos
        
        # Time-of-day x day-of-week interactions
        features['tod_sin_x_dow_sin'] = tod_sin * dow_sin
        features['tod_cos_x_dow_cos'] = tod_cos * dow_cos
        
        X = np.column_stack([features[k] for k in sorted(features.keys())])
        self.feature_names = sorted(features.keys())
        return X
    
    def fit(
        self,
        fixtures_df: pd.DataFrame,
        risk_panel: np.ndarray,
        historical_means: Optional[np.ndarray] = None,
        warm_start: bool = False
    ) -> "RiskPredictor":
        """
        Fit the risk predictor.
        
        Parameters
        ----------
        fixtures_df : DataFrame with columns [fixture_id, attendance, kickoff_hour, fixture_type]
        risk_panel : array (n_fixtures, n_posts, n_buckets) - observed risk per fixture
        historical_means : (n_posts, n_buckets) mean risk for base rate feature
        warm_start : if models exist, continue from current state
        
        Returns
        -------
        self
        """
        n_fixtures = len(fixtures_df)
        X = self._build_design_matrix(fixtures_df, historical_means)
        n_features = X.shape[1]
        
        # Reshape risk panel to (n_fixtures, n_panels)
        Y = risk_panel.reshape(n_fixtures, self.n_panels)
        
        # Fit one sparse model per panel (post-bucket)
        if self.models is None or not warm_start:
            self.models = SparseBayesianRegressorBatch(
                n_models=self.n_panels,
                n_features=n_features,
                a0=self.sparsity_prior,
                b0=self.sparsity_prior,
                verbose=self.verbose
            )
        
        self.models.fit_batch(X, Y, warm_start=warm_start)
        self.fitted_ = True
        return self
    
    def predict(
        self,
        fixture_specs: pd.DataFrame,
        historical_means: Optional[np.ndarray] = None,
        return_std: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict risk for new fixture(s).
        
        Parameters
        ----------
        fixture_specs : DataFrame with columns [attendance, kickoff_hour, fixture_type]
                       one row per fixture to predict
        historical_means : (n_posts, n_buckets) for base rate feature
        return_std : if True, also return predictive std
        
        Returns
        -------
        risk_pred : (n_fixtures, n_posts, n_buckets) predicted risk
        risk_std  : (n_fixtures, n_posts, n_buckets) predictive std (if requested)
        """
        assert self.fitted_, "Call fit() first"
        n_fixtures = len(fixture_specs)
        X = self._build_design_matrix(fixture_specs, historical_means)
        means, stds = self.models.predict_batch(X)
        # Reshape to (n_fixtures, n_posts, n_buckets)
        risk_pred = means.reshape(n_fixtures, self.n_posts, self.n_buckets)
        if return_std:
            risk_std = stds.reshape(n_fixtures, self.n_posts, self.n_buckets)
            return risk_pred, risk_std
        return risk_pred, None
    
    def predict_posterior_samples(
        self,
        fixture_specs: pd.DataFrame,
        n_samples: int = 100,
        historical_means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw posterior predictive samples for uncertainty quantification.
        
        Returns: (n_fixtures, n_samples, n_posts, n_buckets)
        """
        assert self.fitted_, "Call fit() first"
        n_fixtures = len(fixture_specs)
        X = self._build_design_matrix(fixture_specs, historical_means)
        samples = []
        for m in self.models.models:
            w_samps = m.sample_posterior(n_samples, X)  # (n_fixtures, n_samples)
            samples.append(w_samps)
        # Stack: (n_panels, n_fixtures, n_samples) -> (n_fixtures, n_samples, n_panels)
        samples = np.stack(samples, axis=0).transpose(1, 2, 0)
        return samples.reshape(n_fixtures, n_samples, self.n_posts, self.n_buckets)
    
    def leave_one_out_cv(
        self,
        fixtures_df: pd.DataFrame,
        risk_panel: np.ndarray,
        historical_means: Optional[np.ndarray] = None,
        n_jobs: int = 1
    ) -> Dict:
        """
        True leave-one-out cross-validation.
        
        For each fixture, train on all other fixtures and predict the left-out one.
        
        Returns dict with:
        - predictions: (n_fixtures, n_posts, n_buckets)
        - pred_stds: (n_fixtures, n_posts, n_buckets)
        - mse, mae, r2 per fixture and overall
        - coverage_95: proportion of true values within 95% predictive interval
        """
        n_fixtures = len(fixtures_df)
        n_posts, n_buckets = self.n_posts, self.n_buckets
        
        predictions = np.zeros((n_fixtures, n_posts, n_buckets))
        pred_stds = np.zeros((n_fixtures, n_posts, n_buckets))
        
        def fit_and_predict(left_out_idx):
            """Train on all except left_out_idx, predict left_out_idx."""
            mask = np.ones(n_fixtures, dtype=bool)
            mask[left_out_idx] = False
            
            # Create new predictor for this fold
            fold_predictor = RiskPredictor(
                posts_df=self.posts_df,
                n_buckets=self.n_buckets,
                fixture_types=self.fixture_types,
                sparsity_prior=self.sparsity_prior,
                verbose=False
            )
            fold_predictor.fit(
                fixtures_df[mask],
                risk_panel[mask],
                historical_means=historical_means,
                warm_start=False
            )
            
            # Predict left-out fixture
            test_fixture = fixtures_df.iloc[[left_out_idx]]
            pred, std = fold_predictor.predict(
                test_fixture, historical_means=historical_means, return_std=True
            )
            return left_out_idx, pred[0], std[0]
        
        if n_jobs == 1:
            for i in range(n_fixtures):
                idx, pred, std = fit_and_predict(i)
                predictions[idx] = pred
                pred_stds[idx] = std
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_predict)(i) for i in range(n_fixtures)
            )
            for idx, pred, std in results:
                predictions[idx] = pred
                pred_stds[idx] = std
        
        # Compute metrics
        true_vals = risk_panel.reshape(n_fixtures, -1)
        pred_vals = predictions.reshape(n_fixtures, -1)
        std_vals = pred_stds.reshape(n_fixtures, -1)
        
        mse = np.mean((true_vals - pred_vals)**2)
        mae = np.mean(np.abs(true_vals - pred_vals))
        ss_res = np.sum((true_vals - pred_vals)**2)
        ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        
        # Coverage of 95% predictive intervals
        in_interval = np.abs(true_vals - pred_vals) < 1.96 * std_vals
        coverage_95 = np.mean(in_interval) if np.any(std_vals > 0) else np.nan
        
        # Per-fixture metrics
        per_fixture = []
        for i in range(n_fixtures):
            t = true_vals[i]
            p = pred_vals[i]
            s = std_vals[i]
            per_fixture.append({
                "fixture_id": fixtures_df.iloc[i]['fixture_id'],
                "mse": float(np.mean((t - p)**2)),
                "mae": float(np.mean(np.abs(t - p))),
                "r2": float(1 - np.sum((t - p)**2) / np.sum((t - np.mean(t))**2)) if np.var(t) > 0 else np.nan,
                "coverage_95": float(np.mean(np.abs(t - p) < 1.96 * s)) if np.any(s > 0) else np.nan,
            })
        
        return {
            "predictions": predictions,
            "pred_stds": pred_stds,
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2),
            "coverage_95": float(coverage_95),
            "per_fixture": per_fixture,
            "n_fixtures": n_fixtures,
        }
    
    def hyperparameter_sweep(
        self,
        fixtures_df: pd.DataFrame,
        risk_panel: np.ndarray,
        historical_means: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
        n_jobs: int = 1
    ) -> List[Dict]:
        """
        Sweep hyperparameters (sparsity prior, etc.) using SparseSolver.
        
        Returns list of results with metrics for each parameter combination.
        """
        if param_grid is None:
            param_grid = {
                'a0': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                'b0': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            }
        
        X = self._build_design_matrix(fixtures_df, historical_means)
        # Use first panel (post 0, bucket 0) as representative for sweep
        y = risk_panel[:, 0, 0]
        
        solver = SparseSolver(
            base_config={"max_iter": 200, "tol": 1e-5, "verbose": False},
            n_features=X.shape[1]
        )
        
        return solver.sweep_hyperparameters(X, y, param_grid=param_grid, n_jobs=n_jobs)
    
    def random_feature_sweep(
        self,
        fixtures_df: pd.DataFrame,
        risk_panel: np.ndarray,
        historical_means: Optional[np.ndarray] = None,
        n_perturbations: int = 20,
        noise_scale: float = 0.1,
        n_jobs: int = 1
    ) -> List[Dict]:
        """
        Random feature perturbation sweep using SparseSolver.
        Assesses feature selection stability.
        """
        X = self._build_design_matrix(fixtures_df, historical_means)
        y = risk_panel[:, 0, 0]
        
        solver = SparseSolver(
            base_config={"max_iter": 200, "tol": 1e-5, "verbose": False},
            n_features=X.shape[1]
        )
        
        # First fit base model
        base_model = SparseBayesianRegressor(
            n_features=X.shape[1],
            a0=self.sparsity_prior,
            b0=self.sparsity_prior,
            max_iter=200,
            tol=1e-5
        )
        base_model.fit(X, y)
        
        return solver.random_feature_perturbation(
            X, y,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            n_jobs=n_jobs,
            base_model=base_model
        )
    
    def sparsity_path(
        self,
        fixtures_df: pd.DataFrame,
        risk_panel: np.ndarray,
        historical_means: Optional[np.ndarray] = None,
        alpha_grid: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Compute sparsity path (feature selection vs regularization strength).
        """
        X = self._build_design_matrix(fixtures_df, historical_means)
        y = risk_panel[:, 0, 0]
        
        if alpha_grid is None:
            alpha_grid = np.logspace(-6, 1, 20)
        
        return quick_sparsity_path(
            X, y,
            alpha_grid=alpha_grid,
            base_config={"max_iter": 300, "tol": 1e-5}
        )
    
    def get_feature_importance(self, aggregate: str = "mean") -> pd.DataFrame:
        """
        Get feature importance across all panels.
        
        Returns DataFrame with columns: feature, n_active_panels, mean_|w|, mean_alpha
        """
        assert self.fitted_, "Call fit() first"
        n_models = self.n_panels
        feat_data = {name: {"active": 0, "w_abs": [], "alpha": []} for name in self.feature_names}
        
        for model in self.models.models:
            active = model.get_active_features()
            for idx in active:
                feat_data[self.feature_names[idx]]["active"] += 1
                feat_data[self.feature_names[idx]]["w_abs"].append(abs(model.m[idx]))
                feat_data[self.feature_names[idx]]["alpha"].append(model.E_alpha[idx])
        
        rows = []
        for name, d in feat_data.items():
            rows.append({
                "feature": name,
                "n_active_panels": d["active"],
                "prop_active": d["active"] / n_models,
                "mean_abs_weight": np.mean(d["w_abs"]) if d["w_abs"] else 0.0,
                "mean_alpha": np.mean(d["alpha"]) if d["alpha"] else np.inf,
            })
        
        df = pd.DataFrame(rows)
        if aggregate == "mean":
            df["importance"] = df["prop_active"] * df["mean_abs_weight"]
            df = df.sort_values("importance", ascending=False)
        return df
    
    def save(self, path: str):
        """Save model to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'posts_df': self.posts_df,
                'n_buckets': self.n_buckets,
                'fixture_types': self.fixture_types,
                'sparsity_prior': self.sparsity_prior,
                'models': self.models,
                'feature_names': self.feature_names,
                'fitted_': self.fitted_
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "RiskPredictor":
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            posts_df=data['posts_df'],
            n_buckets=data['n_buckets'],
            fixture_types=data['fixture_types'],
            sparsity_prior=data['sparsity_prior']
        )
        obj.models = data['models']
        obj.feature_names = data['feature_names']
        obj.fitted_ = data['fitted_']
        return obj


def build_fixture_dataframe(
    fixtures_list: List[Tuple[str, str]],
    attendances: List[int],
    fixture_type_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Helper to build fixtures_df from the FIXTURES list format.
    
    Parameters
    ----------
    fixtures_list : [(label, ISO_datetime), ...]
    attendances : list of int attendance per fixture
    fixture_type_map : {label: type_string} mapping
    
    Returns
    -------
    DataFrame with fixture_id, attendance, kickoff_hour, fixture_type
    """
    rows = []
    for (label, kiso), att in zip(fixtures_list, attendances):
        dt = pd.Timestamp(kiso)
        rows.append({
            "fixture_id": label,
            "attendance": att,
            "kickoff_hour": dt.hour + dt.minute / 60.0,
            "fixture_type": fixture_type_map.get(label, "Unknown")
        })
    return pd.DataFrame(rows)