"""
Sparse Bayesian Regressor with Automatic Relevance Determination (ARD).

Implements a linear model with sparsity-inducing priors (Student-t / horseshoe-like)
using variational Bayes. Provides:
- Automatic feature selection (many weights driven to ~0)
- Uncertainty quantification via posterior variance
- Fast coordinate-descent updates with warm starts for parameter sweeps
"""

import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import norm
from typing import Optional, Dict, Tuple
import warnings


class SparseBayesianRegressor:
    """
    Sparse linear regression: y = X @ w + eps, eps ~ N(0, sigma^2)
    
    Prior: w_j ~ N(0, alpha_j^-1), alpha_j ~ Gamma(a0, b0)  (ARD)
           sigma^2 ~ Gamma(c0, d0)
    
    Variational posterior: q(w) = N(m, S), q(alpha) = Gamma(a, b), q(sigma2) = Gamma(c, d)
    
    Coordinate ascent updates (Beal 2003; Tipping 2001).
    """
    
    def __init__(
        self,
        n_features: int,
        a0: float = 1e-6,
        b0: float = 1e-6,
        c0: float = 1e-6,
        d0: float = 1e-6,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = False
    ):
        self.n_features = n_features
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Variational parameters
        self.m = np.zeros(n_features)          # E[w]
        self.S = np.eye(n_features)            # Cov[w]
        self.a = np.full(n_features, a0 + 0.5) # Gamma shape for alpha
        self.b = np.full(n_features, b0)       # Gamma rate for alpha
        self.c = c0 + 0.5                      # Gamma shape for sigma^-2
        self.d = d0                            # Gamma rate for sigma^-2
        
        self.fitted_ = False
        self.elbo_history = []
    
    @property
    def E_alpha(self) -> np.ndarray:
        return self.a / self.b
    
    @property
    def E_sigma2(self) -> float:
        return self.d / (self.c - 1) if self.c > 1 else 1.0
    
    @property
    def E_sigma2_inv(self) -> float:
        return self.c / self.d
    
    def _compute_elbo(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evidence lower bound (for monitoring convergence)."""
        N, D = X.shape
        E_w = self.m
        E_w2 = self.m**2 + np.diag(self.S)
        E_alpha = self.E_alpha
        
        # E[log p(y|w,sigma2)]
        resid = y - X @ E_w
        E_loglik = -0.5 * N * np.log(2*np.pi) + 0.5 * N * (digamma(self.c) - np.log(self.d)) \
                   - 0.5 * self.E_sigma2_inv * (resid @ resid + np.trace(X @ self.S @ X.T))
        
        # E[log p(w|alpha)]
        E_logpw = 0.5 * np.sum(digamma(self.a) - np.log(self.b)) \
                  - 0.5 * np.sum(E_alpha * E_w2)
        
        # E[log p(alpha)]
        E_logpa = np.sum(self.a0 * np.log(self.b0) - gammaln(self.a0) \
                         + (self.a0 - 1) * (digamma(self.a) - np.log(self.b)) \
                         - self.b0 * E_alpha)
        
        # E[log p(sigma2)]
        E_logps = self.c0 * np.log(self.d0) - gammaln(self.c0) \
                  + (self.c0 - 1) * (digamma(self.c) - np.log(self.d)) \
                  - self.d0 * self.E_sigma2_inv
        
        # Entropies
        H_w = 0.5 * D * (1 + np.log(2*np.pi)) + 0.5 * np.linalg.slogdet(self.S)[1]
        H_alpha = np.sum(self.a - np.log(self.b) + gammaln(self.a) \
                         + (1 - self.a) * digamma(self.a))
        H_sigma = self.c - np.log(self.d) + gammaln(self.c) + (1 - self.c) * digamma(self.c)
        
        return float(E_loglik + E_logpw + E_logpa + E_logps + H_w + H_alpha + H_sigma)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        warm_start: bool = True,
        max_iter: Optional[int] = None
    ) -> "SparseBayesianRegressor":
        """
        Fit the model using coordinate ascent variational inference.
        
        Parameters
        ----------
        X : (N, D) design matrix
        y : (N,) target vector
        warm_start : if True, continue from current variational params
        max_iter : override default max iterations
        
        Returns
        -------
        self
        """
        N, D = X.shape
        assert D == self.n_features
        
        if not warm_start or not self.fitted_:
            self.m = np.zeros(D)
            self.S = np.eye(D)
            self.a = np.full(D, self.a0 + 0.5)
            self.b = np.full(D, self.b0)
            self.c = self.c0 + 0.5 * N
            self.d = self.d0
        
        iters = max_iter or self.max_iter
        XTX = X.T @ X
        XTy = X.T @ y
        
        for it in range(iters):
            m_old = self.m.copy()
            
            # Update q(w): Gaussian with precision = E[alpha] * I + E[sigma^-2] * X^T X
            prec_w = np.diag(self.E_alpha) + self.E_sigma2_inv * XTX
            self.S = np.linalg.inv(prec_w)
            self.m = self.E_sigma2_inv * self.S @ XTy
            
            # Update q(alpha): Gamma(a, b) with a = a0 + 0.5, b = b0 + 0.5 * E[w_j^2]
            E_w2 = self.m**2 + np.diag(self.S)
            self.a = self.a0 + 0.5
            self.b = self.b0 + 0.5 * E_w2
            
            # Update q(sigma^-2): Gamma(c, d)
            resid = y - X @ self.m
            self.c = self.c0 + 0.5 * N
            self.d = self.d0 + 0.5 * (resid @ resid + np.trace(X @ self.S @ X.T))
            
            # ELBO for monitoring
            elbo = self._compute_elbo(X, y)
            self.elbo_history.append(elbo)
            
            if self.verbose and it % 50 == 0:
                n_active = np.sum(self.E_alpha < 1e3)
                print(f"  iter {it}: ELBO={elbo:.3f}, active={n_active}, sigma2={self.E_sigma2:.4f}")
            
            if np.max(np.abs(self.m - m_old)) < self.tol:
                if self.verbose:
                    print(f"  converged at iter {it}")
                break
        
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict mean and optionally std of predictive distribution."""
        assert self.fitted_, "Call fit() first"
        mean = X @ self.m
        if not return_std:
            return mean, None
        # Predictive variance: x_i^T S x_i + E[sigma^2]
        var = np.sum((X @ self.S) * X, axis=1) + self.E_sigma2
        return mean, np.sqrt(np.maximum(var, 0))
    
    def sample_posterior(self, n_samples: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw posterior samples of w (and predictions if X given)."""
        assert self.fitted_, "Call fit() first"
        # Sample w ~ N(m, S)
        L = np.linalg.cholesky(self.S + 1e-10 * np.eye(self.n_features))
        w_samples = self.m + (L @ np.random.randn(self.n_features, n_samples)).T
        if X is not None:
            return X @ w_samples.T  # (N, n_samples)
        return w_samples
    
    def get_active_features(self, threshold: float = 1e3) -> np.ndarray:
        """Indices of features with E[alpha] < threshold (i.e., not pruned)."""
        return np.where(self.E_alpha < threshold)[0]
    
    def get_sparsity_report(self) -> Dict:
        """Summary of sparsity and uncertainty."""
        active = self.get_active_features()
        return {
            "n_features": self.n_features,
            "n_active": len(active),
            "active_features": active.tolist(),
            "E_alpha": self.E_alpha.tolist(),
            "E_w": self.m.tolist(),
            "E_w_std": np.sqrt(np.diag(self.S)).tolist(),
            "E_sigma2": self.E_sigma2,
            "elbo_final": self.elbo_history[-1] if self.elbo_history else None
        }


class SparseBayesianRegressorBatch:
    """
    Fit multiple independent sparse regressors (e.g., one per post/bucket)
    with shared hyperparameters but independent posteriors.
    Useful for panel data: (post, bucket) pairs each get their own model.
    """
    
    def __init__(self, n_models: int, n_features: int, **kwargs):
        self.models = [SparseBayesianRegressor(n_features, **kwargs) for _ in range(n_models)]
        self.n_models = n_models
        self.n_features = n_features
    
    def fit_batch(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        """Fit all models. X: (N, D), Y: (N, M)"""
        N, D = X.shape
        M = Y.shape[1]
        assert M == self.n_models
        for m in range(M):
            self.models[m].fit(X, Y[:, m], **kwargs)
        return self
    
    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (means (N,M), stds (N,M))"""
        means = np.column_stack([m.predict(X)[0] for m in self.models])
        stds = np.column_stack([m.predict(X, return_std=True)[1] for m in self.models])
        return means, stds
    
    def get_sparsity_reports(self):
        return [m.get_sparsity_report() for m in self.models]