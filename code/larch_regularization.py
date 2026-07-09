"""Elastic-net (L1+L2) penalized-likelihood wrapper for Larch estimation.

metacountregressor adds an elastic-net penalty to its estimation objective
(solution.py: ``regularise_l1`` + ``regularise_l2``, both scaled by
``reg_penalty``, summed into ``-loglik + penalty``). Larch's own
``Model.maximize_loglike()`` has no equivalent hook - it hands
``model.logloss`` / ``model.d_logloss`` straight to
``scipy.optimize.minimize`` (this is the path actually used throughout this
pipeline, since every ``_estimate``/``maximize_loglike`` call site here first
calls ``set_cap()``, which gives every parameter finite bounds and makes
Larch pick SLSQP over BHHH - see ``larch/model/optimization.py``).

``elasticnet_objective`` temporarily wraps ``logloss``/``d_logloss`` on a
Larch model instance so the *same* shrinkage metacountregressor applies to
its count models also applies to Larch's discrete-choice models: weak or
collinear coefficients get pulled toward zero instead of exploding to the
``set_cap()`` bound.  The penalty is added in mean-loss (logloss) space, not
raw log-likelihood space, so its strength doesn't wash out as sample size
grows into the millions of rows typical of this pipeline.

Usage
-----
    from larch_regularization import elasticnet_objective

    with elasticnet_objective(model):
        result = model.maximize_loglike(options=est_options)

The wrapper is a no-op (penalty=0) unless GA_L1_L2_ALPHA is set > 0, or an
explicit alpha is passed.
"""
from __future__ import annotations

import os

import numpy as np

# Elastic-net strength (mean-loss/logloss space) and L1 vs L2 mix, mirroring
# metacountregressor's reg_penalty knob. Overridable via env vars so a run
# can be tuned without code changes; default is a small always-on shrinkage.
DEFAULT_L1_L2_ALPHA = float(os.environ.get("GA_L1_L2_ALPHA", "1e-3"))
DEFAULT_L1_RATIO = float(os.environ.get("GA_L1_RATIO", "0.5"))


def _free_pvals(pvals, holdfast):
    """Zero out holdfasted/fixed parameters so they're excluded from the penalty."""
    pvals = np.asarray(pvals, dtype=float)
    if holdfast is None:
        return pvals
    holdfast = np.asarray(holdfast, dtype=bool)
    if holdfast.shape != pvals.shape:
        return pvals
    return np.where(holdfast, 0.0, pvals)


def elasticnet_penalty_and_grad(pvals, holdfast, alpha: float, l1_ratio: float):
    """Elastic-net penalty value and gradient over free parameters.

    penalty = alpha * (l1_ratio * sum(|beta|) + (1 - l1_ratio) * sum(beta^2))

    Same functional form as metacountregressor's
    ``regularise_l1(betas) + regularise_l2(betas)`` (solution.py), except L1
    genuinely uses |beta| (metacountregressor's ``regularise_l1`` forward
    branch uses squares too - a bug in that package we deliberately don't
    reproduce here).
    """
    free = _free_pvals(pvals, holdfast)
    l1 = np.abs(free).sum()
    l2 = np.square(free).sum()
    penalty = alpha * (l1_ratio * l1 + (1.0 - l1_ratio) * l2)
    grad = alpha * (l1_ratio * np.sign(free) + (1.0 - l1_ratio) * 2.0 * free)
    return float(penalty), grad


class elasticnet_objective:
    """Context manager: temporarily adds an elastic-net penalty to a Larch
    model's ``logloss``/``d_logloss`` (the objective/gradient pair Larch's
    SLSQP path actually optimizes), then restores the originals on exit so
    reported LL/std-errors downstream reflect the true (unpenalized)
    likelihood.
    """

    def __init__(self, model, alpha: float = DEFAULT_L1_L2_ALPHA,
                 l1_ratio: float = DEFAULT_L1_RATIO):
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._orig_logloss = None
        self._orig_d_logloss = None
        self._patched = False

    def __enter__(self):
        model = self.model
        if self.alpha <= 0 or not hasattr(model, "logloss") or not hasattr(model, "d_logloss"):
            return model

        self._orig_logloss = model.logloss
        self._orig_d_logloss = model.d_logloss
        orig_logloss = self._orig_logloss
        orig_d_logloss = self._orig_d_logloss
        alpha, l1_ratio = self.alpha, self.l1_ratio

        def penalized_logloss(x=None, **kwargs):
            base = orig_logloss(x, **kwargs)
            pvals = x if x is not None else model.pvals
            penalty, _ = elasticnet_penalty_and_grad(pvals, getattr(model, "pholdfast", None),
                                                       alpha, l1_ratio)
            return base + penalty

        def penalized_d_logloss(x=None, **kwargs):
            grad = np.asarray(orig_d_logloss(x, **kwargs), dtype=float)
            pvals = x if x is not None else model.pvals
            _, pen_grad = elasticnet_penalty_and_grad(pvals, getattr(model, "pholdfast", None),
                                                        alpha, l1_ratio)
            return grad + pen_grad

        model.logloss = penalized_logloss
        model.d_logloss = penalized_d_logloss
        self._patched = True
        return model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patched:
            self.model.logloss = self._orig_logloss
            self.model.d_logloss = self._orig_d_logloss
        return False
