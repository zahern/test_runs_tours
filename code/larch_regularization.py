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


def elasticnet_penalty_and_grad(pvals, holdfast, alpha: float, l1_ratio: float,
                                reference=None, weights=None):
    """Elastic-net penalty value and gradient over free parameters.

    penalty = alpha * sum_i w_i * (l1_ratio * |beta_i - ref_i|
                                   + (1 - l1_ratio) * (beta_i - ref_i)^2)

    Same functional form as metacountregressor's
    ``regularise_l1(betas) + regularise_l2(betas)`` (solution.py), except L1
    genuinely uses |beta| (metacountregressor's ``regularise_l1`` forward
    branch uses squares too - a bug in that package we deliberately don't
    reproduce here).

    reference : optional per-parameter shrink target (default all-zeros -
        classic elastic net). Lets nesting parameters be shrunk toward 1.0
        (the flat/MNL value) instead of 0.0 - shrinking a mu toward 0 pushes
        the model DEEPER into its degenerate near-argmax regime, the exact
        pathology regularisation is meant to prevent.
    weights : optional per-parameter penalty multiplier (default all-ones).
        Used to penalize weakly-identified parameters (Mu_*, ASCs) harder
        than slope coefficients.
    """
    pvals = np.asarray(pvals, dtype=float)
    dev = pvals if reference is None else pvals - np.asarray(reference, dtype=float)
    dev = _free_pvals(dev, holdfast)
    w = np.ones_like(dev) if weights is None else np.asarray(weights, dtype=float)
    l1 = (w * np.abs(dev)).sum()
    l2 = (w * np.square(dev)).sum()
    penalty = alpha * (l1_ratio * l1 + (1.0 - l1_ratio) * l2)
    grad = alpha * w * (l1_ratio * np.sign(dev) + (1.0 - l1_ratio) * 2.0 * dev)
    return float(penalty), grad


class elasticnet_objective:
    """Context manager: temporarily adds an elastic-net penalty to a Larch
    model's ``logloss``/``d_logloss`` (the objective/gradient pair Larch's
    SLSQP path actually optimizes), then restores the originals on exit so
    reported LL/std-errors downstream reflect the true (unpenalized)
    likelihood.
    """

    def __init__(self, model, alpha: float = DEFAULT_L1_L2_ALPHA,
                 l1_ratio: float = DEFAULT_L1_RATIO,
                 reference: dict = None, weights: dict = None):
        """
        reference / weights: optional {param_name: value} dicts, resolved
        against model.pnames at __enter__ (unnamed params get ref 0 /
        weight 1). See nl_identification_priors() for the ready-made NL
        recipe (mu -> 1, ASC -> 0).
        """
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.reference = reference
        self.weights = weights
        self._orig_logloss = None
        self._orig_d_logloss = None
        self._patched = False

    def _resolve(self, mapping, default):
        """{name: value} dict -> per-parameter array in model.pnames order."""
        if not mapping:
            return None
        try:
            names = list(self.model.pnames)
        except Exception:
            return None
        return np.array([float(mapping.get(str(n), default)) for n in names])

    def __enter__(self):
        model = self.model
        if self.alpha <= 0 or not hasattr(model, "logloss") or not hasattr(model, "d_logloss"):
            return model

        self._orig_logloss = model.logloss
        self._orig_d_logloss = model.d_logloss
        orig_logloss = self._orig_logloss
        orig_d_logloss = self._orig_d_logloss
        alpha, l1_ratio = self.alpha, self.l1_ratio
        ref_arr = self._resolve(self.reference, 0.0)
        w_arr = self._resolve(self.weights, 1.0)

        def penalized_logloss(x=None, **kwargs):
            base = orig_logloss(x, **kwargs)
            pvals = x if x is not None else model.pvals
            penalty, _ = elasticnet_penalty_and_grad(pvals, getattr(model, "pholdfast", None),
                                                       alpha, l1_ratio,
                                                       reference=ref_arr, weights=w_arr)
            return base + penalty

        def penalized_d_logloss(x=None, **kwargs):
            grad = np.asarray(orig_d_logloss(x, **kwargs), dtype=float)
            pvals = x if x is not None else model.pvals
            _, pen_grad = elasticnet_penalty_and_grad(pvals, getattr(model, "pholdfast", None),
                                                        alpha, l1_ratio,
                                                        reference=ref_arr, weights=w_arr)
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


# Identification priors for the stage-5 nested logit. Overridable via env so
# a run can tune the strength without code changes.
NL_PRIOR_ALPHA = float(os.environ.get("GA_NL_PRIOR_ALPHA", "5e-3"))


def nl_identification_priors(model):
    """Reference/weight dicts for regularizing a Larch nested-logit mode model.

    Weakly-identified parameter groups (job 23880279: Mu_Car at its 0.10
    bound, ASC_SR with SE~12) are ridge-shrunk toward identifiable defaults:

      Mu_*   -> 1.0 (the flat/MNL limit; data must EARN departures from it)
      ASC_*  -> 0.0, penalized 5x harder than slopes (ASCs soak up any
                misspecification and are the first to blow up)
      slopes -> 0.0 at weight 1 (mild classic ridge)

    Returns (reference, weights) dicts for elasticnet_objective. Usage:

        ref, w = nl_identification_priors(m)
        with elasticnet_objective(m, alpha=NL_PRIOR_ALPHA, l1_ratio=0.0,
                                  reference=ref, weights=w):
            m.maximize_loglike()
    """
    reference, weights = {}, {}
    try:
        names = [str(n) for n in model.pnames]
    except Exception:
        return reference, weights
    for n in names:
        if n.startswith("Mu_"):
            reference[n] = 1.0
            weights[n] = 10.0
        elif n.startswith("ASC_"):
            reference[n] = 0.0
            weights[n] = 5.0
        else:
            reference[n] = 0.0
            weights[n] = 1.0
    return reference, weights
