"""
regret_choice.py — Random Regret Minimization (RRM) choice models
=================================================================
POLARIS lets each discrete-choice component use a RUM (utility-maximisation) or
an RRM (regret-minimisation) decision rule. This framework already has RRM in the
stage-3 planning-order and stage-4 timing models (via SearchLibrium), but not in
the stage-5 mode / destination choice, and the decision rule is nowhere a
SEARCHABLE hypothesis. This module supplies the reusable RRM engine so the rule
(RUM vs RRM) can be a first-class searched decision, chosen by BIC — exactly like
the nested-vs-flat mode-structure decision.

Decision rules implemented (all as a softmax over available alternatives):
  • 'rum'  — utility:  V_i = ASC_i + Σ_k β_k x_ik           (baseline logit)
  • 'rrm'  — classic Random Regret (Chorus 2010):
        R_i = Σ_{j≠i} Σ_k ln(1 + exp(β_k (x_jk − x_ik))),   V_i = ASC_i − R_i
  • 'prrm' — Pure/Profit RRM (van Cranenburgh & Prato): the attributes are
        transformed once (x̂ depends on the attribute sign, not on β) so regret is
        LINEAR-in-parameters, R_i = Σ_k β_k x̂_ik, hence estimable as an MNL on −x̂.
  • 'hybrid' — the POLARIS / Golshani et al. (2018) hybrid RUM-RRM used in the
        activity-timing (start-time) choice: a convex blend of the two rules with
        a SHARED β and an estimated weight λ∈(0,1),
            V_i = ASC_i + (1−λ) Σ_k β_k x_ik  +  λ (−R_i).
        λ=0 recovers RUM, λ=1 recovers classic RRM; estimating λ lets the data say
        how regret-driven the timing decision is.
An optional scale μ ('mu_rrm') multiplies the regret.

X is a dense (n_cases, n_alts, n_attrs) array; `chosen` a (n_cases,) alt index;
`avail` an optional (n_cases, n_alts) 0/1 mask. Pure numpy/scipy — testable in
the base env, no larch/jax.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from scipy import optimize as _opt
    _SCIPY_OK = True
except Exception:                                    # pragma: no cover
    _SCIPY_OK = False

logger = logging.getLogger(__name__)

_VALID_RULES = ('rum', 'rrm', 'prrm', 'mu_rrm', 'hybrid')


def _softplus(z):
    # numerically stable ln(1 + exp(z))
    return np.logaddexp(0.0, z)


def prrm_transform(X: np.ndarray, attr_sign: np.ndarray) -> np.ndarray:
    """P-RRM attribute transform x̂_ik = Σ_{j≠i} g(x_jk − x_ik), where for a
    'more-is-worse' attribute (sign −1, e.g. cost/time) g = max(0, x_jk − x_ik)
    and for 'more-is-better' (sign +1) g = min(0, x_jk − x_ik). Returns the same
    shape as X; a P-RRM fit is then an MNL on −x̂."""
    n, a, k = X.shape
    xi = X[:, :, None, :]           # (n, i, 1, k)
    xj = X[:, None, :, :]           # (n, 1, j, k)
    diff = xj - xi                  # (n, i, j, k)
    eye = np.eye(a, dtype=bool)[None, :, :, None]
    diff = np.where(eye, 0.0, diff)
    pos = np.maximum(diff, 0.0).sum(axis=2)   # more-is-worse contribution
    neg = np.minimum(diff, 0.0).sum(axis=2)   # more-is-better contribution
    s = np.asarray(attr_sign, dtype=float)[None, None, :]
    return np.where(s < 0, pos, neg)          # (n, i, k)


@dataclass
class RegretChoiceModel:
    rule: str = 'rrm'
    attr_sign: Optional[np.ndarray] = None       # +1 more-is-better, −1 more-is-worse
    l2: float = 0.0
    fit_ascs: bool = True
    beta_: Optional[np.ndarray] = None
    asc_: Optional[np.ndarray] = None            # length n_alts, asc_[0]=0 reference
    mu_: float = 1.0
    lam_: float = 1.0                                # hybrid RUM-RRM weight (0=RUM,1=RRM)
    names_: list = field(default_factory=list)
    n_: int = 0
    k_: int = 0
    ll_: float = np.nan

    def _classic_regret(self, X, beta):
        n, a, kk = X.shape
        xi = X[:, :, None, :]
        xj = X[:, None, :, :]
        d = (xj - xi) * beta[None, None, None, :]
        sp = _softplus(d)                            # (n,i,j,k)
        eye = np.eye(a, dtype=bool)[None, :, :, None]
        sp = np.where(eye, 0.0, sp)
        return sp.sum(axis=(2, 3))                   # R (n,i)

    # -- systematic values V_i (pre-softmax) for a parameter vector -----------
    def _utilities(self, X, asc, beta, mu, lam=1.0, xhat=None):
        if self.rule == 'rum':
            V = X @ beta
        elif self.rule == 'prrm':
            V = -(xhat @ beta)                        # regret is linear; V = −R
        elif self.rule == 'hybrid':
            # convex RUM-RRM blend with shared β: (1−λ)·V_rum + λ·(−R)
            Vrum = X @ beta
            R = self._classic_regret(X, beta)
            V = (1.0 - lam) * Vrum + lam * (-R)
        else:  # 'rrm' / 'mu_rrm' classic
            V = -mu * self._classic_regret(X, beta)
        return V + asc[None, :]

    def _neg_ll(self, theta, X, y, avail, xhat):
        n, a, kk = X.shape
        asc = np.zeros(a)
        off = 0
        if self.fit_ascs:
            asc[1:] = theta[:a - 1]
            off = a - 1
        beta = theta[off:off + kk]
        mu, lam = 1.0, 1.0
        if self.rule == 'mu_rrm':
            mu = np.exp(np.clip(theta[off + kk], -3, 3))
        elif self.rule == 'hybrid':
            lam = 1.0 / (1.0 + np.exp(-theta[off + kk]))   # logistic → (0,1)
        V = self._utilities(X, asc, beta, mu, lam=lam, xhat=xhat)
        V = np.where(avail > 0, V, -1e20)
        V = V - V.max(axis=1, keepdims=True)
        eV = np.exp(V) * (avail > 0)
        denom = eV.sum(axis=1)
        p_chosen = eV[np.arange(n), y] / np.clip(denom, 1e-300, None)
        ll = np.sum(np.log(np.clip(p_chosen, 1e-300, None)))
        pen = self.l2 * np.sum(beta ** 2)
        return -(ll - pen)

    def fit(self, X, chosen, avail=None, names: Optional[Sequence[str]] = None):
        X = np.asarray(X, dtype=float)
        n, a, kk = X.shape
        y = np.asarray(chosen, dtype=int)
        avail = np.ones((n, a)) if avail is None else np.asarray(avail, dtype=float)
        if self.attr_sign is None:
            self.attr_sign = -np.ones(kk)            # default: more-is-worse
        xhat = prrm_transform(X, self.attr_sign) if self.rule == 'prrm' else None
        self.names_ = list(names) if names is not None else [f'x{k}' for k in range(kk)]

        _extra = 1 if self.rule in ('mu_rrm', 'hybrid') else 0
        n_par = (a - 1 if self.fit_ascs else 0) + kk + _extra
        theta0 = np.zeros(n_par)
        if not _SCIPY_OK:
            logger.warning("regret_choice: scipy unavailable — returning zero fit")
            self.beta_ = np.zeros(kk); self.asc_ = np.zeros(a); self.n_ = n; self.k_ = n_par
            return self
        res = _opt.minimize(self._neg_ll, theta0, args=(X, y, avail, xhat),
                            method='BFGS', options={'maxiter': 400, 'gtol': 1e-6})
        theta = res.x
        asc = np.zeros(a); off = 0
        if self.fit_ascs:
            asc[1:] = theta[:a - 1]; off = a - 1
        self.asc_ = asc
        self.beta_ = theta[off:off + kk]
        if self.rule == 'mu_rrm':
            self.mu_ = float(np.exp(np.clip(theta[off + kk], -3, 3)))
        elif self.rule == 'hybrid':
            self.lam_ = float(1.0 / (1.0 + np.exp(-theta[off + kk])))
        self.n_, self.k_ = n, n_par
        self.ll_ = float(-res.fun)
        return self

    def predict_proba(self, X, avail=None):
        X = np.asarray(X, dtype=float)
        n, a, kk = X.shape
        avail = np.ones((n, a)) if avail is None else np.asarray(avail, dtype=float)
        xhat = prrm_transform(X, self.attr_sign) if self.rule == 'prrm' else None
        V = self._utilities(X, self.asc_, self.beta_, self.mu_, lam=self.lam_, xhat=xhat)
        V = np.where(avail > 0, V, -1e20)
        V = V - V.max(axis=1, keepdims=True)
        eV = np.exp(V) * (avail > 0)
        return eV / np.clip(eV.sum(axis=1, keepdims=True), 1e-300, None)

    @property
    def aic(self):
        return 2 * self.k_ - 2 * self.ll_ if np.isfinite(self.ll_) else np.nan

    @property
    def bic(self):
        return self.k_ * np.log(max(self.n_, 1)) - 2 * self.ll_ if np.isfinite(self.ll_) else np.nan


def fit_choice(X, chosen, avail=None, rule='rrm', attr_sign=None, names=None,
               fit_ascs=True, l2=0.0) -> RegretChoiceModel:
    """Convenience: fit one decision rule and return the model (with .bic/.aic)."""
    return RegretChoiceModel(rule=rule, attr_sign=attr_sign, l2=l2,
                             fit_ascs=fit_ascs).fit(X, chosen, avail, names)


def search_decision_rule(X, chosen, avail=None, rules=('rum', 'rrm', 'prrm'),
                         attr_sign=None, names=None, fit_ascs=True, l2=0.0):
    """Fit each candidate decision rule and return (best_model, {rule: bic}).
    This is the searchable-decision primitive: pick the rule with the best BIC —
    RUM vs RRM becomes a data-driven choice, POLARIS-style."""
    fits, bics = {}, {}
    for r in rules:
        if r not in _VALID_RULES:
            continue
        try:
            m = fit_choice(X, chosen, avail, rule=r, attr_sign=attr_sign,
                           names=names, fit_ascs=fit_ascs, l2=l2)
            fits[r] = m
            bics[r] = m.bic
        except Exception as e:
            logger.warning(f"search_decision_rule: rule '{r}' failed ({e})")
            bics[r] = np.inf
    if not fits:
        return None, bics
    best = min(fits, key=lambda r: bics.get(r, np.inf))
    logger.info(f"decision-rule search: BICs={{{', '.join(f'{r}:{bics[r]:.1f}' for r in fits)}}} "
                f"→ best={best}")
    return fits[best], bics


if __name__ == '__main__':
    # Self-check: simulate an RRM data-generating process and confirm the RRM fit
    # beats RUM on BIC (and recovers the sign), while on a RUM DGP, RUM wins.
    rng = np.random.default_rng(0)
    n, a, k = 4000, 4, 2          # 4 alts (e.g. modes), 2 attrs (time, cost)
    X = rng.normal(size=(n, a, k))
    beta_true = np.array([-1.2, -0.8])
    sign = np.array([-1.0, -1.0])  # both more-is-worse

    def _draw(rule, lam=1.0):
        m = RegretChoiceModel(rule=rule, attr_sign=sign)
        m.asc_ = np.zeros(a); m.beta_ = beta_true; m.mu_ = 1.0; m.lam_ = lam
        P = m.predict_proba(X)
        u = rng.random(n)
        return (P.cumsum(1) > u[:, None]).argmax(1)

    for dgp in ('rrm', 'rum'):
        y = _draw(dgp)
        best, bics = search_decision_rule(X, y, rules=('rum', 'rrm', 'prrm', 'hybrid'),
                                          attr_sign=sign, names=['time', 'cost'])
        print(f"DGP={dgp:4s} → chosen rule={best.rule:6s} "
              f"beta={np.round(best.beta_, 2)}  "
              f"BIC(rum={bics['rum']:.0f}, rrm={bics['rrm']:.0f}, prrm={bics['prrm']:.0f}, "
              f"hybrid={bics['hybrid']:.0f})")

    # Hybrid DGP (λ=0.6): the fitted hybrid should recover λ≈0.6 and win/tie the
    # pure rules — the POLARIS activity-timing case.
    yh = _draw('hybrid', lam=0.6)
    mh = fit_choice(X, yh, rule='hybrid', attr_sign=sign, names=['time', 'cost'])
    besth, bicsh = search_decision_rule(X, yh, rules=('rum', 'rrm', 'hybrid'),
                                        attr_sign=sign, names=['time', 'cost'])
    print(f"DGP=hybrid(λ=0.6) → fitted λ={mh.lam_:.2f}  chosen rule={besth.rule} "
          f"BIC(rum={bicsh['rum']:.0f}, rrm={bicsh['rrm']:.0f}, hybrid={bicsh['hybrid']:.0f})")
