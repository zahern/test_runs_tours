"""Choice-set-conscious estimation for destination choice (371 zones).

Three linked pieces (no full 371-alternative denominator anywhere):

1. COMPETING ALTERNATIVES via Bayesian adaptive sampling from HHTS.
   The universal set is cut to a consideration set C_n per case. Sampling
   weights blend an HHTS-learned prior (empirical zone choice shares per
   segment, Dirichlet-smoothed) with the current model's own utilities
   (exploitation), mixed by alpha and tempered by tau. Alpha/tau adapt to
   hold the effective sample size of the sampled denominator near target.

2. GIBBS embedding with McFadden's correction.
   Alternate (i) beta | C  (random-walk Metropolis, explicit adaptive
   Gaussian proposals) with (ii) C | beta (refresh consideration sets from
   the adaptive sampler). Every likelihood uses McFadden offsets
   o_k = log pi(C|k) so unequal sampling probabilities are corrected.
   Valid as K -> full set; standard sampling-of-alternatives practice.

3. ADAPTIVE STOCHASTIC-GRADIENT MLE (second method).
   Maximum likelihood with Monte-Carlo-approximated gradients: minibatches
   over cases x sampled alternatives, Adam steps, K (alt count) adapted to
   gradient noise, SEs from OPG information on a final large sample —
   a Monte Carlo approximation of the full-choice-set costs.

Data layout (vectorised): padded arrays X (N,K,d), chosen (N,),
offsets (N,K), mask (N,K bool). Ragged lists are padded internally.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd


# ===========================================================================
# Padded layout + vectorised sampled-denominator likelihood / gradient
# ===========================================================================

def pad_choice_data(Xlist, yidx, offsets=None):
    """Ragged per-case sampled sets -> padded (N,K,d) arrays + mask."""
    N = len(Xlist)
    K = max(X.shape[0] for X in Xlist)
    d = Xlist[0].shape[1]
    X = np.zeros((N, K, d))
    off = np.zeros((N, K))
    mask = np.zeros((N, K), dtype=bool)
    y = np.asarray(yidx, dtype=int)
    for n, Xi in enumerate(Xlist):
        j = Xi.shape[0]
        X[n, :j] = Xi
        mask[n, :j] = True
        if offsets is not None and offsets[n] is not None:
            off[n, :j] = np.asarray(offsets[n], dtype=float)
    return X, y, off, mask


def loglik_mat(beta, X, y, off, mask):
    """Sum log P(chosen); denominators over SAMPLED alts only."""
    V = X @ beta + np.where(mask, off, -np.inf)
    V = np.where(mask, V, -np.inf)
    m = np.max(V, axis=1)
    lse = m + np.log(np.sum(np.exp(V - m[:, None]), axis=1))
    return float(np.sum(V[np.arange(len(y)), y] - lse))


def grad_mat(beta, X, y, off, mask):
    """Analytic gradient of loglik_mat (for adaptive SGD)."""
    V = np.where(mask, X @ beta + off, -np.inf)
    m = np.max(V, axis=1, keepdims=True)
    P = np.exp(V - m)
    P = P / P.sum(axis=1, keepdims=True)
    chosen = np.zeros_like(P)
    chosen[np.arange(len(y)), y] = 1.0
    return ((chosen - P)[..., None] * np.where(mask[..., None], X, 0.0)).sum(axis=(0, 1))


# ===========================================================================
# Proposal distributions (Metropolis-Hastings)
# ===========================================================================

class GaussianRandomWalkProposal:
    """Adaptive Gaussian random-walk proposal, q(θ*|θ) = N(θ, λΣ_t).

    Haario et al. (2001): recursive mean/covariance (Welford), optimal
    scaling λ = 2.38²/d, regularised with eps·I (never singular), no
    adaptation before `adapt_after` draws. Symmetric ⇒ Hastings ratio = 1.
    """

    def __init__(self, dim, init_cov=None, adapt_after=500,
                 scaling=None, eps=1e-6, seed=0):
        self.dim = int(dim)
        self.Sigma = np.array(init_cov, dtype=float) if init_cov is not None \
            else np.eye(self.dim)
        self.lam = float(scaling) if scaling else (2.38 ** 2 / max(self.dim, 1))
        self.adapt_after = int(adapt_after)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self._n = 0
        self._mean = None
        self._M2 = None

    def propose(self, theta):
        L = np.linalg.cholesky(self.lam * self.Sigma
                               + self.eps * np.eye(self.dim))
        return theta + L @ self.rng.standard_normal(self.dim)

    def adapt(self, theta):
        self._n += 1
        x = np.asarray(theta, dtype=float)
        if self._mean is None:
            self._mean = x.copy()
            self._M2 = np.zeros((self.dim, self.dim))
            return
        n = self._n
        delta = x - self._mean
        self._mean = self._mean + delta / n
        self._M2 = self._M2 + np.outer(delta, x - self._mean)
        if n > self.adapt_after:
            self.Sigma = self._M2 / max(n - 1, 1)

    @property
    def cov(self):
        return self.lam * self.Sigma + self.eps * np.eye(self.dim)


class BlockRandomWalkProposal:
    """Block-wise Metropolis: one adaptive Gaussian RWMH proposal per block
    (e.g. slopes | cut-points/scales). `blocks` = list of index arrays."""

    def __init__(self, blocks, init_covs=None, adapt_after=500,
                 scaling=None, eps=1e-6, seed=0):
        self.blocks = [np.asarray(b, dtype=int) for b in blocks]
        self.props = []
        for i, b in enumerate(self.blocks):
            ic = None if init_covs is None else init_covs[i]
            self.props.append(
                GaussianRandomWalkProposal(len(b), init_cov=ic,
                                           adapt_after=adapt_after,
                                           scaling=scaling, eps=eps,
                                           seed=seed + 977 * (i + 1)))

    def step(self, theta, logpost):
        theta = theta.copy()
        cur = float(logpost(theta))
        accepts = []
        for b, prop in zip(self.blocks, self.props):
            cand = theta.copy()
            cand[b] = prop.propose(theta[b])
            new = float(logpost(cand))
            if np.log(np.random.uniform()) < new - cur:
                theta, cur = cand, new
                accepts.append(1)
                prop.adapt(theta[b])
            else:
                accepts.append(0)
        return theta, accepts, len(self.blocks)


# ===========================================================================
# 1. HHTS-learned competing-alternative prior + adaptive sampler
# ===========================================================================

class HHTSCompetingPrior:
    """Bayesian prior over consideration membership learned from HHTS.

    Per segment: Dirichlet-smoothed empirical zone choice shares
    q0(zone|segment) = (count + a) / (total + a·Z). Zones never chosen in
    HHTS keep mass a (strictly positive ⇒ every zone stays reachable).
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = float(smoothing)
        self.shares_ = {}
        self.zones_ = None

    def fit(self, choices, segment_col='segment', zone_col='zone'):
        """choices: DataFrame with one row per observed HHTS choice."""
        zones = np.sort(choices[zone_col].unique())
        self.zones_ = zones
        zidx = {z: i for i, z in enumerate(zones)}
        for seg, g in choices.groupby(segment_col):
            c = np.zeros(len(zones))
            for z in g[zone_col].to_numpy():
                c[zidx[z]] += 1
            self.shares_[seg] = (c + self.smoothing) / (c.sum() + self.smoothing * len(zones))
        return self

    def q0(self, segment, zone_ids):
        """Prior sampling weights aligned to `zone_ids` (universal order)."""
        q = self.shares_.get(segment)
        if q is None:
            return np.full(len(zone_ids), 1.0 / len(zone_ids))
        zidx = {z: i for i, z in enumerate(self.zones_)}
        w = np.array([q[zidx[z]] if z in zidx else self.smoothing for z in zone_ids],
                     dtype=float)
        return w / w.sum()


class AdaptiveCompetingSampler:
    """Cut the universal set to competing alternatives per case.

    Draw weights blend the HHTS prior with the model's current utilities:
        q ∝ α·q0 + (1-α)·softmax(V(β_cur)/τ).
    α (prior weight) and τ (temperature) adapt to hold the denominator ESS
    near `ess_target`: if sampled sets are degenerate (one alt dominates),
    raise τ / α; if flat and wasteful, lower them. Bounds keep both away
    from 0/∞ so every zone stays samplable (McFadden correction stays valid).

    draw() returns (alts_idx, offsets) with McFadden offsets
    o_k = Σ_{a∈C,a≠k} log q(a) for each member k of C.
    """

    def __init__(self, zone_ids, q0, alpha=0.5, tau=1.0, ess_target=0.5,
                 alpha_bounds=(0.05, 0.95), tau_bounds=(0.25, 4.0),
                 seed=0):
        self.zone_ids = np.asarray(zone_ids)
        self.q0 = np.asarray(q0, dtype=float)
        self.q0 = self.q0 / self.q0.sum()
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.ess_target = float(ess_target)
        self.alpha_bounds = alpha_bounds
        self.tau_bounds = tau_bounds
        self.rng = np.random.default_rng(seed)

    def weights(self, V):
        util = np.exp((V - V.max()) / max(self.tau, 1e-9))
        qu = util / util.sum()
        return self.alpha * self.q0 + (1.0 - self.alpha) * qu

    def draw(self, V, chosen_pos, K, zone_X_fn=None):
        """Sample a consideration set containing the chosen alt.

        V: current utilities over the UNIVERSAL set (for weighting only —
        never a 371-denominator in the likelihood). chosen_pos: index of the
        chosen alt. Returns (positions, offsets); caller builds X rows via
        zone_X_fn or slices a Deze matrix.
        """
        q = self.weights(V)
        others = self.rng.choice(len(q), size=K - 1, replace=True, p=q)
        pos = np.concatenate([[chosen_pos], others])
        # McFadden offsets under iid draws from q (chosen always included):
        # o_k = Σ_{a∈C,a≠k} log q(a).
        logq = np.log(np.maximum(q[pos], 1e-300))
        offs = logq.sum() - logq
        # ESS of this set under current V (for adaptation signal)
        v = V[pos] - V[pos].max()
        p = np.exp(v)
        p = p / p.sum()
        ess = 1.0 / float((p ** 2).sum()) / K
        return pos, offs, float(ess)

    def adapt(self, ess_mean):
        """Nudge α, τ toward the ESS target (bounded)."""
        if ess_mean < self.ess_target:
            self.tau = min(self.tau_bounds[1], self.tau * 1.1)
            self.alpha = min(self.alpha_bounds[1], self.alpha + 0.02)
        else:
            self.tau = max(self.tau_bounds[0], self.tau / 1.05)
            self.alpha = max(self.alpha_bounds[0], self.alpha - 0.01)


# ===========================================================================
# Diagnostics
# ===========================================================================

def effective_sample_size(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    out = []
    for c in range(x.shape[1]):
        v = x[:, c] - x[:, c].mean()
        n = len(v)
        denom = float(v @ v)
        if denom <= 0:
            out.append(float(n))
            continue
        rho_sum, t = 0.0, 1
        while t + 1 < n:
            g1 = float(v[:n - t] @ v[t:]) / denom
            g2 = float(v[:n - t - 1] @ v[t + 1:]) / denom if t + 1 < n else 0.0
            if g1 + g2 < 0:
                break
            rho_sum += g1 + g2
            t += 2
        out.append(float(n) / max(1e-9, 1.0 + 2.0 * rho_sum))
    return out[0] if len(out) == 1 else np.array(out)


def r_hat(chains):
    C = np.asarray(chains, dtype=float)
    m, n = C.shape[0], C.shape[1]
    if m < 2:
        return np.full(C.shape[2], np.nan)
    within = C.var(axis=1, ddof=1).mean(axis=0)
    between = n * C.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (n - 1.0) / n * within + between / n
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(var_hat / np.where(within > 0, within, np.nan))


# ===========================================================================
# Core RWMH driver (vectorised, padded sampled sets)
# ===========================================================================

def _prep_prior(d, prior_mean, prior_sd, ridge_extra):
    mean = np.zeros(d) if prior_mean is None else np.asarray(prior_mean, float)
    Pc = np.eye(d) * float(prior_sd) ** 2
    if ridge_extra:
        Pc = np.linalg.inv(np.linalg.inv(Pc) + np.eye(d) * float(ridge_extra))
    chol = np.linalg.cholesky(Pc)
    return mean, Pc, np.linalg.inv(chol)


def estimate_mnl_mh(Xlist, yidx, offsets=None, prior_mean=None,
                    prior_sd=5.0, fixed_mask=None, init=None,
                    n_burn=2000, n_draw=5000, thin=5,
                    proposal=None, blocks=None, seed=0, verbose=True,
                    ridge_extra=0.0):
    """Random-walk Metropolis for sampled-alternative MNL (vectorised).

    Denominators are over SAMPLED alts only — the universal set is never
    formed. Returns posterior mean/cov/draws/accept_rate/ess.
    """
    rng = np.random.default_rng(seed)
    X, y, off, mask = pad_choice_data(Xlist, yidx, offsets)
    d = X.shape[2]
    free = np.ones(d, dtype=bool)
    if fixed_mask is not None:
        free = ~np.asarray(fixed_mask, dtype=bool)
    fidx = np.where(free)[0]
    dfree = int(free.sum())
    if dfree == 0:
        raise ValueError("no free parameters (all holdfast)")
    Xf = X[:, :, fidx]
    mf, _, chol_inv = _prep_prior(dfree, None if prior_mean is None else
                                  np.asarray(prior_mean, float)[fidx],
                                  prior_sd, ridge_extra)
    Cf = np.eye(dfree) * float(prior_sd) ** 2
    logdet = np.linalg.slogdet(Cf)[1]

    def logpost(th):
        V = np.where(mask, Xf @ th + off, -np.inf)
        V = V.reshape(Xf.shape[0], Xf.shape[1])
        m = np.max(V, axis=1)
        ll = float(np.sum(V[np.arange(len(y)), y] - m
                          - np.log(np.sum(np.exp(V - m[:, None]), axis=1))))
        z = chol_inv @ (th - mf)
        return ll - 0.5 * float(z @ z) - 0.5 * logdet

    start = np.zeros(dfree) if init is None else np.asarray(init, float)[fidx]
    if proposal is None:
        if blocks is not None:
            pos = {g: k for k, g in enumerate(fidx)}
            blks = [np.array(sorted(pos[g] for g in b if g in pos), dtype=int)
                    for b in blocks]
            proposal = BlockRandomWalkProposal([b for b in blks if len(b)],
                                               seed=seed + 13)
        else:
            proposal = GaussianRandomWalkProposal(dfree, seed=seed + 13)

    theta = start.copy()
    cur = float(logpost(theta))
    draws, n_acc, n_prop = [], 0, 0
    total = n_burn + n_draw * thin
    report = max(total // 10, 1)
    for it in range(1, total + 1):
        if isinstance(proposal, BlockRandomWalkProposal):
            theta, accs, _ = proposal.step(theta, logpost)
            n_acc += sum(accs)
            n_prop += len(accs)
        else:
            cand = proposal.propose(theta)
            new = float(logpost(cand))
            n_prop += 1
            if math.log(rng.uniform()) < new - cur:
                theta, cur = cand, new
                n_acc += 1
            if it > proposal.adapt_after:
                proposal.adapt(theta)
        if it > n_burn and (it - n_burn) % thin == 0:
            draws.append(theta.copy())
        if verbose and it % report == 0:
            print(f"  MH {it}/{total} acc={n_acc / max(n_prop, 1):.3f}")
    draws = np.array(draws)
    return {
        'posterior_mean': draws.mean(axis=0),
        'posterior_cov': np.cov(draws.T),
        'draws': draws,
        'accept_rate': n_acc / max(n_prop, 1),
        'ess': effective_sample_size(draws),
        'free_idx': fidx,
        'n_cases': len(Xlist),
        'n_free': dfree,
    }


# ===========================================================================
# 2. Gibbs: alternate beta | C  with  C | beta  (McFadden-corrected)
# ===========================================================================

def gibbs_mnl(Z, y_star, sampler, prior_sd=5.0, fixed_mask=None,
              K=20, n_outer=40, n_inner=200, n_burn_outer=10,
              refresh_every=1, init=None, seed=0, verbose=True):
    """Gibbs sampler embedding consideration-set refreshes.

    Z        : (N_cases, N_zones_univ, d) universal attributes per case
               (weighting/sampling only — never a likelihood denominator)
    y_star   : chosen universal-alt index per case
    sampler  : AdaptiveCompetingSampler (HHTS prior + current-utility blend)
    Each outer sweep: (i) refresh C_n | beta via sampler (McFadden offsets
    from the actual q used); (ii) n_inner RWMH steps for beta | C.
    Returns draws of beta + set-history ESS trace.
    """
    rng = np.random.default_rng(seed)
    N, Zu, d = Z.shape
    free = np.ones(d, dtype=bool)
    if fixed_mask is not None:
        free = ~np.asarray(fixed_mask, dtype=bool)
    fidx = np.where(free)[0]
    Zf = Z[:, :, fidx]
    beta = np.zeros(int(free.sum())) if init is None else np.asarray(init, float)
    prop = GaussianRandomWalkProposal(int(free.sum()), seed=seed + 5)
    mf, _, chol_inv = _prep_prior(int(free.sum()), None, prior_sd, 0.0)
    Cf = np.eye(int(free.sum())) * float(prior_sd) ** 2
    logdet = np.linalg.slogdet(Cf)[1]

    draws, ess_trace, acc_trace = [], [], []
    n_acc = n_prop = 0
    Xc = y_c = oc = mk = None
    for outer in range(n_outer):
        if outer % refresh_every == 0:
            V = Zf @ beta  # (N, Zu): weighting only, never a denominator
            Xs, ys, os, ess = [], [], [], []
            for n in range(N):
                pos, offs, e = sampler.draw(V[n], y_star[n], K)
                Xs.append(Zf[n][pos])
                # chosen is pos[0] by construction
                ys.append(0)
                os.append(offs)
                ess.append(e)
            Xc = np.array(Xs)
            y_c = np.array(ys)
            oc = np.array(os)
            mk = np.ones((N, K), dtype=bool)
            sampler.adapt(float(np.mean(ess)))
            ess_trace.append(float(np.mean(ess)))
        mask = mk

        def logpost(th):
            Vv = np.where(mask, Xc @ th + oc, -np.inf)
            m = np.max(Vv, axis=1)
            ll = float(np.sum(Vv[np.arange(N), y_c] - m
                              - np.log(np.sum(np.exp(Vv - m[:, None]), axis=1))))
            z = chol_inv @ (th - mf)
            return ll - 0.5 * float(z @ z) - 0.5 * logdet

        cur = float(logpost(beta))
        for _ in range(n_inner):
            cand = prop.propose(beta)
            new = float(logpost(cand))
            n_prop += 1
            if math.log(rng.uniform()) < new - cur:
                beta, cur = cand, new
                n_acc += 1
            prop.adapt(beta)
        if outer >= n_burn_outer:
            draws.append(beta.copy())
        acc_trace.append(n_acc / max(n_prop, 1))
        if verbose and (outer + 1) % max(n_outer // 8, 1) == 0:
            print(f"  Gibbs outer {outer + 1}/{n_outer} acc={acc_trace[-1]:.3f} "
                  f"setESS={ess_trace[-1]:.3f} alpha={sampler.alpha:.2f}")
    draws = np.array(draws)
    return {'draws': draws, 'posterior_mean': draws.mean(axis=0),
            'posterior_cov': np.cov(draws.T),
            'accept_rate': n_acc / max(n_prop, 1),
            'ess': effective_sample_size(draws),
            'set_ess_trace': np.array(ess_trace),
            'free_idx': fidx}


# ===========================================================================
# 3. Adaptive stochastic-gradient MLE (Monte Carlo full-set approximation)
# ===========================================================================

def estimate_mnl_sgd(Xlist, yidx, offsets=None, fixed_mask=None, init=None,
                     lr=0.05, batch=256, epochs=40, K_adapt=True,
                     ess_target=0.5, seed=0, verbose=True,
                     sampler=None, Z=None, y_star=None):
    """MLE with Monte-Carlo-approximated gradients (Adam, adaptive K).

    Minibatches over cases; each case's denominator is Monte-Carlo over its
    sampled set (McFadden offsets included). If `sampler` + universal (Z,
    y_star) are given, consideration sets are re-drawn adaptively and K
    grows when set-ESS falls below target (shrinks when comfortably above,
    bounded [8, 64]) — adaptive stochastic-gradient estimation of the
    full-choice-set costs. SEs: OPG information on a final large sample.
    """
    rng = np.random.default_rng(seed)
    X, y, off, mask = pad_choice_data(Xlist, yidx, offsets)
    N, _, d = X.shape
    free = np.ones(d, dtype=bool)
    if fixed_mask is not None:
        free = ~np.asarray(fixed_mask, dtype=bool)
    fidx = np.where(free)[0]
    th = np.zeros(int(free.sum())) if init is None else np.asarray(init, float)[fidx]
    m1 = np.zeros_like(th)
    m2 = np.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    Kcur = int(X.shape[1])
    Xn, yn, offn, maskn = X[:, :, fidx], y, off, mask
    use_adaptive_sets = sampler is not None and Z is not None and y_star is not None
    Zf = Z[:, :, fidx] if use_adaptive_sets else None

    for ep in range(epochs):
        if use_adaptive_sets and K_adapt:
            V = Zf @ th
            Xs, ys, os, ess = [], [], [], []
            for n in range(N):
                pos, o, e = sampler.draw(V[n], y_star[n], Kcur)
                Xs.append(Zf[pos])
                ys.append(0)
                os.append(o)
                ess.append(e)
            Xn, yn, offn = np.array(Xs), np.array(ys), np.array(os)
            maskn = np.ones((N, Kcur), dtype=bool)
            em = float(np.mean(ess))
            sampler.adapt(em)
            if em < ess_target and Kcur < 64:
                Kcur = min(64, Kcur + 4)
            elif em > ess_target + 0.2 and Kcur > 8:
                Kcur = max(8, Kcur - 2)
        perm = rng.permutation(N)
        for s in range(0, N, batch):
            idx = perm[s:s + batch]
            step += 1
            g = grad_mat(th, Xn[idx], yn[idx], offn[idx], maskn[idx])
            m1 = b1 * m1 + (1 - b1) * g
            m2 = b2 * m2 + (1 - b2) * g * g
            m1h, m2h = m1 / (1 - b1 ** step), m2 / (1 - b2 ** step)
            th = th + lr * m1h / (np.sqrt(m2h) + eps)
        if verbose and (ep + 1) % max(epochs // 8, 1) == 0:
            ll = loglik_mat(th, [Xn[i] for i in range(N)], yn,
                            [offn[i] for i in range(N)],
                            [maskn[i] for i in range(N)])
            print(f"  SGD epoch {ep + 1}/{epochs} ll={ll:.1f} K={Kcur}")
    # OPG information on the final sample -> approx SEs
    G = []
    for s in range(0, N, max(batch, 1)):
        idx = np.arange(s, min(s + batch, N))
        V = np.where(maskn[idx], Xn[idx] @ th + offn[idx], -np.inf)
        m_ = np.max(V, axis=1, keepdims=True)
        P = np.exp(V - m_)
        P = P / P.sum(axis=1, keepdims=True)
        for ii, nn in enumerate(idx):
            g_n = ((np.eye(Kcur)[yn[nn]] - P[ii])[:, None]
                   * np.where(maskn[nn][:, None], Xn[nn], 0.0)).sum(axis=0)
            G.append(g_n)
    G = np.array(G)
    info = G.T @ G / max(N, 1) + np.eye(len(th)) * 1e-8
    cov = np.linalg.pinv(info) / max(N, 1)
    return {'coef': th, 'cov': cov,
            'std_err': np.sqrt(np.clip(np.diag(cov), 0, None)),
            'free_idx': fidx, 'K_final': Kcur}


# ===========================================================================
# Long-frame adapter + method dispatch
# ===========================================================================

def from_long_frame(df, case_col='case_id', alt_col='alt_id',
                    chosen_col='chosen', correction_col='log_correction',
                    feature_cols=None):
    """Build (Xlist, yidx, offsets) from a long sampled-alternative frame.

    Each case contributes its SAMPLED alternatives only — the full universal
    denominator is never formed. Exactly one chosen row per case.
    """
    if feature_cols is None:
        drop = {case_col, alt_col, chosen_col, correction_col}
        feature_cols = [c for c in df.columns if c not in drop]
    Xlist, yidx, offsets = [], [], []
    for _, g in df.groupby(case_col, sort=False):
        Xlist.append(g[feature_cols].to_numpy(dtype=float))
        ch = g[chosen_col].to_numpy()
        yidx.append(int(np.flatnonzero(ch)[0]))
        offsets.append(g[correction_col].to_numpy(dtype=float)
                       if correction_col in g.columns else None)
    return Xlist, yidx, offsets, list(feature_cols)


def estimate_dest_choice(frame=None, Xlist=None, yidx=None, offsets=None,
                         method=None, feature_cols=None, **kw):
    """Dispatch: method in {'mh', 'gibbs', 'sgd'} (env STAGE5_DEST_ESTIMATOR).

    'mh'    : RWMH on fixed sampled sets (McFadden offsets in frame).
    'gibbs' : Gibbs with adaptive competing-set refresh (needs Z, y_star,
              sampler in kw).
    'sgd'   : adaptive stochastic-gradient MLE.
    Returns posterior/coef table-compatible dicts (value/std_err/t_stat).
    """
    method = (method or os.environ.get('STAGE5_DEST_ESTIMATOR', 'mh')).lower()
    if frame is not None:
        Xlist, yidx, offsets, names = from_long_frame(
            frame, feature_cols=feature_cols)
    elif Xlist is not None:
        names = kw.pop('names', [f'x{i}' for i in range(Xlist[0].shape[1])])
    else:
        names = kw.pop('names')
    if method == 'sgd':
        out = estimate_mnl_sgd(Xlist, yidx, offsets, **kw)
        mu, sd = out['coef'], out['std_err']
        return {'method': 'sgd', 'names': names, 'value': mu, 'std_err': sd,
                't_stat': np.divide(mu, sd, out=np.full_like(mu, np.nan),
                                    where=sd > 0), 'raw': out}
    if method == 'gibbs':
        out = gibbs_mnl(kw.pop('Z'), kw.pop('y_star'), kw.pop('sampler'),
                        **kw)
    else:
        out = estimate_mnl_mh(Xlist, yidx, offsets, **kw)
    mu = np.asarray(out['posterior_mean'])
    sd = np.sqrt(np.clip(np.diag(out['posterior_cov']), 0, None))
    return {'method': method, 'names': names, 'value': mu, 'std_err': sd,
            't_stat': np.divide(mu, sd, out=np.full_like(mu, np.nan),
                                where=sd > 0), 'raw': out}


def posterior_table(fit, names):
    """Posterior summary DataFrame (works for mh/gibbs raw fits)."""
    mu = np.asarray(fit['posterior_mean'])
    sd = np.sqrt(np.clip(np.diag(fit['posterior_cov']), 0, None))
    lo, hi = np.quantile(fit['draws'], [0.025, 0.975], axis=0)
    ess = np.asarray(fit['ess'], dtype=float).ravel()
    return pd.DataFrame({
        'param': list(names),
        'value': mu,
        'std_err': sd,
        't_stat': np.divide(mu, sd, out=np.full_like(mu, np.nan),
                            where=sd > 0),
        'ci_lo': lo,
        'ci_hi': hi,
        'ess': ess[0] if ess.size == 1 else ess,
    })
