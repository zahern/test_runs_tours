"""Binomial maximum likelihood for the speeding propensity.

WHY THIS EXISTS
---------------
The response is a count of speeding TRIPS, so its natural model is

    k_i ~ Binomial(n_i, p_i),    logit(p_i) = x_i'beta

where n_i is the number of trips that actually got a speed reading. The
metacountregressor package implements only Poisson and negative binomial,
so this likelihood was unavailable there. It is also the model that does
not need the flow projection at all: n_i is measured, not modelled, which
is exactly what the exposure LRT has been complaining about.

Kept separate from the count path on purpose. Fitted by Newton/IRLS on
the exact binomial loglik, which is verifiable against known coefficients
and a brute-force grid -- a fit you can check is worth more here than one
that merely converges.

ESTIMATION
----------
    loglik(b) = sum_i [ k_i * eta_i - n_i * log(1 + exp(eta_i)) ]
    score     = X' (k - n*p)
    info      = X' diag(n*p*(1-p)) X

Newton step: b += (X'WX)^-1 X'(k - n*p). With a ridge for conditioning.
Standard errors are the inverse observed information, matching the
maximum-likelihood convention (asymptotically equivalent to the expected
information; both are reported so the choice is visible).

FAILURE MODES HANDLED EXPLICITLY
--------------------------------
  * complete/quasi-complete separation -- a coefficient diverges and the
    MLE does not exist. Detected and reported, never returned silently,
    because a diverged coefficient with a finite standard error is the
    most dangerous possible output.
  * an all-zero or all-one column in a group -- unidentifiable, dropped.
  * n_i = 0 rows -- contribute nothing, removed.
  * non-finite inputs -- rejected.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_binomial_logit", "logit", "BinomialFitError"]


class BinomialFitError(ValueError):
    """Raised when the binomial cannot be fitted as specified."""


def logit(p):
    """Numerically safe logit."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _loglik(eta, k, n):
    """sum k*eta - n*log(1+exp(eta)), without overflow."""
    # log(1+exp(eta)) computed stably on both tails
    ls = np.where(eta > 0, eta + np.log1p(np.exp(-np.abs(eta))),
                  np.log1p(np.exp(-np.abs(eta))))
    return float(np.sum(k * eta - n * ls))


def _fit_one(eta0, k, n, X, ridge, maxiter, tol, prior_prec=None):
    """Newton/IRLS for the (penalized) likelihood.

    prior_prec, when given, is a per-column Gaussian prior PRECISION vector:
    the objective becomes loglik - 0.5 * sum(prior_prec * beta^2), i.e. a MAP
    fit under independent N(0, 1/prior_prec) priors. The Newton system and
    the damping test both use the penalized objective, so the step can never
    decrease it. Returns (beta, loglik_penalized, iters, diverged).
    """
    def _obj(b):
        ll = _loglik(X @ b, k, n)
        if prior_prec is not None:
            ll = ll - 0.5 * float(np.sum(prior_prec * b * b))
        return ll

    beta = np.array(eta0, dtype=float)
    ll = _obj(beta)
    diverged = False
    it = 0
    for it in range(1, maxiter + 1):
        eta = X @ beta
        # numerically safe p = expit(eta)
        p = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-np.abs(eta))),
                     np.exp(-np.abs(eta)) / (1.0 + np.exp(-np.abs(eta))))
        w = n * p * (1.0 - p)
        w = np.maximum(w, 1e-10)
        grad = X.T @ (k - n * p)
        XtW = X.T * w
        H = XtW @ X
        if prior_prec is not None:
            grad = grad - prior_prec * beta
            H = H + np.diag(prior_prec)
        if ridge:
            H = H + ridge * np.eye(H.shape[0])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # damped step: never let the (penalized) objective get worse
        scale = 1.0
        cand = beta + step
        cand_ll = _obj(cand)
        for _ in range(40):
            if np.isfinite(cand_ll) and cand_ll >= ll - 1e-12:
                break
            scale *= 0.5
            cand = beta + scale * step
            cand_ll = _obj(cand)
        if not np.isfinite(cand_ll):
            diverged = True
            break
        delta = float(np.max(np.abs(cand - beta)))
        beta, ll = cand, cand_ll
        if delta < tol:
            break
    else:
        diverged = diverged or False
    return beta, ll, it, diverged


def fit_binomial_logit(
    k,
    n,
    X,
    names=None,
    intercept=True,
    ridge=0.0,
    maxiter=200,
    tol=1e-10,
    sep_tol=1e4,
    drop_degenerate=True,
    prior_sd=None,
):
    """Fit k ~ Binomial(n, sigmoid(X beta)), optionally as a MAP fit.

    Parameters
    ----------
    k : counts of "successes" (speeding trips), shape (N,)
    n : trials (judged trips), shape (N,)
    X : design matrix without intercept, shape (N, P)
    names : optional column names
    intercept : prepend a column of ones
    ridge : L2 ridge added to the Hessian only (0 = pure MLE)
    sep_tol : |beta| above which the fit is declared separated
    drop_degenerate : silently drop constant / all-zero columns
    prior_sd : optional dict {column name -> prior sd}, evaluated on the
        FINAL column names ("(intercept)" included). Independent Gaussian
        priors, MAP-fitted; the returned cov is then the Laplace posterior
        covariance (inverse posterior precision), and `laplace_evidence`
        is log p(y) up to a prior-independent constant, so different
        prior_sd dicts on the same design/data are directly comparable.

    Returns
    -------
    dict with keys
      coef, se, loglik, aic, bic, n_obs, n_trials, n_successes,
      converged, separated, dropped, names, fitted_p,
      loglik_pen, logdet_H, laplace_evidence, prior_prec
    """
    k = np.asarray(k, dtype=float).ravel()
    n = np.asarray(n, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and k.size > 1 and X.shape[1] != k.size:
        X = X.T
    if k.size != n.size or X.shape[0] != k.size:
        raise BinomialFitError(
            f"shape mismatch: k={k.size} n={n.size} X={X.shape}")
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(n))
            and np.all(np.isfinite(X))):
        raise BinomialFitError("non-finite input")

    # trials must be non-negative integers; k must lie in [0, n]
    if np.any(n < 0):
        raise BinomialFitError("negative trial counts")
    if np.any(k < 0) or np.any(k > n + 1e-6):
        bad = int(np.sum((k < 0) | (k > n + 1e-6)))
        raise BinomialFitError(
            f"{bad} rows have successes outside [0, n]")

    keep = n > 0
    dropped = []
    if not np.all(keep):
        k, n, X = k[keep], n[keep], X[keep]
    if k.size == 0:
        raise BinomialFitError("no rows with n > 0")

    cols = list(names) if names is not None else \
        [f"x{j}" for j in range(X.shape[1])]
    if len(cols) != X.shape[1]:
        cols = [f"x{j}" for j in range(X.shape[1])]

    if drop_degenerate:
        good = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.allclose(col, col[0]):
                dropped.append(cols[j])
            elif np.all(col == 0.0):
                dropped.append(cols[j])
            else:
                good.append(j)
        if len(good) != X.shape[1]:
            X = X[:, good]
            cols = [cols[j] for j in good]

    if intercept:
        X = np.column_stack([np.ones(len(k)), X])
        cols = ["(intercept)"] + cols

    if X.shape[1] == 0:
        raise BinomialFitError("no usable columns after dropping degenerate")

    # Gaussian prior precision aligned to the FINAL columns. A prior_sd entry
    # of None/0/NaN leaves that coefficient flat. Aliases: callers may use
    # "__INTERCEPT__" for the intercept.
    prior_prec = None
    if prior_sd:
        _p = np.zeros(X.shape[1])
        for j, c in enumerate(cols):
            sd = prior_sd.get(c)
            if sd is None and c == "(intercept)":
                sd = prior_sd.get("__INTERCEPT__")
            if sd is not None and np.isfinite(sd) and float(sd) > 0:
                _p[j] = 1.0 / float(sd) ** 2
        if np.any(_p > 0):
            prior_prec = _p

    def _obj(b):
        v = _loglik(X @ b, k, n)
        if prior_prec is not None:
            v = v - 0.5 * float(np.sum(prior_prec * b * b))
        return v

    # start from the null fit (intercept only) or zeros
    if intercept:
        rate = float(np.clip(k.sum() / max(n.sum(), 1e-12), 1e-6, 1 - 1e-6))
        start = np.zeros(X.shape[1])
        start[0] = float(np.log(rate / (1 - rate)))
    else:
        start = np.zeros(X.shape[1])

    beta, ll_pen, iters, diverged = _fit_one(
        start, k, n, X, ridge, maxiter, tol, prior_prec=prior_prec)
    ll = _loglik(X @ beta, k, n)

    # Posterior precision at the MAP. With a prior this is the Laplace
    # posterior covariance (inverse posterior precision); with a flat prior
    # it is the ordinary observed-information covariance.
    eta = X @ beta
    p = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-np.abs(eta))),
                 np.exp(-np.abs(eta)) / (1.0 + np.exp(-np.abs(eta))))
    w = np.maximum(n * p * (1.0 - p), 1e-10)
    XtW = X.T * w
    H = XtW @ X
    if prior_prec is not None:
        H = H + np.diag(prior_prec)
    if ridge:
        H = H + ridge * np.eye(H.shape[0])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    try:
        sign, logdet_H = np.linalg.slogdet(H)
        logdet_H = float(logdet_H) if sign > 0 else float("nan")
    except np.linalg.LinAlgError:
        logdet_H = float("nan")
    # Laplace log evidence log p(y), up to a constant independent of the
    # prior (same design and data):
    #     loglik_pen - 0.5 logdet(H) + 0.5 sum(log prior_prec)
    # The last term is the prior normaliser. WITHOUT it the comparison
    # silently prefers the widest prior (the missing term is exactly the
    # Occam penalty on prior volume), which is the opposite of the intended
    # behaviour. Only defined for proper priors: comparing a truly flat fit
    # against a proper-prior fit is invalid, so every evidence grid must use
    # proper (possibly very weak) priors throughout.
    log_prior_norm = (float(np.sum(np.log(prior_prec[prior_prec > 0])))
                      if prior_prec is not None
                      and np.any(prior_prec > 0) else float("nan"))
    laplace_evidence = (
        float(ll_pen - 0.5 * logdet_H + 0.5 * log_prior_norm)
        if (np.isfinite(logdet_H) and np.isfinite(log_prior_norm))
        else float("nan"))

    # expected information (alternative convention), for transparency
    try:
        Hexp = XtW @ X
        if ridge:
            Hexp = Hexp + ridge * np.eye(Hexp.shape[0])
        se_exp = np.sqrt(np.clip(np.diag(np.linalg.inv(Hexp)), 0.0, np.inf))
    except np.linalg.LinAlgError:
        se_exp = np.full_like(se, np.nan)

    npar = int(np.sum(np.isfinite(beta)))
    # Separation detection. Two signals, because neither alone suffices:
    #
    #  (a) perfect separation -- every bin that had a speeding trip is
    #      predicted p ~ 1 and every bin that had none is predicted p ~ 0.
    #      This is direct and scale-free. Note the loglik CANNOT be used for
    #      this: at a separated fit the likelihood is already at its
    #      supremum of 0, so pushing the coefficient further gains nothing
    #      numerically even though the MLE is genuinely at infinity.
    #
    #  (b) the loglik is still rising as a large coefficient is pushed
    #      further, which catches quasi-separation the perfect-prediction
    #      test can miss.
    #
    # A bare magnitude threshold is NOT used as the primary test: a large
    # coefficient on an unstandardised predictor is not separation.
    perfect = bool(
        np.all(((k > 0) & (p > 1.0 - 1e-6)) | ((k == 0) & (p < 1e-6)))
        and (k > 0).any() and (k == 0).any())
    runaway = False
    for j in range(len(beta)):
        if abs(beta[j]) < 5.0:
            continue
        lls = []
        for t in (1.0, 10.0, 50.0):
            cand = beta.copy()
            cand[j] += t * np.sign(beta[j])
            lls.append(_obj(cand))
        if lls[1] > ll_pen + 1e-9 and lls[2] > ll_pen + 1e-9:
            runaway = True
    separated = bool(diverged or perfect or runaway
                     or not np.all(np.isfinite(beta))
                     or np.any(np.abs(beta) > sep_tol))
    converged = bool(np.all(np.isfinite(beta)) and not diverged)

    return {
        "coef": beta,
        "se": se,
        "se_expected_info": se_exp,
        "cov": cov,
        "loglik": float(ll),
        "aic": float(-2 * ll + 2 * npar),
        "bic": float(-2 * ll + npar * np.log(max(k.size, 2))),
        "n_obs": int(k.size),
        "n_trials": float(n.sum()),
        "n_successes": float(k.sum()),
        "converged": converged,
        "separated": separated,
        "n_iter": int(iters),
        "dropped": dropped,
        "names": cols,
        "fitted_p": p,
        "design_columns": int(X.shape[1]),
        # Laplace-approximation extras. loglik_pen is the MAP objective;
        # logdet_H the log-determinant of the posterior precision; evidence
        # is comparable across priors on the same design/data.
        "loglik_pen": float(ll_pen),
        "logdet_H": logdet_H,
        "log_prior_norm": log_prior_norm,
        "laplace_evidence": laplace_evidence,
        "prior_prec": (None if prior_prec is None
                       else [float(x) for x in prior_prec]),
    }
