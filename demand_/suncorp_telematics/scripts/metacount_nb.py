"""
metacount_nb.py
Shared Negative-Binomial / Poisson random-parameters count model via
metacountregressor — replaces the old statsmodels SPF in 19/20/21/24.

Model (NB random-parameters):
    Y_i ~ NegBinom(lambda_i, alpha)
    log lambda_i = b0 + b1*log(E_i+1) + X_fixed*b_fixed + X_rdm*b_i
    b_i,k ~ Normal(mean_k, sd_k^2)   (independent random parameters)

Fitted with metacountregressor.ExperimentBuilder.fit_manual_model
(model='nb', dispersion=1). Poisson is the same spec with dispersion=0
(model='poisson', no alpha).

Provides:
  - prepare_metacount_frame(): logexp + highway dummies (trunk reference)
  - fit_metacount_nb(): fit + parse MLE/PARTE + predictions + alpha/AIC/BIC
  - predict_metacount(): out-of-sample expected counts from PARTE means
"""
import os
import re
import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Largest NB2 dispersion the package can report. The optimiser works on
# log_alpha and both print_summary and nb2_dispersion() map it through
# exp(clip(theta, -12, 12)), so a *reported* alpha is bounded by e^12. Sourced
# from the package when available so this tracks the real ceiling rather than
# duplicating the constant; e^12 is the documented fallback.
ALPHA_MAX = float(np.exp(12.0))
try:
    from metacountregressor.nb_parameterization import LOG_ALPHA_MAX
    ALPHA_MAX = float(np.exp(float(LOG_ALPHA_MAX)))
except Exception:  # pragma: no cover - older layout without the module
    pass

try:
    from metacountregressor import ExperimentBuilder
    _HAS_MCR = True
except Exception as e:
    _HAS_MCR = False
    _MCR_ERR = str(e)


def prepare_metacount_frame(df, exposure_col="trips",
                            highway_col="highway",
                            extra_fixed=("logexp",),
                            reference="trunk"):
    """Add logexp + highway dummies. Returns (df_out, dummy_cols)."""
    out = df.copy()
    out["logexp"] = np.log(out[exposure_col].astype(float) + 1.0)
    dummy_cols = []
    if highway_col in out.columns:
        hw = out[highway_col].fillna("unknown").astype(str).str.strip()
        # normalise common variants
        hw = hw.str.lower()
        dums = pd.get_dummies(hw, prefix="hw", dtype=float)
        # map back: reference level dropped to match paper ("trunk" reference)
        # dummies look like hw_trunk, hw_residential, ...
        ref_col = None
        for c in list(dums.columns):
            if c == f"hw_{reference}":
                ref_col = c
                break
        if ref_col in dums.columns:
            dums = dums.drop(columns=[ref_col])
        # sanitise names (metacountregressor needs valid identifiers)
        dums.columns = [re.sub(r"\W+", "_", c) for c in dums.columns]
        out = pd.concat([out.reset_index(drop=True),
                         dums.reset_index(drop=True)], axis=1)
        dummy_cols = list(dums.columns)
    return out, dummy_cols


def _parse_summary_text(txt):
    """Parse metacountregressor _summary_text table (v1.0.145 and older).

    New format (v1.0.145+selection-guards):
        Parameter  Estimate  Std.Err  z-value  p-value
    Old format:
        Parameter  Estimate(MLE)  Estimate(PARTE)  Std.Err  z-value  p-value  Shrinkage

    Returns dicts: mle, parte, se, z, p keyed by parameter name.
    For the new format mle==parte==Estimate (transformed scale used for
    predictions: sd rows are actual SDs, dispersion row is alpha directly).
    """
    mle, parte, se, z, p = {}, {}, {}, {}, {}
    for line in txt.splitlines():
        s = line.rstrip()
        if not s.strip() or s.strip().startswith("=") or s.strip().startswith("-"):
            continue
        if "Parameter" in s and ("Estimate" in s or "Std.Err" in s):
            continue
        if s.strip().startswith("Log-Likelihood") or s.strip().startswith("AIC") or s.strip().startswith("BIC"):
            continue
        toks = s.split()
        if len(toks) < 5:
            continue
        name = toks[0]
        try:
            nums = [float(t) for t in toks[1:]]
        except ValueError:
            continue
        if len(nums) >= 6:
            # old format: MLE, PARTE, SE, z, p, shrinkage
            mle[name], parte[name], se[name], z[name], p[name] = nums[0], nums[1], nums[2], nums[3], nums[4]
        elif len(nums) == 5:
            # shouldn't happen, treat as MLE,PARTE,SE,z,p
            mle[name], parte[name], se[name], z[name], p[name] = nums[0], nums[1], nums[2], nums[3], nums[4]
        elif len(nums) == 4:
            # new format: Estimate, SE, z, p
            mle[name], parte[name], se[name], z[name], p[name] = nums[0], nums[0], nums[1], nums[2], nums[3]
        else:
            continue
    return {"mle": mle, "parte": parte, "se": se, "z": z, "p": p}


def _norm_spec_terms(terms, work):
    """Keep 'var' / 'var:dist' terms whose base column exists in work."""
    out = []
    for t in terms or []:
        s = str(t)
        base = s.split(":")[0]
        if base not in work.columns:
            continue
        out.append(s if ":" in s else f"{base}:normal")
    return out


def _exposure_name(fixed_terms):
    """The design column that carries the free exposure coefficient.

    logexp is the exposure scale the pipeline builds (log(1 + seg_flow *
    bin_factor * uplift)). Named rather than hardcoded at the call site so
    the free-coefficient read stays correct if the column is renamed.
    """
    for cand in ("logexp", "log_exposure", "log_exp"):
        if any(c.split(":")[0] == cand for c in (fixed_terms or [])):
            return cand
    return "logexp"


def _refit_sane(parsed, alpha, expected, model):
    """Reject diverged/degenerate fits (e.g. |b|~1e5, all preds at floor)."""
    # A dispersion sitting ON the package clip is not a large dispersion: the
    # optimiser is optimising log_alpha unclipped, print_summary reports
    # exp(clip(theta, -12, 12)), and alpha == e^12 therefore means the NB2
    # likelihood wants infinite dispersion. Coefficients from such a fit are
    # not estimable, so the stage is rejected and the ladder moves on to the
    # SAME structure at the other family.
    if model == "nb" and np.isfinite(alpha) and alpha >= 0.999 * ALPHA_MAX:
        return False, (f"dispersion at clip (alpha={alpha:.3g}): NB2 "
                       "unbounded on this panel")
    vals = [float(v) for k, v in parsed.get("mle", {}).items()
            if "dispersion" not in str(k).lower()]
    if not vals or not all(np.isfinite(vals)):
        return False, "non-finite coefs"
    if max(abs(v) for v in vals) > 500:
        return False, f"diverged coefs (max|b|={max(abs(v) for v in vals):.1f})"
    # alpha is alpha-scale (see ALPHA_MAX), so the valid band is
    # (0, e^12]. The previous 1e4 ceiling rejected legitimately large
    # dispersions -- and _glm_fallback_fit used the same bound.
    if model == "nb" and not (np.isfinite(alpha) and 0 < alpha <= ALPHA_MAX):
        return False, f"bad dispersion alpha={alpha}"
    exp = np.asarray(expected, dtype=float)
    if not np.all(np.isfinite(exp)):
        return False, "non-finite predictions"
    if float(np.std(exp)) <= 0 or not (1e-9 <= float(np.mean(exp)) <= 1e6):
        return False, "degenerate predictions"
    return True, ""


def _screen_design(work, y_col, terms):
    """Drop design columns whose coefficient has no finite MLE.

    An MLE only exists if the likelihood is bounded in every direction. Three
    cheap cases break that on sparse count panels, and all three occur in the
    T9 refit design:

      constant        sd == 0 (e.g. in_event on an event-window frame): the
                      column duplicates the intercept, so (b, b0 - b) traces a
                      flat ridge and MLE drifts along it.
      separated       binary, and every row where it is 1 has Y == 0. Its
                      MLE is at -inf: the optimiser walks the coefficient
                      down while the log-likelihood merely approaches its
                      supremum, never attaining it. Poisson stalls early
                      (the curvature dies), NB2 does not -- alpha absorbs the
                      zeros -- so NB2 is precisely where |b| reaches 1e5.
      duplicate       byte-identical to a column already kept.

    Returned as (kept_terms, drops) with one {term, reason, stat} record per
    drop so the run can report what it removed and why. This is the same
    screening _glm_fallback_fit already performs; the JAX path previously had
    none, so a separated dummy that statsmodels would drop was handed to the
    optimiser, diverged, and cost the whole searched specification.
    """
    y = pd.to_numeric(work[y_col], errors="coerce").fillna(0.0).to_numpy(float)
    pos = y > 0
    kept, drops = [], []
    seen = {}
    for t in terms or []:
        base = str(t).split(":")[0]
        if base not in work.columns:
            continue
        v = pd.to_numeric(work[base], errors="coerce").to_numpy(float)
        v = np.where(np.isfinite(v), v, 0.0)
        sd = float(np.std(v))
        if sd <= 0.0:
            drops.append({"term": base, "reason": "constant",
                          "stat": float(v[0]) if len(v) else None})
            continue
        u = np.unique(v)
        if len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}:
            on = v > 0.5
            n_on = int(on.sum())
            ev_on = float(y[on].sum())
            if n_on and ev_on <= 0.0:
                drops.append({"term": base, "reason": "separated",
                              "stat": {"rows_on": n_on, "events_on": 0}})
                continue
        key = v.tobytes()
        if key in seen:
            drops.append({"term": base, "reason": "duplicate",
                          "stat": seen[key]})
            continue
        seen[key] = base
        kept.append(t)
    return kept, drops


def _drift_bases(parsed, att, limit=500.0):
    """Named design bases whose coefficient has run away (|b| > limit).

    A single drifting coefficient is not evidence that the whole
    specification is wrong: it is evidence that one column has no finite
    MLE. Naming it lets the caller drop that column and refit the SAME stage
    instead of throwing away the hierarchy with it.
    """
    bases = {str(t).split(":")[0] for t in
             list(att.get("fixed") or []) + list(att.get("rdm") or []) +
             list(att.get("rdm_cor") or []) + list(att.get("grouped") or []) +
             list(att.get("hetero") or []) + list(att.get("hetero_var") or [])}
    out = {}
    for name, b in (parsed.get("mle") or {}).items():
        s = str(name)
        if "dispersion" in s.lower():
            continue
        try:
            bv = float(b)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(bv) or abs(bv) <= limit:
            continue
        for cand in bases:
            if s == cand or s.endswith(f"({cand})") or s.endswith(f"_{cand}"):
                out[cand] = bv
                break
    return out


def _staged_attempts(model, fixed_terms, rdm_terms, rdm_cor_terms,
                     grouped_terms, hetro_in_means, hetro_in_variances):
    """The fallback ladder, structure-first and family-second.

    Every structure (full / nohetero / ind) is tried at the requested family
    before the other family's structures, and the no-random-structure stages
    come last. A family swap therefore no longer costs the searched random
    and grouped terms -- the defect that turned one diverged NB2 refit into a
    fixed-effects Poisson surface (T9 job 27247312).
    """
    _families = [model] + [m for m in ("nb", "poisson") if m != model]

    def _mk_stage(fam, kind):
        st = {"fixed": fixed_terms, "model": fam,
              "disp": 1 if fam == "nb" else 0, "label": f"{fam}-{kind}"}
        if kind == "full":
            st.update(rdm=rdm_terms, rdm_cor=rdm_cor_terms,
                      grouped=grouped_terms, hetero=hetro_in_means,
                      hetero_var=hetro_in_variances)
        elif kind == "nohetero":
            st.update(rdm=rdm_terms, rdm_cor=rdm_cor_terms,
                      grouped=grouped_terms, hetero=[], hetero_var=[])
        elif kind == "ind":
            st.update(rdm=rdm_terms + rdm_cor_terms, rdm_cor=[],
                      grouped=[], hetero=[], hetero_var=[])
        else:
            st.update(rdm=[], rdm_cor=[], grouped=[], hetero=[],
                      hetero_var=[])
        return st

    out = [_mk_stage(f, k) for f in _families
           for k in ("full", "nohetero", "ind")]
    out += [_mk_stage(f, "fixed") for f in _families]
    return out


def fit_metacount_nb(df, y_col, id_col, fixed_terms, rdm_terms,
                     model="nb", R=200, de_maxiter=8, de_popsize=8,
                     seed=0, verbose=False, continuous_de_warm_start=None,
                     use_slsqp=None, rdm_cor_terms=None, grouped_terms=None,
                     hetro_in_means=None, hetro_in_variances=None,
                     group_id_col=None, use_prefit_start=False,
                     offset_col=None, screen_design=True):
    """Fit NB (or Poisson) random-parameters count model.

    fixed_terms: list of column names (must exist, numeric)
    rdm_terms: list like ['lanes_z:normal'] (base names must exist)
    rdm_cor_terms / grouped_terms / hetro_in_means: full searched structure
      (correlated RP, highway-grouped RP, heterogeneity-in-means). These were
      previously dropped at refit, silently reducing the searched model to
      fixed-only. Staged fallback (structure first, family second):
      <model>-full -> -nohetero -> -ind -> <other>-full -> -nohetero -> -ind
      -> -fixed (both families); each stage is sanity-gated so a diverged fit
      (|b|>500, zero-variance predictions) is never accepted, and landing on
      any stage but the first is reported as ``degraded``.
    Returns dict with expected, alpha, aic, bic, loglik, coef table, fit, builder.

    Memory trick: the DE warm-start evaluates a whole population with
    jax.vmap (micro-batched x4 in metacountregressor>=1.0.162) and SLSQP
    holds grad buffers. When N*R is large (>~12M, e.g. 100k x 200) this
    dominates peak RSS, so it is auto-disabled unless explicitly requested
    — LBFGS polish on the prefit start gives the same optimum at a fraction
    of the memory.

    offset_col: optional LOG-scale exposure column used as a fixed offset
      (eta += offset in build_eta, so the column must already be
      log(exposure)). This is the hazard/rate parameterisation: the
      exposure coefficient is pinned at 1 instead of estimated. It is
      mutually exclusive with including the same column in fixed_terms --
      keeping both would fit exposure twice (once free, once pinned), so
      any occurrence in fixed/rdm/grouped/hetero is dropped and logged.

    Startup trick: the Poisson prefit (``use_prefit_start=True``) diverges on
    sparse 15-min counts — ``fit_simple_poisson_full`` returned
    [-96896, 3023, 200] on the Suncorp panel. That start puts eta below the
    -25 clip in ``mixed_model_loglik``, where the gradient is exactly zero,
    so LBFGS/SLSQP/DE all stall on the flat plateau and every spec is
    rejected as diverged. The structural search fits with ``use_prefit=False``
    from a small random init; matching that here converges to the same
    optimum (BIC 18293 vs searched 18327). Default is therefore off.
    """
    if not _HAS_MCR:
        raise ImportError(f"metacountregressor unavailable: {_MCR_ERR}")
    # ensure numeric, no NaN/inf (all spec bases, not just fixed)
    _all_bases = {str(t).split(":")[0] for t in
                  list(fixed_terms or []) + list(rdm_terms or []) +
                  list(rdm_cor_terms or []) + list(grouped_terms or []) +
                  list(hetro_in_means or []) + list(hetro_in_variances or [])}
    work = df.copy()
    for base in _all_bases:
        if base in work.columns:
            work[base] = pd.to_numeric(work[base], errors="coerce").fillna(0.0)
            work[base] = work[base].replace([np.inf, -np.inf], 0.0)
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce").fillna(0).astype(float)

    fixed_terms = [c for c in (fixed_terms or []) if c.split(":")[0] in work.columns]
    rdm_terms = _norm_spec_terms(rdm_terms, work)
    rdm_cor_terms = _norm_spec_terms(rdm_cor_terms, work)
    grouped_terms = _norm_spec_terms(grouped_terms, work)
    hetro_in_means = [str(t).split(":")[0] for t in (hetro_in_means or [])
                      if str(t).split(":")[0] in work.columns]
    hetro_in_variances = [str(t).split(":")[0] for t in (hetro_in_variances or [])
                          if str(t).split(":")[0] in work.columns]
    # Hazard parameterisation: pin the exposure coefficient at 1 by moving the
    # exposure column out of the design and into the fixed offset. It must be
    # REMOVED, not just complemented -- leaving it in fixed_terms would fit
    # exposure twice (once free, once pinned) and the two are not identified.
    if offset_col is not None:
        if offset_col not in work.columns:
            raise ValueError(f"offset_col {offset_col!r} not in frame")
        work[offset_col] = pd.to_numeric(work[offset_col], errors="coerce") \
            .replace([np.inf, -np.inf], 0.0).fillna(0.0)
        _drop = {offset_col}
        _before = (len(fixed_terms), len(rdm_terms) + len(rdm_cor_terms),
                   len(grouped_terms), len(hetro_in_means),
                   len(hetro_in_variances))
        fixed_terms = [c for c in fixed_terms if c.split(":")[0] not in _drop]
        rdm_terms = [t for t in rdm_terms if t.split(":")[0] not in _drop]
        rdm_cor_terms = [t for t in rdm_cor_terms
                         if t.split(":")[0] not in _drop]
        grouped_terms = [t for t in grouped_terms
                         if t.split(":")[0] not in _drop]
        hetro_in_means = [c for c in hetro_in_means if c not in _drop]
        hetro_in_variances = [c for c in hetro_in_variances if c not in _drop]
        if verbose:
            _after = (len(fixed_terms),
                      len(rdm_terms) + len(rdm_cor_terms),
                      len(grouped_terms), len(hetro_in_means),
                      len(hetro_in_variances))
            if _before != _after:
                print(f"  [metacount] hazard offset: {offset_col} pinned at "
                      f"coefficient 1.0; removed from design "
                      f"(fixed/rdm/grouped/hetero {_before} -> {_after})")
    gid = (group_id_col if group_id_col in work.columns else None) \
        if grouped_terms else None
    if gid is None and grouped_terms:
        if verbose:
            print("  [metacount] WARN: grouped terms without group column; "
                  "dropping grouped structure")
        grouped_terms = []
    if hetro_in_variances and verbose:
        print("  [metacount] NOTE: heterogeneity-in-variances passed through "
              "as spec key (backend support varies; falls back if rejected)")

    # What the caller asked for, before any screening, so the fidelity flag
    # reports a model that lost structure rather than comparing a screened
    # spec against itself.
    requested = {"fixed": list(fixed_terms), "rdm": list(rdm_terms) +
                 list(rdm_cor_terms), "grouped": list(grouped_terms),
                 "hetero": list(hetro_in_means)}
    # Design screening (see _screen_design). Applied to every role, because a
    # separated or constant base is unidentifiable whichever way it enters;
    # a grouped slope on such a column has no finite MLE either.
    design_drops = []
    if screen_design:
        if fixed_terms:
            fixed_terms, _d = _screen_design(work, y_col, fixed_terms)
            design_drops += _d
        if rdm_terms:
            rdm_terms, _d = _screen_design(work, y_col, rdm_terms)
            design_drops += _d
        if rdm_cor_terms:
            rdm_cor_terms, _d = _screen_design(work, y_col, rdm_cor_terms)
            design_drops += _d
        if grouped_terms:
            grouped_terms, _d = _screen_design(work, y_col, grouped_terms)
            design_drops += _d
        if hetro_in_means:
            _h, _d = _screen_design(work, y_col, hetro_in_means)
            design_drops += _d
            hetro_in_means = [str(t).split(":")[0] for t in _h]
        if not grouped_terms:
            gid = None
        # Heterogeneity-in-means is only defined for a coefficient that is
        # still random; if screening removed its random block, the term would
        # make make_manual_spec fail and take the whole stage (and, by
        # cascade, the hierarchy) down with it.
        if hetro_in_means:
            _rand_now = {str(t).split(":")[0] for t in
                         list(rdm_terms) + list(rdm_cor_terms)
                         + list(grouped_terms)}
            _h_lost = [t for t in hetro_in_means if t not in _rand_now]
            if _h_lost:
                hetro_in_means = [t for t in hetro_in_means
                                  if t in _rand_now]
                if verbose:
                    print("  [metacount] design screen: hetro_in_means "
                          f"dropped {_h_lost} (no longer a random term)")
        if design_drops and verbose:
            for _dr in design_drops:
                print(f"  [metacount] design screen: dropped {_dr['term']!r} "
                      f"({_dr['reason']})")

    # Number of groups the FIT saw. Stored so prediction can size the grouped
    # draw tensor identically even when the prediction frame happens to
    # contain fewer cells: a shorter tensor shifts every group's draws.
    _n_groups = None
    _group_cats = None
    if gid is not None and gid in work.columns:
        _gc = work[gid]
        if isinstance(_gc.dtype, pd.CategoricalDtype):
            _n_groups = int(len(_gc.cat.categories))
            _group_cats = [str(c) for c in _gc.cat.categories]
        else:
            _n_groups = int(_gc.astype(str).nunique())

    builder = ExperimentBuilder(work, id_col=id_col, y_col=y_col,
                                group_id_col=gid, offset_col=offset_col)
    # Auto low-mem guard: N*R large => skip vmap DE warm-start by default.
    _n, _r = int(len(work)), int(R)
    if continuous_de_warm_start is None:
        continuous_de_warm_start = not (_n * _r > 12_000_000)
        if verbose and not continuous_de_warm_start:
            print(f"  [metacount] low-mem: N*R={_n * _r:,} > 12M; "
                  "disabling continuous-DE warm-start (LBFGS polish only)")
    if use_slsqp is None:
        use_slsqp = bool(_n * _r <= 12_000_000)

    # STAGED LADDER, ordered so that STRUCTURE is given up before FAMILY.
    #
    # The old ladder was nb-full -> nb-nohetero -> nb-ind -> nb-fixed ->
    # poisson-fixed: the only Poisson stage was also the only stage with no
    # random or grouped terms. So whenever NB2 was not estimable -- which on
    # a 1.5%-positive panel it is not, because the dispersion has no finite
    # MLE and log_alpha runs to its clip -- the ladder necessarily shipped a
    # FIXED-EFFECTS Poisson model and the searched fixture-grouped slopes
    # were dropped as collateral damage (T9 job 27247312).
    #
    # A family swap and a structure cut are independent decisions; collapsing
    # them means the cheap one costs the expensive one. The ladder now tries
    # every structure at the requested family, then the same structures at the
    # other family, and only then the no-random-structure stages -- so a
    # diverged NB2 fits the SAME hierarchy as Poisson instead of discarding
    # it. Landing anywhere but the first entry is still reported as degraded.
    attempts = _staged_attempts(model, fixed_terms, rdm_terms, rdm_cor_terms,
                                grouped_terms, hetro_in_means,
                                hetro_in_variances)
    last_err = None
    last_reject = None
    drift_retries = []
    rejected_stages = []
    _max_drift_retries = 2
    _queue = list(attempts)
    while _queue:
        att = _queue.pop(0)
        try:
            spec = builder.make_manual_spec(
                fixed_terms=att["fixed"], rdm_terms=att["rdm"],
                rdm_cor_terms=att["rdm_cor"], grouped_terms=att["grouped"],
                hetro_in_means=att["hetero"], dispersion=att["disp"],
                group_id_col=gid)
            if att["hetero_var"]:
                spec["hetro_in_variances"] = list(att["hetero_var"])
            fit = builder.fit_manual_model(
                manual_spec=spec, model=att["model"], R=R,
                print_report=False, de_maxiter=de_maxiter,
                de_popsize=de_popsize, de_seed=seed,
                continuous_de_warm_start=continuous_de_warm_start,
                use_slsqp=use_slsqp, use_prefit_start=use_prefit_start)
            parsed = _parse_summary_text(fit.get("_summary_text", ""))
            summ = fit.get("summary", {})
            # alpha: the summary-text dispersion Estimate is ALREADY on the
            # alpha scale -- print_summary reports exp(clip(theta, -12, 12)),
            # matching nb2_dispersion() in the package. It is never a log
            # value. Alpha therefore ranges over (0, e^12 ~ 1.63e5], and a
            # fit landing in (1e4, e^12] is legitimately large, not diverged.
            #
            # The old chain gated on `0 < alpha < 1e4` and then fell back to
            # np.exp(value) on that same alpha-scale number. A reported
            # dispersion of, say, 5e4 therefore failed the gate and overflowed
            # to inf through exp(), so the fit was rejected as
            # "bad dispersion alpha=inf" and a searched spec was thrown away.
            # Both the gate and the exp() were units errors.
            alpha = 0.0
            alpha_src = "fallback-1.0"
            if att["model"] == "nb":
                dp = parsed["parte"].get("dispersion", np.nan)
                dm = parsed["mle"].get("dispersion", np.nan)
                if np.isfinite(dp) and 0 < dp <= ALPHA_MAX:
                    alpha, alpha_src = float(dp), "summary-dispersion"
                elif np.isfinite(dm) and 0 < dm <= ALPHA_MAX:
                    alpha, alpha_src = float(dm), "summary-dispersion-mle"
                elif np.isfinite(dm) and dm > ALPHA_MAX:
                    # Past the package ceiling: a genuinely diverged fit, and
                    # the raw number is the reportable value (do NOT re-exp).
                    alpha, alpha_src = float(dm), "summary-dispersion-overflow"
                else:
                    # Last resort: the raw unconstrained coordinate. This one
                    # IS on the log scale (the optimiser optimises log_alpha),
                    # so exp() is correct here and only here.
                    try:
                        import numpy as _np
                        praw = float(_np.asarray(fit["result"].params)[fit["param_index"]["dispersion"]])
                        alpha = float(_np.exp(praw))
                        alpha_src = "exp(raw-log-alpha)"
                    except Exception:
                        alpha, alpha_src = 1.0, "fallback-1.0"
            expected = np.asarray(fit["predictions"], dtype=float)
            expected = np.maximum(expected, 1e-9)
            _sane, _sane_reason = _refit_sane(parsed, alpha, expected,
                                              att["model"])
            if not _sane:
                if verbose:
                    print(f"  [metacount] {att['label']} rejected "
                          f"({_sane_reason})")
                last_reject = f"{att['label']}: {_sane_reason}"
                rejected_stages.append(last_reject)
                # Drift is a property of a COLUMN, not of the stage: drop the
                # runaway columns and refit this same stage before giving up
                # on its random/grouped structure. Without this, one
                # separated dummy walked the ladder from nb-full all the way
                # to poisson-fixed (T9 job 27247312) and the searched
                # hierarchy was lost with it.
                _drift = _drift_bases(parsed, att)
                # Only a LOCALISED runaway is worth a retry: if a third of the
                # design is drifting the problem is the start/scale, not one
                # column, and refitting without it just burns another fit.
                _n_fixed = max(len(att.get("fixed") or []), 1)
                if len(_drift) > max(2, int(0.34 * _n_fixed)):
                    _drift = {}
                if _drift and len(drift_retries) < _max_drift_retries:
                    _new = dict(att)
                    for _k in ("fixed", "rdm", "rdm_cor", "grouped",
                               "hetero", "hetero_var"):
                        _new[_k] = [t for t in (att.get(_k) or [])
                                    if str(t).split(":")[0] not in _drift]
                    _new["label"] = (f"{att['label']}-nodrift"
                                     f"{len(drift_retries) + 1}")
                    drift_retries.append({"from": att["label"],
                                          "to": _new["label"],
                                          "dropped": sorted(_drift)})
                    if verbose:
                        print("  [metacount] drift in %s (|b| up to %.0f); "
                              "dropping those columns and refitting %s"
                              % (", ".join(sorted(_drift)),
                                 max(abs(v) for v in _drift.values()),
                                 att["label"]))
                    _queue.insert(0, _new)
                    continue
                if verbose:
                    print("  [metacount] no localised drift; trying a "
                          "simpler spec")
                try:
                    import gc as _gc
                    _gc.collect()
                except Exception:
                    pass
                continue
            # coef table for paper (guaranteed columns even if parse is thin)
            rows = []
            for name, b in parsed["mle"].items():
                rows.append({
                    "variable": name,
                    "coefficient": b,
                    "coef_mle": b,
                    "coef_parte": parsed["parte"].get(name, np.nan),
                    "std_err": parsed["se"].get(name, np.nan),
                    "z_value": parsed["z"].get(name, np.nan),
                    "p_value": parsed["p"].get(name, np.nan),
                })
            if not rows:
                # ultimate fallback: raw params via print_coefficients
                try:
                    ct = builder.print_coefficients(fit)
                    for _, r in ct.iterrows():
                        rows.append({
                            "variable": str(r["Parameter"]),
                            "coefficient": float(r["Estimate"]),
                            "coef_mle": float(r["Estimate"]),
                            "coef_parte": float(r["Estimate"]),
                            "std_err": np.nan, "z_value": np.nan, "p_value": np.nan,
                        })
                except Exception:
                    pass
            coef_df = pd.DataFrame(rows, columns=["variable", "coefficient", "coef_mle",
                                                  "coef_parte", "std_err", "z_value", "p_value"])
            # beta dict on prediction (transformed) scale
            beta = {"__INTERCEPT__": float(parsed["parte"].get("__INTERCEPT__", 0.0))}
            for c in att["fixed"]:
                base = c.split(":")[0]
                if base in parsed["parte"]:
                    beta[base] = float(parsed["parte"][base])
            for r_ in att["rdm"] + att["rdm_cor"]:
                base = r_.split(":")[0]
                key = f"mean({base})"
                if key in parsed["parte"]:
                    beta[f"__mean__{base}"] = float(parsed["parte"][key])
            if verbose:
                _src = ("n/a (poisson)" if att["model"] != "nb"
                        else alpha_src)
                print(f"  [metacount] {att['label']}: "
                      f"n={len(work)} AIC={summ.get('aic', np.nan):.1f} "
                      f"alpha={alpha:.4f} ({_src})")
            try:
                import gc as _gc
                _gc.collect()
                try:
                    import jax as _jax
                    _jax.clear_caches()
                except Exception:
                    pass
            except Exception:
                pass
            return {
                "expected": expected,
                # None (not 0.0) for Poisson: the model has no dispersion
                # parameter, and 0.0 reads downstream as a degenerate
                # dispersion rather than an absent one.
                "alpha": (float(alpha) if att["model"] == "nb" else None),
                "alpha_src": alpha_src,
                "aic": float(summ.get("aic", np.nan)),
                "bic": float(summ.get("bic", np.nan)),
                "loglik": float(summ.get("loglik", np.nan)),
                "n_obs": int(summ.get("n_obs", len(work))),
                "n_parm": int(summ.get("num_parm", len(rows))),
                "coef": coef_df,
                "parsed": parsed,
                "beta": beta,
                "fit": fit,
                "spec": fit.get("spec"),
                "builder": builder,
                "manual_spec": spec,
                "spec_label": att["label"],
                "fixed_terms": att["fixed"],
                "rdm_terms": att["rdm"] + att["rdm_cor"],
                "rdm_cor_terms": att["rdm_cor"],
                "grouped_terms": att["grouped"],
                "model": att["model"],
                # Exposure bookkeeping. offset_col means the exposure
                # coefficient is pinned at 1 by construction and cannot be
                # read off the fit; with no offset the free coefficient is
                # reported so the hazard-vs-free LRT is possible.
                "offset_col": offset_col,
                # FIT FIDELITY. A staged ladder that lands on a simpler
                # stage than the top one has shipped a different model from
                # the one requested; the caller must be able to see that
                # rather than infer it from a label string.
                "design_drops": design_drops,
                "drift_retries": drift_retries,
                "rejected_stages": rejected_stages,
                "requested": requested,
                "group_id_col": gid,
                "n_groups": _n_groups,
                "group_categories": _group_cats,
                "degraded": bool(
                    att["label"] != attempts[0]["label"]
                    or len(att.get("grouped") or []) < len(requested["grouped"])
                    or len(att.get("rdm") or []) + len(att.get("rdm_cor") or [])
                    < len(requested["rdm"])),
                "exposure_mode": ("offset-pinned" if offset_col
                                  else "free-coefficient"),
                "exposure_coef": (None if offset_col
                                  else float(parsed["parte"].get(
                                      _exposure_name(fixed_terms), np.nan))),
            }
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  [metacount] {att['label']} failed: {e}")
            continue
    try:
        return _glm_fallback_fit(work, y_col, fixed_terms, verbose=verbose)
    except Exception as _glm_err:
        raise RuntimeError(f"All metacountregressor specs failed: "
                           f"{last_err or last_reject}; "
                           f"GLM fallback failed: {_glm_err}")


def _glm_fallback_fit(work, y_col, fixed_terms, verbose=False):
    """Terminal non-JAX fallback when every metacountregressor stage diverges.

    Chain: statsmodels NB -> Poisson GLM -> intercept-only Poisson, with
    separation screening (zero-variance columns and dummies with no events
    are dropped up front). The intercept-only stage always converges on
    non-degenerate counts, so this never raises for usable data. Returns
    the same info-dict contract as fit_metacount_nb with spec_label
    'statsmodels-*-fallback' (callers can detect it with str.startswith).
    """
    import statsmodels.api as _sm

    y = pd.to_numeric(work[y_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
    y = np.where(np.isfinite(y) & (y >= 0), y, 0.0)
    n = len(work)
    cols = []
    for c in (fixed_terms or []):
        base = str(c).split(":")[0]
        if base not in work.columns or base in cols:
            continue
        v = pd.to_numeric(work[base], errors="coerce").fillna(0.0).astype(float).to_numpy()
        if not np.all(np.isfinite(v)) or float(np.std(v)) <= 0:
            continue
        u = np.unique(v)
        if len(u) == 2 and set(u.tolist()) <= {0.0, 1.0} \
                and float(y[v > 0.5].sum()) <= 0:
            if verbose:
                print(f"  [metacount] glm-fallback: dropping separated dummy {base}")
            continue
        cols.append(base)
    X_full = pd.DataFrame({"const": np.ones(n)})
    for c in cols:
        X_full[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0).astype(float).to_numpy()
    X0 = pd.DataFrame({"const": np.ones(n)})
    last_reject = "no stages attempted"
    for X, family, label in ((X_full, "nb", "statsmodels-nb-fallback"),
                             (X_full, "poisson", "statsmodels-poisson-fallback"),
                             (X0, "poisson", "statsmodels-intercept-fallback")):
        try:
            if family == "nb":
                res = _sm.NegativeBinomial(y, X).fit(disp=0, maxiter=500)
                alpha = float(res.params.get("alpha", np.nan))
                # statsmodels NB alpha is on the same alpha scale as the
                # package's, so share the ceiling (see ALPHA_MAX).
                if not (np.isfinite(alpha) and 0 < alpha <= ALPHA_MAX):
                    last_reject = f"{label}: bad alpha={alpha}"
                    continue
                model = "nb"
            else:
                res = _sm.GLM(y, X, family=_sm.families.Poisson()).fit()
                alpha = 0.0
                model = "poisson"
            params = res.params
            bse = res.bse
            tvals = getattr(res, "tvalues", pd.Series(np.nan, index=params.index))
            pvals = getattr(res, "pvalues", pd.Series(np.nan, index=params.index))
            f = lambda s: {str(k): float(v) for k, v in s.items()}
            mle, se_d, z_d, p_d = f(params), f(bse), f(tvals), f(pvals)
            parsed = {"mle": mle, "parte": dict(mle), "se": se_d, "z": z_d, "p": p_d}
            exp = np.maximum(np.asarray(res.predict(), dtype=float), 1e-9)
            # same sanity gate as the JAX path (alpha lives outside mle here)
            sane, reason = _refit_sane(
                {"mle": {k: v for k, v in mle.items() if k != "alpha"}},
                alpha, exp, model)
            if not sane:
                last_reject = f"{label}: {reason}"
                continue
            llf = float(res.llf)
            k = int(len(params))
            aic = float(-2.0 * llf + 2.0 * k)
            bic = float(-2.0 * llf + k * np.log(max(n, 1)))
            rows = [{"variable": kk, "coefficient": mle[kk],
                     "coef_mle": mle[kk], "coef_parte": mle[kk],
                     "std_err": se_d.get(kk, np.nan),
                     "z_value": z_d.get(kk, np.nan),
                     "p_value": p_d.get(kk, np.nan)} for kk in mle]
            coef_df = pd.DataFrame(rows, columns=["variable", "coefficient", "coef_mle",
                                                  "coef_parte", "std_err", "z_value", "p_value"])
            beta = {"__INTERCEPT__": float(mle.get("const", 0.0))}
            for c in cols:
                if c in mle:
                    beta[c] = float(mle[c])
            print(f"  [metacount] {label}: n={n} AIC={aic:.1f} alpha={alpha:.4f} "
                  f"(fallback; JAX stages diverged)")
            return {"expected": exp, "alpha": float(alpha), "aic": aic, "bic": bic,
                    "loglik": llf, "n_obs": n, "n_parm": k, "coef": coef_df,
                    "parsed": parsed, "beta": beta, "fit": None, "spec": None,
                    "builder": None, "manual_spec": None, "spec_label": label,
                    "fixed_terms": list(cols), "rdm_terms": [],
                    "rdm_cor_terms": [], "grouped_terms": [], "model": model,
                    # Every JAX stage diverged: the shipped fit carries none of
                    # the requested random/hierarchical structure, so it is
                    # degraded by construction.
                    "design_drops": [],
                    "drift_retries": [],
                    "rejected_stages": [], "requested": {},
                    "group_id_col": None, "n_groups": None,
                    "group_categories": None, "degraded": True}
        except Exception as e:  # noqa: BLE001 - try next simpler stage
            last_reject = f"{label}: {e}"
            continue
    raise RuntimeError(f"GLM fallback exhausted: {last_reject}")


def predict_metacount(df_new, fit_info):
    """Out-of-sample expected counts using prediction-scale means.

    Uses fit_info['beta'] (summary-text transformed scale, verified to
    reproduce fit['predictions'] to ~0.3%). Falls back to parsed PARTE.

    NOTE: this is a deterministic plug-in at the mean random coefficients.
    For random-parameter fits the correct predictive mean is
    E_b[exp(eta)] > exp(E_b[eta]) (log-normal variance inflation), so use
    ``predict_metacount_sim`` whenever the fit has random structure.
    """
    beta = fit_info.get("beta") or {}
    parsed = fit_info.get("parsed", {})
    parte = parsed.get("parte", {}) if isinstance(parsed, dict) else {}
    intercept = float(beta.get("__INTERCEPT__", parte.get("__INTERCEPT__", 0.0)))
    lin = np.full(len(df_new), intercept)
    # A PINNED exposure (offset_col) is not a design column: fit() removed it
    # from fixed/rdm/grouped/hetero terms so the coefficients cannot fit it
    # twice. Predicting from fixed_terms alone therefore returns exp(Xb), i.e.
    # a RATE per unit exposure, not the expected count -- silent, and it looks
    # like a 1/E-level under-prediction rather than a units error. Add it back.
    _off = fit_info.get("offset_col")
    if _off:
        if _off not in df_new.columns:
            raise ValueError(
                f"predict_metacount: the model was fitted with {_off!r} pinned "
                "as its exposure offset, but the prediction frame does not "
                "carry that column. Predicting without it would ship rates "
                "instead of counts.")
        lin = lin + pd.to_numeric(df_new[_off], errors="coerce") \
            .fillna(0.0).values
    # fixed dummies + numeric
    for c in fit_info.get("fixed_terms", []):
        base = c.split(":")[0]
        if base not in df_new.columns:
            continue
        b = beta.get(base, parte.get(c, parte.get(base, None)))
        if b is None or not np.isfinite(b):
            continue
        lin = lin + pd.to_numeric(df_new[base], errors="coerce").fillna(0.0).values * float(b)
    # random means
    for r in fit_info.get("rdm_terms", []):
        base = r.split(":")[0]
        if base not in df_new.columns:
            continue
        b = beta.get(f"__mean__{base}", parte.get(f"mean({base})", None))
        if b is None or not np.isfinite(b):
            continue
        lin = lin + pd.to_numeric(df_new[base], errors="coerce").fillna(0.0).values * float(b)
    return np.maximum(np.exp(lin), 1e-9)


def predict_metacount_sim(df_new, fit_info, id_col="row_id", y_col="Y",
                          R=200, chunk=150000):
    """Draw-averaged predictive mean for random-parameter fits.

    Rebuilds model data for ``df_new`` through the SAME package code path as
    the fit (``build_model_from_manual_spec`` + ``CountModel.predict``), but
    with the FIT's scaler so the fitted parameters apply unchanged, and with
    the FIT's ``offset_col`` re-attached (a pinned exposure was removed from
    the design by fit(), so it has to be passed back explicitly), then
    averages ``exp(eta)`` over R fresh Halton draws. This reproduces the
    in-sample ``expected`` (mean over draws) instead of the plug-in at the
    mean random coefficients, which under-predicts by the log-normal
    variance-inflation factor (observed ~10x on the Suncorp panel).

    Falls back to the linear plug-in when the fit has no random structure.
    Chunked to bound peak memory (draws are (chunk, K, R)).
    """
    fit = fit_info.get("fit")
    spec = fit_info.get("spec")
    manual_spec = fit_info.get("manual_spec")
    has_random = bool(spec is not None and (
        getattr(spec, "Kr_ind", 0) + getattr(spec, "Kr_cor", 0)
        + getattr(spec, "Kg", 0)) > 0)
    if not (fit is not None and manual_spec is not None and has_random):
        return predict_metacount(df_new, fit_info)

    try:
        import metacountregressor.main_hpc as mh
    except Exception:
        return predict_metacount(df_new, fit_info)

    scaler = (fit.get("data") or {}).get("scaler", {}) if isinstance(fit, dict) else {}
    params = np.asarray(fit["result"].params)
    orig_scaler_fn = mh.compute_scaler

    def _fit_scaler(df, cols, exclude_binary=True):
        return {k: v for k, v in scaler.items() if k in cols}

    # One shared Halton draw set per random block. Any iid draw set gives an
    # unbiased estimate of E_b[exp(eta)], so sharing the draws across rows is
    # valid and avoids regenerating N x R Halton points per chunk (the
    # dominant cost: 25k x 1000 x 2 per chunk). Grouped draws are indexed by
    # group code (build_eta: beta_g_all[group_ids]), so they keep one row per
    # group rather than per observation.
    _kr_ind = int(getattr(spec, "Kr_ind", 0) or 0)
    _kr_cor = int(getattr(spec, "Kr_cor", 0) or 0)
    _kg = int(getattr(spec, "Kg", 0) or 0)
    _gid_col = (manual_spec or {}).get("group_id_col")
    base_draws = {}
    if _kr_ind:
        base_draws["draws_ind"] = mh.generate_halton_normal(1, _kr_ind, int(R), seed=42)
    if _kr_cor:
        base_draws["draws_cor"] = mh.generate_halton_normal(1, _kr_cor, int(R), seed=43)
    if _kg:
        # Prefer the group count the FIT used. Sizing this from the prediction
        # frame gave a shorter draw tensor whenever a frame was missing a
        # cell, which silently applied another cell's coefficients.
        if fit_info.get("n_groups"):
            n_groups = int(fit_info["n_groups"])
        elif _gid_col and _gid_col in df_new.columns:
            _gc = df_new[_gid_col]
            n_groups = (int(len(_gc.cat.categories))
                        if isinstance(_gc.dtype, pd.CategoricalDtype)
                        else int(pd.Series(_gc).astype(str).nunique()))
        else:
            n_groups = len(df_new)
        base_draws["draws_g"] = mh.generate_halton_normal(
            max(n_groups, 1), _kg, int(R), seed=44)

    mh.compute_scaler = _fit_scaler
    try:
        outs = []
        for start in range(0, len(df_new), int(chunk)):
            sub = df_new.iloc[start:start + int(chunk)].copy()
            if y_col not in sub.columns:
                sub[y_col] = 0.0
            draws_kwargs = {}
            for k, v in base_draws.items():
                if k == "draws_g":
                    draws_kwargs[k] = v
                else:
                    draws_kwargs[k] = np.broadcast_to(
                        v, (len(sub),) + v.shape[1:])
            # Re-impose the FIT's group categories. The package codes groups
            # with astype("category").cat.codes, which re-derives categories
            # from whatever frame it is given: a prediction frame missing one
            # cell would shift every later code and apply another cell's
            # coefficients. Declaring the fit's categories makes the coding
            # identical by construction.
            _fit_cats = fit_info.get("group_categories")
            if _fit_cats and _gid_col and _gid_col in sub.columns:
                sub[_gid_col] = pd.Categorical(
                    sub[_gid_col].astype(str), categories=_fit_cats)
            # offset_col MUST travel with the spec. make_manual_spec() does
            # not record it (it lives on the builder instance, and the fit
            # gets it as a separate argument to build_model_from_manual_spec),
            # so re-building here without it dropped a pinned exposure from
            # eta entirely: every pinned-offset arm shipped exp(Xb), the
            # exposure-normalised rate, and the "3-4x holdout level deficit"
            # was that missing factor (obs/pred 2.81 vs the observed-weighted
            # harmonic mean of E, 3.00).
            _offset_col = fit_info.get("offset_col")
            if _offset_col and _offset_col not in sub.columns:
                raise ValueError(
                    f"predict_metacount_sim: the model was fitted with "
                    f"{_offset_col!r} pinned as its exposure offset, but the "
                    "prediction frame does not carry that column. Predicting "
                    "without it would ship rates instead of counts.")
            data_new, spec_new = mh.build_model_from_manual_spec(
                df=sub, manual_spec=manual_spec, id_col=id_col, y_col=y_col,
                offset_col=_offset_col,
                R=int(R), **draws_kwargs)
            model = mh.CountModel(spec_new, data_new)
            model.params = params
            mu = np.asarray(model.predict(), dtype=float)
            mu = np.maximum(mu.reshape(len(sub), -1).mean(axis=1), 1e-9)
            # balance_panel_dataframe sorts by id_col; invert that so the
            # returned vector aligns with df_new's row order.
            ids = np.asarray(sub[id_col].to_numpy())
            if len(ids) and np.any(np.diff(ids) < 0):
                order = np.argsort(ids, kind="stable")
                aligned = np.empty_like(mu)
                aligned[order] = mu
                mu = aligned
            outs.append(mu)
            del data_new, spec_new, model, mu, draws_kwargs
        return np.concatenate(outs)
    finally:
        mh.compute_scaler = orig_scaler_fn
