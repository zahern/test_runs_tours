"""Land-use -> activity-participation -> travel-behavior mediation analysis.

Implements the Ding & Lu (2016, J. Transp. Geography, S0966692316000314)
structural-equation idea inside this pipeline's estimators: activity
participation is modelled as a MEDIATOR between land-use patterns and travel
behavior, so each land-use variable decomposes into

    total effect  =  direct effect  +  indirect effect (via participation)

with delta-method standard errors, z-tests and confidence intervals.  The
paper's headline result — indirect paths reinforcing (and sometimes rescuing)
direct land-use effects that look insignificant in isolation — is reproduced
as a first-class screening report for the GA variable-selection loop.

Two-equation setup (person level, SearchLibrium estimators, unpenalized)::

    M-equation (mediator) : ordered logit of participation band
                            P(M <= j) on land-use X + controls C
    Y-equation (outcome)  : binary logit of travel outcome
                            P(Y = 1) on land-use X + participation score M
                            + controls C

For land-use variable x with M-coefficient ``a``, Y-coefficient on the
mediator ``b`` and Y-coefficient on x (``c'``, direct)::

    indirect = a * b,   total = c' + a * b

Variances come from the delta method over the block-diagonal joint
covariance of the two separately-estimated equations
(``SearchLibrium.cov_params``; falls back to diagonal SEs when the
installed SearchLibrium predates the ``cov_params`` accessor, in which
case within-equation covariances are treated as 0 and flagged).

Mediation hook for variable selection: :func:`mediation_screen` returns a
``force_include`` list of land-use variables with significant indirect but
insignificant direct effects — exactly the variables a direct-only GA would
wrongly drop.  Feed it to the constraint overlay / seed list, e.g.::

    screen = mediation_screen(fx)
    # -> {"force_include": ["transit_acc", ...], ...}

References
----------
Ding, Y. & Lu, H. (2016). Activity participation as a mediating variable
to analyze the effect of land use on travel behavior: a structural
equation modeling approach. Journal of Transport Geography, 52, 23-28.
DOI: 10.1016/j.jtrangeo.2016.02.009
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

logger = logging.getLogger(__name__)

_M_COEF = "m_score"          # mediator score column entering the Y equation
_CONST = "const"             # explicit intercept column (SearchLibrium
                             # ordered/MNL paths are used with
                             # fit_intercept=False; see module notes)


# ---------------------------------------------------------------------------
# estimator adapters (SearchLibrium, unpenalized for inference-valid SEs)
# ---------------------------------------------------------------------------

def _sl_coef_frame(model, kind):
    """Extract (names, coefs, cov) from a fitted SearchLibrium model.

    Uses the ``cov_params()`` accessor (SearchLibrium>=0.0.237); falls back
    to diagonal SEs with a logged warning when unavailable.
    Returns (names, beta, V, diag_only).
    """
    names, beta, V, diag_only = None, None, None, False
    try:
        V = model.cov_params()
        names = list(V.index)
        beta = np.asarray(
            getattr(model, 'coeff_est',
                    getattr(model, 'params', [])), dtype=float).ravel()
        if beta.size != len(names):
            raise ValueError("name/coef length mismatch")
        return names, beta, np.asarray(V.values, dtype=float), False
    except Exception as e:
        logger.warning(f"{kind}: cov_params() unavailable ({e}) — "
                       f"diagonal-SE fallback (within-equation covariances "
                       f"treated as 0)")
    # diagonal fallback
    if hasattr(model, 'labels') and model.labels is not None:
        names = [str(n) for n in list(model.labels)]
        beta = np.asarray(getattr(model, 'params'), dtype=float).ravel()
    else:
        names = [str(n) for n in list(getattr(model, 'coeff_names', []))]
        beta = np.asarray(getattr(model, 'coeff_est', []), dtype=float).ravel()
    se = np.asarray(getattr(model, 'stderr',
                            getattr(model, 'se', None)), dtype=float).ravel()
    if beta.size == 0 or se.size != beta.size:
        raise ValueError(f"{kind}: cannot recover coefficients/SEs from "
                         f"fitted model")
    se = np.where(np.isfinite(se) & (se > 0), se, np.nan)
    return names, beta, np.diag(se ** 2), True


def _find_coef(names, target):
    """Index of coefficient ``target`` trying exact, dotted and suffixed forms.

    SearchLibrium names binary-logit slopes like ``x1.1``; ordered slopes are
    plain.  Raises ValueError listing available names when unresolvable.
    """
    if target in names:
        return names.index(target)
    for i, nm in enumerate(names):
        s = str(nm)
        if s == target or s.startswith(target + ".") or s.startswith(target + ":"):
            return i
    raise ValueError(f"coefficient {target!r} not in model "
                     f"(available: {names})")


def fit_mediator(df, x_cols, ctrl_cols=(), m_col="m_band", use_jax=True,
                 distr="logit"):
    """Ordered-logit M-equation: participation band ~ land use + controls.

    No explicit intercept: the J-1 cut-point thresholds serve as intercepts,
    and an added constant would be perfectly collinear with them (flat ridge,
    attenuated slopes).
    """
    from SearchLibrium import OrderedLogit
    feats = [c for c in list(x_cols) + list(ctrl_cols) if c in df.columns]
    feats = [c for c in feats if c != _CONST]
    d = df.dropna(subset=feats + [m_col]).copy()
    y = pd.to_numeric(d[m_col], errors='coerce').fillna(0).astype(int).values
    J = int(y.max()) + 1
    if J < 2:
        raise ValueError("mediator has a single level — cannot fit ordered logit")
    Xdf = d[feats].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # NB: SearchLibrium ordered link defaults to probit — pass logit explicitly.
    m = OrderedLogit(_jax=use_jax, X=Xdf, y=y, J=J, fit_intercept=False,
                     distr=distr)
    m.fit()
    names, beta, V, diag_only = _sl_coef_frame(m, "M-equation")
    return {"model": m, "names": names, "beta": beta, "V": V,
            "diag_only": diag_only, "features": feats,
            "ll": float(m.loglik), "n": int(len(d))}


def fit_outcome(df, x_cols, m_col="m_score", ctrl_cols=(), y_col="y",
                use_jax=True):
    """Binary-logit Y-equation: outcome ~ land use + mediator + controls."""
    from SearchLibrium import LogisticRegression
    feats = [c for c in list(x_cols) + [m_col] + list(ctrl_cols)
             if c in df.columns]
    if _CONST not in feats:
        feats = feats + [_CONST]
    d = df.dropna(subset=feats + [y_col]).copy()
    y = pd.to_numeric(d[y_col], errors='coerce').fillna(0).values
    y = (y > 0).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("outcome has a single class — cannot fit binary logit")
    Xdf = d[feats].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # NB: return_hess flag is inverted upstream (True -> dummy); False yields
    # the Hessian inverse needed for inference. Penalties off for valid SEs.
    m = LogisticRegression(_jax=use_jax)
    m.fit(Xdf, y, varnames=feats, return_hess=False,
          l2_penalty=0.0, l1_penalty=0.0)
    names, beta, V, diag_only = _sl_coef_frame(m, "Y-equation")
    return {"model": m, "names": names, "beta": beta, "V": V,
            "diag_only": diag_only, "features": feats,
            "ll": float(m.loglik), "n": int(len(d))}


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------

def _z_p(est, se):
    est, se = float(est), float(se)
    if not np.isfinite(est) or not np.isfinite(se) or se <= 0:
        return np.nan, np.nan, (np.nan, np.nan)
    z = est / se
    p = 2.0 * float(norm.sf(abs(z)))
    return z, p, (est - 1.96 * se, est + 1.96 * se)


def mediation_effects(m_fit, y_fit, x_cols, m_name=_M_COEF, alpha=0.05):
    """Direct / indirect / total effects per land-use variable.

    Returns a DataFrame with columns: variable, a, se_a, b, se_b, direct,
    se_direct, indirect, se_indirect, total, se_total, z/p/CI for indirect
    and total, prop_mediated, n.  Delta method over the block-diagonal joint
    covariance of the two equations (cross-equation covariances are 0 by
    separate-estimation construction).
    """
    mn, mb, mV = m_fit["names"], m_fit["beta"], m_fit["V"]
    yn, yb, yV = y_fit["names"], y_fit["beta"], y_fit["V"]
    p_m, p_y = len(mn), len(yn)
    V = np.zeros((p_m + p_y, p_m + p_y))
    V[:p_m, :p_m] = mV
    V[p_m:, p_m:] = yV
    try:
        ib = _find_coef(yn, m_name)
    except ValueError:
        raise ValueError(f"mediator {m_name!r} not a Y-equation coefficient "
                         f"(Y names: {yn})")
    b, vb = float(yb[ib]), float(yV[ib, ib])
    rows = []
    for x in x_cols:
        try:
            ia = _find_coef(mn, x)
        except ValueError:
            continue
        try:
            ic = _find_coef(yn, x)
        except ValueError:
            continue
        a, va = float(mb[ia]), float(mV[ia, ia])
        c, vc = float(yb[ic]), float(yV[ic, ic])
        vcb = float(yV[ic, ib])  # within-Y cov(direct, mediator path)
        ind = a * b
        # Var(a*b) = b^2 Var(a) + a^2 Var(b); cross-equation cov = 0.
        v_ind = b * b * va + a * a * vb
        tot = c + ind
        # Var(c + a*b) = Var(c) + Var(ab) + 2 b Cov(c,b).
        v_tot = vc + v_ind + 2.0 * b * vcb
        se_ind = float(np.sqrt(max(v_ind, 0.0)))
        se_tot = float(np.sqrt(max(v_tot, 0.0)))
        se_a = float(np.sqrt(max(va, 0.0)))
        se_c = float(np.sqrt(max(vc, 0.0)))
        z_ind, p_ind, ci_ind = _z_p(ind, se_ind)
        z_tot, p_tot, ci_tot = _z_p(tot, se_tot)
        _, p_c, _ = _z_p(c, se_c)
        prop = float(ind / tot) if np.isfinite(tot) and abs(tot) > 1e-12 else np.nan
        rows.append({
            "variable": x, "a": a, "se_a": se_a,
            "b": b, "se_b": float(np.sqrt(max(vb, 0.0))),
            "direct": c, "se_direct": se_c, "p_direct": p_c,
            "indirect": ind, "se_indirect": se_ind,
            "z_indirect": z_ind, "p_indirect": p_ind,
            "ci_indirect_lo": ci_ind[0], "ci_indirect_hi": ci_ind[1],
            "total": tot, "se_total": se_tot,
            "z_total": z_tot, "p_total": p_tot,
            "ci_total_lo": ci_tot[0], "ci_total_hi": ci_tot[1],
            "prop_mediated": prop,
            "sig_indirect": bool(np.isfinite(p_ind) and p_ind < alpha),
            "sig_direct": bool(np.isfinite(p_c) and p_c < alpha),
            "sig_total": bool(np.isfinite(p_tot) and p_tot < alpha),
        })
    fx = pd.DataFrame(rows)
    fx.attrs["diag_only"] = bool(m_fit.get("diag_only") or y_fit.get("diag_only"))
    fx.attrs["n_m"] = int(m_fit.get("n", 0))
    fx.attrs["n_y"] = int(y_fit.get("n", 0))
    return fx


def llr_mediation(y_full_ll, y_nomed_ll, df_extra=1):
    """Likelihood-ratio test: does the mediator belong in the Y equation?"""
    stat = 2.0 * (float(y_full_ll) - float(y_nomed_ll))
    p = float(chi2.sf(max(stat, 0.0), int(df_extra)))
    return {"lr_stat": stat, "df": int(df_extra), "p_value": p}


# ---------------------------------------------------------------------------
# GA hook: the paper's headline as a variable-selection screen
# ---------------------------------------------------------------------------

def mediation_screen(fx, alpha=0.05):
    """Flag land-use variables a direct-only search would wrongly drop.

    Returns {"force_include": [...], "detail": {...}} where force_include
    holds variables with a significant indirect effect whose direct effect
    is insignificant — Ding & Lu's core finding — plus variables whose
    total effect is significant while the direct path alone is not.
    Feed ``force_include`` to the constraint overlay / GA seed list.
    """
    force, detail = [], {}
    for _, r in fx.iterrows():
        v = str(r["variable"])
        indirect_only = bool(r["sig_indirect"] and not r["sig_direct"])
        total_not_direct = bool(r["sig_total"] and not r["sig_direct"])
        if indirect_only or total_not_direct:
            force.append(v)
        detail[v] = {"indirect_only": indirect_only,
                     "total_not_direct": total_not_direct,
                     "p_indirect": float(r["p_indirect"]),
                     "p_direct": float(r["p_direct"]),
                     "p_total": float(r["p_total"])}
    return {"force_include": force, "alpha": float(alpha), "detail": detail}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def mediation_table_tex(fx, caption=None, label="tab:mediation"):
    """Paper-style LaTeX table: direct / indirect / total per land-use var."""
    cap = caption or ("Mediation decomposition: direct, indirect (via activity "
                      "participation) and total effects of land use on travel "
                      "behavior.")
    L = ["% Auto-generated by mediation_analysis.py — DO NOT EDIT BY HAND.",
         "\\begin{table}[!ht]",
         f"  \\caption{{{cap}}}\\label{{{label}}}",
         "  \\begin{center}",
         "    \\begin{tabular}{lrrrr}",
         "      \\toprule",
         "      Land use & Direct & Indirect & Total & Prop. mediated \\\\",
         "      \\midrule"]
    for _, r in fx.iterrows():
        star = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        try:
            pm = f"{r['prop_mediated']:.2f}"
        except Exception:
            pm = "---"
        L.append(f"      {r['variable']} & {r['direct']:.4f}{star(r['p_direct'])} & "
                 f"{r['indirect']:.4f}{star(r['p_indirect'])} & "
                 f"{r['total']:.4f}{star(r['p_total'])} & {pm} \\\\")
    L += ["      \\bottomrule", "    \\end{tabular}", "  \\end{center}",
          "  \\footnotesize{Indirect = via activity participation; "
          "SEs by the delta method. */**/***: 5/1/0.1\\% significance.}",
          "\\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# HHTS frame builder (person-level X / M / Y)
# ---------------------------------------------------------------------------

def build_mediation_frame(persons, tours, landuse, purpose="nonmand",
                          y_mode=1, x_cols=None, ctrl_cols=None,
                          m_cap=2, person_id="person_id"):
    """Assemble the person-level mediation frame from pipeline tables.

    X: home-zone land-use of each person (via
    ``synth_attributes.attach_person_accessibility``, as in Stage 1).
    M: participation band for ``purpose`` (0 / 1 / 2+ ``n_<p>_out``) plus the
    numeric ``m_score`` used in the Y equation.
    Y: 1 if the person's (first) ``purpose`` tour uses ``y_mode`` (default
    1 = DA), else 0; persons without such a tour are dropped (count logged).
    """
    from stage1_activity_frequency import build_activity_counts
    from synth_attributes import attach_person_accessibility
    if x_cols is None:
        x_cols = ["auto_work_acc", "auto_shop_acc", "walk_acc", "bike_acc",
                  "transit_acc", "emp_density", "is_cbd", "transit_walk_acc"]
    if ctrl_cols is None:
        ctrl_cols = ["incp", "CARLICENCE", "vehrd", "sexp", "nprd"]
    pers = persons.copy()
    if person_id not in pers.columns and "PERSID" in pers.columns:
        pers = pers.rename(columns={"PERSID": person_id})
    counts = build_activity_counts(tours)
    m_col = f"n_{purpose}_out"
    if m_col not in counts.columns:
        raise ValueError(f"purpose {purpose!r}: {m_col} not in tour counts")
    df = pers.merge(counts[[person_id, m_col]], on=person_id, how="left")
    df[m_col] = pd.to_numeric(df[m_col], errors="coerce").fillna(0)
    df = attach_person_accessibility(df, landuse)
    # Y from the person's first purpose tour
    t = tours.copy()
    tmode = "tour_mode" if "tour_mode" in t.columns else None
    if tmode is None:
        raise ValueError("tours table needs a tour_mode column")
    t = t.sort_values(["person_id", "tour_id"] if "tour_id" in t.columns
                      else ["person_id"])
    first = t.groupby("person_id", as_index=False).first()
    mp = "main_purpose" if "main_purpose" in first.columns else "purpose"
    first = first[first[mp].astype(str).str.lower() == purpose]
    first["y"] = (pd.to_numeric(first[tmode], errors="coerce") == y_mode).astype(int)
    df = df.merge(first[[person_id, "y"]], on=person_id, how="inner")
    n_drop = int(len(pers) - len(df))
    logger.info(f"mediation frame: {len(df)} persons with a {purpose} tour "
                f"({n_drop} without dropped)")
    df["m_band"] = pd.to_numeric(df[m_col], errors="coerce").fillna(0).clip(0, m_cap).astype(int)
    df[_M_COEF] = df["m_band"].astype(float)
    if _CONST not in df.columns:
        df[_CONST] = 1.0
    keep = [person_id, "m_band", _M_COEF, "y", _CONST]
    keep += [c for c in list(x_cols) + list(ctrl_cols) if c in df.columns]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# one-call driver
# ---------------------------------------------------------------------------

def run_mediation_report(df, x_cols, ctrl_cols=(), m_col="m_band",
                         y_col="y", out_dir=None, label="mediation",
                         use_jax=True, alpha=0.05):
    """Fit both equations, decompose effects, write csv/tex. Returns dict."""
    x_cols = [c for c in x_cols if c in df.columns]
    ctrl_cols = [c for c in ctrl_cols if c in df.columns]
    m_fit = fit_mediator(df, x_cols, ctrl_cols, m_col=m_col, use_jax=use_jax)
    y_fit = fit_outcome(df, x_cols, ctrl_cols=ctrl_cols, y_col=y_col,
                        use_jax=use_jax)
    fx = mediation_effects(m_fit, y_fit, x_cols, alpha=alpha)
    # nested LLR: Y|X+M vs Y|X
    y_nom = fit_outcome(df[[c for c in df.columns if c != _M_COEF]],
                        [c for c in x_cols], ctrl_cols=ctrl_cols, y_col=y_col,
                        use_jax=use_jax) if _M_COEF in df.columns else None
    llr = llr_mediation(y_fit["ll"], y_nom["ll"],
                        df_extra=1) if y_nom is not None else None
    screen = mediation_screen(fx, alpha=alpha)
    out = {"effects": fx, "screen": screen, "llr": llr,
           "m_ll": m_fit["ll"], "y_ll": y_fit["ll"],
           "n_m": m_fit["n"], "n_y": y_fit["n"],
           "diag_only": fx.attrs.get("diag_only", False)}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fx.to_csv(os.path.join(out_dir, f"{label}_effects.csv"), index=False)
        with open(os.path.join(out_dir, f"{label}_table.tex"), "w") as f:
            f.write(mediation_table_tex(fx))
    return out
