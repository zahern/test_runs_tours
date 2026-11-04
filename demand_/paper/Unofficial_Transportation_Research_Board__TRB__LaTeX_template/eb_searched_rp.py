"""EB prior (count) models on searched specs + random-parameter check (Table 31 replacement job)
+ in-job best-model searches (Part S).

Why this job exists
-------------------
Table 31 (``tab:eb_spf`` in ``eb_results.tex``) fits BOTH the crash SPF and
the hard-braking SPF on the SAME exhaustive 30-variable geometry set
(``trb_empirical_bayes.py`` GEO list, Poisson-GLM + Hauer moment dispersion,
``log_len`` offset). That is why every variable appears in both columns:
the spec was never searched, it is the full candidate set.

Likewise, no EB job ever searched random parameters. The random-parameter
evidence in the paper (``run_rp_check.py`` on the 14-var HB / 10-var crash
DE benchmarks; ``tab:mcr`` role search on the N=4,000 subsample;
``refit_significant.py`` step 5 constrained verdict) was never fed back
into the EB priors -- ``trb_empirical_bayes.py`` cannot fit random
parameters at all.

What this job does
------------------
For each response it uses its OWN searched condensed spec (the paper's
reported parsimonious models, NOT the 30-var baseline):

  HB    (11 vars): AADT, SL, SKD_SP, IP, PSDSG24, IV_VL,
                   IC, IV_M, PA_COM1, RSDSO, RSDSDL5
  Crash (8 vars):  AADT, IP, MEDSFBar, PSDSNIL,
                   PSDS1T24, RSDSDG10, SL, RSDSDL5

(Specs are read from hb_final_features_{nm,crash}.csv when present so the
job tracks the paper; hardcoded fallbacks match refit_significant.py.)

  Part A (Hauer EB, same method as Table 31, log_len offset):
    Poisson-GLM + Hauer moment alpha on the searched spec per response,
    EB posteriors, 5-fold CV, EB-HB encompassing, screening figure.
    Writes the drop-in Table 31 replacement with DIFFERENT rows per model.
  Part B (MCR NB, same estimator as the DE benchmarks, log_len offset):
    fixed vs random-parameter NB via metacountregressor.fit_model (R=200
    Halton draws) with the MCR role blocks from run_rp_check.py, which
    apply cleanly to the condensed specs (all role vars are retained):
      HB:    random-indep IC, IV_M + random-corr PA_COM1, RSDSO, RSDSDL5
      Crash: random-indep PSDS1T24, RSDSDG10 + random-corr SL, RSDSDL5
    Writes the fixed-vs-random verdict (answers "why not RP models").
    Also writes RP-SD forest plots with general SEs (90% CI).
  Part S (best-model SEARCHES inside this job -- new):
    S1 core HB search through metacount (SparseEA-AGDS over the 30
       geometry candidates, fitness = MCR NB penalised BIC, single run,
       no restarts; seed 42).
    S2 core crash search through metacountregressor (same estimator /
       fitness; seed 43).
    S3 joint HB+crash copula search (SparseEA-AGDS over the 2xJ joint
        mask, warm-started at the S1/S2 best masks, score = copula
        joint BIC with IFM (--joint-copula: joe/gumbel/frank/normal/
        kimeldorf; single run, no restarts).
    S4 MCR random-parameter role search (StructureEvaluatorLC over
       roles 0/1/2/3 with normal mixing) on the searched best specs, so
       the RP blocks are searched rather than only hardcoded; general
       SEs (JAX Hessian with numerical fallback, as computed by
       fit_model) are reported per coefficient and plotted per SD.
    Every search writes best-spec CSVs, a per-generation history CSV
    (best-so-far, best-current, mean/median/min/max) and a convergence
    PNG showing the best-so-far line, the best-current line, the
    population mean/median, the min-max band and faint per-individual
    population traces.

Outputs (NEW files; existing eb_results.tex / Table 31 untouched):
  eb_searched_model_features.csv   searched-spec SPF coefs (crash & HB)
  eb_searched_posterior.csv        per-segment SPF, w, EB (searched priors)
  eb_searched_crossval.csv         5-fold CV on searched specs
  eb_searched_encompassing.csv     crash ~ crash-geo + log1p(EB_HB) vs geo
  eb_searched_screening.png        shrinkage figure from searched priors
  eb_searched_spf.tex              Table 31 replacement (union rows, "--"
                                   where a var is not in that model's spec)
  eb_searched_rp.csv / .tex        fixed-vs-RP NB comparison (log_len)
  eb_searched_rp_HB_fixed.csv etc. per-fit coef tables
  eb_searched_verdict.csv          RP accept/reject per response
  eb_searched_rp_sds_{HB,Crash}.png  RP-SD forest plots (general SEs)
  eb_search_best_{hb,crash}.csv    SparseEA-AGDS searched best specs
  eb_search_hist_{hb,crash}.csv    per-generation histories
  eb_search_convergence_{hb,crash}.png  convergence + population traces
  eb_search_joint_best.{csv,tex}   joint copula searched masks
  eb_search_hist_joint.csv         joint per-generation history
  eb_search_convergence_joint.png  joint convergence + population traces
  eb_search_mcr_role_{HB,Crash}.csv  MCR-searched RP roles
  eb_search_vs_condensed.csv       searched vs condensed spec comparison

Usage (HPC): python -u eb_searched_rp.py [--data near_miss_hb_ecf.csv]
             [--search all|hb|crash|joint|mcr-role|none] [--skip-search]
             [--skip-rp] [--hb-maxiter 60 ...]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MCR_MIN_VERSION = (1, 0, 165)

# Searched condensed specs (fallback if the feature CSVs are absent).
HB_SEARCHED = ["AADT", "SL", "SKD_SP", "IP", "PSDSG24", "IV_VL",
               "IC", "IV_M", "PA_COM1", "RSDSO", "RSDSDL5"]
CRASH_SEARCHED = ["AADT", "IP", "MEDSFBar", "PSDSNIL",
                  "PSDS1T24", "RSDSDG10", "SL", "RSDSDL5"]

# MCR random-parameter blocks (from run_rp_check.py; all members survive
# the DE -> condensed pruning, so they apply unchanged here).
HB_RDM = ["IC:normal", "IV_M:normal"]
HB_RDM_COR = ["PA_COM1:normal", "RSDSO:normal", "RSDSDL5:normal"]
CR_RDM = ["PSDS1T24:normal", "RSDSDG10:normal"]
CR_RDM_COR = ["SL:normal", "RSDSDL5:normal"]

# 30-candidate geometry set (same as refit_sparsea_agds.py / SP1). The
# SparseEA-AGDS core searches (Part S) optimise over this set with the
# metacountregressor NB estimator (fit_model), so the "best model" is
# searched inside this job instead of being inherited from CSVs.
GEO_CANDIDATES = [
    "AADT", "SL", "NL", "SKD_SP", "CU_MOD", "CU_SHA", "CU_VSHA",
    "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
    "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
    "IV_VL", "IV_L", "IV_M", "IV_100",
    "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
    "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10",
]

RS = 42


def check_mcr_version(min_ver=None):
    try:
        import metacountregressor as mcr  # noqa: F401
        try:
            from importlib.metadata import version as _pkg_version
            ver = _pkg_version("metacountregressor")
        except Exception:
            ver = getattr(mcr, "__version__", "0.0.0")
    except Exception:
        print("FATAL: metacountregressor not importable -- refusing to run",
              flush=True)
        sys.exit(2)
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
    need = tuple(min_ver) if min_ver else MCR_MIN_VERSION
    need_s = ".".join(map(str, need))
    print(f"metacountregressor {ver} (required >= {need_s})", flush=True)
    if tup < need:
        print(f"FATAL: metacountregressor {ver} < {need_s} -- refusing to run",
              flush=True)
        sys.exit(2)
    return ver


def load_specs():
    """Prefer the paper's feature CSVs; fall back to hardcoded condensed sets."""
    hb, cr = list(HB_SEARCHED), list(CRASH_SEARCHED)
    try:
        f = os.path.join(HERE, "hb_final_features_nm.csv")
        if os.path.isfile(f):
            hb = pd.read_csv(f)["feature"].tolist()
    except Exception as e:
        print(f"WARNING: hb_final_features_nm.csv unreadable ({e}); "
              f"using fallback", flush=True)
    try:
        f = os.path.join(HERE, "hb_final_features_crash.csv")
        if os.path.isfile(f):
            cr = pd.read_csv(f)["feature"].tolist()
    except Exception as e:
        print(f"WARNING: hb_final_features_crash.csv unreadable ({e}); "
              f"using fallback", flush=True)
    print(f"HB searched spec ({len(hb)}): {hb}", flush=True)
    print(f"Crash searched spec ({len(cr)}): {cr}", flush=True)
    return hb, cr


def load_data(feats):
    import statsmodels.api as sm  # noqa: F401  (fail fast if missing)
    df = pd.read_csv(os.path.join(HERE, "near_miss_hb_ecf.csv"),
                     low_memory=False)
    df = df.dropna(subset=["ObsCrashes"])
    for c in sorted(set(feats)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        else:
            raise RuntimeError(f"searched feature absent from data: {c}")
    df["seg_len_km"] = (pd.to_numeric(df.get("segment_length_m", 0),
                                      errors="coerce").fillna(0) / 1000.0
                        ).clip(lower=0.001)
    df["log_len"] = np.log(df["seg_len_km"])
    if "HB_Total" in df.columns:
        df["HB"] = pd.to_numeric(df.get("HB_Total"), errors="coerce").fillna(0)
    else:
        df["HB"] = pd.to_numeric(df.get("nm_total"), errors="coerce").fillna(0)
    # Standardise searched feats globally (same convention as
    # trb_empirical_bayes.py so Table 31 replacement is apples-to-apples).
    for c in sorted(set(feats)):
        s = df[c].std()
        if s > 1e-10:
            df[c] = (df[c] - df[c].mean()) / s
    if "segment_id" not in df.columns:
        df["segment_id"] = np.arange(len(df))
    return df


def fit_nb(df, y, feats, offset, maxiter=2000):
    """Poisson GLM + Hauer moment alpha (identical to trb_empirical_bayes)."""
    import statsmodels.api as sm
    d = df.dropna(subset=[y] + feats + [offset]).copy()
    X = sm.add_constant(d[feats].values.astype(float), has_constant="add")
    off = d[offset].values.astype(float)
    yv = d[y].values.astype(float)
    fam = sm.families.Poisson()
    g = sm.GLM(yv, X, offset=off, family=fam).fit(method="irls",
                                                 maxiter=maxiter)
    lam = np.asarray(g.fittedvalues, float)
    alpha = max(float(np.sum((yv - lam) ** 2 - lam))
                / max(float(np.sum(lam ** 2)), 1e-9), 0.0)
    return g, lam, float(alpha), d


def eb_weight(alpha, lam):
    return 1.0 / (1.0 + np.asarray(alpha, float) * np.asarray(lam, float))


def plot_convergence_with_traces(hist, plot_path, title, ylabel):
    """SparseEA-AGDS convergence plot: best-so-far + current best + traces.

    hist must provide history_best (best-so-far) plus, when available,
    history_gen_best (best current), history_mean/median and history_pop
    (full per-individual fitness vectors). Renders the bold best-so-far
    line, the faint current-best line, mean/median dashed lines, the
    min-max band, and faint per-individual population traces (up to 32).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hb = np.asarray(hist.get("history_best", []), dtype=float)
    if hb.size == 0:
        return
    gb = np.asarray(hist.get("history_gen_best", hb), dtype=float)
    hm = hist.get("history_mean")
    hd = hist.get("history_median")
    hmin = hist.get("history_min")
    hmax = hist.get("history_max")
    hpop = hist.get("history_pop")
    gens = np.arange(1, len(hb) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        if hpop is not None and len(hpop) == len(hb):
            P = min(len(np.atleast_1d(hpop[0])), 32)
            for pi in range(P):
                tr = [float(np.atleast_1d(v)[pi])
                      if len(np.atleast_1d(v)) > pi else float("nan")
                      for v in hpop]
                ax.plot(gens, tr, color="gray", linewidth=0.4, alpha=0.22)
        if hmin is not None and hmax is not None and len(hmin) == len(hb):
            ax.fill_between(gens, np.asarray(hmin, float),
                            np.asarray(hmax, float), color="gray",
                            alpha=0.15, label="Population range")
    except Exception:
        pass
    ax.plot(gens, gb, color="darkorange", linewidth=0.9, alpha=0.55,
            label="Best current (generation best)")
    try:
        if hm is not None and len(hm) == len(hb):
            ax.plot(gens, np.asarray(hm, float), color="#4477AA",
                    linewidth=1.0, alpha=0.75, linestyle="--",
                    label="Population mean")
        if hd is not None and len(hd) == len(hb):
            ax.plot(gens, np.asarray(hd, float), color="#44AA77",
                    linewidth=1.0, alpha=0.75, linestyle=":",
                    label="Population median")
    except Exception:
        pass
    ax.plot(gens, hb, color="#B34700", linewidth=1.9,
            label=f"Best so far = {float(np.min(hb)):.2f}")
    ax.set_xlabel("Generation (single run, no restarts)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sparsea_core_search(dfm, candidates, y_col, offset_col, label, seed,
                            maxiter, popsize, p_threshold, penalty,
                            out_dir, tag):
    """Core HB / crash search inside this job (SparseEA-AGDS + MCR NB).

    Fitness = metacountregressor NB BIC + penalty/insignificant coef
    (same penalised-BIC convention as refit_sparsea_agds.py). Single run,
    no restarts. Writes best-spec CSV/TeX, full history CSV (best-so-far,
    best-current, mean/median/min/max), and a convergence PNG with
    population traces.
    """
    sys.path.insert(0, HERE)
    from nearmiss_hb_analysis import fit_model
    from sparsea_agds import sparsea_agds_search
    cands = [c for c in candidates
             if c in dfm.columns
             and pd.to_numeric(dfm[c], errors="coerce").std() > 1e-10]
    dfs = dfm.dropna(subset=cands + [y_col, offset_col]).reset_index(drop=True)
    print(f"\n[Part S:{tag}] SparseEA-AGDS core search y={y_col} "
          f"J={len(cands)} maxiter={maxiter} popsize={popsize} seed={seed}",
          flush=True)

    def _score_vec(x):
        mask = (np.asarray(x) > 0.5).astype(int)
        if mask.sum() < 2:
            return 1e12
        sel = [cands[i] for i in range(len(cands)) if mask[i] == 1]
        try:
            o = fit_model(dfs, sel, y_col, offset_col, label=f"S-{tag}",
                          verbose=False)
        except Exception:
            return 1e12
        if o is None:
            return 1e12
        ct = o["coef_table"]
        feat_rows = ct.iloc[1:]
        if len(ct) and str(ct["Variable"].iloc[-1]).startswith("ln("):
            feat_rows = feat_rows.iloc[:-1]
        try:
            n_insig = sum(1 for p in feat_rows["p"]
                          if (p is None) or (not np.isfinite(p))
                          or float(p) > p_threshold)
        except Exception:
            n_insig = len(feat_rows)
        return float(o["bic"]) + n_insig * penalty

    def _mask_score(mask):
        return _score_vec(np.asarray(mask, dtype=float))

    out = sparsea_agds_search(_mask_score, len(cands),
                              pop_size=max(popsize * 4, 32),
                              max_gen=maxiter, seed=seed)
    best_mask = np.asarray(out["best_mask"]).astype(int)
    feats = [cands[i] for i in range(len(cands)) if best_mask[i] == 1]
    sys.path.insert(0, HERE)
    from nearmiss_hb_analysis import fit_model as _fm
    best_model = _fm(dfs, feats, y_col, offset_col, label=f"S-best-{tag}",
                     verbose=False)
    hist = {k: (list(v) if k != "history_pop" else v)
            for k, v in out.items() if k.startswith("history_")}
    # history CSV (best-so-far, best-current, mean/median/min/max per gen)
    n = len(out["history_best"])
    hd = pd.DataFrame({"gen": np.arange(1, n + 1),
                       "best_so_far": out["history_best"],
                       "best_current": out.get("history_gen_best",
                                               out["history_best"]),
                       "mean": out.get("history_mean", [np.nan] * n),
                       "median": out.get("history_median", [np.nan] * n),
                       "min": out.get("history_min", [np.nan] * n),
                       "max": out.get("history_max", [np.nan] * n)})
    hd.to_csv(os.path.join(out_dir, f"eb_search_hist_{tag}.csv"), index=False)
    pd.DataFrame({"feature": feats, "response": tag,
                  "step": "sparsea-agds-eb"}).to_csv(
        os.path.join(out_dir, f"eb_search_best_{tag}.csv"), index=False)
    plot_convergence_with_traces(
        out, os.path.join(out_dir, f"eb_search_convergence_{tag}.png"),
        f"Single-run SparseEA-AGDS Convergence — {label} (metacount NB)",
        "Penalised BIC")
    print(f"[Part S:{tag}] best {len(feats)} feats penalised-BIC="
          f"{float(np.min(out['history_best'])):.2f} raw-BIC="
          f"{best_model['bic']:.2f}: {feats}", flush=True)
    return feats, best_model, hist


def run_joint_copula_sparsea_search(dfm, cands, y_hb, y_cr, offset_col,
                                     hb_init, cr_init, out_dir,
                                     pop_size=20, max_gen=12, seed=44,
                                     copula="joe"):
    """Joint (HB mask, crash mask) SparseEA-AGDS search on copula joint BIC.

    Joint mask = 2*J bits (first J = HB inclusion, next J = crash
    inclusion), warm-started at the core-search best masks. Each eval
    refits both fixed NB margins (metacountregressor fit_model) and the
    copula dependence by IFM; score = penalised joint BIC
    (joint BIC + 10 x #insignificant margin coefs at p>0.10 across both
    margins, paper rule). Cached margin fits keep the ~pop*gen evals
    tractable. ``copula`` selects the IFM family: joe (default), gumbel
    (new in metacountregressor 1.0.154 -- upper-tail dependence,
    motivated by the Gaussian-Joe mixture win), frank, or kimeldorf
    (local CDFs). Writes joint best CSV/TeX, history CSV, and a
    convergence PNG with population traces.
    """
    sys.path.insert(0, HERE)
    from nearmiss_hb_analysis import fit_model, _nb_cdf, _joint_pmf_vec
    from nearmiss_hb_analysis import _rebuild_mu
    from nearmiss_hb_analysis import (_COPULA_FNS, _COPULA_BOUNDS,
                                      _COPULA_INIT)
    copula = str(copula).lower()
    if copula == "joe":
        try:
            from copula_extensions import _joe_cop as _cop_fn
        except Exception:
            from nearmiss_crash_analysis import _joe_cop as _cop_fn  # fallback
        _cop_init, _cop_bounds = 2.0, (1.001, 20.0)
    elif copula == "gumbel":
        # Gumbel CDF via the package's tested log-CDF (1.0.154+); hard gate.
        check_mcr_version(min_ver=(1, 0, 154))
        from metacountregressor.bivariate_copula import gumbel_logcdf

        def _cop_fn(u, v, theta):
            return np.asarray(np.exp(gumbel_logcdf(
                np.asarray(u, dtype=float), np.asarray(v, dtype=float),
                float(theta))), dtype=float)
        _cop_init, _cop_bounds = 2.0, (1.001, 20.0)
    elif copula in _COPULA_FNS:
        _cop_fn = _COPULA_FNS[copula]
        _cop_init = _COPULA_INIT[copula]
        _cop_bounds = _COPULA_BOUNDS[copula]
    else:
        raise ValueError(f"unknown joint copula {copula!r}; use joe/gumbel/"
                         f"{'/'.join(sorted(_COPULA_FNS))}")
    from scipy.optimize import minimize as scipy_min
    J = len(cands)
    idx = {c: i for i, c in enumerate(cands)}
    m0 = np.zeros(2 * J, dtype=int)
    for c in hb_init:
        if c in idx:
            m0[idx[c]] = 1
    for c in cr_init:
        if c in idx:
            m0[J + idx[c]] = 1
    print(f"\n[Part S:joint] joint SparseEA-AGDS J={J} pop={pop_size} "
          f"gen={max_gen} seed={seed} copula={copula} warm-start "
          f"HB({int(m0[:J].sum())}) CR({int(m0[J:].sum())})", flush=True)
    from sparsea_agds import sparsea_agds_search
    _marg_cache = {}
    EPS = 1e-10

    def _marg(sel, y):
        key = (tuple(sorted(sel)), y)
        if key not in _marg_cache:
            _marg_cache[key] = fit_model(dfm, list(sel), y, offset_col,
                                         label="J-marg", verbose=False)
        return _marg_cache[key]

    def _joint_score(mask):
        mask = np.asarray(mask).astype(int)
        mh = [cands[i] for i in range(J) if mask[i] == 1]
        mc = [cands[i] for i in range(J) if mask[J + i] == 1]
        if len(mh) < 2 or len(mc) < 2:
            return 1e12
        try:
            m1 = _marg(mh, y_hb)
            m2 = _marg(mc, y_cr)
            if m1 is None or m2 is None:
                return 1e12
            allf = sorted(set(mh) | set(mc))
            df_c = dfm.dropna(
                subset=allf + [y_hb, y_cr, offset_col]).reset_index(drop=True)
            if len(df_c) < 100:
                return 1e12
            mu1, a1 = _rebuild_mu(df_c, m1)
            mu2, a2 = _rebuild_mu(df_c, m2)
            y1 = df_c[y_hb].values.astype(int)
            y2 = df_c[y_cr].values.astype(int)
            u1c = np.clip(_nb_cdf(y1, mu1, a1), EPS, 1 - EPS)
            u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, a1),
                                           0.0, 1 - EPS), 0.0)
            u2c = np.clip(_nb_cdf(y2, mu2, a2), EPS, 1 - EPS)
            u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, a2),
                                           0.0, 1 - EPS), 0.0)

            def _neg(t):
                pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, _cop_fn,
                                     float(t[0]))
                return -float(np.sum(np.log(pmf)))
            r = scipy_min(_neg, [_cop_init], method="L-BFGS-B",
                          bounds=[_cop_bounds],
                          options={"maxiter": 300})
            th, llj = float(r.x[0]), -float(r.fun)
            n = len(df_c)
            k = m1["k"] + m2["k"] + 1
            bic = k * np.log(n) - 2 * llj
            # penalised joint BIC: push significant margins in-search so no
            # separate joint_mask_refit pruning step is needed
            try:
                n_insig = 0
                for _m in (m1, m2):
                    _ct = _m["coef_table"]
                    _main = _ct[~_ct["Variable"].str.startswith(
                        ("RND", "chol(", "param", "ln("))
                        & (_ct["Variable"] != "(Intercept)")]
                    n_insig += int(((_main["p"] > 0.10)
                                    | _main["p"].isna()).sum())
            except Exception:
                n_insig = 0
            return float(bic) + 10.0 * n_insig
        except Exception:
            return 1e12

    # seed one individual at the warm-start mask inside the solver by
    # wrapping fitness (solver itself is randomly initialised; the warm
    # start enters via elitism once evaluated -- also evaluate it now).
    _warm_bic = _joint_score(m0)
    print(f"[Part S:joint] warm-start joint BIC={_warm_bic:.1f}", flush=True)
    out = sparsea_agds_search(_joint_score, 2 * J, pop_size=pop_size,
                              max_gen=max_gen, seed=seed)
    best = np.asarray(out["best_mask"]).astype(int)
    # elitist guard: keep warm start if AGDS never beat it
    if float(np.min(out["history_best"])) > _warm_bic:
        best = m0.copy()
        out["best_mask"] = best.copy()
        out["best_fitness"] = float(_warm_bic)
    mh = [cands[i] for i in range(J) if best[i] == 1]
    mc = [cands[i] for i in range(J) if best[J + i] == 1]
    n = len(out["history_best"])
    pd.DataFrame({"gen": np.arange(1, n + 1),
                  "best_so_far": out["history_best"],
                  "best_current": out.get("history_gen_best",
                                          out["history_best"]),
                  "mean": out.get("history_mean", [np.nan] * n),
                  "median": out.get("history_median", [np.nan] * n),
                  "min": out.get("history_min", [np.nan] * n),
                  "max": out.get("history_max", [np.nan] * n)}
                 ).to_csv(os.path.join(out_dir, "eb_search_hist_joint.csv"),
                          index=False)
    pd.DataFrame([{"copula": copula,
                   "hb_features": ";".join(sorted(mh)),
                    "crash_features": ";".join(sorted(mc)),
                    "joint_bic": float(np.min(out["history_best"])),
                    "warm_start_bic": float(_warm_bic),
                    "n_hb": len(mh), "n_cr": len(mc)}]).to_csv(
        os.path.join(out_dir, "eb_search_joint_best.csv"), index=False)
    plot_convergence_with_traces(
        out, os.path.join(out_dir, "eb_search_convergence_joint.png"),
        "Single-run SparseEA-AGDS Convergence — joint HB+crash "
        f"({copula} joint BIC)", "Joint BIC")
    with open(os.path.join(out_dir, "eb_search_joint_best.tex"), "w") as f:
        f.write("% Auto-generated by eb_searched_rp.py -- joint copula "
                "SparseEA-AGDS search.\n")
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Joint HB--crash structure searched by "
                f"SparseEA-AGDS on {copula}-copula joint BIC "
                f"(warm-start BIC $={_warm_bic:,.1f}$, searched BIC "
                f"$={float(np.min(out['history_best'])):,.1f}$)}}"
                "\\label{tab:eb_search_joint}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lr}\n      \\toprule\n")
        f.write("      Searched HB mask & " + ", ".join(sorted(mh)) +
                " \\\\\n")
        f.write("      Searched crash mask & " + ", ".join(sorted(mc)) +
                " \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write("\\end{table}\n")
    print(f"[Part S:joint] best HB({len(mh)}) CR({len(mc)}) "
          f"{copula} joint-BIC={float(np.min(out['history_best'])):.1f}",
          flush=True)
    return mh, mc, out


def run_mcr_role_search(dfm, feats, y_col, offset_col, label, out_dir,
                        n_starts=1, max_iter=300, n_max=1500,
                        max_time=1200):
    """MCR random-parameter role search (StructureEvaluatorLC, 0/1/2/3).

    Searches which coefficients stay fixed vs random-independent /
    random-correlated (normal), i.e. the metacountregressor structure
    search for the RP block. Returns the remapped (fixed, rdm, rdm_cor)
    triple plus BIC, and writes eb_search_mcr_role_<label>.csv. This is
    the "consider MCR for the random parameters" step: the RP verdict no
    longer rests only on hardcoded blocks. General SEs (JAX Hessian with
    numerical fallback) are already reported per-coefficient by
    fit_model's coef tables.
    """
    try:
        sys.path.insert(0, HERE)
        from nearmiss_hb_analysis import mcr_model_search
    except Exception as e:
        print(f"[Part S:{label}] mcr_model_search import failed: {e}",
              flush=True)
        return None
    try:
        res = mcr_model_search(dfm, y_col, list(feats), offset_col,
                               label=f"EBs-{label}", n_starts=n_starts,
                               max_iter=max_iter, R=5, max_latent_classes=1,
                               out_dir=None, N_max=n_max,
                               max_time_per_start=max_time, solver="de",
                               pop_size=20)
    except Exception as e:
        print(f"[Part S:{label}] MCR role search failed: {e}", flush=True)
        return None
    if not res:
        print(f"[Part S:{label}] MCR role search returned None", flush=True)
        return None
    pd.DataFrame([{"label": label, "bic": res["bic"],
                   "fixed_terms": ";".join(res["fixed_orig"]),
                   "rdm_ind_terms": ";".join(res["rdm_i_orig"]),
                   "rdm_cor_terms": ";".join(res["rdm_c_orig"]),
                   "latent_classes": res["lc"],
                   "nb": int(res["nb"])}]).to_csv(
        os.path.join(out_dir, f"eb_search_mcr_role_{label}.csv"), index=False)
    print(f"[Part S:{label}] MCR roles BIC={res['bic']:.1f} fixed="
          f"{res['fixed_orig']} indep={res['rdm_i_orig']} corr="
          f"{res['rdm_c_orig']}", flush=True)
    return res


def plot_rp_sd_significance(coef_csv, plot_path, title):
    """Forest-style plot of RND sd(...) estimates with general SEs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        ct = pd.read_csv(coef_csv)
    except Exception:
        return
    sd = ct[ct["Variable"].str.startswith("RND sd(")].copy()
    if len(sd) == 0:
        return
    sd["se"] = pd.to_numeric(sd["Std.Err"], errors="coerce")
    sd["est"] = pd.to_numeric(sd["Estimate"], errors="coerce")
    sd = sd.dropna(subset=["est"])
    y = np.arange(len(sd))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(sd) + 1.5)))
    ax.errorbar(sd["est"].values, y,
                xerr=1.645 * sd["se"].fillna(0).values,
                fmt="o", color="#B34700", ecolor="gray", capsize=3,
                label="90% CI (general SEs)")
    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sd["Variable"].tolist(), fontsize=8)
    ax.set_xlabel("Random-parameter SD estimate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "near_miss_hb_ecf.csv"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--skip-rp", action="store_true",
                    help="skip Part B MCR fits (fast Hauer-only mode)")
    ap.add_argument("--skip-search", action="store_true",
                    help="skip Part S SparseEA-AGDS searches (reuse CSV specs)")
    ap.add_argument("--search", default="all",
                    choices=["all", "hb", "crash", "joint", "mcr-role", "none"],
                    help="which Part-S searches to run (default all)")
    ap.add_argument("--hb-maxiter", type=int, default=60)
    ap.add_argument("--hb-popsize", type=int, default=12)
    ap.add_argument("--crash-maxiter", type=int, default=30)
    ap.add_argument("--crash-popsize", type=int, default=8)
    ap.add_argument("--joint-pop", type=int, default=20)
    ap.add_argument("--joint-gen", type=int, default=12)
    ap.add_argument("--joint-seed", type=int, default=44)
    ap.add_argument("--joint-copula", default="joe",
                    choices=["joe", "gumbel", "frank", "normal",
                             "kimeldorf"],
                    help="IFM copula family for the S3 joint search "
                         "(gumbel needs metacountregressor>=1.0.154; "
                         "normal is slow -- use sparingly)")
    ap.add_argument("--load-best", action="store_true",
                    help="load S1/S2 best masks from eb_search_best_{hb,crash}.csv "
                         "instead of the condensed specs (for --search joint/mcr-role "
                         "top-ups; job 25402029 follow-up)")
    ap.add_argument("--best-dir", default=None,
                    help="directory holding eb_search_best_*.csv when --load-best "
                         "is used with a scratch --out-dir (default: --out-dir)")
    ap.add_argument("--mcr-starts", type=int, default=1)
    ap.add_argument("--mcr-maxiter", type=int, default=300)
    ap.add_argument("--mcr-nmax", type=int, default=1500)
    ap.add_argument("--mcr-maxtime", type=int, default=1200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    HB_FEATS, CRASH_FEATS = load_specs()
    df = load_data(HB_FEATS + CRASH_FEATS)
    print(f"segments={len(df):,} crashes={int(df['ObsCrashes'].sum()):,} "
          f"HB={int(df['HB'].sum()):,}", flush=True)

    # ---- Part A: Hauer EB on searched specs (log_len; Table 31 method) ----
    import statsmodels.api as sm
    from scipy import stats as sps
    res_c, lam_c, alph_c, dc = fit_nb(df, "ObsCrashes", CRASH_FEATS, "log_len")
    res_h, lam_h, alph_h, dh = fit_nb(df, "HB", HB_FEATS, "log_len")
    print(f"Searched crash SPF ({len(CRASH_FEATS)} vars): "
          f"LL={res_c.llf:.1f} alpha={alph_c:.4f}", flush=True)
    print(f"Searched HB SPF ({len(HB_FEATS)} vars): "
          f"LL={res_h.llf:.1f} alpha={alph_h:.4f}", flush=True)
    pd.DataFrame({
        "var": ["const"] + CRASH_FEATS + ["alpha"],
        "crash_coef": np.r_[res_c.params, alph_c],
        "crash_se": np.r_[res_c.bse, np.nan],
    }).to_csv(os.path.join(args.out_dir, "eb_searched_model_features.csv"),
              index=False)

    w_c = eb_weight(alph_c, lam_c)
    eb_c = w_c * lam_c + (1.0 - w_c) * dc["ObsCrashes"].values
    w_h = eb_weight(alph_h, lam_h)
    eb_h = w_h * lam_h + (1.0 - w_h) * dh["HB"].values
    post = dc.copy()
    post["SPF_crash_searched"] = lam_c
    post["w_crash_searched"] = w_c
    post["EB_crash_searched"] = eb_c
    Spf_h = pd.Series(lam_h, index=dh["segment_id"].values)
    W_h = pd.Series(w_h, index=dh["segment_id"].values)
    E_h = pd.Series(eb_h, index=dh["segment_id"].values)
    post["SPF_hb_searched"] = post["segment_id"].map(Spf_h)
    post["w_hb_searched"] = post["segment_id"].map(W_h)
    post["EB_hb_searched"] = post["segment_id"].map(E_h)
    ref = pd.to_numeric(post.get("ExpCrashes"), errors="coerce")
    ok = ref.notna() & (ref > 0)
    lam_ref = ref[ok].values.astype(float)
    yref = post.loc[ok, "ObsCrashes"].values.astype(float)
    alph_ref = max((float(np.sum((yref - lam_ref) ** 2))
                    - float(np.sum(lam_ref)))
                   / max(float(np.sum(lam_ref ** 2)), 1e-9), 0.0)
    post["EB_ref"] = np.nan
    post.loc[ok, "EB_ref"] = (eb_weight(alph_ref, lam_ref) * lam_ref
                              + (1 - eb_weight(alph_ref, lam_ref)) * yref)
    post.to_csv(os.path.join(args.out_dir, "eb_searched_posterior.csv"),
                index=False)

    # 5-fold CV (searched specs; ref prior uses train-fold dispersion).
    rng = np.random.RandomState(RS)
    folds = 5
    idx = rng.permutation(len(df))
    sz = len(idx) // folds
    ms = {e: {"rmse": [], "mae": [], "rho": []}
          for e in ["Observed", "SPF-searched", "EB-searched", "EB(ref)"]}
    for f in range(folds):
        te = idx[f * sz:(f + 1) * sz] if f < folds - 1 else idx[f * sz:]
        tr = np.setdiff1d(np.arange(len(df)), te)
        trd, ted = df.iloc[tr].copy(), df.iloc[te].copy()
        rc, _, a_tr, _ = fit_nb(trd, "ObsCrashes", CRASH_FEATS, "log_len")
        Xte = sm.add_constant(ted[CRASH_FEATS].values.astype(float),
                             has_constant="add")
        lam_te = np.asarray(rc.predict(
            Xte, offset=ted["log_len"].values.astype(float)), float)
        yte = ted["ObsCrashes"].values.astype(float)
        wte = eb_weight(a_tr, lam_te)
        eb_te = wte * lam_te + (1 - wte) * yte
        ms["Observed"]["rmse"].append(np.sqrt(np.mean((yte - yte.mean()) ** 2)))
        ms["Observed"]["mae"].append(np.mean(np.abs(yte - yte.mean())))
        ms["Observed"]["rho"].append(1.0)
        ms["SPF-searched"]["rmse"].append(np.sqrt(np.mean((yte - lam_te) ** 2)))
        ms["SPF-searched"]["mae"].append(np.mean(np.abs(yte - lam_te)))
        ms["SPF-searched"]["rho"].append(sps.spearmanr(yte, lam_te).statistic)
        ms["EB-searched"]["rmse"].append(np.sqrt(np.mean((yte - eb_te) ** 2)))
        ms["EB-searched"]["mae"].append(np.mean(np.abs(yte - eb_te)))
        ms["EB-searched"]["rho"].append(sps.spearmanr(yte, eb_te).statistic)
        ref_tr = pd.to_numeric(trd["ExpCrashes"],
                               errors="coerce").values.astype(float)
        okr = np.isfinite(ref_tr) & (ref_tr > 0)
        a_r = max((float(np.sum((trd["ObsCrashes"].values[okr]
                                 - ref_tr[okr]) ** 2))
                   - float(np.sum(ref_tr[okr])))
                  / max(float(np.sum(ref_tr[okr] ** 2)), 1e-9), 0.0) \
            if okr.any() else a_tr
        ref_te = pd.to_numeric(ted["ExpCrashes"],
                               errors="coerce").values.astype(float)
        m_ = np.isfinite(ref_te) & (ref_te > 0)
        wr = eb_weight(a_r, ref_te[m_])
        ebr = wr * ref_te[m_] + (1 - wr) * yte[m_]
        ms["EB(ref)"]["rmse"].append(np.sqrt(np.mean((yte[m_] - ebr) ** 2)))
        ms["EB(ref)"]["mae"].append(np.mean(np.abs(yte[m_] - ebr)))
        ms["EB(ref)"]["rho"].append(sps.spearmanr(yte[m_], ebr).statistic)
    cv = pd.DataFrame({k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                       for k, v in ms.items()}).T
    cv.to_csv(os.path.join(args.out_dir, "eb_searched_crossval.csv"))
    print("\n5-fold CV (searched specs, mean over folds):", flush=True)
    print(cv.round(3).to_string(), flush=True)

    # Encompassing: crash ~ crash-searched-geo + log1p(EB_HB_searched).
    sub = post.dropna(subset=["EB_hb_searched", "SPF_crash_searched",
                              "ObsCrashes", "log_len"]).copy()
    Xg = sm.add_constant(sub[CRASH_FEATS].values.astype(float),
                        has_constant="add")
    off = sub["log_len"].values.astype(float)
    y = sub["ObsCrashes"].values.astype(float)
    fam = sm.families.Poisson()
    r_g = sm.GLM(y, Xg, offset=off, family=fam).fit(method="irls",
                                                   maxiter=2000)
    Xb = sm.add_constant(np.c_[sub[CRASH_FEATS].values.astype(float),
                               np.log1p(sub["EB_hb_searched"].values)],
                        has_constant="add")
    r_b = sm.GLM(y, Xb, offset=off, family=fam).fit(method="irls",
                                                   maxiter=2000)
    lr = float(r_g.deviance - r_b.deviance)
    p_lr = float(sps.chi2.sf(lr, df=1))
    print(f"\nEncompassing (searched EB-HB): LR={lr:.1f} p={p_lr:.2e} "
          f"BIC_geo={r_g.bic:.1f} BIC_geo+EBhb={r_b.bic:.1f}", flush=True)
    pd.DataFrame({"stat": ["LR", "p", "BIC_geo", "BIC_geo_plus_EB_HB",
                           "diff_BIC"],
                  "value": [lr, p_lr, r_g.bic, r_b.bic, r_g.bic - r_b.bic]}
                 ).to_csv(os.path.join(args.out_dir,
                                       "eb_searched_encompassing.csv"),
                          index=False)

    # Figure (same 3-panel layout as eb_screening.png, searched priors).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.hist(w_c, bins=40, color="#4477AA", edgecolor="white")
    ax.set_xlabel("EB weight w (prior reliability)")
    ax.set_ylabel("Segments")
    ax.set_title("(a) Shrinkage toward searched prior mean")
    ax = axes[1]
    n = min(len(post), 4000)
    ax.scatter(post["ObsCrashes"].head(n), post["EB_crash_searched"].head(n),
               s=4, alpha=0.35, c="#CC6677")
    lim = [0, max(post["EB_crash_searched"].max(), post["ObsCrashes"].max())]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Observed crashes")
    ax.set_ylabel("EB posterior mean (searched prior)")
    ax.set_title("(b) EB vs observed")
    ax = axes[2]
    ax.scatter(post["SPF_crash_searched"].head(n),
               post["EB_crash_searched"].head(n),
               s=4, alpha=0.35, c="#88CCEE")
    ax.plot([0, post["SPF_crash_searched"].max()],
            [0, post["SPF_crash_searched"].max()], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Searched prior-mean prediction")
    ax.set_ylabel("EB posterior mean")
    ax.set_title("(c) EB vs searched prior mean")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "eb_searched_screening.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Table 31 replacement: union of the two searched specs ("--" = not in
    # that model's spec, which is exactly why the old table was wrong).
    try:
        from nearmiss_hb_analysis import VLABEL
    except Exception:
        VLABEL = {}
    def _vv(v):
        return VLABEL.get(v, {"AADT": "AADT (std)"}.get(v, v))
    union = []
    for v in ["const"] + CRASH_FEATS + [v for v in HB_FEATS
                                        if v not in CRASH_FEATS]:
        union.append(v)
    # map Hauer results by var name
    mc = dict(zip(["const"] + CRASH_FEATS, res_c.params))
    sc = dict(zip(["const"] + CRASH_FEATS, res_c.bse))
    zc = dict(zip(["const"] + CRASH_FEATS, res_c.tvalues))
    pc = dict(zip(["const"] + CRASH_FEATS, res_c.pvalues))
    mh = dict(zip(["const"] + HB_FEATS, res_h.params))
    sh = dict(zip(["const"] + HB_FEATS, res_h.bse))
    zh = dict(zip(["const"] + HB_FEATS, res_h.tvalues))
    ph = dict(zip(["const"] + HB_FEATS, res_h.pvalues))
    def cell(d, v):
        return f"{float(d[v]):.3f}" if v in d else "--"
    tex = []
    tex.append("% Auto-generated by eb_searched_rp.py -- Table 31 replacement.")
    tex.append("% Searched condensed specs (not the 30-var baseline); "
               "\"--\" = variable not in that response's spec.")
    tex.append("\\begin{table}[!ht]\\footnotesize\\caption{Empirical Bayes "
               "prior-model coefficients on the searched condensed "
               f"specifications (crash: {len(CRASH_FEATS)} vars; hard braking: "
               f"{len(HB_FEATS)} vars; $N=10{{,}}780$; Hauer moment "
               "$\\hat\\alpha$, \\texttt{log\\_len} offset -- same method as "
               "Table~\\ref{tab:eb_spf}). NB count regressions used as "
               "shrinkage priors, not HSM-style SPFs.}\\label{tab:eb_spf_searched}"
               "\\begin{center}\\begin{tabular}{lrrrr|lrrrr}\\toprule")
    tex.append("& \\multicolumn{4}{c}{Crash \\textbf{prior (searched)}}"
               "& \\multicolumn{4}{c}{Hard-braking \\textbf{prior (searched)}}\\\\")
    tex.append("Variable & $\\hat\\beta$ & SE & $z$ & $p$ "
               "& $\\hat\\beta$ & SE & $z$ & $p$\\\\\\midrule")
    for v in union:
        if v == "const":
            lab = "Const"
        else:
            lab = _vv(v)
        tex.append(f"{lab} & {cell(mc, v)} & {cell(sc, v)} & {cell(zc, v)} & "
                   f"{(f'{float(pc[v]):.3f}' if v in pc else '--')} "
                   f"& {cell(mh, v)} & {cell(sh, v)} & {cell(zh, v)} & "
                   f"{(f'{float(ph[v]):.3f}' if v in ph else '--')}\\\\")
    tex.append(f"$\\hat\\alpha$ & \\multicolumn{{4}}{{c}}{{{alph_c:.4f}}}"
               f"& \\multicolumn{{4}}{{c}}{{{alph_h:.4f}}}\\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{center}\\end{table}")
    with open(os.path.join(args.out_dir, "eb_searched_spf.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(tex))
    print("wrote eb_searched_spf.tex (Table 31 replacement)", flush=True)

    # ---- Part B: MCR fixed vs random-parameter NB on searched specs ----
    # dfm is built in both modes so Part S searches always have log_len /
    # HB_Total available even under --skip-rp.
    dfm = pd.read_csv(args.data, low_memory=False).dropna(subset=["ObsCrashes"])
    for c in sorted(set(HB_FEATS) | set(CRASH_FEATS)):
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce").fillna(0.0)
    dfm["seg_len_km"] = (pd.to_numeric(dfm.get("segment_length_m", 0),
                                       errors="coerce").fillna(0) / 1000.0
                         ).clip(lower=0.001)
    dfm["log_len"] = np.log(dfm["seg_len_km"])
    if "HB_Total" in dfm.columns:
        dfm["HB_Total"] = pd.to_numeric(dfm["HB_Total"],
                                        errors="coerce").fillna(0.0)
    else:
        dfm["HB_Total"] = pd.to_numeric(dfm.get("nm_total", 0),
                                        errors="coerce").fillna(0.0)
    if args.skip_rp:
        print("Skipping Part B fits (--skip-rp); continuing to Part S.",
              flush=True)
    else:
        check_mcr_version()
        sys.path.insert(0, HERE)
        from nearmiss_hb_analysis import fit_model
        from scipy import stats as st
        # log_len must exist for fit_model offset; load_data already built it.
    if not args.skip_rp:
        rows = []
        for label, feats, y, rdm, rdm_cor in [
            ("HB-searched fixed", HB_FEATS, "HB_Total", None, None),
            ("HB-searched random", HB_FEATS, "HB_Total", HB_RDM, HB_RDM_COR),
            ("Crash-searched fixed", CRASH_FEATS, "ObsCrashes", None, None),
            ("Crash-searched random", CRASH_FEATS, "ObsCrashes",
             CR_RDM, CR_RDM_COR),
        ]:
            print(f"\n=== Fitting {label} ===", flush=True)
            m = fit_model(dfm, feats, y, "log_len", label=label,
                          rdm_terms=rdm, rdm_cor_terms=rdm_cor, draws_R=200)
            print(f"  LL={m['ll']:.1f} BIC={m['bic']:.1f} k={m['k']} "
                  f"converged={m['converged']}", flush=True)
            ct = m["coef_table"]
            sd = ct[ct["Variable"].str.startswith("RND sd(")]
            n_sig = int((sd["p"] < 0.05).sum()) if len(sd) else 0
            rows.append({"model": label, "ll": m["ll"], "aic": m["aic"],
                         "bic": m["bic"], "k": m["k"], "n": m["n"],
                         "converged": m["converged"], "n_sd": len(sd),
                         "n_sd_sig05": n_sig})
            tag = label.replace(" ", "_").replace("-", "_")
            ct.to_csv(os.path.join(args.out_dir, f"eb_searched_rp_{tag}.csv"),
                      index=False)
        res = pd.DataFrame(rows)
        for resp, fx, rn in [("HB", "HB-searched fixed", "HB-searched random"),
                             ("Crash", "Crash-searched fixed",
                              "Crash-searched random")]:
            a = res.set_index("model")
            lrv = 2 * (a.loc[rn, "ll"] - a.loc[fx, "ll"])
            dfr = int(a.loc[rn, "k"] - a.loc[fx, "k"])
            pv = float(1 - st.chi2.cdf(lrv, dfr))
            print(f"{resp}: LR={lrv:.1f} df={dfr} p={pv:.4g}", flush=True)
            res.loc[res["model"] == rn, "lr_vs_fixed"] = lrv
            res.loc[res["model"] == rn, "lr_df"] = dfr
            res.loc[res["model"] == rn, "lr_p"] = pv
        res.to_csv(os.path.join(args.out_dir, "eb_searched_rp.csv"), index=False)
        # constrained verdict (same gates as refit_significant.py step 5)
        verdicts = []
        for _, r in res[res["model"].str.endswith("random")].iterrows():
            ct = pd.read_csv(os.path.join(
                args.out_dir,
                f"eb_searched_rp_{r['model'].replace(' ', '_').replace('-', '_')}.csv"))
            sd = ct[ct["Variable"].str.startswith("RND sd(")]
            means = ct[ct["Variable"].str.startswith("RND mean(")]
            ok_sd = bool((sd["p"] < 0.10).all()) if len(sd) else True
            ok_mean = bool((means["Estimate"].abs() < 5).all()) if len(means) else True
            try:
                _cti = ct.set_index("Variable")
                _dp = _cti.loc["alpha", "p"] if "alpha" in _cti.index else _cti.loc["ln(overdispersion)", "p"]
                disp_p = float(_dp)
            except KeyError:
                disp_p = np.nan
            v = ("ACCEPT" if (ok_sd and ok_mean and disp_p < 0.10) else "REJECT")
            verdicts.append({"model": r["model"], "ok_sd_p010": ok_sd,
                             "ok_mean_lt5": ok_mean, "disp_p": disp_p,
                             "verdict": v})
        pd.DataFrame(verdicts).to_csv(os.path.join(args.out_dir,
                                                   "eb_searched_verdict.csv"),
                                      index=False)
        with open(os.path.join(args.out_dir, "eb_searched_rp.tex"), "w") as f:
            f.write("% Auto-generated by eb_searched_rp.py -- RP check on searched specs.\n")
            f.write("\\begin{table}[!ht]\n")
            f.write("  \\caption{Fixed vs.\\ random-parameter NB on the searched "
                    "condensed specifications "
                    f"(HB {len(HB_FEATS)} vars; crash {len(CRASH_FEATS)} vars; "
                    "$N=10{,}780$, $R=200$ Halton draws, "
                    "\\texttt{log\\_len} offset)}\\label{tab:eb_searched_rp}\n")
            f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n"
                    "      \\toprule\n"
                    "      Model & LL & AIC & BIC & $k$ & $\\Delta$BIC vs.\\ fixed "
                    "\\\\\n      \\midrule\n")
            for _, r in res.iterrows():
                base = res[(res["model"].str.split().str[0]
                            == r["model"].split()[0])
                           & (res["model"].str.endswith("fixed"))]
                d = r["bic"] - float(base["bic"].iloc[0]) \
                    if r["model"].endswith("random") else 0.0
                f.write(f"      {r['model']} & {r['ll']:,.1f} & {r['aic']:,.1f} & "
                        f"{r['bic']:,.1f} & {int(r['k'])} & {d:+,.1f} \\\\\n")
            f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
            f.write("  \\emph{Note:} Random blocks follow the MCR role search "
                    "(HB: random-independent IC/IV\\_M plus random-correlated "
                    "PA\\_COM1/RSDSO/RSDSDL5; crashes: random-independent "
                    "PSDS1T24/RSDSDG10 plus random-correlated SL/RSDSDL5). "
                    "SD significance and the significance-constrained verdict "
                    "(every retained SD $p<0.10$, $|$mean$|<5$, dispersion "
                    "$p<0.10$) are in \\texttt{eb\\_searched\\_rp.csv} and "
                    "\\texttt{eb\\_searched\\_verdict.csv}. Full coefficient "
                    "tables: \\texttt{eb\\_searched\\_rp\\_*.csv}.}\n")
            f.write("\\end{table}\n")
        print("Wrote eb_searched_rp.tex / eb_searched_rp.csv / "
              "eb_searched_verdict.csv", flush=True)
        for _csv, _tag in [
            ("eb_searched_rp_HB_searched_random.csv", "HB-searched random"),
            ("eb_searched_rp_Crash_searched_random.csv", "Crash-searched random"),
        ]:
            try:
                plot_rp_sd_significance(
                    os.path.join(args.out_dir, _csv),
                    os.path.join(args.out_dir,
                                 f"eb_searched_rp_sds_{_tag.split('-')[0]}.png"),
                    f"Random-parameter SDs — {_tag} (general SEs, 90% CI)")
            except Exception as e:
                print(f"RP SD plot failed for {_tag}: {e}", flush=True)

    # ---- Part S: best-model searches inside this job ----
    # S1 core HB via metacount (SparseEA-AGDS + MCR NB penalised BIC).
    # S2 core crash via metacountregressor (same estimator).
    # S3 joint HB+crash copula search (SparseEA-AGDS on Joe joint BIC).
    # S4 MCR role search for the random-parameter blocks (general SEs).
    # Every search writes a convergence PNG showing best-so-far, best
    # current, population mean/median, min-max band and faint
    # per-individual traces (see plot_convergence_with_traces).
    if args.skip_search or args.search == "none":
        print("Skipping Part S searches (--skip-search / --search none).",
              flush=True)
        return
    check_mcr_version()
    want = set(["hb", "crash", "joint", "mcr-role"]) \
        if args.search == "all" else set([args.search])
    # dfm already holds log_len / HB_Total for Part B; extend to the full
    # 30-candidate set for the searches.
    for c in GEO_CANDIDATES:
        if c in dfm.columns:
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce").fillna(0.0)
    cands = [c for c in GEO_CANDIDATES
             if c in dfm.columns
             and pd.to_numeric(dfm[c], errors="coerce").std() > 1e-10]
    print(f"[Part S] candidates J={len(cands)}: {cands}", flush=True)
    hb_best, hb_model = list(HB_FEATS), None
    cr_best, cr_model = list(CRASH_FEATS), None
    if args.load_best:
        _bdir = args.best_dir or args.out_dir
        _loaded = {}
        for _tag in ["hb", "crash"]:
            try:
                _df = pd.read_csv(
                    os.path.join(_bdir, f"eb_search_best_{_tag}.csv"))
                _fl = [f for f in _df["feature"].astype(str).tolist()
                       if f in cands]
                if _fl:
                    _loaded[_tag] = _fl
                    print(f"[Part S] --load-best {_tag}: {len(_fl)} feats "
                          f"from {_bdir}", flush=True)
            except Exception as e:
                print(f"[Part S] --load-best {_tag} failed ({e}); using "
                      f"condensed.", flush=True)
        if "hb" in _loaded:
            hb_best = _loaded["hb"]
        if "crash" in _loaded:
            cr_best = _loaded["crash"]
    if "hb" in want:
        hb_best, hb_model, _ = run_sparsea_core_search(
            dfm, cands, "HB_Total", "log_len", "HB core (metacount)",
            seed=42, maxiter=args.hb_maxiter, popsize=args.hb_popsize,
            p_threshold=0.10, penalty=10.0, out_dir=args.out_dir, tag="hb")
    if "crash" in want:
        cr_best, cr_model, _ = run_sparsea_core_search(
            dfm, cands, "ObsCrashes", "log_len", "Crash core "
            "(metacountregressor)", seed=43, maxiter=args.crash_maxiter,
            popsize=args.crash_popsize, p_threshold=0.10, penalty=10.0,
            out_dir=args.out_dir, tag="crash")
    # comparison vs the condensed specs used in Parts A/B
    try:
        pd.DataFrame([
            {"response": "HB", "spec": "condensed (Parts A/B)",
             "n": len(HB_FEATS), "features": ";".join(HB_FEATS)},
            {"response": "HB", "spec": "SparseEA-AGDS searched (Part S)",
             "n": len(hb_best), "features": ";".join(hb_best)},
            {"response": "Crash", "spec": "condensed (Parts A/B)",
             "n": len(CRASH_FEATS), "features": ";".join(CRASH_FEATS)},
            {"response": "Crash", "spec": "SparseEA-AGDS searched (Part S)",
             "n": len(cr_best), "features": ";".join(cr_best)},
        ]).to_csv(os.path.join(args.out_dir, "eb_search_vs_condensed.csv"),
                   index=False)
    except Exception as e:
        print(f"[Part S] vs_condensed write failed: {e}", flush=True)
    if "joint" in want:
        try:
            run_joint_copula_sparsea_search(
                dfm, cands, "HB_Total", "ObsCrashes", "log_len",
                hb_best, cr_best, args.out_dir,
                pop_size=args.joint_pop, max_gen=args.joint_gen,
                seed=args.joint_seed, copula=args.joint_copula)
        except Exception as e:
            print(f"[Part S:joint] FAILED: {e}", flush=True)
    if "mcr-role" in want:
        # MCR StructureEvaluatorLC role search on the searched best specs:
        # decides fixed vs random-indep vs random-corr per coefficient
        # (normal). General SEs come from fit_model's JAX Hessian +
        # numerical fallback and are plotted per SD above.
        for _tag, _feats, _y in [("HB", hb_best, "HB_Total"),
                                 ("Crash", cr_best, "ObsCrashes")]:
            try:
                run_mcr_role_search(dfm, _feats, _y, "log_len", _tag,
                                    args.out_dir,
                                    n_starts=args.mcr_starts,
                                    max_iter=args.mcr_maxiter,
                                    n_max=args.mcr_nmax,
                                    max_time=args.mcr_maxtime)
            except Exception as e:
                print(f"[Part S:{_tag}] MCR role search FAILED: {e}",
                      flush=True)
    print("Part S searches complete.", flush=True)


if __name__ == "__main__":
    main()
