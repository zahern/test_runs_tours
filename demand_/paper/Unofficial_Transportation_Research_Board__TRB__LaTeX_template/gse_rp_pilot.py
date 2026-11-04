"""GSE structured random-parameter pilot (metacountregressor>=1.0.177).

Unrestricted random-parameter blocks keep failing the significance verdict
(degenerate means, n.s. SDs; see eb_searched_verdict.csv: REJECT/REJECT).
1.0.177 exposes the canonical structural equation
(beta_nk = T(beta_k + delta'z_n + gamma*g_nk + exp(lnsig_k + omega'w_n)*nu))
via manual_spec keys hetro_in_means / hetro_in_variances / gse_cols+gse_scores
(see random_parameter_structure.py in the 1.0.177 wheel).

Arms per response (searched specs; HB 10 vars / crash 8 vars; same MCR roles
as run_rp_check.py; R=200 Halton draws; log_len offset):
  A  unstructured RP ............... baseline replicate (expect REJECT)
  B  + hetro_in_means [distinct helpers] .... mean heterogeneity
  C  + hetro_in_means + hetro_in_variances [distinct helpers]
  D  + GSE gradient scores only ...... best-effort; SKIP with reason on any
                                     API failure (never fails the job)

Per-term distributions (no forced normal): lognormal for positive-only effects.
Heterogeneity helpers must be DISTINCT from fixed/random (MCR 1.0.177 role exclusivity).

Estimation mirrors nearmiss_hb_analysis.fit_model exactly (standardise,
build_model_from_manual_spec, L-BFGS-B 5000, JAX SEs w/ scipy fallback).
New pi blocks are labelled generically so unknown keys cannot crash the
table builder. Verdict per arm: every retained RND SD p<0.10 AND every
|RND mean|<5 AND dispersion p<0.10 -> ACCEPT else REJECT.

Outputs: gse_rp_pilot.csv (arm fits), gse_rp_pilot_verdict.csv,
         gse_rp_pilot.tex (summary table, tab:gse_rp_pilot).
Usage (HPC): python -u gse_rp_pilot.py [--data ...] [--out-dir ...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
MCR_MIN_VERSION = (1, 0, 177)
ALPHA = 0.10

HB_FEATS = ["AADT", "IC", "IP", "IV_M", "PA_COM1",
            "PSDSG24", "RSDSDL5", "RSDSO", "SKD_SP", "SL"]
CRASH_FEATS = ["AADT", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "RSDSDG10", "RSDSDL5", "SL"]
# Per-term distributions (not forced normal). lognormal for positive-only effects.
HB_RDM = ["IC:normal", "IV_M:lognormal"]
HB_RDM_COR = ["PA_COM1:normal", "RSDSO:normal", "RSDSDL5:normal"]
CR_RDM = ["PSDS1T24:lognormal", "RSDSDG10:lognormal"]
CR_RDM_COR = ["SL:normal", "RSDSDL5:normal"]
# Heterogeneity helpers must be DISTINCT from the fixed/random terms
# (MCR 1.0.177 role exclusivity): a variable cannot be both fixed and a
# hetro_in_means/variances helper. Pick helpers outside each spec.
HB_HELPERS = ["IV_VL", "NL", "IV_L"]      # not in HB fixed/random
CR_HELPERS = ["NL", "SKD_SP", "IV_L"]     # not in Crash fixed/random


def _gate():
    try:
        import metacountregressor as mcr
    except Exception:
        print("FATAL: metacountregressor not importable", flush=True)
        sys.exit(2)
    try:
        from importlib.metadata import version as _pkg_version
        ver = _pkg_version("metacountregressor")
    except Exception:
        ver = getattr(mcr, "__version__", "0.0.0")
    print(f"metacountregressor {ver} (required >= 1.0.177)", flush=True)
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
    if tup < MCR_MIN_VERSION:
        print(f"FATAL: metacountregressor {ver} < 1.0.177 -- refusing to run",
              flush=True)
        sys.exit(2)


def _nb2_grad_scores(y, mu, alpha, X):
    """Per-segment score contributions d(logL_n)/d(beta) for fixed NB2 betas.

    GSE-spirit gradient scores from the fitted fixed model (exact gammaln
    reference, NumPy only): s_nk = d l_n / d eta_n * x_nk with
    d l/d eta = (y - mu) / (1 + alpha*mu) for NB2 (Var = mu + alpha mu^2).
    Returns standardised (mean 0 / sd 1) scores, zeroing flat columns.
    """
    w = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / (
        1.0 + float(alpha) * np.asarray(mu, dtype=float))
    G = w[:, None] * np.asarray(X, dtype=float)
    mu_g, sd_g = G.mean(axis=0, keepdims=True), G.std(axis=0, keepdims=True)
    out = (G - mu_g) / np.where(sd_g < 1e-8, 1.0, sd_g)
    out[:, (sd_g < 1e-8).ravel()] = 0.0
    return out


PENALTY = 10.0  # per insignificant/degenerate coefficient (paper rule)


def _unified_pbic(m):
    """Penalised BIC pushing significant models in-search.

    pBIC = BIC + 10*(fixed insig + RND-sd insig + untestable chol +
    degenerate |mean|>=5 + dispersion insig). Untestable (NaN/--) counts
    as insignificant so unidentified blocks cannot win on raw BIC.
    """
    import pandas as pd

    ct = m["coef_table"]
    main = ct[~ct["Variable"].str.startswith(("RND", "chol(", "param", "ln("))
               & ~ct["Variable"].isin(["(Intercept)", "alpha"])
               & (ct["Variable"] != "(Intercept)")]
    n_fix = int(((main["p"] > ALPHA) | main["p"].isna()).sum())
    sd_rows = ct[ct["Variable"].str.startswith("RND sd(")]
    if len(sd_rows) == 0:
        sd_rows = ct[ct["Variable"].str.contains("sd", case=False)
                     & ~ct["Variable"].isin(["(Intercept)"]
                                            + [_V(f) for f in []])
                     & (ct["Variable"] != "ln(overdispersion)")]
    # restrict fallback to non-fixed rows only
    sd_rows = sd_rows[~sd_rows["Variable"].isin(
        ["(Intercept)", "ln(overdispersion)"])]
    n_sd = int(((sd_rows["p"] > ALPHA) | sd_rows["p"].isna()).sum()) if len(sd_rows) else 0
    chol_rows = ct[ct["Variable"].str.startswith("chol(")]
    n_chol = int(((chol_rows["p"] > ALPHA) | chol_rows["p"].isna()).sum()) if len(chol_rows) else 0
    mean_rows = ct[ct["Variable"].str.startswith("RND mean(")]
    n_degen = int((mean_rows["Estimate"].abs() >= 5).sum()) if len(mean_rows) else 0
    # dispersion is labelled "alpha" (display scale) in these tables
    # (older tables used "ln(overdispersion)"): accept either
    try:
        _cti = ct.set_index("Variable")
        _dp = _cti.loc["alpha", "p"] if "alpha" in _cti.index else _cti.loc["ln(overdispersion)", "p"]
        disp_p = float(_dp)
    except KeyError:
        disp_p = float("nan")
    n_disp = 0 if (pd.notna(disp_p) and disp_p <= ALPHA) else 1
    n_insig = n_fix + n_sd + n_chol + n_degen + n_disp
    return float(m["bic"]) + PENALTY * n_insig, n_insig


def _fit_structured(df, feats, y_col, label, rdm, rdm_cor, helpers,
                    variances_on, gse_pack, draws_R=500):
    """Structured-RP fit mirroring fit_model; returns (model_dict, verdict)."""
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import minimize as scipy_min
    from scipy import stats as scipy_stats
    from dataclasses import replace
    import jax.numpy as jnp
    from nearmiss_hb_analysis import (
        _V, _scipy_hessian_se, build_model_from_manual_spec,
        build_param_index, compute_standard_errors, mixed_model_loglik,
    )

    df_m = df.dropna(subset=feats + [y_col, "log_len"]).reset_index(drop=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_m[feats].values.astype(float))
    for i in range(len(feats)):
        df_m[f"__f{i}__"] = Xs[:, i]
    fstd = [f"__f{i}__" for i in range(len(feats))]
    col2f = {feat: f"__f{i}__" for i, feat in enumerate(feats)}

    def _remap(terms):
        out = []
        for t in terms or []:
            base, sep, dist = t.partition(":")
            std = col2f.get(base, base)
            out.append(std if not sep else f"{std}:{dist}")
        return out

    rdm_s = _remap(rdm)
    rdmc_s = _remap(rdm_cor)

    # Random parameters replace their fixed effects
    rdm_bases = {t.split(":")[0] for t in (rdm or []) + (rdm_cor or [])}
    fstd_active = [col2f[f] for f in feats if f not in rdm_bases]

    # Heterogeneity helpers under 1.0.177 role exclusivity: a helper must be
    # a DISTINCT variable (cannot also be fixed or random). In-spec helpers
    # are skipped; out-of-spec helpers get dedicated __h{i}__ columns
    # (centred, orthogonalised against the fixed design, scaled).
    _used_bases = set(fstd_active) | {t.split(":")[0] for t in (rdm_s or []) + (rdmc_s or [])}
    help_s = []
    for hi, h in enumerate(helpers or []):
        if h not in df_m.columns:
            continue
        # In-spec helper (already fixed or random) is illegal under 1.0.177
        # role exclusivity -> skip it. Out-of-spec helpers (present in the
        # data but not in this spec) get a dedicated __h{i}__ column.
        if h in col2f and col2f[h] in _used_bases:
            continue
        v = df_m[h].values.astype(float)
        v = v - v.mean()
        X_fixed = df_m[fstd_active].values if fstd_active else np.ones((len(df_m), 1))
        X_fixed = np.column_stack([np.ones(len(df_m)), X_fixed])
        beta, *_ = np.linalg.lstsq(X_fixed, v, rcond=None)
        resid = v - X_fixed @ beta
        resid = resid - resid.mean()
        if resid.std() > 1e-12:
            resid = resid / resid.std()
        hc = f"__h{hi}__"
        df_m[hc] = resid
        help_s.append(hc)
    if (helpers or []) and not help_s:
        # all requested helpers were in-spec: heterogeneity inexpressible
        # under 177 exclusivity with these helpers
        raise RuntimeError(
            "SKIP: het helpers " + ",".join(helpers or []) +
            " overlap the fixed/random spec (1.0.177 role exclusivity)")

    # 177+ also forbids one column under TWO hetro roles: means and variances
    # need DISTINCT columns -> clone the orthogonalised helpers as __hv{i}__.
    het_var_s = []
    if variances_on:
        for vi, hc in enumerate(help_s):
            hvc = f"__hv{vi}__"
            df_m[hvc] = df_m[hc].values
            het_var_s.append(hvc)

    y_raw = df_m[y_col].values.astype(float)
    df_m["_id"] = np.arange(len(df_m))
    df_m["_off"] = df_m["log_len"].values.astype(float)

    manual_spec = {
        "fixed_terms": fstd_active, "rdm_terms": rdm_s, "rdm_cor_terms": rdmc_s,
        "grouped_terms": [],
        "hetro_in_means": help_s,
        "hetro_in_variances": het_var_s,
        "transformations": ["no"] * len(fstd_active),
        "dispersion": 1, "latent_classes": 1,
    }
    if gse_pack is not None:
        _, gscores = gse_pack
        G = np.asarray(gscores, dtype=float)
        if G.ndim != 2:
            raise RuntimeError("SKIP: GSE scores must be (N, K_random)")
        # Distinct __g{i}__ columns (order = cor-block then ind-block, as
        # scored) to satisfy 177 role exclusivity; mapping is positional.
        g_new = []
        for gi in range(G.shape[1]):
            gc = f"__g{gi}__"
            df_m[gc] = G[:, gi]
            g_new.append(gc)
        manual_spec["gse_cols"] = g_new
        manual_spec["gse_scores"] = G

    data, spec = build_model_from_manual_spec(
        df_m, manual_spec, id_col="_id", y_col=y_col, offset_col="_off",
        R=draws_R)
    from dataclasses import replace as _replace
    spec = _replace(spec, model="nb")
    pi = build_param_index(spec)
    K = pi["total_params"]

    init = np.zeros(K)
    init[pi["fixed"][0]:pi["fixed"][1]] = 0.0
    init[pi["fixed"][0]] = np.log(max(y_raw.mean(), 0.01))
    # SDs and Cholesky init at 0.0 -> exp(0)=1 SD, zero correlation
    for key in ("ind_mean", "ind_sd", "cor_mean", "chol", "group_mean",
                "group_sd", "het_mean", "het_var", "het_gamma", "gamma_gse",
                "gse_gamma"):
        if key in pi:
            s, e = pi[key]
            init[s:e] = 0.0

    def obj(p):
        return float(mixed_model_loglik(jnp.array(p), data, spec, indivi=False))

    # Multi-start: jitter fixed effects and SD blocks; keep best LL.
    best_params = None
    best_ll = -np.inf
    best_se = None
    rng = np.random.default_rng(12345)
    for attempt in range(5):
        if attempt == 0:
            start = init
        else:
            start = init.copy()
            f_start, f_end = pi["fixed"]
            start[f_start:f_end] += rng.normal(0, 0.1, size=f_end - f_start)
            for key in ("ind_sd", "chol", "group_sd"):
                if key in pi:
                    s, e = pi[key]
                    start[s:e] += rng.normal(0, 0.2, size=e - s)
        res = scipy_min(obj, start, method="L-BFGS-B", options={"maxiter": 5000})
        if res.success:
            ll_attempt = -obj(res.x)
            if ll_attempt > best_ll:
                best_ll = ll_attempt
                best_params = res.x
                try:
                    se_jax = np.asarray(compute_standard_errors(jnp.array(res.x), obj))
                    if np.isfinite(se_jax).any() and np.nanmax(np.abs(se_jax)) < 50:
                        best_se = se_jax
                except Exception:
                    pass
    if best_params is None:
        best_params = res.x
        best_se = None
    params = best_params
    n = len(df_m)
    ll = best_ll
    bic = K * np.log(n) - 2 * ll

    if best_se is None:
        best_se = _scipy_hessian_se(obj, params)
    se = np.where(np.abs(best_se) > 50, np.nan, best_se)

    # z/p on RAW scale initially; will be overridden for SD/chol/disp rows below
    z_raw = np.where(np.isfinite(se) & (se > 1e-12), params / se, np.nan)
    p_raw = np.where(np.isfinite(z_raw), 2 * (1 - scipy_stats.norm.cdf(np.abs(z_raw))),
                     np.nan)

    # Delta-method display transforms for SD/chol/dispersion
    def _exp_display(raw, se_raw, clip_lo=-12.0, clip_hi=6.0):
        raw = np.asarray(raw, dtype=float)
        se_raw = np.asarray(se_raw, dtype=float)
        clipped = np.clip(raw, clip_lo, clip_hi)
        disp = np.exp(clipped)
        return disp, disp * se_raw

    def _disp_display(raw, se_raw):
        raw = np.asarray(raw, dtype=float)
        se_raw = np.asarray(se_raw, dtype=float)
        clipped = np.clip(raw, -12.0, 12.0)
        disp = np.exp(clipped)
        return disp, disp * se_raw

    est = params[:K].copy()
    se_disp = se[:K].copy()
    z_disp = z_raw[:K].copy()
    p_disp = p_raw[:K].copy()

    if "ind_sd" in pi:
        ss, es = pi["ind_sd"]
        sd_disp, sd_se = _exp_display(params[ss:es], se[ss:es])
        est[ss:es] = sd_disp
        se_disp[ss:es] = sd_se
        z_disp[ss:es] = np.where(np.isfinite(sd_se) & (sd_se > 1e-12),
                                 sd_disp / sd_se, np.nan)
        p_disp[ss:es] = np.where(np.isfinite(z_disp[ss:es]),
                                 2*(1-scipy_stats.norm.cdf(np.abs(z_disp[ss:es]))), np.nan)

    if "chol" in pi:
        s_ch, e_ch = pi["chol"]
        mcor = len(rdmc_s)
        if mcor > 0:
            diag_offsets = [i * (i + 3) // 2 for i in range(mcor)]
            chol_disp = params[s_ch:e_ch].copy()
            chol_se = se[s_ch:e_ch].copy()
            chol_z = z_raw[s_ch:e_ch].copy()
            chol_p = p_raw[s_ch:e_ch].copy()
            for d_idx in diag_offsets:
                if d_idx < (e_ch - s_ch):
                    d_raw = params[s_ch + d_idx]
                    d_se = se[s_ch + d_idx]
                    d_disp, d_se_disp = _exp_display(d_raw, d_se)
                    chol_disp[d_idx] = d_disp
                    chol_se[d_idx] = d_se_disp
                    chol_z[d_idx] = d_disp / d_se_disp if (np.isfinite(d_se_disp) and d_se_disp > 1e-12) else np.nan
                    chol_p[d_idx] = 2*(1-scipy_stats.norm.cdf(abs(chol_z[d_idx]))) if np.isfinite(chol_z[d_idx]) else np.nan
            est[s_ch:e_ch] = chol_disp
            se_disp[s_ch:e_ch] = chol_se
            z_disp[s_ch:e_ch] = chol_z
            p_disp[s_ch:e_ch] = chol_p

    if "dispersion" in pi:
        dpos = pi["dispersion"]
        dpos = dpos[0] if isinstance(dpos, (tuple, list)) else int(dpos)
        d_raw = params[dpos]
        d_se = se[dpos]
        d_disp, d_se_disp = _disp_display(d_raw, d_se)
        est[dpos] = d_disp
        se_disp[dpos] = d_se_disp
        z_disp[dpos] = d_disp / d_se_disp if (np.isfinite(d_se_disp) and d_se_disp > 1e-12) else np.nan
        p_disp[dpos] = 2*(1-scipy_stats.norm.cdf(abs(z_disp[dpos]))) if np.isfinite(z_disp[dpos]) else np.nan

    # names mirror fit_model: fixed, RND means/SDs, Cholesky; any extra
    # 1.0.151 hetro/gse blocks fall through to generic param{i} labels
    # (verdict only gates RND-sd / RND-mean / dispersion rows)
    rdm_mean_names = []
    if "ind_mean" in pi:
        sm, em = pi["ind_mean"]
        rdm_mean_names += ["RND mean(" + _V(rdm[i].split(":")[0]) + ")"
                           for i in range(em - sm)]
    if "cor_mean" in pi:
        sc, ec = pi["cor_mean"]
        rdm_mean_names += ["RND mean(" + _V(rdm_cor[i].split(":")[0]) + ")"
                           for i in range(ec - sc)]
    rdm_sd_names = []
    if "ind_sd" in pi:
        ss, es = pi["ind_sd"]
        rdm_sd_names += ["RND sd(" + _V(rdm[i].split(":")[0]) + ")"
                         for i in range(es - ss)]
    chol_names = []
    if "chol" in pi:
        s_ch, e_ch = pi["chol"]
        n_ch = e_ch - s_ch
        mcor = len(rdmc_s)
        if mcor and n_ch == mcor * (mcor + 1) // 2:
            cvars = [_V(t.split(":")[0]) for t in rdmc_s]
            for i in range(mcor):
                for j in range(i + 1):
                    chol_names.append(f"chol({cvars[i]}|{cvars[j]})")
    # Random parameters replace their fixed effects: fixed names cover the
    # active (non-random) vars only, aligned to the fixed param block.
    rdm_bases_all = {t.split(":")[0] for t in (rdm or []) + (rdm_cor or [])}
    active_labels = [_V(f) for f in feats if f not in rdm_bases_all]
    names_full = (["(Intercept)"] + active_labels
                  + rdm_mean_names + rdm_sd_names + chol_names)
    if "dispersion" in pi:
        dpos = pi["dispersion"]
        dpos = dpos[0] if isinstance(dpos, (tuple, list)) else int(dpos)
        for j in range(len(names_full), dpos):
            names_full.append(f"param{j}")
        names_full.append("alpha")  # displayed scale alpha
    if len(names_full) < K:
        names_full += [f"param{i}" for i in range(len(names_full), K)]
    names_full = names_full[:K]
    ct = pd.DataFrame({"Variable": names_full, "Estimate": est,
                       "Std.Err": se_disp, "z": z_disp, "p": p_disp})

    sd_rows = ct[ct["Variable"].str.startswith("RND sd(")]
    mean_rows = ct[ct["Variable"].str.startswith("RND mean(")]
    # fall back: any block whose name mentions sd/mean outside fixed set
    if len(sd_rows) == 0:
        sd_rows = ct[ct["Variable"].str.contains("sd", case=False)
                     & ~ct["Variable"].isin(["(Intercept)"]
                                            + [_V(f) for f in feats]
                                            + ["alpha"])]
    ok_sd = bool((sd_rows["p"] < ALPHA).all()) if len(sd_rows) else True
    ok_mean = True
    if len(mean_rows):
        ok_mean = bool((mean_rows["Estimate"].abs() < 5).all())
    try:
        disp_p = float(ct.set_index("Variable").loc["alpha", "p"])
    except KeyError:
        disp_p = np.nan
    # guard: if random terms were requested but no RND rows were labelled
    # (e.g. 1.0.151 renamed pi blocks), the verdict must not vacuously pass
    if (rdm or rdm_cor) and len(sd_rows) == 0 and len(mean_rows) == 0:
        return {"label": label, "ll": ll, "bic": bic, "k": K, "n": n,
                "converged": bool(res.success), "coef_table": ct,
                "verdict": "NO-RND-ROWS", "ok_sd": "", "ok_mean": "",
                "disp_p": disp_p}, ct
    try:
        _cti2 = ct.set_index("Variable")
        _dp2 = _cti2.loc["alpha", "p"] if "alpha" in _cti2.index else _cti2.loc["ln(overdispersion)", "p"]
        disp_p = float(_dp2)
    except KeyError:
        disp_p = np.nan
    ok_disp = bool(disp_p < ALPHA) if np.isfinite(disp_p) else False
    verdict = "ACCEPT" if (ok_sd and ok_mean and ok_disp) else "REJECT"
    return {"label": label, "ll": ll, "bic": bic, "k": K, "n": n,
            "converged": bool(res.success), "coef_table": ct,
            "verdict": verdict, "ok_sd": ok_sd, "ok_mean": ok_mean,
            "disp_p": disp_p}, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "near_miss_hb_ecf.csv"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--arms", default="A-unstruct,B-hetmean,C-hetmeanvar,D-gse",
                    help="comma list of arms to run; strict subsets write "
                         "gse_rp_pilot_partial.* sidecars instead of the "
                         "full-run aggregates")
    ap.add_argument("--tags", default="HB,Crash",
                    help="comma list of response tags to run, e.g. Crash")
    ap.add_argument("--suffix", default="",
                    help="explicit output suffix override, e.g. _partial_abd")
    args = ap.parse_args()
    sel = [a.strip() for a in args.arms.split(",") if a.strip()]
    tagsel = [t.strip() for t in args.tags.split(",") if t.strip()]
    _all_arms = ["A-unstruct", "B-hetmean", "C-hetmeanvar", "D-gse"]
    suffix = (args.suffix if args.suffix else
              ("" if all(a in sel for a in _all_arms) else "_partial"))
    _gate()

    import pandas as pd
    from nearmiss_hb_analysis import fit_model
    df = pd.read_csv(args.data, low_memory=False)
    _need = (set(HB_FEATS) | set(CRASH_FEATS)
             | set(HB_HELPERS) | set(CR_HELPERS))
    for c in sorted(_need):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    nm_col = "nm_total" if "nm_total" in df.columns else "HB_Total"
    print(f"Loaded {len(df)} rows; HB col={nm_col}", flush=True)

    rows, verdicts = [], []
    for y_col, tag, feats, rdm, rdmc, helpers in [
        (nm_col, "HB", HB_FEATS, HB_RDM, HB_RDM_COR, HB_HELPERS),
        ("ObsCrashes", "Crash", CRASH_FEATS, CR_RDM, CR_RDM_COR, CR_HELPERS),
    ]:
        if tag not in tagsel:
            continue
        # fixed reference (same estimator as reported best models)
        m0 = fit_model(df, feats, y_col, "log_len", label=f"{tag}-fixed")
        # Arms A/B/C: unstructured | +hetro_in_means | +means+variances
        for arm, h, var_on, gse in [
            ("A-unstruct", [], False, None),
            ("B-hetmean", helpers, False, None),
            ("C-hetmeanvar", helpers, True, None),
        ]:
            if arm not in sel:
                continue
            try:
                m, _ = _fit_structured(df, feats, y_col, f"{tag}-{arm}",
                                       rdm, rdmc, h, var_on, gse)
                pb, ni = _unified_pbic(m)
                rows.append({"model": f"{tag}-{arm}", "ll": m["ll"],
                             "bic": m["bic"], "pbic": pb, "n_insig": ni,
                             "k": m["k"], "n": m["n"],
                             "converged": m["converged"]})
                verdicts.append({"model": f"{tag}-{arm}",
                                 "verdict": m["verdict"],
                                 "ok_sd": m["ok_sd"], "ok_mean": m["ok_mean"],
                                 "disp_p": m["disp_p"]})
                print(f"[{tag}-{arm}] BIC={m['bic']:.1f} pBIC={pb:.1f} "
                      f"-> {m['verdict']}", flush=True)
                m["coef_table"].to_csv(os.path.join(
                    args.out_dir, f"gse_rp_{tag}_{arm}.csv"), index=False)
            except Exception as e:  # noqa: BLE001 -- pilot reports, not crashes
                print(f"[{tag}-{arm}] FAILED: {e}", flush=True)
                verdicts.append({"model": f"{tag}-{arm}", "verdict": f"ERROR: {e}",
                                 "ok_sd": "", "ok_mean": "", "disp_p": ""})
        # Arm D: GSE scores (best-effort; SKIP on any API failure)
        if "D-gse" in sel:
            try:
                mf = fit_model(df, feats, y_col, "log_len", label=f"{tag}-fx-gse")
                mu = np.exp(np.clip(
                    mf["params"][0] + mf["scaler"].transform(
                        df.dropna(subset=feats + [y_col, "log_len"])[feats].values.astype(float)
                    ) @ mf["params"][1:len(feats) + 1]
                    + df.dropna(subset=feats + [y_col, "log_len"])["log_len"].values, -20, 20))
                yy = df.dropna(subset=feats + [y_col, "log_len"])[y_col].values.astype(float)
                G = _nb2_grad_scores(yy, mu, mf["alpha"],
                                     mf["scaler"].transform(
                                         df.dropna(subset=feats + [y_col, "log_len"])[feats].values.astype(float)))
                # G must be (N, K_random) in cor-then-ind order (MCR build_eta
                # validates G.shape[-1] == Kr_cor + Kr_ind). Slice the full
                # J-feat gradient scores down to the random-coefficient order.
                rand_order = [t.split(":")[0] for t in rdmc] + [
                    t.split(":")[0] for t in rdm]
                ridx = [feats.index(b) for b in rand_order]
                G = np.asarray(G)[:, ridx]
                gcols = [f"__f{i}__" for i in ridx]
                # No hetro helpers in D-gse to avoid role-exclusivity conflicts
                m, _ = _fit_structured(df, feats, y_col, f"{tag}-D-gse",
                                       rdm, rdmc, [], False, (gcols, G))
                pb, ni = _unified_pbic(m)
                rows.append({"model": f"{tag}-D-gse", "ll": m["ll"], "bic": m["bic"],
                             "pbic": pb, "n_insig": ni,
                             "k": m["k"], "n": m["n"], "converged": m["converged"]})
                verdicts.append({"model": f"{tag}-D-gse", "verdict": m["verdict"],
                                 "ok_sd": m["ok_sd"], "ok_mean": m["ok_mean"],
                                 "disp_p": m["disp_p"]})
                print(f"[{tag}-D-gse] BIC={m['bic']:.1f} pBIC={pb:.1f} "
                      f"-> {m['verdict']}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}-D-gse] SKIP: {e}", flush=True)
                verdicts.append({"model": f"{tag}-D-gse", "verdict": "SKIP",
                                 "ok_sd": "", "ok_mean": "", "disp_p": ""})
        rows.append({"model": f"{tag}-fixed", "ll": m0["ll"], "bic": m0["bic"],
                     "pbic": float(m0["bic"]), "n_insig": 0,
                     "k": m0["k"], "n": m0["n"], "converged": m0["converged"]})

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"gse_rp_pilot{suffix}.csv"),
                              index=False)
    pd.DataFrame(verdicts).to_csv(os.path.join(args.out_dir,
                                               f"gse_rp_pilot_verdict{suffix}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"gse_rp_pilot{suffix}.tex"), "w") as f:
        f.write("% Auto-generated by gse_rp_pilot.py (structured RP pilot)\n")
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Structured random-parameter pilot (metacountregressor "
                "structural equation; $N=10{,}780$, $R=500$ Halton draws; "
                "verdict: every retained SD $p<0.10$, $|$mean$|<5$, "
                "dispersion $p<0.10$)}\\label{tab:gse_rp_pilot}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrrc}\n")
        f.write("      \\toprule\n      Model & LL & BIC & pBIC & $k$ & $\\Delta$pBIC "
                "vs.\\ fixed & Verdict \\\\\n      \\midrule\n")
        res = pd.DataFrame(rows)
        for tag in ["HB", "Crash"]:
            base = float(res[res["model"] == f"{tag}-fixed"]["pbic"].iloc[0])
            for _, r in res[res["model"].str.startswith(tag + "-")
                            & (res["model"] != f"{tag}-fixed")].iterrows():
                v = verdicts and [x for x in verdicts if x["model"] == r["model"]][0]["verdict"]
                f.write(f"      {r['model']} & {r['ll']:,.1f} & {r['bic']:,.1f} "
                        f"& {r['pbic']:,.1f} "
                        f"& {int(r['k'])} & {r['pbic'] - base:+,.1f} & {v} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write("\\end{table}\n")
    print(f"Wrote gse_rp_pilot{suffix}.csv / gse_rp_pilot_verdict{suffix}.csv / "
          f"gse_rp_pilot{suffix}.tex", flush=True)


if __name__ == "__main__":
    main()
