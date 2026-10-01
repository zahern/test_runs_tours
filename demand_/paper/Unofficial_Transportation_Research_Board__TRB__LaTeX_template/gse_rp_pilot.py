"""GSE structured random-parameter pilot (metacountregressor>=1.0.151).

Unrestricted random-parameter blocks keep failing the significance verdict
(degenerate means, n.s. SDs; see eb_searched_verdict.csv: REJECT/REJECT).
1.0.151 exposes the canonical structural equation
(beta_nk = T(beta_k + delta'z_n + gamma*g_nk + exp(lnsig_k + omega'w_n)*nu))
via manual_spec keys hetro_in_means / hetro_in_variances / gse_cols+gse_scores
(see random_parameter_structure.py in the 1.0.151 wheel).

Arms per response (searched specs; HB 10 vars / crash 8 vars; same MCR roles
as run_rp_check.py; R=200 Halton draws; log_len offset):
  A  unstructured RP ............... baseline replicate (expect REJECT)
  B  + hetro_in_means [AADT, SL] .... exposure-like mean helpers
  C  + hetro_in_means + hetro_in_variances [AADT, SL]
  D  + GSE gradient scores ......... best-effort; SKIP with reason on any
                                     API failure (never fails the job)

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
MCR_MIN_VERSION = (1, 0, 165)
ALPHA = 0.10

HB_FEATS = ["AADT", "IC", "IP", "IV_M", "PA_COM1",
            "PSDSG24", "RSDSDL5", "RSDSO", "SKD_SP", "SL"]
CRASH_FEATS = ["AADT", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "RSDSDG10", "RSDSDL5", "SL"]
HB_RDM = ["IC:normal", "IV_M:normal"]
HB_RDM_COR = ["PA_COM1:normal", "RSDSO:normal", "RSDSDL5:normal"]
CR_RDM = ["PSDS1T24:normal", "RSDSDG10:normal"]
CR_RDM_COR = ["SL:normal", "RSDSDL5:normal"]
HELPERS = ["AADT", "SL"]  # exposure-like helpers for means and variances


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
    print(f"metacountregressor {ver} (required >= 1.0.165)", flush=True)
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
    if tup < MCR_MIN_VERSION:
        print(f"FATAL: metacountregressor {ver} < 1.0.165 -- refusing to run",
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
    try:
        disp_p = float(ct.set_index("Variable").loc["ln(overdispersion)", "p"])
    except KeyError:
        disp_p = float("nan")
    n_disp = 0 if (pd.notna(disp_p) and disp_p <= ALPHA) else 1
    n_insig = n_fix + n_sd + n_chol + n_degen + n_disp
    return float(m["bic"]) + PENALTY * n_insig, n_insig


def _fit_structured(df, feats, y_col, label, rdm, rdm_cor, helpers,
                    variances_on, gse_pack, draws_R=200):
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
    help_s = [col2f[h] for h in helpers if h in col2f]

    y_raw = df_m[y_col].values.astype(float)
    df_m["_id"] = np.arange(len(df_m))
    df_m["_off"] = df_m["log_len"].values.astype(float)

    manual_spec = {
        "fixed_terms": fstd, "rdm_terms": rdm_s, "rdm_cor_terms": rdmc_s,
        "grouped_terms": [],
        "hetro_in_means": help_s,
        "hetro_in_variances": help_s if variances_on else [],
        "transformations": ["no"] * len(fstd),
        "dispersion": 1, "latent_classes": 1,
    }
    if gse_pack is not None:
        gcols, gscores = gse_pack
        manual_spec["gse_cols"] = gcols
        manual_spec["gse_scores"] = np.asarray(gscores, dtype=float)

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
    for key in ("ind_mean", "ind_sd", "cor_mean", "chol", "group_mean",
                "group_sd", "het_mean", "het_var", "het_gamma", "gamma_gse",
                "gse_gamma"):
        if key in pi:
            s, e = pi[key]
            init[s:e] = 0.0

    def obj(p):
        return float(mixed_model_loglik(jnp.array(p), data, spec, indivi=False))

    res = scipy_min(obj, init, method="L-BFGS-B", options={"maxiter": 5000})
    params = res.x
    n = len(df_m)
    ll = -obj(params)
    bic = K * np.log(n) - 2 * ll

    try:
        se_jax = np.asarray(compute_standard_errors(jnp.array(params), obj))
        se = se_jax if (np.isfinite(se_jax).any()
                        and np.nanmax(np.abs(se_jax)) < 50) else None
    except Exception:
        se = None
    if se is None:
        se = _scipy_hessian_se(obj, params)
    se = np.where(np.abs(se) > 50, np.nan, se)
    z = np.where(np.isfinite(se) & (se > 1e-12), params / se, np.nan)
    p = np.where(np.isfinite(z), 2 * (1 - scipy_stats.norm.cdf(np.abs(z))),
                 np.nan)

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
        mcor = len(rdm_cor)
        if mcor and n_ch == mcor * (mcor + 1) // 2:
            cvars = [_V(t.split(":")[0]) for t in rdm_cor]
            for i in range(mcor):
                for j in range(i + 1):
                    chol_names.append(f"chol({cvars[i]}|{cvars[j]})")
    names_full = (["(Intercept)"] + [_V(f) for f in feats]
                  + rdm_mean_names + rdm_sd_names + chol_names)
    if "dispersion" in pi:
        dpos = pi["dispersion"]
        dpos = dpos[0] if isinstance(dpos, (tuple, list)) else int(dpos)
        for j in range(len(names_full), dpos):
            names_full.append(f"param{j}")
        names_full.append("ln(overdispersion)")
    if len(names_full) < K:
        names_full += [f"param{i}" for i in range(len(names_full), K)]
    names_full = names_full[:K]
    ct = pd.DataFrame({"Variable": names_full, "Estimate": params[:K],
                       "Std.Err": se[:K], "z": z[:K], "p": p[:K]})

    sd_rows = ct[ct["Variable"].str.startswith("RND sd(")]
    mean_rows = ct[ct["Variable"].str.startswith("RND mean(")]
    # fall back: any block whose name mentions sd/mean outside fixed set
    if len(sd_rows) == 0:
        sd_rows = ct[ct["Variable"].str.contains("sd", case=False)
                     & ~ct["Variable"].isin(["(Intercept)"]
                                            + [_V(f) for f in feats]
                                            + ["ln(overdispersion)"])]
    ok_sd = bool((sd_rows["p"] < ALPHA).all()) if len(sd_rows) else True
    ok_mean = True
    if len(mean_rows):
        ok_mean = bool((mean_rows["Estimate"].abs() < 5).all())
    try:
        disp_p = float(ct.set_index("Variable").loc["ln(overdispersion)", "p"])
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
        disp_p = float(ct.set_index("Variable").loc["ln(overdispersion)", "p"])
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
    args = ap.parse_args()
    _gate()

    import pandas as pd
    from nearmiss_hb_analysis import fit_model
    df = pd.read_csv(args.data, low_memory=False)
    for c in sorted(set(HB_FEATS) | set(CRASH_FEATS)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    nm_col = "nm_total" if "nm_total" in df.columns else "HB_Total"
    print(f"Loaded {len(df)} rows; HB col={nm_col}", flush=True)

    rows, verdicts = [], []
    for y_col, tag, feats, rdm, rdmc in [
        (nm_col, "HB", HB_FEATS, HB_RDM, HB_RDM_COR),
        ("ObsCrashes", "Crash", CRASH_FEATS, CR_RDM, CR_RDM_COR),
    ]:
        # fixed reference (same estimator as reported best models)
        m0 = fit_model(df, feats, y_col, "log_len", label=f"{tag}-fixed")
        # Arms A/B/C: unstructured | +hetro_in_means | +means+variances
        for arm, helpers, var_on, gse in [
            ("A-unstruct", [], False, None),
            ("B-hetmean", HELPERS, False, None),
            ("C-hetmeanvar", HELPERS, True, None),
        ]:
            try:
                m, _ = _fit_structured(df, feats, y_col, f"{tag}-{arm}",
                                       rdm, rdmc, helpers, var_on, gse)
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
            m, _ = _fit_structured(df, feats, y_col, f"{tag}-D-gse",
                                   rdm, rdmc, HELPERS, True, (gcols, G))
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

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "gse_rp_pilot.csv"),
                              index=False)
    pd.DataFrame(verdicts).to_csv(os.path.join(args.out_dir,
                                               "gse_rp_pilot_verdict.csv"), index=False)
    with open(os.path.join(args.out_dir, "gse_rp_pilot.tex"), "w") as f:
        f.write("% Auto-generated by gse_rp_pilot.py (1.0.151 structured RP)\n")
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Structured random-parameter pilot (1.0.151 "
                "structural equation; $N=10{,}780$, $R=200$ Halton draws; "
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
    print("Wrote gse_rp_pilot.csv / gse_rp_pilot_verdict.csv / gse_rp_pilot.tex",
          flush=True)


if __name__ == "__main__":
    main()
