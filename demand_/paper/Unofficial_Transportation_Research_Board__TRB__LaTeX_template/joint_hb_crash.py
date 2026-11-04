"""Joint HB+crash model: best margins -> joint dependence -> linkage.

Pipeline (all on VMT-years exposure offset, see exposure.py):
  1. BEST margins: read the HB / crash specs from best_models_master.csv
     (DE-benchmark search winners; falls back to the hardcoded DE sets).
     Fit both NB margins with metacountregressor.fit_model (>=1.0.151).
  2. JOINT estimation: IFM copulas (Frank, Gaussian, Kimeldorf--Sampson,
     Joe + Gaussian-Joe mixture) over both margins; LR vs independence.
  3. LINKAGE: copula dependence (non-linear by construction) PLUS a
     copula-free rank linkage (Spearman rho of Pearson residuals), so the
     HB<->crash link is shown two independent ways.

Outputs (out-dir): joint_hb_crash.csv/.tex (tab:joint_hb_crash),
joint_linkage.csv/.tex (tab:joint_linkage).

Usage (HPC): python -u joint_hb_crash.py [--rp] [--resume]
"""
import os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _metacount_source import METACOUNT_SOURCE
from nearmiss_hb_analysis import fit_model, _rebuild_mu
from copula_rp_tiers import fit_ifm, fit_mixture, _margins, NAMES
from exposure import OFF, add_exposure_offsets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")
SPEC_CSV = os.path.join(HERE, "best_models_master.csv")

MCR_MIN_VERSION = (1, 0, 151)

# Fallback DE-benchmark sets (used only if the spec CSV is unreadable).
HB_FALLBACK = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
               "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5",
               "RSDSO", "RSDSDL5"]
CR_FALLBACK = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]


def check_mcr_version():
    import metacountregressor as mcr
    if METACOUNT_SOURCE:
        ver = getattr(mcr, "__version__", "0.0.0")
    else:
        try:
            from importlib.metadata import version as _pkg_version
            ver = _pkg_version("metacountregressor")
        except Exception:
            ver = getattr(mcr, "__version__", "0.0.0")
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
        print(f"metacountregressor {ver} from {mcr.__file__} (required >= 1.0.151)",
            flush=True)
    if tup < MCR_MIN_VERSION:
        print(f"FATAL: metacountregressor {ver} < 1.0.151 -- refusing to run",
              flush=True)
        sys.exit(2)
    return ver


def load_best_specs(path):
    """Read HB / crash feature sets from best_models_master.csv."""
    try:
        spec = pd.read_csv(path)
        hb = spec.loc[spec["response"] == "HB", "features"].iloc[0].split(";")
        cr = spec.loc[spec["response"] == "CRASH", "features"].iloc[0].split(";")
        print(f"Best specs from {os.path.basename(path)}: "
              f"HB({len(hb)}) crash({len(cr)})", flush=True)
        return [h.strip() for h in hb], [c.strip() for c in cr]
    except Exception as e:
        print(f"WARNING: spec CSV unreadable ({e}); using DE fallback.",
              flush=True)
        return list(HB_FALLBACK), list(CR_FALLBACK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--spec-csv", default=SPEC_CSV)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--rp", action="store_true",
                    help="Also fit RP margins and report RP copula rows")
    ap.add_argument("--joint-mle", action="store_true",
                    help="Also run package-native joint MLE "
                         "(BivariateCopulaNB: both margins + copula together)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    OUT = a.out_dir
    check_mcr_version()

    hb_feats, cr_feats = load_best_specs(a.spec_csv)

    df = pd.read_csv(a.data, low_memory=False)
    df = add_exposure_offsets(df)
    for c in sorted(set(hb_feats) | set(cr_feats)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows; offset={OFF}", flush=True)
    print(f"HB({len(hb_feats)}): {sorted(hb_feats)}", flush=True)
    print(f"CR({len(cr_feats)}): {sorted(cr_feats)}", flush=True)

    # ---- 1. BEST margins (fixed) ----
    print("Fitting best fixed margins ...", flush=True)
    m_hb = fit_model(df, hb_feats, "HB_Total", OFF, label="J-HB-best")
    m_cr = fit_model(df, cr_feats, "ObsCrashes", OFF, label="J-CR-best")
    print(f"HB: LL={m_hb['ll']:.1f} BIC={m_hb['bic']:.1f} k={m_hb['k']}; "
          f"CR: LL={m_cr['ll']:.1f} BIC={m_cr['bic']:.1f} k={m_cr['k']}",
          flush=True)

    margin_sets = [("fixed", m_hb, m_cr)]
    if a.rp:
        from run_rp_check import HB_RDM, HB_RDM_COR, CR_RDM, CR_RDM_COR
        import pickle
        ckpt = os.path.join(OUT, "margins_joint_ckpt.pkl")
        m_hb_r = m_cr_r = None
        if a.resume and os.path.isfile(ckpt):
            try:
                with open(ckpt, "rb") as pf:
                    saved = pickle.load(pf)
                if saved.get("draws") == a.draws:
                    m_hb_r, m_cr_r = saved["hb"], saved["cr"]
                    print(f"Resumed RP margins from {ckpt}.", flush=True)
            except Exception as e:
                print(f"Checkpoint load FAILED ({e}); refitting.", flush=True)
        if m_hb_r is None:
            for draws in ([a.draws] if a.draws <= 100 else [a.draws, 100]):
                try:
                    print(f"Fitting RP margins (draws_R={draws}) ...", flush=True)
                    m_hb_r = fit_model(df, hb_feats, "HB_Total", OFF,
                                       label="J-HBr", rdm_terms=HB_RDM,
                                       rdm_cor_terms=HB_RDM_COR, draws_R=draws)
                    m_cr_r = fit_model(df, cr_feats, "ObsCrashes", OFF,
                                       label="J-CRr", rdm_terms=CR_RDM,
                                       rdm_cor_terms=CR_RDM_COR, draws_R=draws)
                    try:
                        with open(ckpt, "wb") as pf:
                            pickle.dump({"hb": m_hb_r, "cr": m_cr_r,
                                         "draws": draws}, pf)
                    except Exception as e:
                        print(f"Checkpoint save FAILED ({e}).", flush=True)
                    break
                except Exception as e:
                    print(f"RP draws_R={draws} FAILED: {e}", flush=True)
                    m_hb_r = m_cr_r = None
        if m_hb_r is not None:
            margin_sets.append(("RP", m_hb_r, m_cr_r))
        else:
            print("WARNING: RP margins unavailable; fixed only.", flush=True)

    # ---- 2+3. JOINT copulas + LINKAGE per margin set ----
    for tag, mh_m, mc_m in margin_sets:
        M = _margins(df, mh_m, mc_m, "HB_Total")
        rows = []
        for key in ["frank", "normal", "kimeldorf", "joe"]:
            try:
                r = fit_ifm(M, key)
            except Exception as e:
                print(f"[{tag}] {NAMES[key]} FAILED: {e}", flush=True)
                continue
            r["model"] = NAMES[key]
            rows.append(r)
            print(f"[{tag}] {NAMES[key]}: th={r['theta']:.3f} "
                  f"dBIC={r['dBIC']:+.1f} LR={r['lr']:.1f} p={r['lr_p']:.4g}",
                  flush=True)
        try:
            rm = fit_mixture(M)
            rm["model"] = "Gaussian-Joe mixture"
            rows.append(rm)
            print(f"[{tag}] Mixture: dBIC={rm['dBIC']:+.1f} LR={rm['lr']:.1f}",
                  flush=True)
        except Exception as e:
            print(f"[{tag}] Mixture FAILED: {e}", flush=True)

        # Copula-free rank linkage: Spearman of Pearson residuals.
        df_c = df.dropna(
            subset=list(set(mh_m["features"]) | set(mc_m["features"]))
                   + ["HB_Total", "ObsCrashes", OFF]).reset_index(drop=True)
        mu1, al1 = _rebuild_mu(df_c, mh_m)
        mu2, al2 = _rebuild_mu(df_c, mc_m)
        y1 = df_c["HB_Total"].values.astype(float)
        y2 = df_c["ObsCrashes"].values.astype(float)
        pr1 = (y1 - mu1) / np.sqrt(mu1 + al1 * mu1 ** 2)
        pr2 = (y2 - mu2) / np.sqrt(mu2 + al2 * mu2 ** 2)
        ok = np.isfinite(pr1) & np.isfinite(pr2)
        rho, p_rho = scipy_stats.spearmanr(pr1[ok], pr2[ok])
        print(f"[{tag}] residual Spearman rho={rho:.3f} p={p_rho:.4g} "
              f"(n={int(ok.sum())})", flush=True)

        suf = "" if tag == "fixed" else "_rp"
        if rows:
            jd = pd.DataFrame(rows)
            jd.to_csv(os.path.join(OUT, f"joint_hb_crash{suf}.csv"), index=False)
            with open(os.path.join(OUT, f"joint_hb_crash{suf}.tex"), "w") as f:
                f.write("% Auto-generated by joint_hb_crash.py "
                        f"({tag} best-spec margins, exposure offset)\n")
                f.write("\\begin{table}[!ht]\n")
                f.write(f"  \\caption{{Joint HB--crash dependence on best-spec margins ({tag}, $N={M['n']:,}$)}}\\label{{tab:joint_hb_crash{suf}}}\n")
                f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n      \\toprule\n")
                f.write("      Copula & $\\hat\\theta$ & $\\log L$ & $\\Delta$BIC vs.\\ indep. & LR & $p$ \\\\\n      \\midrule\n")
                for _, r_ in jd.iterrows():
                    th = r_['theta'] if isinstance(r_['theta'], str) else f"{r_['theta']:.3f}"
                    f.write(f"      {r_['model']} & {th} & {r_['ll']:,.1f} & {r_['dBIC']:+,.1f} & {r_['lr']:,.1f} & {r_['lr_p']:.4g} \\\\\n")
                f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
                f.write(f"  \\emph{{Note:}} Margins are the best-spec NB fits (HB {len(hb_feats)}, crash {len(cr_feats)} vars; VMT-years offset). Floored evaluations: "
                        + ", ".join(f"{mname}={ncl}" for mname, ncl in
                                    ((r_['model'] if isinstance(r_['model'], str) else r_['model'],
                                      int(r_['nclip']) if 'nclip' in r_ and pd.notna(r_['nclip']) else 0)
                                     for _, r_ in jd.iterrows())) + ".\n")
                f.write("\\end{table}\n")

        best = min(rows, key=lambda r: r["bic"]) if rows else None
        link = pd.DataFrame([{
            "margins": tag, "n": M["n"],
            "best_copula": best["model"] if best is not None else "none",
            "theta": best["theta"] if best is not None and not isinstance(best["theta"], str) else (best["theta"] if best is not None else ""),
            "dBIC": best["dBIC"] if best is not None else np.nan,
            "lr": best["lr"] if best is not None else np.nan,
            "lr_p": best["lr_p"] if best is not None else np.nan,
            "resid_spearman": rho, "resid_p": p_rho,
        }])
        link.to_csv(os.path.join(OUT, f"joint_linkage{suf}.csv"), index=False)
        with open(os.path.join(OUT, f"joint_linkage{suf}.tex"), "w") as f:
            f.write("% Auto-generated by joint_hb_crash.py (linkage verdict)\n")
            f.write("\\begin{table}[!ht]\n")
            f.write(f"  \\caption{{HB--crash linkage verdict ({tag} margins)}}\\label{{tab:joint_linkage{suf}}}\n")
            f.write("  \\begin{center}\n    \\begin{tabular}{lr}\n      \\toprule\n")
            f.write("      Quantity & Value \\\\\n      \\midrule\n")
            r0 = link.iloc[0]
            f.write(f"      Best copula & {r0['best_copula']} \\\\\n")
            f.write(f"      $\\Delta$BIC vs.\\ indep. & {r0['dBIC']:+.1f} \\\\\n")
            f.write(f"      LR $p$ & {r0['lr_p']:.4g} \\\\\n")
            f.write(f"      Residual Spearman $\\rho$ & {r0['resid_spearman']:.3f} ($p={r0['resid_p']:.4g}$) \\\\\n")
            f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
            verdict = ("dependence + shared rank signal" if (r0['lr_p'] < 0.05 and r0['resid_p'] < 0.05)
                       else "mixed/weak linkage (see copula and residual rows)")
            f.write(f"  \\emph{{Note:}} $N={int(r0['n']):,}$ segments on the VMT-years scale. Reading: {verdict}.\n")
            f.write("\\end{table}\n")
        print(f"Wrote joint_hb_crash{suf} + joint_linkage{suf} (.tex/.csv).",
              flush=True)

    # ---- 4. Package-native JOINT MLE: both NB margins + copula estimated
    # together (BivariateCopulaNB), same best specs + exposure offsets. ----
    if a.joint_mle:
        try:
            from metacountregressor.bivariate_copula import BivariateCopulaNB
        except Exception as e:
            print(f"JOINT-MLE unavailable ({e}); skipping.", flush=True)
            BivariateCopulaNB = None
        if BivariateCopulaNB is not None:
            df_j = df.dropna(subset=list(set(hb_feats) | set(cr_feats))
                             + ["HB_Total", "ObsCrashes", OFF]).reset_index(drop=True)
            y1 = df_j["HB_Total"].values.astype(float)
            y2 = df_j["ObsCrashes"].values.astype(float)
            o1 = df_j[OFF].values.astype(float)
            o2 = df_j[OFF].values.astype(float)
            # Standardize like fit_model: raw scales (AADT in thousands vs
            # 0/1 dummies) make the package L-BFGS-B overflow (job 25363153:
            # all-nan, conv=False on raw X).
            from sklearn.preprocessing import StandardScaler
            x1 = StandardScaler().fit_transform(
                df_j[hb_feats].values.astype(float))
            x2 = StandardScaler().fit_transform(
                df_j[cr_feats].values.astype(float))
            print("[mle] standardized X (job 25363153 fix).", flush=True)
            M0 = _margins(df, m_hb, m_cr, "HB_Total")
            mrows = []
            for cop in ["frank", "normal", "kimeldorf"]:
                try:
                    fit = BivariateCopulaNB(copula=cop).fit(
                        y1, y2, x1, x2, o1, o2,
                        feature_names_1=hb_feats, feature_names_2=cr_feats)
                except Exception as e:
                    print(f"[mle] {cop} FAILED: {e}", flush=True)
                    continue
                if fit.n == M0["n"]:
                    bic0 = M0["k_indep"] * np.log(fit.n) - 2 * M0["ll_indep"]
                    dBIC = fit.bic - bic0
                else:
                    dBIC = float("nan")
                ok = bool(fit.converged) and np.isfinite(fit.loglik)
                mrows.append({"copula": cop, "theta": fit.rho,
                              "theta_p": fit.p_rho, "ll": fit.loglik,
                              "bic": fit.bic, "dBIC": dBIC,
                              "converged": fit.converged, "n": fit.n,
                              "k": fit.k, "ok": ok})
                print(f"[mle] {cop}: th={fit.rho:.3f} p={fit.p_rho:.4g} "
                      f"BIC={fit.bic:,.1f} dBIC={dBIC:+.1f} "
                      f"conv={fit.converged}", flush=True)
            if mrows:
                mj = pd.DataFrame(mrows)
                mj_ok = mj[mj["ok"]].drop(columns=["ok"])
                if len(mj_ok) == 0:
                    print("WARNING: no joint-MLE run converged -- "
                          "joint_mle NOT written (not papering NaNs).",
                          flush=True)
                    mj.to_csv(os.path.join(OUT, "joint_mle_all.csv"),
                              index=False)
                else:
                    mj_ok.to_csv(os.path.join(OUT, "joint_mle.csv"), index=False)
                    mj.to_csv(os.path.join(OUT, "joint_mle_all.csv"),
                              index=False)
                    with open(os.path.join(OUT, "joint_mle.tex"), "w") as f:
                        f.write("% Auto-generated by joint_hb_crash.py --joint-mle "
                                "(BivariateCopulaNB, best specs, exposure offset)\n")
                        f.write("\\begin{table}[!ht]\n")
                        f.write(f"  \\caption{{Joint MLE of HB--crash margins and copula (best specs, $N={len(df_j):,}$)}}\\label{{tab:joint_mle}}\n")
                        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrc}\n      \\toprule\n")
                        f.write("      Copula & $\\hat\\theta$ & $\\log L$ & BIC & $\\Delta$BIC vs.\\ indep. & Conv. \\\\\n      \\midrule\n")
                        for _, r_ in mj_ok.iterrows():
                            f.write(f"      {r_['copula']} & {r_['theta']:.3f} ($p={r_['theta_p']:.4g}$) & {r_['ll']:,.1f} & {r_['bic']:,.1f} & {r_['dBIC']:+.1f} & {r_['converged']} \\\\\n")
                        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
                        f.write("  \\emph{Note:} Both NB margins and the copula dependence estimated together (package-native joint MLE) on the VMT-years scale.\n")
                        f.write("\\end{table}\n")
                    print(f"Wrote joint_mle (.tex/.csv): {len(mj_ok)} rows.",
                          flush=True)
            else:
                print("WARNING: no joint-MLE rows -- joint_mle NOT written.",
                      flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
