"""
hb_randomwalk_validation.py
===========================
Random-walk validation of the metaheuristic feature selections.

The DE/SA searches report a single best penalised-BIC specification per
response. This script asks whether those optima are *stable plateaus* or
isolated spikes, by exploring the specification space with a Metropolis
random walk:

  state        x in {0,1}^30  (inclusion mask over the candidate variables)
  energy       E(x) = BIC(x) + lambda * #{insignificant coefficients}
               (the same significance-penalised BIC used by the DE search,
               Eq. penbic in the paper), evaluated by NB2 maximum likelihood
               on a fixed subsample (n = --rows, seed 42)
  proposal     toggle one uniformly-chosen candidate (symmetric)
  acceptance   min{1, exp(-Delta E / T)} with T = --temp BIC units

Because the proposal is symmetric, the walk targets the Boltzmann
distribution over specifications proportional to exp(-E/T); visitation
frequencies estimate inclusion probabilities, and the walk trace shows
whether the searched optimum sits on a broad low-energy plateau.

Outputs (in --out-dir):
  hb_randomwalk_trace.png        energy traces + inclusion probabilities
  hb_randomwalk_stability.csv    per-variable inclusion rates and mean energies
  hb_randomwalk_summary.csv      acceptance rates, modal-set Jaccard, etc.
  hb_randomwalk.tex              paper-ready table block

Usage:
  python hb_randomwalk_validation.py [--steps 600] [--rows 4000] [--temp 5]
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "near_miss_hb_ecf.csv")
RS = 42
PENALTY = 10.0          # per insignificant coefficient (matches DE search)
P_THRESH = 0.10
CANDIDATES = [
    "AADT", "SL", "NL", "SKD_SP", "CU_MOD", "CU_SHA", "CU_VSHA",
    "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
    "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
    "IV_VL", "IV_L", "IV_M",
    "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
    "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10",
]  # IV_100 dropped a priori (near-zero variance), as in the paper


def load_subsample(rows, seed=RS):
    df = pd.read_csv(DATA, low_memory=False)
    df = df.dropna(subset=["ObsCrashes", "segment_length_m"])
    y_cols = {"HB": "HB_Total", "Crash": "ObsCrashes"}
    for c in CANDIDATES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    df["log_len"] = np.log(
        pd.to_numeric(df["segment_length_m"], errors="coerce")
        .fillna(0).clip(lower=0.1) / 1000.0)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(rows, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True), y_cols


def fit_energy(dsub, y, mask, start_params=None):
    """Negative-Binomial profile-likelihood energy for a mask.

    Direct NB2 MLE diverges on these counts (fold means ~120, max ~11k), so
    the energy uses the stable two-step estimator of the EB SPFs
    (trb_empirical_bayes.fit_nb): Poisson IRLS coefficients, Hauer moment
    dispersion alpha = sum((y-mu)^2 - mu) / sum(mu^2), then the NB log-
    likelihood at (beta_hat, alpha_hat). HC0-robust GLM p-values drive the
    significance penalty. The identical estimator is applied to every
    visited mask, so energies are comparable across the walk.
    """
    feats = [c for c, m in zip(CANDIDATES, mask) if m]
    X = sm.add_constant(dsub[feats].values.astype(float), has_constant="add")
    off = dsub["log_len"].values.astype(float)
    yv = dsub[y].values.astype(float)
    try:
        pois = sm.GLM(yv, X, offset=off,
                      family=sm.families.Poisson()).fit(maxiter=100)
        mu = np.asarray(pois.fittedvalues, float)
        alpha = max((np.sum((yv - mu) ** 2) - np.sum(mu))
                    / max(np.sum(mu ** 2), 1e-9), 1e-6)
        fam = sm.families.NegativeBinomial(alpha=alpha)
        nb = sm.GLM(yv, X, offset=off, family=fam).fit(cov_type="HC0")
        llf = float(fam.loglike(yv, np.asarray(nb.fittedvalues, float)))
        if not np.isfinite(llf):
            return np.inf, None, np.nan
        k = X.shape[1] + 1                     # + alpha
        bic = k * np.log(len(yv)) - 2.0 * llf
        pvals = np.asarray(nb.pvalues)[1:]     # skip intercept
        n_insig = int(np.sum(~np.isfinite(pvals) | (pvals > P_THRESH)))
        return bic + PENALTY * n_insig, np.asarray(nb.params), llf
    except Exception:
        return np.inf, None, np.nan


def random_walk(dsub, y, start_feats, steps, temp, seed=RS):
    """Metropolis walk over inclusion masks. Returns trace and visit stats."""
    rng = np.random.RandomState(seed + hash(y) % 1000)
    K = len(CANDIDATES)
    state = np.array([c in start_feats for c in CANDIDATES], dtype=bool)
    e, params, _ = fit_energy(dsub, y, state)
    trace = np.empty(steps)
    visits = np.zeros(K)
    incl_when_visited = np.zeros(K)
    n_accept = 0
    for t in range(steps):
        trace[t] = e
        j = rng.randint(K)
        cand = state.copy()
        cand[j] = not cand[j]
        if cand.sum() < 2:
            visits[j] += 1
            continue
        e_new, params_new, _ = fit_energy(dsub, y, cand, start_params=params)
        dE = e_new - e
        if dE <= 0 or rng.rand() < np.exp(-dE / temp):
            state, e, params = cand, e_new, params_new
            n_accept += 1
        visits[j] += 1
        if state[j]:
            incl_when_visited[j] += 1
    incl_prob = np.divide(incl_when_visited, visits,
                          out=np.zeros(K), where=visits > 0)
    return {
        "trace": trace, "energy": e, "state": state, "visits": visits,
        "incl_prob": incl_prob, "accept_rate": n_accept / steps,
    }


def main():
    ap = argparse.ArgumentParser(description="Random-walk specification validation")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--out-dir", default=DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dsub, y_cols = load_subsample(args.rows)
    print(f"subsample n={len(dsub)}, candidates={len(CANDIDATES)}, "
          f"steps={args.steps}, T={args.temp}")

    starts = {}
    for resp, fname in (("HB", "hb_final_features_nm.csv"),
                        ("Crash", "hb_final_features_crash.csv")):
        f = pd.read_csv(os.path.join(DIR, fname))
        starts[resp] = set(f["feature"].tolist())

    results, rows = {}, []
    for resp in ("HB", "Crash"):
        print(f"\nrandom walk [{resp}] from final spec "
              f"({len(starts[resp])} features) ...")
        r = random_walk(dsub, y_cols[resp], starts[resp], args.steps, args.temp)
        results[resp] = r
        modal = {c for c, p in zip(CANDIDATES, r["incl_prob"]) if p >= 0.5}
        jac = len(modal & starts[resp]) / max(len(modal | starts[resp]), 1)
        print(f"  accept rate={r['accept_rate']:.2f}, "
              f"E_start={r['trace'][0]:.1f}, E_end={r['energy']:.1f}, "
              f"modal-set Jaccard vs selected={jac:.2f}")
        for c, ip, v in zip(CANDIDATES, r["incl_prob"], r["visits"]):
            rows.append({"response": resp, "variable": c,
                         "inclusion_prob": ip, "visits": int(v),
                         "selected": c in starts[resp]})
        # trace plot ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2),
                             gridspec_kw={"height_ratios": [1.4, 1]})
    for col, (resp, r) in zip((0, 1), results.items()):
        ax = axes[0, col]
        ax.plot(r["trace"], lw=0.8, color="#4477AA", alpha=0.85)
        ax.plot(np.minimum.accumulate(r["trace"]), lw=1.4, color="#CC6677")
        ax.set_title(f"{resp} response  (accept {r['accept_rate']:.0%})",
                     fontsize=10)
        ax.set_ylabel("penalised BIC")
        ax.grid(alpha=0.3)
        ax2 = axes[1, col]
        order = np.argsort(-r["incl_prob"])[:15]
        sel = [CANDIDATES[i] in starts[resp] for i in order]
        ax2.bar(range(len(order)), r["incl_prob"][order],
                color=["#CC6677" if s else "#4477AA" for s in sel])
        ax2.set_xticks(range(len(order)),
                       [CANDIDATES[i] for i in order], rotation=90,
                       fontsize=6.5)
        ax2.set_ylabel("P(include)", fontsize=9)
        ax2.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("Random-walk validation of the searched specifications "
                 f"({args.steps} steps, T={args.temp} BIC, n={len(dsub)})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "hb_randomwalk_trace.png"), dpi=170)
    plt.close(fig)
    print("\nfigure written: hb_randomwalk_trace.png")

    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(args.out_dir, "hb_randomwalk_stability.csv"),
              index=False)
    summ = []
    for resp, r in results.items():
        modal = {c for c, p in zip(CANDIDATES, r["incl_prob"]) if p >= 0.5}
        jac = len(modal & starts[resp]) / max(len(modal | starts[resp]), 1)
        sel_in = {c for c in starts[resp]
                  if r["incl_prob"][CANDIDATES.index(c)] >= 0.5}
        summ.append({
            "response": resp, "steps": args.steps, "temp_bic": args.temp,
            "accept_rate": round(r["accept_rate"], 3),
            "E_start": round(float(r["trace"][0]), 1),
            "E_end": round(float(r["energy"]), 1),
            "modal_jaccard_vs_selected": round(jac, 3),
            "selected_retained_at_P>=0.5":
                f"{len(sel_in)}/{len(starts[resp])}",
        })
    sm_df = pd.DataFrame(summ)
    sm_df.to_csv(os.path.join(args.out_dir, "hb_randomwalk_summary.csv"),
                 index=False)
    print(sm_df.to_string(index=False))

    # ---- LaTeX table block (no heading; paper supplies context) --------
    lines = [
        "% Auto-generated by hb_randomwalk_validation.py - table block only",
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Random-walk validation of the searched specifications "
        f"({args.steps} Metropolis steps per response, temperature "
        f"$T={args.temp:g}$ BIC units, symmetric single-variable toggle "
        f"proposal, $n={len(dsub)}$ subsample).}}\\label{{tab:randomwalk}}",
        "\\begin{tabular}{lrrrr}\\hline",
        "Response & Accept.\\ rate & $E$ start & $E$ end & Modal-set "
        "Jaccard\\\\\\hline",
    ]
    for _, r in sm_df.iterrows():
        lines.append(f"{r['response']} & {r['accept_rate']:.2f} & "
                     f"{r['E_start']:.0f} & {r['E_end']:.0f} & "
                     f"{r['modal_jaccard_vs_selected']:.2f}\\\\")
    lines += ["\\hline\\end{tabular}",
              "\\emph{Walks start from the final searched specifications. "
              "Inclusion probabilities and the modal set (P(include)$\\geq$0.5) "
              "are estimated from visitation frequencies under the Boltzmann "
              "distribution over specifications; see "
              "Figure~\\ref{fig:randomwalk}.}\\end{table}"]
    with open(os.path.join(args.out_dir, "hb_randomwalk.tex"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("latex block written: hb_randomwalk.tex")


if __name__ == "__main__":
    main()
