"""
45_robust_stochastic.py - OR layer for police schedule + placement.

Inspired by Burdett & Corry (Ann. Oper. Res. 2026): robust configuration
across demand scenarios, best-fit max-coverage, multilevel planning, and
matheuristics (master heuristic + exact MIP subproblem). Transferred from
hospital caseloads to fixture demand:

  P7 robust staffing: ONE staffed set S (|S|=P) maximising the number of
     fixtures whose static coverage clears target tau (best-fit max count
     minus lambda per post, cf. their Eq.12 + lambda penalty). Oracle =
     per-fixture best set (upper bound).
  P8 two-stage stochastic roster (SAA over 41 fixtures): first stage opens
     posts z (fixed cost phi each); second stage staffs x[g] within base
     squad P0 + paid overtime o[g] (unit cost gamma). Reports stochastic
     optimum vs deterministic (mean-scenario) value => VSS, and EVPI vs
     per-fixture oracle.
  VNS matheuristic: master perturbs the staffed set (N1 single-swap,
     N2 double-swap shake); subproblem = EXACT P5 movement MIP (HiGHS) on
     the restricted set. Benchmarked against unrestricted exact P5.

Inputs: results/surfaces.npz (acc_fx 41x14x28, crowds, types, labels,
        prank, node_id, plat, plon)
Outputs: results/robust_stochastic.tex, results/robust_stochastic_audit.json,
         figures/robust_tau.png

Usage: python 45_robust_stochastic.py [--squad 5]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import seaborn as sns

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
SURF = os.path.join(RES, "surfaces.npz")
OUT_TEX = os.path.join(RES, "robust_stochastic.tex")
OUT_JSON = os.path.join(RES, "robust_stochastic_audit.json")
FIG_TAU = os.path.join(FIG, "robust_tau.png")

sns.set_style("whitegrid")


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def tau_matrix(plat, plon, delta_min=15.0, v_kmh=12.0, setup_min=5.0, max_move_m=4000.0):
    J = len(plat)
    D = haversine_m(plat[:, None], plon[:, None], plat[None, :], plon[None, :])
    tau = np.full((J, J), 10 ** 6, dtype=int)
    for i in range(J):
        for j in range(J):
            if i == j:
                tau[i, j] = 1
            elif D[i, j] <= max_move_m:
                t = int(np.ceil((setup_min + (D[i, j] / 1000.0) / v_kmh * 60.0) / delta_min))
                tau[i, j] = max(2, t)
    return tau


def solve_p5(r, P, tau, lam=0.02, time_limit=120.0):
    """Exact relocation-aware deployment MIP (Programme P5, compact)."""
    J, T = r.shape
    arcs = [(i, j) for i in range(J) for j in range(J) if tau[i, j] <= T]
    NX, NY = J * T, len(arcs) * T
    N = NX + NY
    c = np.zeros(N)
    for j in range(J):
        for t in range(T):
            c[j * T + t] = -r[j, t]
    for a, (i, j) in enumerate(arcs):
        if i != j:
            for t in range(T):
                c[NX + a * T + t] = lam
    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0

    def row(entries, lo, hi):
        nonlocal nr
        for var, val in entries:
            rows.append(nr)
            cols.append(var)
            vals.append(val)
        lb.append(lo)
        ub.append(hi)
        nr += 1

    def xv(j, t):
        return j * T + t

    def yv(a, t):
        return NX + a * T + t

    in_arcs = {}
    out_arcs = {}
    for a, (i, j) in enumerate(arcs):
        out_arcs.setdefault(i, []).append((a, j))
        in_arcs.setdefault(j, []).append((a, i))
    row([(xv(j, 0), 1.0) for j in range(J)], P, P)
    for j in range(J):
        for t in range(1, T):
            ent = [(xv(j, t), -1.0)]
            for a, i in in_arcs.get(j, []):
                if 0 <= t - tau[i, j] < T:
                    ent.append((yv(a, t - tau[i, j]), 1.0))
            row(ent, 0.0, 0.0)
    for j in range(J):
        for t in range(T - 1):
            row([(xv(j, t), 1.0)] + [(yv(a, t), -1.0) for a, _ in out_arcs.get(j, [])],
                0.0, 0.0)
    A = csr_matrix((vals, (rows, cols)), shape=(nr, N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return None, None, None, res.status
    z = np.round(res.x).astype(int)
    x = z[:NX].reshape(J, T).astype(bool)
    moves = int(sum(z[NX + a * T + t] for a, (i, j) in enumerate(arcs)
                    for t in range(T) if i != j))
    return x, float(r[x].sum() / r.sum()), moves, res.status


def solve_p7(Amat, tot, P, tau, lam=1e-4, time_limit=120.0):
    """P7 robust staffing: max fixtures clearing tau minus lambda per post."""
    G, J = Amat.shape
    N = J + G
    c = np.concatenate([-lam * np.ones(J), -np.ones(G)])
    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0
    for g in range(G):
        for j in range(J):
            if Amat[g, j] > 0:
                rows.append(nr)
                cols.append(j)
                vals.append(Amat[g, j])
        rows.append(nr)
        cols.append(J + g)
        vals.append(-tau * tot[g])
        lb.append(0.0)
        ub.append(np.inf)
        nr += 1
    rows.append(nr)
    for j in range(J):
        rows.append(nr)
        cols.append(j)
        vals.append(1.0)
    lb.append(P)
    ub.append(P)
    nr += 1
    res = milp(c=c, constraints=LinearConstraint(
        csr_matrix((vals, (rows, cols)), shape=(nr, N)), lb, ub),
        integrality=np.ones(N), bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    s = np.round(res.x[:J]).astype(int)
    cov = np.round(res.x[J:]).astype(int)
    return s, cov, res.status


def solve_p8(Amat, tot, P0, Omax=2, phi=0.002, gamma=0.01, time_limit=300.0,
             mean_only=False):
    """P8 two-stage SAA: open posts z; per-scenario staff x[g] + overtime o[g]."""
    G, J = Amat.shape
    if mean_only:
        Am = Amat.mean(axis=0, keepdims=True)
        tm = tot.mean()
        G, Amat, tot = 1, Am, np.array([tm])
    # order: z[J] binary, x[G*J] binary, o[G] integer 0..Omax
    NZ, NX, NO = J, G * J, G
    N = NZ + NX + NO
    p = 1.0 / G
    c = np.zeros(N)
    c[:J] = phi
    for g in range(G):
        for j in range(J):
            c[NZ + g * J + j] = -p * Amat[g, j] / max(tot[g], 1e-9)
    for g in range(G):
        c[NZ + NX + g] = p * gamma
    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0

    def row(entries, lo, hi):
        nonlocal nr
        for var, val in entries:
            rows.append(nr)
            cols.append(var)
            vals.append(val)
        lb.append(lo)
        ub.append(hi)
        nr += 1

    for g in range(G):
        for j in range(J):
            row([(NZ + g * J + j, 1.0), (j, -1.0)], -np.inf, 0.0)  # x <= z
    for g in range(G):
        row([(NZ + g * J + j, 1.0) for j in range(J)] +
            [(NZ + NX + g, -1.0)], -np.inf, P0)  # squad + overtime
    A = csr_matrix((vals, (rows, cols)), shape=(nr, N))
    integ = np.concatenate([np.ones(NZ + NX), np.ones(NO)])
    bnd = Bounds(np.zeros(N), np.concatenate([np.ones(NZ + NX), Omax * np.ones(NO)]))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=integ, bounds=bnd,
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return None
    z = np.round(res.x[:NZ]).astype(int)
    x = np.round(res.x[NZ:NZ + NX]).reshape(G, J).astype(int)
    o = np.round(res.x[NZ + NX:]).astype(int)
    cov = np.array([Amat[g] @ x[g] / max(tot[g], 1e-9) for g in range(G)])
    return {"z": z, "x": x, "o": o, "cov": cov,
            "exp_cov": float((cov / G).sum()) if not mean_only else float(cov[0]),
            "exp_o": float(o.mean()), "status": int(res.status)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    P = int(args.squad)
    rng = np.random.default_rng(args.seed)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    t0 = time.time()
    audit = {"squad": P}
    print("=" * 80)
    print("45_robust_stochastic.py - P7 robust + P8 stochastic + VNS matheuristic")
    print("=" * 80)

    d = np.load(SURF)
    R = d["acc_fx"].astype(float)  # (41, 14, 28)
    G, J, T = R.shape
    labels = [str(x) for x in d["labels"]]
    crowds = d["crowds"].astype(float)
    Amat = R.sum(axis=2)
    tot = Amat.sum(axis=1)
    print(f"   fixtures={G} posts={J} buckets={T}")

    # ---- P7: robust staffing across fixtures ----
    print("\n[P7] Robust staffing (max fixtures clearing tau)...")
    oracle = np.array([np.sort(Amat[g])[-P:].sum() / tot[g] for g in range(G)])
    p7_rows = []
    for tau in [0.40, 0.45, 0.50, 0.55, 0.60]:
        s, cov, st = solve_p7(Amat, tot, P, tau)
        n_cov = int(cov.sum())
        totcov = float((Amat[:, s == 1].sum(axis=1) / tot).mean())
        p7_rows.append({"tau": tau, "fixtures_covered": n_cov,
                        "of": G, "mean_coverage": round(totcov, 4),
                        "oracle_mean": round(float(oracle.mean()), 4),
                        "posts": ",".join(map(str, np.where(s == 1)[0])),
                        "status": st})
        print(f"   tau={tau:.2f}: {n_cov}/{G} fixtures, mean cov={totcov:.3f} "
              f"(oracle {oracle.mean():.3f}), posts={p7_rows[-1]['posts']}")
    audit["p7"] = p7_rows

    # ---- P8: two-stage stochastic roster ----
    print("\n[P8] Two-stage stochastic roster (SAA, 41 scenarios)...")
    st = solve_p8(Amat, tot, P0=P)
    de = solve_p8(Amat, tot, P0=P, mean_only=True)
    # deterministic policy evaluated on scenarios: staff fixed top-mean set
    mean_top = np.argsort(-Amat.mean(axis=0))[:P]
    det_cov = np.array([Amat[g][mean_top].sum() / tot[g] for g in range(G)])
    vss = st["exp_cov"] - float(det_cov.mean())
    evpi = float(oracle.mean()) - st["exp_cov"]
    print(f"   stochastic E[cov]={st['exp_cov']:.4f} E[overtime]={st['exp_o']:.3f} "
          f"posts={sorted(map(int, np.where(st['z'] == 1)[0]))}")
    print(f"   deterministic E[cov]={det_cov.mean():.4f} => VSS={vss:+.4f}")
    print(f"   oracle E[cov]={oracle.mean():.4f} => EVPI={evpi:+.4f}")
    audit["p8"] = {"exp_cov": st["exp_cov"], "exp_overtime": st["exp_o"],
                   "det_cov": float(det_cov.mean()), "vss": vss,
                   "evpi": evpi, "open_posts": sorted(map(int, np.where(st["z"] == 1)[0])),
                   "status": st["status"]}

    # ---- VNS matheuristic vs exact P5 ----
    print("\n[VNS] Matheuristic (master perturbs set, exact P5 subproblem)...")
    plat, plon = d["plat"].astype(float), d["plon"].astype(float)
    tau = tau_matrix(plat, plon)
    rbar = R.mean(axis=0)
    tE = time.time()
    x_opt, cov_opt, mov_opt, st_opt = solve_p5(rbar, P, tau)
    t_exact = time.time() - tE
    print(f"   exact P5: cov={cov_opt:.4f} moves={mov_opt} time={t_exact:.1f}s")
    cur = set(rng.choice(J, size=P, replace=False).tolist())
    best, best_cov = set(cur), -1.0
    evals, tV = 0, time.time()

    def subproblem(S):
        S = sorted(S)
        xs, cv, mv, stt = solve_p5(rbar[S][:, :], P, tau[np.ix_(S, S)],
                                   time_limit=60.0)
        return cv if cv is not None else -1.0

    cur_cov = subproblem(cur)
    evals += 1
    while evals < 60 and time.time() - tV < 600:
        improved = False
        # N1: single swaps, first improvement
        cand = [(a, b) for a in cur for b in range(J) if b not in cur]
        rng.shuffle(cand)
        for a, b in cand:
            trial = set(cur)
            trial.discard(a)
            trial.add(b)
            v = subproblem(trial)
            evals += 1
            if v > cur_cov + 1e-9:
                cur, cur_cov = trial, v
                improved = True
                break
            if evals >= 60:
                break
        if cur_cov > best_cov:
            best, best_cov = set(cur), cur_cov
        if not improved:
            # N2 shake: random double swap
            trial = set(rng.choice(J, size=P, replace=False).tolist())
            v = subproblem(trial)
            evals += 1
            if v > cur_cov:
                cur, cur_cov = trial, v
    t_vns = time.time() - tV
    gap = (cov_opt - best_cov) / max(cov_opt, 1e-9)
    print(f"   VNS: best={best_cov:.4f} set={sorted(best)} evals={evals} "
          f"time={t_vns:.1f}s gap={100 * gap:.2f}%")
    audit["vns"] = {"exact_cov": cov_opt, "vns_cov": best_cov,
                    "gap_pct": round(100 * gap, 3), "evals": evals,
                    "exact_s": round(t_exact, 1), "vns_s": round(t_vns, 1)}

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    taus = [r["tau"] for r in p7_rows]
    ax.plot(taus, [r["fixtures_covered"] for r in p7_rows], "o-",
            label="robust set", lw=2)
    ax.set_xlabel("Coverage target tau")
    ax.set_ylabel(f"Fixtures covered (of {G})")
    ax.set_title(f"Robust staffing: one set of {P} posts across {G} fixtures")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_TAU, dpi=150)
    plt.close(fig)

    # ---- tex ----
    L = []
    L.append("% robust_stochastic.tex - auto-generated by 45_robust_stochastic.py")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{P7 robust staffing:} one set of "
             f"${P}$ posts; fixtures clearing target $\\tau$. Oracle = per-fixture best set.}}")
    L.append("\\label{tab:robust}")
    L.append("\\begin{tabular}{rrrr}")
    L.append("\\hline")
    L.append("$\\tau$ & Covered & Mean cov. & Oracle \\\\")
    L.append("\\hline")
    for r in p7_rows:
        L.append("{0:.2f} & {1}/{2} & {3:.3f} & {4:.3f} \\\\".format(
            r["tau"], r["fixtures_covered"], r["of"], r["mean_coverage"], r["oracle_mean"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{P8 two-stage stochastic roster} "
             "(SAA, 41 fixture scenarios; overtime price $\\gamma$). VSS = value of the stochastic solution.}}")
    L.append("\\label{tab:stochastic}")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\hline")
    L.append("Policy & E[coverage] & E[overtime] & VSS & EVPI \\\\")
    L.append("\\hline")
    q = audit["p8"]
    L.append("Stochastic & {0:.4f} & {1:.3f} & {2:+.4f} & {3:+.4f} \\\\".format(
        q["exp_cov"], q["exp_overtime"], q["vss"], q["evpi"]))
    L.append("Deterministic (mean) & {0:.4f} & -- & -- & -- \\\\".format(q["det_cov"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{VNS matheuristic vs exact P5} "
             "on the season-mean surface.}")
    L.append("\\label{tab:vns}")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\hline")
    L.append("Method & Coverage & Moves & Evals & Time (s) \\\\")
    L.append("\\hline")
    L.append("Exact P5 (HiGHS) & {0:.4f} & {1} & 1 & {2:.1f} \\\\".format(
        cov_opt, mov_opt, t_exact))
    L.append("VNS + exact subproblem & {0:.4f} & -- & {1} & {2:.1f} \\\\".format(
        best_cov, evals, t_vns))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    audit["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nWrote {OUT_TEX} in {audit['runtime_s']}s")
    print("DONE! Robust + stochastic + VNS complete.")


if __name__ == "__main__":
    main()
