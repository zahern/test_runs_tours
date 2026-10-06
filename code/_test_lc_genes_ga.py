"""Tests for the opt-in latent-class gene block wired into the Stage-5
destination GA (STAGE5_LC_GENES=1).

Covers:
  T1-T4  decode + repair (pooled, crash repair, membership fallback, hyper-genes)
  T5-T7  fit_lc_genome: pooled passthrough, recovery on LC-truth data,
         crash-gap hard constraint
  T8     GA selection: with the extended chromosome the GA picks the
         LC-with-crash genome over the pooled MNL on LC-truth data
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("STAGE5_LC_N_INIT", "1")
os.environ.setdefault("STAGE5_LC_MAXITER", "150")
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import pandas as pd

from stage5_lc_genes import (extend_candidates, lc_gene_names, decode_lc_spec,
                             fit_lc_genome, default_membership_vars)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""))


# ── LC-truth synthetic frame (crash-defined classes + person membership) ──
rng = np.random.default_rng(11)
N, J = 800, 4
B0 = np.array([-1.0, -0.2, 0.5])
B1 = np.array([-0.4, -1.5, -0.2])
GAMMA = np.array([0.6, 0.9, -0.7])
rows = []
for n in range(N):
    student = float(rng.random() < 0.35)
    incp = float(rng.normal())
    p1 = 1.0 / (1.0 + np.exp(-(GAMMA[0] + GAMMA[1] * student + GAMMA[2] * incp)))
    cls = 1 if rng.random() < p1 else 0
    x = rng.normal(size=(J, 3))
    x[:, 0] = np.log1p(rng.uniform(1, 60, J))
    x[:, 1] = np.log1p(rng.uniform(0, 300, J))
    q = rng.uniform(0.02, 1.0, J)
    off = -np.log(q)
    u = x @ (B1 if cls else B0) + off
    pr = np.exp(u - u.max()); pr /= pr.sum()
    ch = int(rng.choice(J, p=pr))
    for a in range(J):
        rows.append({"case_id": f"c{n}", "alt_id": a, "chosen": int(a == ch),
                     "log_correction": float(off[a]),
                     "log_DIST_1": x[a, 0], "log_CRASH_1": x[a, 1],
                     "visit_work": x[a, 2],
                     "student": student, "incp_std": incp})
df = pd.DataFrame(rows)
BASE = ["log_DIST_1", "log_CRASH_1", "visit_work"]
MEMB = ["student", "incp_std"]
CANDS = extend_candidates(BASE, membership_vars=MEMB)
IDX = {c: i for i, c in enumerate(CANDS)}


def genome(*names):
    g = np.zeros(len(CANDS), dtype=int)
    for nm in names:
        g[IDX[nm]] = 1
    return g


print("=== T1-T4: decode + repair ===")
sp = decode_lc_spec(genome("log_DIST_1", "log_CRASH_1"), CANDS, BASE)
check("T1 pooled genome decodes lc_on=0", sp["lc_on"] is False,
      f"features={sp['features']}")

sp = decode_lc_spec(genome("log_DIST_1", "visit_work", "lc_on"), CANDS, BASE)
check("T2 lc_on repairs a crash feature in",
      "log_CRASH_1" in sp["features"] and sp["lc_on"],
      f"features={sp['features']}")

sp = decode_lc_spec(genome("log_DIST_1", "lc_on"), CANDS, BASE,
                    available_membership=MEMB)
check("T3 membership fallback = available pool",
      sp["membership_vars"] == MEMB, f"memb={sp['membership_vars']}")

sp_hi = decode_lc_spec(genome("log_DIST_1", "lc_on", "lc_gap_hi", "lc_nc3",
                              "lc_memb:student"), CANDS, BASE)
check("T4 hyper-genes decode (nc3/gap_hi/memb)",
      sp_hi["n_classes"] == 3 and sp_hi["gap_min"] == 1.0
      and sp_hi["membership_vars"] == ["student"],
      f"nc={sp_hi['n_classes']} gap_min={sp_hi['gap_min']} "
      f"memb={sp_hi['membership_vars']}")

print("\n=== T5-T7: fit_lc_genome ===")
score_p, info_p = fit_lc_genome(df, {c: c for c in df.columns},
                                decode_lc_spec(genome("log_DIST_1", "log_CRASH_1"),
                                               CANDS, BASE))
check("T5 pooled spec returns None (caller uses larch)", score_p is None,
      info_p.get("reason", ""))

spec = decode_lc_spec(genome("log_DIST_1", "log_CRASH_1", "visit_work", "lc_on",
                             "lc_memb:student", "lc_memb:incp_std"),
                      CANDS, BASE, available_membership=MEMB)
t0 = time.time()
score, info = fit_lc_genome(df, {c: c for c in df.columns}, spec, seed=7)
print(f"  LC fit {time.time()-t0:.0f}s score(-BIC/n)={score:.4f} ll={info.get('ll'):.1f} "
      f"gap={info.get('crash_gap'):.2f} shares={[round(s,3) for s in info.get('shares',[])]}")
check("T6 LC genome scores finite on LC-truth",
      score is not None and score > -1e8 and np.isfinite(score),
      f"score={score}")
check("T6 recovers a crash-defined class (gap ~1.3)",
      info.get("crash_gap", 0.0) > 0.8, f"gap={info.get('crash_gap'):.2f}")

spec_hi = dict(spec, gap_min=1.0)
score_hi, info_hi = fit_lc_genome(df, {c: c for c in df.columns}, spec_hi, seed=7)
spec_impossible = dict(spec, gap_min=2.5)
score_imp, info_imp = fit_lc_genome(df, {c: c for c in df.columns},
                                    spec_impossible, seed=7)
check("T7 gap_min=1.0 satisfiable, 2.5 rejected",
      score_hi is not None and score_hi > -1e8 and score_imp <= -1e8
      and "crash gap" in str(info_imp.get("reason", "")),
      f"hi={score_hi:.3f} impossible={info_imp.get('reason')}")

print("\n=== T8: GA selects LC-with-crash on the extended chromosome ===")
from ga_hypothesis_testing import GeneticAlgorithm
from SearchLibrium.mh_choice_estimator import ChoiceSetFrame, _fit_frame_mnl
try:
    from ga_external_estimators import _GA_COMPLEXITY_PENALTY as PEN
except Exception:
    PEN = 1.0


def genome_fitness(genes):
    sp = decode_lc_spec(genes, CANDS, BASE, available_membership=MEMB)
    if not sp["lc_on"]:
        feats = [c for c in sp["features"] if c in df.columns]
        if not feats:
            return -1e9
        fr = ChoiceSetFrame.from_long(df, feats)
        b, ll, _ = _fit_frame_mnl(fr, np.zeros(fr.dimension))
        k, n = fr.dimension, fr.n_cases
        return -(-2.0 * ll + PEN * k * np.log(max(n, 2))) / max(n, 1)
    sc, _ = fit_lc_genome(df, {c: c for c in df.columns}, sp,
                          complexity_penalty=PEN, seed=11)
    return -1e9 if sc is None else sc


pooled_score = genome_fitness(genome("log_DIST_1", "log_CRASH_1", "visit_work"))
ga = GeneticAlgorithm(
    len(CANDS), genome_fitness, pop_size=8, n_generations=4,
    mutation_rate=0.15, min_features=1, feature_names=CANDS, seed=3,
    verbose=False, progress_label="lc-smoke")
best, hist, pop = ga.run_with_restarts(n_restarts=1)
best_vars = [v for v, g in zip(CANDS, best.genes) if g == 1]
sp_best = decode_lc_spec(best.genes, CANDS, BASE, available_membership=MEMB)
print(f"  pooled baseline BIC/n={-pooled_score:.4f}")
print(f"  GA best BIC/n={-best.fitness:.4f}  vars={best_vars}")
print(f"  decoded: lc_on={sp_best['lc_on']} nc={sp_best['n_classes']} "
      f"memb={sp_best['membership_vars']}")
check("T8 GA selects lc_on", sp_best["lc_on"] is True, f"vars={best_vars}")
check("T8 selected spec is crash-based",
      any("crash" in c.lower() for c in sp_best["features"]),
      f"features={sp_best['features']}")
check("T8 LC genome beats pooled baseline", best.fitness > pooled_score,
      f"lc={-best.fitness:.4f} pooled={-pooled_score:.4f}")

print("\n=== T9-T12: behavioral pool + doctor gate ===")
pool_d = extend_candidates(BASE)
memb_genes = [c for c in pool_d if c.startswith("lc_memb:")]
check("T9 default membership genes are behavioral-only",
      memb_genes == ["lc_memb:STUDYING", "lc_memb:age5p", "lc_memb:sexp",
                     "lc_memb:safety_sensitivity"],
      f"{memb_genes}")
os.environ["STAGE5_LC_MEMB_EXTRA"] = "incp,vehrd"
pool_e = extend_candidates(BASE)
memb_genes_e = [c for c in pool_e if c.startswith("lc_memb:")]
check("T9 MEMB_EXTRA appends structural vars",
      memb_genes_e[-2:] == ["lc_memb:incp", "lc_memb:vehrd"],
      f"{memb_genes_e}")
del os.environ["STAGE5_LC_MEMB_EXTRA"]

CANDS_D = extend_candidates(BASE)
g12 = np.zeros(len(CANDS_D), dtype=int)
g12[CANDS_D.index("log_DIST_1")] = 1
g12[CANDS_D.index("lc_on")] = 1
sp12 = decode_lc_spec(g12, CANDS_D, BASE,
                      available_membership=["STUDYING", "incp", "vehrd"])
check("T12 silent fallback keeps behavioral components only",
      sp12["membership_vars"] == ["STUDYING"],
      f"memb={sp12['membership_vars']}")

df_d = df.copy()
df_d["log_DIST_1_dup"] = df_d["log_DIST_1"]
c2c_d = {c: c for c in ["log_DIST_1", "log_CRASH_1", "visit_work",
                        "log_DIST_1_dup"]}
spec_d = decode_lc_spec(genome("log_DIST_1", "log_CRASH_1", "visit_work", "lc_on",
                               "lc_memb:student", "lc_memb:incp_std"),
                        CANDS, BASE, available_membership=MEMB)
spec_d["features"] = list(spec_d["features"]) + ["log_DIST_1_dup"]
os.environ["STAGE5_LC_DOCTOR"] = "prune"
score_d, info_d = fit_lc_genome(df_d, c2c_d, spec_d, seed=7)
check("T10 doctor prunes the duplicate, fit proceeds",
      score_d is not None and score_d > -1e8
      and len(info_d.get("pruned", [])) == 1,
      f"score={score_d} pruned={info_d.get('pruned')} "
      f"drops={info_d.get('doctor_drops')} "
      f"cond={info_d.get('doctor_condition')}")
os.environ["STAGE5_LC_DOCTOR"] = "reject"
score_r, info_r = fit_lc_genome(df_d, c2c_d, spec_d, seed=7)
check("T11 doctor reject fails the collinear spec",
      score_r is not None and score_r <= -1e8
      and "doctor" in str(info_r.get("reason", "")).lower(),
      f"reason={info_r.get('reason')}")
del os.environ["STAGE5_LC_DOCTOR"]

n_pass = sum(1 for ok, _, _ in RESULTS if ok)
print(f"\n{n_pass}/{len(RESULTS)} checks passed")
sys.exit(0 if n_pass == len(RESULTS) else 1)
