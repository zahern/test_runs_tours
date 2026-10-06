"""Larch latent-class DESTINATION diagnostic + perturbation.

Tests whether larch's LatentClass can fit a destination model where:
  * each class has its own destination coefficients (distance, crash, logsum,
    visitability, agglomeration, size), and
  * class membership is a function of PERSON TYPES only (income, vehicles,
    student, age, sex, safety sensitivity).

Modes:
  LC_SMOKE=1   tiny synthetic flat dataset -> validates the larch LC API fast
  (default)    real cached QLD work frame, subsampled to LC_N_CASES

Then perturbs the fitted parameter vector (class-specific + membership) and
refits LC_PERTURB_N times, reporting LL / convergence / shares, i.e. the
diagnostic the metaheuristic solver would wrap as fitness.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("PIPELINE_SKIMS_PATH",
                      r"Z:\test_runs_tours\demand_\skims_371.omx")
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import pandas as pd

import larch as lx
from larch import P, X

SMOKE = os.environ.get("LC_SMOKE", "0") == "1"
N_CASES = int(os.environ.get("LC_N_CASES", "1200"))
N_PERT = int(os.environ.get("LC_PERTURB_N", "2"))
SIGMA = float(os.environ.get("LC_PERTURB_SIGMA", "0.5"))
OUT = os.path.join(CODE, "runs", "_s5_larch_perturb", "larch_lc_dest.json")


def patch_larch_flow():
    try:
        from larch.dataset import flow as _flow
        def _safe(self, name):
            if name == "_obj": raise AttributeError(name)
            o = self.__dict__.get("_obj")
            if o is None: raise AttributeError(name)
            r = getattr(o, name)
            try: r = self.transfer_dimension_attrs(r)
            except Exception: pass
            return r
        for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
            k = getattr(_flow, c, None)
            if k is not None: k.__getattr__ = _safe
    except Exception as e:
        print("flow patch skipped:", e)


patch_larch_flow()
import stage5_destination_choice as S5


def build_smoke():
    rng = np.random.default_rng(7)
    N, J = 400, 4
    rows = []
    true = {1: np.array([-1.0, -0.4, 0.5]), 2: np.array([-0.2, -1.2, -0.3])}
    for n in range(N):
        incp = rng.integers(1, 10)
        veh = rng.integers(0, 3)
        p2 = 1.0 / (1.0 + np.exp(-(0.4 * (incp - 5) + 0.5 * veh)))
        cls = 2 if rng.random() < p2 else 1
        x = rng.normal(size=(J, 3))
        x[:, 0] = np.log1p(rng.uniform(1, 60, J))
        x[:, 1] = np.log1p(rng.uniform(0, 300, J))
        u = x @ true[cls]
        pr = np.exp(u - u.max()); pr /= pr.sum()
        ch = int(rng.choice(J, p=pr))
        for a in range(J):
            rows.append({"case_id": f"c{n}", "alt_id": a, "chosen": int(a == ch),
                         "log_correction": 0.0, "log_DIST_1": x[a, 0],
                         "log_CRASH_1": x[a, 1], "visit_work": x[a, 2],
                         "incp": incp, "vehrd": veh})
    df = pd.DataFrame(rows)
    dtaz = df[df["chosen"] == 1].set_index("case_id")["alt_id"]
    df["DTAZ"] = df["case_id"].map(dtaz).astype(int)
    # The `chosen` CA indicator must not leak into the class-membership
    # dataset (it conflicts with the class alternative axis); choice is
    # resolved from the DTAZ CO variable instead.
    case_df = (df.drop_duplicates("case_id").set_index("case_id")
               [["incp", "vehrd"]].copy())
    case_df.index.name = "case_id"
    df = df.drop(columns=["chosen"]).set_index(["case_id", "alt_id"])
    dt = lx.Dataset.construct.from_idca(df)
    dt.attrs["_flat_mode"] = True
    for cov in ("incp", "vehrd"):
        dt = dt.assign(**{f"m_{cov}": ("case_id",
                                       np.asarray(case_df[cov].values, float))})
    return (dt, {"log_DIST_1": "log_dist", "log_CRASH_1": "crash",
                 "visit_work": "visit"}, ["incp", "vehrd"], case_df)


def build_real():
    import cloudpickle
    cache = cloudpickle.load(open(
        r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
    seg = cache["seg_acdc"]
    sa = cache["sampled_alts"]
    if N_CASES and N_CASES < seg["ptour_id"].nunique():
        keep = (seg[["ptour_id"]].drop_duplicates()
                .sample(n=N_CASES, random_state=42)["ptour_id"].astype(str))
        seg = seg[seg["ptour_id"].astype(str).isin(set(keep))].copy()
        sa = sa[sa["ptour_id"].astype(str).isin(set(keep))].copy()
    import openmatrix as omx
    skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
    skim_df = S5._skim_flat(skims)
    S5._skim_lookup(skim_df)
    dt = S5._dest_tree(seg, cache["persons"], cache["lu_with_st"], skims,
                       cache["logsums"], sampled_alts=sa, skim_df=skim_df,
                       crash_skims=skims, segment="work")
    skims.close()
    covs = [c for c in ("incp", "vehrd", "STUDYING", "age5p", "sexp",
                        "safety_sensitivity") if c in dt.data_vars]
    print(f"[real] cases={seg['ptour_id'].nunique()} membership covariates={covs}")
    terms = {}
    for col, name in (("log_DIST_1", "log_dist"), ("log_CRASH_1", "crash"),
                      ("logsum", "logsum"), ("visit_work", "visit_work"),
                      ("acdc_aggl", "aggl"), ("size_term", "size")):
        if col in dt.data_vars:
            terms[col] = name
    # Case-level membership dataset (one row per tour) for the class equation.
    case_vals = {}
    for cov in covs:
        arr = np.asarray(dt[cov].values)
        case_vals[cov] = (arr[:, 0] if arr.ndim > 1 else arr).astype(float)
    case_df = pd.DataFrame(case_vals,
                           index=pd.Index(
                               np.asarray(dt.coords["ptour_id"].values),
                               name="ptour_id"))
    # Attach the membership covariates as case-level CO variables so the
    # membership equation resolves them regardless of which dataset larch
    # ends up using for the class model.
    for cov in covs:
        dt = dt.assign(**{f"m_{cov}": ("ptour_id",
                                       np.asarray(case_df[cov].values, float))})
    return dt, terms, covs, case_df


def class_model(dt, c, terms):
    m = S5._lx_model(dt, compute_engine="jax")
    expr = None
    for col, name in terms.items():
        t = P(f"c{c}_{name}") * X(col)
        expr = t if expr is None else expr + t
    m.utility_ca = expr
    m.choice_co_code = "DTAZ"
    m.availability_ca_var = "DIST < 150" if "DIST" in dt.data_vars else 1
    return m


def membership_model(covs):
    """Membership equation over PERSON TYPES only.

    The covariates are referenced as m_<cov> because they are attached to the
    choice datatree as case-level CO variables (see build_*); referencing the
    raw names hit larch's 'unable to rewrite X to itself' when the same name
    also exists as a CA variable.
    """
    mk = lx.Model()
    expr = None
    for cov in covs:
        t = P(f"g_{cov}") * X(f"m_{cov}")
        expr = t if expr is None else expr + t
    mk.utility_co[102] = expr
    return mk


def make_lc(dt, terms, covs, case_df):
    m1 = class_model(dt, 1, terms)
    m2 = class_model(dt, 2, terms)
    mk = membership_model(covs)
    # larch's supported LC path for covariates is groupid=: the class model's
    # case-level dataset is DERIVED from the choice datatree
    # (dissolve_zero_variance over the alternative axis), which keeps exactly
    # the person-type covariates.  groupid must be available as a CO column,
    # not an index level, hence the explicit group_id variable.  Class keys
    # 101/102 must not collide with the choice alternative ids.
    case_dim = next((c for c in ("ptour_id", "case_id") if c in dt.coords), None)
    if case_dim is None:
        raise KeyError(f"no case coordinate on dataset: {list(dt.coords)}")
    case_ids = np.asarray(dt.coords[case_dim].values)
    dt2 = dt.assign(group_id=(case_dim, case_ids))
    b = lx.LatentClass(mk, {101: m1, 102: m2}, datatree=dt2,
                       groupid="group_id")
    b.set_cap(20)
    return b


def shares(b, case_df):
    """Implied class shares from the membership equation (class 101 = ref)."""
    try:
        pf = b.class_membership_model.pf
        u = np.zeros(len(case_df), dtype=float)
        for col in case_df.columns:
            name = f"g_{col}"
            if name in pf.index:
                u += float(pf.loc[name, "value"]) * np.asarray(case_df[col], float)
        p2 = 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))
        return [float(1.0 - p2.mean()), float(p2.mean())]
    except Exception as e:
        return f"n/a ({type(e).__name__}: {e})"


def membership_coefs(b):
    try:
        pf = b.class_membership_model.pf
        return {str(n): float(pf.loc[n, "value"]) for n in pf.index}
    except Exception:
        return {}


def fit_once(b):
    t0 = time.time()
    try:
        res = b.maximize_loglike(method="slsqp",
                                 options={"maxiter": 300, "ftol": 1e-7},
                                 quiet=True)
        ok = res.get("success")
        msg = res.get("message", "")
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {e}"
    try:
        ll = float(b.loglike(check_if_best=False, error_if_bad=False))
    except Exception:
        ll = float("nan")
    try:
        g = float(np.linalg.norm(np.nan_to_num(np.asarray(b.d_loglike(), float))))
    except Exception:
        g = float("nan")
    return ok, msg, ll, g, time.time() - t0


def run(tag, dt, terms, covs, case_df, base_pvals=None):
    b = make_lc(dt, terms, covs, case_df)
    if base_pvals is not None:
        try:
            b.pvals = base_pvals + 0.0
        except Exception as e:
            print("  pvals set failed:", e)
    ok, msg, ll, g, secs = fit_once(b)
    pf = b.pf
    row = {"tag": tag, "success": bool(ok), "message": str(msg)[:120],
           "ll": ll, "grad_norm": g, "seconds": round(secs, 1),
           "shares": shares(b, case_df),
           "membership": membership_coefs(b)}
    print(f"[{tag}] success={ok} LL={ll:.1f} |grad|={g:.3g} "
          f"shares={row['shares']} ({secs:.0f}s)", flush=True)
    try:
        coefs = {str(n): float(pf.loc[n, "value"]) for n in pf.index}
    except Exception:
        coefs = {}
    row["coefs"] = coefs
    return row, b


dt, terms, covs, case_df = build_smoke() if SMOKE else build_real()
print(f"terms={list(terms)} covariates={covs} cases={len(case_df)}")

rows = []
r0, b0 = run("baseline", dt, terms, covs, case_df)
rows.append(r0)
base = np.asarray(b0.pvals, float).copy()

rng = np.random.default_rng(123)
for i in range(1, N_PERT + 1):
    jitter = rng.normal(0.0, SIGMA * np.maximum(np.abs(base), 0.05))
    r, _ = run(f"perturb {i}", dt, terms, covs, case_df,
               base_pvals=base + jitter)
    rows.append(r)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"smoke": SMOKE, "n_cases": N_CASES, "rows": rows}, fh,
              indent=2, default=str)
print("saved ->", OUT)
