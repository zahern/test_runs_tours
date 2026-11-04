"""_diag_logsum_ident.py -- is arm B's c_logsum a genuine zero effect, or a
column the model cannot separate from distance?

Arm B comes back with c_logsum = 3.6e-16, i.e. pinned at the ridge.  Two very
different stories produce that number:

  (1) GENUINE ZERO   the data place the optimum at 0 and the fit found it;
  (2) COLLINEAR      the logsum column carries almost no within-case variation
                     independent of the distance terms, so its coefficient is
                     unidentified and the ridge parks it at 0.

They are told apart with measurements on the SAME frame the paper arm fitted:

  * within-case variance of every utility column -- a column that is constant
    within a case contributes nothing to a conditional logit;
  * the exact score and information at beta=0 (the estimator's own objects);
  * a profile likelihood in c_logsum, refitting the other four coefficients at
    every fixed value (the ground truth for "does the data care");
  * the same profile for the RAW mode logsum, i.e. with _dest_tree's
    within-case logsum-perp-distance orthogonalization undone, to see whether
    the preprocessing is what removed the signal.

Run with the same environment as job_stage5_paper_arms.pbs.
"""
import json
import os
import time

import numpy as np
import pandas as pd

SEG = os.environ.get("PAPER_SEGMENT", "work")
COEFS = [c.strip() for c in os.environ.get(
    "PAPER_UTILITY",
    "c_log_dist,c_crash,c_logsum,c_visit_work,c_crash_safety_dest"
).replace(";", ",").split(",") if c.strip()]
MEMB_WANT = [c.strip() for c in os.environ.get(
    "PAPER_MEMB", "STUDYING,sexp,incp,vehrd"
).replace(";", ",").split(",") if c.strip()]
CACHE = os.environ.get(
    "PAPER_CACHE", os.path.join("runs", "_s5_larch_perturb", f"{SEG}_dest_cache.pkl"))
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = os.environ.get("DIAG_OUT", os.path.join("runs", "_s5_logsum_diag", STAMP))
os.makedirs(OUT, exist_ok=True)
FRAME_PARQUET = os.path.join(OUT, "work_dest_frame.parquet")

REPORT = {"segment": SEG, "coefs": COEFS, "cache": CACHE}


def say(msg):
    print(msg, flush=True)


# ── frame: identical construction to _paper_stage5_arms.py ─────────────────
import cloudpickle
import openmatrix as omx
import stage5_destination_choice as S5
from SearchLibrium.mh_choice_estimator import (
    ChoiceSetFrame, _fit_frame_mnl, identify_weak_features)

say(f"[frame] loading {CACHE}")
cache = cloudpickle.load(open(CACHE, "rb"))
if cache.get("segment") not in (None, SEG):
    raise SystemExit(f"cache is for {cache.get('segment')!r}, not {SEG!r}")
seg = cache["seg_acdc"]
sa = cache["sampled_alts"]

skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
dt = S5._dest_tree(seg, cache["persons"], cache["lu_with_st"], skims,
                   cache["logsums"], sampled_alts=sa, skim_df=skim_df,
                   crash_skims=skims, segment=SEG)
skims.close()
frame, fmap = S5._adaptive_dest_choice_frame_full(
    dt, sa, SEG, case_covariates=MEMB_WANT)
coef_to_col = {str(v): str(k) for k, v in fmap.items()}
missing = [c for c in COEFS if c not in coef_to_col]
if missing:
    raise KeyError(f"utility coefs missing from frame: {missing}")
COLS = [coef_to_col[c] for c in COEFS]
LS_COL = coef_to_col["c_logsum"]
say(f"[frame] cases={frame['case_id'].nunique()} rows={len(frame)} "
    f"logsum col={LS_COL!r}")

# The frame is expensive to rebuild (~4 min); keep it for follow-up checks.
try:
    frame.to_parquet(FRAME_PARQUET)
    say(f"[frame] saved {FRAME_PARQUET}")
except Exception as exc:                                    # noqa: BLE001
    say(f"[frame] parquet save skipped ({exc})")

# ── raw mode logsum, i.e. before _dest_tree orthogonalized it ──────────────
# The frame's logsum column is (case mean of the raw logsum) + (within-case
# residual orthogonal to DIST and log_DIST_1).  The cache still holds the raw
# cases × zones surface, so the un-orthogonalized column can be recovered
# exactly instead of guessed at.
RAW_COL = "logsum_raw"
raw_ok = False
try:
    ls = cache["logsums"]
    ls_vals = np.asarray(ls.values, dtype=float)
    ls_cases = [str(c) for c in ls.coords["cases"].values]
    ls_zones = [int(z) for z in ls.coords["zones"].values]
    c_row = {c: i for i, c in enumerate(ls_cases)}
    z_col = {z: j for j, z in enumerate(ls_zones)}
    cid = frame["case_id"].astype(str).values
    aid = frame["alt_id"]
    aid = pd.to_numeric(aid, errors="coerce").astype("Int64").values
    vals = np.full(len(frame), np.nan)
    hit = 0
    for i in range(len(frame)):
        r = c_row.get(cid[i])
        z = aid[i]
        if r is None or z is None or pd.isna(z):
            continue
        j = z_col.get(int(z))
        if j is None:
            continue
        vals[i] = ls_vals[r, j]
        hit += 1
    frac = hit / max(len(frame), 1)
    say(f"[raw] raw logsum matched {hit}/{len(frame)} rows ({frac:.3%})")
    if frac > 0.99 and np.isfinite(vals).all():
        frame[RAW_COL] = vals
        raw_ok = True
    REPORT["raw_logsum_match"] = frac
except Exception as exc:                                    # noqa: BLE001
    say(f"[raw] could not rebuild raw logsum ({type(exc).__name__}: {exc})")
    REPORT["raw_logsum_error"] = f"{type(exc).__name__}: {exc}"


# ── 1. within-case variation ───────────────────────────────────────────────
def within_stats(col):
    v = frame[col].astype(float).values
    g = frame.groupby("case_id", sort=False)[col]
    m = g.transform("mean").astype(float).values
    d = v - m
    per_case_var = (pd.Series(d * d, index=frame.index)
                    .groupby(frame["case_id"], sort=False).mean())
    n_alts = frame.groupby("case_id", sort=False).size()
    multi = per_case_var[n_alts >= 2]
    tot = float(np.var(v))
    mean_w = float(multi.mean())
    return dict(total_sd=float(np.sqrt(tot)),
                within_sd=float(np.sqrt(max(mean_w, 0.0))),
                within_frac_of_total=float(np.sqrt(max(mean_w, 0.0) / tot)) if tot > 0 else 0.0,
                cases_zero_within=int((multi <= 0.0).sum()),
                cases=int(multi.size))


say("\n=== within-case variation (the only variation a conditional logit uses) ===")
within = {}
for c in COLS + ([RAW_COL] if raw_ok else []):
    st = within_stats(c)
    within[c] = st
    say(f"  {c:<16} total_sd={st['total_sd']:.6g}  within_sd={st['within_sd']:.6g}"
        f"  within/total={st['within_frac_of_total']:.3g}"
        f"  zero-within cases={st['cases_zero_within']}/{st['cases']}")
REPORT["within"] = within


# ── 2. collinearity of the logsum with the distance terms, within case ─────
# _dest_tree's orthogonalization used raw DIST and log_DIST_1; the frame may
# only carry the log transform, so rebuild DIST from it when absent.
if "DIST" in frame.columns:
    frame["_DIST"] = frame["DIST"].astype(float)
else:
    frame["_DIST"] = np.expm1(frame["log_DIST_1"].astype(float))


def within_collinearity(col):
    g = frame.groupby("case_id", sort=False)
    y = (frame[col].astype(float) - g[col].transform("mean").astype(float)).values
    vd = frame["_DIST"].astype(float).values
    vld = frame["log_DIST_1"].astype(float).values
    x1 = vd - g["_DIST"].transform("mean").astype(float).values
    x2 = vld - g["log_DIST_1"].transform("mean").astype(float).values
    X = np.column_stack([x1, x2])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if np.var(y) <= 0:
        return dict(r_dist=float("nan"), r_logdist=float("nan"), r2=1.0,
                    resid_sd=0.0)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1.0 - float(np.var(res) / np.var(y))
    return dict(r_dist=float(np.corrcoef(x1[ok], y)[0, 1]),
                r_logdist=float(np.corrcoef(x2[ok], y)[0, 1]),
                r2=r2, resid_sd=float(np.sqrt(np.var(res))),
                slope=[float(b[0]), float(b[1])])


say("\n=== within-case collinearity with DIST / log_DIST_1 ===")
collin = {}
for c in [LS_COL] + ([RAW_COL] if raw_ok else []):
    cc = within_collinearity(c)
    collin[c] = cc
    say(f"  {c:<16} r(DIST)={cc['r_dist']:+.4f}  r(log_DIST_1)={cc['r_logdist']:+.4f}"
        f"  R^2={cc['r2']:.5f}  resid_sd={cc['resid_sd']:.4g}"
        f"  slopes={np.round(cc.get('slope', []), 4).tolist()}")
REPORT["within_collinearity"] = collin


# ── 3. exact score + information at beta = 0 (the estimator's own objects) ─
def ident(frame_obj, label):
    info, grad, _ = frame_obj.information(np.zeros(frame_obj.dimension))
    dg = np.clip(np.diag(info), 0.0, None)
    se = np.where(dg > 0, 1.0 / np.sqrt(np.maximum(dg, 1e-300)), np.inf)
    scale = np.sqrt(np.maximum(dg, 1e-300))
    corr = info / np.outer(scale, scale)
    evals = np.linalg.eigvalsh(0.5 * (corr + corr.T))
    rep = dict(names=list(frame_obj.feature_names),
               gradient=[float(x) for x in grad],
               info_diag=[float(x) for x in dg],
               se=[float(x) for x in se],
               cond=float(evals[-1] / max(evals[0], 1e-300)),
               eig_min=float(evals[0]), eig_max=float(evals[-1]))
    say(f"\n=== [{label}] score and information at beta=0 ===")
    for n, g_, s_ in zip(rep["names"], rep["gradient"], rep["se"]):
        say(f"  {n:<16} score={g_:+.4g}  SE={s_:.4g}")
    say(f"  condition(corr)={rep['cond']:.4g}  eig_min={rep['eig_min']:.4g}")
    return rep


fr_full = ChoiceSetFrame.from_long(frame, COLS)
REPORT["ident_frame"] = ident(fr_full, "frame as fitted")
if raw_ok:
    _fraw = frame.copy()
    _fraw[LS_COL] = _fraw[RAW_COL]
    REPORT["ident_raw"] = ident(ChoiceSetFrame.from_long(_fraw, COLS), "raw logsum")


# ── 4. arm B reference fit (must reproduce the reported -8813.6656) ────────
b_b, ll_pen, res_b = _fit_frame_mnl(fr_full, np.zeros(fr_full.dimension),
                                    maxiter=2000)
ll5 = float(fr_full.loglike_and_gradient(b_b)[0])
say(f"\n=== reference 5-parameter fit: LL={ll5:.6f}  "
    f"(arm B reported -8813.665628) ===")
say("  coefs: " + ", ".join(f"{n}={v:+.6g}"
                           for n, v in zip(fr_full.feature_names, b_b)))
REPORT["ref_fit"] = dict(ll=ll5,
                         coefs={n: float(v) for n, v in
                                zip(fr_full.feature_names, b_b)},
                         grad_inf=float(getattr(res_b, "grad_inf", np.nan)))


# ── 5. profile likelihood in c_logsum, other four refitted each time ───────
GRID = [-15.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.0,
        0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]


def profile(ls_col, label):
    others = [c for c in COLS if c != LS_COL]
    rows = []
    for t in GRID:
        df = frame[["case_id", "alt_id", "chosen"]].copy()
        for c in others:
            df[c] = frame[c].astype(float)
        df["off"] = (frame["log_correction"].astype(float)
                     + float(t) * frame[ls_col].astype(float))
        fr = ChoiceSetFrame.from_long(df, others, offset_col="off")
        bb, _, rr = _fit_frame_mnl(fr, np.zeros(len(others)), maxiter=2000)
        ll = float(fr.loglike_and_gradient(bb)[0])
        rows.append(dict(t=float(t), ll=ll,
                         beta={c: float(v) for c, v in zip(others, bb)},
                         grad_inf=float(getattr(rr, "grad_inf", np.nan))))
    base = next(r["ll"] for r in rows if r["t"] == 0.0)
    near = [r for r in rows if abs(r["t"]) <= 1.0]
    tt = np.array([r["t"] for r in near])
    llv = np.array([r["ll"] for r in near])
    se = float("nan")
    if tt.size >= 3:
        A = np.vstack([tt ** 2, tt, np.ones_like(tt)]).T
        coef, *_ = np.linalg.lstsq(A, llv, rcond=None)
        curv = 2.0 * coef[0]                      # d2LL/dt2
        se = float(1.0 / np.sqrt(-curv)) if curv < 0 else float("inf")
    say(f"\n=== profile likelihood in {label}: other four refit at each t ===")
    for r in rows:
        flag = "  <-- bound" if abs(r["t"]) >= 15.0 else ""
        say(f"  t={r['t']:+7.2f}  LL={r['ll']:.4f}  dLL={r['ll'] - base:+.6g}"
            f"  |g|={r['grad_inf']:.1e}{flag}")
    say(f"  implied SE({label}) from curvature near 0: {se:.4g}")
    return dict(rows=rows, base_ll=base, se=se,
                max_dll=float(max(abs(r["ll"] - base) for r in rows)))


prof_frame = profile(LS_COL, "c_logsum (frame as fitted)")
REPORT["profile_frame"] = prof_frame
if raw_ok:
    REPORT["profile_raw"] = profile(RAW_COL, "c_logsum (raw mode logsum)")


# ── 6. drop the term entirely: what does it cost? ─────────────────────────
others = [c for c in COLS if c != LS_COL]
df4 = frame[["case_id", "alt_id", "chosen", "log_correction"] + others].copy()
fr4 = ChoiceSetFrame.from_long(df4, others)
b4, _, res4 = _fit_frame_mnl(fr4, np.zeros(len(others)), maxiter=2000)
ll4 = float(fr4.loglike_and_gradient(b4)[0])
lr = 2.0 * (ll5 - ll4)
say(f"\n=== dropping c_logsum: LL={ll4:.6f} (5-param {ll5:.6f}) ===")
say(f"  LR = {lr:.4g} on 1 df (chi2_95 = 3.84); other coefs shift: "
    + ", ".join(f"{n}={v:+.6g}" for n, v in zip(others, b4)))
REPORT["drop_logsum"] = dict(ll=ll4, lr=float(lr),
                            coefs={n: float(v) for n, v in zip(others, b4)},
                            shift={n: float(v - b_b[fr_full.feature_names.index(n)])
                                   for n, v in zip(others, b4)})

weak_frame = [list(getattr(identify_weak_features(fr_full, beta=b_b),
                           "suggested_drop", []) or [])]
if raw_ok:
    weak_frame.append(list(getattr(identify_weak_features(
        ChoiceSetFrame.from_long(_fraw, COLS), beta=b_b), "suggested_drop", []) or []))
REPORT["identify_weak_features"] = dict(frame=weak_frame[0],
                                        raw=weak_frame[1] if raw_ok else None)
say(f"\nidentify_weak_features -> frame={weak_frame[0]}"
    + (f" raw={weak_frame[1]}" if raw_ok else ""))

with open(os.path.join(OUT, "logsum_ident.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
say(f"\nsaved -> {OUT}/logsum_ident.json")
