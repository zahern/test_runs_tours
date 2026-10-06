"""Mode-model perturbation harness for Stage 5.

Fits the mode model (the same path _run_segment uses), then jitters the fitted
coefficients and refits, recording for every fit:
  * mode LL and the gradient-norm convergence gate,
  * the logsum surface it produces and its WITHIN-CASE across-zone std
    (the quantity that determines whether c_logsum can identify).

This is the mode-side analogue of _s5_larch_perturb.py, and it validates that
perturbed mode solutions still produce healthy logsums for the destination
model.

Env knobs:
  SEG=work              segment
  MODE_N_TOURS=0        subsample tours (0 = all)
  MODE_PERTURB_N=2      number of jittered refits
  MODE_PERTURB_SIGMA=0.5
  MODE_DRAWS...         not used (larch optimizer)
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NEW_POP", "1")
os.environ["NEW_POP_DIR"] = r"Z:\test_runs_tours\new_pop"
os.environ["NEW_POP_MAPPED"] = "0"
os.environ["PIPELINE_SKIMS_PATH"] = r"Z:\test_runs_tours\demand_\skims_371.omx"
os.environ["PIPELINE_SEED"] = "42"
os.environ["PIPELINE_ACCELERATOR"] = "cpu"
os.environ["LARCH_FORCE_JAX"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["STAGE5_CLEAR_SHARROW_CACHE"] = "1"
os.environ["STAGE5_DEST_SAMPLE_SIZE"] = "20"

CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import pipeline_logger as pl
import openmatrix as omx
import stage5_destination_choice as S5
from stage5_destination_choice import (
    _skim_flat, _skim_lookup, _prep_hhts_tours, _filter_seg,
)

SEG = os.environ.get("SEG", "work")
N_TOURS = int(os.environ.get("MODE_N_TOURS", "0"))
N_PERT = int(os.environ.get("MODE_PERTURB_N", "2"))
SIGMA = float(os.environ.get("MODE_PERTURB_SIGMA", "0.5"))
OUT = os.path.join(CODE, "runs", "_s5_larch_perturb", f"{SEG}_mode_perturb.json")

_, out, landuse = pl.load_data()
hhts = _prep_hhts_tours(out["tours"], out["persons"])
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = _skim_flat(skims)
_skim_lookup(skim_df)
seg = _filter_seg(hhts, SEG)
if N_TOURS and N_TOURS < len(seg):
    seg = seg.sample(n=N_TOURS, random_state=42).reset_index(drop=True)
    print(f"[mode-perturb] subsampled to {len(seg)} tours", flush=True)
print(f"[mode-perturb] segment={SEG} tours={len(seg):,}", flush=True)

t0 = time.time()
dt_m = S5._mode_tree(seg, out["persons"], landuse, skims, skim_df=skim_df)
print(f"[mode-perturb] _mode_tree built in {time.time()-t0:.0f}s "
      f"type={type(dt_m).__name__}", flush=True)

HINT = (list(seg.columns) + list(out["persons"].columns))


def fit_model(warm=None, jitter_sigma=0.0, rng=None):
    mm = S5._build_mode_model(dt_m, SEG, avail_vars_hint=HINT)
    try:
        mm.doctor(repair_ch_av="+")
    except Exception:
        pass
    if warm:
        for n, v in warm.items():
            if n in mm.pf.index and np.isfinite(float(v)):
                try:
                    mm.set_value(n, float(v))
                except Exception:
                    pass
    if jitter_sigma > 0 and rng is not None:
        try:
            pf = mm.pf
            hf = (pf['holdfast'].fillna(False).astype(bool)
                  if 'holdfast' in pf.columns else np.zeros(len(pf), bool))
        except Exception:
            hf = np.zeros(0, bool)
        for name in pf.index:
            if bool(hf.get(name, False)):
                continue
            try:
                v = float(mm.pf.loc[name, 'value'])
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            d = float(rng.normal(0.0, jitter_sigma * max(abs(v), 0.05)))
            v2 = v + d
            try:
                lo = float(mm.pf.loc[name, 'minimum'])
                hi = float(mm.pf.loc[name, 'maximum'])
                if np.isfinite(lo):
                    v2 = max(v2, lo + 1e-6)
                if np.isfinite(hi):
                    v2 = min(v2, hi - 1e-6)
            except Exception:
                pass
            try:
                mm.set_value(name, v2)
            except Exception:
                pass
    mm = S5._prefit_mode_model(mm, seg, out["persons"], skims, SEG)
    mm = S5._estimate(mm)
    return mm


def record(tag, mm, seconds):
    ll = float(getattr(mm, "_cached_ll", float("nan")))
    diag = dict(getattr(mm, "_fit_diagnostics", {}) or {})
    cc = dict(getattr(mm, "_coef_cache", {}) or {})
    nz = sum(1 for v in cc.values() if abs(float(v)) > 1e-9)
    try:
        t1 = time.time()
        ls = S5._logsums(mm, dt_m, skims=skims, verbose=False)
        within = np.nanstd(np.asarray(ls.values, dtype=float), axis=1)
        flat = float(np.nanmean(within < 1e-9))
        med = float(np.nanmedian(within))
        p95 = float(np.nanpercentile(within, 95))
        ls_s = time.time() - t1
    except Exception as e:
        flat = med = p95 = float("nan")
        ls_s = 0.0
        print(f"[mode-perturb] {tag}: logsum failed {e!r}", flush=True)
    print(f"[mode-perturb] {tag:10s} LL={ll:12.2f} grad={diag.get('grad_norm')} "
          f"stationary={diag.get('stationary')} conv={diag.get('converged')} "
          f"n_free={diag.get('n_free')} coef_nz={nz} | logsum within-case std "
          f"median={med:.5g} p95={p95:.5g} flat={flat:.1%} "
          f"({seconds:.0f}s fit, {ls_s:.0f}s ls)", flush=True)
    return {"tag": tag, "ll": ll, "grad_norm": diag.get("grad_norm"),
            "stationary": diag.get("stationary"),
            "converged": diag.get("converged"), "n_free": diag.get("n_free"),
            "coef_nonzero": nz, "logsum_within_median": med,
            "logsum_within_p95": p95, "logsum_flat_frac": flat,
            "seconds": round(seconds, 1), "coefs": cc}


results = []
t0 = time.time()
base = fit_model()
results.append(record("baseline", base, time.time() - t0))
warm = dict(getattr(base, "_coef_cache", {}) or {})
rng = np.random.default_rng(123)
for i in range(1, N_PERT + 1):
    t0 = time.time()
    mm = fit_model(warm=warm, jitter_sigma=SIGMA, rng=rng)
    results.append(record(f"jitter {i}", mm, time.time() - t0))

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"segment": SEG, "n_tours": len(seg), "sigma": SIGMA,
               "runs": results}, fh, indent=2, default=str)
print(f"[mode-perturb] results -> {OUT}", flush=True)
try:
    skims.close()
except Exception:
    pass
