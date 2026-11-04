"""Run larch's doctor on the real mode model (subsampled) and print the
near-dependent parameter groups with eigenvector loadings."""
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
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np
import pipeline_logger as pl
import openmatrix as omx
import stage5_destination_choice as S5
from stage5_destination_choice import _skim_flat, _skim_lookup, _prep_hhts_tours, _filter_seg

N = int(os.environ.get("MODE_N_TOURS", "1000"))
_, out, landuse = pl.load_data()
hhts = _prep_hhts_tours(out["tours"], out["persons"])
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = _skim_flat(skims)
_skim_lookup(skim_df)
seg = _filter_seg(hhts, "work")
if N and N < len(seg):
    seg = seg.sample(n=N, random_state=42).reset_index(drop=True)
print(f"[mode-doctor] tours={len(seg)}", flush=True)
t0 = time.time()
dt_m = S5._mode_tree(seg, out["persons"], landuse, skims, skim_df=skim_df)
print(f"[mode-doctor] tree {time.time()-t0:.0f}s", flush=True)
HINT = list(seg.columns) + list(out["persons"].columns)
mm = S5._build_mode_model(dt_m, "work", avail_vars_hint=HINT)
try:
    mm.doctor(repair_ch_av="+")
except Exception:
    pass
mm = S5._prefit_mode_model(mm, seg, out["persons"], skims, "work")
try:
    print("loglike at priors:", float(mm.loglike(check_if_best=False, error_if_bad=False)), flush=True)
except Exception as e:
    print("loglike failed:", e, flush=True)

from larch.model import troubleshooting
mm2, problems = troubleshooting.doctor(
    mm, check_low_variance_data_co="?", check_overspec="?", verbose=3)
print("problem checks:", list(problems.keys()), flush=True)
for name, diag in problems.items():
    print(f"\n--- {name}: {len(diag)} issue(s) ---", flush=True)
    try:
        print(diag.head(40).to_string(), flush=True)
    except Exception as e:
        print("render failed:", e, flush=True)
try:
    _m, diag = troubleshooting.overspecification(mm, repair="?", verbose=3)
    if diag is not None:
        print("\n=== top overspecification directions ===", flush=True)
        for prob in diag.index.get_level_values(0).unique()[:8]:
            sub = diag.xs(prob, level=0)
            ev = float(sub.index.get_level_values(0)[0])
            s = sub.droplevel(0)
            s = s.reindex(s.abs().sort_values(ascending=False).index)
            print(f"dir {prob}: eigenvalue={ev:.3e}", flush=True)
            for pname, val in s.head(8).items():
                print(f"    {str(pname):34s} {float(val):+.4f}", flush=True)
except Exception as e:
    print("overspec print failed:", type(e).__name__, e, flush=True)
try:
    skims.close()
except Exception:
    pass
