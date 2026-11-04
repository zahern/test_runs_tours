import os, sys, warnings, time, traceback
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
os.environ["STAGE5_DEST_SAMPLE_SIZE"] = os.environ.get("STAGE5_DEST_SAMPLE_SIZE", "20")

CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import pandas as pd
import pipeline_logger as pl
import openmatrix as omx
import larch as lx

import stage5_destination_choice as S5
from stage5_destination_choice import (
    _skim_flat, _skim_lookup, _prep_hhts_tours, _filter_seg,
    _dest_tree, _build_dest_model, _larch_data_array_guard,
)

_, out, landuse = pl.load_data()
hhts = _prep_hhts_tours(out["tours"], out["persons"])
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = _skim_flat(skims)
_skim_lookup(skim_df)

SEG = os.environ.get("SEG", "work")
seg = _filter_seg(hhts, SEG)
print(f"\n[LARCH-DEST] segment={SEG}  tours={len(seg):,}", flush=True)

# ── 1. sampled alternatives ──────────────────────────────────────────────────
# Pass None first: _dest_tree then uses ALL zones (log_correction = 0), which
# isolates the question "does the Larch destination formulation work at all?".
# The sampled-alternative variant is exercised via the pipeline's own
# run_stage5 path, which supplies the Manski-Lerman design properly.
sampled = None
print("[LARCH-DEST] sampled_alts=None -> using ALL 371 zones "
      "(no sampling correction needed)", flush=True)

# ── 2. build the dest DataTree / Dataset ─────────────────────────────────────
t0 = time.time()
dt = _dest_tree(seg, out["persons"], landuse, skims, logsums=None,
                sampled_alts=sampled, skim_df=skim_df,
                crash_skims=skims, segment=SEG)
print(f"[LARCH-DEST] _dest_tree OK in {time.time()-t0:.0f}s  "
      f"type={type(dt).__name__}", flush=True)
try:
    print(f"[LARCH-DEST] attrs: flat_mode={getattr(dt,'attrs',{}).get('_flat_mode')} "
          f"ortho={'yes' if getattr(dt,'attrs',{}).get('_ortho_betas') else 'no'}", flush=True)
except Exception:
    pass

# ── 3. build the model and REACH THE LIKELIHOOD (the original failure) ──────
t1 = time.time()
mm = _build_dest_model(dt, SEG)
print(f"[LARCH-DEST] _build_dest_model OK in {time.time()-t1:.0f}s", flush=True)

ok = _larch_data_array_guard(mm, dt, "jax")
print(f"[LARCH-DEST] data-array guard -> {ok}  (True == arrays healthy)", flush=True)

try:
    ll0 = float(mm.loglike())
    print(f"[LARCH-DEST] INITIAL loglike = {ll0:.4f}   <-- was the nan/.wt failure",
          flush=True)
    n_free = int((~mm.pf['holdfast'].fillna(False)).sum()) if 'holdfast' in mm.pf else -1
    print(f"[LARCH-DEST] n_params={len(mm.pf)}  n_free={n_free}", flush=True)
    print("[LARCH-DEST] 'log_correction' in pf:",
          'log_correction' in mm.pf.index, flush=True)
    if 'log_correction' in mm.pf.index:
        print(f"[LARCH-DEST]   locked value = {mm.pf.loc['log_correction','value']} "
              f"(1.0 == Manski-Lerman held at 1)", flush=True)
except Exception as e:
    print(f"[LARCH-DEST] loglike FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
finally:
    try: skims.close()
    except Exception: pass
