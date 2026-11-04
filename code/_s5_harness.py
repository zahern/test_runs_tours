import os, sys, warnings, time, traceback, json
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
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import pipeline_logger as pl
import openmatrix as omx
from stage5_destination_choice import (
    _skim_flat, _skim_lookup, _run_segment, _prep_hhts_tours,
)

t0 = time.time()
_, out, landuse = pl.load_data()
print(f"\n[harness] load_data {time.time()-t0:.0f}s | tours={len(out['tours']):,} "
      f"persons={len(out['persons']):,} landuse={len(landuse):,}", flush=True)

# The pipeline preps tours BEFORE _run_segment (pipeline_logger.py:2544):
# maps tour_origin->OTAZ, tour_destination->DTAZ, primary_purpose->main_purpose,
# and tour_mode->integer codes. Without this, _run_segment's OD skim merge dies
# with KeyError: 'DTAZ'.
hhts_tours = _prep_hhts_tours(out["tours"], out["persons"])
print(f"[harness] prep_hhts_tours: {len(hhts_tours):,} tours | "
      f"OTAZ={'OTAZ' in hhts_tours.columns} DTAZ={'DTAZ' in hhts_tours.columns} "
      f"main_purpose={'main_purpose' in hhts_tours.columns} "
      f"tour_mode={'tour_mode' in hhts_tours.columns}", flush=True)

skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = _skim_flat(skims)
_skim_lookup(skim_df)

SEG = os.environ.get("SEG", "work")
print(f"[harness] === stage 5 segment '{SEG}' ===", flush=True)
t1 = time.time()
try:
    res = _run_segment(
        SEG, hhts_tours, out["persons"], landuse,
        skims, skim_df=skim_df, seed=42, verbose=True,
        crash_skims=skims,          # CRASH comes from skims_371.omx
    )
    dt = time.time() - t1
    print(f"\n[harness] segment '{SEG}' finished in {dt:.0f}s", flush=True)
    print(f"[harness] result keys: {sorted(res.keys())}", flush=True)
    for k in ("mode_ll", "dest_ll", "mode_model", "dest_model",
              "mode_coef", "dest_coef", "logsums"):
        v = res.get(k)
        if v is None:
            print(f"  {k}: None", flush=True)
        elif isinstance(v, dict):
            print(f"  {k}: dict({len(v)}) {list(v.items())[:4]}", flush=True)
        elif isinstance(v, float):
            print(f"  {k}: {v}", flush=True)
        else:
            print(f"  {k}: <{type(v).__name__}>", flush=True)
    with open(f"runs/_s5_{SEG}_summary.json", "w", encoding="utf-8") as fh:
        json.dump({k: (v if isinstance(v, (int, float, str, list, type(None)))
                       else f"<{type(v).__name__}>")
                   for k, v in res.items()}, fh, indent=2, default=str)
except Exception:
    print(f"\n[harness] segment '{SEG}' FAILED after {time.time()-t1:.0f}s", flush=True)
    traceback.print_exc()
finally:
    try: skims.close()
    except Exception: pass
print(f"\n[harness] total {time.time()-t0:.0f}s", flush=True)
