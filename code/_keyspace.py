import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NEW_POP", "1")
os.environ["NEW_POP_DIR"] = r"Z:\test_runs_tours\new_pop"
os.environ["NEW_POP_MAPPED"] = "0"
os.environ["PIPELINE_SKIMS_PATH"] = r"Z:\test_runs_tours\demand_\skims_371.omx"
os.environ["PIPELINE_SEED"] = "42"
os.environ["PIPELINE_ACCELERATOR"] = "cpu"
os.environ["LARCH_FORCE_JAX"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import pipeline_logger as pl
import openmatrix as omx
from stage5_destination_choice import (
    _skim_flat, _skim_lookup, _prep_hhts_tours, _filter_seg,
    _mode_tree, _build_od_dataset,
)

_, out, landuse = pl.load_data()
hhts = _prep_hhts_tours(out["tours"], out["persons"])
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = _skim_flat(skims)
_skim_lookup(skim_df)

def show(tag, ds):
    try:
        co = dict(ds.coords)
        print(f"\n=== {tag} ===", flush=True)
        print(f"  dims   : {dict(ds.sizes)}", flush=True)
        for name in ("otaz", "dtaz"):
            c = ds.coords.get(name, None)
            if c is None:
                print(f"  coords[{name}]: ABSENT", flush=True)
                continue
            v = np.asarray(c.values).ravel()
            print(f"  coords[{name}]: n={v.size} dtype={v.dtype} "
                  f"min={v.min()} max={v.max()} sample={v[:3].tolist()}", flush=True)
        print(f"  all coords: {list(co.keys())[:12]}", flush=True)
    except Exception as exc:
        print(f"  {tag} inspect failed: {type(exc).__name__}: {exc}", flush=True)

# ── A. the OD dataset _dest_tree uses for its alternative axis ────────────────
try:
    od_dest = _build_od_dataset(skims, skim_df)
    show("A. _build_od_dataset(skims, skim_df)  <- what _dest_tree uses", od_dest)
except Exception as e:
    print(f"A failed: {type(e).__name__}: {e}", flush=True)

# ── B. the mode DataTree's od node <- what _logsums reads ────────────────────
SEG = os.environ.get("SEG", "work")
seg = _filter_seg(hhts, SEG)
print(f"\n[diag] segment={SEG} tours={len(seg):,}", flush=True)
try:
    land_seg = landuse   # _mode_tree takes landuse directly
    dt_m = _mode_tree(seg, out["persons"], land_seg, skims, skim_df=skim_df)
    print(f"[diag] _mode_tree built: {type(dt_m).__name__}", flush=True)
    try:
        od_mode = dt_m['od']
        show("B. dt_mode['od']  <- what _logsums reads", od_mode)
    except Exception as e:
        print(f"  dt_mode['od'] unavailable: {type(e).__name__}: {e}", flush=True)
        try:
            print(f"  available nodes: {list(dt_m.keys())[:15]}", flush=True)
        except Exception:
            pass
except Exception as e:
    import traceback
    print(f"B failed: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()

# ── C. keyspace comparison ───────────────────────────────────────────────────
print("\n=== C. keyspace verdict ===", flush=True)
try:
    a = set(int(z) for z in np.asarray(od_dest.coords['otaz'].values).ravel() if int(z) > 0)
    b = set(int(z) for z in np.asarray(dt_m['od'].coords.get('dtaz').values).ravel() if int(z) > 0)
    print(f"  _dest_tree universe : n={len(a)} sample={sorted(a)[:3]}", flush=True)
    print(f"  logsum dtaz universe: n={len(b)} sample={sorted(b)[:3]}", flush=True)
    print(f"  intersection        : n={len(a & b)}", flush=True)
    print(f"  missing from logsum : n={len(a - b)}  <-- alternatives with NO logsum", flush=True)
    print(f"  SAME KEYSPACE       : {len(a & b) > 0}", flush=True)
except Exception as e:
    print(f"  verdict failed: {type(e).__name__}: {e}", flush=True)

try: skims.close()
except Exception: pass
