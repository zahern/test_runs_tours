"""HPC verification for the E1 STAGE5_POWER_ACC flag.

Run with STAGE5_POWER_ACC=1 (forwarded by qsub -v). Exits 0 with
'E1 FLAG VERIFIED' only if every wiring point is live; raises otherwise.
"""
import os
import sys

import numpy as np
import pandas as pd

TAG = os.environ.get("RUN_TAG", "powerflag_verify")
print(f"[e1verify] STAGE5_POWER_ACC={os.environ.get('STAGE5_POWER_ACC')!r} run={TAG}",
      flush=True)
assert os.environ.get("STAGE5_POWER_ACC", "0").strip() == "1", \
    "E1 flag not set in job environment — qsub -v STAGE5_POWER_ACC=1 required"

import stage5_destination_choice as s5

# 1. UEC terms live
for t in ("c_power_reliability", "c_power_headroom", "c_power_cost", "c_ev_hosting"):
    assert t in s5.DEST_UEC_FLAT, f"missing DEST term {t}"
for t in ("c_power_rel_da", "c_power_rel_sr", "c_power_rel_transit",
          "c_power_ev_da", "c_power_ev_transit"):
    assert t in s5.MODE_UEC, f"missing MODE term {t}"
print("[e1verify] 4 dest + 5 mode UEC terms live", flush=True)

# 2. landuse merge on a small ORDER frame
lu = pd.DataFrame({"TOTEMP": [100.0, 200.0]},
                  index=pd.Index([1, 2], name="ORDER"))
out = s5._merge_power_accessibility(lu)
for c in s5._POWER_ACC_COLS:
    assert c in out.columns, f"merge missed {c}"
    assert np.isfinite(out[c].values).all(), f"NaN in {c}"
print("[e1verify] landuse merge OK:",
      {c: round(float(out[c].iloc[0]), 3) for c in s5._POWER_ACC_COLS}, flush=True)

# 3. GA pools offer the genes and they map to live terms
import ga_hypothesis_testing as g
assert {"c_power_reliability", "c_power_headroom",
        "c_power_cost", "c_ev_hosting"} <= set(g._stage5_dest_candidates())
assert {"power_reliability", "ev_hosting"} <= set(g._stage5_mode_candidates())
assert {"power_reliability", "ev_hosting"} <= set(
    g._stage5_mode_candidates_for_segment("work"))
assert {"c_power_reliability", "c_power_headroom",
        "c_power_cost", "c_ev_hosting"} <= set(
    g._stage5_dest_candidates_for_segment("work"))
print("[e1verify] GA pools OK (global + per-segment, dest + mode)", flush=True)

# 4. larch stack present (estimation itself runs on larch)
import larch
print(f"[e1verify] larch {larch.__version__} importable", flush=True)

print("E1 FLAG VERIFIED")
