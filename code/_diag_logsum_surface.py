"""_diag_logsum_surface.py -- does the CURRENT code produce a mode logsum that
varies across destination zones?

The Oct-2 work cache carries a logsum surface that is constant across all 372
zones for every one of its 6720 cases (per-row sd exactly 0), which makes the
destination c_logsum a per-tour constant and therefore unidentified.  That
cache was built before the 2026-10-08/10-13 logsum changes (Larch per-zone
DTAZ substitution returning byte-identical columns, then default OFF).

This script evaluates _logsums() with the code on disk today, on a small
sample of tours from an already-fitted segment, and reports the across-zone
variation of the resulting surface.  It fits nothing.
"""
import os
import numpy as np

SEG = os.environ.get("SEG", "nonmand")
N = int(os.environ.get("N_TOURS", "200"))

import openmatrix as omx  # noqa: E402
import pipeline_logger as pl  # noqa: E402
import stage5_destination_choice as S5  # noqa: E402

print(f"[surface] segment={SEG} sample={N}", flush=True)
_, out, landuse = pl.load_data()
tours = S5._prep_hhts_tours(out["tours"], out["persons"])
_PURP = next(c for c in ("main_purpose", "primary_purpose", "tour_purpose")
             if c in tours.columns)
print(f"[surface] purpose column={_PURP!r}: "
      f"{tours[_PURP].value_counts().head(8).to_dict()}", flush=True)
seg_of = tours[_PURP].map(S5._purpose_to_seg)
seg = tours[seg_of == SEG].copy()
print(f"[surface] {SEG} tours available: {len(seg)}", flush=True)
if len(seg) > N:
    seg = seg.sample(N, random_state=42).reset_index(drop=True)
if "ptour_id" not in seg.columns:
    seg["ptour_id"] = np.arange(len(seg)).astype(str)

blob = S5.load_models(os.environ.get("PAPER_MODELS", "saved_data/stage5_models.pkl")) or {}
mdl = blob[SEG]["mode_model"]

skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
dt = S5._mode_tree(seg, out["persons"], landuse, skims, skim_df=skim_df)

for force in (True, False):
    tag = "numpy (forced)" if force else "default path"
    try:
        ls = S5._logsums(mdl, dt, skims=skims, force_numpy=force)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[surface] {tag}: FAILED {type(exc).__name__}: {exc}", flush=True)
        continue
    v = np.asarray(ls.values, dtype=float)
    rowsd = np.nanstd(v, axis=1)
    print(f"\n[surface] {tag}: shape={v.shape}")
    print(f"  across-zone sd per case: min={np.nanmin(rowsd):.6g} "
          f"median={np.nanmedian(rowsd):.6g} mean={np.nanmean(rowsd):.6g} "
          f"max={np.nanmax(rowsd):.6g}")
    flat = int((rowsd < 1e-12).sum())
    print(f"  cases with zero across-zone variation: {flat}/{rowsd.size}")
    print(f"  value range: {np.nanmin(v):.6g} .. {np.nanmax(v):.6g}")
    print(f"  first case, 8 zones: {np.round(v[0, :8], 6).tolist()}", flush=True)
    # The fringe/external composite zone carries near-infinite impedance and
    # can dominate an across-zone SD; report the real SA2 surface separately,
    # since that is the universe destination choice actually samples from.
    zids = np.asarray([int(z) for z in ls.coords["zones"].values])
    sa2 = (zids > 99999) | ((zids > 0) & (zids < 900))
    if sa2.sum() >= 10:
        vs = v[:, sa2]
        rsd = np.nanstd(vs, axis=1)
        cen = np.nanmedian(np.abs(vs - np.nanmedian(vs, axis=1, keepdims=True)),
                           axis=1)
        print(f"  SA2-only ({int(sa2.sum())} zones): across-zone sd "
              f"min={np.nanmin(rsd):.4g} median={np.nanmedian(rsd):.4g} "
              f"max={np.nanmax(rsd):.4g}")
        print(f"  SA2-only median |dev from case median|: "
              f"{np.nanmedian(cen):.4g}; value range "
              f"{np.nanmin(vs):.4g} .. {np.nanmax(vs):.4g}", flush=True)
        _lo = float(np.nanmin(vs[0]))
        _hi = float(np.nanmax(vs[0]))
        print(f"  first case SA2 spread: {_lo:.4g} .. {_hi:.4g}"
              f" (n={int(np.isfinite(vs[0]).sum())})", flush=True)

skims.close()
