"""
11_convert_skims_to_openmatrix.py
Convert sa2_skims_507.omx (written by demand_'s lightweight h5py-based
omx_utils.write_omx — see that module's docstring re: an openmatrix/tables
binary incompatibility on the Windows dev machine) into a fully spec-compliant
OMX file using the real `openmatrix` package, with a properly tagged zone
mapping (OMX_Lookup attribute).

Also merges in the model's ECF (Expected Crash Frequency) safety component —
crash_rate_total from crash_matrices_507.omx — as a matrix named 'CRASH', so
it lands in the same OMX file the ABM stages already expect a 'CRASH' matrix
to come from (e.g. stage5_destination_choice._mode_probs_numpy's
`_mat('CRASH', None)` lookup). Both files share the same 507-zone ordering
(both built from zone_ids_sa_only_507.txt, sorted identically), so no
remapping is needed.

Run this on whichever environment actually executes the ABM pipeline
(pipeline_logger.py) — it calls `openmatrix.open_file()` directly in several
places (bypassing read_skims.py's h5py fallback), so it needs a file the
`openmatrix` package itself can open, not just h5py.

Requirements: openmatrix, tables (PyTables), numpy, h5py
    pip install openmatrix tables

Usage:
    python 11_convert_skims_to_openmatrix.py
    # or specify paths:
    python 11_convert_skims_to_openmatrix.py --in sa2_skims_507.omx --crash crash_matrices_507.omx --out sa2_skims_507_omxspec.omx
"""
import argparse
import os
import h5py
import numpy as np


def _read_h5py_omx(path):
    with h5py.File(path, "r") as f:
        matrix_names = list(f["data"].keys())
        lookup_names = list(f["lookup"].keys()) if "lookup" in f else []
        if not lookup_names:
            raise ValueError(f"No /lookup group found in {path} — cannot build zone mapping")
        lookup_name = lookup_names[0]
        raw = f["lookup"][lookup_name][...]
        zone_ids = [int(z.decode() if isinstance(z, bytes) else z) for z in raw]
        matrices = {name: np.array(f["data"][name], dtype=np.float32) for name in matrix_names}
    return matrices, zone_ids, lookup_name


def convert(in_path, crash_path, out_path):
    import openmatrix as omx

    matrices, zone_ids, lookup_name = _read_h5py_omx(in_path)
    print(f"Source matrices ({len(matrices)}): {list(matrices.keys())}")
    print(f"Zone mapping '{lookup_name}': {len(zone_ids)} zones, "
          f"IDs {min(zone_ids)}-{max(zone_ids)}")

    if crash_path and os.path.exists(crash_path):
        crash_matrices, crash_zone_ids, _ = _read_h5py_omx(crash_path)
        if "crash_rate_total" not in crash_matrices:
            raise ValueError(f"crash_rate_total not found in {crash_path}")
        if crash_zone_ids == zone_ids:
            matrices["CRASH"] = crash_matrices["crash_rate_total"]
            print(f"Merged crash_rate_total from {crash_path} as matrix 'CRASH' "
                  f"(the model's ECF safety component, {len(zone_ids)} zones)")
        else:
            # Zone scopes differ (e.g. crash=507 zones, skims=529 zones after
            # adding Tweed Heads). Build the CRASH matrix by mapping the crash
            # zone pairs into the larger skim zone index; Tweed Heads OD pairs
            # (not in the crash OD analysis) get ECF=0, which is correct.
            crash_idx = {z: i for i, z in enumerate(crash_zone_ids)}
            n = len(zone_ids)
            crash_mat = np.zeros((n, n), dtype=np.float32)
            src = crash_matrices["crash_rate_total"]
            for oi, oz in enumerate(zone_ids):
                ci = crash_idx.get(oz)
                if ci is None:
                    continue
                for di, dz in enumerate(zone_ids):
                    cj = crash_idx.get(dz)
                    if cj is not None:
                        crash_mat[oi, di] = src[ci, cj]
            n_nonzero = int((crash_mat > 0).sum())
            print(f"Merged crash_rate_total from {crash_path} as matrix 'CRASH' "
                  f"(crash scope={len(crash_zone_ids)}, skim scope={n}, "
                  f"non-zero cells={n_nonzero:,} — OD pairs outside crash scope get ECF=0)")
    else:
        print(f"WARNING: crash matrix file not found at {crash_path} — "
              f"output will have no 'CRASH' matrix, safety component will be 0 everywhere")

    if os.path.exists(out_path):
        os.remove(out_path)

    out = omx.open_file(out_path, "w")
    try:
        for name, arr in matrices.items():
            out[name] = arr
        out.create_mapping(lookup_name, zone_ids)
    finally:
        out.close()

    fsize = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\nWritten: {out_path}  ({fsize:.1f} MB, {len(matrices)} matrices, "
          f"{len(zone_ids)}x{len(zone_ids)} zones)")
    print("Verifying with openmatrix...")
    chk = omx.open_file(out_path, "r")
    print(f"  list_matrices(): {chk.list_matrices()}")
    print(f"  list_mappings(): {chk.list_mappings()}")
    chk.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="sa2_skims_507.omx")
    p.add_argument("--crash", dest="crash_path", default="crash_matrices_507.omx")
    p.add_argument("--out", dest="out_path", default="sa2_skims_507_omxspec.omx")
    args = p.parse_args()
    convert(args.in_path, args.crash_path, args.out_path)
