#!/usr/bin/env python3
"""70_stationary_topP.py - exact best stationary plan on each surface.

A stationary plan with P officers covers P distinct sites for all 28 bins,
so the optimum is simply the P sites with the largest total weight. Reports
that optimum under two metrics:
  o      : predicted finable speeding x obs_frac  (the MIP objective)
  o_f    : ... x min(1, 100m / length)            (the 70_score weighted metric)
and compares it with the selected plans' coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "control_mip_gt10"
OBS_M = 100.0

SURFACES = {
    "event_obs": RESULTS / "obs_event_surface_gt10.parquet",
    "ctrl_obs": RESULTS / "obs_control_surface_gt10.parquet",
    "ctrl_d1": RESULTS / "obs_control_surface_d1_gt10.parquet",
}


def base_label(label: str) -> str:
    for pref in ("CTRL1: ", "CTRL: "):
        if str(label).startswith(pref):
            return str(label)[len(pref):]
    return str(label)


def main() -> None:
    geo = pd.read_csv(ROOT / "data" / "road_geometry_features.csv")
    lengths = dict(zip(geo["road_osmid"].astype(np.int64),
                       geo["length_m"].astype(float)))
    f_map = {int(o): (min(1.0, OBS_M / L) if L and L > 0 else 1.0)
             for o, L in lengths.items()}

    rows = []
    for name, path in SURFACES.items():
        s = pd.read_parquet(path, columns=[
            "fixture", "road_osmid", "rel_bin", "pred_speeding"])
        s["fixture"] = s["fixture"].map(base_label)
        s["road_osmid"] = s["road_osmid"].astype(np.int64)
        s["f"] = s["road_osmid"].map(f_map).fillna(1.0)
        for fx, g in s.groupby("fixture"):
            tot = g["pred_speeding"].sum()
            tot_f = (g["pred_speeding"] * g["f"]).sum()
            site_mass = g.groupby("road_osmid").agg(
                m=("pred_speeding", "sum")).sort_values("m", ascending=False)
            site_mass_f = (g.assign(w=g["pred_speeding"] * g["f"])
                           .groupby("road_osmid")["w"].sum()
                           .sort_values(ascending=False))
            for P in range(3, 9):
                rows.append({
                    "surface": name, "fixture": fx, "P": P,
                    "topP_o": float(site_mass["m"].head(P).sum()
                                    / max(tot, 1e-12)),
                    "topP_o_f": float(site_mass_f.head(P).sum()
                                      / max(tot_f, 1e-12)),
                })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "stationary_topP.csv", index=False)
    print("=== best stationary plan (top-P sites, all bins) ===")
    for surf in SURFACES:
        print(f"\nsurface={surf}")
        sub = per[per["surface"] == surf]
        print(sub.groupby("P")[["topP_o", "topP_o_f"]].mean().round(4)
              .to_string())
    print("\nper-fixture top-5:")
    print(per[(per["P"] == 5)][["surface", "fixture", "topP_o", "topP_o_f"]]
          .to_string(index=False))
    with open(OUT / "stationary_topP.json", "w") as fh:
        json.dump(per.to_dict(orient="records"), fh, indent=2)


if __name__ == "__main__":
    main()
