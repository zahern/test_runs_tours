#!/usr/bin/env python3
"""70_move_value.py - value of the relocations the observed-data MIPs choose.

The mobility A/B in 60 compares fractional vs binary capacity, NOT move vs
no-move. This script takes each selected plan and builds the stationary
counterfactual directly: every officer stays at the site of their first leg
for all 28 bins. Coverage is then computed on the same surface for both.

Reports, per arm and fixture:
  cov_full      weighted coverage of the selected plan
  cov_stationary weighted coverage with all second legs removed
  delta_pp      (full - stationary) * 100
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
PLANS = {
    "event_obs": OUT / "event_obs",
    "ctrl_d1": OUT / "ctrl_d1",
    "ctrl_obs": OUT / "ctrl_obs",
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

    surf_cache = {}
    for name, path in SURFACES.items():
        s = pd.read_parquet(path, columns=[
            "fixture", "road_osmid", "rel_bin", "pred_speeding"])
        s["fixture"] = s["fixture"].map(base_label)
        s["road_osmid"] = s["road_osmid"].astype(np.int64)
        s["f"] = s["road_osmid"].map(f_map).fillna(1.0)
        s["w"] = s["pred_speeding"] * s["f"]
        surf_cache[name] = s

    rows = []
    for arm, path in PLANS.items():
        sch = pd.read_csv(path / "game_day_mip_schedule.csv")
        sch["fixture"] = sch["fixture"].map(base_label)
        for fx, g in sch.groupby("fixture"):
            start_sites, full, move_bins = {}, set(), []
            for off, h in g.groupby("officer"):
                h = h.sort_values("rel_bin")
                stationed = h[h["state"] == "stationed"]
                moving = h[h["state"] == "moving"]
                if stationed.empty:
                    continue
                first = stationed.iloc[0]
                start_sites[off] = int(first["road_osmid"])
                if len(moving):
                    move_bins.append(int(moving["rel_bin"].min()))
                for r in stationed.itertuples():
                    full.add((int(r.rel_bin), int(r.road_osmid)))
            stat = {(t, o) for t in range(28) for o in start_sites.values()}
            for surf_name, surf in surf_cache.items():
                s = surf[surf["fixture"] == fx]
                tot = float(s["w"].sum())
                full_w = float(s[s.apply(
                    lambda r: (int(r.rel_bin), int(r.road_osmid)) in full,
                    axis=1)]["w"].sum())
                stat_w = float(s[s.apply(
                    lambda r: (int(r.rel_bin), int(r.road_osmid)) in stat,
                    axis=1)]["w"].sum())
                rows.append({
                    "arm": arm, "fixture": fx, "surface": surf_name,
                    "n_officers": len(start_sites),
                    "moves": len(move_bins),
                    "cov_full": full_w / max(tot, 1e-12),
                    "cov_stationary": stat_w / max(tot, 1e-12),
                    "delta_pp": 100.0 * (full_w - stat_w) / max(tot, 1e-12),
                })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "relocation_value.csv", index=False)
    print("=== relocation value: selected plan vs its stationary "
          "counterfactual ===")
    for surf_name in SURFACES:
        sub = per[per["surface"] == surf_name]
        tab = (sub.groupby("arm")[["cov_full", "cov_stationary", "delta_pp",
                                   "moves"]].mean())
        print(f"\nsurface={surf_name}")
        print(tab.round(4).to_string())
    with open(OUT / "relocation_value.json", "w") as fh:
        json.dump(per.to_dict(orient="records"), fh, indent=2)
    print(f"\nwrote {OUT / 'relocation_value.csv'}")


if __name__ == "__main__":
    main()
