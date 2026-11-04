#!/usr/bin/env python3
"""70_score_plans.py - score every candidate plan on the SAME observed surfaces.

Each arm's own summary measures coverage of its own input surface, which is not
comparable across arms (predicted vs observed; event vs control). This script
takes the schedules produced by the event and non-event MIPs and scores each
one on the event observed surface and the matched-control surface with one
definition:

  cov_weighted = sum(obs * min(1, 100m / length)) over staffed cells
                 / same sum over ALL cells (network)
  cov_raw      = sum(obs) over staffed cells / sum(obs) over all cells

Also reports pairwise site-set Jaccard between arms (per fixture, then mean).
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
    "event_pred": RESULTS,
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
    f_map = {int(o): (float(min(1.0, OBS_M / L)) if L and L > 0 else 1.0)
             for o, L in lengths.items()}

    scheds = {}
    sets = {}
    for arm, path in PLANS.items():
        s = pd.read_csv(path / "game_day_mip_schedule.csv",
                        usecols=["fixture", "rel_bin", "road_osmid"])
        s["fixture"] = s["fixture"].map(base_label)
        s = s.dropna(subset=["road_osmid"]).copy()
        s["road_osmid"] = s["road_osmid"].astype(np.int64)
        s["f"] = s["road_osmid"].map(f_map).fillna(1.0)
        scheds[arm] = s.drop_duplicates(["fixture", "rel_bin", "road_osmid"])
        sets[arm] = {fx: set(g["road_osmid"].unique())
                     for fx, g in s.groupby("fixture")}

    rows = []
    for surf_name, path in SURFACES.items():
        surf = pd.read_parquet(path, columns=[
            "fixture", "road_osmid", "rel_bin", "pred_speeding"])
        surf["fixture"] = surf["fixture"].map(base_label)
        surf["road_osmid"] = surf["road_osmid"].astype(np.int64)
        surf["f"] = surf["road_osmid"].map(f_map).fillna(1.0)
        surf["w"] = surf["pred_speeding"] * surf["f"]
        for fx, g in surf.groupby("fixture"):
            tot_w = float(g["w"].sum())
            tot_r = float(g["pred_speeding"].sum())
            for arm, sch in scheds.items():
                sel = sch[sch["fixture"] == fx]
                if sel.empty:
                    continue
                m = g.merge(sel[["rel_bin", "road_osmid"]],
                            on=["rel_bin", "road_osmid"], how="inner")
                rows.append({
                    "surface": surf_name, "plan": arm, "fixture": fx,
                    "cov_weighted": float(m["w"].sum()) / max(tot_w, 1e-12),
                    "cov_raw": float(m["pred_speeding"].sum()) / max(tot_r, 1e-12),
                    "n_cells": int(len(m)),
                })
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "plan_scored_on_common_surfaces.csv", index=False)

    pooled = (per.groupby(["surface", "plan"])[
        ["cov_weighted", "cov_raw", "n_cells"]].mean().reset_index())
    print("=== mean over fixtures (each plan scored on each surface) ===")
    print(pooled.pivot(index="plan", columns="surface",
                       values="cov_weighted").round(3).to_string())
    print()
    print(pooled.pivot(index="plan", columns="surface",
                       values="cov_raw").round(3).to_string())

    arms = list(PLANS)
    jac = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            per_fx = []
            for fx, sa in sets[a].items():
                sb = sets.get(b, {}).get(fx)
                if sb:
                    per_fx.append(len(sa & sb) / len(sa | sb))
            jac[f"{a}|{b}"] = float(np.mean(per_fx)) if per_fx else float("nan")
    print("\n=== pairwise site-set Jaccard (mean over fixtures) ===")
    for k, v in sorted(jac.items(), key=lambda kv: -kv[1]):
        print(f"   {k:28s} {v:.3f}")

    out = {"pooled": pooled.to_dict(orient="records"), "jaccard": jac,
           "note": ("cov_weighted = observed*min(1,100m/length) covered / "
                    "all cells; cov_raw = observed covered / all observed")}
    with open(OUT / "plan_common_surface_score.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT / 'plan_scored_on_common_surfaces.csv'}")


if __name__ == "__main__":
    main()
