#!/usr/bin/env python3
"""Test if RP heterogeneity can run."""
import sys
sys.path.insert(0, "Z:\test_runs_tours\demand_\suncorp_telematics\scripts")
import numpy as np
import pandas as pd
import os

SEGMENT_ROADS = r"Z:\test_runs_tours\demand_\suncorp_telematics\data\segment_roads_part0_5000.csv.gz"
FEATURES = r"Z:\test_runs_tours\demand_\suncorp_telematics\data\road_geometry_features.csv"

df = pd.read_csv(SEGMENT_ROADS)
df = df[df["trips"] > 0].copy()
df["Y"] = df["speeding"].astype(float)

df["highway"] = df.get("highway", "unknown").fillna("unknown").astype(str)
for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
for c in ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
          "dist_to_intersection_m", "node_degree", "length_m"]:
    if c in df.columns:
        sd = df[c].std()
        df[c + "_z"] = (df[c] - df[c].mean()) / (sd if sd and sd > 0 else 1.0)
df["highway"] = df.get("highway", "unknown").fillna("unknown").astype(str)
df["ref_dt"] = pd.to_datetime(df["ref_dt"])
df["kickoff_h"] = df["ref_dt"].dt.hour + df["ref_dt"].dt.minute / 60.0

from metacount_nb import prepare_metacount_frame, fit_metacount_nb
mc, _ = prepare_metacount_frame(df[df["type"] == "Event"], exposure_col="trips", highway_col="highway")
fixed, rdm = _mc_terms_for(mc)

try:
    info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                            fixed_terms=fixed, rdm_terms=rdm,
                            model="nb", R=100, de_maxiter=5, de_popsize=5, seed=0, verbose=True)
    print(f"SUCCESS: RP fit completed")
    print(f"AIC: {info['aic']:.0f}, alpha: {info['alpha']:.3f}")
    print(f"Jensen: {info['jensen_factor']:.2f}x")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {str(e)[:300]}")