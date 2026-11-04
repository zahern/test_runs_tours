"""Standalone sanity test for stage5b_ca_duckdb (no larch needed)."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage5b_ca_duckdb as ca


def main() -> int:
    print("duckdb available:", ca.available())
    rng = np.random.default_rng(0)
    n_zones = 30
    zone_ids = np.arange(100, 100 + n_zones)
    n_obs = 400

    stop = pd.DataFrame({
        "origin_taz": rng.choice(zone_ids, n_obs),
        "dest_taz": rng.choice(zone_ids, n_obs),
        "anchor_down_taz": rng.choice(zone_ids, n_obs),
        "is_outbound": rng.integers(0, 2, n_obs),
        "depart_time_min": rng.integers(300, 1200, n_obs).astype(float),
        "tour_mode": rng.choice([1, 2, 3, 4, 5], n_obs),
        "income_hi": rng.random(n_obs),
        "is_female": rng.integers(0, 2, n_obs).astype(float),
    })
    zones = pd.DataFrame({"zone_id": zone_ids})
    taz_index = {int(z): i for i, z in enumerate(zone_ids)}
    mat = rng.random((n_zones, n_zones)) * 20.0
    skims = {"NetworkDist": mat, "AutoIVT_AM": mat * 2,
             "TransitIVT_AM": mat * 3, "AutoCrashRisk": rng.random((n_zones, n_zones))}
    landuse = pd.DataFrame({
        "zone_id": zone_ids,
        "TotalEmployment": rng.random(n_zones) * 100,
        "RetailEmp": rng.random(n_zones) * 50,
        "TOTPOP": rng.random(n_zones) * 1000,
        "TOTHH": rng.random(n_zones) * 400,
    })

    with tempfile.TemporaryDirectory() as tmp:
        for sample_size in (5, 0):
            info = ca.build_or_get(tmp, "work", stop, zones, skims, landuse,
                                   taz_index, "zone_id", sample_size, 42)
            df, n_obs_out = ca.load_ca(
                info["path"],
                ["detour_dist_km", "log_size", "dist_to_anchor",
                 "same_as_anchor", "same_as_prev", "is_inbound",
                 "tour_mode_auto", "tour_mode_transit", "tour_mode_active",
                 "AutoIVT_AM", "AutoCrashRisk", "income_hi"],
                has_correction=info["sampled"])
            assert df is not None and len(df) > 0
            assert df["chosen"].sum() == n_obs_out, (
                f"chosen rows {int(df['chosen'].sum())} != cases {n_obs_out}")
            assert df["case_id"].nunique() == n_obs_out
            per_case = df.groupby("case_id").size()
            if info["sampled"]:
                assert 1 <= per_case.min() <= per_case.max() <= sample_size, per_case.describe()
                assert "log_correction" in df.columns
            else:
                assert 1 <= per_case.min() <= per_case.max() <= n_zones
            assert np.isfinite(df.select_dtypes("number").to_numpy()).all()
            print(f"sample_size={sample_size}: rows={len(df):,} cases={n_obs_out} "
                  f"alts/case={per_case.iloc[0]} sampled={info['sampled']} "
                  f"detour[min,max]=({df['detour_dist_km'].min():.2f},"
                  f"{df['detour_dist_km'].max():.2f})")
        print("CA_DUCKDB_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
