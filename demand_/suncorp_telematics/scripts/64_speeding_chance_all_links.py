"""
64_speeding_chance_all_links.py
========================================================================
Gap-fill per-bin speeding chance onto EVERY road link so maps have no
holes ("visible on every segment").

Source (script 59, PBS 59_game_day_search_predict.pbs):
    results/game_day_pred_15min.parquet
    fixture x segment x 15-min bin, 6 fixtures x 7,558 segments x 28 bins.
    pred_propensity = per-bin speeding chance (no nulls, ~0.004..0.017).
    267 of 7,825 road_meta links have no prediction (zero exposure in the
    15-min panel) and are simply absent from that file.

This script left-joins the predictions onto the full link set from
data/road_meta.csv (7,825 OSM links), filling the 267 missing links
with 0 (no observed trips -> no measurable speeding chance), and writes:

  results/game_day_speeding_chance_all_links.parquet
      fixture x ALL 7,825 links x 28 bins (6 x 7825 x 28 rows),
      pred_propensity (+ pred_speeding, pred_flow, exp) with osmid key
      for a direct join to data/osm_roads_suncorp.geojson in QGIS
      (join: osmid <-> osmid).
  results/speeding_chance_per_segment.csv
      one row per link (7,825): link attributes + mean/max propensity
      across bins+fixtures, total/mean predicted speeding, covered flag.
      Instant choropleth without touching the 1.3M-row bin file.

Usage:
  python scripts/64_speeding_chance_all_links.py
"""
import os

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")

PRED = os.path.join(RES, "game_day_pred_15min.parquet")
META = os.path.join(DATA, "road_meta.csv")
OUT_BIN = os.path.join(RES, "game_day_speeding_chance_all_links.parquet")
OUT_SEG = os.path.join(RES, "speeding_chance_per_segment.csv")


def main():
    print("[64] loading predictions + link set ...", flush=True)
    g = pd.read_parquet(PRED)
    meta = pd.read_csv(META)
    links = meta["osmid"].unique()
    print(f"[64] pred rows: {len(g):,} | pred links: "
          f"{g['road_osmid'].nunique():,} | all links: {len(links):,}",
          flush=True)
    missing = set(links) - set(g["road_osmid"])
    print(f"[64] links absent from predictions (filled 0): {len(missing):,}",
          flush=True)

    # Per fixture, cross its 28 bin labels with the full link set, then
    # left-join predictions. (Bin timestamps differ per fixture.)
    filled = []
    for fixture, gf in g.groupby("fixture", sort=False):
        bins = gf[["bin", "rel_bin"]].drop_duplicates()
        grid = (pd.DataFrame({"road_osmid": links})
                .merge(bins, how="cross"))
        grid["fixture"] = fixture
        grid["code"] = gf["code"].iloc[0]
        m = gf.drop(columns=["fixture", "code"])
        grid = grid.merge(m, on=["road_osmid", "bin", "rel_bin"],
                          how="left")
        filled.append(grid)
    full = pd.concat(filled, ignore_index=True)
    for c in ["pred_flow", "flow_uplift", "pred_speeding_count",
              "pred_speeding", "pred_speeding_network", "exp",
              "pred_denom", "pred_propensity", "pred_speeding_plugin"]:
        if c in full.columns:
            full[c] = full[c].fillna(0.0)
    full["covered"] = (full["exp"] > 0).astype(int)
    print(f"[64] bin file rows: {len(full):,} "
          f"(expect {g['fixture'].nunique()}x{len(links)}x28 = "
          f"{g['fixture'].nunique()*len(links)*28:,})", flush=True)
    assert len(full) == g["fixture"].nunique() * len(links) * 28
    assert int(full["pred_propensity"].isna().sum()) == 0
    full.to_parquet(OUT_BIN, index=False)
    print(f"[64] wrote {OUT_BIN}", flush=True)

    seg = (full.groupby("road_osmid")
           .agg(mean_propensity=("pred_propensity", "mean"),
                max_propensity=("pred_propensity", "max"),
                mean_pred_speeding=("pred_speeding", "mean"),
                total_pred_speeding=("pred_speeding", "sum"),
                covered=("covered", "max")))
    seg = seg.reset_index().merge(
        meta.rename(columns={"osmid": "road_osmid"}),
        on="road_osmid", how="left")
    assert len(seg) == len(links)
    seg.to_csv(OUT_SEG, index=False)
    print(f"[64] wrote {OUT_SEG} ({len(seg):,} links, "
          f"{int(seg['covered'].sum()):,} covered)", flush=True)
    print("[64] top-5 mean propensity:",
          flush=True)
    print(seg.nlargest(5, "mean_propensity")
          [["road_osmid", "road_name", "mean_propensity",
            "max_propensity"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
