#!/usr/bin/env python3
"""69_transfer_eval.py - realised (observed) coverage and site-set transfer
for the game-day deployment MIP.

Evaluates any MIP output directory (game_day_mip_schedule.csv + sites.csv +
summary.json) against the observed holdout counts in
results/game_day_pred_vs_actual_cells.csv.gz, so coverage is no longer only a
share of the predicted surface:

  realised_obsfrac : sum over staffed (bin, segment) of
                     observed * obs_frac, over total observed mass
  realised_raw     : same without the observability fraction (whole-segment)
  oracle_top5      : best five observed segments, stationary (upper bound)
  pred_top5        : top five by predicted observable mass, scored on observed
  random5_*        : random five-site draws from the 250-candidate shortlist
  transfer_*       : another fixture's selected site set applied to this one
                     (leave-one-fixture-out plan transfer)

Writes <out>.json and <out>.csv.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL_BINS = 28


def kickoff_utc(fixture: str) -> pd.Timestamp:
    m = re.search(r"@\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})", str(fixture))
    if not m:
        raise ValueError(f"cannot parse kickoff from {fixture!r}")
    ts = pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}",
                      tz="Australia/Brisbane")
    return ts.tz_convert("UTC")


def rel_bins_for(fixtures: pd.Series, bins: pd.Series) -> pd.Series:
    out = pd.Series(-1, index=fixtures.index, dtype=int)
    for fx in fixtures.unique():
        ko = kickoff_utc(fx)
        start = ko - pd.Timedelta(hours=3)
        mask = fixtures == fx
        out.loc[mask] = np.rint(
            (bins.loc[mask] - start) / pd.Timedelta(minutes=15)).astype(int)
    return out


def frac_map_from_sites(sites: pd.DataFrame) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for fx in sites["fixture"].unique():
        out[fx] = {}
        sub = sites[sites["fixture"] == fx]
        for _, r in sub.iterrows():
            out[fx].update({int(o): float(f) for o, f in zip(
                str(r["segment_osmids"]).split("|"),
                str(r["obs_frac"]).split("|"))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default=str(ROOT / "results"
                    / "game_day_pred_vs_actual_cells.csv.gz"))
    ap.add_argument("--pred", default=str(ROOT / "results"
                    / "game_day_pred_15min.parquet"))
    ap.add_argument("--mip-dir", default=str(ROOT / "results"))
    ap.add_argument("--geometry", default=str(ROOT / "data"
                    / "road_geometry_features.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "transfer_eval.json"))
    ap.add_argument("--random-draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mip = Path(args.mip_dir)
    obs_raw = pd.read_csv(args.obs)
    obs_raw["bin"] = pd.to_datetime(obs_raw["bin"], utc=True)
    obs_raw["road_osmid"] = obs_raw["road_osmid"].astype(np.int64)
    obs_raw["rel_bin"] = rel_bins_for(obs_raw["fixture"], obs_raw["bin"])
    obs_raw = obs_raw[(obs_raw["rel_bin"] >= 0)
                      & (obs_raw["rel_bin"] < REL_BINS)].copy()
    obs_cell = (obs_raw.groupby(["fixture", "road_osmid", "rel_bin"],
                                as_index=False)["observed"].sum())
    obs_seg = obs_raw.groupby(["fixture", "road_osmid"],
                              as_index=False)["observed"].sum()
    obs_tot = obs_raw.groupby("fixture")["observed"].sum()

    pred = pd.read_parquet(args.pred, columns=[
        "fixture", "road_osmid", "rel_bin", "pred_speeding"])
    pred["road_osmid"] = pred["road_osmid"].astype(np.int64)
    pred_cell = pred.groupby(["fixture", "road_osmid", "rel_bin"],
                             as_index=False)["pred_speeding"].sum()
    pred_seg = pred.groupby(["fixture", "road_osmid"],
                            as_index=False)["pred_speeding"].sum()
    pred_tot = pred.groupby("fixture")["pred_speeding"].sum()

    geo = pd.read_csv(args.geometry)
    lengths = dict(zip(geo["road_osmid"].astype(np.int64),
                       geo["length_m"].astype(float)))

    summary_path = mip / "game_day_mip_summary.json"
    obs_m = 100.0
    per_game_cov: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path) as fh:
            _summary = json.load(fh)
        obs_m = float(_summary["summary"].get("obs_m", 100.0))
        for _r in _summary.get("per_game", []):
            per_game_cov[_r["fixture"]] = _r

    def frac(fx: str, osmid: int, site_maps) -> float:
        if fx in site_maps and osmid in site_maps[fx]:
            return site_maps[fx][osmid]
        L = lengths.get(int(osmid))
        return 1.0 if not L else float(min(1.0, obs_m / float(L)))

    sites = pd.read_csv(mip / "game_day_mip_sites.csv")
    sched = pd.read_csv(mip / "game_day_mip_schedule.csv")
    site_maps = frac_map_from_sites(sites)

    sched = sched.drop_duplicates(["fixture", "rel_bin", "road_osmid"]).copy()
    sched["f"] = [frac(fx, int(o), site_maps)
                  for fx, o in zip(sched["fixture"], sched["road_osmid"])]
    o_cov = sched.merge(obs_cell, on=["fixture", "road_osmid", "rel_bin"],
                        how="left").fillna({"observed": 0.0})
    o_cov["cov"] = o_cov["observed"] * o_cov["f"]
    o_cov_raw = sched.merge(obs_cell, on=["fixture", "road_osmid", "rel_bin"],
                            how="left").fillna({"observed": 0.0})
    p_cov = sched.merge(pred_cell, on=["fixture", "road_osmid", "rel_bin"],
                        how="left").fillna({"pred_speeding": 0.0})
    p_cov["cov"] = p_cov["pred_speeding"] * p_cov["f"]
    covered_obs = o_cov.groupby("fixture")["cov"].sum()
    covered_obs_raw = o_cov_raw.groupby("fixture")["observed"].sum()
    covered_pred = p_cov.groupby("fixture")["cov"].sum()

    rng = np.random.default_rng(args.seed)
    per: list[dict] = []
    fixtures = list(obs_tot.index)
    for fx in fixtures:
        fs = obs_seg[obs_seg["fixture"] == fx].set_index("road_osmid")["observed"]
        ps = pred_seg[pred_seg["fixture"] == fx].set_index("road_osmid")["pred_speeding"]
        total_o = float(obs_tot.get(fx, 0.0))
        total_p = float(pred_tot.get(fx, 0.0))
        fl = {int(o): frac(fx, int(o), site_maps) for o in ps.index}

        obs_rank = fs.sort_values(ascending=False)
        oracle5 = [int(o) for o in obs_rank.index[:5]]
        obs_rate = {int(o): float(v) for o, v in fs.items()}

        def cov_set(osmids, fr=None):
            fr = fr or fl
            return sum(fr.get(int(o), 0.0)
                       * obs_rate.get(int(o), 0.0) for o in osmids)

        oracle_cov = cov_set(oracle5)

        obs_mass = {int(o): float(ps.get(o, 0.0)) * fl.get(int(o), 0.0)
                    for o in ps.index}
        cand = [int(o) for o in sorted(obs_mass, key=obs_mass.get,
                                       reverse=True)[:250]]
        pred5 = cand[:5]
        pred5_cov = cov_set(pred5)
        cand_mass = float(obs_seg[(obs_seg["fixture"] == fx)
                                  & (obs_seg["road_osmid"].isin(cand))]
                          ["observed"].sum())
        draws = np.empty(args.random_draws)
        for i in range(args.random_draws):
            pick = rng.choice(len(cand), size=min(5, len(cand)), replace=False)
            draws[i] = sum(fl.get(cand[j], 0.0)
                           * obs_rate.get(cand[j], 0.0) for j in pick)
        set_here = site_maps.get(fx, {})
        transfer: list[float] = []
        for gx in fixtures:
            if gx == fx:
                continue
            transfer.append(cov_set(site_maps.get(gx, {}).keys()))
        row = {
            "fixture": fx,
            "squad": int(sites.loc[sites["fixture"] == fx, "squad"].iloc[0]),
            "n_sites": int(sites.loc[sites["fixture"] == fx, "sites"]
                           .iloc[0].count("|") + 1),
            "pred_cov_sample": float(per_game_cov.get(fx, {}).get(
                "coverage", covered_pred.get(fx, 0.0) / max(total_p, 1e-12))),
            "pred_cov_network": float(per_game_cov.get(fx, {}).get(
                "coverage_network",
                covered_pred.get(fx, 0.0) / max(total_p, 1e-12))),
            "realised_obsfrac": float(covered_obs.get(fx, 0.0)
                                      / max(total_o, 1e-12)),
            "realised_raw": float(covered_obs_raw.get(fx, 0.0)
                                  / max(total_o, 1e-12)),
            "oracle_top5": float(oracle_cov / max(total_o, 1e-12)),
            "pred_top5": float(pred5_cov / max(total_o, 1e-12)),
            "random5_mean": float(draws.mean() / max(total_o, 1e-12)),
            "random5_p10": float(np.percentile(draws, 10) / max(total_o, 1e-12)),
            "random5_p90": float(np.percentile(draws, 90) / max(total_o, 1e-12)),
            "candidate_obs_share": float(cand_mass / max(total_o, 1e-12)),
            "transfer_mean": float(np.mean(transfer) / max(total_o, 1e-12)),
            "transfer_min": float(np.min(transfer) / max(total_o, 1e-12)),
            "transfer_max": float(np.max(transfer) / max(total_o, 1e-12)),
            "total_observed": total_o,
            "total_predicted": total_p,
        }
        per.append(row)

    keys = [k for k in per[0] if k != "fixture"]
    pooled = {k: float(np.mean([r[k] for r in per])) for k in keys}
    result = {
        "meta": {
            "mip_dir": str(mip),
            "obs_m": obs_m,
            "n_fixtures": len(per),
            "random_draws": args.random_draws,
            "coverage_note": ("realised_obsfrac is observed mass on staffed "
                              "segment-bins weighted by min(1, obs_m/length); "
                              "transfer_* applies another fixture's site set "
                              "stationary to this fixture"),
        },
        "pooled": pooled,
        "per_fixture": per,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    pd.DataFrame(per).to_csv(out.with_suffix(".csv"), index=False)

    print("=== realised coverage (observed, obs_frac-weighted) ===")
    for r in per:
        print(f"   {r['fixture'][:44]:44s} P={r['squad']}  "
              f"realised={r['realised_obsfrac']:.3f} "
              f"(raw {r['realised_raw']:.3f})  pred={r['pred_cov_sample']:.3f}  "
              f"oracle={r['oracle_top5']:.3f}  random={r['random5_mean']:.3f}  "
              f"transfer={r['transfer_mean']:.3f}")
    print(f"   POOLED realised={pooled['realised_obsfrac']:.3f} "
          f"raw={pooled['realised_raw']:.3f} pred={pooled['pred_cov_sample']:.3f} "
          f"oracle={pooled['oracle_top5']:.3f} random={pooled['random5_mean']:.3f} "
          f"transfer={pooled['transfer_mean']:.3f} "
          f"cand_share={pooled['candidate_obs_share']:.3f}")
    print(f"   wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
