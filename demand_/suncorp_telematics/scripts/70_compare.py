#!/usr/bin/env python3
"""70_compare.py - event vs non-event MIP comparison (R3.3/R3.2 evidence).

Compares four arms that use the SAME solver settings:
  event_pred : shipped predicted surface (results/, run 27008291)
  event_obs  : observed counts on the six held-out fixture evenings
  ctrl_d1    : observed counts on the nearest clean non-event evening
  ctrl_obs   : mean observed counts across up to 3 clean evenings

Reports per-fixture and pooled coverage (candidate and network), squad,
relocations, and the overlap (Jaccard) between each arm's chosen site set
and the shipped event plan's site set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CTRL = RESULTS / "control_mip_gt10"
ARMS = [
    ("event_pred", RESULTS),
    ("event_obs", CTRL / "event_obs"),
    ("ctrl_d1", CTRL / "ctrl_d1"),
    ("ctrl_obs", CTRL / "ctrl_obs"),
]


def base_label(label: str) -> str:
    for pref in ("CTRL1: ", "CTRL: "):
        if label.startswith(pref):
            return label[len(pref):]
    return label


def read_arm(path: Path) -> tuple[pd.DataFrame, dict]:
    sites = pd.read_csv(path / "game_day_mip_sites.csv")
    summ = json.load(open(path / "game_day_mip_summary.json"))["summary"]
    return sites, summ


def main() -> None:
    ref = pd.read_csv(RESULTS / "game_day_mip_sites.csv")
    ref_sets = {r.fixture: set(str(r.segment_osmids).split("|"))
                for r in ref.itertuples()}
    rows = []
    pooled = {}
    for arm, path in ARMS:
        if not (path / "game_day_mip_summary.json").exists():
            print(f"WARN: {arm} missing in {path}, skipping")
            continue
        sites, summ = read_arm(path)
        pooled[arm] = {
            "mean_coverage": summ.get("mean_coverage"),
            "mean_coverage_network": summ.get("mean_coverage_network"),
            "mean_squad": summ.get("mean_squad"),
            "mean_relocations": summ.get("mean_relocations"),
            "distinct_site_sets": summ.get("distinct_site_sets"),
            "network_mass_in_candidates_pct":
                summ.get("network_mass_in_candidates_pct"),
            "mobility_ab": summ.get("mobility_ab"),
        }
        for r in sites.itertuples():
            fx = base_label(r.fixture)
            mine = set(str(r.segment_osmids).split("|"))
            theirs = ref_sets.get(fx)
            jac = (len(mine & theirs) / len(mine | theirs)
                   if theirs else float("nan"))
            rows.append({
                "arm": arm, "fixture": fx, "squad": r.squad,
                "relocations": r.relocations, "n_sites": r.n_sites,
                "coverage": r.coverage, "coverage_network": r.coverage_network,
                "jaccard_vs_event_plan": jac,
            })
    per = pd.DataFrame(rows)
    out = CTRL / "control_compare.csv"
    CTRL.mkdir(parents=True, exist_ok=True)
    per.to_csv(out, index=False)
    for arm in pooled:
        sub = per[per["arm"] == arm]
        pooled[arm]["mean_jaccard_vs_event_plan"] = float(
            sub["jaccard_vs_event_plan"].mean())
    with open(CTRL / "control_compare.json", "w") as fh:
        json.dump({"pooled": pooled,
                   "fixtures": per.to_dict(orient="records")}, fh, indent=2)
    print(per.groupby("arm")[["coverage", "coverage_network", "squad",
                              "relocations", "jaccard_vs_event_plan"]]
          .mean().to_string())
    print()
    print(pd.DataFrame(pooled).T.to_string())
    print(f"\nwrote {out} and {CTRL / 'control_compare.json'}")


if __name__ == "__main__":
    main()
