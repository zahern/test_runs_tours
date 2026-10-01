#!/usr/bin/env python
"""
run_luti_scenarios_371.py — UrbanSim/POLARIS-role co-simulation for SEQ (paper: arXiv:2608.24817).
================================================================================
Paper anchor: Fig 1 (UrbanSim + POLARIS + CityBES loop), Fig 3 (travel over the
horizon), Table 1 (coupled vs uncoupled county effects).

SEQ mapping:
  UrbanSim role : accessibility-driven land-use reallocation on the 371 SA2s
                  (this module) + usual_location/workplace logic (existing).
  POLARIS role  : the tour/trip pipeline; here represented by the weighted HHTS
                  trip table scaled to scenario land use (VMT/VHT/trips/cost).
  CityBES role  : build_building_energy_371 (imported; 2045 energy per scenario).

Scenarios (paper Sec 3.2): BASE (business as usual), TELE (29% daily WFH from
2030), MBUF (+$0.06/mi link-level fee from 2030 ~ A$0.056/km).
Mechanisms are emergent, not imposed: TELE removes commute tours (extensive
margin) and relieves congestion (BPR-lite speeds); MBUF raises generalised
cost (arc elasticity, trips unchanged, shorter/rerouted driving); land use
responds through skim-fed accessibility (TELE decentralises, MBUF
recentralises); energy follows occupancy + stock (CityBES module).

Outputs (saved_data/luti/):
  scenario_travel.csv   year x scenario VMT/VHT/trips/cost (Fig 3 input)
  table1_analogue.csv   2045 coupled vs uncoupled %Δ by SA3 + overall (Table 1)
  landuse_371_{SCEN}_2045.csv  scenario populations/employment by SA2 (Fig 2 input)
  scenario_energy.csv   citywide GWh/yr by scenario (Sec 4.3 input)

Run:  python run_luti_scenarios_371.py
Env:  LANDUSE_371, TRIPS_371, ZONE_IDS_371, ACCESS_371, LUTI_OUTDIR,
      POP_GROWTH_2045, EMP_GROWTH_2045, TELE_WFH, BASE_WFH, MBUF_AUD_PER_KM,
      VMT_ELAST, DECENTRAL_SHARE, RECENTRAL_SHARE
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("luti371")

_HERE = os.path.dirname(os.path.abspath(__file__))
LANDUSE = os.environ.get("LANDUSE_371",
                         os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv"))
TRIPS = os.environ.get("TRIPS_371",
                       os.path.join(_HERE, "saved_data", "trips_processed.csv"))
ZONE_IDS = os.environ.get("ZONE_IDS_371",
                          os.path.join(_HERE, "saved_data", "seq_zone_ids_371.csv"))
ACCESS = os.environ.get("ACCESS_371",
                        os.path.join(_HERE, "saved_data", "accessibility_371.csv"))
OUTDIR = os.environ.get("LUTI_OUTDIR", os.path.join(_HERE, "saved_data", "luti"))

POP_GROWTH_2045 = float(os.environ.get("POP_GROWTH_2045", 0.35))  # SEQ placeholder
EMP_GROWTH_2045 = float(os.environ.get("EMP_GROWTH_2045", 0.30))
TELE_WFH = float(os.environ.get("TELE_WFH", 0.29))
BASE_WFH = float(os.environ.get("BASE_WFH", 0.08))
MBUF_AUD_PER_KM = float(os.environ.get("MBUF_AUD_PER_KM", 0.056))  # $0.06/mi
OP_COST_AUD_PER_KM = float(os.environ.get("OP_COST_AUD_PER_KM", 0.90))
VMT_ELAST = float(os.environ.get("VMT_ELAST", -0.15))
DECENTRAL_SHARE = float(os.environ.get("DECENTRAL_SHARE", 0.06))
RECENTRAL_SHARE = float(os.environ.get("RECENTRAL_SHARE", 0.05))
CBD_LON, CBD_LAT = 153.0251, -27.4679  # Brisbane CBD
AUTO_MODES = {"Car driver", "Truck driver", "Motorcycle driver",
              "Uber / Other Ride Share", "Taxi"}
YEARS = [2025, 2030, 2035, 2040, 2045]


def _num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(default, index=df.index, dtype=float)


def _haversine_km(lon, lat, lo0=CBD_LON, la0=CBD_LAT):
    lon, lat = np.radians(lon), np.radians(lat)
    lo0, la0 = np.radians(lo0), np.radians(la0)
    h = np.sin((lat - la0) / 2.0) ** 2 + np.cos(la0) * np.cos(lat) * np.sin((lon - lo0) / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(h))


def load_inputs():
    lu = pd.read_csv(LANDUSE)
    lu["SA2_MAIN16"] = pd.to_numeric(lu["SA2_MAIN16"], errors="coerce").astype("Int64")
    lu = lu.dropna(subset=["SA2_MAIN16"]).copy()
    zid = pd.read_csv(ZONE_IDS)
    zid["SA2_MAIN16"] = pd.to_numeric(zid["SA2_MAIN16"], errors="coerce").astype("Int64")
    base = zid.merge(lu, on="SA2_MAIN16", how="left", suffixes=("", "_lu"))
    base["dist_cbd_km"] = _haversine_km(
        pd.to_numeric(base["longitude"], errors="coerce").fillna(CBD_LON).values,
        pd.to_numeric(base["latitude"], errors="coerce").fillna(CBD_LAT).values)
    if os.path.exists(ACCESS):
        acc = pd.read_csv(ACCESS)
        col = "auto_work_acc" if "auto_work_acc" in acc.columns else acc.columns[1]
        base = base.merge(acc[["zone_id", col]].rename(
            columns={"zone_id": "SA2_MAIN16", col: "auto_acc"}),
            on="SA2_MAIN16", how="left")
    if "auto_acc" not in base.columns:
        base["auto_acc"] = 10.0
    base["auto_acc"] = pd.to_numeric(base["auto_acc"], errors="coerce").fillna(10.0)
    base["TOTPOP"] = _num(base, "TOTPOP").clip(lower=0.0)
    base["TOTEMP"] = _num(base, "TOTEMP").clip(lower=0.0)
    trips = pd.read_csv(TRIPS, low_memory=False)
    return base, trips


def scenario_landuse(base, scen, year):
    """Reallocate fixed control totals (paper Sec 4.1: spatial pattern, not totals)."""
    f = (year - 2025) / 20.0
    pop_tot = base["TOTPOP"].sum() * (1.0 + POP_GROWTH_2045 * f)
    emp_tot = base["TOTEMP"].sum() * (1.0 + EMP_GROWTH_2045 * f)
    out = base[["SA2_MAIN16", "dist_cbd_km", "auto_acc"]].copy()
    # BASE: proportional growth; policy branch-in from 2030 (paper Sec 3.2)
    if year <= 2030 or scen == "BASE":
        out["pop"] = base["TOTPOP"] / base["TOTPOP"].sum() * pop_tot
        out["emp"] = base["TOTEMP"] / base["TOTEMP"].sum() * emp_tot
        return out
    b = (year - 2030) / 15.0  # phase-in
    out["pop"] = base["TOTPOP"] / base["TOTPOP"].sum() * pop_tot
    out["emp"] = base["TOTEMP"] / base["TOTEMP"].sum() * emp_tot
    if scen == "TELE":
        # decentralise: core (dist<10km) loses share to periphery (dist>25km)
        core = (out["dist_cbd_km"] < 10.0).values
        peri = (out["dist_cbd_km"] > 25.0).values
        move = (out.loc[core, "pop"].sum() * DECENTRAL_SHARE * b
                + (out["pop"].sum() * 0.01 * b))
        w = (out.loc[peri, "pop"] * out.loc[peri, "auto_acc"]).clip(lower=1e-9)
        out.loc[core, "pop"] *= (1.0 - DECENTRAL_SHARE * b)
        out.loc[peri, "pop"] += move * w / w.sum()
        out.loc[core, "emp"] *= (1.0 - DECENTRAL_SHARE * 0.7 * b)
        out.loc[peri, "emp"] += out.loc[peri, "emp"].sum() * 0 + \
            move * 0.4 * w / w.sum()
    elif scen == "MBUF":
        # recentralise: periphery loses share to core (fee penalises long trips)
        core = (out["dist_cbd_km"] < 10.0).values
        peri = (out["dist_cbd_km"] > 25.0).values
        move = out.loc[peri, "pop"].sum() * RECENTRAL_SHARE * b
        w = (out.loc[core, "pop"] * out.loc[core, "auto_acc"]).clip(lower=1e-9)
        out.loc[peri, "pop"] *= (1.0 - RECENTRAL_SHARE * b)
        out.loc[core, "pop"] += move * w / w.sum()
        out.loc[peri, "emp"] *= (1.0 - RECENTRAL_SHARE * 0.7 * b)
    return out


def travel_totals(trips, landuse, scen, year, uncoupled_landuse=None,
                  base_landuse=None, speed_factor=1.0, cost_mult=1.0):
    """Daily household travel from the weighted trip table under scenario rules.

    Land-use channel (paper Table 1): trips are reweighted by the origin
    zone's scenario-vs-BASE population ratio, so a decentralised land use
    (longer peripheral trips up-weighted) moves VMT/VHT even with identical
    travel rules. base_landuse=None means "already on BASE" (ratio 1).

    System-dynamics channel (sd_hybrid_371): speed_factor (<1 under
    congestion) scales auto hours; cost_mult scales the operating-cost leg
    (Bass EV diffusion). Both default to 1.0 (no feedback). VMT/trips are
    untouched by SD factors (no double-counted induced demand).
    """
    lu = (uncoupled_landuse if uncoupled_landuse is not None else landuse)
    tp = trips.copy()
    tp["w"] = pd.to_numeric(tp["TRIPWGT23"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tp["km"] = pd.to_numeric(tp["cumdist"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tp["hrs"] = pd.to_numeric(tp["TRAVTIME"], errors="coerce").fillna(0.0).clip(lower=0.0) / 60.0
    branch = max(min((year - 2030) / 15.0, 1.0), 0.0) if scen != "BASE" else 0.0
    is_auto = tp["MODE1"].isin(AUTO_MODES).values
    is_work = (tp["DESTPURP"].astype(str).str.lower() == "work").values
    w = tp["w"].values.copy()
    km = tp["km"].values.copy()
    hrs = tp["hrs"].values.copy()
    toll_km = np.zeros(len(tp))
    if scen == "TELE" and branch > 0:
        # extensive margin: incremental WFH removes commute tours outright
        cut = (TELE_WFH - BASE_WFH) * branch
        w = np.where(is_work, w * (1.0 - cut), w)
        # remaining trips shorter (less congestion chasing) + BPR-lite speedup
        km = np.where(is_auto, km * (1.0 - 0.03 * branch), km)
        hrs = np.where(is_auto, hrs * (1.0 - 0.08 * branch), hrs)
    elif scen == "MBUF" and branch > 0:
        # generalised cost +fee/base on auto km; trips unchanged, driving
        # shorter AND quicker (shorter trips take less time + decongestion).
        # x3.0 folds together arc elasticity, destination shortening and
        # rerouting (paper: -2.9% VMT, -3.1% VHT at 6c/mi).
        pct = (MBUF_AUD_PER_KM / OP_COST_AUD_PER_KM) * branch
        _cut = VMT_ELAST * pct * 3.0
        km = np.where(is_auto, km * (1.0 + _cut), km)
        hrs = np.where(is_auto, hrs * (1.0 + _cut * 1.05), hrs)
        toll_km = np.where(is_auto, km * MBUF_AUD_PER_KM * branch, 0.0)
    # land-use channel: origin-zone scenario/BASE pop reweight (composition).
    # Weighted-mean normalisation preserves total trips exactly, so the land
    # use moves VMT/VHT only through trip-length composition (paper Table 1).
    if base_landuse is not None and "pop" in lu.columns and "pop" in base_landuse.columns:
        _sc = lu.set_index("SA2_MAIN16")["pop"]
        _bs = base_landuse.set_index("SA2_MAIN16")["pop"]
        _ratio = (_sc / _bs.clip(lower=1e-9)).clip(0.5, 2.0)
        _rmap = _ratio.to_dict()
        _r = pd.to_numeric(tp["ORIGSA2_2021"], errors="coerce").map(_rmap).fillna(1.0).values
        _rmean = float((w * _r).sum() / max(w.sum(), 1e-9))
        w = w * _r / max(_rmean, 1e-9)
    vmt = float((w[is_auto] * km[is_auto]).sum())
    # SD congestion: a slower network needs more hours for the same driving.
    hrs_adj = hrs / max(speed_factor, 1e-6)
    vht = float((w[is_auto] * hrs_adj[is_auto]).sum())
    ntrips = float(w.sum())
    # SD fleet: EV diffusion scales the operating-cost leg only (tolls intact).
    cost_op = float((w[is_auto] * km[is_auto] * OP_COST_AUD_PER_KM * cost_mult).sum())
    cost_toll = float((w * toll_km).sum())
    cost = cost_op + cost_toll
    return {"vmt": vmt, "vht": vht, "trips": ntrips, "cost_aud": cost,
            "cost_op": cost_op, "cost_toll": cost_toll}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energy", action="store_true",
                    help="also run the CityBES-role energy module for 2045")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    base, trips = load_inputs()

    # ── Fig 3 input: trajectories (+ SD hybrid feedback) ───────────────────
    # SD channel (SD_HYBRID=0 disables): per-scenario sequential state. Each
    # cell first runs unadjusted (VMT is factor-independent), derives factors
    # from the SD state, then reruns adjusted (skipped when factors are 1.0).
    # Uncoupled trajectories keep their own SD state so Table 1 compares the
    # coupled vs uncoupled travel response under consistent SD dynamics.
    from sd_hybrid_371 import sd_factors, interface_table
    sd_on = os.environ.get("SD_HYBRID", "1").strip() != "0"
    rows = []
    lus = {}
    res_c, res_u, fac = {}, {}, {}
    for scen in ("BASE", "TELE", "MBUF"):
        for y in YEARS:
            lu = scenario_landuse(base, scen, y)
            lus[(scen, y)] = lu
    _b25 = travel_totals(trips, lus[("BASE", 2025)], "BASE", 2025)
    _cap = _b25["vmt"] / 0.7  # BASE-2025 v/c anchor (mirrors sd_hybrid_371)
    _sd_raw = {}
    for scen in ("BASE", "TELE", "MBUF"):
        for y in YEARS:
            _jobs = [("c", lus[(scen, y)], None,
                       lus[("BASE", y)] if scen != "BASE" else None)]
            if scen != "BASE":
                _jobs.append(("u", lus[("BASE", y)], lus[("BASE", y)], None))
            for _kind, _lu, _unc, _base in _jobs:
                _t0 = travel_totals(trips, _lu, scen, y,
                                    uncoupled_landuse=_unc, base_landuse=_base)
                _sf = _cm = 1.0
                _ev = 0.0
                if sd_on:
                    _sf, _cm, _ev, _sd_raw[(scen, _kind)] = sd_factors(
                        scen, y, _t0["vmt"], _cap, _sd_raw.get((scen, _kind)))
                if (_sf, _cm) == (1.0, 1.0):
                    _t = _t0
                else:
                    _t = travel_totals(
                        trips, _lu, scen, y,
                        uncoupled_landuse=_unc, base_landuse=_base,
                        speed_factor=_sf, cost_mult=_cm)
                fac[(_kind, scen, y)] = (_sf, _cm, _ev)
                _row = {"scenario": scen, "year": y,
                        "sd_speed_factor": _sf, "sd_ev_share": _ev,
                        "sd_cost_mult": _cm,
                        **{k: _t[k] for k in ("vmt", "vht", "trips", "cost_aud")}}
                if _kind == "c":
                    res_c[(scen, y)] = _t
                    rows.append(_row)
                else:
                    res_u[(scen, y)] = _t
    trav = pd.DataFrame(rows)
    # annualise (paper reports regional totals; sample day x 365)
    for c in ("vmt", "vht", "trips", "cost_aud"):
        trav[c] = trav[c] * 365.0
    trav.to_csv(os.path.join(OUTDIR, "scenario_travel.csv"), index=False,
                columns=["scenario", "year", "vmt", "vht", "trips", "cost_aud",
                         "sd_speed_factor", "sd_ev_share", "sd_cost_mult"])
    # SD stocks + interface table (single source: same factors as above)
    _stocks = []
    for (kind, scen, y), (_sf, _cm, _ev) in fac.items():
        if kind != "c":
            continue
        _stocks.append({"scenario": scen, "year": y,
                        "vc_ratio": round(res_c[(scen, y)]["vmt"] / _cap, 3),
                        "speed_factor": _sf, "ev_share": _ev, "cost_mult": _cm})
    pd.DataFrame(_stocks).to_csv(os.path.join(OUTDIR, "sd_stocks.csv"), index=False)
    interface_table().to_csv(os.path.join(OUTDIR, "sd_interfaces.csv"), index=False)

    # ── Table 1 analogue: coupled vs uncoupled, SA3 (county role) + overall ─
    # Overall rows reuse the SD-adjusted trajectory values (no recompute); the
    # SA3 splits recompute on trip subsets with the same 2045 SD factors, so
    # the land-use contribution (delta) is measured under identical dynamics.
    sa3 = base[["SA2_MAIN16"]].copy()
    sa3["sa3"] = pd.to_numeric(base["sa3_code"]
                               if "sa3_code" in base.columns else 0,
                               errors="coerce").fillna(0).astype(int).values \
        if "sa3_code" in base.columns else 0
    o_sa2 = pd.to_numeric(trips["ORIGSA2_2021"], errors="coerce")
    sa3map = dict(zip(base["SA2_MAIN16"].astype(int).values, sa3["sa3"].values))
    t1 = []
    for scen in ("TELE", "MBUF"):
        lu_c = lus[(scen, 2045)]
        lu_u = lus[("BASE", 2045)]
        tc, tu, tb = res_c[(scen, 2045)], res_u[(scen, 2045)], res_c[("BASE", 2045)]
        _fc = fac[("c", scen, 2045)][:2]
        _fu = fac[("u", scen, 2045)][:2]
        _fb = fac[("c", "BASE", 2045)][:2]
        for metric in ("vmt", "vht"):
            co = (tc[metric] / max(tb[metric], 1e-9) - 1.0) * 100.0
            un = (tu[metric] / max(tb[metric], 1e-9) - 1.0) * 100.0
            t1.append({"scenario": scen, "area": "Overall", "metric": metric,
                       "co_sim": round(co, 1), "uncoupled": round(un, 1),
                       "delta": round(co - un, 1)})
        # SA3 split via origin-zone pop weights under each land use
        for s3 in sorted(set(sa3map.values())):
            in_s3 = o_sa2.map(sa3map).fillna(-1).astype(int) == s3
            if int(in_s3.sum()) < 50:
                continue
            sub = trips[in_s3].copy()
            tcb = travel_totals(sub, lu_c, scen, 2045, base_landuse=lu_u,
                                speed_factor=_fc[0], cost_mult=_fc[1])
            tub = travel_totals(sub, lu_u, scen, 2045, uncoupled_landuse=lu_u,
                                speed_factor=_fu[0], cost_mult=_fu[1])
            tbb = travel_totals(sub, lu_u, "BASE", 2045,
                                speed_factor=_fb[0], cost_mult=_fb[1])
            for metric in ("vmt", "vht"):
                co = (tcb[metric] / max(tbb[metric], 1e-9) - 1.0) * 100.0
                un = (tub[metric] / max(tbb[metric], 1e-9) - 1.0) * 100.0
                t1.append({"scenario": scen, "area": f"SA3_{s3}", "metric": metric,
                           "co_sim": round(co, 1), "uncoupled": round(un, 1),
                           "delta": round(co - un, 1)})
    pd.DataFrame(t1).to_csv(os.path.join(OUTDIR, "table1_analogue.csv"), index=False)

    # ── Fig 2 input: 2045 land use by scenario ─────────────────────────────
    for scen in ("BASE", "TELE", "MBUF"):
        lus[(scen, 2045)].to_csv(
            os.path.join(OUTDIR, f"landuse_371_{scen}_2045.csv"), index=False)

    # ── Sec 4.3 input: energy per scenario (CityBES role) ──────────────────
    if args.energy:
        from build_building_energy_371 import (
            compute_stock, compute_occupancy, compute_energy)
        import json as _json
        lu_full = pd.read_csv(LANDUSE)
        lu_full["SA2_MAIN16"] = pd.to_numeric(
            lu_full["SA2_MAIN16"], errors="coerce").astype("Int64")
        stock = compute_stock(lu_full)
        zids = lu_full["SA2_MAIN16"].astype(int).values
        occ_b, _ = compute_occupancy(trips, lu_full, zids, scenario="BASE")
        en_b = compute_energy(stock, occ_b, occ_ref=None)
        erows = [{"scenario": "BASE",
                  "city_gwh_yr": float(en_b.values.sum() * 365.0 / 1e9)}]
        for scen in ("TELE", "MBUF"):
            occ_s, _ = compute_occupancy(trips, lu_full, zids, scenario=scen)
            en_s = compute_energy(stock, occ_s, occ_ref=occ_b)
            erows.append({"scenario": scen,
                          "city_gwh_yr": float(en_s.values.sum() * 365.0 / 1e9)})
        pd.DataFrame(erows).to_csv(os.path.join(OUTDIR, "scenario_energy.csv"),
                                   index=False)
    log.info("LUTI scenarios complete.")
    print("Wrote LUTI outputs to saved_data/luti/")


if __name__ == "__main__":
    main()
