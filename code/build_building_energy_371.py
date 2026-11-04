#!/usr/bin/env python
"""
build_building_energy_371.py — CityBES-role urban building energy for the 371 SA2 zones.
================================================================================
Paper anchor: arXiv:2608.24817 (UrbanSim + POLARIS + CityBES co-simulation).
This module plays the CityBES role in our SEQ pipeline:

  activity generation (tours/trips, the POLARIS role)  -->  dynamic building
  occupancy by type/vintage  -->  zone site energy  (+ DOE-reference schedules
  for the Fig-4 analogue and EUI benchmarking for the Fig-5 analogue).

Method (documented SEQ assumptions, all env-overridable):
  STOCK: per-zone floorspace by building type from zone_landuse_overture.csv
    (Overture buildings: res_floor_m2, com_floor_m2, bldg_apt_share,
    bldg_office/retail/industrial_m2, jobs_*_m2). Missing splits fall back to
    TOTPOP/TOTEMP shares. 8 ASHRAE-90.1 vintage bins with default SEQ shares.
  OCCUPANCY (paper Sec 2.3.3): each trip's DESTPURP maps to building types via
    PURPOSE_TO_BLDG; its DURATION (min, x TRIPWGT23) accrues occupant-hours in
    the destination SA2 zone. Work hours split across office/retail/industrial/
    edu/health by the destination zone's jobs mix. Home hours = pop*24 -
    out-of-home hours. Hourly curves spread DURATION from ARRTIME.
  ENERGY: E(type,zone) = area * EUI * (1 + OCC_ELAST*(occ/occ_ref - 1)), with
    occ_ref = BASE occupancy, so BASE reproduces benchmark EUIs and scenarios
    deviate only through occupancy/stock shifts (paper Sec 4.3 mechanism).
  SCENARIOS: TELE shifts work occupant-hours home at WFH_RATE (default 0.29,
    paper Sec 3.2.2); MBUF applies a small recentralising shift of occupant-
    hours into central multifamily (paper Fig 7b mechanism).

Outputs (saved_data/energy/):
  building_energy_371_{SCEN}.csv  zone x type site kWh/day
  occupancy_371_{SCEN}.csv        hourly normalised occupancy by type (Fig 4 input)
  energy_totals_{SCEN}.json       citywide GWh/yr by sector + scenario delta
  doe_reference_schedules.csv     stylised ASHRAE reference curves (Fig 4 baseline)

Run:  python build_building_energy_371.py [--scenario BASE|TELE|MBUF]
Env:  LANDUSE_371, TRIPS_371, WFH_RATE, MBUF_MF_SHIFT, OCC_ELAST, ENERGY_OUTDIR
"""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("citybes371")

_HERE = os.path.dirname(os.path.abspath(__file__))
LANDUSE = os.environ.get("LANDUSE_371",
                         os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv"))
TRIPS = os.environ.get("TRIPS_371",
                       os.path.join(_HERE, "saved_data", "trips_processed.csv"))
OUTDIR = os.environ.get("ENERGY_OUTDIR", os.path.join(_HERE, "saved_data", "energy"))

WFH_RATE = float(os.environ.get("WFH_RATE", 0.29))
MBUF_MF_SHIFT = float(os.environ.get("MBUF_MF_SHIFT", 0.03))
OCC_ELAST = float(os.environ.get("OCC_ELAST", 0.6))

# ── prototype EUIs, site kWh/m2/yr (SEQ climate; DOE/NatHERS-anchored defaults) ─
EUI = {
    "single_family": 95.0, "multifamily": 118.0,
    "office_small": 150.0, "office_large": 185.0,
    "retail_standalone": 175.0, "retail_stripmall": 165.0,
    "supermarket": 320.0, "restaurant_full": 420.0, "restaurant_quick": 380.0,
    "school_primary": 110.0, "school_secondary": 120.0,
    "hospital": 380.0, "hotel": 200.0, "warehouse": 90.0,
}
# vintage multipliers (pre-1980 .. 2019, ASHRAE 90.1 bins); default SEQ shares
VINTAGE_MULT = [1.35, 1.25, 1.15, 1.08, 1.0, 0.95, 0.90, 0.85]
VINTAGE_SHARE = [0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10]
AVG_VINT_MULT = float(np.dot(VINTAGE_MULT, VINTAGE_SHARE))

# HHTS DESTPURP -> building types (paper Sec 2.3.3 building-to-activity map)
PURPOSE_TO_BLDG = {
    "work": ["office_small", "office_large", "retail_standalone", "warehouse",
             "school_primary", "hospital"],
    "edu": ["school_primary", "school_secondary"],
    "shopping": ["retail_standalone", "retail_stripmall", "supermarket"],
    "eat": ["restaurant_full", "restaurant_quick"],
    "social": ["single_family", "multifamily", "restaurant_full"],
    "othmaint": ["office_small", "retail_standalone"],
    "othdiscr": ["retail_stripmall", "hotel"],
    "escort": ["school_primary", "school_secondary"],
    "home": ["single_family", "multifamily"],
}
RES_TYPES = ["single_family", "multifamily"]
WORK_SPLIT_JOBS = {  # dest-zone jobs columns used to split work hours
    "office_small": ["jobs_office_n", "FPSEMPN"],
    "office_large": ["jobs_office_m2", "FPSEMPN"],
    "retail_standalone": ["jobs_retail_n", "RETEMPN"],
    "warehouse": ["jobs_industrial_n", "MWTEMPN"],
    "school_primary": ["jobs_education_n", "HSENROLL"],
    "hospital": ["jobs_health_n", "HEREMPN"],
}


def _num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(default, index=df.index, dtype=float)


def compute_stock(lu):
    """Per-zone floorspace (m2) by building type from Overture + fallbacks."""
    n = len(lu)
    res = _num(lu, "res_floor_m2")
    if (res == 0).all():
        res = _num(lu, "TOTPOP") * 55.0  # 55 m2/person fallback
    apt = _num(lu, "bldg_apt_share").clip(upper=1.0)
    if (apt == 0).all():
        apt = pd.Series(0.25, index=lu.index)  # SEQ default unit share
    stock = pd.DataFrame(index=lu.index)
    stock["single_family"] = res * (1.0 - apt)
    stock["multifamily"] = res * apt
    com = _num(lu, "com_floor_m2")
    if (com == 0).all():
        com = _num(lu, "TOTEMP") * 30.0
    off = (_num(lu, "bldg_office_m2") + _num(lu, "jobs_office_m2")).clip(lower=0.0)
    ret = (_num(lu, "bldg_retail_m2") + _num(lu, "jobs_retail_m2")).clip(lower=0.0)
    ind = (_num(lu, "bldg_industrial_m2") + _num(lu, "jobs_industrial_m2")).clip(lower=0.0)
    edu = _num(lu, "jobs_education_m2")
    hea = _num(lu, "jobs_health_m2")
    known = off + ret + ind + edu + hea
    # scale known splits to com where com observed, else use as-is
    scale = (com / known.clip(lower=1.0)).clip(upper=3.0)
    scale = scale.where(known > 0, 0.0)
    stock["office_small"] = off * 0.4 * scale + com * 0.10 * (known == 0).astype(float)
    stock["office_large"] = off * 0.6 * scale + com * 0.15 * (known == 0).astype(float)
    stock["retail_standalone"] = ret * 0.5 * scale + com * 0.10 * (known == 0).astype(float)
    stock["retail_stripmall"] = ret * 0.3 * scale
    stock["supermarket"] = ret * 0.2 * scale + com * 0.03 * (known == 0).astype(float)
    stock["restaurant_full"] = com * 0.02
    stock["restaurant_quick"] = com * 0.015
    stock["school_primary"] = edu * 0.6 * scale + com * 0.02 * (known == 0).astype(float)
    stock["school_secondary"] = edu * 0.4 * scale + com * 0.015 * (known == 0).astype(float)
    stock["hospital"] = hea * scale + com * 0.02 * (known == 0).astype(float)
    stock["hotel"] = com * 0.03
    stock["warehouse"] = ind * scale + com * 0.05 * (known == 0).astype(float)
    return stock.fillna(0.0)


def _work_split_weights(lu):
    """Dest-zone jobs mix -> share of work occupant-hours per building type."""
    w = pd.DataFrame(index=lu.index)
    for b, cols in WORK_SPLIT_JOBS.items():
        w[b] = sum((_num(lu, c) for c in cols), start=pd.Series(0.0, index=lu.index))
    tot = w.sum(axis=1).clip(lower=1e-9)
    return (w.T / tot.values).T.fillna(1.0 / len(w.columns))


def compute_occupancy(trips, lu, zone_ids, scenario="BASE", wfh_rate=WFH_RATE):
    """Occupant-hours/day by zone x type + 24-h normalised curves by type."""
    lu = lu.copy()
    lu["SA2_MAIN16"] = pd.to_numeric(lu["SA2_MAIN16"], errors="coerce")
    lidx = {int(z): i for i, z in enumerate(zone_ids)}
    n = len(zone_ids)
    types = list(EUI)
    occ = pd.DataFrame(0.0, index=range(n), columns=types)
    hourly = pd.DataFrame(0.0, index=range(24), columns=types)
    wsplit = _work_split_weights(lu.set_index("SA2_MAIN16").reindex(zone_ids).reset_index())

    tp = trips.copy()
    tp["DESTSA2"] = pd.to_numeric(tp["DESTSA2_2021"], errors="coerce")
    tp = tp[tp["DESTSA2"].isin(lidx)].copy()
    tp["dur_h"] = pd.to_numeric(tp["DURATION"], errors="coerce").fillna(0.0).clip(lower=0, upper=20) / 60.0
    tp["w"] = pd.to_numeric(tp["TRIPWGT23"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tp["arr_h"] = pd.to_numeric(tp["ARRTIME"], errors="coerce").fillna(0.0) / 60.0
    tp["purp"] = tp["DESTPURP"].astype(str).str.lower()

    for purp, blds in PURPOSE_TO_BLDG.items():
        if purp == "home":
            continue  # home handled from population residual below
        sub = tp[tp["purp"] == purp]
        if len(sub) == 0:
            continue
        zi = sub["DESTSA2"].map(lidx).values.astype(int)
        hrs = (sub["dur_h"] * sub["w"] / 365.0).values  # avg-day occupant-hours
        if purp == "work":
            W = wsplit[[b for b in blds]].values  # (n, n_b)
            for j, b in enumerate(blds):
                np.add.at(occ[b].values, zi, hrs * W[zi, j])
        else:
            share = 1.0 / len(blds)
            for b in blds:
                np.add.at(occ[b].values, zi, hrs * share)
        # hourly curves (spread duration from arrival hour)
        ah = np.clip(sub["arr_h"].values.astype(int), 0, 23)
        dh = np.clip(np.ceil(sub["dur_h"].values).astype(int), 1, 24)
        share = 1.0 / len(blds)
        for (h0, dd, hh) in zip(ah, dh, hrs):
            for h in range(h0, min(h0 + dd, 24)):
                for b in blds:
                    hourly.loc[h, b] += hh * share / max(dd, 1)

    # home hours: residents home = pop*24 - out-of-home (non-home occ attributed)
    pop = lu.set_index("SA2_MAIN16").reindex(zone_ids)["TOTPOP"]
    pop = pd.to_numeric(pop, errors="coerce").fillna(0.0).clip(lower=0.0).values / 365.0
    ooh = occ.sum(axis=1).values  # avg-day person-hours out (one occupant per hour)
    home_h = np.maximum(pop * 24.0 - ooh, pop * 8.0)
    sf_share = 1.0 - _num(lu.set_index("SA2_MAIN16").reindex(zone_ids).reset_index(),
                          "bldg_apt_share").clip(upper=1.0).fillna(0.75).values
    occ["single_family"] = occ["single_family"].values + home_h * sf_share
    occ["multifamily"] = occ["multifamily"].values + home_h * (1.0 - sf_share)
    hourly.loc[:, "single_family"] += home_h.mean() / 24.0
    hourly.loc[:, "multifamily"] += home_h.mean() / 24.0

    # ── scenario shifts (paper Sec 4.3 mechanisms) ──────────────────────────
    if scenario.upper() == "TELE":
        # WFH is office-work biased: full rate on offices, partial on
        # retail/warehouse (frontline), none on schools/hospitals.
        for b, f in [("office_small", 1.0), ("office_large", 1.0),
                     ("retail_standalone", 0.3), ("warehouse", 0.2)]:
            moved = occ[b].values * wfh_rate * f
            occ[b] = occ[b].values - moved
            occ["single_family"] = occ["single_family"].values + moved * 0.8
            occ["multifamily"] = occ["multifamily"].values + moved * 0.2
        hourly.loc[9:16, "single_family"] *= (1.0 + wfh_rate)
        hourly.loc[9:16, "multifamily"] *= (1.0 + wfh_rate)
        hourly.loc[9:16, ["office_small", "office_large"]] *= (1.0 - wfh_rate)
    elif scenario.upper() == "MBUF":
        # Central-MF concentration capped: MF is a small stock, so an uncapped
        # SF->MF flow would explode the MF occupancy ratio (clip x2.2 energy).
        _sf = occ["single_family"].values
        _mf = occ["multifamily"].values
        shift = np.minimum(_sf * MBUF_MF_SHIFT, _mf * 0.10)
        occ["single_family"] = _sf - shift
        occ["multifamily"] = _mf + shift

    occ.index = zone_ids
    # normalised daily curves per type
    norm = hourly.div(hourly.max(axis=0).clip(lower=1e-9), axis=1)
    return occ, norm


def doe_reference_schedules():
    """Stylised ASHRAE/DOE weekday occupancy curves (Fig-4 baseline)."""
    h = np.arange(24)
    office = np.clip(np.exp(-((h - 13.0) ** 2) / 18.0), 0, 1) * 0.95 + 0.05
    office[(h < 7) | (h > 19)] = 0.05
    retail = np.clip(np.exp(-((h - 14.0) ** 2) / 20.0), 0, 1) * 0.9 + 0.05
    retail[(h < 8) | (h > 21)] = 0.03
    school = np.clip(np.exp(-((h - 12.0) ** 2) / 8.0), 0, 1) * 0.95
    school[(h < 7) | (h > 16)] = 0.0
    home = 0.55 + 0.40 * np.exp(-((h - 2.0) ** 2) / 18.0) + 0.25 * np.exp(-((h - 20.0) ** 2) / 10.0)
    home = home / home.max()
    restaurant = np.clip(np.exp(-((h - 13.0) ** 2) / 6.0) + np.exp(-((h - 19.5) ** 2) / 6.0), 0, 1) * 0.9 + 0.05
    hospital = pd.Series(0.7, index=h) + 0.2 * np.exp(-((h - 12.0) ** 2) / 30.0)
    warehouse = np.clip(np.exp(-((h - 11.0) ** 2) / 25.0), 0, 1) * 0.8 + 0.1
    hotel = 0.6 + 0.35 * np.exp(-((h - 1.0) ** 2) / 12.0)
    ref = pd.DataFrame({
        "single_family": home, "multifamily": home,
        "office_small": office, "office_large": office,
        "retail_standalone": retail, "retail_stripmall": retail,
        "supermarket": retail, "restaurant_full": restaurant,
        "restaurant_quick": restaurant, "school_primary": school,
        "school_secondary": school, "hospital": hospital.values,
        "hotel": hotel, "warehouse": warehouse,
    }, index=h)
    ref.index.name = "hour"
    return ref


def compute_energy(stock, occ, occ_ref=None):
    """Site kWh/day by zone x type. occ_ref=None -> returns BASE (EUI) energy."""
    eui_day = {t: v * AVG_VINT_MULT / 365.0 for t, v in EUI.items()}
    en = pd.DataFrame(index=stock.index, columns=list(EUI), dtype=float)
    for t in EUI:
        base = stock[t].values * eui_day[t]
        if occ_ref is None:
            en[t] = base
        else:
            # 0/0 (no occupancy in either case) means "no change" -> ratio 1,
            # not 0 (which would wrongly discount empty-but-stocked cells).
            _o = occ[t].values.astype(float)
            _r = occ_ref[t].clip(lower=1e-9).values.astype(float)
            ratio = np.divide(_o, _r, out=np.ones_like(_o), where=_r > 1e-6)
            ratio = np.clip(ratio, 0.2, 3.0)
            en[t] = base * (1.0 + OCC_ELAST * (ratio - 1.0))
    return en.fillna(0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="BASE")
    args = ap.parse_args()
    scen = args.scenario.upper()
    os.makedirs(OUTDIR, exist_ok=True)

    lu = pd.read_csv(LANDUSE)
    key = next((c for c in ("SA2_MAIN16",) if c in lu.columns), lu.columns[0])
    lu["SA2_MAIN16"] = pd.to_numeric(lu[key], errors="coerce").astype("Int64")
    lu = lu.dropna(subset=["SA2_MAIN16"]).copy()
    zone_ids = lu["SA2_MAIN16"].astype(int).values

    trips = pd.read_csv(TRIPS, low_memory=False)
    stock = compute_stock(lu)
    occ_base, _ = compute_occupancy(trips, lu, zone_ids, scenario="BASE")
    occ_scen, norm = compute_occupancy(trips, lu, zone_ids, scenario=scen)
    en_base = compute_energy(stock, occ_base, occ_ref=None)
    en = compute_energy(stock, occ_scen, occ_ref=occ_base)

    en_out = en.copy()
    en_out.insert(0, "SA2_MAIN16", zone_ids)
    en_out.to_csv(os.path.join(OUTDIR, f"building_energy_371_{scen}.csv"), index=False)
    norm.to_csv(os.path.join(OUTDIR, f"occupancy_371_{scen}.csv"))
    doe_reference_schedules().to_csv(os.path.join(OUTDIR, "doe_reference_schedules.csv"))

    tot = {t: float(en[t].sum() * 365.0 / 1e9) for t in EUI}  # GWh/yr
    tot_b = {t: float(en_base[t].sum() * 365.0 / 1e9) for t in EUI}
    city = float(en.values.sum() * 365.0 / 1e9)
    city_b = float(en_base.values.sum() * 365.0 / 1e9)
    rep = {"scenario": scen, "city_gwh_yr": round(city, 1),
           "base_gwh_yr": round(city_b, 1),
           "delta_pct": round((city / max(city_b, 1e-9) - 1.0) * 100.0, 2),
           "by_type_gwh": {t: round(v, 2) for t, v in tot.items()},
           "base_by_type_gwh": {t: round(v, 2) for t, v in tot_b.items()}}
    with open(os.path.join(OUTDIR, f"energy_totals_{scen}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    log.info("%s: city %.1f GWh/yr (%+.2f%% vs BASE)", scen, city, rep["delta_pct"])
    print(f"Wrote energy outputs for {scen}")


if __name__ == "__main__":
    main()
