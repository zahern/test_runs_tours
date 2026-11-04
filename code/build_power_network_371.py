#!/usr/bin/env python
"""
build_power_network_371.py — synthetic distribution network for the 371 SA2 zones.
================================================================================
URBANopt-anchored (docs.urbanopt.net/workflows/opendss/opendss.html + disco):

  URBANopt OpenDSS workflow inputs = GeoJSON electric network + electrical
  database (equipment specs) + timeseries building loads -> DiTTo/OpenDSS
  powerflow -> overload / voltage-violation flags.

At 371-SA2 scale a building-level URBANopt model is infeasible, so this builder
produces the SAME three inputs at representative-feeder scale, one feeder set
per SA2 zone:

  1. saved_data/power/sa2_371_electric.geojson  (URBANopt-style Feature file:
     Building + ElectricalJunction + ElectricalConnector + Substation features)
  2. saved_data/power/electrical_database.json  (equipment catalogue, same role
     as urbanopt-ditto-reader/example/electrical_database.json)
  3. saved_data/power/profiles/zone_<SA2>_load.csv  (24-h kW timeseries loads,
     same role as URBANopt default_feature_report timeseries)
  4. saved_data/power/zone_power_371.csv  (per-zone peak/energy/sizing summary)
  5. saved_data/power/opendss_config.json  (ditto-reader-style run config)

Load model (all coefficients env-tunable, documented assumptions for SEQ):
  residential peak  = TOTHH * RES_ADMD_KW        (default 1.8 kW/hh after-diversity)
  commercial peak   = TOTEMP * COM_KW_PER_JOB    (default 1.2) + com_floor_m2 * 0.08
  industrial peak   = jobs_industrial_m2 * 0.05  (falls back to MWTEMPN * 2.0)
  EV peak           = n_ev_charging * 11 * 0.4 + TOTHH * EV_ADOPT * 7 * 0.1
  daily energy      = peak * load-factor mix (res 0.45 / com 0.55 / ind 0.65)

Network sizing per zone (representative LV feeder):
  n_transformers = ceil(peak_kVA / XFMR_KVA)  (default 500 kVA unit)
  feeder_km      = max(0.5, sqrt(area_sqkm) * FEEDER_KM_FACTOR)  (default 1.2,
                   derated by customer density so CBD zones get short feeders)
  n_feeders      = clamp(ceil(n_customers / 250), 1, 12)

Offline, pure pandas/numpy (+ geopandas only if present for polygon areas).
Idempotent: re-running overwrites the power/ outputs.

Run:  python build_power_network_371.py
Env:  ZONE_IDS_371, LANDUSE_371, SA2_GEOJSON, POWER_OUTDIR,
      RES_ADMD_KW, COM_KW_PER_JOB, XFMR_KVA, FEEDER_KM_FACTOR, EV_ADOPT
"""
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("power371")

_HERE = os.path.dirname(os.path.abspath(__file__))
ZONE_IDS = os.environ.get("ZONE_IDS_371",
                           os.path.join(_HERE, "saved_data", "seq_zone_ids_371.csv"))
LANDUSE = os.environ.get("LANDUSE_371",
                         os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv"))
SA2_GEOJSON = os.environ.get("SA2_GEOJSON",
                             os.path.join(_HERE, "saved_data", "sa2_371.geojson"))
OUTDIR = os.environ.get("POWER_OUTDIR", os.path.join(_HERE, "saved_data", "power"))

# ── tunable assumptions ──────────────────────────────────────────────────────
RES_ADMD_KW = float(os.environ.get("RES_ADMD_KW", 1.8))
COM_KW_PER_JOB = float(os.environ.get("COM_KW_PER_JOB", 1.2))
COM_KW_PER_M2 = float(os.environ.get("COM_KW_PER_M2", 0.08))
IND_KW_PER_M2 = float(os.environ.get("IND_KW_PER_M2", 0.05))
IND_KW_PER_JOB = float(os.environ.get("IND_KW_PER_JOB", 2.0))
XFMR_KVA = float(os.environ.get("XFMR_KVA", 500.0))
FEEDER_KM_FACTOR = float(os.environ.get("FEEDER_KM_FACTOR", 1.2))
EV_ADOPT = float(os.environ.get("EV_ADOPT", 0.06))  # share of hh with home EV
PV_KW_PER_HH = float(os.environ.get("PV_KW_PER_HH", 1.5))  # rooftop proxy


def _num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(default, index=df.index, dtype=float)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "profiles"), exist_ok=True)

    zones = pd.read_csv(ZONE_IDS)
    zid = next((c for c in ("SA2_MAIN16", "sa2_code", "SA2_CODE") if c in zones.columns), None)
    if zid is None:
        raise SystemExit(f"FATAL: no SA2 code column in {ZONE_IDS}")
    zones["SA2_MAIN16"] = pd.to_numeric(zones[zid], errors="coerce").astype("Int64")
    zones = zones.dropna(subset=["SA2_MAIN16"]).copy()
    zones["SA2_MAIN16"] = zones["SA2_MAIN16"].astype(int)

    lu = pd.read_csv(LANDUSE)
    lkey = next((c for c in ("SA2_MAIN16", "sa2_code", "SA2_CODE") if c in lu.columns), None)
    if lkey is None:
        raise SystemExit(f"FATAL: no SA2 code column in {LANDUSE}")
    lu["SA2_MAIN16"] = pd.to_numeric(lu[lkey], errors="coerce").astype("Int64")

    df = zones.merge(lu, on="SA2_MAIN16", how="left", suffixes=("", "_lu"))
    log.info("zones=%d landuse_cols=%d", len(df), len(lu.columns))

    tothh = _num(df, "TOTHH")
    totemp = _num(df, "TOTEMP")
    if (totemp == 0).all():  # fallback: sum employment categories
        emp_cols = ["RETEMPN", "FPSEMPN", "HEREMPN", "AGREMPN", "MWTEMPN", "OTHEMP"]
        totemp = sum((_num(df, c) for c in emp_cols), start=pd.Series(0.0, index=df.index))
    com_m2 = _num(df, "com_floor_m2") + _num(df, "jobs_office_m2") + _num(df, "jobs_retail_m2")
    ind_m2 = _num(df, "jobs_industrial_m2")
    ind_jobs = _num(df, "MWTEMPN") + _num(df, "AGREMPN")
    n_ev = _num(df, "n_ev_charging")

    res_kw = tothh * RES_ADMD_KW
    com_kw = totemp * COM_KW_PER_JOB + com_m2 * COM_KW_PER_M2
    ind_kw = np.where(ind_m2 > 0, ind_m2 * IND_KW_PER_M2, ind_jobs * IND_KW_PER_JOB)
    ind_kw = pd.Series(ind_kw, index=df.index, dtype=float).clip(lower=0.0)
    ev_kw = n_ev * 11.0 * 0.4 + tothh * EV_ADOPT * 7.0 * 0.1
    peak_kw = res_kw + com_kw + ind_kw + ev_kw
    peak_kva = peak_kw / 0.92  # SEQ power factor assumption
    energy_kwh = res_kw * 24 * 0.45 + com_kw * 24 * 0.55 + ind_kw * 24 * 0.65 + ev_kw * 24 * 0.30
    pv_kw = tothh * PV_KW_PER_HH

    area = _num(df, "area_sqkm", 10.0).clip(lower=0.1)
    n_customers = (tothh + totemp / 3.0).clip(lower=1.0)
    n_xfmr = np.ceil(peak_kva / XFMR_KVA).astype(int).clip(lower=1)
    density = n_customers / area
    urban = np.clip(np.log1p(density / 500.0) / np.log1p(20000.0 / 500.0), 0.15, 1.0)
    feeder_km = np.maximum(0.5, np.sqrt(area) * FEEDER_KM_FACTOR * (1.3 - urban))
    n_feeders = np.ceil(n_customers / 250.0).astype(int).clip(lower=1, upper=12)

    power = pd.DataFrame({
        "SA2_MAIN16": df["SA2_MAIN16"].values,
        "ORDER": pd.to_numeric(df["ORDER"] if "ORDER" in df.columns else df["SA2_MAIN16"],
                               errors="coerce").fillna(0).astype(int).values,
        "res_kw": res_kw.values, "com_kw": com_kw.values,
        "ind_kw": np.asarray(ind_kw, dtype=float), "ev_kw": ev_kw.values,
        "peak_kw": peak_kw.values, "peak_kva": peak_kva.values,
        "energy_kwh_day": energy_kwh.values, "pv_kw": pv_kw.values,
        "n_customers": n_customers.values, "n_transformers": n_xfmr,
        "xfmr_mva": (n_xfmr * XFMR_KVA / 1000.0),
        "feeder_km": np.asarray(feeder_km, dtype=float), "n_feeders": n_feeders,
        "longitude": pd.to_numeric(df.get("longitude", 153.0), errors="coerce").fillna(153.0).values,
        "latitude": pd.to_numeric(df.get("latitude", -27.5), errors="coerce").fillna(-27.5).values,
    })
    zpath = os.path.join(OUTDIR, "zone_power_371.csv")
    power.to_csv(zpath, index=False)
    log.info("wrote %d zone rows -> %s (peak %.1f MW system)",
             len(power), zpath, power["peak_kw"].sum() / 1000.0)

    # ── 24-h normalised profiles per zone (res/com/ind/EV shapes) ────────────
    hrs = np.arange(24)
    res_shape = 0.55 + 0.45 * np.exp(-((hrs - 18.5) ** 2) / 8.0) + 0.15 * np.exp(-((hrs - 7.5) ** 2) / 6.0)
    com_shape = 0.35 + 0.65 * np.exp(-((hrs - 13.0) ** 2) / 18.0)
    ind_shape = 0.70 + 0.30 * np.exp(-((hrs - 10.0) ** 2) / 40.0)
    ev_shape = 0.25 + 0.75 * np.exp(-((hrs - 21.0) ** 2) / 10.0)
    for _, r in power.iterrows():
        prof = (r["res_kw"] * res_shape / res_shape.max()
                + r["com_kw"] * com_shape / com_shape.max()
                + r["ind_kw"] * ind_shape / ind_shape.max()
                + r["ev_kw"] * ev_shape / ev_shape.max()
                - r["pv_kw"] * np.clip(np.sin((hrs - 6) / 12.0 * np.pi), 0, 1) * 0.8)
        prof = np.clip(prof, 0.05 * r["peak_kw"] if r["peak_kw"] > 0 else 0.0, None)
        pd.DataFrame({"hour": hrs, "kw": prof}).to_csv(
            os.path.join(OUTDIR, "profiles", f"zone_{int(r['SA2_MAIN16'])}_load.csv"), index=False)

    # ── electrical database (URBANopt-ditto-reader role) ─────────────────────
    edb = {
        "source": "synthetic SEQ catalogue (placeholder for Energex/Powerlink specs)",
        "transformers": [
            {"name": "SEQ_PAD_500kVA", "kva": 500, "primary_kv": 11.0,
             "secondary_kv": 0.415, "impedance_pct": 4.5},
            {"name": "SEQ_PAD_1000kVA", "kva": 1000, "primary_kv": 11.0,
             "secondary_kv": 0.415, "impedance_pct": 5.0},
            {"name": "SEQ_PAD_1500kVA", "kva": 1500, "primary_kv": 11.0,
             "secondary_kv": 0.415, "impedance_pct": 5.5},
        ],
        "lines": [
            {"name": "SEQ_OH_ACSR_150", "r_ohm_per_km": 0.21, "x_ohm_per_km": 0.35,
             "ampacity_a": 400, "kv": 11.0},
            {"name": "SEQ_UG_XLPE_240", "r_ohm_per_km": 0.09, "x_ohm_per_km": 0.12,
             "ampacity_a": 550, "kv": 11.0},
        ],
        "notes": "Replace values with utility specs before investment-grade use.",
    }
    with open(os.path.join(OUTDIR, "electrical_database.json"), "w") as f:
        json.dump(edb, f, indent=2)

    # ── URBANopt-style electric GeoJSON (buildings + junctions/connectors) ───
    features = []
    for _, r in power.iterrows():
        sa2 = int(r["SA2_MAIN16"])
        lon, lat = float(r["longitude"]), float(r["latitude"])
        features.append({
            "type": "Feature",
            "properties": {"id": f"bldg_{sa2}", "type": "Building",
                           "name": f"SA2 {sa2} aggregate load",
                           "peak_kw": round(float(r["peak_kw"]), 1),
                           "pv_kw": round(float(r["pv_kw"]), 1)},
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
        features.append({
            "type": "Feature",
            "properties": {"id": f"sub_{sa2}", "type": "Substation",
                           "name": f"SA2 {sa2} zone substation",
                           "kv": 33.0, "mva": round(float(r["peak_kva"] / 1000 * 1.25 + 1.0), 2)},
            "geometry": {"type": "Point", "coordinates": [lon + 0.002, lat + 0.002]},
        })
        for k in range(int(r["n_feeders"])):
            features.append({
                "type": "Feature",
                "properties": {"id": f"junc_{sa2}_{k}", "type": "ElectricalJunction",
                               "feeder": k},
                "geometry": {"type": "Point",
                             "coordinates": [lon + 0.001 * (k + 1), lat - 0.001 * (k + 1)]},
            })
            features.append({
                "type": "Feature",
                "properties": {"id": f"line_{sa2}_{k}", "type": "ElectricalConnector",
                               "from": f"sub_{sa2}", "to": f"junc_{sa2}_{k}",
                               "equipment": "SEQ_OH_ACSR_150",
                               "length_km": round(float(r["feeder_km"] / max(int(r["n_feeders"]), 1)), 3)},
                "geometry": {"type": "LineString", "coordinates": [
                    [lon + 0.002, lat + 0.002],
                    [lon + 0.001 * (k + 1), lat - 0.001 * (k + 1)]]},
            })
    with open(os.path.join(OUTDIR, "sa2_371_electric.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    cfg = {
        "urbanopt_scenario_file": "saved_data/power/zone_power_371.csv",
        "equipment_file": "saved_data/power/electrical_database.json",
        "opendss_folder": "saved_data/power/opendss",
        "urbanopt_geojson_file": "saved_data/power/sa2_371_electric.geojson",
        "load_profiles": "saved_data/power/profiles/zone_<SA2>_load.csv",
        "use_reopt": False, "number_of_timepoints": 24,
    }
    with open(os.path.join(OUTDIR, "opendss_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("electric GeoJSON: %d features | profiles: %d zones",
             len(features), len(power))
    print("Power network build complete.")


if __name__ == "__main__":
    main()
