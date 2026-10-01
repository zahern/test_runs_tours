"""
fuel_price.py — FuelWatch-based fuel price lookup per vehicle/fuel type,
location and day-of-week, for trip fuel-cost computation.

Why: Stage 6 currently uses a flat proxy (stage6_mode_choice.py FUEL_RATE =
$0.12/km; stage6_plan_finalisation.py FUEL_COST_PER_KM = $0.15/km) with no
variation by fuel type (QTS vehicles: petrol 11.5k, diesel 3.9k, hybrid 331,
electric 160, lpg 28), location or weekday. FuelWatch publishes daily retail
prices by product and site, and Perth exhibits a weekly price cycle, so the
same trip costs different amounts on different days.

Sources:
  - Live (opt-in): WA FuelWatch public site (fuelwatch.wa.gov.au) daily
    prices by product x suburb/town. NOTE the site is JS-heavy and HPC
    egress may be blocked — fetch failures fall back to cache, then to the
    built-in weekday table. FuelWatch covers WA only; for SEQ runs supply
    --price-csv with local (e.g. RACQ-average) daily prices in the same
    schema instead.
  - Cache: saved_data/fuel_prices/fuelwatch_daily.csv
    (date, location, fuel_type, price_cpl)
  - Built-in fallback: weekday median c/L per fuel type (Perth-cycle shape:
    midweek low, weekend high), clearly labelled estimates.

Vehicle mapping (3_QTS_VEHICLES.xlsx fuelType/type):
  fuelType {petrol, diesel, hybrid, electric, lpg} x type {car, van,
  motorcycle, truck} -> consumption L/100km (or kWh/100km). EVs priced per
  kWh (overnight charging tariff fallback 32c/kWh).

Trip fuel cost:  cost_$ = price_$/L x L_per_km(veh) x dist_km.
Stage 6 hook (opt-in): attach_fuel_cost(plan_or_trips, vehicles, ...) adds
`fuel_cost_$`; pass it as cost_proxy instead of dist*FUEL_RATE. NOT wired
into stage6_mode_choice.py — estimation default (flat proxy on real HHTS)
is untouched.

PIPELINE ISOLATION: import-safe, __main__-guarded, stdlib+pandas/numpy
(+requests only for live fetch). Writes only saved_data/fuel_prices/.

Usage:
  python fuel_price.py --demo                      # offline smoke test
  python fuel_price.py --fetch --product ULP        # live FuelWatch fetch
  python fuel_price.py --price-csv my_prices.csv    # custom daily prices
"""
import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_ROOT = os.path.normpath(os.path.join(CODE_DIR, ".."))
OUT_DIR = os.path.join(CODE_DIR, "saved_data", "fuel_prices")

# FuelWatch product codes (site search params)
PRODUCTS = {"ULP": 1, "PULP": 2, "98RON": 3, "DIESEL": 4, "LPG": 5, "E10": 6}
FUEL_OF_PRODUCT = {"ULP": "petrol", "PULP": "petrol", "98RON": "petrol",
                   "E10": "petrol", "DIESEL": "diesel", "LPG": "lpg"}

# QTS fuelType -> FuelWatch product (hybrid runs petrol; EV priced per kWh)
QTS_TO_PRODUCT = {"petrol": "ULP", "diesel": "DIESEL", "hybrid": "ULP",
                  "lpg": "LPG", "electric": "EV"}

# Consumption L/100km by (fuelType, type); EV in kWh/100km. Fleet-average
# Australian figures (ABS/Green Vehicle Guide ballparks), documented estimates.
CONSUMPTION = {
    ("petrol", "car"): 9.5, ("petrol", "van"): 11.5, ("petrol", "motorcycle"): 5.0,
    ("petrol", "truck"): 16.0, ("petrol", "other"): 10.0,
    ("diesel", "car"): 7.5, ("diesel", "van"): 9.5, ("diesel", "motorcycle"): 4.5,
    ("diesel", "truck"): 14.0, ("diesel", "other"): 8.5,
    ("hybrid", "car"): 5.0, ("hybrid", "van"): 6.5, ("hybrid", "motorcycle"): 4.0,
    ("hybrid", "truck"): 12.0, ("hybrid", "other"): 5.5,
    ("lpg", "car"): 12.0, ("lpg", "van"): 14.0, ("lpg", "motorcycle"): 6.0,
    ("lpg", "truck"): 18.0, ("lpg", "other"): 12.5,
    ("electric", "car"): 18.0, ("electric", "van"): 24.0, ("electric", "motorcycle"): 8.0,
    ("electric", "truck"): 60.0, ("electric", "other"): 20.0,
}
EV_TARIFF_CPKWH = 32.0  # overnight charging fallback

# Built-in fallback weekday medians (c/L; EVs c/kWh). Labelled estimates with
# the Perth weekly-cycle shape (Tue/Wed low). Replace via --price-csv for SEQ.
FALLBACK_CPL = {
    # Mon..Sun
    "ULP":    [178.5, 172.9, 171.8, 174.6, 179.9, 183.2, 182.4],
    "PULP":   [191.5, 185.9, 184.8, 187.6, 192.9, 196.2, 195.4],
    "98RON":  [199.5, 193.9, 192.8, 195.6, 200.9, 204.2, 203.4],
    "E10":    [175.5, 169.9, 168.8, 171.6, 176.9, 180.2, 179.4],
    "DIESEL": [186.0, 184.5, 183.8, 184.9, 186.8, 188.1, 187.6],
    "LPG":    [102.9, 101.9, 101.5, 102.0, 103.1, 104.0, 103.7],
    "EV":     [32.0] * 7,
}


def fallback_price(product, weekday, location=None):
    """weekday: Mon=0..Sun=6. Location currently selects metro profile;
    regional uplift (+8c/L) applies when location maps outside metro."""
    vals = FALLBACK_CPL[product]
    p = vals[weekday % 7]
    if location and str(location).upper().startswith("REGIONAL:"):
        p += 8.0
    return float(p)


def load_price_table(price_csv=None):
    """Daily price table -> (date, location, fuel_type, price_cpl)."""
    if price_csv and os.path.exists(price_csv):
        t = pd.read_csv(price_csv)
        t["date"] = pd.to_datetime(t["date"]).dt.date
        return t
    p = os.path.join(OUT_DIR, "fuelwatch_daily.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        t["date"] = pd.to_datetime(t["date"]).dt.date
        return t
    return None


def price_cpl(product, when, location="METRO", table=None):
    """Price in cents/L for product on date `when` at `location`.

    Priority: supplied/cached daily table (matching location, else all-site
    median that day, else weekday profile of table) -> built-in fallback
    weekday profile. Never raises for missing data.
    """
    if isinstance(when, dt.datetime):
        when = when.date()
    if table is not None and len(table):
        day = table[table["date"] == when]
        if len(day):
            m = day[(day["location"].astype(str) == str(location))
                    & (day["fuel_type"].astype(str) == product)]
            if len(m):
                return float(m["price_cpl"].median())
            m2 = day[day["fuel_type"].astype(str) == product]
            if len(m2):
                return float(m2["price_cpl"].median())
        wk = when.weekday()
        m3 = table[(table["fuel_type"].astype(str) == product)
                   & (pd.to_datetime(table["date"]).dt.weekday == wk)]
        if len(m3):
            return float(m3["price_cpl"].median())
    return fallback_price(product, when.weekday(), location)


def fetch_fuelwatch_day(product="ULP", suburb="Perth", day="today", out_csv=None):
    """Live fetch from FuelWatch public search (opt-in; needs requests+net).

    Returns DataFrame(date, location, fuel_type, price_cpl) or raises.
    Failures are the caller's signal to use cache/fallback — main() handles.
    """
    import requests  # local import: module stays importable without it
    url = ("https://www.fuelwatch.wa.gov.au/fuelwatch/fuelwatchController.cmd")
    params = {"event": "SEARCH", "product": PRODUCTS[product],
              "suburb": suburb, "day": day}
    r = requests.get(url, params=params, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # The results table is server-rendered HTML; parse station rows.
    tables = pd.read_html(r.text)
    if not tables:
        raise ValueError("no price tables parsed from FuelWatch response")
    best = max(tables, key=len)
    cols = [str(c).lower() for c in best.columns]
    price_col = next((c for c in best.columns
                      if "price" in str(c).lower()), None)
    if price_col is None:
        raise ValueError(f"no price column in {cols}")
    prices = pd.to_numeric(best[price_col].astype(str).str.extract(r"(\d+\.?\d*)")[0],
                           errors="coerce").dropna()
    if not len(prices):
        raise ValueError("no numeric prices parsed")
    today = dt.date.today()
    return pd.DataFrame({"date": [today] * len(prices),
                         "location": [suburb] * len(prices),
                         "fuel_type": [product] * len(prices),
                         "price_cpl": prices.values})


# ------------------------------------------------------- vehicle / trip API
def load_vehicles():
    v = pd.read_excel(os.path.join(TRT_ROOT, "3_QTS_VEHICLES.xlsx"), engine="openpyxl")
    v["fuelType"] = v["fuelType"].astype(str).str.lower()
    v["type"] = v["type"].astype(str).str.lower()
    return v


def consumption_per_km(fuel_type, veh_type):
    ft, tp = str(fuel_type).lower(), str(veh_type).lower()
    key = (ft, tp if (ft, tp) in CONSUMPTION else "other")
    per100 = CONSUMPTION.get(key, CONSUMPTION.get((ft, "other"), 10.0))
    return per100 / 100.0  # L/km (kWh/km for electric)


def trip_fuel_cost(dist_km, fuel_type, veh_type, when, location="METRO", table=None):
    """Fuel cost in $ for one trip leg."""
    ft = str(fuel_type).lower()
    product = QTS_TO_PRODUCT.get(ft, "ULP")
    if product == "EV":
        return float(dist_km) * consumption_per_km(ft, veh_type) * EV_TARIFF_CPKWH / 100.0
    cpl = price_cpl(product, when, location, table)
    return float(dist_km) * consumption_per_km(ft, veh_type) * cpl / 100.0


def attach_fuel_cost(trips, vehicles=None, date_col=None, default_date=None,
                     location_col=None, default_location="METRO",
                     table=None, dist_col="cumdist"):
    """Opt-in Stage 6 helper: adds `fuel_cost_$` per trip row.

    trips: needs dist_col + HHID (+ date_col/location_col if per-day/zone
      pricing wanted; else every row uses default_date/location).
    vehicles: QTS vehicles table (one row per VEHID); uses the household's
      median-consumption vehicle when the exact VEHID is unknown.
    """
    trips = trips.copy()
    vehicles = load_vehicles() if vehicles is None else vehicles.copy()
    vehicles["l_per_km"] = [consumption_per_km(f, t) for f, t in
                            zip(vehicles["fuelType"], vehicles["type"])]
    vehicles["product"] = vehicles["fuelType"].map(QTS_TO_PRODUCT).fillna("ULP")
    hh_med = vehicles.groupby("HHID").agg(
        l_per_km=("l_per_km", "median"),
        product=("product", lambda s: s.value_counts().index[0])).reset_index()
    t = trips.merge(hh_med, on="HHID", how="left")
    t["l_per_km"] = t["l_per_km"].fillna(CONSUMPTION[("petrol", "car")] / 100.0)
    t["product"] = t["product"].fillna("ULP")

    def _when(r):
        if date_col and date_col in t.columns and pd.notna(r[date_col]):
            d = pd.to_datetime(r[date_col]).date()
            return d
        return default_date or dt.date.today()

    def _loc(r):
        if location_col and location_col in t.columns and pd.notna(r[location_col]):
            return str(r[location_col])
        return default_location

    costs = []
    for _, r in t.iterrows():
        w = _when(r)
        if r["product"] == "EV":
            costs.append(float(r[dist_col]) * float(r["l_per_km"]) * EV_TARIFF_CPKWH / 100.0)
        else:
            cpl = price_cpl(r["product"], w, _loc(r), table)
            costs.append(float(r[dist_col]) * float(r["l_per_km"]) * cpl / 100.0)
    trips["fuel_cost_$"] = np.round(costs, 3)
    return trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline smoke test")
    ap.add_argument("--fetch", action="store_true", help="live FuelWatch fetch")
    ap.add_argument("--product", default="ULP", choices=list(PRODUCTS))
    ap.add_argument("--suburb", default="Perth")
    ap.add_argument("--price-csv", default=None)
    a = ap.parse_args()

    if a.fetch:
        try:
            df = fetch_fuelwatch_day(a.product, a.suburb)
            os.makedirs(OUT_DIR, exist_ok=True)
            p = os.path.join(OUT_DIR, "fuelwatch_daily.csv")
            hdr = not os.path.exists(p)
            df.to_csv(p, mode="a", header=hdr, index=False)
            print(f"fetched {len(df)} {a.product} prices @ {a.suburb} -> {p}")
        except Exception as e:
            print(f"live fetch failed ({e}); cache/fallback still available")
    table = load_price_table(a.price_csv)
    print(f"price table: {'none (fallback weekday profile)' if table is None else len(table)}")

    # demo: weekday profile + per-vehicle-type trip cost, fully offline
    v = load_vehicles()
    print(f"vehicles: {len(v)} | " +
          ", ".join(f"{k}={n}" for k, n in v.fuelType.value_counts().items()))
    monday, sunday = dt.date(2026, 9, 14), dt.date(2026, 9, 20)
    for prod in ["ULP", "DIESEL", "LPG"]:
        print(f"  {prod}: Mon {price_cpl(prod, monday, table=table):.1f}c/L, "
              f"Sun {price_cpl(prod, sunday, table=table):.1f}c/L (fallback cycle)")
    d = 10.0  # 10 km trip
    for ft, tp in [("petrol", "car"), ("diesel", "van"), ("hybrid", "car"),
                   ("lpg", "car"), ("electric", "car")]:
        c = trip_fuel_cost(d, ft, tp, monday, table=table)
        flat = d * 0.12
        print(f"  10km {ft}/{tp}: ${c:.2f} vs flat ${flat:.2f}")
    print("pipeline untouched: pass fuel_cost_$ as cost_proxy explicitly to experiment.")


if __name__ == "__main__":
    main()
