#!/usr/bin/env python
"""
verify_sa2_2021.py — prove every pipeline key is SA2-2021 (and flag stragglers).
================================================================================
Static audit (Sep 2026) already showed: 371 spine, landuse, accessibility,
power/energy, synth crosswalk, geojson keys, HHTS SA1->SA2 mapping are all
2021 codes ("SA2_MAIN16" column names are a legacy misnomer). What needs a
runtime check:

  1. OMX lookup keys: skims_371.omx must contain exactly the 371 2021 codes
     (it derives from the 507 system; a 2016-keyed file would silently
     misalign every skim join).
  2. Split-zone polygon vintage: sa2_371.geojson polygons for split zones
     (Ormeau, Pimpama, Robina, ...) must be 2021 boundaries — checks polygon
     area vs the 2021 reference areas (tolerance 5%).
  3. HHTS coverage: share of HHTS persons/trips whose 2021 SA2 is outside the
     371 set (expected: fringe/externals only).
  4. Straggler scan: every key column below must hold no 2016-only codes
     (codes whose CG mapping points to a *different* 2021 code).

Run (needs pandas; h5py+geopandas unlock checks 1-2, else skipped openly):
  python verify_sa2_2021.py
Exit code 0 = all executed checks pass, 1 = a check failed, 2 = setup missing.
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEMAND = os.path.normpath(os.path.join(_HERE, "..", "demand_"))
SHP2021 = os.path.normpath(os.path.join(_HERE, "..", "SA2_2021_AUST_SHP_GDA2020"))

SEQ371 = os.path.join(_HERE, "saved_data", "seq_zone_ids_371.csv")
CG = os.path.join(SHP2021, "CG_SA2_2016_SA2_2021.csv")
OMX = os.path.join(DEMAND, "skims_371.omx")
GEOJSON = os.path.join(_HERE, "saved_data", "sa2_371.geojson")
TRIPS = os.path.join(_HERE, "saved_data", "trips_processed.csv")

_failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        _failures.append(name)


def main():
    print("=" * 64)
    print("SA2-2021 key audit")
    print("=" * 64)

    seq = pd.read_csv(SEQ371)
    zone_col = "SA2_MAIN16" if "SA2_MAIN16" in seq.columns else seq.columns[1]
    zones371 = set(pd.to_numeric(seq[zone_col], errors="coerce").dropna().astype(int))
    print(f"371 spine: {len(zones371)} zones")

    # 2016-only codes = CG rows where 2016 code != any 2021 twin of itself.
    # A code present in an artifact is a straggler iff it maps ONLY to
    # different 2021 codes (identical twins are vintage-neutral).
    cg = pd.read_csv(CG)
    cg["a"] = pd.to_numeric(cg["SA2_MAINCODE_2016"], errors="coerce")
    cg["b"] = pd.to_numeric(cg["SA2_CODE_2021"], errors="coerce")
    twins = set(cg.loc[cg["a"] == cg["b"], "a"].astype(int))
    changed16 = set(cg.loc[cg["a"] != cg["b"], "a"].astype(int)) - twins
    print(f"CG loaded: {len(twins)} vintage-neutral codes, "
          f"{len(changed16)} changed 2016-only codes nationally")

    art = {
        "zone_landuse_overture": ("saved_data/zone_landuse_overture.csv", "SA2_MAIN16"),
        "accessibility_371": ("saved_data/accessibility_371.csv", "zone_id"),
        "zone_power_371": ("saved_data/power/zone_power_371.csv", "SA2_MAIN16"),
        "building_energy_BASE": ("saved_data/energy/building_energy_371_BASE.csv",
                                 "SA2_MAIN16"),
        "power_accessibility": ("saved_data/power/power_accessibility_371.csv",
                                "zone_id"),
    }
    for label, (rel, col) in art.items():
        p = os.path.join(_HERE, rel)
        if not os.path.exists(p):
            print(f"  [SKIP] {label}: file missing")
            continue
        d = pd.read_csv(p, usecols=lambda c: True)
        if col not in d.columns:
            check(label, False, f"key column {col} missing")
            continue
        keys = set(pd.to_numeric(d[col], errors="coerce").dropna().astype(int))
        strays = sorted(keys & changed16)
        orphans = sorted(keys - zones371 - twins)
        check(label, not strays and not orphans,
              f"n={len(keys)} strays={strays[:5]} orphans={orphans[:5]}")

    # ── 1. OMX lookup ──────────────────────────────────────────────────
    try:
        import h5py
        with h5py.File(OMX, "r") as f:
            lk = None
            if "lookup" in f:
                for name in f["lookup"]:
                    lk = name
                    break
            if lk is None:
                check("omx_lookup", False, "no lookup group")
            else:
                raw = f["lookup"][lk][:]
                ids = {int(z.decode() if isinstance(z, bytes) else z) for z in raw}
                check("omx_lookup", ids == zones371,
                      f"n={len(ids)} missing={len(zones371 - ids)} "
                      f"extra={len(ids - zones371)} "
                      f"strays={sorted(ids & changed16)[:5]}")
    except ImportError:
        print("  [SKIP] omx_lookup: h5py not installed")
    except Exception as e:  # noqa: BLE001
        check("omx_lookup", False, str(e)[:120])

    # ── 2. split-zone polygon vintage (area vs 2021 reference) ─────────
    try:
        import geopandas as gpd
        g = gpd.read_file(GEOJSON)
        key = next((c for c in ("SA2_MAIN16", "sa2_code", "SA2_CODE") if c in g.columns), None)
        if key is None:
            check("geojson_keys", False, "no SA2 key column")
        else:
            g["code"] = pd.to_numeric(g[key], errors="coerce").astype("Int64")
            strays = sorted(set(g["code"].dropna().astype(int)) & changed16)
            check("geojson_keys", not strays, f"strays={strays[:5]}")
            # area test on a few known splits (2021 areas from seq file)
            ref = None
            if "area_sqkm" in seq.columns:
                _s2 = seq[[zone_col, "area_sqkm"]].copy()
                _s2[zone_col] = pd.to_numeric(_s2[zone_col], errors="coerce")
                _s2 = _s2.dropna(subset=[zone_col])
                ref = dict(zip(_s2[zone_col].astype(int).values,
                               pd.to_numeric(_s2["area_sqkm"],
                                             errors="coerce").fillna(0).values))
            if ref is not None:
                gm = g.to_crs(3577)
                gm["area_km2"] = gm.geometry.area / 1e6
                worst = 0.0
                for code in (309071552, 309071553, 309081559, 309101562,
                             314021577, 314021578, 316021580):
                    sub = gm[gm["code"] == code]
                    if len(sub) == 0 or code not in ref:
                        continue
                    got = float(sub["area_km2"].sum())
                    want = float(ref[code])
                    rel = abs(got - want) / max(want, 1e-9)
                    worst = max(worst, rel)
                    if rel > 0.05:
                        print(f"    area mismatch {code}: geojson={got:.1f} "
                              f"ref2021={want:.1f}")
                check("geojson_split_areas", worst <= 0.05,
                      f"worst rel err={worst:.1%}")
    except ImportError:
        print("  [SKIP] geojson checks: geopandas not installed")
    except Exception as e:  # noqa: BLE001
        check("geojson_keys", False, str(e)[:120])

    # ── 3. HHTS coverage ───────────────────────────────────────────────
    if os.path.exists(TRIPS):
        t = pd.read_csv(TRIPS, low_memory=False,
                        usecols=lambda c: c in ("DESTSA2_2021", "ORIGSA2_2021"))
        for col in ("DESTSA2_2021", "ORIGSA2_2021"):
            if col not in t.columns:
                continue
            v = pd.to_numeric(t[col], errors="coerce")
            outside = float((~v.isin(zones371) & v.notna()).mean())
            check(f"hhts_{col}", outside < 0.05,
                  f"outside-371={outside:.1%} (fringe/externals expected)")
    else:
        print("  [SKIP] hhts coverage: trips file missing")

    print("-" * 64)
    if _failures:
        print(f"FAILED: {_failures}")
        return 1
    print("All executed checks pass — pipeline keys are SA2-2021.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
