"""
09_build_landuse_507.py
Aggregate SEQ-STM employment/landuse (land_use_resciacre_test.csv, 2062 EMME
zones) up to all 507 SA2 zones, using the same EMME->SA2 crosswalk and
nearest-neighbor imputation pattern as 04_aggregate_skims.py.

For the ~330 zones already present in the ABM's existing
zone_order_and_landuse.csv (test_runs_tours/code), the activity-purpose and
OSM-landuse-category columns that have no traceable source in this codebase
(EATOUT, Shopping, airport..village_green, etc.) are carried forward
unchanged via a SA2-code merge. For the ~177 newly-added zones those columns
are left blank (NaN) — there is no existing pipeline to derive them, so they
are reported as a gap rather than guessed at.

Output: Z:\\test_runs_tours\\new_pop\\saved_data\\zone_landuse_507.csv,
keyed directly by SA2_MAIN16 (no ORDER column).
"""
import os
import re
import glob
import numpy as np
import pandas as pd

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
LEGACY_LANDUSE = r"Z:\test_runs_tours\code\zone_order_and_landuse.csv"
SOURCE_LANDUSE = (r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM"
                   r"\#Model_Masters\ActivitySim\asim_model_inputs\land_use_resciacre_test.csv")
ZONE_LIST_507 = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OUT_DIR = r"Z:\test_runs_tours\new_pop\saved_data"
OUT_PATH = os.path.join(OUT_DIR, "zone_landuse_507.csv")
NEW_POP_ZONE_DIR = os.path.join(OUT_DIR, "zone_ids_sa2")

# Additive ActivitySim landuse columns -> output column name (renamed where
# the legacy file uses a different name, e.g. OTHEMPN -> OTHEMP).
SOURCE_TO_OUT = {
    "TOTHH": "TOTHH", "TOTPOP": "TOTPOP", "AGE0519": "AGE0519",
    "HSENROLL": "HSENROLL", "AGREMPN": "AGREMPN", "MWTEMPN": "MWTEMPN",
    "OTHEMPN": "OTHEMP", "RETEMPN": "RETEMPN", "FPSEMPN": "FPSEMPN",
    "HEREMPN": "HEREMPN", "TOTEMP": "TOTEMP",
}
ADDITIVE_OUT_COLS = list(SOURCE_TO_OUT.values())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(ZONE_LIST_507) as f:
        zone_ids_507 = sorted(int(x) for x in re.split(r"[\s,]+", f.read().strip()) if x)
    n_sa2 = len(zone_ids_507)
    sa2_to_idx = {code: i for i, code in enumerate(zone_ids_507)}
    print(f"Target SA2 zones: {n_sa2}")

    src = pd.read_csv(SOURCE_LANDUSE)
    src["emme_zone_id"] = src["numpy_index_plus_one"].astype(int)
    n_emme = len(src)
    print(f"Source EMME-zone landuse rows: {n_emme}")

    crosswalk = pd.read_csv(os.path.join(DEMAND_DIR, "sa2_crosswalk_507.csv"))
    crosswalk["sa2_code"] = crosswalk["sa2_code"].astype(int)
    crosswalk = crosswalk.set_index("sa2_code").reindex(zone_ids_507).reset_index()

    expanded = pd.read_csv(os.path.join(DEMAND_DIR, "sa2_crosswalk_507_expanded.csv"))
    expanded["sa2_code"] = expanded["sa2_code"].astype(int)
    expanded["emme_zone_id"] = expanded["emme_zone_id"].astype(int)
    emme_to_sa2 = dict(zip(expanded["emme_zone_id"], expanded["sa2_code"]))

    agg_sa2_idx = np.full(n_emme, -1, dtype=np.int32)
    for i, emme_id in enumerate(src["emme_zone_id"].values):
        sa2_code = emme_to_sa2.get(int(emme_id))
        if sa2_code is not None and sa2_code in sa2_to_idx:
            agg_sa2_idx[i] = sa2_to_idx[sa2_code]

    mapped = agg_sa2_idx >= 0
    print(f"Mapped EMME zones: {mapped.sum()}/{n_emme}")

    out = pd.DataFrame({"SA2_MAIN16": zone_ids_507})
    for col in ADDITIVE_OUT_COLS:
        out[col] = 0.0

    src_cols = [c for c in SOURCE_TO_OUT]
    sums = (
        pd.DataFrame(src.loc[mapped, src_cols].values, columns=src_cols)
        .assign(sa2_idx=agg_sa2_idx[mapped])
        .groupby("sa2_idx")[src_cols].sum()
    )
    for src_col, out_col in SOURCE_TO_OUT.items():
        out.loc[sums.index, out_col] = sums[src_col].values

    has_emme = crosswalk["n_emme_zones"].fillna(0).astype(int) > 0
    coords = crosswalk[["sa2_lat", "sa2_lon"]].values.astype(float)
    mapped_sa2 = np.where(has_emme)[0]
    unmapped_sa2 = np.where(~has_emme)[0]
    print(f"SA2 zones with direct EMME coverage: {has_emme.sum()}/{n_sa2}")
    print(f"SA2 zones needing nearest-neighbor imputation: {len(unmapped_sa2)}")

    for u_idx in unmapped_sa2:
        if np.isnan(coords[u_idx, 0]) or len(mapped_sa2) == 0:
            continue
        dists = np.sqrt(
            (coords[mapped_sa2, 0] - coords[u_idx, 0]) ** 2
            + (coords[mapped_sa2, 1] - coords[u_idx, 1]) ** 2
        )
        nn_idx = mapped_sa2[np.argmin(dists)]
        out.loc[u_idx, ADDITIVE_OUT_COLS] = out.loc[nn_idx, ADDITIVE_OUT_COLS].values

    # sa2_area from the crosswalk's official area_sqkm (more accurate than the
    # legacy TAZ-acreage-derived sa2_area column).
    out["sa2_area"] = crosswalk["area_sqkm"].values
    out["POPULATION"] = out["TOTPOP"]

    # Carry forward legacy activity-purpose / OSM-landuse-category columns for
    # zones that already had them; leave NaN (and report) for new zones.
    legacy = pd.read_csv(LEGACY_LANDUSE)
    legacy_extra_cols = [c for c in legacy.columns
                         if c not in ("ORDER", "SA2_MAIN16") and c not in out.columns]
    legacy_subset = legacy[["SA2_MAIN16"] + legacy_extra_cols].copy()
    legacy_subset["SA2_MAIN16"] = legacy_subset["SA2_MAIN16"].astype(int)

    out = out.merge(legacy_subset, on="SA2_MAIN16", how="left")

    n_gap = out[legacy_extra_cols[0]].isna().sum() if legacy_extra_cols else 0
    print(f"\nZones with no legacy activity-purpose/OSM-category data "
          f"({len(legacy_extra_cols)} columns: {legacy_extra_cols[:5]}...): {n_gap}/{n_sa2}")

    # ── Fill the gap zones from the fresh OSM POI counts ───────────────────
    # 10_build_osm_counts_507.py fetches per-zone POI tag-value counts for
    # the FULL zone list (SA2_2016_OSM_counts_507.csv), so zones the legacy
    # 330-zone file never covered (Toowoomba SA4 317, NSW Richmond Valley,
    # the 2021-vintage infill SA2s) get real attractor data instead of NaN→0.
    _osm_path = os.path.join(OUT_DIR_NEWPOP, "SA2_2016_OSM_counts_507.csv") \
        if 'OUT_DIR_NEWPOP' in dir() else \
        os.path.join(os.path.dirname(OUT_PATH), "SA2_2016_OSM_counts_507.csv")
    if os.path.exists(_osm_path) and n_gap:
        osm = pd.read_csv(_osm_path)
        osm["SA2_MAIN16"] = osm["SA2_MAIN16"].astype(int)
        osm = osm.set_index("SA2_MAIN16")

        def _osm_sum(row_zone, cols):
            if row_zone not in osm.index:
                return np.nan
            vals = [osm.at[row_zone, c] for c in cols if c in osm.columns]
            return float(np.nansum(vals)) if vals else np.nan

        # Composite activity-purpose columns approximated from standard
        # POI groupings; direct-name columns (park, sport, retail, ...)
        # copy straight across. Conservative: only fills where legacy is NaN.
        _COMPOSites = {
            'EATOUT':      ['cafe', 'restaurant', 'fast_food', 'pub', 'bar', 'food_court'],
            'Shopping':    ['supermarket', 'mall', 'department_store', 'convenience',
                            'clothes', 'bakery', 'butcher', 'greengrocer', 'chemist'],
            'Entertainment': ['cinema', 'theatre', 'nightclub', 'arts_centre', 'casino'],
            'Healthcare':  ['hospital', 'clinic', 'doctors', 'dentist', 'pharmacy',
                            'physiotherapist'],
            'Education':   ['school', 'university', 'college', 'language_school'],
            'ESCORT':      ['school', 'kindergarten', 'childcare'],
            'childcare':   ['kindergarten', 'childcare'],
            'Public_Services': ['townhall', 'courthouse', 'police', 'fire_station',
                                'library', 'post_office'],
            'OTHERMAINTENANCE': ['bank', 'car_repair', 'laundry', 'post_office',
                                 'fuel', 'atm'],
            'OTHERDISCRFacilities': ['place_of_worship', 'community_centre',
                                     'sports_centre', 'fitness_centre', 'swimming_pool'],
        }
        _gap_mask = out[legacy_extra_cols[0]].isna()
        _n_filled_cols = 0
        for col in legacy_extra_cols:
            if col in ('SA4_NAME16',):
                continue
            if col in _COMPOSites:
                srcs = _COMPOSites[col]
            elif col in osm.columns:
                srcs = [col]
            else:
                continue
            fill_vals = out.loc[_gap_mask, "SA2_MAIN16"].map(
                lambda z, s=srcs: _osm_sum(z, s))
            out.loc[_gap_mask, col] = out.loc[_gap_mask, col].fillna(fill_vals)
            _n_filled_cols += 1
        # tally against a column the fill can actually populate (the first
        # legacy col is school enrollment, which has no OSM equivalent)
        _tally_col = next((c for c in ('EATOUT', 'park', 'Shopping')
                           if c in out.columns), None)
        _still_gap = (int(out.loc[_gap_mask, _tally_col].isna().sum())
                      if _tally_col else -1)
        print(f"OSM gap-fill: {_n_filled_cols} columns filled for "
              f"{int(_gap_mask.sum())} gap zones from {os.path.basename(_osm_path)} "
              f"(zones still lacking OSM attractors: {_still_gap})")

    # ── Reconcile TOTHH/TOTPOP with the actual exported synthetic population ──
    # The SEQ-STM landuse source is a different model vintage/scenario (see
    # plan caveat) and its household/population totals don't exactly match
    # the synth pop the ABM will actually run on. Overwrite TOTHH/TOTPOP (and
    # the POPULATION alias) with real per-zone counts from the new_pop
    # zone_<sa2>.csv exports so landuse-driven accessibility/destination-choice
    # terms stay internally consistent with the population itself. Employment/
    # school-enrollment columns are left as-is — synth pop has no equivalent.
    out = reconcile_hh_pop_with_synth(out, NEW_POP_ZONE_DIR)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}  ({len(out)} zones, {len(out.columns)} columns)")


def reconcile_hh_pop_with_synth(out, zone_dir):
    """Overwrite TOTHH/TOTPOP/POPULATION with real counts from the exported
    new_pop zone_<sa2>.csv files (person count = rows, household count =
    nunique(abshid)). Zones with no exported synth pop yet (e.g. the 4
    HPC-backfill gap zones) keep their SEQ-STM-derived values, flagged below.
    """
    sa2_counts = {}
    for fp in glob.glob(os.path.join(zone_dir, "zone_*.csv")):
        sa2 = int(re.search(r"zone_(\d+)\.csv", os.path.basename(fp)).group(1))
        df = pd.read_csv(fp, usecols=lambda c: c in ("abshid", "abspid"))
        n_persons = len(df)
        n_hh = df["abshid"].nunique() if "abshid" in df.columns else np.nan
        sa2_counts[sa2] = (n_persons, n_hh)

    out["TOTPOP"] = out["SA2_MAIN16"].map(lambda z: sa2_counts.get(z, (np.nan, np.nan))[0]).combine_first(out["TOTPOP"])
    out["TOTHH"] = out["SA2_MAIN16"].map(lambda z: sa2_counts.get(z, (np.nan, np.nan))[1]).combine_first(out["TOTHH"])
    out["POPULATION"] = out["TOTPOP"]

    n_reconciled = sum(1 for z in out["SA2_MAIN16"] if z in sa2_counts)
    n_unreconciled = len(out) - n_reconciled
    print(f"\nTOTHH/TOTPOP reconciled against synth pop for {n_reconciled}/{len(out)} zones "
          f"(remaining {n_unreconciled} keep SEQ-STM-derived values — no synth pop exported yet, "
          f"e.g. the HPC-backfill gap zones)")
    return out


if __name__ == "__main__":
    main()
