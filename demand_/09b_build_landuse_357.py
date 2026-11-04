"""
09b_build_landuse_357.py
Rebuild per-SA2 land use for the corrected 357-zone SEQ + Tweed Shire study
area (see sa2_zones_reference_357.csv - built 2026-07 to replace the stale
324-zone list, which was missing Toowoomba (SA4 317) and over/under-scoped
the NSW side).

Employment/parcel-area source of truth (unchanged from 09_build_landuse_507.py):
  land_use_resciacre_test.csv - 2062-zone SEQ-STM EMME land use, workplace-
  based (i.e. jobs located AT each zone - the right variable for a destination
  -choice work-attraction term), aggregated up to SA2 via the existing EMME-
  zone-to-SA2 crosswalk. Covers 308 of the 357 zones (see in_emme_network flag
  in sa2_zones_reference_357.csv) - essentially the old ~330-zone TAZ
  footprint plus a couple of zones outside it that happen to have EMME
  representation already (e.g. Tweed Heads / Tweed Heads South).

For the 49 zones with NO EMME representation (all 15 Toowoomba SA2s + 6 of
the 8 Tweed Valley SA2s + a handful of others at the edge of the old network),
there is no local workplace-based employment/school-capacity data, so this
script fills them from real ABS 2021 Census SA2-level tables instead of
nearest-neighbor guessing:
  - Employment by type: sa2indp.csv (Industry of Employment, 1-digit ANZSIC).
    This is RESIDENCE-based (where employed people live, not where the job
    is), which is a real, honestly-flagged approximation - not the same
    variable as the EMME source, but the best real data available locally,
    and reasonable for a self-contained regional city like Toowoomba where
    most residents' jobs are also local.
  - Education by type: sa2typp.csv (Type of Educational Institution
    Attending). Also residence-based (where students live, not school
    location/capacity) - same caveat as above, flagged in the output.
  - Parcel land area (RESACRE/CIACRE/TOTACRE): no equivalent ABS source is
    available locally (would need a land-use/cadastre dataset e.g. QLUMP,
    not present in this repo) - filled by nearest-neighbour from the nearest
    EMME-covered zone, same method the old 507-script used for ALL gap
    columns. Flagged with parcel_area_is_nn=True so it's not mistaken for
    real data.

Output: Z:\\test_runs_tours\\new_pop\\saved_data\\zone_landuse_357.csv
keyed by SA2_MAIN16.
"""
import csv
import os
import re
import numpy as np
import pandas as pd

# Portable paths: derive from this script's location so the job runs on the
# Linux HPC (paths under ~/) as well as the Windows Z:\ checkout. The former
# hardcoded Z:\ / z:\ paths only existed on Windows and broke every HPC run
# (FileNotFoundError on sa2_zones_reference_357.csv). External roots
# (activity_sim_resources, Paper_Plan) are siblings of test_runs_tours under the
# home root, and are env-overridable.
DEMAND_DIR = os.path.dirname(os.path.abspath(__file__))
_TRT_ROOT  = os.path.normpath(os.path.join(DEMAND_DIR, ".."))
_HOME_ROOT = os.path.normpath(os.path.join(_TRT_ROOT, ".."))
ZONE_REF_357 = os.path.join(DEMAND_DIR, "sa2_zones_reference_357.csv")
SOURCE_LANDUSE = os.environ.get("SOURCE_LANDUSE") or os.path.join(
    _HOME_ROOT, "activity_sim_resources", "new_asim", "SEQ-STM-ASIM", "SEQ-STM-ASIM",
    "#Model_Masters", "ActivitySim", "asim_model_inputs", "land_use_resciacre_test.csv")
EXPANDED_CROSSWALK = os.path.join(DEMAND_DIR, "sa2_crosswalk_507_expanded.csv")
LEGACY_LANDUSE = os.path.join(_TRT_ROOT, "code", "zone_order_and_landuse.csv")
CENSUS_CONTROLS_DIR = os.environ.get("CENSUS_CONTROLS_DIR") or os.path.join(
    _HOME_ROOT, "Paper_Plan", "synth_pop", "data_sa2_controls")
OUT_DIR = os.path.join(_TRT_ROOT, "new_pop", "saved_data")
OUT_PATH = os.path.join(OUT_DIR, "zone_landuse_357.csv")

# EMME-source employment columns pulled straight through, PLUS the parcel-area
# and tertiary-enrolment columns the old 09_build_landuse_507.py silently
# dropped even though the source file has them.
SOURCE_TO_OUT = {
    "TOTHH": "TOTHH", "TOTPOP": "TOTPOP", "AGE0519": "AGE0519",
    "HSENROLL": "HSENROLL", "AGREMPN": "AGREMPN", "MWTEMPN": "MWTEMPN",
    "OTHEMPN": "OTHEMP", "RETEMPN": "RETEMPN", "FPSEMPN": "FPSEMPN",
    "HEREMPN": "HEREMPN", "TOTEMP": "TOTEMP",
    "RESACRE": "RESACRE", "CIACRE": "CIACRE", "TOTACRE": "TOTACRE",
    "COLLFTE": "COLLFTE", "COLLPTE": "COLLPTE",
}
ADDITIVE_OUT_COLS = list(SOURCE_TO_OUT.values())

# ANZSIC 1-digit (sa2indp.csv) -> EMME employment category. Standard SEQ-STM
# grouping: FPS=Finance/Professional/Services, HER=Health/Education/Recreation,
# MWT=Manufacturing/Wholesale/Transport (+construction/mining/utilities).
INDP_TO_EMPCAT = {
    "Agriculture, Forestry and Fishing": "AGREMPN",
    "Manufacturing": "MWTEMPN",
    "Wholesale Trade": "MWTEMPN",
    "Transport, Postal and Warehousing": "MWTEMPN",
    "Construction": "MWTEMPN",
    "Mining": "MWTEMPN",
    "Electricity, Gas, Water and Waste Services": "MWTEMPN",
    "Retail Trade": "RETEMPN",
    "Accommodation and Food Services": "RETEMPN",
    "Financial and Insurance Services": "FPSEMPN",
    "Professional, Scientific and Technical Services": "FPSEMPN",
    "Information Media and Telecommunications": "FPSEMPN",
    "Rental, Hiring and Real Estate Services": "FPSEMPN",
    "Administrative and Support Services": "FPSEMPN",
    "Public Administration and Safety": "FPSEMPN",
    "Health Care and Social Assistance": "HEREMPN",
    "Education and Training": "HEREMPN",
    "Arts and Recreation Services": "HEREMPN",
    "Other Services": "HEREMPN",
    "Inadequately described": "OTHEMP",
}
TYPP_TO_EDUCAT = {
    "Primary - Government": "Primary",
    "Primary - Non Government": "Primary",
    "Secondary - Government": "Secondary",
    "Secondary - Non-Government": "Secondary",
    "Preschool": "Preschool",
    "Tertiary - Non University": "TAFE_college",
    "Tertiary - University or other higher education": "University",
}


def _read_abs_sa2_table(path):
    """Parse an ABS TableBuilder SA2-by-category export (junk header rows,
    'Person Records','<sa2>','<category>',<count> data rows)."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = next(csv.reader([line]))
            if len(parts) >= 4 and re.match(r"^\d{9}$", parts[1].strip()):
                rows.append((parts[1].strip(), parts[2].strip(), parts[3].strip()))
    df = pd.DataFrame(rows, columns=["sa2", "category", "count"])
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    zones = pd.read_csv(ZONE_REF_357, dtype={"sa2_code": str})
    zone_ids_357 = sorted(int(z) for z in zones["sa2_code"])
    n_sa2 = len(zone_ids_357)
    sa2_to_idx = {code: i for i, code in enumerate(zone_ids_357)}
    print(f"Target SA2 zones: {n_sa2}")

    # -- 1. EMME-sourced employment/parcel-area, aggregated to SA2 ----------
    src = pd.read_csv(SOURCE_LANDUSE)
    src["emme_zone_id"] = src["numpy_index_plus_one"].astype(int)
    n_emme = len(src)
    print(f"Source EMME-zone landuse rows: {n_emme}")

    expanded = pd.read_csv(EXPANDED_CROSSWALK)
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

    out = pd.DataFrame({"SA2_MAIN16": zone_ids_357})
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

    # has_emme starts from the sa2_zones_reference_357.csv flag (a name-match
    # against lol.csv's SA2_NAME column), but the actual aggregation above
    # goes through a SEPARATE numeric emme_zone_id<->sa2_code crosswalk
    # (sa2_crosswalk_507_expanded.csv) that may not agree zone-for-zone (e.g.
    # Gatton: name-matched True, but its EMME zones never landed in that
    # crosswalk, so TOTHH/TOTPOP/TOTEMP all aggregated to 0). Any zone with
    # TOTHH+TOTPOP+TOTEMP==0 after aggregation demonstrably got nothing from
    # the EMME source regardless of what the flag says, so route those to the
    # ABS-census fallback too rather than silently keeping them at zero.
    name_match_emme = np.array([str(zones.iloc[i]["in_emme_network"]) == "True" for i in range(n_sa2)])
    got_real_data = (out["TOTHH"] + out["TOTPOP"] + out["TOTEMP"]).to_numpy() > 0
    has_emme = name_match_emme & got_real_data
    n_flag_mismatch = int((name_match_emme & ~got_real_data).sum())
    if n_flag_mismatch:
        print(f"NOTE: {n_flag_mismatch} zones flagged in_emme_network=True but got no "
              f"aggregated data (crosswalk mismatch, e.g. Gatton) - routed to ABS fallback below")
    print(f"SA2 zones with direct EMME coverage: {has_emme.sum()}/{n_sa2}")
    gap_idx = np.where(~has_emme)[0]
    print(f"SA2 zones needing ABS-census apportionment: {len(gap_idx)}")

    # -- 1b. Carry forward legacy OSM-landuse/activity-purpose/school columns
    # (Primary, Secondary, university, college, Education, etc.) for zones
    # that already had them. Done BEFORE the TYPP/INDP fills below so those
    # fills only apply where this merge left a real gap (NaN), not
    # unconditionally - otherwise zones with good legacy data (e.g. Tweed
    # Heads, which has EMME employment coverage but was never in the old
    # 330-zone legacy file) would wrongly be skipped just for having
    # in_emme_network=True.
    legacy = pd.read_csv(LEGACY_LANDUSE)
    legacy_extra_cols = [c for c in legacy.columns
                         if c not in ("ORDER", "SA2_MAIN16") and c not in out.columns]
    legacy_subset = legacy[["SA2_MAIN16"] + legacy_extra_cols].copy()
    legacy_subset["SA2_MAIN16"] = legacy_subset["SA2_MAIN16"].astype(int)
    out = out.merge(legacy_subset, on="SA2_MAIN16", how="left")
    n_gap_legacy = out[legacy_extra_cols[0]].isna().sum() if legacy_extra_cols else 0
    print(f"Zones with no legacy OSM-landuse/activity-purpose data "
          f"({len(legacy_extra_cols)} columns): {n_gap_legacy}/{n_sa2}")

    # -- 1c. Real population/household counts for EMME-gap zones (ABS Census
    # sex-by-SA2 / tenure-by-SA2), since the EMME source only covers 308/357
    # zones and TOTPOP/TOTHH would otherwise silently read 0 for the other 49.
    # sa2sexp.csv only has 349/569 possible SA2 codes (it predates the
    # SEQ_SA4_PREFIXES fix and was extracted under the old, narrower/wrong
    # 301-311 numeric range) - sa2indp.csv has full coverage (569, unfiltered
    # statewide+Tweed), so its ALL-CATEGORY total (incl. "Not applicable" /
    # "Not stated", i.e. every enumerated person, not just the employed) is
    # used as a fallback population count where sexp has no row. Cross-checked
    # against sexp on zones with both: within ~1-2% (ABS randomised cell
    # perturbation), close enough for an interim zone-total population count.
    sexp = _read_abs_sa2_table(os.path.join(CENSUS_CONTROLS_DIR, "sa2sexp.csv"))
    pop_by_sa2 = sexp.groupby("sa2")["count"].sum()
    indp_all = _read_abs_sa2_table(os.path.join(CENSUS_CONTROLS_DIR, "sa2indp.csv"))
    pop_by_sa2_fallback = indp_all.groupby("sa2")["count"].sum()
    tend = _read_abs_sa2_table(os.path.join(CENSUS_CONTROLS_DIR, "sa2tend.csv"))
    hh_by_sa2 = tend.groupby("sa2")["count"].sum()
    n_pop_filled, n_pop_fallback = 0, 0
    for i in gap_idx:
        sa2 = str(zone_ids_357[i])
        if sa2 in pop_by_sa2.index:
            out.loc[i, "TOTPOP"] = pop_by_sa2.loc[sa2]
            n_pop_filled += 1
        elif sa2 in pop_by_sa2_fallback.index:
            out.loc[i, "TOTPOP"] = pop_by_sa2_fallback.loc[sa2]
            n_pop_filled += 1
            n_pop_fallback += 1
        if sa2 in hh_by_sa2.index:
            out.loc[i, "TOTHH"] = hh_by_sa2.loc[sa2]
    print(f"TOTPOP apportioned from ABS Census for {n_pop_filled}/{len(gap_idx)} gap zones "
          f"({n_pop_fallback} via the sa2indp all-category fallback, rest via sa2sexp)")

    # -- 2. Employment-by-type for gap zones: ABS INDP (residence-based) ----
    indp = _read_abs_sa2_table(os.path.join(CENSUS_CONTROLS_DIR, "sa2indp.csv"))
    indp["empcat"] = indp["category"].map(INDP_TO_EMPCAT)
    indp_emp = (indp.dropna(subset=["empcat"])
                .groupby(["sa2", "empcat"])["count"].sum().unstack(fill_value=0.0))
    emp_cats = ["AGREMPN", "MWTEMPN", "RETEMPN", "FPSEMPN", "HEREMPN", "OTHEMP"]
    for c in emp_cats:
        if c not in indp_emp.columns:
            indp_emp[c] = 0.0

    n_indp_filled = 0
    for i in gap_idx:
        sa2 = str(zone_ids_357[i])
        if sa2 in indp_emp.index:
            row = indp_emp.loc[sa2]
            for c in emp_cats:
                out.loc[i, c] = row[c]
            out.loc[i, "TOTEMP"] = row[emp_cats].sum()
            n_indp_filled += 1
    print(f"Employment apportioned from ABS INDP (residence-based, flagged): "
          f"{n_indp_filled}/{len(gap_idx)} gap zones")

    # -- 3. Education-by-type for ALL zones: ABS TYPP (residence-based) -----
    # Kept SEPARATE from the EMME-sourced HSENROLL/Primary/Secondary columns
    # rather than overwriting them - EMME's Primary/Secondary represents
    # school PLACES located at the zone (destination-side, the right variable
    # for a destination-choice attraction term); TYPP is where students LIVE
    # (origin-side). Exposed as *_typp_resident columns so callers can choose;
    # only used to FILL Primary/Secondary/tertiary counts where the EMME
    # source has no data at all (gap zones).
    typp = _read_abs_sa2_table(os.path.join(CENSUS_CONTROLS_DIR, "sa2typp.csv"))
    typp["educat"] = typp["category"].map(TYPP_TO_EDUCAT)
    typp_edu = (typp.dropna(subset=["educat"])
                .groupby(["sa2", "educat"])["count"].sum().unstack(fill_value=0.0))
    edu_cats = ["Preschool", "Primary", "Secondary", "TAFE_college", "University"]
    for c in edu_cats:
        if c not in typp_edu.columns:
            typp_edu[c] = 0.0

    for c in edu_cats:
        out[f"{c}_typp_resident"] = 0.0
    n_typp_filled = 0
    for i in range(n_sa2):
        sa2 = str(zone_ids_357[i])
        if sa2 in typp_edu.index:
            row = typp_edu.loc[sa2]
            for c in edu_cats:
                out.loc[i, f"{c}_typp_resident"] = row[c]
            n_typp_filled += 1
    print(f"ABS TYPP (education-by-type, residence-based) attached for {n_typp_filled}/{n_sa2} zones")

    # Ensure Primary/Secondary/university/college exist as columns (legacy
    # names used by stage5_destination_choice.py's SIZE_TERM_COEFFICIENTS).
    # Fill from TYPP wherever the legacy merge above left them missing (NaN) -
    # this is NOT the same set as gap_idx: e.g. Tweed Heads has EMME
    # employment coverage (in_emme_network=True) but was never part of the
    # old 330-zone legacy TAZ file, so it has NaN Primary/Secondary too.
    for legacy_col, typp_col in [("Primary", "Primary_typp_resident"),
                                  ("Secondary", "Secondary_typp_resident"),
                                  ("university", "University_typp_resident"),
                                  ("college", "TAFE_college_typp_resident")]:
        if legacy_col not in out.columns:
            out[legacy_col] = np.nan
        missing = out[legacy_col].isna()
        out.loc[missing, legacy_col] = out.loc[missing, typp_col]
        out[legacy_col] = out[legacy_col].fillna(0.0)

    # -- 4. Parcel area (RESACRE/CIACRE/TOTACRE) for gap zones: nearest-neighbour --
    # No local ABS/QLUMP land-use-by-parcel source exists for these zones -
    # flagged explicitly rather than silently presented as real.
    coords = zones[["centroid_lat", "centroid_lon"]].values.astype(float)
    mapped_idx = np.where(has_emme)[0]
    out["parcel_area_is_nn"] = False
    for u_idx in gap_idx:
        if len(mapped_idx) == 0:
            continue
        dists = np.sqrt(
            (coords[mapped_idx, 0] - coords[u_idx, 0]) ** 2
            + (coords[mapped_idx, 1] - coords[u_idx, 1]) ** 2
        )
        nn_idx = mapped_idx[np.argmin(dists)]
        for c in ("RESACRE", "CIACRE", "TOTACRE"):
            out.loc[u_idx, c] = out.loc[nn_idx, c]
        out.loc[u_idx, "parcel_area_is_nn"] = True

    # -- 1d. Fill REMAINING NaN legacy/QSpatial columns from the 507 build --
    # The 60 universe zones missing from the 330-zone legacy file (step 1b
    # LEFT join) would otherwise keep NaN QSpatial/activity/area cells, which
    # downstream fillna(0) then silently zeroes — wiping real attractors such
    # as recreation_ground for Toowoomba/Tweed/fringe gap zones (seen as a
    # degenerate all-zero panel in accessibility_vs_landuse.png, job s5check).
    # zone_landuse_507.csv covers the same 371 SA2 keys with no gaps and was
    # verified unit-identical to the 357 values on the 311-zone overlap
    # (2026-09: median ratio 1.0, corr 1.0 for every QSpatial/activity
    # column), so it is a safe same-unit donor. Provenance of the donor's own
    # gap values: the 507 build fills ITS gaps from fresh OSM POI counts
    # (09_build_landuse_507.py OSM gap-fill — direct tag counts for park/
    # sport/retail, composites for EATOUT/Shopping/etc.), i.e. real local
    # data, not nearest-neighbour fabrication — verified 2026-09 (e.g.
    # Wilsonton park=11 equals its OSM park POI count). Employment/pop/education have
    # their own EMME/Census/TYPP fills above — 507 carries different vintages
    # there (corr 0.3-0.5) — so those columns are NEVER touched here.
    # sa2_area converts km²→m² (×1e6, ratio verified 1e6 on the overlap).
    _FILL_507_PATH = os.path.join(OUT_DIR, "zone_landuse_507.csv")
    _FILL_507_MANAGED = {
        "TOTHH", "TOTPOP", "POPULATION", "AGE0519", "HSENROLL",
        "AGREMPN", "MWTEMPN", "OTHEMP", "RETEMPN", "FPSEMPN", "HEREMPN",
        "TOTEMP", "RESACRE", "CIACRE", "TOTACRE", "COLLFTE", "COLLPTE",
        "Primary", "Secondary", "university", "college",
        "Preschool_typp_resident", "Primary_typp_resident",
        "Secondary_typp_resident", "TAFE_college_typp_resident",
        "University_typp_resident", "parcel_area_is_nn",
    }
    out["landuse_is_507fill"] = False
    try:
        _f507 = pd.read_csv(_FILL_507_PATH)
        _f507["SA2_MAIN16"] = _f507["SA2_MAIN16"].astype(int)
        _f507 = _f507.drop_duplicates(subset=["SA2_MAIN16"]).set_index("SA2_MAIN16")
        _n_507_zones, _n_507_cols = 0, 0
        for _c in legacy_extra_cols:
            if _c in _FILL_507_MANAGED or _c not in _f507.columns:
                continue
            _m = out[_c].isna()
            if not bool(_m.any()):
                continue
            _donor = pd.to_numeric(_f507[_c], errors="coerce").reindex(
                out["SA2_MAIN16"].astype(int).values)
            # reindex keys by SA2 code → reset to positional alignment with out
            _donor = pd.Series(_donor.values, index=out.index)
            if _c == "sa2_area":
                _donor = _donor * 1e6  # 507 stores km²; 357 stores m²
            _got = _m & _donor.notna()
            if bool(_got.any()):
                out.loc[_got, _c] = _donor[_got]
                out.loc[_got, "landuse_is_507fill"] = True
                _n_507_cols += 1
                print(f"  507-fill {_c}: {int(_got.sum())} zone(s)")
        _n_507_rows = int(out["landuse_is_507fill"].sum())
        print(f"507-backfill: {_n_507_cols} column(s) touched, "
              f"{_n_507_rows}/{n_sa2} zones carry ≥1 507-filled cell "
              f"(flagged landuse_is_507fill=True)")
        _still_nan = [c for c in legacy_extra_cols
                      if c not in _FILL_507_MANAGED and out[c].isna().any()]
        if _still_nan:
            # String columns (LandUse/SA4_NAME16/city) are never backfilled —
            # report numeric gaps separately so one unfillable label column
            # doesn't implicate all 60 gap zones. A zone is listed only if it
            # is STILL NaN somewhere the donor actually covers it (donor-wide
            # gaps such as *_total_priv, which 507 lacks for every gap zone,
            # are a donor limitation, not a per-zone surprise).
            _str_cols = {"LandUse", "SA4_NAME16", "city"}
            _still_num = [c for c in _still_nan if c not in _str_cols]
            # Gap universe = zones that needed any fill (sa2_area filled all
            # 60, so the flag marks exactly the legacy-gap set). A column
            # counts for the zone report only if the donor covers ≥1 gap
            # zone in it — donor-wide gaps (*_total_priv, covered nowhere)
            # are reported via the column NOTE above, not per zone.
            _gap = out["landuse_is_507fill"]
            _still_mask = pd.Series(False, index=out.index)
            _donor_gap_mask = pd.Series(False, index=out.index)
            for _c in _still_num:
                try:
                    _dz = pd.to_numeric(_f507[_c], errors="coerce").reindex(
                        out["SA2_MAIN16"].astype(int).values)
                    _dz = pd.Series(_dz.values, index=out.index)
                    _covers_gap = bool((_dz[_gap].notna()).any())
                    _still_mask = (_still_mask
                                   | (out[_c].isna() & _dz.notna()))
                    if _covers_gap:
                        _donor_gap_mask = (_donor_gap_mask
                                           | (out[_c].isna() & _dz.isna()))
                except Exception:
                    continue
            _still_zones = sorted(
                set(out.loc[_still_mask, "SA2_MAIN16"].astype(int).tolist()))
            _donor_gap_zones = sorted(
                set(out.loc[_donor_gap_mask, "SA2_MAIN16"].astype(int).tolist()))
            print(f"NOTE: {len(_still_nan)} column(s) still have NaN after "
                  f"507-fill (donor also gaps — e.g. 317011453 Mullumbimby, "
                  f"which has no QSpatial extract in either build): "
                  f"{_still_nan}")
            print(f"      zones fillable-but-unfilled ({len(_still_zones)}): "
                  f"{_still_zones} (empty means the backfill is complete)")
            print(f"      zones NaN where the 507 donor is also gapped "
                  f"({len(_donor_gap_zones)}): {_donor_gap_zones} — downstream "
                  f"fillna(0) treats their missing attractors as zero")
    except Exception as _e507:
        print(f"NOTE: 507-backfill skipped ({_e507!r}) — NaN gap cells kept")

    # -- 5. Attach ABS geography attributes + provenance flag ----------------
    out["sa2_area_sqkm"] = zones["area_sqkm"].values
    out["sa2_name"] = zones["sa2_name"].values
    out["sa4_name"] = zones["sa4_name"].values
    out["in_emme_network"] = has_emme   # corrected (data-based), not the raw name-match flag
    out["POPULATION"] = out["TOTPOP"]

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}  ({len(out)} zones, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()
