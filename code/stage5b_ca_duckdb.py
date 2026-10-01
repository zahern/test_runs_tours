"""
stage5b_ca_duckdb.py
====================
DuckDB-backed case-alternative (CA) materialiser for the stage-5b / stage-51
trip-destination MNL.

Why
---
The GA refits the same trip-destination model for every chromosome; only the
ACTIVE feature subset changes. Rebuilding the `n_obs x n_zones` pandas design
on each evaluation is what dominates memory and runtime. This module:

1. Builds the wide (case, alt) feature table ONCE per purpose with DuckDB
   joins (obs x alt pairs x skim matrices x landuse x person attributes),
   using the SAME importance sampling as `stage5b_trip_destination`
   (Manski-Lerman correction stored as `log_correction`).
2. Writes it to parquet as float32.
3. Serves per-evaluation column-projected reads (`SELECT` only the current
   chromosome's columns), so no pandas rebuild and no unused columns in RAM.

If duckdb is unavailable, `available()` returns False and callers fall back to
the in-memory pandas design transparently.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
    _HAVE_DUCKDB = True
except Exception:  # pragma: no cover - optional dependency
    duckdb = None
    _HAVE_DUCKDB = False

logger = logging.getLogger(__name__)

SKIM_ALIASES = ("NetworkDist", "AutoIVT_AM", "TransitIVT_AM", "AutoCrashRisk")
LU_ATTRS = ("TotalEmployment", "RetailEmp", "HealthEduRetEmp", "Shopping",
            "EATOUT", "Entertainment", "TotalPop", "TotalHH")
PERSON_COVS = ("income_hi", "is_female", "hh_child", "occp_group",
               "ptype_FullTimeWorker", "ptype_UniversityStudent", "mtwp",
               "WFHWEEK")
_TOUR_MODE_AUTO = (1, 2)
_TOUR_MODE_TRANSIT = (5,)
_TOUR_MODE_ACTIVE = (3, 4)


def available() -> bool:
    return _HAVE_DUCKDB


def build_or_get(cache_dir, purpose, stop_trips, all_zones, skims, landuse,
                 taz_index, zone_id_col, sample_size, seed,
                 full_grid: bool = False, logsums=None) -> dict:
    """Return {'path': parquet, 'sampled': bool}; build it on first call.

    full_grid=True materialises the full rectangular case×zone grid (no
    alternative sampling, no Manski-Lerman correction) instead of the ragged
    importance-sampled pairs. The ragged CA cannot be estimated: Larch 6's
    dense ``from_idca`` unstacks missing (case, alt) cells to NaN and every
    engine (jax/numba, full/backbone) then fails — the silent "ALL estimation
    attempts failed" seen in every Stage-51 shard. The full grid needs no
    correction and fits with the identical engine chain. Pair with a case cap
    upstream (STAGE5B_CA_FULLGRID_CASES) to keep the grid tractable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    sampled = bool(not full_grid and sample_size
                   and 0 < int(sample_size) < len(all_zones))
    _fg = "fg1_" if full_grid else ""
    _ls = "ls1" if logsums is not None or "logsum" in stop_trips.columns else "ls0"
    key = (f"ca_{purpose}_{_fg}{_ls}s{int(sample_size)}_n{len(stop_trips)}_"
           f"seed{int(seed)}_{zone_id_col}.parquet")
    path = os.path.join(cache_dir, key)
    if not os.path.isfile(path):
        _build(path, purpose, stop_trips, all_zones, skims, landuse,
               taz_index, zone_id_col, int(sample_size), int(seed),
             full_grid=full_grid, logsums=logsums)
    return {"path": path, "sampled": sampled}


def load_ca(path, columns, has_correction: bool = True):
    """Read the CA parquet with column projection.

    Returns (DataFrame, n_cases). Missing requested columns are ignored so a
    chromosome can never fail on an unavailable candidate.
    """
    if not _HAVE_DUCKDB:
        return None, 0
    have = _parquet_columns(path)
    cols = ["case_id", "alt_id", "chosen"]
    for c in list(columns) + (["log_correction"] if has_correction else []):
        if c in have and c not in cols:
            cols.append(c)
    sel = ", ".join(f'"{c}"' for c in cols)
    df = duckdb.query(
        f"SELECT {sel} FROM read_parquet('{path}')").to_df()
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df["chosen"] = df["chosen"].astype(int)
    n_obs = int(df["case_id"].nunique())
    return df, n_obs


def _parquet_columns(path) -> set:
    try:
        info = duckdb.query(
            f"SELECT * FROM read_parquet('{path}') LIMIT 0").to_df()
        return set(info.columns)
    except Exception:  # noqa: BLE001
        return set()


def _build(path, purpose, stop_trips, all_zones, skims, landuse, taz_index,
           zone_id_col, sample_size, seed, full_grid: bool = False,
           logsums=None) -> None:
    from stage5b_trip_destination import (
        _choice_set_metrics, _normalise_mode_codes, _resolve_logsum_values,
        _sample_alt_pairs, compute_size_term, LANDUSE_ALIAS)

    zones = all_zones[[zone_id_col]].drop_duplicates().reset_index(drop=True)
    log_size_all = compute_size_term(
        zones.assign(**{zone_id_col: zones[zone_id_col]}),
        purpose, landuse, zone_id_col)

    stop = stop_trips.copy().reset_index(drop=True)
    z2i = {int(z): i for i, z in enumerate(zones[zone_id_col].values)}
    valid = stop["dest_taz"].map(
        lambda z: pd.notna(z) and int(z) in z2i).fillna(False)
    if not bool(valid.all()):
        logger.warning(
            f"CA build [{purpose}]: dropping {int((~valid).sum())}/"
            f"{len(stop)} obs whose DTAZ is outside the zone universe")
        stop = stop.loc[valid].reset_index(drop=True)

    n_obs = len(stop)
    _sampled_choice_set = 0 < int(sample_size) < len(zones)
    if full_grid:
        # Rectangular full cartesian grid (see build_or_get): no feasibility
        # mask, no envelope repair, no correction — every (case, zone) pair
        # present so dense from_idca stays NaN-free.
        obs_idx, alt_idx, log_corr = _sample_alt_pairs(
            stop["dest_taz"].values, zones[zone_id_col].values,
            log_size_all, 0, seed, feasible_by_obs=None)
    elif _sampled_choice_set:
        # Sampled SearchLibrium choice sets do not need a dense feasibility
        # matrix; _sample_alt_pairs always retains the chosen alternative.
        feasible_by_obs = None
        obs_idx, alt_idx, log_corr = _sample_alt_pairs(
            stop["dest_taz"].values, zones[zone_id_col].values,
            log_size_all, sample_size, seed,
            feasible_by_obs=feasible_by_obs)

    else:
        full_obs_idx = np.repeat(np.arange(n_obs), len(zones))
        full_alt_ids = np.tile(zones[zone_id_col].to_numpy(), n_obs)
        full_obs_rows = stop.iloc[full_obs_idx].reset_index(drop=True)
        full_metrics = _choice_set_metrics(
            full_obs_rows, full_alt_ids, skims, taz_index)
        feasible_by_obs = full_metrics["feasible"].reshape(n_obs, len(zones))
        chosen_idx = stop["dest_taz"].map(z2i).astype(int).to_numpy()
        chosen_feasible = feasible_by_obs[np.arange(n_obs), chosen_idx]
        feasible_by_obs[np.arange(n_obs), chosen_idx] = True
        if not bool(chosen_feasible.all()):
            logger.warning(
                "CA build [%s]: retaining %d observed destinations outside the "
                "mode-aware detour envelope", purpose, int((~chosen_feasible).sum()))

        obs_idx, alt_idx, log_corr = _sample_alt_pairs(
            stop["dest_taz"].values, zones[zone_id_col].values,
            log_size_all, sample_size, seed,
            feasible_by_obs=feasible_by_obs)

    alt_zone_vals = zones[zone_id_col].values[alt_idx]
    dest_vals = pd.to_numeric(stop["dest_taz"], errors="coerce").values

    pairs = pd.DataFrame({
        "case_id": np.asarray(obs_idx, dtype=np.int64),
        "obs_row": np.asarray(obs_idx, dtype=np.int64),
        "alt_row": np.asarray(alt_idx, dtype=np.int64),
        "chosen": (dest_vals[np.asarray(obs_idx, dtype=np.int64)]
                   == alt_zone_vals).astype(np.int8),
    })
    if log_corr is not None:
        pairs["log_correction"] = np.asarray(log_corr, dtype=np.float64)

    if logsums is not None:
        pair_obs = stop.iloc[np.asarray(obs_idx, dtype=np.int64)].reset_index(drop=True)
        pairs["logsum"] = _resolve_logsum_values(
            logsums, pair_obs, alt_zone_vals)

    def _num(col, default=0.0):
        s = (stop[col] if col in stop.columns
             else pd.Series(default, index=stop.index))
        return pd.to_numeric(s, errors="coerce").fillna(default).values

    origin_zone = _num("origin_taz").astype(np.int64)
    anchor_zone = _num("anchor_down_taz").astype(np.int64)
    obs_tbl = pd.DataFrame({
        "obs_row": np.arange(len(stop), dtype=np.int64),
        "origin_zone": origin_zone,
        "anchor_zone": anchor_zone,
        "origin_idx": [int(taz_index.get(int(z), -1)) for z in origin_zone],
        "anchor_idx": [int(taz_index.get(int(z), -1)) for z in anchor_zone],
        "is_outbound": _num("is_outbound", 1.0).astype(np.int8),
        "depart_time_min": _num("depart_time_min", 0.0),
    })
    obs_tbl["logsum"] = _num("logsum", 0.0)
    tm_num = pd.to_numeric(stop.get("tour_mode", pd.Series(1, index=stop.index)),
                           errors="coerce")
    if float(tm_num.notna().mean()) < 0.5:
        tm_num = (stop.get("tour_mode", pd.Series("", index=stop.index))
                  .astype(str).str.strip()
                  .map({"DA": 1, "SR": 2, "Walk": 3, "Bike": 4, "Transit": 5}))
    obs_tbl["tour_mode"] = tm_num.fillna(1).astype(int).values
    obs_tbl["choice_mode"] = _normalise_mode_codes(stop)
    for c in PERSON_COVS:
        if c in stop.columns:
            obs_tbl[c] = _num(c, 0.0)

    zones_tbl = pd.DataFrame({
        "alt_row": np.arange(len(zones), dtype=np.int64),
        "alt_zone": zones[zone_id_col].values,
        "alt_idx": [int(taz_index.get(int(z), -1))
                    for z in zones[zone_id_col].values],
        "log_size": log_size_all,
    })
    lu = landuse.copy()
    for alias, orig in LANDUSE_ALIAS.items():
        if orig in lu.columns and alias not in lu.columns:
            lu[alias] = lu[orig]
    lu_idx = lu.set_index(zone_id_col) if zone_id_col in lu.columns else lu
    for c in LU_ATTRS:
        if c in lu_idx.columns:
            zones_tbl[c] = (zones[zone_id_col].map(lu_idx[c])
                            .fillna(0.0).astype(float).values)
        else:
            zones_tbl[c] = 0.0

    con = duckdb.connect()
    con.register("pairs", pairs)
    con.register("obs", obs_tbl)
    con.register("zones", zones_tbl)
    skim_joins = []
    skim_selects = []
    for alias in SKIM_ALIASES:
        if alias not in skims:
            continue
        mat = np.asarray(skims[alias], dtype=np.float64)
        n = mat.shape[0]
        con.register(f"skim_{alias}", pd.DataFrame({
            "o": np.repeat(np.arange(n, dtype=np.int64), n),
            "d": np.tile(np.arange(n, dtype=np.int64), n),
            "v": mat.ravel(),
        }))
        skim_joins.append(
            f"LEFT JOIN skim_{alias} sk_{alias} "
            f"ON sk_{alias}.o = o.origin_idx AND sk_{alias}.d = z.alt_idx")
        skim_selects.append(
            f'CAST(COALESCE(sk_{alias}.v, 0) AS FLOAT) AS "{alias}"')
    if "NetworkDist" in skims:
        skim_joins += [
            "LEFT JOIN skim_NetworkDist d_oa "
            "ON d_oa.o = o.origin_idx AND d_oa.d = z.alt_idx",
            "LEFT JOIN skim_NetworkDist d_ad "
            "ON d_ad.o = z.alt_idx AND d_ad.d = o.anchor_idx",
            "LEFT JOIN skim_NetworkDist d_od "
            "ON d_od.o = o.origin_idx AND d_od.d = o.anchor_idx",
        ]

    for alias, prefix in (("WalkDist", "walk"), ("BikeDist", "bike")):
        if alias in skims:
            # Register the mode-specific skim (SKIM_ALIASES loop above only
            # registers NetworkDist/AutoIVT_AM/TransitIVT_AM/AutoCrashRisk,
            # so skim_WalkDist/skim_BikeDist would otherwise not exist and the
            # detour query fails with "Table with name skim_WalkDist does not
            # exist"). Also add the origin->alt (oa) leg used by the detour.
            try:
                _mm = np.asarray(skims[alias], dtype=np.float64)
                _nn = _mm.shape[0]
                con.register(f"skim_{alias}", pd.DataFrame({
                    "o": np.repeat(np.arange(_nn, dtype=np.int64), _nn),
                    "d": np.tile(np.arange(_nn, dtype=np.int64), _nn),
                    "v": _mm.ravel(),
                }))
            except Exception:
                pass
            skim_joins += [
                f"LEFT JOIN skim_{alias} {prefix}_oa "
                f"ON {prefix}_oa.o = o.origin_idx AND {prefix}_oa.d = z.alt_idx",
                f"LEFT JOIN skim_{alias} {prefix}_ad "
                f"ON {prefix}_ad.o = z.alt_idx AND {prefix}_ad.d = o.anchor_idx",
                f"LEFT JOIN skim_{alias} {prefix}_od "
                f"ON {prefix}_od.o = o.origin_idx AND {prefix}_od.d = o.anchor_idx",
            ]

    _network_detour = (
        "GREATEST(COALESCE(d_oa.v,0) + COALESCE(d_ad.v,0) "
        "- COALESCE(d_od.v,0), 0)"
    )
    _distance_expr = "COALESCE(d_ad.v, 0)"
    _detour_expr = _network_detour
    if "WalkDist" in skims:
        _walk_detour = (
            "GREATEST(COALESCE(walk_oa.v,0) + COALESCE(walk_ad.v,0) "
            "- COALESCE(walk_od.v,0), 0)"
        )
        _detour_expr = (
            f"CASE WHEN o.choice_mode = 3 THEN {_walk_detour} "
            f"ELSE {_detour_expr} END"
        )
        _distance_expr = (
            "CASE WHEN o.choice_mode = 3 THEN COALESCE(walk_ad.v, 0) "
            f"ELSE {_distance_expr} END"
        )
    if "BikeDist" in skims:
        _bike_detour = (
            "GREATEST(COALESCE(bike_oa.v,0) + COALESCE(bike_ad.v,0) "
            "- COALESCE(bike_od.v,0), 0)"
        )
        _detour_expr = (
            f"CASE WHEN o.choice_mode = 4 THEN {_bike_detour} "
            f"ELSE {_detour_expr} END"
        )
        _distance_expr = (
            "CASE WHEN o.choice_mode = 4 THEN COALESCE(bike_ad.v, 0) "
            f"ELSE {_distance_expr} END"
        )

    mode_case = ("CASE "
                 f"WHEN o.tour_mode IN {_TOUR_MODE_AUTO} THEN 1 ELSE 0 END")
    mode_transit = ("CASE "
                    f"WHEN o.tour_mode IN {_TOUR_MODE_TRANSIT} THEN 1 ELSE 0 END")
    mode_active = ("CASE "
                   f"WHEN o.tour_mode IN {_TOUR_MODE_ACTIVE} THEN 1 ELSE 0 END")
    person_select = ", ".join(
        f'CAST(COALESCE(o."{c}", 0) AS FLOAT) AS "{c}"'
        for c in PERSON_COVS if c in obs_tbl.columns)
    corr_select = ('CAST(p.log_correction AS FLOAT) AS log_correction'
                   if "log_correction" in pairs.columns else
                   'CAST(0 AS FLOAT) AS log_correction')
    logsum_select = ('CAST(p.logsum AS FLOAT) AS logsum'
                     if "logsum" in pairs.columns else
                     'CAST(o.logsum AS FLOAT) AS logsum')
    lu_select = ", ".join(f'CAST(z."{c}" AS FLOAT) AS "{c}"' for c in LU_ATTRS)

    query = f"""
    SELECT
        p.case_id,
        CAST(z.alt_zone AS BIGINT) AS alt_id,
        p.chosen,
        {corr_select},
        {logsum_select},
        CAST({_detour_expr} AS FLOAT) AS detour_dist_km,
        CAST(z.log_size AS FLOAT)                AS log_size,
        CAST({_distance_expr} AS FLOAT)          AS dist_to_anchor,
        CAST(LN(1 + GREATEST({_distance_expr}, 0)) AS FLOAT)
                                                 AS log_dist_to_anchor,
        CAST(CASE WHEN z.alt_zone = o.anchor_zone THEN 1 ELSE 0 END
             AS FLOAT)                           AS same_as_anchor,
        CAST(CASE WHEN z.alt_zone = o.origin_zone THEN 1 ELSE 0 END
             AS FLOAT)                           AS same_as_prev,
        CAST(1 - o.is_outbound AS FLOAT)         AS is_inbound,
        CAST(o.depart_time_min AS FLOAT)         AS depart_time_min,
        CAST(o.is_outbound AS FLOAT)             AS is_outbound,
        CAST({mode_case} AS FLOAT)               AS tour_mode_auto,
        CAST({mode_transit} AS FLOAT)            AS tour_mode_transit,
        CAST({mode_active} AS FLOAT)             AS tour_mode_active,
        {", ".join(skim_selects) if skim_selects else "CAST(0 AS FLOAT) AS AutoIVT_AM"},
        {lu_select}
        {"," if person_select else ""} {person_select}
    FROM pairs p
    JOIN zones z ON p.alt_row = z.alt_row
    JOIN obs   o ON p.obs_row = o.obs_row
    {" ".join(skim_joins)}
    """
    tmp = path + ".tmp"
    con.execute(
        f"COPY ({query}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    os.replace(tmp, path)
    logger.info(f"CA build [{purpose}]: {len(pairs):,} (case, alt) rows -> {path} "
                f"({os.path.getsize(path)/1e6:.1f} MB)")
