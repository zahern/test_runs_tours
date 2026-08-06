"""
crash_smoothing.py — spatial (GP-over-space) smoothing of the crash field
=========================================================================
The destination-choice model carries an OD crash covariate (log(CRASH+1)), but
raw OD crash counts are SPARSE: most O-D cells are 0, so within a purpose segment
the column is near-constant → its coefficient is unidentified (pval≈1.0, the
"CRASH dead" diagnostic). A raw sparse count is the wrong functional form for a
spatial safety signal.

This module replaces it with a spatially-smoothed crash EXPOSURE surface over
destination zones — the tractable realisation of the "GP-over-OD" idea. A kernel
smoother over zone centroids is exactly the posterior mean of a Gaussian Process
with that kernel, so:

    f* = K (K + σ²I)⁻¹ y            (GP posterior mean at the zones)

with K_ij = exp(-½ (d_ij / ℓ)²) a Gaussian kernel over great-circle zone
distances (length-scale ℓ km) and y the noisy per-zone crash exposure. This
borrows strength across neighbouring zones, turning a sparse 0/1-ish column into
a smooth, varying, identifiable field — so the crash coefficient can actually
move. A plain Nadaraya-Watson smoother (row-normalised kernel) is also provided.

Pure numpy — testable in the base env; consumed (gated) by stage5's _dest_tree.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EARTH_KM = 6371.0088


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full N×N great-circle distance matrix (km) from centroid lat/lon arrays."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2)
    return 2.0 * _EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def gaussian_kernel(dist_km: np.ndarray, length_scale_km: float) -> np.ndarray:
    return np.exp(-0.5 * (dist_km / max(length_scale_km, 1e-6)) ** 2)


def gp_smooth_field(y: np.ndarray, dist_km: np.ndarray,
                    length_scale_km: float = 5.0, noise: float = 1.0) -> np.ndarray:
    """GP posterior mean of a noisy per-zone field y over zone space:
    f* = K (K + noise·I)⁻¹ y. Larger `noise` ⇒ smoother; larger length_scale ⇒
    wider spatial borrowing. Falls back to the row-normalised (Nadaraya-Watson)
    smoother if the linear solve is singular."""
    y = np.asarray(y, dtype=float)
    K = gaussian_kernel(dist_km, length_scale_km)
    n = len(y)
    try:
        A = K + float(noise) * np.eye(n)
        return K @ np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        denom = K.sum(axis=1)
        denom[denom == 0] = 1.0
        return (K @ y) / denom


def nadaraya_watson(y: np.ndarray, dist_km: np.ndarray,
                    length_scale_km: float = 5.0) -> np.ndarray:
    """Row-normalised kernel smoother: smooth[i] = Σ_j K_ij y_j / Σ_j K_ij."""
    K = gaussian_kernel(dist_km, length_scale_km)
    denom = K.sum(axis=1)
    denom[denom == 0] = 1.0
    return (K @ np.asarray(y, dtype=float)) / denom


def smooth_zone_field(values: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                      length_scale_km: float = 5.0, noise: float = 1.0,
                      method: str = 'gp') -> np.ndarray:
    """Smooth a per-zone value array (aligned by position to lat/lon) over space."""
    D = haversine_km(lat, lon)
    if method == 'nw':
        return nadaraya_watson(values, D, length_scale_km)
    return gp_smooth_field(values, D, length_scale_km, noise)


def smoothing_operator(lat: np.ndarray, lon: np.ndarray,
                       length_scale_km: float = 5.0, noise: float = 1.0,
                       method: str = 'gp') -> np.ndarray:
    """Return the N×N spatial smoothing matrix S such that `S @ y` (or, for a
    stack of fields, `Y @ S.T`) is the smoothed field over zone space. This lets
    a whole (cases × zones) surface — e.g. the destination-choice mode logsum —
    be smoothed across zones in one matmul with a precomputed operator.

    GP posterior-mean operator: S = K (K + noise·I)⁻¹  (computed via a solve).
    Nadaraya-Watson: S = row-normalised K. Falls back to NW if the solve fails."""
    D = haversine_km(lat, lon)
    K = gaussian_kernel(D, length_scale_km)
    n = len(lat)
    if method == 'nw':
        denom = K.sum(axis=1, keepdims=True); denom[denom == 0] = 1.0
        return K / denom
    try:
        A = K + float(noise) * np.eye(n)
        # S = K A⁻¹; since K,A symmetric, S = solve(A, K).T
        return np.linalg.solve(A, K).T
    except np.linalg.LinAlgError:
        denom = K.sum(axis=1, keepdims=True); denom[denom == 0] = 1.0
        return K / denom


def smooth_destination_crash(od_crash: pd.DataFrame, centroids: pd.DataFrame,
                             length_scale_km: float = 5.0, noise: float = 1.0,
                             agg: str = 'sum', method: str = 'gp',
                             zone_col: str = 'ORDER') -> dict:
    """Aggregate an OD crash table (OTAZ, DTAZ, CRASH) to a per-destination
    exposure, spatially smooth it over zone centroids, and return a
    {destination_zone: smoothed_crash} dict aligned to `centroids[zone_col]`.

    Zones present in `centroids` but absent from `od_crash` get 0 exposure before
    smoothing (so the kernel still fills them from neighbours — the whole point).
    """
    if od_crash is None or not len(od_crash) or centroids is None or not len(centroids):
        return {}
    dcol = next((c for c in ('DTAZ', 'dtaz', 'D', 'dest') if c in od_crash.columns), None)
    ccol = next((c for c in ('CRASH', 'crash', 'crashes') if c in od_crash.columns), None)
    if dcol is None or ccol is None or zone_col not in centroids.columns:
        logger.warning("smooth_destination_crash: missing DTAZ/CRASH/zone columns — skipping")
        return {}
    per_dest = od_crash.groupby(dcol)[ccol].agg(agg)
    cz = centroids[[zone_col, 'lat', 'lon']].dropna().drop_duplicates(zone_col)
    zones = cz[zone_col].values
    y = np.array([float(per_dest.get(z, 0.0)) for z in zones], dtype=float)
    if not np.any(y > 0):
        logger.warning("smooth_destination_crash: no positive crash exposure — skipping")
        return {}
    sm = smooth_zone_field(y, cz['lat'].values, cz['lon'].values,
                           length_scale_km=length_scale_km, noise=noise, method=method)
    logger.info(f"smooth_destination_crash: {len(zones)} zones, raw nonzero="
                f"{int((y > 0).sum())} → smoothed nonzero={int((sm > 1e-9).sum())} "
                f"(ℓ={length_scale_km}km, {method})")
    return {z: float(v) for z, v in zip(zones, sm)}


if __name__ == '__main__':
    # Self-check: a sparse crash field on a line of zones smooths into a
    # continuous bump, and the smoothed field has far more distinct values.
    rng = np.random.default_rng(0)
    n = 60
    lat = -27.4 + 0.01 * np.arange(n)     # a north-south line of zones
    lon = np.full(n, 153.0)
    y = np.zeros(n)
    y[[20, 21, 40]] = [5.0, 3.0, 8.0]     # crashes at a few zones only
    D = haversine_km(lat, lon)
    gp = gp_smooth_field(y, D, length_scale_km=2.0, noise=1.0)
    nw = nadaraya_watson(y, D, length_scale_km=2.0)
    print(f"raw   : nonzero={int((y>0).sum())}/{n}  distinct={len(np.unique(y))}  "
          f"var={y.var():.3f}")
    print(f"GP    : nonzero={int((gp>1e-6).sum())}/{n}  distinct={len(np.unique(gp.round(6)))}  "
          f"var={gp.var():.3f}  peak@{int(np.argmax(gp))}")
    print(f"NW    : nonzero={int((nw>1e-6).sum())}/{n}  distinct={len(np.unique(nw.round(6)))}  "
          f"peak@{int(np.argmax(nw))}")
    # crash mass should concentrate near zone 40 (largest) for both
    assert 35 <= int(np.argmax(gp)) <= 45, "GP peak not near the largest crash zone"
    print("OK — sparse crash field smoothed into an identifiable spatial surface")
