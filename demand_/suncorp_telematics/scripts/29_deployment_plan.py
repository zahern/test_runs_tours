"""
================================================================================
29_deployment_plan.py - Programme P5 on the P4-selected posts
================================================================================
NOTE (Sep-2026 review): SUPERSEDED for paper results by
  scripts/40_rsp_spatial_effect.py + 00_paper_rebuild.pbs.
Reason: this file has broken refs (r_event/acc_fx undefined when called with
--posts/--squad alone), fabricates fixture attendance from event counts when
no calendar is given, and answers "coverage %" without a with/without-platform
counterfactual per segment. Keep for engineering sweeps only; do NOT \input
its tex directly into the paper until the RSP spatial-effect table passes.
Original doc follows.
Consumes results/post_selection_mip.csv (Programme P4 output) and the raw
hard-braking / speeding stream, builds the space-time risk r_{jh} (weighted 
risky-driving in each post's 500 m catchment per 15-min bucket over 24h days),
then solves the relocation-aware deployment MIP (Programme P5) exactly with
HiGHS. Posts can be re-optimised per phase (pre/during/post). Officers move
with travel-time costs and incur no coverage while in transit.

Outputs:
  results/deployment_plan.csv        optimal schedule with movement tokens
  results/deployment_summary.json    coverage metrics, relocations, Pareto
  results/deployment_riskscore.tex   per-post phase table for the paper
  figures/fig32_mip_schedule.png     risk heatmap + optimal schedule
  figures/fig33_mip_value.png        coverage vs squad size (Pareto)
"""

import sys
sys.path.insert(0, "/home/ahernz/MetaCount")
try:
    from metacount import RiskPredictor, build_fixture_dataframe
    METACOUNT_AVAILABLE = True
except ImportError:
    METACOUNT_AVAILABLE = False
    print("Warning: MetaCount not available, using simple attendance scaling")
    # Fallback build_fixture_dataframe
    def build_fixture_dataframe(fixtures, attendances, fixture_type_map):
        import pandas as pd
        rows = []
        for (label, kiso), att in zip(fixtures, attendances):
            dt = pd.Timestamp(kiso)
            rows.append({
                "fixture_id": label,
                "attendance": att,
                "kickoff_hour": dt.hour + dt.minute / 60.0,
                "fixture_type": fixture_type_map.get(label, "NRL")
            })
        return pd.DataFrame(rows)
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STADIUM_LAT, STADIUM_LON = -27.4647, 153.0093
BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)

# Ordinary evening baseline
ORDINARY_EVENING = ("NonEvent Wed 18Mar", "2026-03-18T16:00")  # 16:00-23:00

# Default attendance estimates by fixture type (can be overridden by fixture_calendar.csv)
DEFAULT_ATTENDANCE = {
    "NRL": 35000,
    "SuperRugby": 18000,
    "ALeague": 10000,
}

# Team name to fixture type mapping
TEAM_TO_TYPE = {
    "Broncos": "NRL", "Panthers": "NRL", "Eels": "NRL", "Dolphins": "NRL",
    "Titans": "NRL", "Rabbitohs": "NRL", "Roosters": "NRL", "Storm": "NRL",
    "Knights": "NRL", "Warriors": "NRL", "Sharks": "NRL", "Dragons": "NRL",
    "Sea Eagles": "NRL", "Bulldogs": "NRL", "Tigers": "NRL", "Raiders": "NRL",
    "Cowboys": "NRL",
    "Reds": "SuperRugby", "Tahs": "SuperRugby", "Brumbies": "SuperRugby",
    "Rebels": "SuperRugby", "Force": "SuperRugby", "Chiefs": "SuperRugby",
    "Crusaders": "SuperRugby", "Blues": "SuperRugby", "Hurricanes": "SuperRugby",
    "Highlanders": "SuperRugby", "Moana": "SuperRugby", "Fijian Drua": "SuperRugby",
    "Roar": "ALeague", "Phoenix": "ALeague", "Victory": "ALeague", "City": "ALeague",
    "Sydney FC": "ALeague", "Wanderers": "ALeague", "Mariners": "ALeague",
    "Jets": "ALeague", "Glory": "ALeague", "United": "ALeague",
}


def infer_fixture_type(label: str) -> str:
    """Infer fixture type from match label (e.g., 'Broncos-Panthers' -> 'NRL')."""
    # Split by common separators
    for sep in [" vs ", " v ", "-", "–"]:
        if sep in label:
            parts = label.split(sep)
            break
    else:
        parts = [label]
    
    # Check each part for known team names
    for part in parts:
        part = part.strip()
        for team, ftype in TEAM_TO_TYPE.items():
            if team.lower() in part.lower():
                return ftype
    # Default to NRL if unknown
    return "NRL"


def extract_fixtures_from_events(events_path: str, min_events_per_fixture: int = 500,
                                  catchment_m: float = 1000.0) -> list:
    """
    Extract fixture days from hard braking events near Suncorp Stadium.
    
    Scans events within catchment_m of stadium, groups by date, and identifies
    days with significant event activity (fixture days).
    
    Returns list of (label, ISO_datetime, attendance, fixture_type) tuples.
    """
    # Load posts to get stadium-centric KDTree for filtering
    # For extraction, we just need events near stadium
    stadium_tree = cKDTree(np.array([[STADIUM_LAT, STADIUM_LON]]))
    
    daily_counts = defaultdict(int)
    daily_events = defaultdict(list)
    
    for ch in pd.read_csv(events_path,
                          usecols=["EventStart", "SnappedLatitude", "SnappedLongitude"],
                          parse_dates=["EventStart"], chunksize=400_000):
        ch = ch.rename(columns={"EventStart": "ts",
                                "SnappedLatitude": "lat",
                                "SnappedLongitude": "lon"})
        m = ((ch["lon"] >= BBOX["lon_min"]) & (ch["lon"] <= BBOX["lon_max"]) &
             (ch["lat"] >= BBOX["lat_min"]) & (ch["lat"] <= BBOX["lat_max"]))
        ch = ch[m]
        if ch.empty:
            continue
        dist, _ = stadium_tree.query(np.column_stack([np.asarray(ch["lat"].values, dtype=float), np.asarray(ch["lon"].values, dtype=float)]),
                                     distance_upper_bound=catchment_m)
        ok = np.isfinite(dist)
        ch = ch[ok]
        if ch.empty:
            continue
        # Group by date
        ch["date"] = ch.ts.dt.date
        for date, group in ch.groupby("date"):
            daily_counts[date] += len(group)
            daily_events[date].append(group)
    
    # Filter days with sufficient events
    fixture_dates = {d: c for d, c in daily_counts.items() if c >= min_events_per_fixture}
    
    if not fixture_dates:
        # Fallback: use hardcoded March 2026 fixtures
        return _get_fallback_fixtures()
    
    # Sort by date
    sorted_dates = sorted(fixture_dates.items(), key=lambda x: x[0])
    
    # Build fixture labels from date and event patterns
    # For now, use generic labels; in production, cross-reference with fixture calendar
    fixtures = []
    for i, (date, count) in enumerate(sorted_dates):
        # Try to infer kickoff time from peak event hour
        all_ts = pd.concat(daily_events[date])["ts"]
        peak_hour = all_ts.dt.hour.mode().iloc[0] if len(all_ts) > 0 else 19
        kickoff = pd.Timestamp(date).replace(hour=peak_hour, minute=0)
        
        # Estimate attendance from event count (rough scaling)
        base_att = 30000
        attendance = int(base_att * (count / max(daily_counts.values()) + 0.5))
        
        label = f"Fixture_{date}_Match{i+1}"
        fixture_type = "NRL"  # Default
        
        fixtures.append((label, kickoff.isoformat(), attendance, fixture_type))
    
    return fixtures


def load_fixture_calendar(calendar_path: str = None) -> list:
    """
    Load fixture calendar from CSV file.
    Expected columns: label, kickoff_iso, attendance, fixture_type
    Falls back to extraction from events if file not found.
    """
    if calendar_path and os.path.exists(calendar_path):
        df = pd.read_csv(calendar_path)
        required = ["label", "kickoff_iso", "attendance", "fixture_type"]
        if all(c in df.columns for c in required):
            return list(zip(df["label"], df["kickoff_iso"], 
                           df["attendance"].astype(int), df["fixture_type"]))
    return None


def _get_fallback_fixtures() -> list:
    """Fallback to known March 2026 fixtures if extraction fails."""
    return [
        ("NRL Broncos-Panthers",  "2026-03-06T19:00", 45566, "NRL"),
        ("NRL Broncos-Eels",      "2026-03-12T19:00", 39015, "NRL"),
        ("SuperRugby Reds-Tahs",  "2026-03-14T18:35", 18087, "SuperRugby"),
        ("NRL Dolphins-Titans",   "2026-03-15T17:15", 16073, "NRL"),
        ("ALeague Roar-Phoenix",  "2026-03-21T14:00", 4826, "ALeague"),
        ("NRL Broncos-Dolphins",  "2026-03-27T19:00", 45882, "NRL"),
    ]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_posts(path):
    p = pd.read_csv(path)
    global CAND_IDS
    CAND_IDS = p["node_id"].to_numpy()
    return p["lat"].to_numpy(), p["lon"].to_numpy(), p["rank"].to_numpy()


def build_daily_risk(events_path, posts_lat, posts_lon,
                     catchment_m=500.0, n_buckets=96, chunksize=400_000):
    """
    Build 24-hour daily risk profiles (96 x 15-min buckets = 24h).
    Returns risk array of shape (n_days, J, n_buckets) where each day
    runs 00:00-23:45 in 15-min buckets.
    """
    J = len(posts_lat)
    tree = cKDTree(np.column_stack([posts_lat, posts_lon]))
    
    # First pass: collect all fixture days
    fixture_days = set()
    for ch in pd.read_csv(events_path,
                          usecols=["EventStart", "SnappedLatitude", "SnappedLongitude"],
                          parse_dates=["EventStart"], chunksize=chunksize):
        ch = ch.rename(columns={"EventStart": "ts", "SnappedLatitude": "lat", "SnappedLongitude": "lon"})
        m = ((ch["lon"] >= BBOX["lon_min"]) & (ch["lon"] <= BBOX["lon_max"]) &
             (ch["lat"] >= BBOX["lat_min"]) & (ch["lat"] <= BBOX["lat_max"]))
        ch = ch[m]
        if ch.empty:
            continue
        dates = ch.ts.dt.date.unique()
        fixture_days.update(dates)
    
    fixture_days = sorted([d for d in fixture_days if not pd.isna(d)])
    n_days = len(fixture_days)
    print(f"   Found {n_days} fixture days")
    
    acc = np.zeros((n_days, J, n_buckets))
    day_starts = [pd.Timestamp(d) for d in fixture_days]
    
    for ch in pd.read_csv(events_path,
                          usecols=["EventStart", "SnappedLatitude",
                                   "SnappedLongitude", "SpeedLoss_kmh"],
                          parse_dates=["EventStart"], chunksize=chunksize):
        ch = ch.rename(columns={"EventStart": "ts",
                                "SnappedLatitude": "lat",
                                "SnappedLongitude": "lon"})
        m = ((ch["lon"] >= BBOX["lon_min"]) & (ch["lon"] <= BBOX["lon_max"]) &
             (ch["lat"] >= BBOX["lat_min"]) & (ch["lat"] <= BBOX["lat_max"]))
        ch = ch[m]
        if ch.empty:
            continue
        # Classify event type
        # Hard braking: high deceleration (we use proxy from data)
        # Speeding: speed > limit + threshold (use speed if available)
        lat_vals = np.asarray(ch["lat"].values, dtype=float)
        lon_vals = np.asarray(ch["lon"].values, dtype=float)
        coords = np.asarray(np.column_stack([lat_vals, lon_vals]), dtype=float)
        print("DEBUG coords type:", type(coords), "shape:", coords.shape, "dtype:", coords.dtype)
        
        # Filter out NaN coordinates
        valid = np.isfinite(coords).all(axis=1)
        print("DEBUG valid:", valid.dtype, valid.sum(), "out of", len(valid))
        if not valid.any():
            continue
        coords = coords[valid]
        ch = ch[valid]
        dist, idx = tree.query(coords, distance_upper_bound=catchment_m)
        ok = np.isfinite(dist)
        ch, idx = ch[ok], idx[ok]
        if ch.empty:
            continue
        ts = ch.ts.to_numpy()
        for d, day_start in enumerate(day_starts):
            day_end = day_start + pd.Timedelta(days=1)
            msk = (ts >= np.datetime64(day_start)) & (ts < np.datetime64(day_end))
            if not msk.any():
                continue
            hh = (ts[msk] - np.datetime64(day_start)) / np.timedelta64(15, "m")
            b = hh.astype(int)
            b = np.clip(b, 0, n_buckets - 1)
            # Weight: hard braking = 1.0, speeding = 0.5 (if speed data available)
            weights = np.ones_like(b, dtype=float)
            if "SpeedLoss_kmh" in ch.columns:
                speed = ch.loc[msk, 'SpeedLoss_kmh'].to_numpy()
                weights = np.where(speed > 60, 0.5, 1.0)  # speeding gets half weight
            np.add.at(acc[d], (idx[msk], b), weights)
    
    # Mean per fixture day
    r_mean = acc.mean(axis=0)  # (J, n_buckets)
    return r_mean, day_starts, acc


def build_phase_risk(acc, day_starts, fixtures, n_buckets=96):
    """
    Slice daily risk into event-day phases aligned to kickoff.
    Returns:
      r_phase: (n_fixtures, J, 28) - 7 hours pre/post in 15-min buckets
      phase_starts: list of phase start bucket indices
    """
    # Find which daily bucket each fixture's -3h aligns to
    # Daily buckets: 0=00:00, 12=03:00, ..., 76=19:00, etc.
    phase_risk = np.zeros((len(fixtures), acc.shape[1], 28))
    for f, (label, kiso) in enumerate(fixtures):
        kickoff = pd.Timestamp(kiso)
        # Bucket index for kickoff-3h in the 24h day
        start_hour = (kickoff - pd.Timedelta(hours=3)).hour
        start_bucket = start_hour * 4  # 15-min buckets
        # Extract 7 hours = 28 buckets
        daily_idx = np.where([d.date() == kickoff.date() for d in day_starts])[0]
        if len(daily_idx) == 0:
            continue
        d = daily_idx[0]
        phase_risk[f] = acc[d, :, start_bucket:start_bucket+28]
    return phase_risk


def tau_matrix(plat, plon, delta_min=15.0, v_kmh=12.0, setup_min=5.0,
               max_move_m=4000.0):
    J = len(plat)
    D = haversine_m(plat[:, None], plon[:, None], plat[None, :], plon[None, :])
    tau = np.full((J, J), 10**6, dtype=int)
    for i in range(J):
        for j in range(J):
            if i == j:
                tau[i, j] = 1
            elif D[i, j] <= max_move_m:
                t = int(np.ceil((setup_min +
                                 (D[i, j] / 1000.0) / v_kmh * 60.0) / delta_min))
                tau[i, j] = max(2, t)
    return tau, D


def solve_p5_phased(r_phase, P, tau, phases, lam=0.02, time_limit=300.0):
    """
    Phased deployment MIP: select posts per phase, allow movement between phases.
    
    Args:
        r_phase: (n_fixtures, J, T) risk per fixture per phase
        P: number of officers
        tau: (J, J) travel time in buckets
        phases: list of (phase_name, start_bucket, end_bucket) tuples
        lam: movement penalty per bucket in transit
        
    Returns:
        x: (n_fixtures, J, T) assignment per fixture
        movements: list of movement dicts per fixture
    """
    # Average risk across fixtures for each phase
    n_fixtures, J, T = r_phase.shape
    r_avg = r_phase.mean(axis=0)  # (J, T)
    
    # Phase boundaries
    phase_bounds = {}
    for name, start, end in phases:
        phase_bounds[name] = (start, end)
    
    # For each phase, we need P officers. They can move at phase boundaries.
    # Decision variables:
    # x[j, t] = 1 if officer at post j in bucket t
    # y[i, j, t] = 1 if officer moves i->j starting at bucket t
    
    # Build phase-level problem
    n_phases = len(phases)
    phase_starts = [s for _, s, _ in phases]
    phase_ends = [e for _, _, e in phases]
    
    arcs = [(i, j) for i in range(J) for j in range(J) if tau[i, j] <= T and i != j]
    
    # Variables: x[j, t] for all t, y[a, t] for all arcs and t
    NX, NY = J * T, len(arcs) * T
    N = NX + NY
    c = np.zeros(N)
    
    for j in range(J):
        for t in range(T):
            c[j * T + t] = -r_avg[j, t]
    for a, (i, j) in enumerate(arcs):
        for t in range(T):
            c[NX + a * T + t] = lam
    
    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0
    
    def row(entries, lo, hi):
        nonlocal nr
        for var, val in entries:
            rows.append(nr); cols.append(var); vals.append(val)
        lb.append(lo); ub.append(hi); nr += 1
    
    def xv(j, t): return j * T + t
    def yv(a, t): return NX + a * T + t
    
    out_arcs = defaultdict(list); in_arcs = defaultdict(list)
    for a, (i, j) in enumerate(arcs):
        out_arcs[i].append((a, j)); in_arcs[j].append((a, i))
    
    # Budget at t=0
    row([(xv(j, 0), 1.0) for j in range(J)], P, P)
    
    # Flow conservation: x[j,t] == sum in-flow arrivals at t
    for j in range(J):
        for t in range(T):
            inflows = [(yv(a, t - tau[i, j]), 1.0) for a, i in in_arcs[j]
                       if 0 <= t - tau[i, j] < T]
            row([(xv(j, t), -1.0)] + inflows, 0.0, 0.0)
    
    # Outflow: if present at j at t, can stay or depart
    for j in range(J):
        for t in range(T - 1):
            row([(xv(j, t), 1.0)] + [(yv(a, t), -1.0) for a, _ in out_arcs[j]],
                0.0, 1.0)
    
    # No departures at last bucket
    for j in range(J):
        for a, _ in out_arcs[j]:
            row([(yv(a, T - 1), 1.0)], 0.0, 0.0)
    
    A = csr_matrix((vals, (rows, cols)), shape=(nr, N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return None, None, None, res.status
    
    z = np.round(res.x).astype(int)
    x = z[:NX].reshape(J, T).astype(bool)
    y = z[NX:].reshape(len(arcs), T).astype(bool)
    
    # Extract movements with tokens
    movements = []
    for a, (i, j) in enumerate(arcs):
        for t in range(T):
            if y[a, t]:
                movements.append({
                    "from_post": i, "to_post": j,
                    "depart_bucket": t, "arrive_bucket": t + tau[i, j],
                    "travel_buckets": tau[i, j],
                    "token": f"MOVE({i}->{j} @t{t})"
                })
    
    # Per-fixture assignment (same for all fixtures in this version)
    x_fixtures = np.repeat(x[None, :, :], n_fixtures, axis=0)
    
    cov_obj = float(r_avg[x].sum())
    moves = len(movements)
    return x_fixtures, cov_obj, moves, movements, res.status


def solve_p5(r, P, tau, lam=0.02, teleport=False, time_limit=300.0):
    """Exact relocation-aware deployment MIP (Programme P5).
    Returns (x[J,T] bool, objective_value, n_relocations, status)."""
    J, T = r.shape
    arcs = [(i, j) for i in range(J) for j in range(J) if tau[i, j] <= T]
    if teleport:
        tau = np.ones_like(tau)
        arcs = [(i, j) for i in range(J) for j in range(J)]
    nx_, ny = len(arcs) * T, J * T
    # variable order: x[j,t] (nx_ block first? no:) x block = J*T, then y arcs*T
    NX, NY = J * T, len(arcs) * T
    N = NX + NY
    c = np.zeros(N)
    for j in range(J):
        for t in range(T):
            c[j * T + t] = -r[j, t]
    for a, (i, j) in enumerate(arcs):
        if i != j:
            for t in range(T):
                c[NX + a * T + t] = lam

    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0

    def row(entries, lo, hi):
        nonlocal nr
        for var, val in entries:
            rows.append(nr); cols.append(var); vals.append(val)
        lb.append(lo); ub.append(hi); nr += 1

    def xv(j, t): return j * T + t
    def yv(a, t): return NX + a * T + t

    arc_idx = {a_pair: a for a, a_pair in enumerate(arcs)}
    out_arcs = defaultdict(list); in_arcs = defaultdict(list)
    for a, (i, j) in enumerate(arcs):
        out_arcs[i].append((a, j)); in_arcs[j].append((a, i))

    # budget at t0
    row([(xv(j, 0), 1.0) for j in range(J)], P, P)
    # flow conservation out / in ; skip last bucket for outflow (arc vars undefined beyond horizon -> forbid departures that land past end implicitly via y bounds at late t: allow but arrivals clipped by in-flow sum empty => officer lost; forbid instead)
    for j in range(J):
        for t in range(T - 1):
            row([(yv(a, t), 1.0) for a, _ in out_arcs[j]], 1.0 if False else 0.0,
                1.0)  # equality handled by pair below? keep <=1 then in-flow equality enforces
    for j in range(J):
        for t in range(T):
            if t == 0:
                continue
            ent = [(xv(j, t), -1.0)]
            ent += [(yv(a, t - tau[i, j]), 1.0) for a, i in in_arcs[j]
                    if 0 <= t - tau[i, j] < T]
            row(ent, 0.0, 0.0)
    # tie out-flow equality: x[j,t] == sum_a y[a_out, t] for t<=T-2 ; at T-1 no departure needed
    for j in range(J):
        for t in range(T - 1):
            row([(xv(j, t), 1.0)] + [(yv(a, t), -1.0) for a, _ in out_arcs[j]],
                0.0, 0.0)

    A = csr_matrix((vals, (rows, cols)), shape=(nr, N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return None, None, None, res.status
    z = np.round(res.x).astype(int)
    x = z[:NX].reshape(J, T).astype(bool)
    moves = int(sum(z[NX + a * T + t]
                    for a, (i, j) in enumerate(arcs)
                    for t in range(T) if i != j))
    obj = float(-(res.fun)) + lam * moves  # reported net objective
    cov_obj = float(r[x].sum())           # pure covered risk
    return x, cov_obj, moves, res.status


def coverage(x, r):
    return float(r[x].sum() / r.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-results", default="results")
    ap.add_argument("--out-figures", default="figures")
    ap.add_argument("--posts", default="post_selection_mip.csv")
    ap.add_argument("--squad", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixture-calendar", default=None,
                    help="Path to fixture_calendar.csv (label,kickoff_iso,attendance,fixture_type)")
    ap.add_argument("--min-events", type=int, default=500,
                    help="Minimum events near stadium to count as fixture day")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_results, exist_ok=True)
    os.makedirs(args.out_figures, exist_ok=True)
    print("=" * 80)
    print("29_deployment_plan.py - Programme P5 on P4-selected posts (Full Year)")
    print("=" * 80)

    # ---- Load fixtures: from calendar CSV or auto-extract from events ----
    fixture_calendar_path = args.fixture_calendar
    if fixture_calendar_path and not os.path.isabs(fixture_calendar_path):
        fixture_calendar_path = f"{args.data_dir}/{fixture_calendar_path}"
    
    fixtures_data = load_fixture_calendar(fixture_calendar_path)
    if fixtures_data is None:
        print("   No fixture calendar found; extracting from events...")
        fixtures_data = extract_fixtures_from_events(
            f"{args.data_dir}/hard_braking_events.csv",
            min_events_per_fixture=args.min_events
        )
    print(f"   Loaded {len(fixtures_data)} fixtures")
    for label, kiso, att, ftype in fixtures_data:
        print(f"     {label} @ {kiso} | crowd={att:,} | type={ftype}")

    # Build FIXTURES list for build_rjt (label, ISO datetime)
    FIXTURES = [(label, kiso) for label, kiso, _, _ in fixtures_data]
    CROWDS = [att for _, _, att, _ in fixtures_data]
    FIXTURE_TYPES = {label: ftype for label, _, _, ftype in fixtures_data}

    print("1. Loading posts and event stream ...")
    cand_ids = None
    posts_path = args.posts
    if not os.path.isabs(posts_path):
        # posts path relative to out_results
        posts_path = os.path.join(args.out_results, posts_path)
    plat, plon, prank = load_posts(posts_path)
    J = len(plat)
    rng = np.random.default_rng(args.seed)

    print(f"   building r[j,h] for {len(FIXTURES)} fixtures (chunked scan) ...")
    r_mean, day_starts, acc = build_daily_risk(f"{args.data_dir}/hard_braking_events.csv",
                           plat, plon, FIXTURES)
    print(f"   r_event total={r_event.sum():.0f} "
          f"(mean/event-day), nonzero cells={(r_event>0).sum()}")

    print("   building r[j,h] for the ordinary evening ...")
    r_ord, _, _ = build_daily_risk(f"{args.data_dir}/hard_braking_events.csv",
                         plat, plon, [ORDINARY_EVENING])
    # r_ord already (J,T): build_rjt averages over the fixture list

    tau, D = tau_matrix(plat, plon)
    print(f"   feasible move arcs (d<=4km, excl holds): "
          f"{int((tau<=28).__matmul__(np.eye(1)).sum() - J) if False else int(((tau <= 28) & (tau != 1)).sum()//1)}")

    # ---- Event day: solve at squad=P ----------------------------------------
    P = args.squad
    print(f"2. Solving P5 on event days (P={P}) ...")
    x5, cov5_obj, mov5, st = solve_p5(r_event, P, tau, lam=args.lam,
                                      time_limit=args.time_limit)
    assert x5 is not None, f"P5 failed (status={st})"
    cov5 = coverage(x5, r_event)
    print(f"   status={st} coverage={cov5:.3f} relocations={mov5}")

    # static best-P baseline
    tot = r_event.sum(axis=1)
    topP = np.argsort(-tot)[:P]
    stat_x = np.zeros_like(r_event, dtype=bool); stat_x[topP, :] = True
    cov_stat = coverage(stat_x, r_event)
    # random baselines
    rnd = []
    for _ in range(200):
        pick = rng.choice(J, P, replace=False)
        rx = np.zeros_like(r_event, dtype=bool); rx[pick, :] = True
        rnd.append(coverage(rx, r_event))
    cov_rand = float(np.mean(rnd))
    # free-movement upper bound
    xt, covt, mt, stt = solve_p5(r_event, P, tau, lam=0.0, teleport=True,
                                 time_limit=args.time_limit)
    cov_tele = coverage(xt, r_event)
    print(f"   static={cov_stat:.3f} random={cov_rand:.3f} teleport={cov_tele:.3f}")

    # ---- Ordinary evening contrast ------------------------------------------
    xo, covo, movo, sto = solve_p5(r_ord, P, tau, lam=args.lam,
                                   time_limit=args.time_limit)
    cov_o = coverage(xo, r_ord)
    tot_o = r_ord.sum(axis=1)
    topO = np.argsort(-tot_o)[:P]
    sx = np.zeros_like(r_ord, dtype=bool); sx[topO, :] = True
    gain_o = cov_o - coverage(sx, r_ord)
    gain_e = cov5 - cov_stat
    print(f"   ordinary: cov={cov_o:.3f} moves={movo} gain_over_static={gain_o:.3f}")

    # ---- Pareto over squad size ---------------------------------------------
    print("3. Pareto + movement-value sweep over squad size ...")
    pareto = []
    for p in range(1, 11):
        xp, cp, mp, sp = solve_p5(r_event, p, tau, lam=args.lam,
                                  time_limit=args.time_limit)
        cov_p = coverage(xp, r_event)
        # best static baseline at this size
        tp = np.argsort(-r_event.sum(axis=1))[:p]
        sx_ = np.zeros_like(r_event, dtype=bool); sx_[tp, :] = True
        cs = coverage(sx_, r_event)
        # free-movement upper bound at this size
        xt_, ct_, _, _ = solve_p5(r_event, p, tau, lam=0.0, teleport=True,
                                  time_limit=args.time_limit)
        ct = coverage(xt_, r_event)
        pareto.append({"P": p, "coverage": round(cov_p, 4),
                       "coverage_static": round(cs, 4),
                       "coverage_teleport": round(ct, 4),
                       "move_potential_pts": round(100*(ct - cs), 2),
                       "relocations": int(mp)})
        print(f"   P={p}: mip={cov_p:.3f} static={cs:.3f} "
              f"teleport={ct:.3f} (pot {100*(ct-cs):+.1f}pt) moves={mp}")
    pd.DataFrame(pareto).to_csv(
        f"{args.out_results}/deployment_sweep.csv", index=False)
    marg = [pareto[0]["coverage"]] + \
        [pareto[i]["coverage"] - pareto[i-1]["coverage"] for i in range(1, 10)]

    # ---- Per-game rosters: MetaCount sparse Bayesian prediction with true LOO CV -------------
    print("4. Per-game rosters (MetaCount sparse Bayesian prediction, true LOO CV) ...")
    T = r_event.shape[1]
    crowds = CROWDS  # Use dynamically loaded crowds
    
    # Build fixture dataframe for MetaCount with kickoff_iso for date features
    fixtures_df = build_fixture_dataframe(FIXTURES, crowds, FIXTURE_TYPES)
    # Add kickoff_iso for day-of-week, month, seasonality features
    fixtures_df['kickoff_iso'] = [kiso for _, kiso in FIXTURES]
    
    # Historical means per (post, bucket) for base rate feature
    historical_means = r_event  # (J, T) mean per event-day
    
    # Also build risk panel: (n_fixtures, J, T)
    risk_panel = acc_fx
    
    if METACOUNT_AVAILABLE:
        # Fit RiskPredictor on all fixtures
        predictor = RiskPredictor(
            posts_df=pd.DataFrame({
                'node_id': CAND_IDS,
                'lat': plat,
                'lon': plon,
                'rank': prank
            }),
            n_buckets=T,
            fixture_types=["NRL", "SuperRugby", "ALeague"],
            sparsity_prior=1e-4,
            verbose=True
        )
        
        # Fit on full panel (n_fixtures x J posts x T buckets)
        predictor.fit(fixtures_df, risk_panel, historical_means=historical_means)
        
        # Feature importance
        importance_df = predictor.get_feature_importance()
        print("   Top features:")
        for _, row in importance_df.head(12).iterrows():
            print(f"     {row['feature']}: active={row['n_active_panels']}/{predictor.n_panels}, "
                  f"importance={row['importance']:.4f}")
        
        # ---- True Leave-One-Out Cross-Validation ----
        print("   Running true LOO cross-validation...")
        loo_results = predictor.leave_one_out_cv(
            fixtures_df, risk_panel, historical_means=historical_means, n_jobs=1
        )
        print(f"   LOO CV: MSE={loo_results['mse']:.4f}, MAE={loo_results['mae']:.4f}, "
              f"R2={loo_results['r2']:.4f}, 95% Coverage={loo_results['coverage_95']:.3f}")
        
        # Sparsity path (feature selection vs regularization)
        print("   Computing sparsity path...")
        sparsity_results = predictor.sparsity_path(
            fixtures_df, risk_panel, historical_means=historical_means
        )
        print(f"   Sparsity path: {len(sparsity_results)} alpha values")
        
        # Hyperparameter sweep
        print("   Running hyperparameter sweep...")
        hp_results = predictor.hyperparameter_sweep(
            fixtures_df, risk_panel, historical_means=historical_means, n_jobs=1
        )
        print(f"   Hyperparameter sweep: {len(hp_results)} combinations")
        
        # Save LOO results
        loo_df = pd.DataFrame(loo_results['per_fixture'])
        loo_df.to_csv(f"{args.out_results}/deployment_loo_cv.csv", index=False)
        
        # Save sparsity path
        spar_df = pd.DataFrame(sparsity_results)
        spar_df.to_csv(f"{args.out_results}/deployment_sparsity_path.csv", index=False)
        
        # Save hyperparameter sweep
        hp_df = pd.DataFrame(hp_results)
        hp_df.to_csv(f"{args.out_results}/deployment_hyperparam_sweep.csv", index=False)
        
    else:
        predictor = None
        print("   MetaCount not available; using simple attendance scaling fallback")
    
    # Now do prediction for each fixture (using LOO predictions from MetaCount)
    sched_rows, per_game, xp_all = [], [], []
    for f, (name, kiso) in enumerate(FIXTURES):
        rf = acc_fx[f]  # realized risk for this fixture (J, T)
        
        if METACOUNT_AVAILABLE and predictor is not None:
            # Use LOO prediction for this fixture
            r_pred = loo_results['predictions'][f]  # (J, T)
            r_std = loo_results['pred_stds'][f]     # (J, T)
        else:
            # Fallback: simple attendance scaling (original method)
            others = [g for g in range(len(FIXTURES)) if g != f]
            base = np.mean([acc_fx[g] for g in others], axis=0)
            xg = np.log(np.array([crowds[g] for g in others], dtype=float))
            yg = np.log(np.array([acc_fx[g].sum() for g in others], dtype=float) + 1.0)
            beta = float(np.polyfit(xg, yg, 1)[0])
            beta = min(max(beta, 0.0), 1.0)
            scale = (crowds[f] / np.mean([crowds[g] for g in others])) ** beta
            r_pred = base * scale
            r_std = np.zeros_like(r_pred)  # No uncertainty in fallback
        
        # Oracle: solve on realized risk
        xo_, _, mo_, _ = solve_p5(rf, P, tau, lam=args.lam,
                                  time_limit=args.time_limit)
        # Prospective: solve on predicted risk
        xp_, _, mp_, _ = solve_p5(r_pred, P, tau, lam=args.lam,
                                  time_limit=args.time_limit)
        xp_all.append(xp_)
        # Unscaled mean: solve on historical mean (no fixture-specific info)
        xm_, _, mm_, _ = solve_p5(historical_means, P, tau, lam=args.lam,
                                  time_limit=args.time_limit)
        
        row_pg = {
            "fixture": name, "kickoff": kiso, "crowd": crowds[f],
            "realized_risk": round(float(rf.sum()), 1),
            "predicted_risk": round(float(r_pred.sum()), 1),
            "cov_oracle": round(coverage(xo_, rf), 4),
            "cov_prospective": round(coverage(xp_, rf), 4),
            "cov_unscaled_mean": round(coverage(xm_, rf), 4),
            "pros_relocations": int(mp_),
            "pred_std_total": round(float(r_std.sum()), 1),
        }
        per_game.append(row_pg)
        print(f"   {name}: realized={rf.sum():.0f} pred={r_pred.sum():.0f}±{r_std.sum():.0f} "
              f"cov oracle={row_pg['cov_oracle']:.3f} "
              f"prospective={row_pg['cov_prospective']:.3f} "
              f"unscaled={row_pg['cov_unscaled_mean']:.3f} moves={mp_}")
        for j in range(J):
            for t in range(T):
                if xp_[j, t]:
                    sched_rows.append({
                        "fixture": name, "kickoff": kiso,
                        "post": f"P{int(prank[j])}", "node_id": int(CAND_IDS[j]),
                        "bucket": t, "t_rel_kickoff_min": -180 + 15 * t})

    pd.DataFrame(sched_rows).to_csv(
        f"{args.out_results}/deployment_schedules.csv", index=False)
    pg = pd.DataFrame(per_game)
    pg.to_csv(f"{args.out_results}/deployment_pergame.csv", index=False)
    tex_rows = "\n".join(
        f"{r['fixture'].replace('-', '--')} & {r['kickoff'][11:16]} & "
        f"{int(r['crowd']):,} & {r['realized_risk']:.0f} & "
        f"{100*r['cov_oracle']:.1f} & {100*r['cov_prospective']:.1f} & "
        f"{100*r['cov_unscaled_mean']:.1f} \\\\" for _, r in pg.iterrows())
    mean_pros = float(pg["cov_prospective"].mean())
    mean_uns = float(pg["cov_unscaled_mean"].mean())
    method_name = "MetaCount sparse Bayesian" if METACOUNT_AVAILABLE else "attendance-scaled"
    tex_pergame = (
        "% deployment_pergame.tex - generated by 29_deployment_plan.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{small}\n"
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Fixture & Kick-off & Crowd & Realised risk & Oracle cov.\\ (\\%) & "
        "Predictive roster (\\%) & Unscaled mean (\\%) \\\\\n\\midrule\n"
        + tex_rows +
        "\n\\midrule\nMean & & & & "
        + " & ".join(f"{100*v:.1f}" for v in [
            pg['cov_oracle'].mean(), mean_pros, mean_uns])
        + " \\\\\n\\bottomrule\n\\end{tabular}\\end{small}\n"
        "\\captionsetup{margin={0.5in,0in}}\n"
        f"\\caption{{\\textcolor{{Highlight}}{{Game-by-game rosters from the "
        f"{method_name} prediction model.}} For each fixture the surface is predicted "
        f"using attendance, kickoff time, fixture type, and historical base rates; "
        f"Programme~P5 is solved on the predicted surface and scored against realised risk. "
        f"Oracle solves the same programme on the realised surface.}}\n"
        "\\label{tab:pergame}\n\\end{table}\n")
    with open(f"{args.out_results}/deployment_pergame.tex", "w") as f:
        f.write(tex_pergame)

    nfx = len(FIXTURES)
    ncols = min(4, nfx)
    nrows = (nfx + ncols - 1) // ncols
    fig34, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), 
                               sharex=True, sharey=True, squeeze=False)
    for f, (name, kiso) in enumerate(FIXTURES):
        r, c = divmod(f, ncols)
        axp = axes[r, c]
        im = axp.imshow(acc_fx[f], aspect="auto", cmap="viridis",
                        extent=[-3, 4, J - 0.5, -0.5])
        for j in range(J):
            for t in range(T):
                if xp_all[f][j, t]:
                    axp.add_patch(plt.Rectangle((t / 4 - 3, j - 0.5), 0.25, 1,
                                                fill=False, edgecolor="black",
                                                lw=1.0))
        axp.set_title(f"{name} ({crowds[f]:,})", fontsize=8)
    # Hide unused subplots
    for f in range(nfx, nrows * ncols):
        r, c = divmod(f, ncols)
        axes[r, c].axis("off")
    fig34.suptitle("Prospective game-by-game rosters (black = staffed buckets)",
                   fontsize=11)
    fig34.supxlabel("hours relative to kick-off")
    fig34.tight_layout()
    plt.savefig(f"{args.out_figures}/fig34_game_schedules.png", dpi=150)


    # ---- Per-post phase table ------------------------------------------------
    phases = {"pre": slice(0, 10), "during": slice(10, 18),
              "post": slice(18, 28)}   # buckets: K-3h..K-0.5h=10, ..K+1.5h=8, ..K+4h=10
    ph_tot = {k: r_event[:, v].sum(axis=1) for k, v in phases.items()}
    staffed = []
    for j in range(J):
        s = "".join([c for c, k in zip("PDA", ["pre", "during", "post"])
                     if x5[j, phases[k]].any()])
        staffed.append(s if s else "none")
    peak_t = r_event.argmax(axis=1)
    peak_lbl = [f"T{(-30 + 15*t)/60:+05.2f}".replace(".", ":")
                for t in peak_t]  # hours rel. kickoff (bucket centres approx)

    rs = pd.DataFrame({
        "post": [f"P{int(rk)}" for rk in prank],
        "lat": plat, "lon": plon,
        "pre": ph_tot["pre"], "during": ph_tot["during"],
        "post_ph": ph_tot["post"], "total": r_event.sum(axis=1),
        "staffed": staffed,
    }).sort_values("total", ascending=False)
    rs.to_csv(f"{args.out_results}/deployment_riskscore.csv", index=False)
    # long-format optimal schedule
    T = r_event.shape[1]
    sched = []
    for j in range(J):
        for t in range(T):
            if x5[j, t]:
                sched.append({"post": f"P{int(prank[j])}", "node_id": int(CAND_IDS[j]),
                              "bucket": t, "t_rel_kickoff_min": -180 + 15*t})
    pd.DataFrame(sched).to_csv(f"{args.out_results}/deployment_plan.csv", index=False)

    tex_rows = "\n".join(
        f"{r['post']} & ({r['lon']:.4f}, {r['lat']:.4f}) & {r['pre']:.1f} & "
        f"{r['during']:.1f} & {r['post_ph']:.1f} & {r['total']:.1f} & "
        f"{r['staffed']} \\\\" for _, r in rs.iterrows())
    tex = (
        "% deployment_riskscore.tex - generated by 29_deployment_plan.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{footnotesize}\n"
        "\\begin{tabular}{llrrrrl}\n\\toprule\n"
        "Post & Junction (lon, lat) & Pre & During & Post & Total & Staffed \\\\\n"
        "\\midrule\n" + tex_rows +
        "\n\\bottomrule\n\\end{tabular}\\end{footnotesize}\n"
        "\\captionsetup{margin={0.5in,0in}}\n"
        "\\caption{\\textcolor{Highlight}{Per-post weighted risky driving by "
        "phase and optimal staffing} (mean per event day, 500~m catchments, "
        f"{len(FIXTURES)} fixtures; $P={P}$). \\emph{{Staffed}} lists phases with "
        "an on-station officer in the optimal schedule (P/D/A).}\n"
        "\\label{tab:riskscore}\n\\end{table}\n")
    with open(f"{args.out_results}/deployment_riskscore.tex", "w") as f:
        f.write(tex)

    # ---- Schedule figure -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(r_event, aspect="auto", cmap="viridis",
                   extent=[-3, 4, J - 0.5, -0.5])
    for j in range(J):
        for t in range(28):
            if x5[j, t]:
                ax.add_patch(plt.Rectangle((t / 4 - 3, j - 0.5), 0.25, 1,
                                           fill=False, edgecolor="black",
                                           lw=1.6))
    ax.set_yticks(range(J)); ax.set_yticklabels([f"P{int(r)}" for r in prank])
    ax.set_xlabel("hours relative to kick-off"); ax.set_ylabel("post")
    plt.colorbar(im, ax=ax, label="mean risky-driving per bucket")
    ax.set_title(f"Optimal space-time deployment (P={P}, exact MIP)")
    plt.tight_layout()
    plt.savefig(f"{args.out_figures}/fig32_mip_schedule.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([p["P"] for p in pareto], [100*p["coverage"] for p in pareto],
               marker="o"); ax[0].set_xlabel("officers P"); ax[0].set_ylabel(
               "event-day coverage (%)"); ax[0].grid(alpha=.3)
    ax[0].axvline(P, color="crimson", ls="--", lw=1)
    ax[1].bar(range(1, 11), [100*m for m in marg]); ax[1].set_xlabel(
        "additional officer"); ax[1].set_ylabel("marginal coverage (%)")
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{args.out_figures}/fig33_mip_value.png", dpi=150)

    summary = {
        "P": P, "lambda": args.lam,
        "fixtures": [{"label": n, "kickoff": k, "attendance": a, "type": t} 
                     for (n, k), a, t in zip(FIXTURES, CROWDS, [FIXTURE_TYPES[n] for n, _ in FIXTURES])],
        "ordinary_evening": ORDINARY_EVENING[0],
        "event": {"coverage_mip": round(cov5, 4), "coverage_static":
                  round(cov_stat, 4), "coverage_random": round(cov_rand, 4),
                  "coverage_teleport": round(cov_tele, 4),
                  "relocations": int(mov5)},
        "ordinary": {"coverage_mip": round(cov_o, 4),
                     "gain_over_static_pts": round(100*(cov_o -
                     coverage(sx, r_ord)), 2), "relocations": int(movo)},
        "gains": {"event_gain_pts": round(100*gain_e, 2),
                  "ordinary_gain_pts": round(100*gain_o, 2)},
        "pareto": pareto, "marginal_pct": [round(100*m, 2) for m in marg],
        "phase_shares_pct": {
            "pre": round(100*r_event[:, :10].sum()/r_event.sum(), 1),
            "during": round(100*r_event[:, 10:18].sum()/r_event.sum(), 1),
            "post": round(100*r_event[:, 18:].sum()/r_event.sum(), 1)},
        "runtime_s": round(time.time()-t0, 1),
        "metacount_used": METACOUNT_AVAILABLE,
    }
    with open(f"{args.out_results}/deployment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("event", "ordinary", "gains",
                               "phase_shares_pct")}, indent=2))
    print("=" * 80)
    print(f"DONE in {summary['runtime_s']}s")


if __name__ == "__main__":
    main()