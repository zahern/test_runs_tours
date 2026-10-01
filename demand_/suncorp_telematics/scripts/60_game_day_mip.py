"""
60_game_day_mip.py - segment-based police MIP on predicted game-day surfaces.

Consumes the future-game predictions from 59_game_day_search_predict.py
(results/game_day_pred_15min.parquet: fixture x segment x 15-min bin) and solves,
for every future fixture, which road-segment midpoint posts to staff:

  (1) each officer is a PLAN: hold one site all window, or move once between two
      sites (travel time gap)  ->  <=1 relocation per officer
  (2) squad size P_g is solved per game by an absolute marginal-coverage rule
      (keep adding officers while the marginal officer still covers at least
      --min-marg predicted Compass speeding events)
    (3) segment-level coverage varies over the game through the selected plans
  (4) coverage is measured on the predicted surface (deployment decision), and
      reported at sample scale and penetration-scaled network scale

MIP (maximal covering over plans):
    max  sum_{j,t} r_{jt} c_{jt}
    s.t. c_{jt} <= sum_{k covers (j,t)} u_k ;  c_{jt} <= 1 ;
         sum_k u_k <= P ;  u_k, c_{jt} in {0,1}
with r_{jt} = f_j * pred_{jt} the *observable* speeding,
f_j = min(1, obs_m / length_j): one officer sees ~obs_m metres, so a
200 m segment yields only half its events.
solved exactly with HiGHS (scipy.optimize.milp).

Inputs:
  results/game_day_pred_15min.parquet     (from 59)
    data/osm_roads_suncorp.geojson          (road geometries for exact midpoints)
Outputs:
  results/game_day_mip_sites.csv          per game: P, sites, relocations, coverage
  results/game_day_mip_plans.csv          per game: every officer's plan cells
    results/game_day_mip_schedule.csv       per game/officer/15-min segment state
  results/game_day_mip_summary.json       audit + headline numbers
  results/game_day_mip.tex                LaTeX table
  figures/game_day_mip_map.png            predicted site load + selected sites
  figures/game_day_mip_schedules.png      officer tokens per game (<=1 move)
  figures/game_day_mip_coverage.png       coverage vs squad size curves
  figures/game_day_mip_summary.png        squad size / coverage bars

Usage:
  python scripts/60_game_day_mip.py --pilot
    python scripts/60_game_day_mip.py [--min-marg 0.5] [--max-segments 250]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIG = PROJ / "figures"
PRED = RES / "game_day_pred_15min.parquet"
GEOJSON = DATA / "osm_roads_suncorp.geojson"
TZ = "Australia/Brisbane"
REL_BINS = 28
# Kickoff sits 3 h after the window opens (-3h..+4h at 15-min bins), so
# bin 12. Defined here rather than borrowed from stage 59, which is where
# WINDOW_HOURS lives; this stage only ever sees the 28-bin axis.
KICKOFF_BIN = 12
PMAX = 10
MOVE_BINS = [4, 8, 12, 16, 20, 24]
OFFICER_COLORS = None


def haversine_m(a1, o1, a2, o2):
    radius = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2 - p1, np.radians(o2 - o1)
    return 2 * radius * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2
                                          + np.cos(p1) * np.cos(p2)
                                          * np.sin(dl / 2) ** 2))


def load_segment_sites(pred, path, max_segments, obs_m=100.0):
    """Create candidates at exact road-segment centerline midpoints.

    Each site also carries the road name, segment length and the
    observable fraction f = min(1, obs_m / length_m): one officer sees
    ~obs_m metres, so a 200 m segment yields only half its speeding.
    """
    import geopandas as gpd

    roads = gpd.read_file(path).to_crs(28356)
    roads["road_osmid"] = pd.to_numeric(roads["osmid"], errors="coerce")
    roads = roads.dropna(subset=["road_osmid"]).copy()
    roads["road_osmid"] = roads["road_osmid"].astype(np.int64)
    roads = roads.drop_duplicates("road_osmid").reset_index(drop=True)
    roads["length_m"] = roads.geometry.length
    midpoint = roads.geometry.interpolate(0.5, normalized=True).to_crs(4326)
    _nm = roads["name"].fillna("").astype(str) if "name" in roads.columns else ""
    _rf = roads["ref"].fillna("").astype(str) if "ref" in roads.columns else ""
    _hw = roads["highway"].fillna("").astype(str) if "highway" in roads.columns else ""
    sites = pd.DataFrame({
        "road_osmid": roads["road_osmid"].to_numpy(),
        "lat": midpoint.y.to_numpy(float),
        "lon": midpoint.x.to_numpy(float),
        "length_m": roads["length_m"].to_numpy(float),
        "road_name": [
            (n.strip() or r.strip() or f"unnamed {_h}".strip())
            for n, r, _h in zip(
                _nm if not isinstance(_nm, str) else [""] * len(roads),
                _rf if not isinstance(_rf, str) else [""] * len(roads),
                _hw if not isinstance(_hw, str) else [""] * len(roads))],
    })
    sites["obs_frac"] = np.minimum(
        1.0, float(obs_m) / np.maximum(sites["length_m"].to_numpy(float), 1e-9))
    score = pred.groupby("road_osmid")["pred_speeding"].sum()
    sites = sites.merge(score.rename("pred_total"), on="road_osmid", how="inner")
    # Rank candidates by *observable* speeding (what police can capture).
    sites["pred_observable"] = sites["pred_total"] * sites["obs_frac"]
    sites = sites.sort_values(["pred_observable", "road_osmid"],
                              ascending=[False, True]).reset_index(drop=True)
    if max_segments > 0 and len(sites) > max_segments:
        sites = sites.head(int(max_segments)).copy()
    sites = sites.reset_index(drop=True)
    sites["site_label"] = [f"S{i + 1}" for i in range(len(sites))]
    return sites


def build_plans(site_lat, site_lon, speed_mps=12.0, setup_min=5.0,
                max_m=4000.0, detour_factor=1.3, max_per_site=1):
    """Enumerate hold and relocation plans.

    max_per_site is the number of officers a single hold plan may host. The
    default of 1 preserves the shipped behaviour exactly. Raising it lets
    the capacity-aware (fractional) arm post two or three officers at one
    long, under-observed site, which is the only way that arm can express
    "more people on the hot street". Relocation plans always hold exactly
    one officer, so they are never scaled.
    """
    n = len(site_lat)
    tau = np.full((n, n), 999, dtype=int)
    for i in range(n):
        dist = haversine_m(site_lat[i], site_lon[i], site_lat, site_lon)
        for j in range(n):
            if i == j:
                tau[i, j] = 1
                continue
            road_dist = float(dist[j]) * float(detour_factor)
            if road_dist <= max_m:
                minutes = setup_min + road_dist / 1000.0 / speed_mps * 60.0
                tau[i, j] = max(2, int(np.ceil(minutes / 15.0)))
    plans = []
    for j in range(n):
        plans.append({"cells": [(j, t) for t in range(REL_BINS)],
                      "move": False, "a": j, "b": j, "t1": None,
                      "cap": int(max(1, max_per_site))})
    for i in range(n):
        for j in range(n):
            if i == j or tau[i, j] > REL_BINS:
                continue
            for t1 in MOVE_BINS:
                arrival = t1 + int(tau[i, j])
                if arrival >= REL_BINS:
                    continue
                cells = ([(i, t) for t in range(t1)]
                         + [(j, t) for t in range(arrival, REL_BINS)])
                plans.append({"cells": cells, "move": True, "a": i, "b": j,
                              "t1": t1, "cap": 1})
    cover_by_cell = {}
    for k, plan in enumerate(plans):
        for cell in plan["cells"]:
            cover_by_cell.setdefault(cell, []).append(k)
    return plans, cover_by_cell, tau


def fixture_variation(surfaces, top_n=(5, 10, 20)):
    """How much of the predicted surface actually differs between games?

    WHY THIS IS REPORTED. Job 26107921 returned distinct_site_sets = 1: the
    same four streets for all six fixtures, which makes a "game-day" plan a
    static street list. Measured on that run's surfaces, the cause is the
    model, not the optimiser:

      * variance of log1p(pred) per segment x fixture splits 92.0% between
        segments, 5.4% between fixtures, 2.6% residual
      * the top-10 and top-20 site sets are 100% IDENTICAL across all six
        fixtures, and the argmax site is the same segment every time

    So the optimiser cannot choose different sites -- it is given the same
    ranking every time. Meanwhile the TIMING does differ: the mean
    off-diagonal correlation of the per-fixture relative-bin profiles is 0.095,
    and the peak bin splits 7-8 (pre-kickoff arrival) for the 19:00 NRL games
    against 27 (end of window) for the others. Officers at the same streets on
    a different clock is still a usable plan, but "where" is not a game-day
    decision and the summary should say so rather than leave it implied by
    distinct_site_sets.

    Returns a dict for game_day_mip_summary.json. Purely diagnostic: nothing
    here changes the optimisation.
    """
    out = {"note": "diagnostic only; does not affect the optimisation"}
    keys = list(surfaces.keys())
    if len(keys) < 2:
        out["reason"] = "need at least 2 fixtures to compare"
        return out
    mats = {k: np.asarray(surfaces[k], dtype=float) for k in keys}
    # Per (fixture, site) totals on the original scale for ranking...
    tot = {k: m.sum(axis=1) for k, m in mats.items()}
    # ...and on a log scale for the variance split, because the raw spread is
    # dominated by the few biggest arterials.
    lg = {k: np.log1p(m).mean(axis=1) for k, m in mats.items()}
    common = sorted(set.intersection(*[set(np.nonzero(t)[0].tolist())
                                       for t in tot.values()]))
    if len(common) < 2:
        out["reason"] = "fixtures do not share enough segments to compare"
        return out
    L = np.column_stack([lg[k][common] for k in keys])
    grand = L.mean()
    seg_means = L.mean(axis=1)
    fx_means = L.mean(axis=0)
    total_ss = float(((L - grand) ** 2).sum())
    if total_ss > 0:
        out["variance_between_segments_pct"] = round(
            100 * float(((seg_means - grand) ** 2).sum()) * L.shape[1]
            / total_ss, 1)
        out["variance_between_fixtures_pct"] = round(
            100 * float(((fx_means - grand) ** 2).sum()) * L.shape[0]
            / total_ss, 1)

    # Ranking stability: do the same sites come out on top every time?
    jac = {}
    for n in top_n:
        tops = [set(np.argsort(-tot[k])[:n].tolist()) for k in keys]
        vals = []
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                vals.append(len(tops[i] & tops[j])
                            / max(len(tops[i] | tops[j]), 1))
        jac["top%d" % n] = round(float(np.mean(vals)), 3) if vals else None
    out["site_overlap_jaccard"] = jac
    out["same_argmax_for_all_fixtures"] = bool(
        len({int(np.argmax(tot[k])) for k in keys}) == 1)

    # Timing: is the relative-bin profile game-specific?
    prof = {k: np.asarray(mats[k], dtype=float).sum(axis=0) for k in keys}
    C = np.column_stack([prof[k] / max(prof[k].sum(), 1e-12) for k in keys])
    cc = np.corrcoef(C)
    off = [cc[i, j] for i in range(len(keys)) for j in range(len(keys)) if i < j]
    out["timing_profile_corr_mean"] = round(float(np.mean(off)), 3) if off \
        else None
    out["peak_rel_bin_per_fixture"] = {
        str(k): int(np.argmax(prof[k])) for k in keys}
    distinct_peaks = len(set(out["peak_rel_bin_per_fixture"].values()))
    out["timing_is_game_specific"] = bool(distinct_peaks > 1)
    if out.get("same_argmax_for_all_fixtures"):
        out["interpretation"] = (
            "the same streets win every fixture, so WHERE to post is not a "
            "game-day decision; check timing_is_game_specific for whether WHEN "
            "still is")
    return out


def solve_mclp(r, plans, cover_by_cell, squad):
    cells = [(j, t) for j in range(r.shape[0]) for t in range(r.shape[1])
             if r[j, t] > 0]
    if not cells:
        return [], 0.0
    cell_index = {cell: n for n, cell in enumerate(cells)}
    n_plans, n_cells = len(plans), len(cells)
    n_total = n_plans + n_cells
    cost = np.zeros(n_total)
    for n, cell in enumerate(cells):
        cost[n_plans + n] = -r[cell]
    rows, cols, vals, lower, upper = [], [], [], [], []
    n_rows = 0

    def add_row(entries, lo, hi):
        nonlocal n_rows
        for col, val in entries:
            rows.append(n_rows)
            cols.append(col)
            vals.append(val)
        lower.append(lo)
        upper.append(hi)
        n_rows += 1
    for cell, n in cell_index.items():
        ks = cover_by_cell.get(cell, [])
        add_row([(n_plans + n, 1.0)] + [(k, -1.0) for k in ks],
                -np.inf, 0.0)
    add_row([(k, 1.0) for k in range(n_plans)], 0.0, float(squad))
    matrix = csr_matrix((vals, (rows, cols)), shape=(n_rows, n_total))
    result = milp(c=cost, constraints=LinearConstraint(matrix, lower, upper),
                  integrality=np.ones(n_total), bounds=Bounds(0, 1),
                  options={"time_limit": 60.0, "mip_rel_gap": 1e-4})
    if result.x is None:
        return [], 0.0
    chosen = np.round(result.x).astype(int)
    selected = [k for k in range(n_plans) if chosen[k]]
    covered = float(sum(r[cell] for n, cell in enumerate(cells)
                        if chosen[n_plans + n]))
    return selected, covered


def solve_mclp_fractional(r_raw, obs_frac, plans, cover_by_cell, squad,
                         time_limit=60.0):
    """Capacity-aware covering MIP: coverage is FRACTIONAL, not binary.

    WHY THIS EXISTS. The binary formulation uses c_jt in {0,1}, so a second
    officer posted at an already-staffed site contributes EXACTLY nothing --
    the cell is already covered. Combined with r being pre-scaled by a fixed
    per-site obs_frac, that makes holding strictly dominate relocating on
    every fixture (observed: relocations = 0 across all six games on job
    26075201), and it is structural, not a threshold artefact.

    Here the covered FRACTION of a cell grows with the number of officers
    present, capped at 1:

        f_jt  <=  obs_frac_j * sum_k u_k     (linear in the decision vars)
        f_jt  <=  1
        max   sum_jt  r_raw_jt * f_jt
        s.t.  sum_k u_k <= P,  u_k in {0,1},  f in [0,1]

    so a second officer at a long, under-observed site is worth something,
    and splitting an officer's window between two such sites can beat
    concentrating them. Whether it ACTUALLY does is an empirical question --
    mobility_ab() answers it rather than assuming it.

    r_raw must be the UNSCALED surface; obs_frac enters here instead.

    Officers per plan: a plan's `cap` says how many officers it can host.
    A hold plan is allowed several (posting two officers at a long,
    under-observed site is a real deployment option and is exactly what the
    binary objective cannot represent), while a relocation is inherently one
    officer, so move plans keep cap 1. So u_k is INTEGER in [0, cap_k],
    not binary.

    Returns (selected_counts, covered_in_raw_units) where selected_counts is
    a list of (plan_index, n_officers).
    """
    cells = [(j, t) for j in range(r_raw.shape[0])
             for t in range(r_raw.shape[1]) if r_raw[j, t] > 0]
    if not cells:
        return [], 0.0
    cell_index = {cell: n for n, cell in enumerate(cells)}
    n_plans, n_cells = len(plans), len(cells)
    n_total = n_plans + n_cells
    cost = np.zeros(n_total)
    for n, cell in enumerate(cells):
        cost[n_plans + n] = -float(r_raw[cell])
    rows, cols, vals, lower, upper = [], [], [], [], []
    n_rows = 0

    def add_row(entries, lo, hi):
        nonlocal n_rows
        for col, val in entries:
            rows.append(n_rows)
            cols.append(col)
            vals.append(val)
        lower.append(lo)
        upper.append(hi)
        n_rows += 1

    for cell, n in cell_index.items():
        j = cell[0]
        ks = cover_by_cell.get(cell, [])
        # f_n - obs_frac_j * sum_k u_k <= 0
        add_row([(n_plans + n, 1.0)]
                + [(k, -float(obs_frac[j])) for k in ks], -np.inf, 0.0)
    add_row([(k, 1.0) for k in range(n_plans)], 0.0, float(squad))
    matrix = csr_matrix((vals, (rows, cols)), shape=(n_rows, n_total))
    # plans: integer officers, bounded per plan; coverage fraction continuous
    integrality = np.zeros(n_total)
    integrality[:n_plans] = 1
    hi = np.ones(n_total)
    for k, p in enumerate(plans):
        hi[k] = float(p.get("cap", 1))
    result = milp(c=cost, constraints=LinearConstraint(matrix, lower, upper),
                  integrality=integrality, bounds=Bounds(0, hi),
                  options={"time_limit": float(time_limit),
                           "mip_rel_gap": 1e-4})
    if result.x is None:
        return [], 0.0
    x = result.x
    selected = [(k, int(round(x[k]))) for k in range(n_plans)
                if round(x[k]) > 0]
    covered = float(sum(float(r_raw[cell]) * float(x[n_plans + n])
                        for n, cell in enumerate(cells)))
    return selected, covered


def mobility_ab(r_scaled, r_raw, obs_frac, plans, cover_by_cell, squad,
                time_limit=60.0):
    """Hold-only (binary) vs capacity-aware (fractional) for one game.

    Reports coverage and relocations for both so the movement question is
    answered with a number instead of an assumption.

      r_scaled : surface already multiplied by obs_frac, i.e. exactly what
                 the shipped binary objective sees, so that arm stays
                 byte-identical to the current behaviour
      r_raw    : the same surface BEFORE obs_frac, needed by the
                 fractional arm so officer count can scale the covered
                 fraction
    """
    out = {"squad": int(squad)}
    sel_b, cov_b = solve_mclp(r_scaled, plans, cover_by_cell, squad)
    sel_f, cov_f = solve_mclp_fractional(r_raw, obs_frac, plans, cover_by_cell,
                                        squad, time_limit=time_limit)
    for tag, sel, cov in (("binary", sel_b, cov_b),
                          ("fractional", sel_f, cov_f)):
        if tag == "binary":
            relocs = sum(plans[k]["move"] for k in sel)
            n_off = len(sel)
        else:
            # sel_f is [(plan_index, n_officers)]
            relocs = sum(n for k, n in sel if plans[k]["move"])
            n_off = sum(n for _k, n in sel)
        out[f"{tag}_relocations"] = int(relocs)
        out[f"{tag}_covered"] = round(float(cov), 3)
        out[f"{tag}_n_officers"] = int(n_off)
    out["coverage_gain_pct"] = round(
        100.0 * (out["fractional_covered"] - out["binary_covered"])
        / max(out["binary_covered"], 1e-9), 2)
    return out


def coverage_of(selected, r, plans):
    covered = set()
    for k in selected:
        covered.update(plans[k]["cells"])
    total = float(r.sum())
    return float(sum(r[j, t] for (j, t) in covered) / total) if total > 0 else 0.0


def prune_plans(plans, r, max_moves):
    """Cap move plans per fixture by standalone covered value.

    The full plan set is ~O(n_sites^2 x |MOVE_BINS|) (117k at 150 sites),
    which is far too large for a 60s MILP once cells are non-empty. Keep all
    stationed plans plus the top ``max_moves`` moves by their standalone
    coverage value; the MIP is then exact over this candidate set.
    """
    stationed = [k for k, p in enumerate(plans) if not p["move"]]
    moves = [k for k, p in enumerate(plans) if p["move"]]
    if max_moves > 0 and len(moves) > int(max_moves):
        val = np.empty(len(moves), dtype=float)
        for n, k in enumerate(moves):
            val[n] = sum(r[c] for c in plans[k]["cells"])
        order = np.argsort(-val, kind="stable")[:int(max_moves)]
        moves = [moves[i] for i in np.sort(order)]
    keep = stationed + moves
    remap = {old: new for new, old in enumerate(keep)}
    new_plans = [plans[k] for k in keep]
    new_cover = {}
    for k in keep:
        for cell in plans[k]["cells"]:
            new_cover.setdefault(cell, []).append(remap[k])
    return new_plans, new_cover


def solve_sweep(r, plans, cover_by_cell, pmax):
    """Solve the covering MIP for every squad size 1..pmax.

    One MILP per P (<=60s each); the headline min-marg rule and the
    requested per-P plans are both derived from this single sweep so no
    squad size is ever solved twice.
    """
    sols, curve, previous = {}, [], 0.0
    for squad in range(1, int(pmax) + 1):
        selected, covered = solve_mclp(r, plans, cover_by_cell, squad)
        marginal = covered - previous
        curve.append({"squad": squad, "covered": covered,
                      "marginal": marginal})
        sols[squad] = (selected, covered)
        previous = covered
    return sols, curve


def select_squad(curve, min_marg, min_marg_frac, penetration=0.05):
    """Walk the sweep, adding officers while the marginal gain is worth it.

    TWO independent stop reasons, either sufficient:

      * absolute floor -- marginal/penetration < min_marg (0 disables).
        Kept for backwards compatibility.
      * relative floor -- marginal < min_marg_frac * covered_sample
        (0 disables). This is the DEFAULT path and the only one that
        works, because the absolute form cannot be rescued by tuning:
        marginals at P=10 run 19-56 events while a sensible threshold is a
        small number, and the six fixtures span 30k-83k total events, so any
        single constant over-invests the small games or under-invests the
        big ones. On job 26075201 `min_marg 0.5` sat 372-1120x below every
        marginal, never fired, and pinned every fixture to the sweep
        ceiling.

    Both sides of the relative test are SAMPLE scale, so penetration
    cancels. Dividing the marginal by penetration and comparing it against a
    sample-scale coverage mixes units and makes the gate ~penetration times
    too strict -- an earlier version of this did exactly that and never
    fired at any threshold.

    Returns (squad, stop_reason).
    """
    chosen = 1
    reason = "reached sweep ceiling"
    for entry in curve:
        p = int(entry["squad"])
        if p < 2:
            chosen = p
            continue
        net = entry["marginal"] / max(penetration, 1e-9)
        if min_marg and min_marg > 0 and net < min_marg:
            reason = f"absolute floor (net {net:.1f} < {min_marg:g})"
            break
        cov = entry.get("covered", entry.get("covered_sample", 0.0))
        if min_marg_frac and min_marg_frac > 0 and cov > 0 and \
                entry["marginal"] < min_marg_frac * cov:
            share = 100.0 * entry["marginal"] / cov
            reason = (f"relative floor (marginal {share:.1f}% of covered "
                      f"< {100 * min_marg_frac:.0f}%)")
            break
        chosen = p
    return chosen, reason


def pick_squad(r, plans, cover_by_cell, min_marg, pmax=None, penetration=0.05,
               min_marg_frac=0.0):
    """Headline squad via the marginal-coverage rule, network scale."""
    sols, curve = solve_sweep(r, plans, cover_by_cell, pmax or PMAX)
    chosen_p, reason = select_squad(curve, min_marg, min_marg_frac,
                                    penetration)
    chosen_plans, coverage = [], 0.0
    for p in range(1, chosen_p + 1):
        chosen_plans, coverage = sols[p][0], sols[p][1]
    return chosen_p, chosen_plans, coverage, curve, reason


def draw_selected_heatmap(ax, fx, r_all, chosen, labels, sites, tau,
                          officer_colors, rel_bins=REL_BINS, site_row=None):
    """One panel: ONLY the staffed sites, by predicted load, with occupancy.

    Split out of the plotting block so it can be exercised on synthetic
    inputs. The previous schedules figure plotted every candidate row of the
    surface, so with --max-segments 250 and a squad of 5 the chosen rows were
    5 lines among 250 -- effectively invisible. Here the rows ARE the
    deployment, which is what the figure is for.
    """
    if chosen is None or len(chosen) == 0:
        ax.axis("off")
        return None
    used = []
    for _, prow in chosen.iterrows():
        for lbl in (str(prow["from_site"]), str(prow["to_site"])):
            if lbl in labels and labels.index(lbl) not in used:
                used.append(labels.index(lbl))
    if not used:
        ax.axis("off")
        return None
    used = sorted(used, key=lambda j: -r_all[j].sum())
    sub = r_all[used, :]
    ax.imshow(sub, aspect="auto", cmap="magma",
              vmin=0.0, vmax=max(float(np.percentile(sub, 99)), 1e-9),
              interpolation="nearest")
    row_of = {j: n for n, j in enumerate(used)}
    ko = KICKOFF_BIN
    for officer, (_, prow) in enumerate(chosen.iterrows()):
        color = officer_colors[officer % len(officer_colors)]
        a = row_of[labels.index(str(prow["from_site"]))]
        moving = bool(prow.get("move", False))
        if not moving:
            ax.axhline(a, color=color, lw=1.7, alpha=0.95)
        else:
            b = row_of[labels.index(str(prow["to_site"]))]
            t1 = int(prow["move_bin"])
            arr = int(prow["arrival_bin"]) if prow["arrival_bin"] != "" \
                else t1
            ax.plot([0, t1], [a, a], color=color, lw=1.7)
            ax.plot([t1, rel_bins - 1], [b, b], color=color, lw=1.7)
            ax.plot([t1, max(arr, t1 + 1)], [a, b], color=color, lw=1.0,
                    ls=":", alpha=0.85)
        ax.text(rel_bins - 0.4, a, f" {officer + 1}", va="center", ha="right",
                fontsize=7.5, color=color)
    ax.set_yticks(range(len(used)))
    ax.set_yticklabels(
        [f"{labels[j]}  {str(sites.iloc[j].get('road_name', ''))[:26]}"
         for j in used], fontsize=6.5)
    ticks = [t for t in (0, ko // 2, ko, ko + 6, ko + 12, rel_bins - 1)
             if 0 <= t < rel_bins]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [("KO" if t == ko else f"{t - ko:+d}b") for t in ticks], fontsize=6.5)
    ax.axvline(ko - 0.5, color="cyan", lw=1.0, ls="--", alpha=0.95)
    nrel = 0
    cov = float("nan")
    if site_row is not None:
        try:
            nrel = int(site_row["relocations"])
            cov = 100 * float(site_row["coverage"])
        except Exception:  # noqa: BLE001
            pass
    ax.set_title(f"{str(fx)[:34]}\nP={len(chosen)}  coverage {cov:.1f}%  "
                 f"relocations {nrel}", fontsize=8)
    return used


def parse_squads(text, pmax):
    """Parse --squads 'all' or comma list into sorted squad sizes."""
    if str(text).strip().lower() == "all":
        return list(range(1, int(pmax) + 1))
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    out = sorted({p for p in out if 1 <= p <= int(pmax)})
    return out or [int(pmax)]


def solution_rows(fx, squad, selected, covered, total, plans_fx, sites,
                    labels, tau, penetration):
    """Site/plan/schedule dicts for one (fixture, squad) solution."""
    moves = sum(plans_fx[k]["move"] for k in selected)
    used_sites = sorted({plans_fx[k]["a"] for k in selected}
                        | {plans_fx[k]["b"] for k in selected})
    coverage = covered / total if total > 0 else 0.0
    site_row = {
        "fixture": fx, "squad": squad, "relocations": int(moves),
        "n_sites": len(used_sites),
        "sites": "|".join(labels[j] for j in used_sites),
        "segment_osmids": "|".join(str(int(sites.iloc[j]["road_osmid"]))
                                    for j in used_sites),
        "road_names": "|".join(str(sites.iloc[j].get("road_name", ""))
                               for j in used_sites),
        "site_latlon": "|".join(
            f"{float(sites.iloc[j]['lat']):.5f},{float(sites.iloc[j]['lon']):.5f}"
            for j in used_sites),
        "obs_frac": "|".join(f"{float(sites.iloc[j]['obs_frac']):.2f}"
                             for j in used_sites),
        "pred_events_sample": round(total, 3),
        "pred_events_network": round(total / max(penetration, 1e-9), 1),
        "covered_sample": round(covered, 3),
        "coverage": round(coverage, 4)}
    plan_rows, schedule_rows = [], []
    for officer, k in enumerate(selected):
        plan = plans_fx[k]
        plan_rows.append({
            "fixture": fx, "squad": squad, "officer": officer + 1,
            "from_site": labels[plan["a"]],
            "to_site": labels[plan["b"]],
            "from_road_osmid": int(sites.iloc[plan["a"]]["road_osmid"]),
            "to_road_osmid": int(sites.iloc[plan["b"]]["road_osmid"]),
            "from_road_name": str(sites.iloc[plan["a"]].get("road_name", "")),
            "to_road_name": str(sites.iloc[plan["b"]].get("road_name", "")),
            "move": bool(plan["move"]),
            "move_bin": plan["t1"] if plan["move"] else "",
            "arrival_bin": (plan["t1"] + int(tau[plan["a"], plan["b"]])
                            if plan["move"] else ""),
            "cells": len(plan["cells"])})
        occupied = {(int(j), int(t)) for j, t in plan["cells"]}
        for rel_bin in range(REL_BINS):
            cell = next((j for j in range(len(sites))
                         if (j, rel_bin) in occupied), None)
            if cell is None:
                schedule_rows.append({
                    "fixture": fx, "squad": squad, "officer": officer + 1,
                    "rel_bin": rel_bin, "state": "moving",
                    "road_osmid": "", "segment": ""})
            else:
                schedule_rows.append({
                    "fixture": fx, "squad": squad, "officer": officer + 1,
                    "rel_bin": rel_bin, "state": "stationed",
                    "road_osmid": int(sites.iloc[cell]["road_osmid"]),
                    "segment": labels[cell],
                    "road_name": str(sites.iloc[cell].get("road_name", ""))})
    return site_row, plan_rows, schedule_rows


def main():
    global OFFICER_COLORS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(PRED))
    ap.add_argument("--min-marg", type=float, default=0.0,
                    help="secondary stop rule in ABSOLUTE predicted speeding "
                         "events on the network scale (the sample marginal is "
                         "divided by --penetration). 0 = disabled, which is "
                         "the default because this form cannot be calibrated: "
                         "on job 26075201 it was 0.5 against marginals of "
                         "19-56 events, i.e. 372-1120x under the threshold, "
                         "so it never fired and the ceiling decided. Use "
                         "--min-marg-frac instead")
    ap.add_argument("--obs-m", type=float, default=100.0,
                    help="metres of road one officer can observe; a segment "
                         "contributes min(1, obs-m / length-m) of its speeding")
    ap.add_argument("--max-squad", type=int, default=0,
                    help="largest squad size the sweep may consider "
                         "(0 = PMAX = 10). This used to be a bare module "
                         "constant, so no run could ever consider more than "
                         "10 officers. One MILP is solved per P, so the "
                         "budget scales with this value")
    ap.add_argument("--mobility-ab", default="off", choices=["on", "off"],
                    help="per game, also solve a CAPACITY-AWARE fractional-"
                         "coverage MIP and report both arms. The shipped "
                         "binary objective makes a second officer at an "
                         "already-staffed site worth exactly zero (the cell "
                         "is already covered), so relocating never wins and "
                         "relocations came out 0 on all six games of job "
                         "26075201 -- structural, not a threshold. The "
                         "fractional arm lets covered fraction grow with "
                         "officers present, capped at 1, and reports whether "
                         "movement earns its place")
    ap.add_argument("--max-officers-per-site", type=int, default=1,
                    help="officers a single hold plan may host, used ONLY by "
                         "the --mobility-ab capacity-aware arm. Default 1 "
                         "reproduces the shipped one-officer-per-site view. "
                         "Relocation plans are always 1 officer")
    ap.add_argument("--ab-time-limit", type=float, default=60.0,
                    help="per-MILP time limit in seconds for the mobility A/B")
    ap.add_argument("--min-marg-frac", type=float, default=0.10,
                    help="DEFAULT stop rule: keep adding officers while the "
                         "marginal coverage is at least this fraction of the "
                         "coverage already achieved. Scale-free, so it "
                         "behaves the same on a 30k-event game and an "
                         "83k-event one. 0.10 selects P=5-6 and ~15%% "
                         "coverage on the current surfaces; 0.05 reaches the "
                         "ceiling. Set 0 to disable")
    ap.add_argument("--squads", default="2,4,6,8,10",
                    help="squad sizes to emit full per-P plans for "
                         "('all' = 1..PMAX); headline table still uses "
                         "--min-marg")
    ap.add_argument("--penetration", type=float, default=0.05)
    ap.add_argument("--max-segments", type=int, default=250,
                    help="top predicted segment-midpoint candidates; 0 = all")
    ap.add_argument("--max-move-plans", type=int, default=3000,
                    help="cap move plans per fixture by standalone value "
                         "(0 = keep all; full set is ~O(n^2) and blows the "
                         "MILP budget)")
    ap.add_argument("--travel-speed-kmh", type=float, default=43.2,
                    help="police moving speed used for relocation feasibility")
    ap.add_argument("--detour-factor", type=float, default=1.3,
                    help="road-distance multiplier applied to midpoint distance")
    ap.add_argument("--setup-min", type=float, default=5.0,
                    help="setup/parking time added to each relocation")
    ap.add_argument("--max-travel-m", type=float, default=4000.0,
                    help="maximum permitted road distance for a relocation")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--allow-concentrated", action="store_true",
                    help="run the MIP even when one segment holds >50% of "
                         "predicted speeding (degenerate coverage)")
    ap.add_argument("--out-dir", default=str(RES))
    ap.add_argument("--fig-dir", default=str(FIG))
    args = ap.parse_args()
    if args.pilot:
        args.limit_games = args.limit_games or 2
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "pilot" if args.pilot else "full"

    print("[60/1] loading predictions + sites...")
    pred = pd.read_parquet(args.pred)
    pred["bin"] = pd.to_datetime(pred["bin"], utc=True).dt.tz_convert(TZ)
    pred["road_osmid"] = pd.to_numeric(pred["road_osmid"], errors="coerce")
    pred = pred.dropna(subset=["road_osmid"]).copy()
    pred["road_osmid"] = pred["road_osmid"].astype(np.int64)
    _ct = pred.groupby("road_osmid")["pred_speeding"].sum().sort_values(ascending=False)
    _top1 = float(_ct.iloc[0] / max(_ct.sum(), 1e-12))
    print(f"   concentration: top segment {int(_ct.index[0])} holds "
          f"{_top1:.1%} of predicted speeding")
    if _top1 > 0.50 and not args.allow_concentrated:
        raise SystemExit(
            f"REFUSING MIP: top segment holds {_top1:.1%} of predicted speeding; "
            "coverage would be degenerate (one site ~= all coverage). Refit the "
            "search model with winsorized geometry (59) or pass --allow-concentrated")
    sites = load_segment_sites(pred, GEOJSON, int(args.max_segments),
                               obs_m=float(args.obs_m))
    if len(sites) < 2:
        raise SystemExit("need at least two predicted segment candidates")
    site_lat = sites["lat"].to_numpy(float)
    site_lon = sites["lon"].to_numpy(float)
    labels = sites["site_label"].tolist()
    seg_map = dict(zip(sites["road_osmid"], sites.index))
    pred["site"] = pred["road_osmid"].map(seg_map)
    pred = pred.dropna(subset=["site"]).copy()
    pred["site"] = pred["site"].astype(int)

    plans, cover_by_cell, tau = build_plans(
        site_lat, site_lon,
        speed_mps=float(args.travel_speed_kmh) / 3.6,
        setup_min=float(args.setup_min), max_m=float(args.max_travel_m),
        detour_factor=float(args.detour_factor),
        max_per_site=int(args.max_officers_per_site))
    print(f"   {len(sites)} sites, {len(plans)} plans "
          f"({len(plans) - len(sites)} single-move)")

    fixtures = list(pred["fixture"].unique())
    if args.limit_games:
        fixtures = fixtures[:args.limit_games]
    print(f"[60/2] solving MIP per game ({len(fixtures)} fixtures)...")
    # The squad ceiling is now a flag. It used to be the bare constant PMAX,
    # so officers above 10 were never even considered and the headline was
    # structurally pinned to 10 on every fixture.
    pmax = int(args.max_squad) if args.max_squad else int(PMAX)
    sweep_squads = parse_squads(args.squads, pmax)
    print(f"   per-P sweep squads: {sweep_squads} "
          f"(1 MILP per P per game, ceiling P={pmax})")
    site_rows, plan_rows, schedule_rows, curves = [], [], [], {}
    mobility_rows = []
    perP_site_rows, perP_plan_rows, perP_sched_rows, curve_rows = [], [], [], []
    surfaces, raws = {}, {}
    for fx in fixtures:
        sub = pred[pred["fixture"] == fx]
        # Keep the UNSCALED surface: the binary objective consumes the
        # obs_frac-scaled version (unchanged behaviour), while the
        # capacity-aware arm needs the raw values so that the covered
        # FRACTION can grow with the number of officers at a site.
        r_raw = (sub.groupby(["site", "rel_bin"])["pred_speeding"].sum()
                 .unstack(fill_value=0.0)
                 .reindex(index=range(len(sites)), columns=range(REL_BINS),
                          fill_value=0.0).to_numpy(float))
        # Observation constraint: scale each site's speeding by the
        # fraction one officer can actually see (f_j <= 1).
        obs_frac = sites["obs_frac"].to_numpy(float)
        r = r_raw * obs_frac[:, None]
        surfaces[fx] = r
        raws[fx] = r_raw
        plans_fx, cover_fx = prune_plans(plans, r, args.max_move_plans)
        sols, curve = solve_sweep(r, plans_fx, cover_fx, pmax)
        curves[fx] = curve
        total = float(r.sum())
        # Headline squad via the min-marg rule (backward compatible).
        # min_marg is compared on the NETWORK scale (divide the sample-scale
        # marginal by --penetration) to match the flag's documented units;
        # the un-scaled comparison never fires because sample-scale
        # marginals are orders of magnitude above any useful threshold,
        # which pinned squad to PMAX on every fixture.
        # Headline squad from the SHARED select_squad rule, so the sweep loop
        # and pick_squad() can never disagree about which P was chosen.
        squad, stop_reason = select_squad(
            curve, args.min_marg, args.min_marg_frac, args.penetration)
        selected, covered = 1, 0.0
        for p in range(1, squad + 1):
            selected, covered = sols[p]
        site_row, prows, srows = solution_rows(
            fx, squad, selected, covered, total, plans_fx, sites,
            labels, tau, args.penetration)
        site_rows.append(site_row)
        plan_rows.extend(prows)
        schedule_rows.extend(srows)
        for entry in curve:
            curve_rows.append({
                "fixture": fx, "squad": entry["squad"],
                "covered_sample": round(entry["covered"], 3),
                "marginal": round(entry["marginal"], 3),
                "coverage": round(entry["covered"] / total, 4)
                if total > 0 else 0.0})
        for p in sweep_squads:
            sel_p, cov_p = sols[p]
            srow, grows, grows_sched = solution_rows(
                fx, p, sel_p, cov_p, total, plans_fx, sites,
                labels, tau, args.penetration)
            perP_site_rows.append(srow)
            perP_plan_rows.extend(grows)
            perP_sched_rows.extend(grows_sched)
        moves = site_row["relocations"]
        coverage = site_row["coverage"]
        # Mobility A/B: does allowing capacity-aware fractional coverage
        # make relocating worthwhile? Reported, not assumed -- if the
        # fractional arm never wins, then holding is genuinely right at this
        # squad size and that is the answer.
        if str(getattr(args, "mobility_ab", "off")).lower() == "on":
            try:
                _ab = mobility_ab(r, raws[fx], obs_frac, plans_fx, cover_fx,
                                  squad, time_limit=float(args.ab_time_limit))
                _ab["fixture"] = str(fx)
                mobility_rows.append(_ab)
                print(f"      mobility A/B: binary cov {_ab['binary_covered']:.0f} "
                      f"reloc {_ab['binary_relocations']} | fractional cov "
                      f"{_ab['fractional_covered']:.0f} reloc "
                      f"{_ab['fractional_relocations']} | gain "
                      f"{_ab['coverage_gain_pct']:+.2f}%")
            except Exception as exc:  # noqa: BLE001
                print(f"      WARN: mobility A/B failed ({exc})")
        print(f"   {str(fx)[:40]:40s} P={squad} moves={moves} "
              f"sites={site_row['n_sites']} "
              f"coverage={100 * coverage:.1f}%  [{stop_reason}]")
        sweep_status = " ".join(
            f"P={p}:{100 * sols[p][1] / total:.1f}%"
            if total > 0 else f"P={p}:0.0%"
            for p in sweep_squads)
        print(f"      per-P solved coverage: {sweep_status}")
    site_df = pd.DataFrame(site_rows)
    plan_df = pd.DataFrame(plan_rows)
    schedule_df = pd.DataFrame(schedule_rows)
    # Degenerate-solution guard: empty frames have no columns, which would
    # raise KeyError('fixture') in the figure block and skip the remaining
    # figures. Coerce the join/filter columns so plots degrade gracefully.
    for _c in ("fixture", "officer", "from_site", "to_site", "move",
               "move_bin"):
        if _c not in plan_df.columns:
            plan_df[_c] = pd.Series(dtype=object)
    if "fixture" not in schedule_df.columns:
        schedule_df["fixture"] = pd.Series(dtype=object)
    site_df.to_csv(out_dir / "game_day_mip_sites.csv", index=False)
    plan_df.to_csv(out_dir / "game_day_mip_plans.csv", index=False)
    schedule_df.to_csv(out_dir / "game_day_mip_schedule.csv", index=False)
    # Full per-P sweep: every requested squad size gets deployable plans.
    perP_site_df = pd.DataFrame(perP_site_rows)
    perP_plan_df = pd.DataFrame(perP_plan_rows)
    perP_sched_df = pd.DataFrame(perP_sched_rows)
    curve_df = pd.DataFrame(curve_rows)
    for _c in ("fixture", "squad", "officer", "from_site", "to_site",
               "move", "move_bin"):
        if _c not in perP_plan_df.columns:
            perP_plan_df[_c] = pd.Series(dtype=object)
    if "fixture" not in perP_sched_df.columns:
        perP_sched_df["fixture"] = pd.Series(dtype=object)
    perP_site_df.to_csv(out_dir / "game_day_mip_sites_perP.csv", index=False)
    perP_plan_df.to_csv(out_dir / "game_day_mip_plans_perP.csv", index=False)
    perP_sched_df.to_csv(out_dir / "game_day_mip_schedule_perP.csv",
                         index=False)
    curve_df.to_csv(out_dir / "game_day_mip_curves.csv", index=False)
    if mobility_rows:
        _mdf = pd.DataFrame(mobility_rows)
        _mdf.to_csv(out_dir / "game_day_mip_mobility_ab.csv", index=False)
        print("   mobility A/B -> game_day_mip_mobility_ab.csv "
              f"({len(_mdf)} rows)")
        _mr = _mdf["fractional_relocations"].sum()
        _mg = float(_mdf["coverage_gain_pct"].mean())
        print(f"   mobility summary: fractional relocations total {_mr} "
              f"(binary {int(_mdf['binary_relocations'].sum())}), "
              f"mean coverage gain {_mg:+.2f}%")
        summary_mobility = {
            "fractional_relocations_total": int(_mr),
            "binary_relocations_total": int(_mdf["binary_relocations"].sum()),
            "mean_coverage_gain_pct": round(_mg, 3),
        }
    else:
        summary_mobility = None
    print(f"   per-P sweep: {len(perP_site_df)} site rows, "
          f"{len(perP_plan_df)} plan rows, {len(curve_df)} curve rows")
    summary = {
        "n_fixtures": len(fixtures),
        "sweep_squads": list(sweep_squads),
        "mean_squad": round(float(site_df["squad"].mean()), 2),
        "squad_range": [int(site_df["squad"].min()),
                        int(site_df["squad"].max())],
        "mean_coverage": round(float(site_df["coverage"].mean()), 4),
        "mean_relocations": round(float(site_df["relocations"].mean()), 2),
        "distinct_site_sets": int(site_df["sites"].nunique()),
        "fixture_variation": fixture_variation(surfaces),
        "penetration": args.penetration,
        "min_marg": args.min_marg,
        "min_marg_scale": "network",
        "segment_based": True,
        "candidate_segments": len(sites),
        "candidate_cap": int(args.max_segments),
        "travel_speed_kmh": float(args.travel_speed_kmh),
        "detour_factor": float(args.detour_factor),
        "setup_min": float(args.setup_min),
        "max_travel_m": float(args.max_travel_m),
        "obs_m": float(args.obs_m),
        "mobility_ab": summary_mobility,
    }
    (out_dir / "game_day_mip_summary.json").write_text(
        json.dumps({"summary": summary, "per_game": site_rows}, indent=2,
                   default=str))

    body = "\n".join(
        f"{i + 1} & {str(r['fixture'])[:34]} & {r['squad']} & "
        f"{r['relocations']} & {r['n_sites']} & {r['sites']} & "
        f"{100 * r['coverage']:.1f} \\\\"
        for i, (_, r) in enumerate(site_df.iterrows()))
    tex = (
        "% game_day_mip.tex - generated by 60_game_day_mip.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{scriptsize}\n"
        "\\begin{tabular}{rllrrrr}\n\\toprule\n"
        "\\# & Fixture & $P_g$ & Moves & Sites & Staffed sites & Cov (\\%) \\\\\n"
        "\\midrule\n" + body + "\n\\midrule\n"
        f"\\multicolumn{{2}}{{l}}{{Mean ({len(site_df)} games)}} & "
        f"{site_df['squad'].mean():.1f} & {site_df['relocations'].mean():.1f} & "
        f"{site_df['n_sites'].mean():.1f} & & "
        f"{100 * site_df['coverage'].mean():.1f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\\end{scriptsize}\n"
        "\\caption{Game-day police site selection solved on the predicted "
        "15-minute speeding surface (MetaCount search + NB refit). Each officer "
        "relocates at most once; squad size and sites are solved per game. "
        "Coverage is the share of observable predicted speeding events on "
        "covered site x 15-min cells (one officer sees "
        f"{float(args.obs_m):.0f} m per segment). Road names, midpoints and "
        "observable fractions per staffed site are in "
        "game_day_mip_sites.csv.}\n\\label{tab:gamedaymip}\\end{table}\n")
    (out_dir / "game_day_mip.tex").write_text(tex)

    print("[60/3] figures...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        OFFICER_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
        n_fx = len(fixtures)
        ncols = min(3, n_fx)
        nrows = int(np.ceil(n_fx / ncols))
        # Road-segment map: draw each candidate segment itself, coloured by
        # observable predicted speeding (log scale); selected sites get a
        # thick green outline + label. Falls back to bubbles if geopandas
        # is unavailable.
        try:
            import geopandas as _gpd
            _roads = _gpd.read_file(GEOJSON).to_crs(4326)
            _roads["road_osmid"] = pd.to_numeric(
                _roads["osmid"], errors="coerce").astype("Int64")
            _roads = _roads.dropna(subset=["road_osmid"]).copy()
            _roads["road_osmid"] = _roads["road_osmid"].astype(np.int64)
            _roads = _roads.drop_duplicates("road_osmid").reset_index(drop=True)

            def _coords_of(g):
                try:
                    return list(g.coords)
                except Exception:  # noqa: BLE001 - multi-part
                    try:
                        return list(g.geoms[0].coords)
                    except Exception:  # noqa: BLE001
                        return None

            # BACKGROUND NETWORK: every feature in the extract, not just the
            # --max-segments candidates. The old map drew only the candidate
            # rows, so 250 isolated OSM way fragments appeared with no
            # continuity and read as a broken network. Drawing all 7,825
            # features underneath gives a connected network for context; the
            # candidate overlay then shows what the model actually scored.
            _bg = []
            for _g in _roads.geometry:
                _c = _coords_of(_g)
                if _c and len(_c) >= 2:
                    _bg.append([(float(a), float(b)) for a, b in _c])
            _geom = {int(o): g for o, g in
                     zip(_roads["road_osmid"], _roads.geometry)}
            _have_geom = True
            print(f"   map background: {len(_bg):,} road features from "
                  # GEOJSON is a pathlib.Path, which has no .rsplit -- calling it
            # raised AttributeError and the whole map silently degraded to
            # bubbles on every run, including job 26107921:
            #   WARN: segment map geometries unavailable ('PosixPath' object
            #          has no attribute 'rsplit'); using bubbles
            f"{GEOJSON.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: segment map geometries unavailable ({exc}); "
                  "using bubbles")
            _geom, _have_geom, _bg = {}, False, []
        from matplotlib.colors import LogNorm
        from matplotlib.collections import LineCollection
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows),
                                 squeeze=False)
        for i, fx in enumerate(fixtures):
            ax = axes[i // ncols][i % ncols]
            r = surfaces[fx]
            totals = r.sum(axis=1)
            row = site_df[site_df["fixture"] == fx].iloc[0]
            selected = set(row["sites"].split("|")) if row["sites"] else set()
            _sel_idx = {labels.index(s) for s in selected if s in labels}
            _vmax = max(float(totals.max()), 1e-9)
            if _have_geom:
                if _bg:
                    # zorder 1: the whole study-area network, faint, so the
                    # panel reads as a road network rather than loose fragments
                    ax.add_collection(LineCollection(
                        _bg, colors="#B8BEC4", linewidths=0.45,
                        alpha=0.55, zorder=1))
                _segs, _vals, _sel = [], [], []
                for j in range(len(sites)):
                    _g = _geom.get(int(sites.iloc[j]["road_osmid"]))
                    if _g is None:
                        continue
                    try:
                        _coords = list(_g.coords)
                    except Exception:  # noqa: BLE001 - multi-part: first part
                        try:
                            _coords = list(_g.geoms[0].coords)
                        except Exception:
                            continue
                    _segs.append(_coords)
                    _vals.append(float(totals[j]))
                    _sel.append(j in _sel_idx)
                _lc = LineCollection(
                    _segs, cmap="magma", norm=LogNorm(vmin=max(min(_vals + [1e-3]), 1e-3),
                                                      vmax=_vmax),
                    array=np.array(_vals), linewidths=1.2, zorder=2)
                ax.add_collection(_lc)
                for j in range(len(sites)):
                    if j not in _sel_idx:
                        continue
                    _g = _geom.get(int(sites.iloc[j]["road_osmid"]))
                    if _g is None:
                        continue
                    try:
                        _xs, _ys = _g.xy
                    except Exception:
                        try:
                            _xs, _ys = _g.geoms[0].xy
                        except Exception:
                            continue
                    ax.plot(list(_xs), list(_ys), color="#0B6E4F", lw=3.0,
                            solid_capstyle="round", zorder=3)
                    ax.annotate(labels[j], (site_lon[j], site_lat[j]),
                                fontsize=6, ha="center", va="center",
                                color="white",
                                bbox=dict(fc="#0B6E4F", ec="none", pad=1),
                                zorder=4)
                ax.set_aspect(1.0 / max(np.cos(np.radians(
                    float(np.mean(site_lat)))), 1.0))  # ~equal at this latitude
                if _bg:
                    _allx = [p[0] for seg in _bg for p in seg]
                    _ally = [p[1] for seg in _bg for p in seg]
                    ax.set_xlim(min(_allx), max(_allx))
                    ax.set_ylim(min(_ally), max(_ally))
                else:
                    ax.autoscale_view()
            else:
                _lmax = max(np.log10(totals + 1.0).max(), 1e-9)
                for j in range(len(sites)):
                    ax.scatter(site_lon[j], site_lat[j],
                               s=12 + 260 * (np.log10(totals[j] + 1.0) / _lmax),
                               c="#0B6E4F" if labels[j] in selected else "#BBBBBB",
                               alpha=0.85, edgecolors="k", linewidths=0.4, zorder=3)
                    ax.annotate(labels[j], (site_lon[j], site_lat[j]), fontsize=5,
                                ha="center", va="center", zorder=4)
                ax.set_aspect(1.0 / max(np.cos(np.radians(
                    float(np.mean(site_lat)))), 1.0))
                ax.autoscale_view()
            # A grid over a map is noise; drop it for the map panels only.
            ax.grid(False)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{str(fx)[:34]}\nP={int(row['squad'])}, "
                         f"cov={100 * row['coverage']:.0f}%", fontsize=8)
            ax.set_xlabel("lon", fontsize=7)
            ax.set_ylabel("lat", fontsize=7)
            if _have_geom and _vals:
                sm = plt.cm.ScalarMappable(
                    cmap="magma",
                    norm=LogNorm(vmin=max(min(_vals + [1e-3]), 1e-3),
                                 vmax=max(max(_vals), 1e-9)))
                cb = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
                cb.set_label("observable predicted speeding", fontsize=6)
                cb.ax.tick_params(labelsize=5)
        for j in range(n_fx, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("Study-area road network (grey) with candidate segments "
                     "coloured by observable\npredicted speeding (log scale); "
                     "staffed sites outlined green", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "game_day_mip_map.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.4 * nrows),
                                 squeeze=False)
        for i, fx in enumerate(fixtures):
            ax = axes[i // ncols][i % ncols]
            r = surfaces[fx]
            sns.heatmap(r, ax=ax, cmap="Greys", cbar=False,
                        vmax=max(np.percentile(r, 99) * 1.3, 1e-9))
            chosen = plan_df[plan_df["fixture"] == fx]
            for officer, (_, prow) in enumerate(chosen.iterrows()):
                color = OFFICER_COLORS[officer % 10]
                if not prow["move"]:
                    ax.plot(range(REL_BINS),
                            [labels.index(prow["from_site"])] * REL_BINS,
                            "-", color=color, lw=1.1)
                else:
                    t1 = int(prow["move_bin"])
                    a = labels.index(prow["from_site"])
                    b = labels.index(prow["to_site"])
                    move = int(tau[a, b])
                    ax.plot(range(t1), [a] * t1, "-", color=color, lw=1.1)
                    ax.plot(range(t1 + move, REL_BINS),
                            [b] * (REL_BINS - t1 - move), "-", color=color,
                            lw=1.1)
                    if t1 > 0:
                        ax.plot([t1 - 1, t1 + move], [a, b], ":", color=color,
                                lw=0.8)
            ax.set_yticks(range(len(sites)))
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_title(str(fx)[:38], fontsize=8)
            ax.set_xlabel("15-min bin")
        for j in range(n_fx, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("Officer schedules on predicted surfaces (<=1 relocation each)")
        fig.tight_layout()
        fig.savefig(fig_dir / "game_day_mip_schedules.png", dpi=130)
        plt.close(fig)

        # ---- selected-site heatmap -------------------------------------
        # The schedules figure above plots all `--max-segments` candidate
        # rows, of which only P are ever chosen, so 240 of 250 rows per
        # panel are dead weight and the chosen rows are hard to find. This
        # figure shows ONLY the sites that were actually staffed, ordered by
        # predicted load, with each officer's occupancy drawn on top -- so
        # you can see, per game, what the smaller squad is doing and where.
        ncols2 = min(2, max(len(fixtures), 1))
        nrows2 = int(np.ceil(len(fixtures) / ncols2))
        fig, axes = plt.subplots(
            nrows2, ncols2,
            figsize=(7.2 * ncols2, max(2.8, 0.55 * PMAX + 2.0) * nrows2),
            squeeze=False)
        for i, fx in enumerate(fixtures):
            ax = axes[i // ncols2][i % ncols2]
            _row = site_df[site_df["fixture"] == fx]
            draw_selected_heatmap(
                ax, fx, surfaces[fx],
                plan_df[plan_df["fixture"] == fx], labels, sites, tau,
                OFFICER_COLORS,
                site_row=_row.iloc[0] if len(_row) else None)
        for j in range(len(fixtures), nrows2 * ncols2):
            axes[j // ncols2][j % ncols2].axis("off")
        fig.suptitle(
            "Staffed sites only: predicted speeding by site x 15-min bin, "
            "with officer occupancy\n(coloured line = holding, dotted = "
            "relocating, dashed cyan = kickoff)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(fig_dir / "game_day_mip_selected_heatmap.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for fx in fixtures:
            curve = pd.DataFrame(curves[fx])
            axes[0].plot(curve["squad"], curve["covered"], "o-",
                         label=str(fx)[:24], lw=1.1, ms=3)
        axes[0].set_xlabel("squad size P")
        axes[0].set_ylabel("predicted speeding events covered")
        axes[0].set_title("Cumulative speeding coverage by squad size")
        axes[0].legend(fontsize=6, ncols=2)
        _marg = curve_df.pivot_table(index="squad", columns="fixture",
                                     values="marginal", aggfunc="mean")
        _x = np.arange(len(_marg.index))
        _w = min(0.8 / max(len(fixtures), 1), 0.35)
        for k, fx in enumerate(_marg.columns):
            axes[1].bar(_x + (k - (len(fixtures) - 1) / 2) * _w,
                        _marg[fx].to_numpy(float), width=_w,
                        label=str(fx)[:24])
        axes[1].set_xticks(_x)
        axes[1].set_xticklabels([int(s) for s in _marg.index])
        axes[1].set_xlabel("squad size P")
        axes[1].set_ylabel("marginal speeding events at P")
        axes[1].set_title("What each extra squad sees (marginal, speeding)")
        axes[1].legend(fontsize=6, ncols=2)
        fig.suptitle("Squad-size sweep on predicted speeding (future games)")
        fig.tight_layout()
        fig.savefig(fig_dir / "game_day_mip_coverage.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(range(len(site_df)), site_df["squad"], color="#0B6E4F")
        axes[0].set_xticks(range(len(site_df)))
        axes[0].set_xticklabels([str(f)[:18] for f in site_df["fixture"]],
                                rotation=45, ha="right", fontsize=6)
        axes[0].set_ylabel("squad size")
        axes[0].set_title("Officers per game")
        axes[1].bar(range(len(site_df)), 100 * site_df["coverage"],
                    color="#7A5195")
        axes[1].set_xticks(range(len(site_df)))
        axes[1].set_xticklabels([str(f)[:18] for f in site_df["fixture"]],
                                rotation=45, ha="right", fontsize=6)
        axes[1].set_ylabel("coverage (%)")
        axes[1].set_title("Predicted speeding coverage")
        fig.tight_layout()
        fig.savefig(fig_dir / "game_day_mip_summary.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plots must not kill the run
        print(f"   WARN: plotting failed ({exc})")
    print(f"[60] DONE ({tag}): {json.dumps(summary, default=str)}")


if __name__ == "__main__":
    main()
