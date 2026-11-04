"""
99_obs_site_bin_shape_diag.py - model-free diagnostic on OBSERVED gameday panels.

No modelling: this reads the raw observed panels
(results/segment_15min_panel.parquet = speeding,
 results/segment_15min_panel_braking.parquet = hard braking,
 data/fixture_calendar.csv = kickoffs) and answers, from TRUE VALUES only:

Q1  Do per-site rankings flip across TTE (rel) bins?        (rank instability)
Q2  Do HB and speeding genuinely differ in their per-site
    time shapes - vs each other, and across gamedays?       (shape structure)

Conventions match 59_game_day_search_predict.py: 28 rel-bins of 15 min,
rel_bin 0 = KO-3h, rel_bin 12 = KO, rel_bin 27 = KO+4h (bin label = start).

Rates: events per sampled Compass vehicle in the same bin (5% sample; the
count is a sample count, rate is penetration-invariant within a bin/topic).

Outputs -> results/obs_site_bin_diag/ (+ console report).
Usage:  python scripts/99_obs_site_bin_shape_diag.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results"
OUT = RES / "obs_site_bin_diag"
OUT.mkdir(exist_ok=True)

REL_BINS = 28
KO_BIN = 12
MIN_VEH_BIN = 10       # drop site-bin cells with < this many sampled vehicles
TOPK = 25
SEG = "road_osmid"


# --------------------------------------------------------------------------- #
def load():
    cal = pd.read_csv(PROJ / "data" / "fixture_calendar.csv")
    cal["fixture_id"] = cal["label"] + "|" + cal["kickoff_iso"].str[:10]

    panel_path = RES / "segment_15min_panel_braking_full.parquet"
    if not panel_path.exists():
        panel_path = RES / "segment_15min_panel_braking.parquet"
    print(f"braking panel: {panel_path.name}")
    hb_all = pd.read_parquet(panel_path)
    bmin, bmax = hb_all["bin"].min(), hb_all["bin"].max()
    print(f"coverage: {bmin} .. {bmax}")

    ko = pd.to_datetime(cal["kickoff_iso"]).dt.tz_localize("Australia/Brisbane")
    in_cov = (ko >= bmin - pd.Timedelta(days=0)) & (ko < bmax)
    cal = cal[in_cov].copy()
    print(f"fixtures in coverage: {len(cal)}\n"
          + cal[["label", "kickoff_iso", "attendance", "fixture_type"]]
          .to_string(index=False))

    rows = []
    for _, f in cal.iterrows():
        ko_f = pd.Timestamp(f["kickoff_iso"]).tz_localize("Australia/Brisbane")
        for r in range(REL_BINS):
            rows.append({"fixture_id": f["fixture_id"],
                         "fixture_type": f["fixture_type"],
                         "attendance": f["attendance"],
                         "rel_bin": r,
                         "bin": ko_f - pd.Timedelta(hours=3)
                                + pd.Timedelta(minutes=15 * r),
                         "phase": "pre" if r < 10 else
                                  "during" if r < 18 else "post"})
    win = pd.DataFrame(rows)

    hb = hb_all[hb_all["bin"].isin(win["bin"])].copy()

    import duckdb
    con = duckdb.connect()
    con.register("win_bins", win[["bin"]].drop_duplicates())
    sp = con.execute("""
        SELECT road_osmid, strftime(bin, '%Y-%m-%dT%H:%M:%S%z') AS bin_str,
               vehicle_count, trips_any_point_gt_5, trips_any_point_gt_10
        FROM read_parquet(?)
        WHERE bin IN (SELECT bin::TIMESTAMPTZ_NS FROM win_bins)
    """, [str(RES / "segment_15min_panel.parquet")]).df()
    sp["bin"] = pd.to_datetime(sp.pop("bin_str"), format="%Y-%m-%dT%H:%M:%S%z")
    print(f"speeding rows in game windows: {len(sp)}, "
          f"sites {sp['road_osmid'].nunique()}")

    hb = hb.merge(sp[[SEG, "bin", "vehicle_count"]], on=[SEG, "bin"], how="left")
    print(f"HB cells in windows: {len(hb)}, "
          f"HB events {hb['braking_events'].sum():.0f}")
    return win, hb, sp


def cells(win, events, num_col):
    """site x fixture x rel_bin observed counts, sums of events & exposure."""
    m = events.merge(win[["fixture_id", "fixture_type", "rel_bin", "bin"]],
                     on="bin", how="left")
    g = m.groupby([SEG, "fixture_id", "fixture_type", "rel_bin"], as_index=False) \
        .agg(veh=("vehicle_count", "sum"), num=(num_col, "sum"))
    g["rate"] = g["num"] / g["veh"].clip(lower=1)
    g.loc[g["veh"] < MIN_VEH_BIN, "rate"] = np.nan
    g = g.dropna(subset=["rate"])
    return g


def pooled_cells(g):
    p = g.groupby([SEG, "rel_bin"], as_index=False) \
        .agg(veh=("veh", "sum"), num=("num", "sum"))
    p["rate"] = p["num"] / p["veh"].clip(lower=1)
    p.loc[p["veh"] < 3 * MIN_VEH_BIN, "rate"] = np.nan
    return p.dropna(subset=["rate"])


def rate_matrix(g):
    return g.pivot_table(index=SEG, columns="rel_bin", values="rate")


NBLOCK = 9


def block_cells(g):
    """coarsen rel bins into 9 blocks of 3 bins (45 min); pure aggregation."""
    b = g.copy()
    b["block"] = (b["rel_bin"] // 3).astype(int)
    ab = b.groupby([SEG, "fixture_id", "fixture_type", "block"], as_index=False) \
        .agg(veh=("veh", "sum"), num=("num", "sum"))
    ab["rate"] = ab["num"] / ab["veh"].clip(lower=1)
    ab.loc[ab["veh"] < 3 * MIN_VEH_BIN, "rate"] = np.nan
    return ab.dropna(subset=["rate"])


def block_matrix(g):
    return g.pivot_table(index=SEG, columns="block", values="rate")


def pool_by_site_cells(g):
    """pool across fixtures at the cell level first (sum num, sum veh)."""
    p = g.groupby([SEG, "rel_bin"], as_index=False) \
        .agg(veh=("veh", "sum"), num=("num", "sum"))
    p["rate"] = p["num"] / p["veh"].clip(lower=1)
    p.loc[p["veh"] < 3 * MIN_VEH_BIN, "rate"] = np.nan
    return p.dropna(subset=["rate"])


def block_pool(g):
    """pool across all fixtures, coarsened to 45-min blocks."""
    b = g.copy()
    b["block"] = (b["rel_bin"] // 3).astype(int)
    p = b.groupby([SEG, "block"], as_index=False) \
        .agg(veh=("veh", "sum"), num=("num", "sum"))
    p["rate"] = p["num"] / p["veh"].clip(lower=1)
    p.loc[p["veh"] < 6 * MIN_VEH_BIN, "rate"] = np.nan
    return p.dropna(subset=["rate"])


def with_names(idx):
    """street label for a list of road_osmid values."""
    try:
        meta = pd.read_csv(PROJ / "data" / "road_meta.csv",
                           usecols=["osmid", "road_name"])
        m = dict(zip(meta["osmid"], meta["road_name"]))
    except Exception:
        m = {}
    return [f"{s}:{m.get(s, '?')}" for s in idx]


# --------------------------------------------------------------------------- #
def kendall(Ra, Rb):
    both = pd.concat([Ra.rename("a"), Rb.rename("b")], axis=1).dropna()
    if len(both) < 8:
        return np.nan
    return stats.kendalltau(both["a"], both["b"]).statistic


def topk_jaccard(Ra, Rb, k=TOPK):
    ta = set(Ra.nlargest(k).dropna().index)
    tb = set(Rb.nlargest(k).dropna().index)
    if not ta or not tb:
        return np.nan
    return len(ta & tb) / len(ta | tb)


def band_mean(tseries, lo=0, hi=8, label=""):
    vals = [v for (a, b), v in tseries.items() if lo <= abs(a - b) <= hi]
    print(f"   bins {lo}..{hi} apart: mean {label} = {np.mean(vals):.3f}")
    return np.mean(vals)


def shape_volatility(A, label):
    """all-bin-pair Spearman of site rankings + top-K Jaccard overlap."""
    tc, tg = {}, {}
    for a in A.columns:
        for b in A.columns:
            if a < b:
                common = A[a].dropna().index.intersection(A[b].dropna().index)
                if len(common) >= 8:
                    tc[(a, b)] = stats.spearmanr(A.loc[common, a],
                                                 A.loc[common, b]).statistic
                tg[(a, b)] = topk_jaccard(A[a], A[b])
    tc = pd.Series(tc, dtype=float)
    tg = pd.Series(tg, dtype=float)
    print(f"[{label}] n_sites={A.notna().any(axis=1).sum()}  "
          f"pairwise rank-Spearman: mean {tc.mean():.3f}, min {tc.min():.3f} | "
          f"top-{TOPK} Jaccard: mean {tg.mean():.3f}, min {tg.min():.3f}")
    band_mean(tc, 0, 2, "rank-rho")
    band_mean(tc, 6, 10, "rank-rho")
    band_mean(tc, 14, 20, "rank-rho")
    return tc, tg


def hb_vs_sp_shape(HB, SP, label, min_overlap=12):
    """per-site Spearman between that site's HB profile and its SP profile."""
    common = HB.index.intersection(SP.index)
    rhos = {}
    for s in common:
        h, p = HB.loc[s], SP.loc[s]
        m = h.notna() & p.notna()
        if m.sum() >= min_overlap:
            rhos[s] = stats.spearmanr(h[m], p[m]).statistic
    rho = pd.Series(rhos, dtype=float)
    if rho.notna().sum() == 0:
        print(f"[{label}] no sites with enough overlap")
        return rho
    print(f"[{label}] HB-SP within-site profile Spearman across bins: "
          f"mean {rho.mean():.3f}, median {rho.median():.3f}, "
          f"neg frac {float((rho < 0).mean()):.2f} (n={rho.notna().sum()})")
    return rho


def variance_decomposition(A, label):
    """Observed value decomposition of log rate: site / bin / interaction,
    plus raw scale shares.  No nulls, no simulation - just the data."""
    X = np.log(A.values + 1.0e-3)
    fin = np.isfinite(X)
    filler = np.array([np.nanmean(row) if fin[i].any() else 0.0
                       for i, row in enumerate(X)])
    F = X.copy()
    for i in range(X.shape[0]):
        F[i] = np.where(fin[i], F[i], filler[i])
    tot = np.var(F, ddof=1)
    site = np.var(np.mean(F, axis=1), ddof=1)
    binv = np.var(np.mean(F, axis=0), ddof=1)
    resid = F - np.mean(F) - (np.mean(F, axis=1) - np.mean(F))[:, None] \
                    - (np.mean(F, axis=0) - np.mean(F))[None, :]
    inter = np.var(resid, ddof=1)
    print(f"[{label}] log-rate variance shares: "
          f"site {100*site/tot:.0f}%, bin {100*binv/tot:.0f}%, "
          f"interaction {100*inter/tot:.0f}%  "
          f"(total var {tot:.3f}, n_sites={A.notna().any(axis=1).sum()})")
    return {"label": label, "site_pct": 100 * site / tot,
            "bin_pct": 100 * binv / tot, "interaction_pct": 100 * inter / tot}


# --------------------------------------------------------------------------- #
def main():
    win, hb, sp = load()

    hb_cell = cells(win, hb.reset_index(drop=True), "braking_events")
    sp_cell = cells(win, sp.reset_index(drop=True), "trips_any_point_gt_5")
    hb_cell.to_csv(OUT / "hb_cell_rates.csv", index=False)
    sp_cell.to_csv(OUT / "sp_cell_rates.csv", index=False)
    hb_cell.to_parquet(OUT / "hb_cell_rates.parquet")
    sp_cell.to_parquet(OUT / "sp_cell_rates.parquet")

    HB = rate_matrix(pooled_cells(hb_cell))
    SP = rate_matrix(pooled_cells(sp_cell))
    print(f"\npooled tables: HB {HB.shape}, SP {SP.shape}")

    print("\n=== Q1a / pooled: site-rank stability across rel bins ===")
    tc, tg = shape_volatility(HB, "HB")
    tsu, tsu_g = shape_volatility(SP, "SP")

    pd.DataFrame([{"bin_a": a, "bin_b": b, "spearman": v}
                  for (a, b), v in tc.items()]).to_csv(
        OUT / "hb_rank_pairs_pooled.csv", index=False)
    pd.DataFrame([{"bin_a": a, "bin_b": b, "spearman": v}
                  for (a, b), v in tsu.items()]).to_csv(
        OUT / "sp_rank_pairs_pooled.csv", index=False)

    print("\n=== key phase contrasts (pooled across gamedays) ===")
    rows = []
    for a, b in [(6, 16), (8, 16), (10, 14), (10, 20), (2, 26), (4, 22)]:
        rows.append({"bins": f"{a}->{b}",
                     "hb_tau": kendall(HB[a], HB[b]) if a in HB.columns and b in HB.columns else np.nan,
                     "hb_topk_jac": topk_jaccard(HB[a], HB[b]) if a in HB.columns and b in HB.columns else np.nan,
                     "sp_tau": kendall(SP[a], SP[b]) if a in SP.columns and b in SP.columns else np.nan,
                     "sp_topk_jac": topk_jaccard(SP[a], SP[b]) if a in SP.columns and b in SP.columns else np.nan})
    phase = pd.DataFrame(rows)
    print(phase.round(3).to_string(index=False))
    phase.to_csv(OUT / "phase_rank_contrasts.csv", index=False)

    print("\n=== Q2 / pooled: HB vs SP within-site shape agreement ===")
    rho = hb_vs_sp_shape(HB, SP, "pooled @15min")
    rho.sort_values().to_frame("spearman").to_csv(
        OUT / "hb_vs_sp_shape_per_site_pooled.csv")
    # coarsened 45-min blocks: HB is sparse per 15-min cell, blocks give real
    # per-site profiles (still observed sums, no model)
    HBb = block_matrix(block_pool(hb_cell))
    SPb = block_matrix(block_pool(sp_cell))
    print(f"block tables: HB {HBb.shape}, SP {SPb.shape}")
    rho_b = hb_vs_sp_shape(HBb, SPb, "pooled @45min-blocks", min_overlap=5)
    rho_b.sort_values().to_frame("spearman").to_csv(
        OUT / "hb_vs_sp_shape_per_site_blocks.csv")

    print("\n=== site-shape 'uniqueness': site profile vs pooled mean profile ===")
    prof = []
    for s in HB.index:
        h, m = HB.loc[s].dropna(), HB.mean(axis=0)
        prof.append({"site": s,
                     "hb_corr_to_mean": stats.spearmanr(h, m[h.index]).statistic
                     if len(h) >= 12 else np.nan,
                     "hb_events_total": float(hb_cell[hb_cell[SEG] == s]["num"].sum())})
    for s in HBb.index:  # block-level uniqueness gets more sites
        h, m = HBb.loc[s].dropna(), HBb.mean(axis=0)
        prof.append({"site": s, "level": "45min-block",
                     "hb_corr_to_mean": stats.spearmanr(h, m[h.index]).statistic
                     if len(h) >= 6 else np.nan,
                     "hb_events_total": float(hb_cell[hb_cell[SEG] == s]["num"].sum())})
    prof = pd.DataFrame(prof)
    print(prof.dropna(subset=["hb_corr_to_mean"])
          ["hb_corr_to_mean"].describe().round(3).to_string())
    prof.to_csv(OUT / "site_shape_vs_mean_pooled.csv", index=False)

    print("\n=== Q1b / per-gameday: does the ranking flip within a gameday? ===")
    per_day = []
    for fid, sub in hb_cell.groupby("fixture_id"):
        spsub = sp_cell[sp_cell["fixture_id"] == fid]
        for tag, A, level in [("hb15", rate_matrix(sub), "15min"),
                              ("hb45", block_matrix(block_cells(sub)), "45min"),
                              ("sp15", rate_matrix(spsub), "15min"),
                              ("sp45", block_matrix(block_cells(spsub)), "45min")]:
            tA, tj = shape_volatility(A, f"{tag} {fid}")
            per_day.append({"fixture_id": fid, "metric": tag, "level": level,
                            "n_sites": int(A.notna().any(axis=1).sum()),
                            "pair_rho_mean": float(tA.mean()),
                            "pair_rho_min": float(tA.min()),
                            "topk_jac_mean": float(tj.mean())})
    per_day = pd.DataFrame(per_day)
    per_day.to_csv(OUT / "per_gameday_rank_volatility.csv", index=False)

    print("\n=== Q2b / per-gameday: HB-SP shape agreement per gameday ===")
    day = []
    for fid in hb_cell["fixture_id"].unique():
        H = rate_matrix(hb_cell[hb_cell["fixture_id"] == fid])
        S = rate_matrix(sp_cell[sp_cell["fixture_id"] == fid])
        r = hb_vs_sp_shape(H, S, f"{fid} @15min")
        if r is not None and r.notna().sum():
            day.append({"fixture_id": fid, "level": "15min",
                        "mean_rho": float(r.mean()), "median_rho": float(r.median()),
                        "neg_frac": float((r < 0).mean()),
                        "n_sites": int(r.notna().sum())})
        Hb = block_matrix(block_cells(hb_cell[hb_cell["fixture_id"] == fid]))
        Sb = block_matrix(block_cells(sp_cell[sp_cell["fixture_id"] == fid]))
        rb = hb_vs_sp_shape(Hb, Sb, f"{fid} @45min")
        if rb is not None and rb.notna().sum():
            day.append({"fixture_id": fid, "level": "45min",
                        "mean_rho": float(rb.mean()), "median_rho": float(rb.median()),
                        "neg_frac": float((rb < 0).mean()),
                        "n_sites": int(rb.notna().sum())})
    day = pd.DataFrame(day)
    day.to_csv(OUT / "per_gameday_hb_vs_sp_shape.csv", index=False)

    # named top-15 tables for eyeballing rank flips, pre vs during vs post
    for col, tag in [(4, "pre"), (13, "ko"), (20, "post")]:
        tops = pd.DataFrame({
            f"hb_{tag}": with_names(HB[col].nlargest(15).index),
            f"hb_{tag}_rate": HB[col].nlargest(15).values,
            f"sp_{tag}": with_names(SP[col].nlargest(15).index),
            f"sp_{tag}_rate": SP[col].nlargest(15).values})
        tops.to_csv(OUT / f"top15_pooled_relbin{col}_{tag}.csv", index=False)
        print(f"\ntop-15 by observed rate at rel bin {col} ({tag}):\n"
              + tops.to_string(index=False))

    print("\n=== observed variance decomposition of log rate ===")
    dec_rows = [variance_decomposition(HB, "HB pooled @15min"),
                variance_decomposition(SP, "SP pooled @15min"),
                variance_decomposition(HBb, "HB pooled @45min"),
                variance_decomposition(SPb, "SP pooled @45min")]
    pd.DataFrame(dec_rows).to_csv(OUT / "variance_decomposition.csv", index=False)

    figures(HB, SP, tc, tsu, hb_cell)
    print(f"\noutputs -> {OUT}")


# --------------------------------------------------------------------------- #
def figures(HB, SP, tc, tsu, hb_cell):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def heatmap(vals, title, path, xlabel, ylabel, **kw):
        cmap = kw.pop("cmap", "viridis")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(vals, aspect="auto", cmap=cmap, **kw)
        plt.colorbar(im)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()

    rank = (-HB.rank(ascending=False)).astype(float)
    heatmap(rank.values, "HB observed rate rank by site x rel bin "
            f"(pooled gamedays, {HB.notna().any(axis=1).sum()} sites)",
            OUT / "hb_rank_heatmap.png", "rel_bin (12=KO)", "site")
    rank = (-SP.rank(ascending=False)).astype(float)
    heatmap(rank.values, "Speeding observed rate rank by site x rel bin "
            f"(pooled gamedays, {SP.notna().any(axis=1).sum()} sites)",
            OUT / "sp_rank_heatmap.png", "rel_bin (12=KO)", "site")

    for M, name in [(tc, "hb"), (tsu, "sp")]:
        Sm = np.full((28, 28), np.nan)
        for (a, b), v in M.items():
            Sm[a, b] = v
            Sm[b, a] = v
        heatmap(Sm, f"{name.upper()} rank stability (Spearman) between bins",
                OUT / f"{name}_rank_stability_matrix.png", "rel_bin", "rel_bin",
                vmin=-1, vmax=1, cmap="RdYlGn")

    for A, tag, cap in [(HB, "hb", "HB rate"), (SP, "sp", "speeding rate")]:
        top = A.mean(axis=1).nlargest(12).index
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in top:
            ax.plot(A.columns, A.loc[s].values, alpha=0.75, lw=1.2)
        ax.axvline(12, ls="--", c="k", lw=0.6)
        ax.set_title(f"Observed {cap} per site, top-12 {tag.upper()} sites (pooled)")
        ax.set_xlabel("rel_bin (12 = KO)")
        plt.tight_layout()
        plt.savefig(OUT / f"{tag}_profile_top12.png", dpi=130)
        plt.close()


if __name__ == "__main__":
    main()
