"""
99b_phase_surface_logo_validate.py - first predictive layer from OBSERVED data.

Per-site risk surfaces for HB and speeding at 3 phases (pre/during/post K0-3h,
0-4h15, 0-4h15..+4h) and 9 x 45-min blocks, built purely by empirical-Bayes
shrinkage of observed counts, validated leave-one-GAMEDAY-out.

Scores
  s_rate_static   : shrunk site rate (ignores timing)
  s_rate_add      : site x phase mean (pooled shape)
  s_rate_inter    : add + EB-shrunk per-site phase deviation
  s_cnt_add       : s_rate_add x TRAINING exposure  (expected events) <- deploy
  s_cnt_oracle    : s_rate_add x observed hold-out exposure (upper bound)
  s_expo          : exposure only (busiest segments), no event data

Metrics per fold x phase: Spearman(score, obs rate), top-25 Jaccard,
capture@25 = share of the phase's observed events on the selected 25 segments.
Phases/blocks with fewer than MIN_EVAL evaluable cells are skipped (capture
and Jaccard are meaningless there).

Outputs -> results/obs_site_bin_diag/logo/ ; final all-data deployment
surface -> logo/deploy_surface_<marker>.csv (named streets).
Usage:  python scripts/99b_phase_surface_logo_validate.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
DIAG = PROJ / "results" / "obs_site_bin_diag"
OUT = DIAG / "logo"
OUT.mkdir(exist_ok=True)

SEG = "road_osmid"
K = 2.0
KE = 5.0        # prior strength for exposure shrink (pseudo gamedays)
TOPK = 25
MIN_VEH_EVAL = 10
MIN_EVAL = 25   # min evaluable cells in a phase/block to score it

GRANS = {
    "phase": lambda r: "pre" if r <= 9 else ("during" if r <= 17 else "post"),
    "block": lambda r: int(r // 3),
}
RATE_SCORES = ["s_rate_static", "s_rate_add", "s_rate_inter"]
CNT_SCORES = ["s_cnt_add", "s_cnt_oracle", "s_expo"]


def with_names(ids):
    meta = pd.read_csv(PROJ / "data" / "road_meta.csv", usecols=["osmid", "road_name"])
    m = dict(zip(meta["osmid"], meta["road_name"]))
    return [f"{s}:{m.get(s, 'osm?')}" for s in ids]


def prep(marker):
    df = pd.read_parquet(DIAG / f"{marker}_cell_rates.parquet")
    df = df[df["veh"] >= MIN_VEH_EVAL].copy()
    df["rate"] = df["num"] / df["veh"]
    return df


def fit_scores(tr, gran):
    r0 = tr["num"].sum() / max(tr["veh"].sum(), 1.0)

    a_s = tr.groupby(SEG).agg(num=("num", "sum"), veh=("veh", "sum"))
    a_s["r"] = (a_s["num"] + K * r0) / (a_s["veh"] + K)
    a_s = a_s["r"]

    ph = tr.groupby(gran).agg(num=("num", "sum"), veh=("veh", "sum"))
    b_p = (ph["num"] + K * r0) / (ph["veh"] + K)
    b_p = b_p / max(float(b_p.mean()), 1e-9)

    cell = tr.groupby([SEG, gran]).agg(num=("num", "sum"), veh=("veh", "sum"))
    cell["r"] = (cell["num"] + K * r0) / (cell["veh"] + K)
    cell["a"] = a_s.reindex(cell.index.get_level_values(0)).values
    cell["b"] = b_p.reindex(cell.index.get_level_values(1)).values
    cell["mu"] = (r0 * cell["a"] * cell["b"]).clip(lower=1e-9)
    cell["d"] = np.log((cell["r"] + 1e-9) / cell["mu"])
    sig2 = 1.0 / cell["veh"].clip(lower=1)
    tau2 = max(float(np.var(cell["d"].values, ddof=1) - np.mean(sig2.values)), 1e-6)
    cell["dhat"] = (tau2 / (tau2 + sig2)) * cell["d"]

    # shrunk training exposure per (site, gran): mean veh across fixtures
    gmean = float(tr.groupby(gran)["veh"].mean().mean())
    expo = tr.groupby([SEG, gran]).agg(veh_sum=("veh", "sum"),
                                       nfx=("fixture_id", "nunique"))
    expo["e"] = (expo["veh_sum"] + KE * gmean) / (expo["nfx"] + KE)
    return {"r0": r0, "a": a_s, "b": b_p, "cell": cell,
            "expo": expo["e"], "gmean_expo": tr["veh"].mean()}


def predict(model, ev, gran):
    a, b, cell = model["a"], model["b"], model["cell"]
    sa = a.reindex(ev[SEG]).values
    sa = np.where(np.isfinite(sa), sa, 1.0) * model["r0"]
    sb = sa * b.reindex(ev[gran]).fillna(1.0).values
    dhat = cell["dhat"].reindex(pd.MultiIndex.from_arrays([ev[SEG], ev[gran]])).values
    si = sb * np.exp(np.nan_to_num(dhat, nan=0.0).clip(-3, 3))
    e = model["expo"].reindex(pd.MultiIndex.from_arrays([ev[SEG], ev[gran]])).values
    e = np.nan_to_num(e, nan=model["gmean_expo"])
    veh = ev["veh"].values if "veh" in ev.columns else np.full(len(ev), np.nan)
    out = pd.DataFrame({
        "s_rate_static": sa, "s_rate_add": sb, "s_rate_inter": si,
        "s_cnt_add": sb * e, "s_cnt_oracle": sb * veh,
        "s_expo": e}, index=ev.index)
    return out


def evaluate(ev, score_list):
    mask = (ev["veh"] >= MIN_VEH_EVAL).values
    rs = ev["rate_obs"].values
    rows = []
    n_eval = int(mask.sum())
    if n_eval < MIN_EVAL:
        return rows, n_eval
    for score in score_list:
        s = ev[score].values
        ok = mask & np.isfinite(s) & np.isfinite(rs)
        if ok.sum() < MIN_EVAL:
            continue
        sub, s2, r2 = ev[ok], s[ok], rs[ok]
        rho = float(stats.spearmanr(s2, r2).statistic)
        top = sub.index[np.argsort(-s2)][:TOPK]
        obs_top = sub.index[np.argsort(-r2)][:TOPK]
        jac = len(set(top) & set(obs_top)) / len(set(top) | set(obs_top))
        cap = float(sub.loc[top, "num"].sum()) / max(float(sub["num"].sum()), 1.0)
        rows.append({"score": score, "spearman": rho, "topk_jaccard": jac,
                     f"capture@{TOPK}": cap, "n_eval": int(ok.sum()),
                     "events": float(sub["num"].sum())})
    return rows, n_eval


def run(marker, gran_name):
    df = prep(marker)
    df[gran_name] = [GRANS[gran_name](r) for r in df["rel_bin"]]
    rows, skipped = [], []
    for fid, ev_f in df.groupby("fixture_id"):
        tr = df[df["fixture_id"] != fid]
        model = fit_scores(tr, gran_name)
        by = ev_f.groupby([SEG, gran_name]).agg(num=("num", "sum"),
                                                veh=("veh", "sum")).reset_index()
        by["rate_obs"] = by["num"] / by["veh"].clip(lower=1)
        preds = predict(model, by, gran_name)
        ev = pd.concat([by, preds], axis=1)
        for gval, sub in ev.groupby(gran_name):
            rr, n_eval = evaluate(sub, RATE_SCORES + CNT_SCORES)
            if not rr:
                skipped.append({"fold": fid, gran_name: gval, "n_eval": n_eval})
                continue
            for r in rr:
                rows.append({"fold": fid, gran_name: gval, **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"logo_{marker}_{gran_name}.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(OUT / f"skipped_{marker}_{gran_name}.csv",
                                     index=False)
    agg = res.groupby([gran_name, "score"]).agg(
        spearman=("spearman", "mean"),
        topk_jaccard=("topk_jaccard", "mean"),
        capture=(f"capture@{TOPK}", "mean"),
        events=("events", "mean"),
        folds=("fold", pd.Series.nunique)).reset_index()
    agg.to_csv(OUT / f"logo_agg_{marker}_{gran_name}.csv", index=False)
    return agg


def deploy_surface(marker):
    """All-data EB surface -> ranked deployment tables per phase."""
    df = prep(marker)
    df["phase"] = [GRANS["phase"](r) for r in df["rel_bin"]]
    model = fit_scores(df, "phase")
    sites = model["a"].index
    out = []
    for p in ["pre", "during", "post"]:
        ev = pd.DataFrame({SEG: sites, "phase": p})
        pr = predict(model, ev, "phase")
        pr[SEG] = sites
        pr["rate_pred"] = pr["s_rate_add"]
        pr["cnt_pred"] = pr["s_cnt_add"]
        top_r = pr.nlargest(10, "rate_pred")
        top_c = pr.nlargest(10, "cnt_pred")
        for tag, sub in [("by_rate", top_r), ("by_expected_count", top_c)]:
            for rank, (_, row) in enumerate(sub.iterrows(), 1):
                out.append({"marker": marker, "phase": p, "list": tag,
                            "rank": rank, SEG: row[SEG],
                            "street": with_names([row[SEG]])[0],
                            "rate_pred_per_veh": row["s_rate_add"],
                            "expected_events": row["s_cnt_add"]})
    return pd.DataFrame(out)


def main():
    for marker in ["hb", "sp"]:
        for gran_name in ["phase", "block"]:
            agg = run(marker, gran_name)
            print(f"\n=== LOGO {marker.upper()} @ {gran_name} "
                  f"(mean over {int(agg['folds'].max())} folds) ===")
            print(agg.round(3).to_string(index=False))
    surf = pd.concat([deploy_surface(m) for m in ["hb", "sp"]])
    surf.to_csv(OUT / "deploy_surface.csv", index=False)
    print(f"\ndeployment surface (all-data EB fit) -> {OUT / 'deploy_surface.csv'}")


if __name__ == "__main__":
    main()
