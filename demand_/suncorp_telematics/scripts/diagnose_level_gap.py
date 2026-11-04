#!/usr/bin/env python
"""Why does the holdout run at 1.5x the training rate while the surface ships 0.55x?

This is the level companion to diagnose_exposure_transfer.py, which settled the
*shape* question (the pinned offset transfers; the residual is a gradient). Here
the question is where the missing MASS sits and whether the crowd channel has
any capacity to carry it.

Four separations, each of which rules out a different story:

  0. world side -- is the holdout's 1.54x a uniform regime shift, or is its
     rate-vs-exposure curve a different shape from training's? Training rates
     come from the cell probe (a post-refit, post-calibration design sample of
     the fitted frame), so the comparison is against the rows the model was
     actually fitted on, with Poisson intervals.

  1. model side -- an exposure-share-weighted split of the aggregate deficit.
     A uniform bias and a gradient produce the same total; they produce very
     different weights, and only one of them can be fixed by a level knob.

  2. location -- the deficit by fixture, by fixture x exposure tercile, and by
     event-relative bin. A per-fixture (crowd) scalar can only address the
     between-fixture part; whatever is left inside a fixture needs a
     within-fixture correction, i.e. a different model term.

  3. capacity -- the crowd channel's required size. Inverting the deficit into
     the attendance elasticity that WOULD have carried it, per fixture, and
     comparing that with the fitted elasticity and its standard error. If the
     required elasticity changes sign across fixtures, no monotone crowd
     response -- and therefore no attendance channel -- can express it.

Nothing is refitted; this reads one arm's artifacts.

    python scripts/diagnose_level_gap.py [arm_dir]

Writes <arm_dir>/level_gap.csv and prints the decomposition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_tail = sys.argv[1:] or ["results_tests/t9_ab_band_cross_pindrop"]
ARM = Path(_tail[0])
ARM = ARM if ARM.is_absolute() else ROOT / ARM
CAL = ROOT / "data" / "fixture_calendar.csv"
TZ = "Australia/Brisbane"
CROWD_MED = 38223.0          # audit.crowd_med: the reference crowd in the design


def load_calendar(frame: pd.DataFrame, col: str = "bin") -> pd.DataFrame:
    t = pd.to_datetime(frame[col], utc=True).dt.tz_convert(TZ)
    frame = frame.copy()
    frame[col] = t
    frame["hour"] = t.dt.hour
    return frame


def poisson_ci(k: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact-ish (Garwood) interval for a count, via chi2 quantiles."""
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2.0
    hi = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2.0
    return float(lo), float(hi)


def load_arm() -> tuple[pd.DataFrame, pd.DataFrame]:
    """probe = training design rows; hold = holdout rows with exposure + obs."""
    probe = load_calendar(pd.read_csv(ARM / "game_day_cell_probe.csv.gz"))
    if "in_event" in probe.columns:
        probe = probe[probe["in_event"].astype(float) > 0.5]
    # The probe stores `exp` as the rate scale and logexp beside it (59 writes
    # keys["exp"] = fw["logexp"] for the grid only). Verify before using.
    assert np.allclose(np.log1p(probe["exp"]), probe["logexp"], atol=1e-6), \
        "probe exp is not on the rate scale; the parameterisation changed"

    grid = load_calendar(pd.read_parquet(
        ARM / "game_day_pred_15min.parquet",
        columns=["fixture", "road_osmid", "bin", "rel_bin", "exp",
                 "pred_flow", "flow_uplift", "pred_speeding"]))
    obs = load_calendar(pd.read_csv(
        ARM / "game_day_pred_vs_actual_cells.csv.gz",
        usecols=["fixture", "road_osmid", "bin", "observed"]))
    n_cells = len(obs)
    hold = grid.merge(obs.drop(columns=["hour"]),
                      on=["fixture", "road_osmid", "bin"], how="inner")
    assert len(hold) == n_cells, (
        f"exposure grid covers {len(hold):,} of {n_cells:,} scored cells")
    # 59 writes keys["exp"] = fw["logexp"], i.e. log1p of the offset exposure.
    assert (np.corrcoef(hold["exp"], np.log1p(hold["pred_flow"]))[0, 1]
            > np.corrcoef(hold["exp"], hold["pred_flow"])[0, 1]), \
        "grid `exp` is not log-scale here; expm1() below would be wrong"
    hold["E"] = np.expm1(hold["exp"])
    return probe, hold


def main() -> None:
    probe, hold = load_arm()
    rows: list[dict] = []

    def add(segment: str, bucket: str, n: int, obs: float, pred: float,
            E: float, **extra) -> dict:
        r = dict(segment=segment, bucket=bucket, n=n, obs=obs, pred=pred, E=E,
                 obs_rate=obs / max(E, 1e-12), pred_rate=pred / max(E, 1e-12),
                 deficit=obs / max(pred, 1e-12), **extra)
        rows.append(r)
        return r

    # ---- 0. the world side ------------------------------------------------
    tr_rate = float(probe["Y"].sum() / probe["exp"].sum())
    tr_k = float(probe["Y"].sum())
    tr_lo, tr_hi = poisson_ci(tr_k)
    tr_expden = float(probe["exp"].sum())
    ho_rate = float(hold["observed"].sum() / hold["E"].sum())
    print("== 0. world side: is 1.54x a shift or a shape change? ==")
    print(f"   train Y/E  = {tr_rate:.5f}  ({tr_k:.0f} events on "
          f"{len(probe):,} probe rows, E={tr_expden:,.0f})  "
          f"CI {tr_lo / tr_expden:.5f}-{tr_hi / tr_expden:.5f}")
    print(f"   holdout obs/E = {ho_rate:.5f}  ({hold['observed'].sum():.0f} "
          f"events on {len(hold):,} rows, E={hold['E'].sum():,.0f})  "
          f"ratio {ho_rate / tr_rate:.2f}x")

    # Same E cut points on both sides: does the rate-vs-exposure curve match?
    cuts = np.quantile(hold["E"], [1 / 3, 2 / 3])
    def band(x: pd.Series) -> pd.Series:
        return pd.cut(x, [-np.inf, cuts[0], cuts[1], np.inf],
                      labels=["low E", "mid E", "high E"])
    hold["E_band"] = band(hold["E"])
    probe = probe.assign(E_band=band(probe["exp"]))
    print("\n   rate per unit exposure by band (same cut points both sides)")
    print(f"   {'band':8s} {'train Y/E':>10s} {'obs/E':>9s} {'ratio':>7s} "
          f"{'train k':>8s}")
    for b in ["low E", "mid E", "high E"]:
        tp = probe[probe["E_band"] == b]
        hp = hold[hold["E_band"] == b]
        if not len(tp) or not len(hp):
            continue
        tr = tp["Y"].sum() / max(tp["exp"].sum(), 1e-12)
        hr = hp["observed"].sum() / max(hp["E"].sum(), 1e-12)
        print(f"   {b:8s} {tr:10.5f} {hr:9.5f} {hr / max(tr, 1e-12):6.2f}x "
              f"{tp['Y'].sum():8.0f}")
        add("world by E band", b, len(hp), float(hp["observed"].sum()),
            float(hp["E"].sum()), float(hp["E"].sum()), train_rate=tr)
    tr_hi_rate = float(probe.loc[probe["E_band"] == "high E", "Y"].sum()
                       / max(probe.loc[probe["E_band"] == "high E",
                                       "exp"].sum(), 1e-12))
    tr_lo_rate = float(probe.loc[probe["E_band"] == "low E", "Y"].sum()
                       / max(probe.loc[probe["E_band"] == "low E",
                                       "exp"].sum(), 1e-12))
    ho_hi_rate = float(hold.loc[hold["E_band"] == "high E", "observed"].sum()
                       / max(hold.loc[hold["E_band"] == "high E", "E"].sum(),
                             1e-12))
    ho_lo_rate = float(hold.loc[hold["E_band"] == "low E", "observed"].sum()
                       / max(hold.loc[hold["E_band"] == "low E", "E"].sum(),
                             1e-12))
    print(f"   train high/low = {tr_hi_rate / max(tr_lo_rate, 1e-12):.2f}x, "
          f"holdout high/low = {ho_hi_rate / max(ho_lo_rate, 1e-12):.2f}x")

    # ---- 1. the model side: where the mass is ------------------------------
    oct_cuts = np.quantile(hold["E"], np.arange(1, 8) / 8)
    hold["oct"] = pd.cut(hold["E"], [-np.inf, *oct_cuts, np.inf],
                         labels=[f"o{i}" for i in range(1, 9)])
    tot_obs = float(hold["observed"].sum())
    tot_pred = float(hold["pred_speeding"].sum())
    print("\n== 1. model side: exposure-weighted location of the deficit ==")
    print(f"   total obs={tot_obs:,.0f} pred={tot_pred:,.0f} "
          f"deficit={tot_obs / tot_pred:.2f}x")
    print(f"   {'octile':7s} {'n':>7s} {'E share':>8s} {'obs share':>10s} "
          f"{'pred/E':>8s} {'obs/E':>8s} {'deficit':>8s} {'contrib':>8s}")
    contribs: dict[str, float] = {}
    for o, g in hold.groupby("oct", observed=True):
        o_sum = float(g["observed"].sum())
        p_sum = float(g["pred_speeding"].sum())
        w = o_sum / max(tot_obs, 1e-12)
        c = w * np.log(max(o_sum, 1e-12) / max(p_sum, 1e-12))
        contribs[str(o)] = c
        print(f"   {str(o):7s} {len(g):7,} "
              f"{g['E'].sum() / hold['E'].sum():8.1%} {w:10.1%} "
              f"{p_sum / max(g['E'].sum(), 1e-12):8.4f} "
              f"{o_sum / max(g['E'].sum(), 1e-12):8.4f} "
              f"{o_sum / max(p_sum, 1e-12):7.2f}x {c:8.3f}")
        add("holdout by E octile", str(o), len(g), o_sum, p_sum,
            float(g["E"].sum()))
    tot_c = sum(contribs.values())
    print(f"   sum of contributions = {tot_c:.3f} "
          f"(ln deficit = {np.log(tot_obs / tot_pred):.3f}); "
          f"top octile alone = {contribs['o8'] / tot_c:.0%}")

    # ---- 2. location: fixture, fixture x tercile, rel_bin ------------------
    print("\n== 2. location ==")
    print(f"   {'fixture':46s} {'n':>7s} {'E':>10s} {'obs':>7s} {'pred':>7s} "
          f"{'deficit':>8s} {'obs sh':>7s}")
    fix_stats = {}
    for f, g in hold.groupby("fixture", observed=True):
        o_sum = float(g["observed"].sum())
        p_sum = float(g["pred_speeding"].sum())
        fix_stats[f] = (o_sum, p_sum, float(g["E"].sum()), len(g))
        print(f"   {f:46s} {len(g):7,} {g['E'].sum():10.0f} {o_sum:7.0f} "
              f"{p_sum:7.0f} {o_sum / max(p_sum, 1e-12):7.2f}x "
              f"{o_sum / tot_obs:7.1%}")
        add("holdout by fixture", f, len(g), o_sum, p_sum, float(g["E"].sum()))

    # between-fixture vs within-fixture spread of the log deficit
    lo = hold.assign(lr=np.log((hold["observed"] + 0.5)
                               / (hold["pred_speeding"] + 0.5)))
    fw = lo.groupby("fixture", observed=True).apply(
        lambda g: pd.Series({"m": (g["lr"] * g["observed"]).sum()
                             / max(g["observed"].sum(), 1e-12),
                             "n": g["observed"].sum()}), include_groups=False)
    between = float(np.average((fw["m"] - fw["m"].mean()) ** 2,
                               weights=fw["n"]))
    within = float(((lo["lr"] - lo["fixture"].map(fw["m"])) ** 2
                    * lo["observed"]).sum() / max(lo["observed"].sum(), 1e-12))
    print(f"   obs-weighted variance of log deficit: between-fixture "
          f"{between:.4f}, within-fixture {within:.4f} -> between is "
          f"{between / max(between + within, 1e-12):.0%}")

    # within one fixture, across exposure terciles
    print(f"\n   deficit inside each fixture (E terciles):")
    print(f"   {'fixture':46s} {'low':>7s} {'mid':>7s} {'high':>7s} {'spread':>7s}")
    for f, g in hold.groupby("fixture", observed=True):
        tert = pd.qcut(g["E"], 3, labels=["low", "mid", "high"],
                       duplicates="drop")
        d = (g.assign(t=tert).groupby("t", observed=True)
             .apply(lambda x: x["observed"].sum()
                    / max(x["pred_speeding"].sum(), 1e-12),
                    include_groups=False))
        sp = float(d.max() / max(d.min(), 1e-12))
        print(f"   {f:46s} {d.get('low', np.nan):6.2f}x "
              f"{d.get('mid', np.nan):6.2f}x {d.get('high', np.nan):6.2f}x "
              f"{sp:6.2f}x")
        for k, v in d.items():
            sub = g[tert == k]
            add("holdout by fixture x tercile", f"{f} | {k}", len(sub),
                float(sub["observed"].sum()),
                float(sub["pred_speeding"].sum()), float(sub["E"].sum()))

    # event-relative time profile: is the response mis-timed?
    print(f"\n   event-relative profile (rel_bin 0 = kickoff-3h):")
    prof = (hold.groupby("rel_bin", observed=True)
            .apply(lambda g: pd.Series({
                "n": len(g),
                "lift": float(g["flow_uplift"].mean()),
                "obs": float(g["observed"].sum()),
                "pred": float(g["pred_speeding"].sum()),
                "E": float(g["E"].sum())}), include_groups=False))
    prof["obs_rate"] = prof["obs"] / prof["E"]
    prof["pred_rate"] = prof["pred"] / prof["E"]
    prof["deficit"] = prof["obs"] / prof["pred"].clip(lower=1e-12)
    print(f"   {'rel_bin':>7s} {'n':>7s} {'uplift':>7s} {'obs/E':>8s} "
          f"{'pred/E':>8s} {'deficit':>8s}")
    for i, r in prof.iterrows():
        print(f"   {i:7d} {r['n']:7.0f} {r['lift']:7.3f} {r['obs_rate']:8.4f} "
              f"{r['pred_rate']:8.4f} {r['deficit']:7.2f}x")
        add("holdout by rel_bin", str(int(i)), int(r["n"]), r["obs"],
            r["pred"], r["E"])
    o_peak = int(prof["obs_rate"].idxmax())
    p_peak = int(prof["pred_rate"].idxmax())
    print(f"   observed peak at rel_bin {o_peak}, model peak at {p_peak} "
          f"({(o_peak - p_peak) * 15:+d} min offset)")

    # ---- 3. capacity: what elasticity would the crowd channel have needed? -
    print("\n== 3. could the crowd channel have carried it? ==")
    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"])
    audit = json.load(open(ARM / "game_day_holdout_metrics.json"))["audit"]
    eta_raw = float(audit["demand_elasticity"]["eta_raw"])
    eta_se = float(audit["demand_elasticity"]["eta_se"])
    eta = float(audit["demand_elasticity"]["eta"])
    print(f"   fitted crowd elasticity: raw {eta_raw:+.3f} (se {eta_se:.2f}, "
          f"p={audit['demand_elasticity']['eta_p']:.3f}) -> shipped {eta:+.3f}")
    print(f"   need it to be monotone in attendance AND big enough:")
    print(f"   {'fixture':46s} {'att':>7s} {'log(att/ref)':>13s} "
          f"{'deficit':>8s} {'required eta':>13s}")
    req = {}
    for f, (o_sum, p_sum, _E, _n) in fix_stats.items():
        label = f.split("@")[0].strip()
        hit = cal[cal["label"] == label]
        att = float(hit["attendance"].iloc[0]) if len(hit) else np.nan
        lr = float(np.log(att / CROWD_MED)) if np.isfinite(att) else np.nan
        need = float(np.log(o_sum / max(p_sum, 1e-12)))
        e_req = need / lr if np.isfinite(lr) and abs(lr) > 1e-9 else np.nan
        req[f] = (att, lr, need, e_req)
        print(f"   {f:46s} {att:7.0f} {lr:+13.3f} "
              f"{o_sum / max(p_sum, 1e-12):7.2f}x {e_req:+13.2f}")
        add("crowd capacity", f, _n, o_sum, p_sum, _E,
            attendance=att, log_att_rel_ref=lr, required_eta=e_req)
    signs = {np.sign(v[3]) for v in req.values() if np.isfinite(v[3])}
    print(f"   required-elasticity signs: "
          f"{sorted(int(s) for s in signs)} "
          f"-> {'NO monotone attendance response can fit all four'
             if len(signs) > 1 else 'sign-consistent'}")
    print(f"   fitted |eta| = {abs(eta):.3f}; largest required |eta| = "
          f"{max(abs(v[3]) for v in req.values() if np.isfinite(v[3])):.2f} "
          f"(= {(max(abs(v[3]) for v in req.values() if np.isfinite(v[3]))
               / max(abs(eta), 1e-9)):.0f}x the fitted shrink target)")

    print("   after a perfect per-fixture rescale the total is exact by "
          "construction; what remains is the within-fixture gradient above "
          "(spread column), i.e. a crowd scalar matches the TOTAL and leaves "
          "the ordering untouched")

    # ---- 4. is the gradient in flow, or in flow x street width? -----------
    # The octile table alone cannot tell these apart, because E mixes street
    # width with time of day. Cross the two: if the deficit rises with E
    # inside every width class the surface is too steep in flow; if its sign
    # depends on width it is not a flow-elasticity problem at all.
    print("\n== 4. flow gradient, or flow x width? ==")
    feats = pd.read_csv(ROOT / "data" / "road_geometry_features.csv",
                        usecols=["road_osmid", "lanes"]).drop_duplicates(
                            "road_osmid").set_index("road_osmid")["lanes"]
    hold["lanes"] = hold["road_osmid"].map(feats)
    hold["lb"] = pd.cut(hold["lanes"], [0, 1, 2, 3, 99],
                        labels=["1", "2", "3", "4+"])
    hold["oq"] = pd.qcut(hold["E"], 4, labels=["E1", "E2", "E3", "E4"],
                         duplicates="drop")
    cell = (hold.groupby(["lb", "oq"], observed=True)
            .agg(n=("E", "size"), E=("E", "sum"),
                 obs=("observed", "sum"), pred=("pred_speeding", "sum")))
    if not cell.empty:
        cell["deficit"] = cell["obs"] / cell["pred"].clip(lower=1e-12)
        piv = cell["deficit"].unstack()
        print("   deficit (obs/pred) by lanes x E quartile:")
        print(piv.round(2).to_string())
        print("   E4/E1 deficit ratio inside each width class: "
              + ", ".join(f"{k}: {v:.2f}" for k, v in
                          (piv["E4"] / piv["E1"]).items()))
        print("   monotone in E inside every class? "
              + ("yes -> a flow-elasticity problem"
                 if bool(np.all(piv["E4"] > piv["E1"]))
                 else "NO -> the sign flips with width, so this is an "
                      "interaction, not a missing elasticity"))
        for (lb, oq), r in cell.iterrows():
            add("deficit by lanes x E quartile", f"{lb} lanes | {oq}",
                int(r["n"]), float(r["obs"]), float(r["pred"]), float(r["E"]))
    # where the observed counts actually live
    sh = (hold.groupby("lb", observed=True)
          .agg(obs=("observed", "sum"), E=("E", "sum")))
    all_obs = float(hold["observed"].sum())
    all_E = float(hold["E"].sum())
    sh["obs_share"] = sh["obs"] / max(all_obs, 1e-12)
    sh["E_share"] = sh["E"] / max(all_E, 1e-12)
    print("\n   observed counts vs exposure by width "
          "(shares of the whole holdout):")
    print(sh[["obs", "obs_share", "E_share"]].round(3).to_string())
    nan_obs = float(hold.loc[hold["lanes"].isna(), "observed"].sum())
    print(f"   (rows with no geometry: {nan_obs:,.0f} counts, "
          f"{nan_obs / max(all_obs, 1e-12):.0%} of all counts)")

    out = pd.DataFrame(rows)
    out.to_csv(ARM / "level_gap.csv", index=False)
    print(f"\nwrote {ARM / 'level_gap.csv'}")


if __name__ == "__main__":
    main()
