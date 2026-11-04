"""Why does the NB refit fall through to poisson-fixed?

Reproduces the T9 failure on a synthetic frame carrying the two properties
the Suncorp panel has and a well-posed regression does not:

  1. counts are ~1.5% non-zero (85,778 cells / 1,261 events -> mean 0.0376),
  2. the design carries a *perfectly separating* level -- a dummy that is 1
     for a handful of rows and 0 for every row that has an event.

(2) has no finite MLE: the coefficient on such a column can be driven toward
-minus-infinity while the log-likelihood only approaches its supremum. Poisson
stops early because the curvature dies; NB2 tolerates the zeros far better
(alpha absorbs them), so LBFGS/SLSQP keep walking and the refit sanity gate
sees |b| in the thousands -> "diverged coefs" -> the whole staged ladder
collapses to poisson-fixed and the searched hierarchy is lost with it.

The comparison that matters is FIXED-ONLY, because nb-fixed -- no random, no
correlated, no grouped, no heterogeneity structure at all -- is the stage that
diverged on the cluster (max|b|=130087), while poisson-fixed on the identical
design converged. That isolates the NB2 likelihood plus the design, and
excludes the hierarchy from the cause.

Run:  python diag_refit_divergence.py [--n 8000] [--full]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metacount_nb import fit_metacount_nb  # noqa: E402


def build_frame(n=8000, n_fixtures=39, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"row_id": np.arange(n)})
    g = rng.integers(0, n_fixtures, n)
    d["fixture"] = g
    d["lanes_z"] = rng.normal(0, 1, n)
    d["maxspeed_z"] = rng.normal(0, 1, n)
    d["aadt_z"] = rng.normal(0, 1, n)
    d["log_flow_z"] = rng.normal(0, 1, n)
    d["log_length"] = rng.normal(4.1, 1.1, n)
    d["dist_stad"] = np.abs(rng.normal(3.0, 2.0, n))
    d["seg_event_uplift_z"] = rng.normal(0, 1, n)
    hb = rng.integers(0, 28, n).astype(float)
    d["rel_sin"] = np.sin(2 * np.pi * hb / 28)
    d["rel_cos"] = np.cos(2 * np.pi * hb / 28)
    # in_event must VARY: on the real 100k refit frame it does (the shipped
    # coefficient is 0.88 with SE 0.47). Holding it constant makes it an exact
    # duplicate of the intercept, which diverges every likelihood -- that is a
    # different defect from the one under test.
    d["in_event"] = (rng.random(n) < 0.35).astype(float)
    d["rel_hour"] = hb
    d["log_att"] = rng.normal(9.3, 0.5, n)
    d["event_x_dist"] = d["in_event"] * d["dist_stad"]
    d["att_x_maxspeed"] = d["log_att"] * d["maxspeed_z"]
    d["att_x_length"] = d["log_att"] * d["log_length"]
    d["att_x_lanes"] = d["log_att"] * d["lanes_z"]
    d["att_x_dist"] = d["log_att"] * d["dist_stad"]
    d["ko_tod_sin"] = rng.normal(0, 1, n)
    d["ko_tod_cos"] = rng.normal(0, 1, n)
    d["clock_prof"] = rng.normal(0, 1, n)
    d["flow_anom"] = rng.normal(0, 1, n)
    d["evu_x_att"] = d["seg_event_uplift_z"] * d["log_att"]
    d["ko_x_dist"] = d["ko_tod_cos"] * d["dist_stad"]
    d["tod_sin"] = np.sin(2 * np.pi * hb / 28)
    # three competition codes, so the two dummies are not exhaustive and hence
    # not collinear with the intercept (the panel also has SuperRugby)
    comp = rng.integers(0, 3, n)
    d["fxt_NRL"] = (comp == 0).astype(float)
    d["fxt_ALeague"] = (comp == 1).astype(float)
    d["big_game"] = (g % 7 == 0).astype(float)
    for name, share in (("hw_residential", 0.45), ("hw_primary", 0.30),
                        ("hw_service", 0.06), ("hw_tertiary", 0.10),
                        ("hw_motorway", 0.01)):
        d[name] = (rng.random(n) < share).astype(float)
    # the pathology: this level's rows never carry an event
    sepv = np.zeros(n)
    sepv[rng.choice(n, max(40, n // 300), replace=False)] = 1.0
    d["hw_trunk_link"] = sepv
    eta = (-7.0 + 0.35 * d["log_length"] + 0.55 * d["hw_residential"]
           + 0.40 * d["hw_primary"] + 3.2 * d["hw_service"]
           + 0.9 * d["lanes_z"] - 0.5 * d["dist_stad"]
           + 0.4 * d["seg_event_uplift_z"] + 0.6 * d["in_event"]
           + 0.25 * d["log_att"] + 0.15 * d["clock_prof"])
    y = rng.poisson(np.exp(eta)).astype(float)
    y[d["hw_trunk_link"] > 0.5] = 0.0     # enforce EXACT separation
    d["Y"] = y
    d["logexp"] = np.log(rng.lognormal(4.0, 1.0, n) + 1.0)
    print(f"  synthetic: n={n:,} non-zero={int((d['Y'] > 0).sum()):,} "
          f"({100 * (d['Y'] > 0).mean():.2f}%) mean={d['Y'].mean():.4f} "
          f"events={int(d['Y'].sum()):,}")
    print(f"  separated level hw_trunk_link: {int(d['hw_trunk_link'].sum())} "
          f"rows, events there = "
          f"{int(d.loc[d['hw_trunk_link'] > 0.5, 'Y'].sum())}")
    return d


FIXED = ["hw_residential", "hw_primary", "hw_service", "hw_tertiary",
         "hw_motorway", "hw_trunk_link", "log_length", "tod_sin", "in_event",
         "rel_hour", "log_att", "event_x_dist", "fxt_NRL", "fxt_ALeague",
         "big_game", "ko_tod_sin", "ko_tod_cos", "att_x_dist", "evu_x_att",
         "att_x_lanes", "att_x_maxspeed", "att_x_length", "ko_x_dist",
         "clock_prof", "flow_anom"]
GROUPED = ["dist_stad:normal", "lanes_z:normal", "maxspeed_z:normal",
           "rel_cos:normal", "rel_sin:normal", "seg_event_uplift_z:normal"]
RDM = ["maxspeed_z:normal", "log_flow_z:normal"]
RDM_COR = ["lanes_z:normal", "aadt_z:normal"]
# metacountregressor 1.0.202 (cluster) also supports 'neglognormal'; the local
# install is 1.0.161. Prior family does not affect whether the FIXED design has
# a finite MLE, which is what this test varies, so 'normal' is used throughout.


def run(model, screen, n=8000, full=False, tag=""):
    d = build_frame(n=n)
    fixed = [c for c in FIXED if c not in {"lanes_z", "maxspeed_z"}]
    print(f"\n== {tag or ''} model={model} screen={screen} "
          f"{'grouped' if full else 'fixed-only'} ==")
    info = fit_metacount_nb(
        d, y_col="Y", id_col="row_id", model=model, R=8,
        fixed_terms=fixed,
        rdm_terms=RDM if full else [], rdm_cor_terms=RDM_COR if full else [],
        grouped_terms=GROUPED if full else [], hetro_in_means=[],
        group_id_col="fixture" if full else None,
        offset_col="logexp", verbose=True, continuous_de_warm_start=False,
        use_slsqp=True, screen_design=screen)
    b = info["coef"]["coefficient"].abs()
    worst = info["coef"].loc[b.idxmax()]
    print(f"  -> spec_label={info['spec_label']} "
          f"max|b|={b.max():.1f} on {worst['variable']!r} "
          f"degraded={info.get('degraded')} bic={info['bic']:.1f}")
    if info.get("design_drops"):
        print(f"  -> design_drops={info['design_drops']}")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--full", action="store_true",
                    help="also try the grouped hierarchical spec")
    a = ap.parse_args()
    run("poisson", screen=False, n=a.n, tag="[1] control")
    run("nb", screen=False, n=a.n, tag="[2] T9 repro")
    run("nb", screen=True, n=a.n, tag="[3] fixed")
    if a.full:
        run("nb", screen=False, n=a.n, full=True, tag="[4] grouped")
        run("nb", screen=True, n=a.n, full=True, tag="[5] grouped fixed")
