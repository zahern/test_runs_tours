"""
59_game_day_search_predict.py - MetaCount search + future-game 15-min predictions.

Stage 1 (model search): a full structural search through metacountregressor on
segment x 15-minute speeding counts (from 58_build_15min_panel.py). Candidate
variables span exposure, calendar, game context, segment geometry and road
class; roles are searched (excluded / fixed / random-independent normal) with
SA, minimising BIC, checkpointing the best spec each chunk.

Memory notes (64M-row panel): the background is sampled BEFORE any enrichment
(calendar / static context / exposure), so the full ~60M-row off-window frame
is never materialised. The profile is estimated from a random background
sample, seg_flow from the whole panel (stable per-segment means), and the
search itself runs on <= --train-cap rows.

The SA runs in wall-budgeted chunks: after each chunk the best-so-far spec
AND decision vector are written to game_day_search_best.json (atomic), so a
walltime kill / OOM loses at most the current chunk. Re-submitting the job
with --resume warm-starts the next chunk from that decision vector.

Stage 2 (future games): the searched best spec is refit and used to predict
every segment x 15-minute bin for the future fixtures (fixtures on/after
--future-from, default = last --n-future calendar fixtures). Metrics compare
predictions against observed future bins (Spearman, top-K, calibration).

Compass counts are ~5% penetration sample counts; the parquet carries
pred_speeding (sample) and pred_speeding_network (sample / --penetration).

Outputs:
  results/game_day_search_best.json        best spec + BIC + decision (resumable)
  results/game_day_search_progress.csv     chunk-by-chunk BIC
  results/game_day_pred_15min.parquet      future games x segment x bin predictions
  results/game_day_holdout_metrics.json    future-game validation metrics
  results/game_day_model_coefs.csv         refit coefficients
  results/game_day_predict.tex             LaTeX table snippet
  figures/game_day_search_convergence.png  search progress
  figures/game_day_calibration.png         predicted vs observed (future games)
  figures/game_day_surfaces.png            per-game predicted 15-min surfaces
  figures/game_day_coefs.png               coefficient magnitudes

Usage:
  python scripts/59_game_day_search_predict.py --pilot
  python scripts/59_game_day_search_predict.py [--future-from 2026-02-01]
  python scripts/59_game_day_search_predict.py --resume   # continue chunks
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import (fit_metacount_nb, predict_metacount,  # noqa: E402
                          predict_metacount_sim, prepare_metacount_frame)

try:
    import jax  # noqa: E402
except Exception:  # noqa: BLE001 - JAX is optional at import time
    jax = None

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIG = PROJ / "figures"
PANEL = RES / "segment_15min_panel.parquet"
CAL = DATA / "fixture_calendar.csv"
ECF = DATA / "near_miss_hb_ecf.csv"
FEATS = DATA / "road_geometry_features.csv"
AADT_CSV = DATA / "segment_aadt.csv"
# The current PyPI engine is JAX-first; sparse-agds is legacy and falls back
# automatically. Default to jax.
# Wall-budgeted chunked SA needs ExperimentBuilder.run(max_time=, refit=),
# available from 1.0.155.
ENGINE_DEFAULT = "jax"
TZ = "Australia/Brisbane"
# The response the deployment is actually justified on.
    #
    # WAS trips_mean_gt_10, described here as "at least 10 km/h over the posted
    # limit". That description was wrong: the panel column counts trips whose
    # MEAN speed is BELOW 10 km/h, so it measures congestion, not speeding, and
    # every downstream label ("speeding", pred_speeding, "ranked by speeding")
    # inherited the error. It also falls with traffic volume -- a binomial MLE
    # over 350,946 trials put the flow elasticity at -0.954 -- so the
    # flow-projection offset pushed prediction onto the busiest arterials,
    # which structurally never register slow trips. The consequence was a
    # NEGATIVE correlation between predicted and observed speeding on all six
    # holdout games (Pearson r = -0.06 to -0.14, top-quartile share 0.35-0.49
    # against 0.25 for a coin toss).
    #
    # NOW speeding_trips, built by stage 58 as vehicle_count where the segment's
    # mean speed exceeds a CREDIBLE posted limit (>= 40 km/h; below that this
    # extract posts 5-20 km/h limits on living/service streets, where the
    # comparison is meaningless). It rises with speed, so a flow offset is at
    # least directionally sensible, and it retains 68.7% of the unfiltered
    # target with the top 100 segments carrying 61.1%.
TARGET_DEFAULT = "speeding_trips"
# The measured denominator for the speeding propensity: trips that actually
# got a speed reading. Defined here (not next to search_frame) because the
# panel reader's column list needs it. See CHECKPOINT_COLS for why its
# absence was so costly.
BINOMIAL_DENOM = "trips_with_judged_speed"
# The successes and the trials of a binomial must be counted over the SAME
# thing. `speeding_trips` counts VEHICLES on an over-limit segment-bin, so its
# denominator has to be vehicle_count; pairing it with trips_with_judged_speed
# mixed units, and because one vehicle can make several trips in a bin, k > n
# was reachable. The exact likelihood caught it:
#     [binom] fit failed: BinomialFitError: 4 rows have successes outside [0, n]
# which was the only symptom -- the count fit carried on regardless.
# The trips_* targets are genuinely trip counts, so they keep the judged-trip
# denominator.
BINOMIAL_DENOM_BY_TARGET = {
    "speeding_trips": "vehicle_count",
}


def binomial_denominator(target):
    """Denominator column matching the unit of `target`."""
    return BINOMIAL_DENOM_BY_TARGET.get(target, BINOMIAL_DENOM)
WINDOW_HOURS = (-3.0, 4.0)
BINS_PER_HOUR = 4
REL_BINS = 28
CONT_VARS = ["lanes_z", "maxspeed_z", "curvature_z", "dist_inter_z",
             "node_degree_z", "log_length", "lanes_known"]
CAL_VARS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "weekend"]
GAME_VARS = ["in_event", "rel_hour", "log_att", "dist_stad", "event_x_dist"]
# Game-day dynamics that must ALWAYS survive the search (user requirement:
# a game-day plan needs game-day structure even when BIC would drop it).
GAME_FORCE = ["in_event", "rel_hour", "log_att", "event_x_dist"]
# Within-window temporal shape (serial persistence is carried by these
# engineered features plus segment-level grouping; metacountregressor
# >=1.0.165 adds a real AR(1)/AR(2) NB2 estimator used in stage 2b below).
TEMPORAL_VARS = ["rel_hour_sq", "rel_sin", "rel_cos", "pre_game", "post_game"]
# Fixture attributes (NOT fixture identity: future games are unseen labels,
# so differentiation must come from attributes — code, crowd tier, kickoff).
# fxt_* code dummies (SuperRugby reference), big_game crowd tier, ko_* kickoff
# time-of-day. Spatiotemporal interactions let shape vary by game:
# rel_x_dist (near-stadium timing), att_x_dist (big-game footprint),
# ev_x_lanes (game traffic uses different road types).
GAME_ATTR_VARS = ["fxt_NRL", "fxt_ALeague", "big_game", "ko_tod_sin",
                  "ko_tod_cos"]
INTERACT_VARS = ["rel_x_dist", "att_x_dist", "ev_x_lanes"]
GAME_WIDE_VARS = GAME_ATTR_VARS + INTERACT_VARS
SLIM_CONT = ["lanes_z", "maxspeed_z", "dist_inter_z", "lanes_known"]
SLIM_CAL = ["tod_sin", "tod_cos", "weekend"]
SLIM_GAME = ["in_event", "rel_hour", "log_att", "dist_stad", "event_x_dist"]
BEST_JSON = RES / "game_day_search_best.json"
PROGRESS_CSV = RES / "game_day_search_progress.csv"
CHECKPOINT_COLS = ["road_osmid", "bin", TARGET_DEFAULT, "vehicle_count",
                   "mean_vehicle_speed_kmh",
                   # The posted limit the speeding target was derived against.
                   # Carried so that stage 59 can REPORT why a segment is or is
                   # not eligible (a segment under the trusted limit is scored
                   # 0 by construction, which is different from a segment that
                   # was measured and genuinely had no speeding). Without this
                   # column a reader cannot tell those two apart.
                   "speed_limit",
                   # The measured denominator for the speeding propensity.
                   # The response is a COUNT of speeding TRIPS, so its true
                   # denominator is the number of trips that actually got a
                   # speed reading -- not trip_count, because a trip with no
                   # speed cannot be classified as speeding (trip_exceed is
                   # NaN and NaN > margin is False, so unjudged trips bias
                   # the rate down exactly where judging is worst).
                   #
                   # This column was MISSING from the list, so the panel was
                   # read without it. That silently disabled three features
                   # at once on cluster job 26077240: the exposure
                   # calibration reported "requested-but-unavailable" in all
                   # five arms, the per-class uplift found 0 classes, and the
                   # binomial would have been unavailable too. The failure was
                   # invisible because the first check bailed without a
                   # print. Load it explicitly.
                   BINOMIAL_DENOM]
# Memory-safety caps for the full (non-pilot) run. The search frame is
# <= train_cap rows; the profile comes from a random background sample.
EVENT_CAP = 100000
BG_CAP = 150000
TRAIN_CAP = 100000
PROFILE_ROWS = 500000
# Bump whenever the candidate set / constraints change shape: older
# checkpoints (different dim) can neither warm-start nor be trusted as-is.
# GEN 6: --group-by event (fixture-grouped per-event split); fixture/code
# columns carried through search_frame; wide + temporal-x block unchanged.
# GEN 7: seg_id proxy column (constant-per-segment) added to the search frame
# so the SEGMENT ITSELF can carry the grouped (role 4) effect. Previously the
# only grouped-eligible variables were geometry covariates, so "group by
# segment" could only ever mean "let lanes_z vary by segment" -- the segment
# identity itself was never a candidate. Candidate set grows by 1, so the
# decision dimension changes and old checkpoints must not warm-start.
SEARCH_GEN = 7
# Transferable temporal structure (all train-anchored, all known for future
# fixtures — unlike autoregression, which needs unobserved future lags):
#   seg_risk_z: EB-shrunk per-segment speeding propensity from train history
#   exp_lag1:   within-window lagged exposure (buildup/dissipation)
#   clock_prof: fixture-type x rel_bin empirical event-clock shape
SPLIT_CODES = ["NRL", "ALeague", "SuperRugby"]
RISK_VARS = ["seg_risk_z"]
LAG_VARS = ["exp_lag1"]
CLOCK_VARS = ["clock_prof"]
TIME_X_VARS = RISK_VARS + LAG_VARS + CLOCK_VARS
# Segment-identity term for the grouped (role 4) effect.
#
# This started as a CONSTANT 1.0 column, on the reasoning that a
# within-group-constant covariate makes the grouped coefficient a pure
# per-segment random intercept. That is right about the algebra and wrong
# about this package: build_spec() screens every variable for
# identifiability and rejects any with a single unique value --
#   "variable 'seg_group_id' was flagged unidentifiable (only one unique
#    value) but assigned role 4"
# so the whole structure came back None, fitness() returned the 1e12
# sentinel, and the probe reported the segment identity as untestable.
#
# A varying column is required. Any covariate that is constant WITHIN a
# group achieves the same thing: its grouped coefficient is estimated
# from how the groups differ in that covariate, which for a per-segment
# grouping is precisely a segment-level propensity contrast. log_length
# is constant within a segment and is therefore the natural carrier.
SEG_GROUP_VAR = "log_length"


def _utcnow_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)
    return path


def _clear_jax_caches():
    if jax is not None:
        try:
            jax.clear_caches()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 10:
        return float("nan")
    ra = np.array(pd.Series(a).rank(), dtype=float)
    rb = np.array(pd.Series(b).rank(), dtype=float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / max(denom, 1e-12))


def _jaccard_topk(obs, pred, k):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    k = int(min(k, len(obs)))
    if k <= 0:
        return float("nan")
    top_obs = set(np.argsort(-obs, kind="stable")[:k].tolist())
    top_pred = set(np.argsort(-pred, kind="stable")[:k].tolist())
    return float(len(top_obs & top_pred) / max(len(top_obs | top_pred), 1))


def _calibration(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 10 or np.var(pred) <= 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(pred, obs, 1)
    return float(slope), float(intercept)


def load_panel(path, target=None):
    """Read the panel with the columns this run actually needs.

    The requested target is added to CHECKPOINT_COLS rather than assumed to be
    in it. CHECKPOINT_COLS carries TARGET_DEFAULT, so a run given any other
    --target used to read a frame without its own response column and then die
    far downstream inside the search with a bare KeyError. That is a trap worth
    removing now that the target is a deliberate, changeable decision.
    """
    cols = list(CHECKPOINT_COLS)
    if target and target not in cols:
        cols.append(target)
    cols = list(dict.fromkeys(cols))
    if Path(path).exists():
        available = set(_pq_names(path))
        missing = [c for c in cols if c not in available]
        if missing:
            raise SystemExit(
                "panel %s is missing %s. Rebuild it with\n"
                "  python scripts/58_build_15min_panel.py --all\n"
                "(delete the existing results/segment_15min_panel_????-??.parquet "
                "first -- finalize skips months that are already present)."
                % (path, ", ".join(missing)))
        return pd.read_parquet(path, columns=cols)
    parts = sorted(RES.glob("segment_15min_panel_????-??.parquet"))
    if not parts:
        raise SystemExit("no 15-min panel found (run 58 first)")
    return pd.concat([pd.read_parquet(p, columns=cols)
                      for p in parts], ignore_index=True)


def _pq_names(path):
    import pyarrow.parquet as pq
    return set(pq.ParquetFile(path).schema_arrow.names)


def _assert_split(cal, future_ids, train_ids):
    """Fail loudly if the holdout is not actually held out.

    Three invariants, each of which was violated by the old label-based
    split: fixture_ids must be unique; train and future must partition the
    calendar exactly; and no future row may sit earlier in time than any
    train row (a future label matching a 2025 twin is exactly this bug).
    """
    ids = cal["fixture_id"]
    if bool(ids.duplicated().any()):
        dupes = sorted(ids[ids.duplicated(keep=False)].unique())
        raise SystemExit(f"FATAL: fixture_id is not unique: {dupes[:5]}")
    overlap = future_ids & train_ids
    if overlap:
        raise SystemExit(f"FATAL: {len(overlap)} fixtures are in BOTH train "
                         f"and future: {sorted(overlap)[:5]}")
    if (future_ids | train_ids) != set(ids):
        raise SystemExit("FATAL: train/future do not partition the calendar")
    fut = cal[cal["fixture_id"].isin(future_ids)]
    trn = cal[cal["fixture_id"].isin(train_ids)]
    if len(fut) and len(trn):
        latest_train = trn["ko"].max()
        earliest_future = fut["ko"].min()
        if earliest_future < latest_train:
            raise SystemExit(
                f"FATAL: holdout is interleaved with train. latest train "
                f"kickoff {latest_train} >= earliest future kickoff "
                f"{earliest_future}; the 'future' fixtures must all postdate "
                "every training fixture")


def load_fixtures():
    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"]).dt.tz_localize(TZ)
    cal["attendance"] = pd.to_numeric(cal["attendance"], errors="coerce")
    cal = cal.sort_values("ko").reset_index(drop=True)
    # `label` is NOT unique: the same fixture appears in both the 2025 and
    # 2026 seasons (NRL Broncos-Eels, NRL Broncos-Dolphins and SuperRugby
    # Reds-Waratahs each have two rows). Every downstream selection keyed on
    # the label alone therefore matched both seasons, so the "future" holdout
    # pulled in three 2025 games WITH observed counts, and those same games
    # were dropped from training too (train = all labels - future labels).
    #
    # fixture_id is the unique per-row key (label + kickoff date). All
    # train/future selection now goes through it; `label` is kept for display.
    cal["fixture_id"] = (cal["label"].astype(str)
                         + " @ " + cal["ko"].dt.strftime("%Y-%m-%d %H:%M"))
    dup = cal["label"].duplicated(keep=False)
    if bool(dup.any()):
        print(f"   NOTE: {int(dup.sum())} calendar rows share a label with "
              f"another row across {cal.loc[dup, 'label'].nunique()} names; "
              "using fixture_id for the train/future split")
    return cal


def static_context():
    feats = pd.read_csv(FEATS)
    feats["road_osmid"] = pd.to_numeric(feats["road_osmid"], errors="coerce")
    feats = feats.dropna(subset=["road_osmid"]).copy()
    feats["road_osmid"] = feats["road_osmid"].astype(np.int64)
    feats = feats.groupby("road_osmid").first().reset_index()
    ecf = pd.read_csv(ECF, usecols=["road_osmid", "highway", "dist_stadium_km"])
    ecf["road_osmid"] = pd.to_numeric(ecf["road_osmid"], errors="coerce")
    ecf = ecf.dropna(subset=["road_osmid"]).copy()
    ecf["road_osmid"] = ecf["road_osmid"].astype(np.int64)
    ecf = ecf.groupby("road_osmid").first().reset_index()
    ctx = feats.merge(ecf, on="road_osmid", how="left")
    # lanes: 3,810 of 7,825 segments (48.7%) have no OSM lanes tag, and the
    # gap is structural rather than random -- coverage runs from 8.9% on
    # service roads to 100% on motorway. Filling the gap with 1 (as
    # requested) asserts every untagged road is single-lane, which is
    # backwards for service roads (kerbside, usually multi-lane) and would
    # bias lanes_z exactly where the search is trying to estimate it. So:
    # fill with 1 as asked, and carry lanes_known so the model can correct
    # for the fill instead of inheriting it. The indicator is searchable
    # (CONT_VARS), so BIC decides whether the gap carries information.
    _lanes_raw = pd.to_numeric(ctx["lanes"], errors="coerce")
    for col in ("maxspeed_kmh", "lanes", "curvature_deg_km",
                "dist_to_intersection_m", "node_degree", "length_m"):
        ctx[col] = pd.to_numeric(ctx[col], errors="coerce")
        ctx[col] = ctx[col].fillna(float(ctx[col].median()))
    # Applied AFTER the median fill above so the 1 wins for untagged rows.
    ctx["lanes"] = _lanes_raw.fillna(1.0)
    ctx["lanes_known"] = (~_lanes_raw.isna()).astype(float)
    # Missing OSM `highway` tag -> "unclassified", not a synthetic "unknown".
    # 2,744 of 7,825 segments (35%) have no tag, and "unknown" was being
    # searched as its own dummy: hw_unknown came back Fixed in the 61b spec,
    # so a third of the network carried a private coefficient. OSM's own
    # convention is that untagged ways are unclassified, and the real
    # unclassified class is tiny (232 segments), so this merges a large
    # spurious group into a small real one instead of the reverse.
    _hw = ctx["highway"].astype("string").str.strip().str.lower()
    _hw = _hw.replace("", pd.NA).fillna("unclassified")
    ctx["highway"] = _hw.astype(str)
    ctx["dist_stadium_km"] = pd.to_numeric(ctx["dist_stadium_km"],
                                           errors="coerce")
    ctx["dist_stadium_km"] = ctx["dist_stadium_km"].fillna(
        float(ctx["dist_stadium_km"].median()))
    for base, out in (("lanes", "lanes_z"), ("maxspeed_kmh", "maxspeed_z"),
                      ("curvature_deg_km", "curvature_z"),
                      ("dist_to_intersection_m", "dist_inter_z"),
                      ("node_degree", "node_degree_z")):
        std = float(ctx[base].std())
        ctx[out] = (ctx[base] - float(ctx[base].mean())) / (std if std > 0 else 1.0)
    ctx["log_length"] = np.log1p(ctx["length_m"].clip(lower=0.0))
    # AADT hook: prefer TMR-joined data/segment_aadt.csv (road_osmid, aadt)
    # when present; otherwise the probe-flow proxy is built later per frame
    # (log_flow_z from seg_flow / vehicle_count) and searched alongside
    # geometry. Either way AADT enters as a candidate and survives only if
    # BIC-significant (random-parameter allowed).
    try:
        if AADT_CSV.exists():
            _a = pd.read_csv(AADT_CSV)
            _a["road_osmid"] = pd.to_numeric(_a["road_osmid"], errors="coerce")
            _a = _a.dropna(subset=["road_osmid"]).copy()
            _a["road_osmid"] = _a["road_osmid"].astype(np.int64)
            _col = "aadt" if "aadt" in _a.columns else ("AADT" if "AADT" in _a.columns else None)
            if _col is not None:
                _a[_col] = pd.to_numeric(_a[_col], errors="coerce")
                ctx = ctx.merge(_a[["road_osmid", _col]].rename(columns={_col: "aadt"}),
                                on="road_osmid", how="left")
                _m, _s = float(ctx["aadt"].median()), float(ctx["aadt"].std())
                ctx["aadt"] = ctx["aadt"].fillna(_m)
                ctx["aadt_z"] = (ctx["aadt"] - float(ctx["aadt"].mean())) / (_s if _s > 0 else 1.0)
    except Exception:
        pass
    # Winsorize standardized candidates: one degenerate segment (e.g. 45 m
    # with imputed lanes + top-capped AADT, see o25899100 where a single
    # osmid hogged 84% of predicted speeding) must not drive the
    # correlated random slopes to 250x explosions.
    for _zc in [c for c in ctx.columns if c.endswith("_z")]:
        _zv = pd.to_numeric(ctx[_zc], errors="coerce")
        ctx[_zc] = _zv.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
    return ctx


def bin_ns_array(panel):
    """UTC epoch nanoseconds for the panel bin column (int64, no tz copies)."""
    return (panel["bin"].dt.tz_convert("UTC").dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]").view("int64"))


def _window_bounds_ns(ko):
    start = ko + pd.Timedelta(hours=WINDOW_HOURS[0])
    end = ko + pd.Timedelta(hours=WINDOW_HOURS[1])
    s_ns = int(pd.Timestamp(start).tz_convert("UTC").value)
    e_ns = int(pd.Timestamp(end).tz_convert("UTC").value)
    return start, s_ns, e_ns


def covered_mask(bin_ns, cal):
    """Boolean mask of rows inside any fixture's [-3h, +4h) window."""
    mask = np.zeros(len(bin_ns), dtype=bool)
    for ko in cal["ko"]:
        _, s_ns, e_ns = _window_bounds_ns(ko)
        mask |= (bin_ns >= s_ns) & (bin_ns < e_ns)
    return mask


def sample_background(panel, bin_ns, covered, rng, sample_n):
    """Random off-window rows, sampled BEFORE any frame copy/enrichment.

    The previous version did panel.loc[~mask].copy() over ~60M rows for every
    call, then sampled -- which alone needed ~15-20GB before the search even
    started. Sampling integer positions first keeps this at ~sample_n rows.
    """
    cand = np.flatnonzero(~covered)
    if cand.size == 0:
        return panel.iloc[[]].copy()
    n = int(sample_n)
    if n and cand.size > n:
        picks = rng.choice(cand, n, replace=False)
        picks.sort()
    else:
        picks = cand
    return panel.iloc[picks].copy()


def window_rows(panel, bin_ns, ko, fixture_id):
    start, s_ns, e_ns = _window_bounds_ns(ko)
    idx = np.flatnonzero((bin_ns >= s_ns) & (bin_ns < e_ns))
    if idx.size == 0:
        return panel.iloc[[]].copy()
    sub = panel.iloc[idx].copy()
    # `fixture` is the UNIQUE fixture_id. Every downstream groupby/unique()
    # (metrics, MIP per-fixture loop, top-20 table) keys on this column, so
    # it must never be a label that repeats across seasons.
    sub["fixture"] = fixture_id
    sub["rel_bin"] = ((sub["bin"] - start) / pd.Timedelta(minutes=15)
                      ).astype(int).clip(0, REL_BINS - 1)
    return sub


def add_calendar(frame):
    dt = pd.to_datetime(frame["bin"], utc=True).dt.tz_convert(TZ)
    tod = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek
    frame["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    frame["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    frame["weekend"] = (dow >= 5).astype(float)
    frame["tod_quarter"] = dt.dt.hour * BINS_PER_HOUR + dt.dt.minute // 15
    frame["dow"] = dow
    return frame


def build_profile(background):
    overall = float(background["vehicle_count"].mean())
    prof = (background.groupby(["dow", "tod_quarter"])["vehicle_count"].mean()
            / max(overall, 1e-9))
    return prof.to_dict()


def add_exposure(frame, ctx, seg_flow, profile):
    if profile:
        prof = pd.DataFrame(
            [(dow, quarter, factor) for (dow, quarter), factor
             in profile.items()],
            columns=["dow", "tod_quarter", "bin_factor"])
        frame = frame.merge(prof, on=["dow", "tod_quarter"], how="left")
        frame["bin_factor"] = frame["bin_factor"].fillna(1.0)
    else:
        frame["bin_factor"] = 1.0
    frame["exp"] = (frame["road_osmid"].map(seg_flow).fillna(0.0)
                    * frame["bin_factor"])
    frame["logexp"] = np.log1p(frame["exp"])
    # Probe AADT proxy (sample mean flow per segment): standardised for search.
    # If TMR AADT (aadt_z) already came via ctx, both compete and BIC decides.
    try:
        _lf = np.log1p(frame["road_osmid"].map(seg_flow).fillna(0.0).to_numpy(float))
        _m, _s = float(np.mean(_lf)), float(np.std(_lf))
        frame["log_flow_z"] = np.clip((_lf - _m) / (_s if _s > 0 else 1.0), -3.0, 3.0)
    except Exception:
        frame["log_flow_z"] = 0.0
    return frame


def _shrunk_ratio(y_sum, e_sum, prior, k):
    """(Y + k*prior) / (E + k). Shrink thin cells toward the global ratio.

    A raw Y/E blows up wherever the exposure sum is small (a rare segment
    with one busy 15-min bin), which would inject enormous fake variation
    into the offset. The pseudo-count k is in exposure units, so it plays
    the same role for both segments and time cells.
    """
    y = np.asarray(y_sum, dtype=float)
    e = np.asarray(e_sum, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (y + k * prior) / np.maximum(e + k, 1e-9)
    return np.where(np.isfinite(r) & (r > 0), r, prior)


def fit_exposure_calibration(frame, denom_col="trips_with_judged_speed",
                             k_seg=8.0, k_cell=40.0, min_rows=500):
    """Calibrate the flow proxy onto the OBSERVED trip denominator.

    WHY: the response is a count of speeding TRIPS, so its true
    denominator is the number of trips that actually got a speed reading.
    The offset in use is exp = seg_flow * bin_factor * uplift -- a TMR flow
    projection standing in for a denominator that stage 58 already
    measured. beta_exposure drifts off 1 because that proxy does not
    reproduce real trip counts, which is what the exposure LRT is picking
    up when it rejects the pinned offset at chi2(1)=364.

    Note the denominator is trips_with_judged_speed and NOT trip_count: a
    trip with no speed reading cannot be classified as speeding
    (trip_exceed is NaN and NaN > 10 is False), so unjudged trips would
    bias the rate down exactly where judging is worst -- motorways and
    night bins.

    Two-step shrunk multiplicative calibration:
      1. per-segment ratio  r_i    = (Y_i + k*g) / (E_i + k)
      2. per dow x tod ratio c_j   = (Y_j + k*g) / (E_j + k)
    with g the global Y/E ratio. Both are ratios on the SAME exp column the
    model already uses, so the calibrated exposure preserves the existing
    shape (and therefore still works in the future, where only exp exists)
    while fixing the level per segment and per time cell.

    This is a property of the network, not of the response, so it is
    estimated once on the training rows and then FROZEN for prediction.
    """
    if denom_col not in frame.columns or "exp" not in frame.columns:
        # Never bail silently. This exact silent return made the calibration
        # look "unavailable" in all five arms of job 26077240 when the real
        # cause was that the panel reader never loaded the column.
        _miss = [c for c in (denom_col, "exp") if c not in frame.columns]
        print(f"   [calib] cannot fit: missing {', '.join(_miss)} from the "
              f"frame. If {denom_col} is missing, check that "
              "CHECKPOINT_COLS includes it (the panel is read with a fixed "
              "column subset).")
        return None
    y = pd.to_numeric(frame[denom_col], errors="coerce")
    e = pd.to_numeric(frame["exp"], errors="coerce")
    ok = y.notna() & e.notna() & np.isfinite(e) & (e > 0)
    if int(ok.sum()) < min_rows:
        print(f"   [calib] only {int(ok.sum())} usable rows for "
              f"{denom_col}; skipping calibration")
        return None
    f = frame.loc[ok, ["road_osmid", "dow", "tod_quarter"]].copy() \
        if "dow" in frame.columns and "tod_quarter" in frame.columns \
        else frame.loc[ok, ["road_osmid"]].copy()
    f["_y"] = y[ok].to_numpy(float)
    f["_e"] = e[ok].to_numpy(float)

    g = float(f["_y"].sum() / max(f["_e"].sum(), 1e-9))
    if not np.isfinite(g) or g <= 0:
        print("   [calib] global Y/E ratio is not positive; skipping")
        return None

    seg = f.groupby("road_osmid")[["_y", "_e"]].sum()
    seg_ratio = _shrunk_ratio(seg["_y"].to_numpy(), seg["_e"].to_numpy(),
                              g, k_seg)
    cal = {"seg": {str(k): float(v) for k, v
                   in zip(seg.index, seg_ratio)},
           "seg_default": float(g), "k_seg": k_seg,
           "k_cell": k_cell, "n_rows": int(len(f)),
           "n_segments": int(len(seg)), "denom_col": denom_col}

    # Step 2 operates on the SEGMENT-ADJUSTED exposure, so the two steps
    # compose instead of fighting: cell ratios are estimated against
    # e * r_seg, not against the raw proxy. Estimating both against the raw
    # proxy and multiplying the results double-counts the global level --
    # each step is shrunk toward g, so the product lands near g^2 and the
    # offset ends up ~2x too small when g is near 0.5.
    _r_seg = np.array([cal["seg"].get(str(s), g)
                       for s in f["road_osmid"]], dtype=float)
    e1 = f["_e"].to_numpy(float) * _r_seg
    f["_e1"] = e1

    if "dow" in f.columns and "tod_quarter" in f.columns:
        cell = f.groupby(["dow", "tod_quarter"])[["_y", "_e1"]].sum()
        cell_ratio = _shrunk_ratio(cell["_y"].to_numpy(),
                                   cell["_e1"].to_numpy(), g, k_cell)
        cal["cell"] = {(str(a), str(b)): float(v) for (a, b), v
                       in zip(cell.index, cell_ratio)}
    else:
        cal["cell"] = {}

    # Final level from the FULL composition, so that
    # e * r_seg * r_cell * level reproduces sum(observed) exactly. The
    # offset is pinned at coefficient 1, so this level is not free: getting
    # it wrong biases the fitted exposure directly.
    _r_cell = np.array([cal["cell"].get((str(a), str(b)), 1.0)
                        for a, b in zip(f["dow"], f["tod_quarter"])],
                       dtype=float) if cal["cell"] else np.ones(len(f))
    e2 = e1 * _r_cell
    cal["level"] = float(f["_y"].sum() / max(e2.sum(), 1e-9))

    _r = np.array([v for v in cal["seg"].values() if np.isfinite(v)])
    if _r.size:
        print(f"   [calib] {denom_col}: {len(f):,} rows / "
              f"{cal['n_segments']:,} segments, global Y/E={g:.3f}, "
              f"final level={cal['level']:.3f}, seg ratio "
              f"p10/50/90={np.percentile(_r, 10):.2f}/"
              f"{np.percentile(_r, 50):.2f}/{np.percentile(_r, 90):.2f}")
    return cal


def calibrate_exposure(frame, cal):
    """Rewrite frame['exp'] onto the calibrated scale, in place.

    Writes `exp` rather than a new column so that logexp, the offset, and
    pred_flow all follow automatically -- prepare_metacount_frame derives
    logexp = log1p(exp), and the future path reads the same exp.
    """
    if not cal:
        return frame
    e = pd.to_numeric(frame["exp"], errors="coerce").fillna(0.0).to_numpy(float)
    seg = cal.get("seg") or {}
    if seg:
        _s = frame["road_osmid"].astype(str) if "road_osmid" in frame.columns \
            else None
        if _s is not None:
            dflt = cal.get("seg_default", 1.0)
            r = _s.map(seg).fillna(dflt).to_numpy(float)
        else:
            r = np.full(len(frame), cal.get("seg_default", 1.0), dtype=float)
    else:
        r = np.full(len(frame), 1.0, dtype=float)
    cell = cal.get("cell") or {}
    if cell and "dow" in frame.columns and "tod_quarter" in frame.columns:
        keys = list(zip(frame["dow"].astype(str), frame["tod_quarter"].astype(str)))
        c = np.array([cell.get(kk, 1.0) for kk in keys], dtype=float)
    else:
        c = np.full(len(frame), 1.0, dtype=float)
    out = np.maximum(e * r * c * float(cal.get("level", 1.0)), 1e-6)
    frame["exp"] = out
    frame["logexp"] = np.log1p(out)
    return frame


def add_exposure_shape(frame, params=None, hinge=False, interact_hw=False):
    """Curvature and road-class interaction on the log exposure.

    logexp is force-fixed, so with the offset pinned at 1 the model has no
    way to bend the flow->speeding curve. The exposure LRT is a test of
    exactly that curve. Two opt-in enrichments:

    hinge      : linear-hinge (piecewise linear) basis on logexp at two
                 frozen quantile knots, ORTHOGONALISED against
                 [1, logexp]. Without that projection the hinges correlate
                 ~0.65 with the forced linear term, which inflates their
                 standard errors and makes the BIC comparison against the
                 plain model unfair. The projection coefficients are
                 estimated once on TRAIN and stored in `params`, then
                 reapplied to the future path, so both use an identical
                 basis. A full natural cubic spline would be the textbook
                 choice, but the deviation form needs no spline machinery
                 that could behave differently between fit and prediction.
    interact_hw: logexp times each highway dummy, letting the flow
                 ELASTICITY differ by road class -- the standard
                 difference-in-slopes parameterisation. Deliberately NOT
                 centred: centring makes the model unable to represent a
                 pure slope difference, because non-reference rows then
                 force the intercept to zero while the centring constant
                 must simultaneously vanish. A more robust alternative to
                 trusting the per-class uplift u, since it changes the
                 elasticity rather than trying to fix the exposure scale
                 per class.

    Returns (frame, params). `params` is None unless something was added.
    """
    if not hinge and not interact_hw:
        return frame, params
    out = frame
    params = dict(params) if params else {}
    le = pd.to_numeric(out["logexp"], errors="coerce")
    le = le.replace([np.inf, -np.inf], np.nan)
    le_v = le.fillna(0.0).to_numpy(float)
    finite = np.isfinite(le_v) & np.isfinite(le.to_numpy(float))

    if hinge:
        knots = params.get("knots")
        if not knots:
            q = le.dropna()
            knots = ([float(np.quantile(q, 0.50)),
                      float(np.quantile(q, 0.80))] if len(q) >= 100 else None)
        if knots:
            params["knots"] = [float(k) for k in knots]
            hs = [np.maximum(0.0, le_v - k) for k in knots]
            # Orthogonalise against the forced linear term. Coefficients are
            # estimated on the first call (train) and reused afterwards.
            ortho = params.get("ortho")
            if not ortho:
                base = np.column_stack([np.ones(len(le_v)), le_v])[finite]
                ortho = []
                for h in hs:
                    bh, *_ = np.linalg.lstsq(base, h[finite], rcond=None)
                    ortho.append([float(bh[0]), float(bh[1])])
                params["ortho"] = ortho
            for i, (h, c) in enumerate(zip(hs, ortho), 1):
                out[f"logexp_h{i}"] = h - (c[0] + c[1] * le_v)
            print(f"   exposure shape: hinge knots at "
                  f"{[round(k, 3) for k in knots]}, orthogonalised")

    if interact_hw:
        _hw = [c for c in out.columns if c.startswith("hw_")]
        for c in _hw:
            _d = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            out[f"logexp_x_{c}"] = le_v * _d.to_numpy(float)
        if _hw:
            print(f"   exposure shape: logexp x {len(_hw)} highway classes")
    return out, (params if params else None)


def add_static(frame, ctx):
    merged = frame.merge(ctx, on="road_osmid", how="left")
    for col in CONT_VARS:
        merged[col] = merged[col].fillna(0.0)
    if "aadt_z" in merged.columns:
        merged["aadt_z"] = pd.to_numeric(merged["aadt_z"], errors="coerce").fillna(0.0)
    merged["dist_stad"] = merged["dist_stadium_km"].fillna(
        float(ctx["dist_stadium_km"].median()))
    # Keep in step with static_context(): a left-join miss on highway means
    # the segment was absent from the ECF entirely, which is the same
    # untagged case, so it takes "unclassified" rather than reviving the
    # "unknown" dummy that static_context() no longer emits.
    merged["highway"] = (merged["highway"].astype("string")
                         .fillna("unclassified").astype(str))
    return merged


def add_game_temporal(frame):
    """Within-window temporal shape (AR(1) substitute).

    metacountregressor has no autoregressive term, so the rise/peak/decay
    around kickoff is encoded explicitly: quadratic + Fourier pair over the
    28-bin window plus pre/post-kickoff steps. KO is at rel_bin 12
    (rel_hour 0); full-time ~ rel_hour +2.
    """
    rh = pd.to_numeric(frame["rel_bin"], errors="coerce").fillna(0.0).to_numpy(float)
    rel_hour = rh / BINS_PER_HOUR + WINDOW_HOURS[0]
    frame["rel_hour_sq"] = rel_hour ** 2
    frame["rel_sin"] = np.sin(2 * np.pi * rh / REL_BINS)
    frame["rel_cos"] = np.cos(2 * np.pi * rh / REL_BINS)
    frame["pre_game"] = (rel_hour < 0).astype(float)
    frame["post_game"] = (rel_hour > 2.0).astype(float)
    return frame


def add_game_wide(frame, crowd_med=None):
    """Fixture-attribute + spatiotemporal interaction candidates.

    Expects columns: fixture_type, ko_hour, attendance, log_att, rel_hour,
    dist_stad, lanes_z, in_event (background rows carry zeros/empty and
    degrade gracefully). crowd_med anchors the big_game tier to the TRAIN
    attendance distribution so future tiers are model-consistent.
    """
    n = len(frame)
    fxt = frame["fixture_type"].fillna("").astype(str) if "fixture_type" in frame.columns else pd.Series([""] * n, index=frame.index)
    frame["fxt_NRL"] = (fxt == "NRL").astype(float)
    frame["fxt_ALeague"] = (fxt == "ALeague").astype(float)
    att = pd.to_numeric(frame["attendance"], errors="coerce").fillna(0.0).to_numpy(float) if "attendance" in frame.columns else np.zeros(n)
    med = float(crowd_med) if crowd_med is not None and np.isfinite(float(crowd_med)) and float(crowd_med) > 0 else float(np.median(att)) if n else 0.0
    frame["big_game"] = (att >= max(med, 1.0)).astype(float)
    koh = pd.to_numeric(frame["ko_hour"], errors="coerce").fillna(0.0).to_numpy(float) if "ko_hour" in frame.columns else np.zeros(n)
    frame["ko_tod_sin"] = np.sin(2 * np.pi * koh / 24.0)
    frame["ko_tod_cos"] = np.cos(2 * np.pi * koh / 24.0)
    rh = pd.to_numeric(frame["rel_hour"], errors="coerce").fillna(0.0).to_numpy(float) if "rel_hour" in frame.columns else np.zeros(n)
    la = pd.to_numeric(frame["log_att"], errors="coerce").fillna(0.0).to_numpy(float) if "log_att" in frame.columns else np.zeros(n)
    ds = pd.to_numeric(frame["dist_stad"], errors="coerce").fillna(0.0).to_numpy(float) if "dist_stad" in frame.columns else np.zeros(n)
    lz = pd.to_numeric(frame["lanes_z"], errors="coerce").fillna(0.0).to_numpy(float) if "lanes_z" in frame.columns else np.zeros(n)
    ie = pd.to_numeric(frame["in_event"], errors="coerce").fillna(0.0).to_numpy(float) if "in_event" in frame.columns else np.zeros(n)
    frame["rel_x_dist"] = rh * ds
    frame["att_x_dist"] = la * ds
    frame["ev_x_lanes"] = ie * lz
    return frame


def estimate_seg_risk(train_events, target, pseudo_exp=20.0):
    """EB-shrunk per-segment speeding propensity from TRAIN windows only.

    risk_s = (Y_s + k*r0) / (E_s + k): segments with little exposure shrink
    to the global rate r0. Standardised across segments; unseen segments map
    to 0. Call BEFORE row-sampling (needs full-window sums, not samples).
    """
    g = train_events.groupby("road_osmid").agg(
        Y=(target, "sum"), E=("exp", "sum"))
    r0 = float(g["Y"].sum() / max(g["E"].sum(), 1e-9))
    risk = (g["Y"] + pseudo_exp * r0) / (g["E"] + pseudo_exp)
    mu, sd = float(risk.mean()), float(risk.std())
    return (((risk - mu) / (sd if sd > 0 else 1.0)).to_dict(), r0)


def estimate_clock_prof(train_events, target):
    """Fixture-type x rel_bin empirical event-clock shape (TRAIN only).

    Mean target rate per (fixture_type, rel_bin), normalised per code and
    smoothed. A nonparametric shape the sparse 15-min search keeps rejecting
    as parametric curvature — estimated pooled, so it survives aggregation.
    """
    if "fixture_type" not in train_events.columns:
        return {}
    g = train_events.groupby(["fixture_type", "rel_bin"])[target].mean()
    prof = {}
    for ft in g.index.get_level_values(0).unique():
        v = g.loc[str(ft)].reindex(range(REL_BINS))
        v = v.fillna(v.mean()).fillna(0.0)
        m = float(v.mean())
        v = (v / max(m, 1e-9)).rolling(3, center=True,
                                       min_periods=1).mean()
        prof[str(ft)] = v.to_numpy(float)
    return prof


def apply_temporal_x(frame, seg_risk_map=None, clock_prof=None):
    """Attach seg_risk_z / clock_prof / exp_lag1 to an enriched frame."""
    n = len(frame)
    if seg_risk_map and "road_osmid" in frame.columns:
        frame["seg_risk_z"] = (frame["road_osmid"].map(seg_risk_map)
                               .fillna(0.0).astype(float))
    elif "seg_risk_z" not in frame.columns:
        frame["seg_risk_z"] = 0.0
    if (clock_prof and "rel_bin" in frame.columns
            and "fixture_type" in frame.columns):
        rb = pd.to_numeric(frame["rel_bin"], errors="coerce").fillna(0).to_numpy(int).clip(0, REL_BINS - 1)
        ft = frame["fixture_type"].fillna("").astype(str).to_numpy()
        ones = np.ones(REL_BINS)
        frame["clock_prof"] = np.array(
            [np.asarray(clock_prof.get(f, ones), dtype=float)[b]
             for f, b in zip(ft, rb)], dtype=float)
    elif "clock_prof" not in frame.columns:
        frame["clock_prof"] = 1.0
    if ("exp" in frame.columns and "rel_bin" in frame.columns
            and "fixture" in frame.columns and "road_osmid" in frame.columns):
        order = frame.sort_values(["fixture", "road_osmid", "rel_bin"]).index
        lagged = frame.loc[order].groupby(["fixture", "road_osmid"])["exp"].shift(1)
        frame["exp_lag1"] = (lagged.reindex(frame.index)
                             .fillna(frame["exp"]).astype(float))
    elif "exp_lag1" not in frame.columns:
        frame["exp_lag1"] = (pd.to_numeric(frame["exp"], errors="coerce")
                             .fillna(0.0).astype(float)
                             if "exp" in frame.columns else 0.0)
    return frame


def estimate_flow_uplift_by_class(train_events, background, bg_by_class,
                                  class_col="highway", fallback=None):
    """Per-highway-class game-day flow uplift u[b, class].

    WHY: a single u[b] for the whole network is the most likely reason the
    exposure LRT rejects the flow channel at chi2(1)=364. One scalar per
    rel_bin collapses 35 fixtures x ~1,500 segments into 28 numbers, so it
    cannot represent that a residential street and a 4-lane arterial
    respond differently to a game. A 20 km/h arterial and a 20 km/h side
    street are given the same multiplier.

    Returns {class_value: np.ndarray(REL_BINS)}. Classes are estimated
    from the train event windows; anything with too little support falls
    back to the pooled `fallback` vector, and the pooled vector is used for
    the background divisor per class so the ratio stays self-consistent
    (a class's own background level is the right denominator when asking
    "how much does THIS class of road speed up on game day").
    """
    out = {}
    pooled = fallback if fallback is not None else np.ones(REL_BINS)
    try:
        if class_col not in train_events.columns:
            return {}
        _ev = train_events.copy()
        _ev[class_col] = _ev[class_col].astype("string").fillna("unclassified")
        _ev[class_col] = _ev[class_col].astype(str)
        ev_by = _ev.groupby([class_col, "rel_bin"])["vehicle_count"].mean()
        for cls in sorted(_ev[class_col].unique()):
            sub = ev_by.loc[cls] if cls in ev_by.index.get_level_values(0) \
                else None
            if sub is None or sub.empty:
                continue
            denom = bg_by_class.get(cls)
            if denom is None or not np.isfinite(denom) or denom <= 0:
                continue
            u = (sub / denom).reindex(range(REL_BINS)).fillna(1.0)
            u = u.replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.25, 4.0)
            out[str(cls)] = u.to_numpy(float)
    except Exception:  # noqa: BLE001
        return {}
    return out


def estimate_flow_uplift(train_events, background, bg_mean=None):
    """Stage 1 of the joint flow->speeding model: game-day flow uplift.

    u[b] = mean probe flow inside train event windows at rel_bin b, divided
    by the background mean. Future flow = seg_flow * bin_factor * u[rel_bin],
    so predicted exposure — and hence predicted speeding — inherits game-day
    attendance dynamics instead of reusing the static background profile.
    Falls back to 1.0 (static exposure) where a bin has no train support.

    bg_mean: pass a DETERMINISTIC background mean (e.g. the full-panel
    aggregate) to make u reproducible. When omitted it is taken from
    `background`, which is a random subsample, so u then varies run to run
    from sampling noise alone — which is what made three measurements of
    the same quantity disagree (0.97/0.96/0.88, 0.955/0.963/0.869,
    0.96/1.05/0.95) while the pipeline was unchanged.
    """
    try:
        ev = train_events.groupby("rel_bin")["vehicle_count"].mean()
        if bg_mean is None:
            bg_mean = float(background["vehicle_count"].mean())
        if not np.isfinite(bg_mean) or bg_mean <= 0:
            return np.ones(REL_BINS)
        u = (ev / bg_mean).reindex(range(REL_BINS)).fillna(1.0)
        u = u.replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(0.25, 4.0)
        return u.to_numpy(float)
    except Exception:
        return np.ones(REL_BINS)


def build_game_frame(panel, bin_ns, cal, ctx, seg_flow, profile, fixture_ids,
                     crowd_med=None):
    """Rows for the fixtures in `fixture_ids` (unique per calendar row).

    The frame's `fixture` column carries fixture_id, not `label`: labels
    repeat across seasons, so keying on them merged two different games
    into one surface. `label` is carried alongside for reporting.
    """
    parts = []
    for _, row in cal.iterrows():
        if row["fixture_id"] not in fixture_ids:
            continue
        sub = window_rows(panel, bin_ns, row["ko"], row["fixture_id"])
        if sub.empty:
            continue
        sub["fixture_label"] = str(row["label"])
        sub["attendance"] = float(row["attendance"]) if np.isfinite(
            row["attendance"]) else 0.0
        sub["fixture_type"] = str(row.get("fixture_type", "") or "")
        try:
            sub["ko_hour"] = float(row["ko"].hour) + float(row["ko"].minute) / 60.0
        except Exception:
            sub["ko_hour"] = 0.0
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["code"] = frame["fixture_type"].fillna("").astype(str)
    frame = add_calendar(frame)
    frame["in_event"] = 1.0
    frame["rel_hour"] = frame["rel_bin"] / BINS_PER_HOUR + WINDOW_HOURS[0]
    frame["log_att"] = np.log1p(frame["attendance"])
    frame = add_game_temporal(frame)
    frame = add_static(frame, ctx)
    frame["event_x_dist"] = frame["in_event"] * frame["dist_stad"]
    frame = add_game_wide(frame, crowd_med=crowd_med)
    frame = add_exposure(frame, ctx, seg_flow, profile)
    return frame


def enrich_background(bg, ctx, seg_flow, profile):
    bg = add_calendar(bg)
    bg["code"] = "BG"
    bg["fixture"] = "BG"
    bg["in_event"] = 0.0
    bg["rel_hour"] = 0.0
    bg["attendance"] = 0.0
    bg["log_att"] = 0.0
    bg["fixture_type"] = ""
    bg["ko_hour"] = 0.0
    bg["rel_hour_sq"] = 0.0
    bg["rel_sin"] = 0.0
    bg["rel_cos"] = 1.0
    bg["pre_game"] = 0.0
    bg["post_game"] = 0.0
    bg = add_static(bg, ctx)
    bg["event_x_dist"] = 0.0
    bg = add_game_wide(bg, crowd_med=1e18)
    bg = add_exposure(bg, ctx, seg_flow, profile)
    return bg


def search_frame(frame, target):
    cols = (["road_osmid", "bin", target, "code", "fixture"] + CAL_VARS
            + GAME_VARS + GAME_WIDE_VARS + TEMPORAL_VARS + TIME_X_VARS
            + CONT_VARS + [
                "exp", "logexp", "highway", "aadt_z", "log_flow_z",
                # The measured denominator. The response is a count of
                # speeding TRIPS, so its true denominator is the number of
                # trips that got a speed reading -- NOT trip_count, because
                # a trip with no speed cannot be classified as speeding
                # (trip_exceed is NaN and NaN > margin is False, so
                # unjudged trips would bias the rate down exactly where
                # judging is worst). Stage 58 measures it; the count
                # refit never used it, which is the whole reason the
                # exposure LRT keeps rejecting the flow proxy.
                BINOMIAL_DENOM])
    # Every denominator a known target could resolve to, so search_frame keeps
    # whichever one this run actually needs. The filter below keeps only
    # columns present in the frame, so listing them all is safe.
    cols = cols + [c for c in sorted(set(
        list(BINOMIAL_DENOM_BY_TARGET.values()) + [BINOMIAL_DENOM]))
        if c not in cols]
    work = frame[[c for c in cols if c in frame.columns]].copy()
    if target in work.columns:
        work = work.rename(columns={target: "Y"})
    else:
        work["Y"] = 0.0
    # SEG_GROUP_VAR is log_length, which is already present and already
    # forced fixed -- it is the within-group-constant carrier for the
    # grouped (role 4) effect. An earlier version added a constant
    # seg_group_id column instead; the package's identifiability screen
    # rejects single-valued variables, so that structure never built.
    if SEG_GROUP_VAR not in work.columns and "road_osmid" in work.columns:
        # Fallback only if log_length was not carried through upstream.
        work[SEG_GROUP_VAR] = 0.0
    work["row_id"] = np.arange(len(work))
    for col in (CAL_VARS + GAME_VARS + GAME_WIDE_VARS + TEMPORAL_VARS
                + TIME_X_VARS + CONT_VARS
                + ["logexp", "Y", "aadt_z", "log_flow_z"]):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            work[col] = work[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return work


def _build_evaluator(builder, variables, constraints, roles, R, engine,
                     max_latent_classes=1):
    try:
        ev = builder.build_evaluator(
            variables=variables, constraints=constraints,
            default_roles=roles, max_latent_classes=int(max_latent_classes),
            R=int(R), engine=engine)
        return ev, engine
    except (TypeError, ValueError) as exc:
        print(f"   WARN: engine={engine} unsupported ({exc}); using default")
        ev = builder.build_evaluator(
            variables=variables, constraints=constraints,
            default_roles=roles, max_latent_classes=int(max_latent_classes),
            R=int(R))
        return ev, "default-jax"


def _grouped_eligible(evaluator):
    """Variables whose allowed roles include 4 (Grouped)."""
    out = []
    try:
        allowed = getattr(evaluator, "allowed_roles", {}) or {}
        for var in (getattr(evaluator, "vars", []) or []):
            if 4 in (allowed.get(var) or []):
                out.append(var)
    except Exception:  # noqa: BLE001
        pass
    return out


def _base_seed_decision(evaluator, dispersion=0, term=None, term_role=0):
    """A MINIMAL valid decision vector for the grouped probe.

    Only the forced terms are active; everything else is Excluded (role 0).
    The probe term is set to `term_role` (1 = Fixed for the control, 4 =
    Grouped for the treatment), so the ONLY difference between the two fits
    is that single gene -- which is what makes the BIC delta attributable
    to grouping rather than to structure.

    An earlier version set every variable Fixed. That is structurally
    guaranteed to fail: 15 mutually-exclusive highway dummies plus the
    calendar/game/interaction blocks are collinear, build_spec() returns
    None, and fitness() hands back the 1e12 failure sentinel. Both arms
    then read 1e12, the delta came out exactly 0.0, and the probe
    reported a confident "grouped loses" for every term while having
    fitted nothing at all.
    """
    try:
        vars_ = list(getattr(evaluator, "vars", []) or [])
        allowed = getattr(evaluator, "allowed_roles", {}) or {}
    except Exception:  # noqa: BLE001
        return None
    if not vars_:
        return None
    D = len(vars_)
    decision = np.zeros(2 * D + 1, dtype=int)
    for i, var in enumerate(vars_):
        ok = list(allowed.get(var) or [0])
        # A variable is "forced" when Excluded is not among its options.
        if 0 not in ok:
            decision[i] = 1 if 1 in ok else (2 if 2 in ok else 0)
    if term is not None and term in vars_:
        i = vars_.index(term)
        if term_role in (allowed.get(term) or [0]):
            decision[i] = int(term_role)
    # dist genes: index 0 of each variable's allowed list is the package
    # default, so 0 is always a decodable choice.
    decision[D:2 * D] = 0
    decision[2 * D] = int(dispersion) % 2
    return decision


def _probe_grouped(evaluator, group_col, audit, args):
    """Evaluate a Grouped structure so the grouped path is actually tested.

    Why this is necessary: build_jax_data only activates the group machinery
    when `grouped_cols` is non-empty, and the SA can only propose role 4 for
    a candidate that already carries it. With ~40 role genes and 40
    iterations the chance of the random walk landing on a grouped structure
    is negligible, so grouped_terms came back [] in both 61 and 61b and the
    grouping axis was silently untested.

    Each arm is a MINIMAL scaffold (forced terms only): the control has the
    term Fixed, the treatment has it Grouped, everything else Excluded. The
    BIC delta between the two arms is therefore attributable to grouping
    alone. Role 4 costs log(n)*2 BIC, so a grouped arm only "wins" if the
    delta is better than that -- the verdict below states the break-even
    rather than just the sign.
    """
    if not _use_group_probe(args):
        return None
    elig = _grouped_eligible(evaluator)
    audit["grouped_eligible"] = list(elig)
    if not elig:
        audit["grouped_probe"] = {"attempted": True, "skipped":
                                  "no variable allows role 4"}
        print("   [grouped] probe skipped: no variable allows role 4")
        return None
    # SEG_GROUP_VAR (log_length) is the segment-identity carrier: it is
    # constant within a segment, so its grouped coefficient is a
    # per-segment propensity contrast. Probed first because that is the
    # question "group by segment" is really asking. Then the geometry
    # covariates, which answer the different question of whether e.g.
    # lanes_z should vary by segment.
    order = [v for v in (SEG_GROUP_VAR, "lanes_z", "aadt_z",
                         "maxspeed_z", "dist_inter_z") if v in elig]
    if str(group_col) != "segment":
        # Grouping is by highway/event, so the segment-identity carrier is
        # not the question being asked; keep it out of the report.
        order = [v for v in order if v != SEG_GROUP_VAR]
    order += [v for v in elig if v not in order]
    results = {}

    def _bic(decision):
        """Evaluate, mapping the 1e12 failure sentinel to None.

        A failed arm must never become a number. Returning 1e12 on both
        sides previously produced delta == 0.0 and a confident-looking
        'grouped loses' for every term, having fitted nothing.
        """
        raw = float(np.min(np.asarray(
            evaluator.fitness(decision), dtype=float)))
        return None if not np.isfinite(raw) or raw >= 1e11 else raw

    try:
        n_obs = int(getattr(getattr(evaluator, "df_train", None),
                            "shape", [0])[0]) or 0
        # Role 4 spends 2 parameters (group_mean + group_sd) vs role 1's one.
        breakeven = float(np.log(max(n_obs, 2)) * 2.0)
        audit["grouped_probe_breakeven"] = round(breakeven, 3)
        for term in order:
            for disp in (0, 1):
                control = _base_seed_decision(evaluator, dispersion=disp,
                                              term=term, term_role=1)
                treat = _base_seed_decision(evaluator, dispersion=disp,
                                            term=term, term_role=4)
                if control is None or treat is None:
                    return None
                b_base = _bic(control)
                b_grp = _bic(treat)
                rec = {"term": term, "dispersion": disp,
                       "base_bic": b_base, "grouped_bic": b_grp,
                       "breakeven_bic": round(breakeven, 3)}
                if b_base is None or b_grp is None:
                    rec.update({"ok": False,
                                "reason": "arm failed to fit (1e12 sentinel)",
                                "delta": None, "grouped_wins": None})
                else:
                    rec["ok"] = True
                    rec["delta"] = round(b_grp - b_base, 3)
                    rec["grouped_wins"] = bool(rec["delta"] < -breakeven)
                results[f"{term}|disp{disp}"] = rec
    except Exception as exc:  # noqa: BLE001
        audit["grouped_probe"] = {"attempted": True, "error": str(exc)[:200]}
        print(f"   [grouped] probe raised ({type(exc).__name__}: {exc}); "
              "continuing without it")
        return None
    audit["grouped_probe"] = {"attempted": True, "group_col": group_col,
                              "terms_probed": list(order), "results": results}
    n_bad = sum(1 for r in results.values() if not r.get("ok"))
    for r in results.values():
        marker = "  <-- segment identity" if r["term"] == SEG_GROUP_VAR else ""
        if not r.get("ok"):
            print(f"   [grouped] term={r['term']:<14} disp={r['dispersion']} "
                  f"INCONCLUSIVE: {r['reason']}{marker}")
            continue
        verdict = ("GROUPED WINS" if r["grouped_wins"]
                   else "grouped loses (over BIC penalty)")
        print(f"   [grouped] term={r['term']:<14} disp={r['dispersion']} "
              f"base={r['base_bic']:.1f} grouped={r['grouped_bic']:.1f} "
              f"delta={r['delta']:+.1f} (breakeven {-breakeven:.1f}) "
              f"-> {verdict}{marker}")
    if n_bad:
        print(f"   [grouped] WARNING: {n_bad}/{len(results)} probes were "
              "INCONCLUSIVE (an arm returned the failure sentinel). "
              "Treat those terms as untested, not as rejected.")
    return results


def _use_group_probe(args):
    return str(getattr(args, "group_probe", "on") or "on").lower() != "off"


def _extract_search_best(evaluator):
    """Best (score, spec, n_eval, decision) from the evaluator cache."""
    n_eval = 0
    try:
        items = list(getattr(evaluator, "cache", {}).items())
    except Exception:  # noqa: BLE001
        items = []
    n_eval = len(items)
    want = None
    try:
        want = len(getattr(evaluator, "vars", []) or [])
    except Exception:  # noqa: BLE001
        want = None
    candidates = []
    for key, val in items:
        if not isinstance(key, (tuple, list)):
            continue
        if want and len(key) not in (2 * want + 1, 3 * want + 2):
            continue
        try:
            score = float(np.min(np.asarray(val, dtype=float)))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(score) and score < 1e12:
            candidates.append((score, key))
    candidates.sort(key=lambda item: item[0])
    for score, key in candidates[:200]:
        # Genes are integer codes; decode from a rounded int vector so list
        # indexing inside build_spec never sees float role/dist codes.
        decision = np.rint(np.asarray(key, dtype=float)).astype(int)
        try:
            spec = evaluator.build_spec(decision)
        except Exception:  # noqa: BLE001 - unsupported role; try next best
            continue
        if spec is not None:
            return float(score), spec, n_eval, decision
    return float("nan"), None, n_eval, None


def _append_progress(row):
    header = not PROGRESS_CSV.exists()
    pd.DataFrame([row]).to_csv(PROGRESS_CSV, mode="a", header=header,
                               index=False)


def run_search(work, args, audit, out_dir, resume_state=None):
    from metacountregressor import ExperimentBuilder, ModelConstraints

    if args.search_vars == "slim":
        cont_list, cal_list, game_list = SLIM_CONT, SLIM_CAL, SLIM_GAME
    else:
        cont_list, cal_list, game_list = CONT_VARS, CAL_VARS, GAME_VARS
    # AADT candidates: external TMR join (aadt_z) if present, else probe
    # AADT proxy (log_flow_z) built from seg_flow exposure downstream.
    # Both are searched and kept only if BIC-significant.
    aadt_cands = [c for c in ("aadt_z", "log_flow_z") if c in work.columns]
    # SEG_GROUP_VAR is appended unconditionally: it is the segment identity
    # and must be allowed to take the grouped (role 4) effect. It is NOT
    # force_fixed -- forcing it would ban roles 2/3/4 and make grouping
    # impossible, which is exactly the bug that left grouped_terms empty.
    cand_cont = [c for c in cont_list + ["log_length"] + aadt_cands
                 + [SEG_GROUP_VAR] if c in work.columns]
    cand_cal = [c for c in cal_list if c in work.columns]
    cand_game = [c for c in game_list if c in work.columns]
    # Temporal shape vars are searchable fixed-only terms; --no-temporal-vars
    # drops them from the candidate pool (peak shape then comes only from
    # rel_hour/game dummies + the optional stage-2b AR).
    cand_temp = [c for c in TEMPORAL_VARS
                 if c in work.columns
                 and not getattr(args, "no_temporal_vars", False)]
    # Fixture-attribute + interaction block: always searchable (slim or
    # full) — this is what lets future fixtures differ by shape, not scale.
    cand_wide = [c for c in GAME_WIDE_VARS if c in work.columns]
    # Transferable-temporal candidates: deterministic transforms of history
    # and the event clock — fixed-only, never random parameters.
    cand_timex = [c for c in TIME_X_VARS if c in work.columns]
    hw = [c for c in work.columns if c.startswith("hw_")]
    # Exposure-shape terms (hinge curvature and/or logexp x highway). Searchable
    # but fixed-only: they describe the shape of the flow->speeding curve, not
    # per-segment taste, so a random coefficient on them would be unidentified
    # noise. BIC decides whether any of them earn their place.
    shape_cols = [c for c in work.columns
                  if c.startswith("logexp_h") or c.startswith("logexp_x_")]
    variables = (["logexp"] + hw + cand_cont + cand_cal + cand_game
                 + cand_temp + cand_wide + cand_timex + shape_cols)
    # Game-day dynamics are FORCED (never droppable): exposure + the game
    # block enter every spec. Temporal-shape vars stay searchable but
    # fixed-only (stable within-window curvature, no random taste variation).
    # Random parameters are restricted to continuous geometry/flow terms —
    # the old code re-allowed cal/game into allow_random, letting the search
    # waste draws on binary game dummies.
    force_game = [c for c in GAME_FORCE if c in work.columns] \
        if not args.no_force_game else []
    # Segment length is exposure scale (like logexp): counts must scale
    # with metres, otherwise 4 m driveways get the same baseline as
    # 400 m arterials (o25908597: 45 m osmid hogged 72% of predictions).
    force_len = ["log_length"] if "log_length" in work.columns else []
    # Role 6 (zero inflation) candidates, resolved BEFORE the constraint chain
    # because no_zi() is destructive and accumulates: it does
    # `allowed.discard(6)` per variable, so a later call with a SHORTER ban
    # list cannot restore the role. Calling it twice -- once for everything,
    # then again for the non-candidates -- silently left every variable
    # banned and made --zero-inflation on a no-op that looked like "ZI was
    # tried and did not help". So the ban list is computed once, here.
    zi_cands = ([c for c in cand_cont + cand_cal if c in work.columns]
                if str(getattr(args, "zero_inflation", "off")) == "on" else [])
    zi_banned = [c for c in variables if c not in zi_cands]
    constraints = (ModelConstraints()
                   .force_include("logexp", *force_len, *force_game)
                   .force_fixed("logexp", *force_len, *force_game)
                   .no_random("logexp", *force_len, *hw, *cand_cal,
                              *cand_game, *cand_temp, *cand_wide,
                              *cand_timex, *shape_cols)
                   .no_zi("logexp", *zi_banned))
    allow = [c for c in cand_cont if c not in hw]
    if allow:
        constraints = constraints.allow_random(*allow, distributions=["normal"])
    audit["forced_game_terms"] = list(force_game)
    # Grouped panel structure. --group-by event splits by fixture: each
    # event gets its own grouped coefficients (the per-event split), shrunk
    # toward means driven by heterogeneity vars (game attributes), so unseen
    # future events predict from attributes. Background rows form the "BG"
    # control group. Segment/highway groupings are place-effect variants.
    _group_col = str(args.group_by or "highway")
    if _group_col == "segment":
        _g = work["road_osmid"].astype(str) if "road_osmid" in work.columns else None
        _gid = "road_osmid" if "road_osmid" in work.columns else None
    elif _group_col == "event":
        _g = (work["fixture"].fillna("BG").astype(str)
              if "fixture" in work.columns else None)
        _gid = "fixture" if "fixture" in work.columns else None
    else:
        _group_col, _gid = "highway", "highway"
        _g = (work["highway"].astype("string").fillna("unclassified").astype(str)
              if "highway" in work.columns else None)
    _use_g = False
    if _g is not None:
        _vc = _g.value_counts()
        _use_g = bool(len(_vc) >= 2 and int((_vc >= 20).sum()) >= 2)
    # SEG_GROUP_VAR is log_length: constant within a segment, so its grouped
    # (role 4) coefficient is a per-segment propensity contrast. It is
    # already a forced fixed term and needs no candidate-pool change; the
    # probe simply skips it when the grouping is not by segment, because a
    # highway-level length coefficient is a re-parameterised length effect
    # rather than a segment propensity.
    audit["seg_group_term"] = (
        SEG_GROUP_VAR if _group_col == "segment"
        else f"{SEG_GROUP_VAR} (probe skipped: --group-by {_group_col} is "
             "not a segment grouping)")
    if _use_g and _gid is not None:
        builder = ExperimentBuilder(work, id_col="row_id", y_col="Y",
                                    group_id_col=_gid)
        roles = [0, 1, 2, 3, 4, 5]
    else:
        builder = ExperimentBuilder(work, id_col="row_id", y_col="Y")
        roles = [0, 1, 2, 3, 5]
    # Role 6 is offered to the search only when ZI was requested. The package
    # derives zero_inflated from len(zi_cols) > 0, so with no candidate
    # actually assigned the fitted model stays an ordinary count model.
    if zi_cands:
        roles = roles + [6]
    audit["zi_requested"] = (str(getattr(args, "zero_inflation", "off"))
                             == "on")
    audit["zi_candidates"] = list(zi_cands)
    if zi_cands:
        print(f"   zero inflation: role 6 offered for {len(zi_cands)} "
              f"candidate terms")
    audit["group_by"] = _group_col if _use_g else "none"
    audit["grouped_gse"] = bool(_use_g)
    audit["default_roles"] = list(roles)
    evaluator, engine_used = _build_evaluator(
        builder, variables, constraints, roles, int(args.search_R),
        str(args.engine), max_latent_classes=int(args.max_latent_classes))
    audit["engine_used"] = engine_used
    # Fail loudly if ZI was requested but is not actually reachable. The
    # failure mode is silent: role 6 missing from every allowed set means the
    # search simply never proposes a zero-inflated spec, and the run looks
    # exactly like "ZI was tried and BIC rejected it".
    zi_reachable = []
    try:
        _ar = getattr(evaluator, "allowed_roles", {}) or {}
        zi_reachable = [v for v in zi_cands if 6 in (_ar.get(v) or [])]
    except Exception:  # noqa: BLE001
        pass
    audit["zi_reachable"] = list(zi_reachable)
    if audit["zi_requested"] and not zi_reachable:
        print("   [zi] *** REQUESTED BUT NOT REACHABLE: no candidate can take "
              "role 6, so this run is NOT a zero-inflation test ***")
    elif zi_reachable:
        print(f"   [zi] role 6 reachable for {len(zi_reachable)}/{len(zi_cands)} "
              f"candidates")
    audit["n_variables"] = len(variables)
    # Probe the grouped axis BEFORE the SA loop. The search cannot reach a
    # role-4 structure on its own (see _probe_grouped), and without this the
    # grouping decision is made by default rather than by evidence.
    _probe_grouped(evaluator, _group_col if _use_g else None, audit, args)
    search_dir = out_dir / "search59"
    search_dir.mkdir(parents=True, exist_ok=True)
    try:
        from metacountregressor import SearchOutputConfig
        out_cfg = SearchOutputConfig(
            output_dir=str(search_dir), experiment_name="game_day",
            search_description="15-min speeding structure search (BIC)",
            checkpoint_every=int(args.checkpoint_every))
    except Exception:
        out_cfg = None

    total_iters = int(args.search_iters)
    chunk = max(1, int(args.search_chunk))
    done = int((resume_state or {}).get("iters_completed", 0) or 0)
    best_bic = (resume_state or {}).get("best_bic")
    best_spec = (resume_state or {}).get("best_spec")
    best_decision = (resume_state or {}).get("best_decision")
    if best_decision is not None:
        best_decision = np.asarray(best_decision, dtype=float)
    if best_spec is not None and not isinstance(best_spec, dict):
        best_spec = None

    t0 = time.time()
    search_error = None
    chunk_failures = 0
    cwd = os.getcwd()
    try:
        os.chdir(search_dir)
        while done < total_iters:
            elapsed_total = time.time() - t0
            if args.wall_budget and elapsed_total >= float(args.wall_budget):
                print(f"   [search] wall budget reached "
                      f"({args.wall_budget:.0f}s); stopping at "
                      f"{done}/{total_iters} iters (resubmit --resume to "
                      "continue)")
                break
            n = min(chunk, total_iters - done)
            warm = best_decision is not None
            n_starts = int(args.n_starts) if not warm else 1
            # Bound the chunk wall time: --chunk-time bounds a whole chunk,
            # the wall budget bounds the search stage, and SA's max_time is
            # per restart -- so divide by the number of restarts.
            chunk_budget = float(args.chunk_time) if args.chunk_time else None
            if args.wall_budget:
                remaining = float(args.wall_budget) - elapsed_total
                chunk_budget = (remaining if chunk_budget is None
                                else min(chunk_budget, remaining))
            max_time = (chunk_budget / n_starts
                        if chunk_budget and chunk_budget > 0 else None)
            run_kwargs = dict(
                algo=args.search_algo,
                max_iter=int(n),
                n_starts=n_starts,
                seed=int(args.seed),
                n_jobs=1,
                output_config=out_cfg,
                max_time=max_time,
                refit=False,
                warm_start_cache=True,
            )
            if warm:
                run_kwargs["init_solutions"] = [best_decision]
            print(f"   [search] chunk {done // chunk + 1}: iters "
                  f"{done + 1}..{done + n} of {total_iters} "
                  f"(starts={run_kwargs['n_starts']}, warm={warm})")
            error = None
            result = None
            try:
                result = builder.run(evaluator, **run_kwargs)
            except Exception as exc:  # noqa: BLE001 - keep the best cached spec
                error = f"{type(exc).__name__}: {exc}"
                print(f"   WARN: chunk raised ({error}); keeping best "
                      "cached spec")
                traceback.print_exc()
            score, spec, n_eval, decision = _extract_search_best(evaluator)
            if np.isfinite(score) and (best_bic is None
                                       or score < float(best_bic)):
                best_bic = float(score)
                best_spec = spec
                best_decision = decision
            search_error = error or search_error
            # Count the generations that actually ran: a chunk stopped by
            # --chunk-time (max_time) completes fewer than its nominal n, and
            # the shortfall is picked up by the next chunk.
            ran = n
            if result is not None:
                try:
                    gens = sum(len(s) for s in (result.get("search_stats") or [])
                               if s is not None)
                    if gens > 0:
                        ran = max(1, min(int(n), int(gens)))
                except Exception:  # noqa: BLE001 - non-list stats
                    ran = n
            elif n_eval == 0:
                ran = 0
            done += int(ran)
            _write_json_atomic(BEST_JSON, {
                "best_spec": best_spec, "best_bic": best_bic,
                "best_decision": None if best_decision is None
                else np.asarray(best_decision, dtype=float).tolist(),
                "algo": args.search_algo, "iters": total_iters,
                "iters_completed": int(done), "chunk_size": int(chunk),
                "R": int(args.search_R), "seed": int(args.seed),
                "target": args.target, "search_gen": SEARCH_GEN,
                "engine": engine_used, "variables": variables,
                "search_error": search_error,
                "updated_iso": _utcnow_iso()})
            _append_progress({
                "chunk": int(done // chunk), "iters_completed": int(done),
                "total_iters": int(total_iters),
                "elapsed_s": round(time.time() - t0, 1),
                "best_bic": None if best_bic is None
                else round(float(best_bic), 2),
                "n_cache_evals": int(n_eval),
                "timestamp_iso": _utcnow_iso()})
            if error and n_eval == 0:
                chunk_failures += 1
                if chunk_failures >= 2:
                    print("   [search] two consecutive empty/failed chunks;"
                          " stopping")
                    break
            else:
                chunk_failures = 0
            _clear_jax_caches()
    finally:
        os.chdir(cwd)

    complete = done >= total_iters
    # Rejection diagnostics. Every 1e12 the search saw used to be anonymous,
    # so a structural axis that failed silently looked identical to one that
    # was never tried. Report the breakdown.
    try:
        _rej = evaluator.reject_summary()
    except Exception:  # noqa: BLE001
        _rej = {"counts": {}, "recent": []}
    audit["search"] = {
        "resumed": bool(resume_state),
        "best_bic": best_bic,
        "done_iters": int(done),
        "total_iters": int(total_iters),
        "complete": bool(complete),
        "engine": engine_used,
        "variables": variables,
        "search_error": search_error,
        "rejections": _rej,
    }
    _tot_rej = sum(int(v.get("count", 0)) for v in _rej.get("counts", {}).values())
    if _tot_rej:
        print(f"   [search] rejected structures: {_tot_rej} total")
        for _why, _v in sorted(_rej.get("counts", {}).items(),
                               key=lambda kv: -int(kv[1].get("count", 0))):
            _d = str(_v.get("detail", ""))[:90]
            print(f"      {int(_v.get('count', 0)):>6}  {_why}"
                  + (f"  -- {_d}" if _d else ""))
    if best_spec is not None:
        print(f"   [search] best_BIC={best_bic:.1f} "
              f"iters={done}/{total_iters} complete={complete}")
    else:
        print("   WARN: search produced no spec; using fallback")
    return best_spec, best_bic, variables, engine_used, done, complete


DEFAULT_FIXED = ["logexp", "in_event", "rel_hour", "log_att", "dist_stad",
                 "event_x_dist", "rel_hour_sq", "rel_sin", "rel_cos",
                 "clock_prof", "seg_risk_z", "exp_lag1",
                 "tod_sin", "tod_cos", "weekend", "lanes_z", "maxspeed_z"]


def ensure_spec(spec, work, audit, force_game=True, keep_temporal=True):
    """Guarantee a usable spec: exposure + game-day block always in.

    The SA search on sparse 15-min counts can select an all-excluded solution
    (and evaluator.build_spec does not re-apply force_include). Exposure is
    forced in; the GAME_FORCE block (in_event/rel_hour/log_att/event_x_dist)
    is re-attached unless --no-force-game was passed, so a game-day plan can
    never be driven by a static-only model again. Near-empty specs fall back
    to DEFAULT_FIXED (which now carries the game + temporal block).
    With keep_temporal=False (--no-temporal-vars), temporal-shape terms are
    stripped so the peak can be compared with/without them.
    """
    if spec is None:
        spec = {}
    fixed = [c for c in spec.get("fixed_terms", []) if c in work.columns]
    if not keep_temporal:
        fixed = [c for c in fixed if c not in TEMPORAL_VARS]
    if "logexp" not in fixed:
        fixed = ["logexp"] + fixed
    missing_game = []
    if force_game:
        for c in GAME_FORCE:
            if c in work.columns and c not in fixed:
                fixed.append(c)
                missing_game.append(c)
    audit["game_terms_patched"] = list(missing_game)
    if len(fixed) <= 1:
        fallback = [c for c in DEFAULT_FIXED
                    if c in work.columns
                    and (keep_temporal or c not in TEMPORAL_VARS)]
        audit["spec_fallback"] = True
        fixed = fallback
    else:
        audit["spec_fallback"] = False
    spec = dict(spec)
    spec["fixed_terms"] = fixed
    spec["rdm_terms"] = [t for t in spec.get("rdm_terms", [])
                         if str(t).split(":")[0] in work.columns]
    return spec


def _rss_gb(tag=""):
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1024**3
        print(f"   [mem] {tag}: RSS={rss:.2f} GB")
        return rss
    except Exception:
        return float("nan")


def _spec_fields(spec, work):
    """Full searched structure filtered to columns present in work.

    Previously only fixed+rdm_terms were passed, silently dropping the
    searched correlated/grouped/heterogeneity structure (and refitting a
    fixed-only model that then diverged).
    """
    from metacount_nb import _norm_spec_terms
    fixed = [c for c in spec.get("fixed_terms", [])
             if c.split(":")[0] in work.columns]
    fields = {
        "fixed_terms": fixed,
        "rdm_terms": _norm_spec_terms(spec.get("rdm_terms"), work),
        "rdm_cor_terms": _norm_spec_terms(spec.get("rdm_cor_terms"), work),
        "grouped_terms": _norm_spec_terms(spec.get("grouped_terms"), work),
        "hetro_in_means": [str(t).split(":")[0]
                           for t in (spec.get("hetro_in_means") or [])
                           if str(t).split(":")[0] in work.columns],
        "hetro_in_variances": [str(t).split(":")[0]
                               for t in (spec.get("hetro_in_variances") or [])
                               if str(t).split(":")[0] in work.columns],
        "group_id_col": spec.get("group_id_col") or "highway",
    }
    if "highway" not in work.columns:
        fields["grouped_terms"] = []
        fields["group_id_col"] = None
    return fields


def refit_only(work, spec, args, audit):
    """JAX refit on the 100k-row search frame only (no future data alive).

    Panel/future frames must already be freed or reduced to light keys
    before calling — peak is then work + draws + eta, not panel + work.
    Carries the FULL searched spec (correlated/grouped/hetero); staged
    fallback inside fit_metacount_nb guarantees a sane accepted fit.

    The likelihood follows the SEARCH rather than being hardcoded. The
    structural search scores each candidate with a real fit and the
    dispersion gene picks Poisson (0) vs NB2 (1) on BIC, so refitting a
    forced "nb" ships a model the search never selected and reports a BIC
    that does not describe it. --refit-model overrides (nb|poisson); the
    default "searched" honours spec["dispersion"].
    """
    fields = _spec_fields(spec, work)
    _mode = str(getattr(args, "refit_model", "searched") or "searched").lower()
    if _mode in ("searched", "auto", ""):
        # build_spec emits dispersion 0/1 from the search's dispersion gene.
        _mode = "nb" if int(spec.get("dispersion", 1) or 0) == 1 else "poisson"
    if _mode not in ("nb", "poisson"):
        raise SystemExit(f"--refit-model must be searched|nb|poisson, got {_mode!r}")
    # Exposure parameterisation. 'offset' pins the exposure coefficient at 1
    # (the hazard/rate form): logexp moves from a free fixed effect into the
    # likelihood's fixed offset. The two are nested, so this is a genuine
    # testable restriction rather than a different model -- see the LRT below.
    _exp = str(getattr(args, "exposure", "free") or "free").lower()
    if _exp not in ("free", "offset", "hazard"):
        raise SystemExit(f"--exposure must be free|offset, got {_exp!r}")
    _offset_col = None
    if _exp in ("offset", "hazard"):
        if "logexp" not in work.columns:
            raise SystemExit("--exposure offset needs a 'logexp' column "
                             "(log-scale exposure) in the search frame")
        _offset_col = "logexp"
    audit["exposure_mode"] = _exp
    audit["refit_model_requested"] = _mode
    audit["refit_dispersion_gene"] = int(spec.get("dispersion", -1))
    print(f"   refit likelihood: {_mode} "
          f"(search dispersion gene={spec.get('dispersion')}); "
          f"exposure: {_exp}"
          + (f" pinned at coef 1.0 via offset[{_offset_col}]"
             if _offset_col else " free coefficient"))
    _rss_gb("refit-start")
    info = fit_metacount_nb(work, y_col="Y", id_col="row_id",
                            model=_mode, R=int(args.refit_R),
                            offset_col=_offset_col,
                            de_maxiter=int(args.de_maxiter),
                            de_popsize=int(args.de_popsize),
                            verbose=True,
                            continuous_de_warm_start=(not args.no_de_warm),
                            use_slsqp=(not args.no_slsqp), **fields)
    if str(info.get("spec_label", "")).startswith("statsmodels-"):
        # JAX stages all diverged (typical: LBFGS-only from a cold init on
        # sparse 15-min counts stalls on the flat eta plateau). One
        # warm-started retry preserves the searched random structure when
        # the cold start was the problem; the GLM fallback above stays as
        # the safety net. R capped: 100k x 60 draws fits the 96GB node.
        print("  [metacount] JAX refit diverged; retrying once warm-started "
              "(DE + SLSQP) to preserve searched structure")
        try:
            warm = fit_metacount_nb(
                work, y_col="Y", id_col="row_id",
                model=_mode, R=min(int(args.refit_R), 60),
                offset_col=_offset_col,
                de_maxiter=int(args.de_maxiter),
                de_popsize=int(args.de_popsize),
                verbose=True, continuous_de_warm_start=True,
                use_slsqp=True, seed=int(args.seed), **fields)
            if not str(warm.get("spec_label", "")).startswith("statsmodels-"):
                info = warm
                print(f"  [metacount] warm retry accepted "
                      f"({warm.get('spec_label')})")
            else:
                print("  [metacount] warm retry also fell back; keeping "
                      "GLM fallback")
        except RuntimeError as exc:
            print(f"  [metacount] warm retry failed ({exc}); keeping "
                  "GLM fallback")
    _rss_gb("refit-done")
    audit["refit"] = {"spec_label": info["spec_label"],
                      "searched_spec": {k: spec.get(k) for k in
                                        ("fixed_terms", "rdm_terms",
                                         "rdm_cor_terms", "grouped_terms",
                                         "hetro_in_means")},
                      "fitted": {k: info.get(k) for k in
                                 ("fixed_terms", "rdm_terms",
                                  "rdm_cor_terms", "grouped_terms")},
                      "aic": info["aic"], "bic": info["bic"],
                      "alpha": info["alpha"],
                      # Likelihood actually fitted, so the reported BIC is
                      # attributable. alpha is None for Poisson (no dispersion
                      # parameter) rather than 0.0, which read as "degenerate".
                      "model": info.get("model", _mode),
                      "refit_model": _mode,
                      "exposure_mode": info.get("exposure_mode", _exp),
                      "exposure_coef": info.get("exposure_coef")}
    _clear_jax_caches()
    return info


def refit_binomial(work, args, audit, spec=None, label="binomial refit"):
    """Fit the speeding propensity as a BINOMIAL and report it beside the
    count refit.

        k_i ~ Binomial(n_i, sigmoid(x_i'b)),  n_i = trips_with_judged_speed

    Why this is not the same as the count model. The count refit needs an
    EXPOSURE to scale a count with, and the only one available is a
    TMR-derived flow projection -- which is why the exposure LRT keeps
    rejecting it. Here the denominator is MEASURED, so there is no
    projection to get wrong, no offset to pin, and no chi2 to explain: the
    elasticity of speeding to flow is just another coefficient, estimated
    by the same Newton step as everything else.

    The design is the searched fixed-term set, so this is a like-for-like
    refit of the SAME structure under a different likelihood, not a new
    search. It does not replace the count path: it is reported alongside
    it so the two can be compared on the same holdout.
    """
    audit = audit if audit is not None else {}
    out = {"label": label, "available": False}
    # Resolve the denominator from the unit of the response. For speeding_trips
    # that is vehicle_count, not trips_with_judged_speed; using the trip count
    # against vehicle successes allowed k > n and the exact likelihood refused
    # the data with "successes outside [0, n]" (1,491 offending rows on the
    # real panel).
    #
    # Preference order: the target's own mapping, then whichever candidate is
    # actually present in the frame. The second step matters for frames built
    # without a target (synthetic and legacy callers): assuming the global
    # default there silently switched them to a column they do not have and
    # reported "unavailable" instead of fitting.
    _target = getattr(args, "target", None)
    _cands = []
    if _target:
        _cands.append(binomial_denominator(_target))
    _cands.append(binomial_denominator(TARGET_DEFAULT))
    _cands.append(BINOMIAL_DENOM)
    _seen, _ordered = set(), []
    for _c in _cands:
        if _c not in _seen:
            _seen.add(_c)
            _ordered.append(_c)
    denom = next((c for c in _ordered if c in work.columns), None)
    out["denominator_candidates"] = _ordered
    if denom is None:
        out["reason"] = ("no usable denominator in the frame; wanted one of "
                         + ", ".join(_ordered))
        audit["binomial"] = out
        print(f"   [binom] unavailable: {out['reason']}")
        return None, out
    out["denominator"] = denom

    k = pd.to_numeric(work["Y"], errors="coerce")
    n = pd.to_numeric(work[denom], errors="coerce")
    ok = (k.notna() & n.notna() & np.isfinite(k.to_numpy(float))
          & np.isfinite(n.to_numpy(float)))
    ok &= n.to_numpy(float) > 0
    k, n = k[ok], n[ok]
    if len(k) < 100:
        out["reason"] = f"only {len(k)} rows with a positive denominator"
        audit["binomial"] = out
        print(f"   [binom] unavailable: {out['reason']}")
        return None, out

    # Design: prefer the searched fixed terms, fall back to the forced
    # exposure block. Kept small and fixed -- the point is a like-for-like
    # refit of the chosen structure, not a second search.
    cands = []
    for c in ["logexp", "in_event", "rel_hour", "log_att", "dist_stad",
              SEG_GROUP_VAR]:
        if c in work.columns:
            cands.append(c)
    cands += [c for c in ("lanes", "lanes_known", "aadt_z", "log_flow_z")
              if c in work.columns and c not in cands]
    if not cands:
        out["reason"] = "no candidate design columns present"
        audit["binomial"] = out
        print(f"   [binom] unavailable: {out['reason']}")
        return None, out

    Xw = work.loc[ok, cands].apply(pd.to_numeric, errors="coerce")
    Xw = Xw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # NO standardisation. An earlier version standardised every column to
    # guard against spurious separation flags, which forced a separate
    # magnitude test. That was a mistake: standardising rescales the
    # coefficients (coef_std = coef_raw * sd), and the one number this whole
    # comparison exists to produce is the flow elasticity on logexp, which
    # is only interpretable -- and only comparable with the count model's
    # pinned 1.0 -- on the RAW scale. Standardising reported 0.445 where the
    # truth was 0.55 purely because sd(logexp) = 0.8. Separation is now
    # detected scale-free (perfect-prediction test), so no rescaling is
    # needed to keep the fit honest.
    Xs = Xw.to_numpy(float)

    from binomial_mle import fit_binomial_logit
    try:
        r = fit_binomial_logit(
            k.to_numpy(float), n.to_numpy(float), Xs, names=cands)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"{type(exc).__name__}: {exc}"
        audit["binomial"] = out
        print(f"   [binom] fit failed: {out['reason']}")
        return None, out

    out.update({
        "available": True,
        "loglik": r["loglik"], "aic": r["aic"], "bic": r["bic"],
        "n_obs": r["n_obs"], "n_trials": r["n_trials"],
        "n_successes": r["n_successes"],
        "mean_rate": r["n_successes"] / max(r["n_trials"], 1.0),
        "converged": r["converged"], "separated": r["separated"],
        "dropped": r["dropped"], "n_params": r["design_columns"],
        "terms": {c: [float(b), float(s)] for c, b, s
                  in zip(r["names"], r["coef"], r["se"])},
    })
    audit["binomial"] = out
    out["n_trials_total"] = int(r["n_trials"])
    print(f"   [binom] k~Binomial(n={out['mean_rate']*r['n_trials']:.0f} "
          f"successes / {r['n_trials']:.0f} {denom}, "
          f"rate={out['mean_rate']:.4f})")
    print(f"   [binom] loglik={r['loglik']:.2f}  AIC={r['aic']:.2f}  "
          f"BIC={r['bic']:.2f}  params={r['design_columns']}")
    if r["separated"]:
        print("   [binom] *** SEPARATED: the MLE does not exist (a coefficient"
              " runs to infinity). Coefficients below are NOT interpretable.")
    if r["dropped"]:
        print(f"   [binom] dropped degenerate columns: {r['dropped']}")
    for c, b, s in zip(r["names"], r["coef"], r["se"]):
        print(f"           {c:<18} {b:+9.4f}  (se {s:.4f})")
    # The headline comparison: in the count model the exposure elasticity is
    # pinned at 1 by the offset and the LRT fights it. Here it is free.
    if "logexp" in r["names"]:
        i = r["names"].index("logexp")
        out["logexp_coef"] = float(r["coef"][i])
        out["logexp_se"] = float(r["se"][i])
        print(f"   [binom] flow elasticity (logexp) = {r['coef'][i]:.4f} "
              f"+/- {r['se'][i]:.4f}   (count model pins this at 1.0 by"
              " construction)")
    return r, out


def plot_pred_vs_actual(out, merged, fig_dir, out_dir, top_n=20):
    """PROOF OF CONCEPT: does the predicted ranking hold up against ACTUAL
    observed speeding for the same games?

    The surfaces figure shows what the model predicted. This shows whether
    the prediction was any good, which is the only thing that justifies
    deploying on it. Two views per game:

      * predicted vs observed per top-N segment, with a 1:1 line
      * the share of OBSERVED speeding falling in each quartile of the
        PREDICTED ranking -- a model that ranks well puts most of the real
        events in its top quartile

    Honest framing, stated in the caption: this is a top-N comparison, not
    a network-wide coverage claim. The deployment figure is the share of
    observed speeding on the staffed sites, reported separately by stage 60.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"   WARN: pred-vs-actual figure unavailable ({exc})")
        return None

    if merged is None or not len(merged):
        print("   WARN: no observed rows; skipping pred-vs-actual figure")
        return None

    _fx = list(out["fixture"].unique())
    ncol = min(2, max(len(_fx), 1))
    nrow = int(np.ceil(len(_fx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 5.2 * nrow),
                             squeeze=False)
    rows = []
    for i, fx in enumerate(_fx):
        ax = axes[i // ncol][i % ncol]
        _p = (out[out["fixture"] == fx]
              .groupby("road_osmid")["pred_speeding"].sum()
              .sort_values(ascending=False).head(int(top_n)))
        try:
            _o = (merged[merged["fixture"] == fx]
                  .groupby("road_osmid")["observed"].sum())
        except Exception:  # noqa: BLE001
            _o = pd.Series(dtype=float)
        _j = pd.concat([_p.rename("pred"), _o.rename("obs")], axis=1)
        _j = _j.dropna()
        if len(_j) < 4:
            ax.axis("off")
            continue
        r = float(np.corrcoef(_j["pred"], _j["obs"])[0, 1])
        ax.scatter(_j["obs"], _j["pred"], s=26, alpha=0.75,
                   color="#0B6E4F", edgecolors="k", linewidths=0.4)
        _mx = float(max(_j["obs"].max(), _j["pred"].max())) * 1.08
        ax.plot([0, _mx], [0, _mx], ls="--", lw=1.0, color="#888888",
                label="1:1")
        ax.set_xlim(0, _mx)
        ax.set_ylim(0, _mx)
        ax.set_xlabel("OBSERVED speeding (actual)")
        ax.set_ylabel("PREDICTED speeding")
        # share of observed by quartile of the PREDICTED ranking
        _s = _j.sort_values("pred", ascending=False)
        _nq = min(4, max(len(_s) // 2, 2))
        _q = np.array_split(_s["obs"].to_numpy(), _nq)
        _sh = [float(x.sum()) for x in _q]
        _tot = max(sum(_sh), 1e-9)
        _pct = [100.0 * x / _tot for x in _sh]
        _ax2 = ax.inset_axes([0.63, 0.63, 0.34, 0.33])
        _ax2.bar(range(_nq), _pct, color="#0B6E4F", width=0.7)
        _ax2.set_xticks(range(_nq))
        _ax2.set_xticklabels([f"Q{i + 1}" for i in range(_nq)], fontsize=5)
        _ax2.tick_params(labelsize=5)
        _ax2.set_title("obs %% by pred rank", fontsize=5.5)
        for _k, _v in enumerate(_pct):
            _ax2.text(_k, _v, f"{_v:.0f}", ha="center", va="bottom",
                      fontsize=5)
        ax.set_title(f"{str(fx)[:44]}\nr={r:.2f}  n={len(_j)}  "
                     f"top quartile holds {_pct[0]:.0f}% of observed",
                     fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        rows.append({"fixture": str(fx), "n_segments": int(len(_j)),
                     "pearson_r": round(r, 4),
                     "top_quartile_obs_share": round(_pct[0] / 100.0, 4),
                     "obs_total": round(float(_j["obs"].sum()), 2),
                     "pred_total": round(float(_j["pred"].sum()), 2)})
    for j in range(len(_fx), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        "PROOF OF CONCEPT: predicted ranking vs ACTUAL observed speeding, "
        "per future game\n(top-%d segments by prediction; inset = share of "
        "observed speeding in each predicted-rank quartile.\n"
        "A ranking that works puts most real events in Q1. Top-N only -- not "
        "a network-wide coverage claim.)" % int(top_n), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _p_out = Path(fig_dir) / "game_day_pred_vs_actual.png"
    fig.savefig(_p_out, dpi=130)
    plt.close(fig)
    try:
        pd.DataFrame(rows).to_csv(
            Path(out_dir) / "game_day_pred_vs_actual.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"   WARN: pred-vs-actual table failed ({exc})")
    for rr in rows:
        print(f"   [value] {str(rr['fixture'])[:34]:34s} r={rr['pearson_r']:.2f}"
              f"  Q1 holds {100 * rr['top_quartile_obs_share']:.0f}% of observed"
              f"  obs_total={rr['obs_total']:.0f}")
    print(f"   pred-vs-actual -> {_p_out.name}, "
          f"game_day_pred_vs_actual.csv")
    return rows


def report_exposure_coefficient(info, audit, label="shipped refit"):
    """Print the free exposure coefficient with its uncertainty.

    WHY: the exposure LRT only says the pinned offset (coefficient 1) is
    rejected. On a panel of ~1e5 rows the SE on that coefficient is around
    2e-3, so a large chi2 can correspond to a deviation of only a few per
    cent -- statistically decisive, practically negligible. The decision to
    keep or abandon the calibrated exposure has to be made on the MAGNITUDE
    of the deviation, not on the chi2, so the number belongs in the log
    and the audit.

    In offset/hazard mode logexp is moved into the likelihood's offset and
    dropped from the design, so there is no coefficient to report; that is
    stated explicitly rather than silently skipped.
    """
    audit = audit if audit is not None else {}
    out = {"label": label, "available": False}
    try:
        coefs = info.get("coef") if isinstance(info, dict) else None
    except Exception:  # noqa: BLE001
        coefs = None
    if coefs is None or len(coefs) == 0 or "variable" not in coefs.columns:
        out["reason"] = "no coefficient table returned"
        audit["exposure_coef_report"] = out
        print(f"   [exposure] coefficient unavailable ({out['reason']})")
        return out

    row = coefs[coefs["variable"].astype(str).str.strip().str.lower()
                == "logexp"]
    if row.empty:
        cand = [v for v in coefs["variable"].astype(str)
                if "logexp" in v.lower()]
        out["reason"] = ("logexp absent from the design (expected when the "
                         "exposure is pinned as an offset)")
        out["candidates"] = cand
        audit["exposure_coef_report"] = out
        print("   [exposure] no free logexp coefficient (pinned offset mode"
              + (f"; near matches: {cand}" if cand else "") + ")")
        return out

    r = row.iloc[0]
    beta = float(r.get("coef_mle", r.get("coefficient", np.nan)))
    try:
        se = float(r.get("std_err", np.nan))
    except (TypeError, ValueError):
        se = np.nan
    out.update({"available": True, "beta": beta, "std_err": se})
    if not np.isfinite(beta):
        out["available"] = False
        out["reason"] = "coefficient is not finite"
        out["beta_raw"] = beta
        audit["exposure_coef_report"] = out
        print("   [exposure] logexp coefficient is not finite "
              f"({beta!r}) -- no magnitude verdict")
        return out

    print(f"   [exposure] free logexp coefficient ({label}):")
    print(f"       beta_hat = {beta:.5f}   SE = {se:.5f}")
    if np.isfinite(se):
        lo, hi = beta - 1.96 * se, beta + 1.96 * se
        out["ci95"] = [float(lo), float(hi)]
        print(f"       95% CI = [{lo:.5f}, {hi:.5f}]")
    if np.isfinite(se) and se > 0:
        z1 = (beta - 1.0) / se
        dev = abs(beta - 1.0)
        out["z_vs_1"] = float(z1)
        out["abs_deviation"] = float(dev)
        out["rel_deviation_pct"] = float(100.0 * dev)
        # Compare against a lever already in play: the per-class game-day
        # uplift alone spans roughly 20%, and the exposure calibration
        # rescales by the measured Y/E ratio.
        scale = 0.20
        out["deviation_vs_uplift_span"] = float(dev / scale)
        verdict = ("PRACTICALLY NEGLIGIBLE" if dev < 0.05 else
                   "SMALL" if dev < 0.10 else
                   "MATERIAL" if dev < 0.25 else "LARGE")
        out["verdict"] = verdict
        print(f"       deviation from 1.0 = {dev:.5f} "
              f"({100.0 * dev:.2f}%), z vs 1 = {z1:+.1f}")
        print(f"       as a fraction of the per-class uplift span (~20%): "
              f"{100.0 * dev / scale:.1f}%  -> {verdict}")
        print("       Decide on this magnitude, not on the LRT chi2: a large"
              " chi2 on a big panel can still be a per-cent effect.")
    audit["exposure_coef_report"] = out
    return out


def _exposure_lrt(work, spec, args, audit, refit_info):
    """Likelihood-ratio test: free exposure coefficient vs pinned at 1.

    The two parameterisations are NESTED -- pinning the coefficient at 1
    is a one-parameter restriction of estimating it -- so the comparison
    is a legitimate LRT and not a BIC race between incomparable models.

    It answers a question the pipeline currently cannot: is the exposure
    channel (seg_flow * bin_factor * uplift) correctly specified? A free
    coefficient far from 1 means the projected flow is misspecified in a
    way a free coefficient was silently absorbing. That is the real
    argument for the hazard form -- not that it is a different model, but
    that it stops hiding flow-projection error.
    """
    if str(getattr(args, "exposure_lrt", "off") or "off").lower() != "on":
        return None
    want = str(getattr(args, "exposure", "free") or "free").lower()
    # Fit the OPPOSITE arm from whatever was shipped.
    other = "free" if want in ("offset", "hazard") else "offset"
    try:
        from metacount_nb import fit_metacount_nb
        fields = _spec_fields(spec, work)
        _mode = refit_info.get("model", "nb")
        print(f"   [exposure] LRT: fitting the {other} arm for comparison...")
        alt = fit_metacount_nb(
            work, y_col="Y", id_col="row_id", model=_mode,
            R=min(int(args.refit_R), 60),
            offset_col=("logexp" if other == "offset" else None),
            de_maxiter=int(args.de_maxiter), de_popsize=int(args.de_popsize),
            verbose=False, continuous_de_warm_start=True, use_slsqp=True,
            seed=int(args.seed), **fields)
    except Exception as exc:  # noqa: BLE001
        audit["exposure_lrt"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"   [exposure] LRT arm failed ({type(exc).__name__}: {exc})")
        return None
    ll_free = float(refit_info["loglik"]) if other == "offset" \
        else float(alt["loglik"])
    ll_off = float(alt["loglik"]) if other == "offset" \
        else float(refit_info["loglik"])
    # 2*loglik difference for 1 restriction; chi2_1 95% critical = 3.84.
    lr = 2.0 * (ll_free - ll_off)
    p = float(np.exp(-0.5 * max(lr, 0.0))) if lr > 0 else 1.0
    verdict = ("free coefficient is significantly better "
               "(exposure channel looks misspecified)"
               if lr > 3.84 else
               "no evidence against pinning the exposure coefficient at 1")
    coef = alt.get("exposure_coef")
    res = {"shipped": want, "compared": other,
           "loglik_free": ll_free, "loglik_offset": ll_off,
           "lr_stat": round(lr, 4), "p_value": round(p, 6),
           "df": 1, "critical_95": 3.84, "verdict": verdict,
           "free_exposure_coef": (None if coef is None
                                  else round(float(coef), 5)),
           "bic_free": float(alt["bic"]) if other == "free" else None,
           "bic_offset": float(refit_info["bic"]) if want != "free" else None}
    audit["exposure_lrt"] = res
    print(f"   [exposure] LRT chi2(1)={lr:+.3f} (95% crit 3.84) -> {verdict}")
    if coef is not None and np.isfinite(coef):
        print(f"   [exposure] free exposure coefficient = {float(coef):.4f} "
              "(1.0 = exposure channel consistent with the count model)")
    # The free arm is the SHIPPED refit whenever --exposure free (the
    # default), in which case `alt` is the pinned arm and carries no
    # coefficient -- so resolve the free arm explicitly. Without this the
    # coefficient is never printed in the default configuration, which is
    # how a chi2 of 364 was able to travel without its magnitude ever being
    # looked at.
    free_info = refit_info if other == "offset" else alt
    try:
        rep = report_exposure_coefficient(
            free_info, audit, label=f"free arm ({'shipped' if other == 'offset' else 'LRT'})")
    except Exception as exc:  # noqa: BLE001
        rep = None
        print(f"   WARN: LRT coefficient report failed ({exc})")
    if rep and rep.get("available"):
        dev = float(rep.get("abs_deviation", float("nan")))
        pct = float(rep.get("rel_deviation_pct", float("nan")))
        v = rep.get("verdict", "")
        print(f"   [exposure] LRT VERDICT: chi2={lr:+.1f} but the free "
              f"coefficient is {pct:.2f}% from 1.0 ({v}).")
        if v in ("PRACTICALLY NEGLIGIBLE", "SMALL"):
            print("   [exposure] -> the exposure channel is NOT materially "
                  "misspecified. Prefer the simpler model unless holdout "
                  "metrics improve.")
        else:
            print("   [exposure] -> the departure is large enough to matter; "
                  "the exposure specification is doing real work.")
        res["coef_magnitude_verdict"] = v
        res["coef_rel_deviation_pct"] = pct
    return res


def refit_and_predict(work, future_work, spec, args, audit):
    # Legacy batched path (main now uses refit_only + per-fixture predict).
    # Kept consistent: full spec + sanity-gated staged fallback.
    info = refit_only(work, spec, args, audit)
    preds = predict_metacount_sim(future_work, info, id_col="row_id",
                                  y_col="Y",
                                  R=max(1000, int(args.refit_R)))
    return info, np.maximum(preds, 1e-9)


def predict_future_chunked(info, future_cal, segs, ctx, seg_flow, profile,
                           work, args, flow_uplift=None, crowd_med=None,
                           temporal_fit=None, seg_risk_map=None,
                           clock_prof=None, flow_uplift_by_hw=None,
                           exp_cal=None):
    """Per-fixture future prediction (numpy only, no JAX).

    Joint flow->speeding stage 2: future exposure uses PREDICTED flow
    (seg_flow * bin_factor * game-day uplift) rather than the static
    background profile, so attendance dynamics propagate into speeding via
    the exposure channel as well as the forced fixed game effects.
    Builds one fixture's segment x 28-bin grid at a time (~211k rows, not
    1.27M), so peak is one chunk. Numerics identical to the batched path
    (row-wise exp(Xb)); ~6x lower peak for n_future=6.
    When temporal_fit (stage 2b AR(1) NB2) is given, its population-mean
    surface is ALSO computed (cheap exp(Xb)) into pred_speeding_temporal;
    the shipped pred_speeding stays JAX unless --temporal-predict is set.
    """
    out_parts = []
    hw_cols = [c for c in work.columns if c.startswith("hw_")]
    _u = (np.asarray(flow_uplift, dtype=float)
          if flow_uplift is not None else np.ones(REL_BINS))
    for _, row in future_cal.iterrows():
        start = row["ko"] + pd.Timedelta(hours=WINDOW_HOURS[0])
        bins = pd.date_range(start, periods=REL_BINS, freq="15min", tz=TZ)
        grid = pd.DataFrame({"road_osmid": np.repeat(segs, REL_BINS),
                             "bin": np.tile(bins, len(segs))})
        # Unique per row, matching the `fixture` values in the observed
        # frame; the display label travels separately.
        grid["fixture"] = row["fixture_id"]
        grid["fixture_label"] = str(row["label"])
        grid["code"] = str(row.get("fixture_type", "") or "")
        grid["attendance"] = float(row["attendance"]) if np.isfinite(
            row["attendance"]) else 0.0
        grid["fixture_type"] = str(row.get("fixture_type", "") or "")
        try:
            grid["ko_hour"] = float(row["ko"].hour) + float(row["ko"].minute) / 60.0
        except Exception:
            grid["ko_hour"] = 0.0
        grid["rel_bin"] = np.tile(np.arange(REL_BINS), len(segs))
        grid = add_calendar(grid)
        grid["in_event"] = 1.0
        grid["rel_hour"] = grid["rel_bin"] / BINS_PER_HOUR + WINDOW_HOURS[0]
        grid["log_att"] = np.log1p(grid["attendance"])
        grid = add_game_temporal(grid)
        grid = add_static(grid, ctx)
        grid["event_x_dist"] = grid["dist_stad"]
        grid = add_game_wide(grid, crowd_med=crowd_med)
        grid = add_exposure(grid, ctx, seg_flow, profile)
        # Per-class u takes precedence over the pooled vector. A single
        # u[b] for the whole network cannot let a residential street and a
        # 4-lane arterial respond differently to a game, which is the most
        # likely reason the exposure LRT rejects the channel.
        if flow_uplift_by_hw and "highway" in grid.columns:
            _hw = (grid["highway"].astype("string")
                   .fillna("unclassified").astype(str).to_numpy())
            _rel = np.asarray(grid["rel_bin"].to_numpy(int)).clip(0, REL_BINS - 1)
            _pu = np.ones(len(grid), dtype=float)
            _hit = np.zeros(len(grid), dtype=bool)
            for _cls, _vec in flow_uplift_by_hw.items():
                _m = _hw == _cls
                if _m.any():
                    _pu[_m] = _vec[_rel[_m]]
                    _hit |= _m
            # classes with no train support keep the pooled multiplier
            if (~_hit).any() and _u is not None and len(_u) == REL_BINS:
                _pu[~_hit] = _u[_rel[~_hit]]
            grid["flow_uplift"] = _pu
            grid["pred_flow"] = grid["exp"] * _pu
            grid["exp"] = grid["pred_flow"].clip(lower=1e-6)
            grid["logexp"] = np.log1p(grid["exp"])
        elif _u is not None and len(_u) == REL_BINS:
            grid["flow_uplift"] = _u[np.asarray(grid["rel_bin"].to_numpy(int)).clip(0, REL_BINS - 1)]
            grid["pred_flow"] = grid["exp"] * grid["flow_uplift"]
            grid["exp"] = grid["pred_flow"].clip(lower=1e-6)
            grid["logexp"] = np.log1p(grid["exp"])
        else:
            grid["flow_uplift"] = 1.0
            grid["pred_flow"] = grid["exp"]
        # Same frozen calibration as training, applied AFTER the game-day
        # uplift so the future exposure sits on the same scale the model was
        # fitted on. Frozen ratios only -- no future counts exist to refit.
        if exp_cal:
            grid = calibrate_exposure(grid, exp_cal)
        # Temporal-x AFTER exposure is final: exp_lag1 rides predicted flow
        # (free lags for the whole window); risk/clock map by segment/code.
        grid = apply_temporal_x(grid, seg_risk_map, clock_prof)
        fw = search_frame(grid.assign(Y=0.0), args.target)
        fw, _ = prepare_metacount_frame(fw, exposure_col="exp",
                                        highway_col="highway")
        # Same shape basis as training, reusing the frozen params.
        fw, _ = add_exposure_shape(
            fw, params=getattr(args, "_exp_shape_params", None),
            hinge=str(getattr(args, "exposure_basis", "none")) in
            ("hinge", "hinge+hw"),
            interact_hw=str(getattr(args, "exposure_basis", "none"))
            in ("hinge+hw", "linear+hw", "hw"))
        for col in hw_cols:
            if col not in fw.columns:
                fw[col] = 0.0
        # Draw-averaged predictive mean (matches the fitted expected; the
        # plug-in at mean random coefficients under-predicts by the
        # log-normal variance inflation). R>=1000 keeps per-row MC error
        # small for the heavy-tailed random coefficients.
        pred = np.maximum(predict_metacount_sim(
            fw, info, id_col="row_id", y_col="Y",
            R=max(1000, int(args.refit_R)), chunk=25000), 1e-9)
        keys = grid[["fixture", "code", "road_osmid", "bin", "rel_bin",
                       "pred_flow", "flow_uplift"]].copy()
        keys["pred_speeding"] = np.asarray(pred, dtype=float)
        keys["pred_speeding_network"] = (
            keys["pred_speeding"] / max(args.penetration, 1e-9))
        keys["exp"] = fw["logexp"].to_numpy()
        if temporal_fit is not None:
            try:
                tpred = np.maximum(
                    np.asarray(temporal_fit.predict(fw), dtype=float), 1e-9)
                # Magnitude guard at the WRITE site, not just at fit time.
                # A near-unit-root AR integrates noise into astronomical
                # counts: 61b (o25938148) shipped a parquet whose
                # pred_speeding_temporal reached 5.18e21 across 1,676,836
                # rows, because the phi_max guard did not exist yet. The
                # fit-time guard now catches that, but a parquet column
                # that large is a live hazard for anything reading it by
                # name, so refuse to write it at all and say so.
                _tmax = float(np.max(tpred)) if tpred.size else 0.0
                _ref = float(np.max(keys["pred_speeding"])) \
                    if "pred_speeding" in keys.keys() and len(keys["pred_speeding"]) \
                    else 0.0
                _cap = max(1e3, 1e6 * max(_ref, 1.0))
                if not np.isfinite(_tmax) or _tmax > _cap:
                    print(f"   [temporal] REFUSING to write "
                          f"pred_speeding_temporal: max={_tmax:.3g} exceeds "
                          f"cap={_cap:.3g} (ref max={_ref:.3g}). The AR "
                          "surface is non-finite or near-unit-root; keeping "
                          "the JAX surface only.")
                    if getattr(args, "temporal_predict", False):
                        print("   [temporal] --temporal-predict requested but "
                              "refused; shipping the JAX surface instead")
                else:
                    keys["pred_speeding_temporal"] = tpred
                    if getattr(args, "temporal_predict", False):
                        keys["pred_speeding"] = tpred
                        keys["pred_speeding_network"] = (
                            tpred / max(args.penetration, 1e-9))
            except Exception as exc:  # noqa: BLE001 - JAX surface stands
                print(f"   WARN: temporal predict failed ({exc}); "
                      "keeping JAX surface")
        out_parts.append(keys)
        del grid, fw, pred, keys
        gc.collect()
    return pd.concat(out_parts, ignore_index=True)


def fit_temporal_stage(train_events, spec, work, args, audit, target):
    """Stage 2b: NB2 + AR(1)/AR(2) refit on contiguous train windows.

    Requires metacountregressor>=1.0.165 (temporal_models). Panels are whole
    segment x fixture windows (id=fixture|road_osmid, time=rel_bin) sampled
    BEFORE row-level caps break contiguity. Fixed terms come from the
    searched spec; random terms are the searched RP bases that also appear
    in fixed (capped at 3 for speed). Returns the fit or None (JAX path
    continues regardless).
    """
    audit["temporal_stage"] = {"attempted": True}
    try:
        from metacountregressor.temporal_models import (
            fit_temporal_random_parameter_model)
    except Exception as exc:
        print(f"   [temporal] skipped (needs metacountregressor>=1.0.165: {exc})")
        audit["temporal_stage"] = {"attempted": True, "skipped": str(exc)}
        return None
    if str(args.temporal_refit or "none").lower() == "none":
        audit["temporal_stage"] = {"attempted": False, "disabled": True}
        return None
    if train_events is None or len(train_events) == 0:
        audit["temporal_stage"]["error"] = "no train events"
        return None
    keep = ["fixture", "road_osmid", "rel_bin", target, "vehicle_count",
            "attendance"] + list(spec.get("fixed_terms", []))
    keep = [c for c in keep if c in train_events.columns]
    panel = train_events[keep].copy()
    panel["panel_id"] = (panel["fixture"].astype(str) + "|" +
                         panel["road_osmid"].astype(str))
    # FULL windows only: AR on gappy series hallucinates unit roots
    # (o25938147: 11.5/28 bins avg -> phi1=0.9976). Drop partial panels,
    # then whole-window sample to cap.
    full_ids = panel.groupby("panel_id")["rel_bin"].nunique()
    full_ids = full_ids[full_ids >= REL_BINS].index.to_numpy()
    if len(full_ids) < 50:
        audit["temporal_stage"]["error"] = (
            f"only {len(full_ids)} full windows (<50); AR needs contiguous "
            "series — use the transferable temporal-x features instead")
        print(f"   [temporal] skipped: {len(full_ids)} full windows "
              "(need >=50)")
        return None
    cap = max(100, int(args.temporal_panels))
    ids = full_ids
    if len(ids) > cap:
        rng = np.random.default_rng(int(args.seed))
        ids = rng.choice(ids, cap, replace=False)
        panel = panel[panel["panel_id"].isin(ids)].copy()
    panel = panel.sort_values(["panel_id", "rel_bin"]).reset_index(drop=True)
    panel["__row"] = np.arange(len(panel))
    fixed = [c for c in spec.get("fixed_terms", []) if c in panel.columns]
    rdm_bases = {str(t).split(":")[0] for t in
                 list(spec.get("rdm_terms", [])) +
                 list(spec.get("rdm_cor_terms", []))}
    rands = [c for c in fixed if c in rdm_bases][:3]
    if not fixed:
        audit["temporal_stage"]["error"] = "no fixed terms overlap panel"
        return None
    print(f"   [temporal] fitting NB2+{args.temporal_refit} on "
          f"{len(panel):,} rows / {panel['panel_id'].nunique():,} panels "
          f"(fixed={len(fixed)}, rdm={len(rands)})...")
    t0 = time.time()
    fit = fit_temporal_random_parameter_model(
        panel, id_col="panel_id", time_col="rel_bin", y_col=target,
        fixed_terms=fixed, random_terms=rands, likelihood="nb",
        temporal_model=str(args.temporal_refit), n_draws=32, maxiter=150)
    phi = {k: float(v) for k, v in dict(fit.temporal_parameters).items()}
    phi_max = max([abs(v) for v in phi.values()] + [0.0])
    if phi_max > float(getattr(args, "temporal_phi_max", 0.95)):
        # Near-unit-root: the AR integrates noise into astronomical counts
        # (o25938147: phi1=0.9976, rmse=1.7e21). Reject; JAX surface ships.
        print(f"   [temporal] REJECTED: non-stationary AR "
              f"(max|phi|={phi_max:.4f} > {args.temporal_phi_max}); "
              "keeping JAX surface")
        audit["temporal_stage"] = {
            "attempted": True, "model": fit.temporal_model,
            "converged": bool(fit.converged),
            "phi": {k: round(v, 4) for k, v in phi.items()},
            "rejected": f"max|phi|={phi_max:.4f} non-stationary",
            "bic": round(float(fit.bic), 1), "aic": round(float(fit.aic), 1),
            "loglik": round(float(fit.loglik), 1),
        }
        return None
    audit["temporal_stage"] = {
        "attempted": True, "model": fit.temporal_model,
        "converged": bool(fit.converged), "message": str(fit.message)[:200],
        "phi": {k: round(float(v), 4) for k, v in phi.items()},
        "bic": round(float(fit.bic), 1), "aic": round(float(fit.aic), 1),
        "loglik": round(float(fit.loglik), 1),
        "n_panels": int(fit.n_panels), "n_obs": int(fit.n_obs),
        "elapsed_s": round(time.time() - t0, 1),
        "fixed_terms": list(fixed), "random_terms": list(rands),
    }
    print(f"   [temporal] {fit.temporal_model} converged={fit.converged} "
          f"phi={audit['temporal_stage']['phi']} BIC={fit.bic:.1f} "
          f"({audit['temporal_stage']['elapsed_s']}s)")
    # In-sample overflow check: population means must live on the observed
    # scale. A "converged" fit with exp-blown predictions (o25938147:
    # rmse 1.7e21) is unusable for --temporal-predict — discard now.
    try:
        _in = np.asarray(fit.predict(panel), dtype=float)
        _ob = pd.to_numeric(panel[target], errors="coerce").fillna(0.0).to_numpy(float)
        if (not np.all(np.isfinite(_in)) or float(np.mean(_in))
                > 100.0 * max(float(np.mean(_ob)), 1e-9)):
            print(f"   [temporal] REJECTED: in-sample mean "
                  f"{float(np.mean(_in[np.isfinite(_in)])):.3g} vs observed "
                  f"{float(np.mean(_ob)):.3g}; keeping JAX surface")
            audit["temporal_stage"]["rejected"] = "in-sample overflow"
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"   [temporal] REJECTED: predict check failed ({exc})")
        audit["temporal_stage"]["rejected"] = f"predict check: {exc}"
        return None
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--future-from", default=None)
    ap.add_argument("--n-future", type=int, default=6)
    ap.add_argument("--penetration", type=float, default=0.05)
    ap.add_argument("--search-algo", default="sa")
    ap.add_argument("--search-iters", type=int, default=120)
    ap.add_argument("--search-chunk", type=int, default=25,
                    help="SA iterations per chunk (checkpoint after each)")
    ap.add_argument("--chunk-time", type=float, default=0.0,
                    help="max seconds per chunk (0 = unlimited)")
    ap.add_argument("--wall-budget", type=float, default=0.0,
                    help="max seconds for the whole search stage "
                         "(refit/predict run after); 0 = unlimited")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="package-side checkpoint every N SA gens (0 = off)")
    ap.add_argument("--n-starts", type=int, default=2,
                    help="SA restarts on the first chunk (1 on warm chunks)")
    ap.add_argument("--search-R", type=int, default=40)
    ap.add_argument("--refit-R", type=int, default=200)
    ap.add_argument("--de-maxiter", type=int, default=8)
    ap.add_argument("--de-popsize", type=int, default=8)
    ap.add_argument("--no-de-warm", action="store_true",
                    help="skip JAX vmap DE warm-start in refit (LBFGS polish "
                         "only; ~16x less peak for N*R large, same optimum)")
    ap.add_argument("--no-slsqp", action="store_true",
                    help="use LBFGS instead of SLSQP in refit (halves grad "
                         "buffers for N*R large)")
    ap.add_argument("--engine", default=ENGINE_DEFAULT)
    ap.add_argument("--surface-scale", default="linear",
                    choices=["linear", "log"],
                    help="colour scale for the per-game predicted surfaces. "
                         "Linear by default: per-15-min-bin values run "
                         "10^0-10^1, and a log axis compressed that range and "
                         "printed counts as '6 x 10^0'. Use log only if a "
                         "game's surface really is multiplicative across bins")
    ap.add_argument("--binomial-refit", default="off", choices=["on", "off"],
                    help="also fit the speeding propensity as a BINOMIAL, "
                         "k ~ Binomial(trips_with_judged_speed, p), and report "
                         "it beside the count refit. The count model needs an "
                         "exposure to scale a count with and the only one "
                         "available is a TMR flow projection, which is what "
                         "the exposure LRT keeps rejecting. Here the "
                         "denominator is MEASURED, so there is no projection "
                         "to get wrong and the flow elasticity becomes a "
                         "coefficient instead of a pinned offset. Reported "
                         "alongside, not instead of, the count path")
    ap.add_argument("--exposure-basis", default="none",
                    choices=["none", "hinge", "hinge+hw", "linear+hw", "hw"],
                    help="shape of the flow->speeding exposure curve. "
                         "'hinge' adds a linear-hinge (piecewise linear) "
                         "deviation basis on logexp at two frozen train "
                         "quantiles, letting the curve bend while the forced "
                         "linear logexp term stays. '+hw' additionally adds "
                         "logexp x highway-class interactions, so the flow "
                         "elasticity differs by road class -- a more robust "
                         "alternative to trusting the per-class uplift u, "
                         "since it changes the ELASTICITY rather than trying "
                         "to fix the exposure scale per class. Off by default")
    ap.add_argument("--zero-inflation", default="off", choices=["on", "off"],
                    help="offer role 6 (zero inflation) to the continuous "
                         "and calendar candidate terms, letting the search fit "
                         "a structural-zero process alongside the count. The "
                         "package derives zero_inflated from len(zi_cols) > 0, "
                         "so with no candidate assigned the model stays "
                         "ordinary. Off by default")
    ap.add_argument("--max-latent-classes", type=int, default=1,
                    help="latent classes for the SEARCH (was hardcoded to 1). "
                         "This is NOT a refit-only knob: with a value > 1 the "
                         "evaluator appends an LC gene plus per-class outcome "
                         "and membership masks, growing the decision vector "
                         "from 2D+1 to 3D+2, activates membership roles 7/8, "
                         "and forces struct_lc to at least this value in "
                         "EVERY spec -- so it is a hard constraint, not a "
                         "soft option. It also inflates the search space, so "
                         "a 3-iteration smoke budget will not find a good "
                         "class structure; raise --search-iters "
                         "substantially before trusting it")
    ap.add_argument("--exposure-calibration", default="off",
                    choices=["off", "on"],
                    help="calibrate the flow proxy onto the OBSERVED trip "
                         "denominator (trips_with_judged_speed) with a "
                         "shrunk per-segment and per-dow/tod ratio, then use "
                         "the calibrated value as the exposure. The response "
                         "is a count of speeding TRIPS, so its true "
                         "denominator is measured, not modelled; the offset "
                         "in use is a TMR flow projection, and that mismatch "
                         "is the leading suspect for the exposure LRT "
                         "rejecting the pinned offset at chi2(1)=364. "
                         "OFF by default so the current channel stays the "
                         "baseline in any comparison")
    ap.add_argument("--calib-k-seg", type=float, default=8.0,
                    help="shrinkage pseudo-count (exposure units) for the "
                         "per-segment calibration ratio")
    ap.add_argument("--calib-k-cell", type=float, default=40.0,
                    help="shrinkage pseudo-count (exposure units) for the "
                         "per dow x time-of-day calibration ratio")
    ap.add_argument("--flow-uplift-by-class", default="off",
                    choices=["on", "off"],
                    help="estimate the game-day flow uplift PER HIGHWAY CLASS "
                         "instead of one pooled scalar per rel_bin. A pooled "
                         "u cannot let a residential street and a 4-lane "
                         "arterial respond differently to a game, and is the "
                         "leading suspect for the exposure LRT rejecting the "
                         "flow channel (chi2(1)=364). OFF by default: it "
                         "changes the shipped exposure channel, so validate "
                         "it against the holdout before relying on it")
    ap.add_argument("--exposure", default="free",
                    choices=["free", "offset", "hazard"],
                    help="exposure parameterisation. 'free' (default) "
                         "estimates the logexp coefficient; 'offset'/'hazard' "
                         "pins it at 1.0 by moving logexp into the "
                         "likelihood's fixed offset (eta += logexp). Nested, "
                         "so --exposure-lrt gives a valid 1-df test of whether "
                         "the projected flow is misspecified")
    ap.add_argument("--exposure-lrt", default="off", choices=["on", "off"],
                    help="after the shipped refit, fit the opposite exposure "
                         "arm and report the 1-df likelihood-ratio test "
                         "(doubles refit time)")
    ap.add_argument("--refit-model", default="searched",
                    choices=["searched", "nb", "poisson"],
                    help="likelihood for the refit. 'searched' (default) "
                         "follows the structural search's dispersion gene, so "
                         "the shipped model is the one BIC chose; 'nb'/'poisson' "
                         "force it")
    ap.add_argument("--search-vars", default="slim", choices=["slim", "full"])
    ap.add_argument("--group-by", default="highway",
                    choices=["highway", "segment", "event"],
                    help="grouped-RP panel dimension: highway class (stable), "
                         "road_osmid segment propensity, or fixture event "
                         "split (per-event coefficients with "
                         "attribute-driven means for future games)")
    ap.add_argument("--no-force-game", action="store_true",
                    help="allow the search to drop game-day terms (default: "
                         "in_event/rel_hour/log_att/event_x_dist are forced)")
    ap.add_argument("--no-flow-uplift", action="store_true",
                    help="predict future exposure from the static background "
                         "profile only (default: joint flow->speeding with "
                         "game-day flow uplift)")
    ap.add_argument("--temporal-refit", default="ar1",
                    choices=["none", "ar1", "ar2"],
                    help="stage 2b: refit searched spec as NB2 + AR model on "
                         "contiguous train windows (needs "
                         "metacountregressor>=1.0.165; skipped otherwise)")
    ap.add_argument("--temporal-panels", type=int, default=500,
                    help="max segment x fixture windows for stage 2b "
                         "(whole windows kept contiguous; ~45 min at 500)")
    ap.add_argument("--temporal-predict", action="store_true",
                    help="ship the AR population-mean surface to the MIP "
                         "instead of the JAX draw-averaged surface")
    ap.add_argument("--no-temporal-vars", action="store_true",
                    help="drop within-window temporal-shape terms "
                         "(rel_hour_sq/sin/cos, pre/post_game) from the "
                         "search pool (peak shape then comes only from "
                         "game dummies + optional stage-2b AR)")
    # Single definition: this flag was declared twice during concurrent
    # edits, which made argparse raise at import. Kept one copy with the
    # fuller help text.
    ap.add_argument("--temporal-phi-max", type=float, default=0.95,
                    help="stage-2b AR stationarity bound: reject the AR fit "
                         "if any |phi| exceeds this (phi~=1 integrates "
                         "noise into astronomical counts, e.g. the "
                         "phi1=0.9976 / rmse=1.7e21 fit)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true",
                    help="continue the search from results/game_day_search_best.json")
    ap.add_argument("--out-dir", default=str(RES))
    ap.add_argument("--fig-dir", default=str(FIG))
    args = ap.parse_args()
    global BEST_JSON, PROGRESS_CSV
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    BEST_JSON = out_dir / "game_day_search_best.json"
    PROGRESS_CSV = out_dir / "game_day_search_progress.csv"
    if args.pilot:
        args.search_iters = min(args.search_iters, 6)
        args.search_chunk = min(args.search_chunk, 5)
        args.search_R = min(args.search_R, 15)
        args.refit_R = min(args.refit_R, 60)
        args.n_future = min(args.n_future, 2)
        args.n_starts = min(args.n_starts, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "pilot" if args.pilot else "full"

    print("[59/1] loading panel + fixtures...")
    panel = load_panel(args.panel, target=args.target)
    panel["bin"] = pd.to_datetime(panel["bin"], utc=True).dt.tz_convert(TZ)
    bin_ns = bin_ns_array(panel)
    cal = load_fixtures()
    ctx = static_context()
    # Split on fixture_id (unique per calendar ROW), never on `label`
    # (which repeats across seasons). Selecting the last n rows rather than
    # the last n names is what makes the holdout actually held out.
    if args.future_from:
        cutoff = pd.Timestamp(args.future_from, tz=TZ)
        _fut = cal.loc[cal["ko"] >= cutoff]
    else:
        _fut = cal.tail(int(args.n_future))
    future_ids = set(_fut["fixture_id"])
    future_labels = set(_fut["label"])
    train_ids = set(cal["fixture_id"]) - future_ids
    _assert_split(cal, future_ids, train_ids)
    audit = {"tag": tag, "target": args.target,
             "penetration": args.penetration,
             "n_train_fixtures": len(train_ids),
             "n_future_fixtures": len(future_ids),
             "train_fixture_rows": int(cal["fixture_id"].isin(train_ids).sum()),
             "future_fixture_rows": int(cal["fixture_id"].isin(future_ids).sum()),
             "future_fixtures": sorted(future_labels),
             "future_fixture_ids": sorted(future_ids)}
    print(f"   train fixtures={len(train_ids)} rows="
          f"{audit['train_fixture_rows']} future fixtures={len(future_ids)} "
          f"rows={audit['future_fixture_rows']}")
    print(f"   future={sorted(future_labels)}")
    # Downstream callers select on fixture_id from here on.
    # Alias kept only so the build_game_frame call site below reads clearly;
    # the values are fixture_ids, NOT labels.

    print("[59/2] building features (sampled background + train windows)...")
    rng = np.random.default_rng(args.seed)
    covered = covered_mask(bin_ns, cal)
    prof_rows = PROFILE_ROWS if not args.pilot else 100000
    prof_sample = sample_background(panel, bin_ns, covered, rng, prof_rows)
    profile = build_profile(add_calendar(prof_sample))
    del prof_sample
    gc.collect()
    # Per-segment mean flow from the WHOLE panel (stable for rare segments);
    # the profile above captures the dow x time-of-day shape from the sample.
    seg_flow = (panel.groupby("road_osmid")["vehicle_count"].mean()).to_dict()
    event_cap = 20000 if args.pilot else EVENT_CAP
    bg_cap = 40000 if args.pilot else BG_CAP
    train_cap = 25000 if args.pilot else TRAIN_CAP
    train_events = build_game_frame(panel, bin_ns, cal, ctx, seg_flow, profile,
                                     train_ids)
    if train_events.empty:
        raise SystemExit("no train event rows; check panel coverage")
    # Crowd-tier anchor from TRAIN attendance (model-consistent future tiers).
    try:
        _tr_att = pd.to_numeric(
            cal[cal["fixture_id"].isin(train_ids)]["attendance"],
            errors="coerce").dropna()
        crowd_med = float(_tr_att.median()) if len(_tr_att) else 0.0
    except Exception:
        crowd_med = 0.0
    audit["crowd_med"] = round(crowd_med, 1)
    # Transferable temporal structure, estimated on FULL train windows
    # (pre-sample) and anchored to train scope — no future leakage:
    # future fixtures/segments map to global-mean fallbacks.
    seg_risk_map, _seg_r0 = estimate_seg_risk(train_events, args.target)
    clock_prof = estimate_clock_prof(train_events, args.target)
    audit["seg_risk"] = {"n_segments": len(seg_risk_map),
                         "global_rate": round(float(_seg_r0), 6)}
    audit["clock_prof"] = {k: [round(float(v), 3) for v in arr]
                           for k, arr in clock_prof.items()}
    train_events = apply_temporal_x(train_events, seg_risk_map, clock_prof)
    print(f"   temporal-x: {len(seg_risk_map)} segment risks, "
          f"clock codes={sorted(clock_prof)})")
    # Stage 2b snapshot: whole-window sample BEFORE row caps break the
    # contiguous 28-bin series the AR estimator needs.
    temporal_snapshot = None
    if str(args.temporal_refit or "none").lower() != "none":
        try:
            _ids = train_events[["fixture", "road_osmid"]].drop_duplicates()
            _cap = max(100, int(args.temporal_panels))
            if len(_ids) > _cap:
                _ids = _ids.sample(_cap, random_state=args.seed)
            temporal_snapshot = train_events.merge(_ids, on=["fixture", "road_osmid"], how="inner").copy()
            print(f"   temporal snapshot: {len(temporal_snapshot):,} rows / "
                  f"{len(_ids):,} windows")
            del _ids
        except Exception as exc:  # noqa: BLE001 - JAX path unaffected
            print(f"   WARN: temporal snapshot failed ({exc})")
            temporal_snapshot = None
    if len(train_events) > event_cap:
        train_events = train_events.sample(event_cap, random_state=args.seed)
    bg_sample = sample_background(panel, bin_ns, covered, rng,
                                  min(len(train_events) * 2, bg_cap))
    background = enrich_background(bg_sample, ctx, seg_flow, profile)
    del bg_sample
    background = apply_temporal_x(background, seg_risk_map, clock_prof)
    # Stage 1 (joint model): game-day flow uplift per rel_bin, estimated
    # while both frames are alive. Disabled with --no-flow-uplift.
    flow_uplift_by_hw = {}
    if args.no_flow_uplift:
        flow_uplift = np.ones(REL_BINS)
    else:
        # Deterministic denominator. estimate_flow_uplift divides by the
        # background mean, and `background` is a random subsample, so the
        # divisor (and therefore every u) moved between runs of identical
        # code. Pass the full-panel aggregate instead; the panel is already
        # in memory here and a scalar mean costs nothing.
        # `panel` has no `highway` column -- road class lives in the static
        # context (road_geometry_features), and the panel is read with a fixed
        # column subset that does not include it. Reading it off the panel
        # therefore always produced an empty dict, so
        # --flow-uplift-by-class silently estimated ZERO classes on job
        # 26077240. Join the class labels on from ctx instead, over the FULL
        # panel (not the background subsample, which would reintroduce the
        # sampling noise the pooled path just removed).
        bg_by_class = {}
        if ctx is not None and "highway" in getattr(ctx, "columns", []):
            try:
                _hw = ctx[["road_osmid", "highway"]].drop_duplicates(
                    subset=["road_osmid"])
                _bgp = (panel.loc[~covered, ["road_osmid", "vehicle_count"]]
                        .merge(_hw, on="road_osmid", how="left"))
                _bgp["highway"] = (_bgp["highway"].astype("string")
                                   .fillna("unclassified").astype(str))
                bg_by_class = (_bgp.groupby("highway")["vehicle_count"]
                               .mean().to_dict())
            except Exception as exc:  # noqa: BLE001
                print(f"   WARN: per-class background failed ({exc})")
                bg_by_class = {}
        _bg_full = float(panel.loc[~covered, "vehicle_count"].mean())
        flow_uplift = estimate_flow_uplift(
            train_events, background, bg_mean=_bg_full)
        # Per-class u. Off by default because it changes the shipped exposure
        # channel; opt in with --flow-uplift-by-class.
        if str(getattr(args, "flow_uplift_by_class", "off")) == "on":
            flow_uplift_by_hw = estimate_flow_uplift_by_class(
                train_events, background, bg_by_class,
                class_col="highway", fallback=flow_uplift)
            # Guard: a requested option that produces nothing must never look
            # like a tested-and-rejected result. Same pattern as the ZI
            # reachability check.
            if not flow_uplift_by_hw:
                print("   [uplift] *** REQUESTED BUT NOT REACHABLE: zero "
                      "highway classes estimated, so this run is NOT a "
                      "per-class-uplift test ***")
            else:
                print(f"   flow uplift by highway class: "
                      f"{len(flow_uplift_by_hw)} classes")
    audit["flow_uplift"] = [round(float(u), 3) for u in flow_uplift]
    audit["flow_joint"] = not args.no_flow_uplift
    print(f"   flow uplift rel_bin 0/12/27: {flow_uplift[0]:.2f}/"
          f"{flow_uplift[12]:.2f}/{flow_uplift[27]:.2f}")
    train_frame = pd.concat([train_events, background], ignore_index=True)
    del train_events, background
    # Calibrate the flow proxy onto the observed trip denominator BEFORE the
    # row cap, so the per-segment ratios see the full event+background set.
    # Estimated on the same rows and then frozen; the future path applies the
    # same ratios to its predicted exp.
    exp_cal = None
    if str(getattr(args, "exposure_calibration", "off")) != "off":
        exp_cal = fit_exposure_calibration(
            train_frame, k_seg=float(getattr(args, "calib_k_seg", 8.0)),
            k_cell=float(getattr(args, "calib_k_cell", 40.0)))
        if exp_cal is not None:
            train_frame = calibrate_exposure(train_frame, exp_cal)
        audit["exposure_calibration"] = (
            "on" if exp_cal is not None else "requested-but-unavailable")
        audit["exposure_calib_segments"] = (
            int(exp_cal["n_segments"]) if exp_cal else 0)
    else:
        audit["exposure_calibration"] = "off"
        audit["exposure_calib_segments"] = 0
    # Printed in BOTH cases, not only when the flag is on: the audit dict is
    # only ever written to game_day_holdout_metrics.json and never reaches
    # stdout, so a log-based test harness cannot see the status otherwise.
    print(f"   exposure calibration: {audit['exposure_calibration']} "
          f"({audit['exposure_calib_segments']:,} segments)")
    if len(train_frame) > train_cap:
        train_frame = train_frame.sample(train_cap, random_state=args.seed)
    work = search_frame(train_frame, args.target)
    del train_frame
    from metacount_nb import prepare_metacount_frame
    work, _ = prepare_metacount_frame(work, exposure_col="exp",
                                      highway_col="highway")
    # Exposure shape is added AFTER prepare_metacount_frame because the
    # road-class interactions need the hw_ dummies it creates. Knots are
    # frozen here and stashed on args for the future path, so both use the
    # identical basis.
    _shape = str(getattr(args, "exposure_basis", "none"))
    work, exp_shape_params = add_exposure_shape(
        work, params=None,
        hinge=_shape in ("hinge", "hinge+hw"),
        interact_hw=_shape in ("hinge+hw", "linear+hw", "hw"))
    if exp_shape_params:
        args._exp_shape_params = exp_shape_params
    audit["exposure_basis"] = _shape
    audit["exposure_hinge_knots"] = (
        [round(float(k), 4) for k in exp_shape_params["knots"]]
        if exp_shape_params and exp_shape_params.get("knots") else [])
    audit["exposure_shape_cols"] = [
        c for c in work.columns
        if c.startswith("logexp_h") or c.startswith("logexp_x_")]
    print(f"   exposure basis: {_shape}  shape cols="
          f"{len(audit['exposure_shape_cols'])}")
    audit["n_train_rows"] = int(len(work))
    print(f"   train rows={len(work):,} (panel kept for future frames; "
          "background sampled, never materialised in full)")
    _clear_jax_caches()

    print("[59/3] metacountregressor structural search...")
    resume_state = None
    if args.resume and BEST_JSON.exists():
        try:
            resume_state = json.loads(BEST_JSON.read_text())
            if resume_state.get("target") != args.target:
                print(f"   WARN: ignoring stale search checkpoint target="
                      f"{resume_state.get('target')!r}; requested {args.target!r}")
                resume_state = None
            elif int(resume_state.get("search_gen", 0)) != int(SEARCH_GEN):
                # Candidate set / constraints changed shape (e.g. wide game
                # block, AR stage): old warm starts and specs are unusable.
                print(f"   WARN: checkpoint search_gen="
                      f"{resume_state.get('search_gen')} != {SEARCH_GEN}; "
                      "starting fresh")
                resume_state = None
            elif not args.no_force_game:
                # Pre-forced-game checkpoints (e.g. BIC=18327 static spec)
                # lack the game block: re-search rather than re-freeze them.
                _rs_fixed = ((resume_state.get("best_spec") or {})
                             .get("fixed_terms", []))
                if not set(GAME_FORCE) <= set(_rs_fixed):
                    print("   WARN: checkpoint predates forced game dynamics "
                          f"(fixed={_rs_fixed}); discarding resume so the "
                          "search re-runs with the game block forced")
                    resume_state = None
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: could not parse {BEST_JSON.name} ({exc}); "
                  "starting fresh")
            resume_state = None
    if resume_state is None and PROGRESS_CSV.exists():
        # Fresh (non-resumed) run: rotate the stale progress trace so the
        # convergence plot can't zigzag across runs (x would jump 80->20).
        try:
            import time as _t
            _bak = PROGRESS_CSV.with_name(
                "game_day_search_progress.bak_"
                f"{_t.strftime('%Y%m%d_%H%M%S')}.csv")
            PROGRESS_CSV.rename(_bak)
            print(f"   rotated stale progress trace -> {_bak.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: progress rotation failed ({exc})")
    total_iters = int(args.search_iters)
    done_iters = int((resume_state or {}).get("iters_completed", 0) or 0)
    if resume_state and done_iters >= total_iters and resume_state.get("best_spec"):
        spec = resume_state.get("best_spec")
        best_bic = resume_state.get("best_bic")
        variables = resume_state.get("variables", [])
        audit["search"] = {"resumed": True, "best_bic": best_bic,
                           "done_iters": done_iters,
                           "total_iters": total_iters, "complete": True}
        print(f"   resumed spec (BIC={best_bic}, "
              f"iters {done_iters}/{total_iters} complete)")
    else:
        spec, best_bic, variables, engine_used, done_iters, complete = \
            run_search(work, args, audit, out_dir, resume_state)
        print(f"   best BIC={best_bic} ({done_iters}/{total_iters} iters, "
              f"complete={complete})")

    spec = ensure_spec(spec, work, audit,
                       force_game=not args.no_force_game,
                       keep_temporal=not getattr(args, "no_temporal_vars", False))
    print(f"   spec: fixed={len(spec.get('fixed_terms', []))} "
          f"rdm={len(spec.get('rdm_terms', []))} "
          f"fallback={audit.get('spec_fallback')}")
    print("[59/4] refit best spec + predict future games...")
    _rss_gb("pre-future-events")
    # Segment list first: after this the 64M-row panel is no longer needed
    # for the grid (grid = segs x 28 bins per fixture from ctx/seg_flow).
    segs = np.sort(panel["road_osmid"].unique())
    future_events = build_game_frame(panel, bin_ns, cal, ctx, seg_flow, profile,
                                     future_ids, crowd_med=crowd_med)
    if future_events.empty:
        raise SystemExit("no future event rows; check panel coverage")
    # Trust the frame, not the calendar: some future windows may have no
    # panel coverage and drop out, and the metrics/MIP iterate this set.
    future_ids = set(future_events["fixture"].unique())
    audit["n_future_fixtures"] = len(future_ids)
    audit["future_fixture_ids"] = sorted(future_ids)
    if not future_ids:
        raise SystemExit("no future fixtures survived panel coverage")
    # Light observed frame for metrics; drop the heavy enriched frame AND
    # the full panel before any JAX allocation or grid build.
    _obs_cols = ["fixture", "road_osmid", "bin", args.target]
    if "vehicle_count" in future_events.columns:
        _obs_cols.append("vehicle_count")
    obs_light = future_events[_obs_cols].copy()
    obs_light = obs_light.rename(columns={args.target: "observed"})
    if "vehicle_count" in obs_light.columns:
        obs_light = obs_light.rename(columns={"vehicle_count": "obs_flow"})
    del future_events
    del panel, bin_ns, covered
    gc.collect()
    _clear_jax_caches()
    _rss_gb("post-panel-free")
    # Keyed on fixture_id so a 2025 twin of a "future" label cannot be
    # dragged back in as a prediction row.
    future_cal = cal[cal["fixture_id"].isin(future_ids)].sort_values(
        "ko").reset_index(drop=True)
    if len(future_cal) != len(future_ids):
        raise SystemExit(
            f"FATAL: future_cal has {len(future_cal)} rows for "
            f"{len(future_ids)} fixture_ids; label collision persists")
    info = refit_only(work, spec, args, audit)
    _rss_gb("post-refit")
    # Report the free exposure coefficient with its SE BEFORE the LRT, so the
    # log shows the magnitude of the departure from 1.0 alongside the chi2.
    # In free mode this is the coefficient the LRT is testing; in offset mode
    # logexp is pinned and the report says so explicitly.
    try:
        report_exposure_coefficient(info, audit, label="shipped refit")
    except Exception as exc:  # noqa: BLE001 - reporting must not kill a run
        print(f"   WARN: exposure coefficient report failed ({exc})")
    # Binomial refit of the same design under a different likelihood, so the
    # two can be compared on the same holdout. Reported, never shipped in
    # place of the count path.
    if str(getattr(args, "binomial_refit", "off") or "off").lower() == "on":
        try:
            _rb, _bo = refit_binomial(work, args, audit, spec=spec)
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: binomial refit failed ({exc})")
            _bo = {"available": False, "reason": str(exc)}
    # Nested LRT on the exposure parameterisation. Runs AFTER the shipped
    # refit so the panel is already freed and the work frame is the only
    # thing alive. Doubles the refit cost, hence opt-in.
    if str(getattr(args, "exposure_lrt", "off") or "off").lower() == "on":
        _exposure_lrt(work, spec, args, audit, info)
        _clear_jax_caches()
    # Stage 2b: NB2 + AR refit on the contiguous-window snapshot (needs
    # metacountregressor>=1.0.165; JAX path continues regardless).
    print("[59/4b] temporal AR refit...")
    temporal_fit = fit_temporal_stage(temporal_snapshot, spec, work, args,
                                      audit, args.target)
    del temporal_snapshot
    gc.collect()
    out = predict_future_chunked(info, future_cal, segs, ctx, seg_flow,
                                 profile, work, args,
                                 flow_uplift=flow_uplift,
                                 flow_uplift_by_hw=flow_uplift_by_hw,
                                 crowd_med=crowd_med,
                                 temporal_fit=temporal_fit,
                                 seg_risk_map=seg_risk_map,
                                 clock_prof=clock_prof,
                                 exp_cal=exp_cal)
    del segs, future_cal
    gc.collect()
    out.to_parquet(out_dir / "game_day_pred_15min.parquet", index=False)
    # Print the path that was ACTUALLY written. This message used to claim
    # "results/game_day_pred_15min.parquet" while the only file touched was
    # the per-run out_dir copy, which led to stage 60 being pointed at a
    # stale shared parquet (dated 29 Aug) instead of the run just finished.
    print(f"   predictions: {len(out):,} rows -> "
          f"{out_dir / 'game_day_pred_15min.parquet'}")

    print("[59/5] future-game metrics + figures...")
    obs = obs_light
    del obs_light
    merged = out.merge(obs, on=["fixture", "road_osmid", "bin"], how="inner")
    del obs
    metrics = {"n_future_rows": int(len(out)),
               "n_observed_rows": int(len(merged))}
    if len(merged) > 10:
        metrics["bin_spearman"] = _spearman(merged["observed"],
                                            merged["pred_speeding"])
        metrics["rmse"] = float(np.sqrt(np.mean(
            (merged["observed"] - merged["pred_speeding"]) ** 2)))
        metrics["mae"] = float(np.mean(np.abs(
            merged["observed"] - merged["pred_speeding"])))
        slope, intercept = _calibration(merged["observed"],
                                        merged["pred_speeding"])
        metrics["calibration_slope"] = slope
        metrics["calibration_intercept"] = intercept
        seg_pred = merged.groupby(["fixture", "road_osmid"])["pred_speeding"].sum()
        seg_obs = merged.groupby(["fixture", "road_osmid"])["observed"].sum()
        seg = pd.concat([seg_pred.rename("pred"), seg_obs.rename("obs")], axis=1)
        metrics["segment_spearman"] = _spearman(seg["obs"], seg["pred"])
        metrics["segment_top20"] = _jaccard_topk(seg["obs"], seg["pred"], 20)
        metrics["segment_top50"] = _jaccard_topk(seg["obs"], seg["pred"], 50)
        # Joint-model diagnostics: flow stage + residual serial correlation.
        # The NB panel has no AR(1) term; |resid_lag1| >> 0 means temporal
        # persistence is left on the table (segment grouping only gives
        # compound symmetry, not decay).
        if "obs_flow" in merged.columns and "pred_flow" in merged.columns:
            metrics["flow_spearman"] = _spearman(merged["obs_flow"],
                                                merged["pred_flow"])
            metrics["flow_mae"] = float(np.mean(np.abs(
                merged["obs_flow"] - merged["pred_flow"])))
        try:
            _m = merged.sort_values(["fixture", "road_osmid", "rel_bin"])
            _r = (_m["observed"] - _m["pred_speeding"]).to_numpy(float)
            _g = (_m["fixture"].astype(str) + "|" +
                  _m["road_osmid"].astype(str)).to_numpy()
            _num, _den, _n = 0.0, 0.0, 0
            for _k in np.unique(_g):
                _rr = _r[_g == _k]
                if len(_rr) >= 4 and np.isfinite(_rr).all():
                    _z = _rr - _rr.mean()
                    _num += float((_z[:-1] * _z[1:]).sum())
                    _den += float((_z ** 2).sum())
                    _n += 1
            metrics["resid_lag1"] = float(_num / max(_den, 1e-12))
            metrics["resid_lag1_groups"] = int(_n)
        except Exception:
            metrics["resid_lag1"] = float("nan")
    try:
        # Concentration guard: one segment must not hog predicted mass
        # (o25899100: a single 45 m osmid held 84% -> degenerate MIP).
        _seg_tot = (out.groupby("road_osmid")["pred_speeding"].sum()
                    .sort_values(ascending=False))
        _top1 = float(_seg_tot.iloc[0] / max(_seg_tot.sum(), 1e-12))
        metrics["top1_share"] = round(_top1, 4)
        metrics["top1_osmid"] = int(_seg_tot.index[0])
        metrics["top5_share"] = round(
            float(_seg_tot.iloc[:5].sum() / max(_seg_tot.sum(), 1e-12)), 4)
        audit["concentration"] = {"top1_share": metrics["top1_share"],
                                  "top1_osmid": metrics["top1_osmid"],
                                  "top5_share": metrics["top5_share"]}
        if _top1 > 0.25:
            print(f"   WARN: concentrated predictions: segment "
                  f"{metrics['top1_osmid']} holds {_top1:.1%} of predicted "
                  "speeding; MIP coverage will be degenerate (60 gate refuses "
                  "above 50% without --allow-concentrated)")
    except Exception as exc:  # noqa: BLE001
        print(f"   WARN: concentration check failed ({exc})")
    if "pred_speeding_temporal" in merged.columns:
        # Stage 2b head-to-head on the same holdout bins.
        try:
            tp = merged["pred_speeding_temporal"].to_numpy(float)
            ob = merged["observed"].to_numpy(float)
            metrics["temporal_bin_spearman"] = _spearman(ob, tp)
            metrics["temporal_rmse"] = float(
                np.sqrt(np.mean((ob - tp) ** 2)))
            _sp = merged.groupby(["fixture", "road_osmid"])[
                "pred_speeding_temporal"].sum()
            _so = merged.groupby(["fixture", "road_osmid"])["observed"].sum()
            _sg = pd.concat([_sp.rename("pred"), _so.rename("obs")], axis=1)
            metrics["temporal_segment_spearman"] = _spearman(
                _sg["obs"], _sg["pred"])
            metrics["temporal_segment_top20"] = _jaccard_topk(
                _sg["obs"], _sg["pred"], 20)
            print(f"   temporal surface: spearman="
                  f"{metrics['temporal_bin_spearman']:.3f} rmse="
                  f"{metrics['temporal_rmse']:.3f} seg_top20="
                  f"{metrics['temporal_segment_top20']:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: temporal metrics failed ({exc})")
    audit["metrics"] = metrics
    (out_dir / "game_day_holdout_metrics.json").write_text(
        json.dumps({"metrics": metrics, "audit": audit}, indent=2, default=str))

    coefs = info.get("coef")
    if coefs is not None and len(coefs):
        coefs.to_csv(out_dir / "game_day_model_coefs.csv", index=False)
    with open(out_dir / "game_day_predict.tex", "w") as tex:
        tex.write("% game_day_predict.tex - generated by 59_game_day_search_predict.py\n")
        tex.write(f"% best BIC: {best_bic}\n")
        for key, val in metrics.items():
            tex.write(f"% {key}: {val}\n")
        tex.write(f"% refit: {json.dumps(audit.get('refit', {}), default=str)}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        if PROGRESS_CSV.exists():
            prog = pd.read_csv(PROGRESS_CSV)
            if "timestamp_iso" in prog.columns:
                prog = prog.sort_values("timestamp_iso").reset_index(drop=True)
            # Split into runs wherever iters_completed steps back (fresh
            # restart after rotation missed, or resumed run): only the last
            # run is the current search; earlier runs plot faint.
            _breaks = [0] + [i for i in range(1, len(prog))
                             if prog["iters_completed"].iloc[i]
                             < prog["iters_completed"].iloc[i - 1]] + [len(prog)]
            fig, ax = plt.subplots(figsize=(7, 4))
            for _a, _b in zip(_breaks[:-1], _breaks[1:]):
                _seg = prog.iloc[_a:_b]
                _cur = (_b == len(prog))
                ax.plot(_seg["iters_completed"], _seg["best_bic"],
                        "o-" if _cur else "o:",
                        color="#0B6E4F" if _cur else "#9DBEAF",
                        ms=5 if _cur else 3,
                        label="current run" if _cur else "prior run")
            ax.set_xlabel("search iterations")
            ax.set_ylabel("best BIC")
            ax.set_title("MetaCount structural search convergence")
            if len(_breaks) > 2:
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / "game_day_search_convergence.png", dpi=120)
            plt.close(fig)
        if len(merged) > 10:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(merged["pred_speeding"], merged["observed"], s=3,
                       alpha=0.25)
            lim = max(float(merged["pred_speeding"].max()),
                      float(merged["observed"].max()))
            ax.plot([0, lim], [0, lim], "r--", lw=1)
            ax.set_xlabel("predicted speeding trips (15-min bin)")
            ax.set_ylabel("observed speeding trips")
            ax.set_title(f"Future games: {metrics.get('bin_spearman', float('nan')):.3f} "
                         "Spearman")
            fig.tight_layout()
            fig.savefig(fig_dir / "game_day_calibration.png", dpi=120)
            plt.close(fig)
        # Per-game surfaces: vehicles vs speeding side by side, rows =
        # per-game top-20 segments ranked by speeding events (log scale so
        # non-hog segments stay visible).
        _dist = {}
        try:
            _dd = ctx[["road_osmid", "dist_stadium_km"]].copy()
            _dd["road_osmid"] = pd.to_numeric(_dd["road_osmid"], errors="coerce")
            _dist = dict(zip(_dd["road_osmid"].astype(np.int64),
                             pd.to_numeric(_dd["dist_stadium_km"], errors="coerce")))
        except Exception:  # noqa: BLE001 - distance is nice-to-have
            _dist = {}
        # Road names + segment midpoints from the OSM extract (for the
        # top-20 table: name and where along the segment).
        _nm, _mid = {}, {}
        try:
            import json as _js
            with open(DATA / "osm_roads_suncorp.geojson") as _gf:
                _gj = _js.load(_gf)
            for _ft in _gj.get("features", []):
                _pr = _ft.get("properties", {}) or {}
                try:
                    _oid = int(float(_pr.get("osmid")))
                except (TypeError, ValueError):
                    continue
                _n = str(_pr.get("name") or "").strip()
                _rr = str(_pr.get("ref") or "").strip()
                _nm[_oid] = _n or _rr or "unnamed"
                try:
                    _coords = _ft["geometry"]["coordinates"]
                    _pts = (_coords if isinstance(_coords[0][0], (int, float))
                            else [p for _part in _coords for p in _part])
                    _mx = _pts[len(_pts) // 2]
                    _mid[_oid] = (round(float(_mx[1]), 5), round(float(_mx[0]), 5))
                except Exception:
                    continue
        except Exception:  # noqa: BLE001 - names are nice-to-have
            pass
        fixtures = list(out["fixture"].unique())
        from matplotlib.colors import LogNorm
        fig, axes = plt.subplots(len(fixtures), 2,
                                 figsize=(13, 3.0 * len(fixtures)),
                                 squeeze=False)
        _top_rows = []
        for i, fx in enumerate(fixtures):
            _fo = out[out["fixture"] == fx]
            _tot = (_fo.groupby("road_osmid")["pred_speeding"].sum()
                    .sort_values(ascending=False))
            _top = _tot.head(20)
            _flow_tot = _fo.groupby("road_osmid")["pred_flow"].sum()
            try:
                _mo = merged[merged["fixture"] == fx]
                _obs_spd = _mo.groupby("road_osmid")["observed"].sum()
                _obs_flo = (_mo.groupby("road_osmid")["obs_flow"].sum()
                            if "obs_flow" in _mo.columns else None)
            except Exception:  # noqa: BLE001
                _obs_spd, _obs_flo = None, None
            for _osm in _top.index:
                _ml = _mid.get(int(_osm), (float("nan"), float("nan")))
                _top_rows.append({
                    "fixture": fx, "road_osmid": int(_osm),
                    "road_name": _nm.get(int(_osm), "unnamed"),
                    "mid_lat": _ml[0], "mid_lon": _ml[1],
                    "pred_speeding": round(float(_top[_osm]), 3),
                    "pred_vehicles": round(float(_flow_tot.get(_osm, float("nan"))), 3),
                    "obs_speeding": round(float(_obs_spd.get(_osm, float("nan"))), 3)
                    if _obs_spd is not None else float("nan"),
                    "obs_vehicles": round(float(_obs_flo.get(_osm, float("nan"))), 3)
                    if _obs_flo is not None else float("nan"),
                    "dist_stadium_km": round(float(_dist.get(int(_osm), float("nan"))), 3)})
            for j, (_col, _cmap) in enumerate((("pred_flow", "viridis"),
                                               ("pred_speeding", "magma"))):
                sub = _fo[_fo["road_osmid"].isin(_top.index)]
                pivot = sub.pivot_table(index="road_osmid", columns="rel_bin",
                                        values=_col, aggfunc="sum").reindex(_top.index)
                ax = axes[i][j]
                _unit = ("vehicles" if _col == "pred_flow"
                         else "speeding events")
                # Scale: linear by default. The values here are order 10^0-10^1
                # per 15-min bin (colorbars ran 2e-1 .. 6e0), and a log axis
                # over that range both compresses the interesting variation
                # and prints tick labels like "6 x 10^0" for what are simply
                # counts of speeding trips. --surface-scale log restores the
                # old behaviour for when a game's surface really is
                # multiplicative.
                if str(getattr(args, "surface_scale", "linear")) == "log":
                    _pmin = max(float(pivot.min().min()), 1e-3)
                    _norm = LogNorm(vmin=_pmin,
                                    vmax=float(pivot.max().max()))
                    _cbl = f"{_unit} per 15-min bin (log scale)"
                else:
                    _norm = None
                    _cbl = f"{_unit} per 15-min bin"
                sns.heatmap(pivot, ax=ax, cmap=_cmap, norm=_norm,
                            cbar_kws={"label": _cbl})
                # Row labels: road name, not osmid. An osmid is meaningless
                # to a reader, and the previous labels made the "ranked by
                # speeding" axis impossible to check by eye.
                #
                # The ticks MUST be set alongside the labels. seaborn's
                # heatmap installs its own y tick locations, and they do not
                # necessarily number len(labels) -- calling
                # set_yticklabels() with a different count raises
                #   ValueError: The number of FixedLocator locations (10) ...
                #             does not match the number of labels (20).
                # That exception was swallowed by the enclosing try/except
                # and took the ENTIRE figure block down with it, so on job
                # 26095208 the surfaces figure, the coefs figure AND
                # game_day_pred_vs_actual.png were all silently skipped with
                # only a one-line WARN to show for it.
                _rowlab = []
                for _osm in _top.index:
                    _nmv = str(_nm.get(int(_osm), "") or "unnamed")
                    _rowlab.append(f"{_nmv[:26]}  {_osm}")
                if len(_rowlab):
                    # Row centres are at i + 0.5 for an imshow-style extent
                    # of [-0.5, n-0.5].
                    ax.set_yticks(np.arange(len(_rowlab)) + 0.5)
                    ax.set_yticklabels(_rowlab, fontsize=5.5, rotation=0)
                ax.tick_params(labelsize=6)
                ax.set_title(_unit + f" | {str(fx)[:44]}", fontsize=8)
                ax.set_xlabel("15-min bin relative to kickoff")
                ax.set_ylabel("road, ranked by predicted speeding (best at top)"
                              if j == 0 else "")
        fig.suptitle("Predicted 15-min surfaces per future game: vehicles vs speeding "
                     "events\n(per-game top-20 segments, ranked by predicted "
                     "speeding; rows identical in both panels so the two are "
                     "comparable)")
        fig.tight_layout()
        fig.savefig(fig_dir / "game_day_surfaces.png", dpi=130)
        plt.close(fig)
        # The proof figure and the top-20 table are separate deliverables
        # from the surfaces plot. They get their own try/except so a
        # cosmetic failure in one figure cannot silently swallow the others
        # -- which is exactly what happened on job 26095208, where one
        # tick/label mismatch cost the surfaces figure, the coefs figure and
        # this proof figure while the job still reported exit 0.
        try:
            # Proof of concept: does the predicted ranking hold up against
            # the ACTUAL observed events for the same games?
            plot_pred_vs_actual(out, merged, fig_dir, out_dir,
                                top_n=int(getattr(args, "value_top_n", 20)))
        except Exception as exc:  # noqa: BLE001
            print("   WARN: pred-vs-actual figure failed (%s)" % exc)
        try:
            pd.DataFrame(_top_rows).to_csv(
                out_dir / "game_day_top20_per_game.csv", index=False)
            print(f"   top-20 per-game table -> game_day_top20_per_game.csv "
                  f"({len(_top_rows)} rows)")
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: top-20 table failed ({exc})")
        if coefs is not None and len(coefs):
            top = coefs.reindex(coefs["coefficient"].abs()
                                .sort_values(ascending=False).index).head(20)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(top["variable"][::-1], top["coefficient"][::-1])
            ax.set_title("Refit coefficients (top-20 by |value|)")
            fig.tight_layout()
            fig.savefig(fig_dir / "game_day_coefs.png", dpi=120)
        plt.close(fig)
        # Success manifest. Written even when nothing failed so the harness can
        # distinguish "figures were produced" from "this run never got that far".
        try:
            (Path(out_dir) / "figures_manifest.json").write_text(json.dumps(
                {"tag": tag, "ok": True, "error": None,
                 "figures_present": sorted(
                     p.name for p in Path(fig_dir).glob("*.png"))},
                indent=2, default=str))
        except Exception as _mexc:  # noqa: BLE001
            print(f"   WARN: could not write figures manifest ({_mexc})")
    except Exception as exc:  # noqa: BLE001 - plots must not kill the run
        print(f"   WARN: plotting failed ({exc})")
        # Record the failure in a file the job's own artefact harness can read.
        #
        # The harness used to grep the PBS log for the string "plotting
        # failed". That cannot work: PBS only copies the job's output file when
        # the job ENDS, so while the assertions are running their own log does
        # not exist yet. The harness therefore fell back to the newest t4mip.o*
        # it could see -- the PREVIOUS run's log -- and reported job 26107921
        # as having a plotting failure when it had none, on the strength of a
        # fault from the killed job 26095208.
        #
        # A manifest the script writes is self-contained and unambiguous.
        try:
            _mf = {
                "tag": tag, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "figures_present": sorted(
                    p.name for p in Path(fig_dir).glob("*.png")),
            }
            (Path(out_dir) / "figures_manifest.json").write_text(
                json.dumps(_mf, indent=2, default=str))
        except Exception as _mexc:  # noqa: BLE001
            print(f"   WARN: could not write figures manifest ({_mexc})")
    print(f"[59] DONE ({tag}). metrics: {json.dumps(metrics, default=str)}")


if __name__ == "__main__":
    main()
