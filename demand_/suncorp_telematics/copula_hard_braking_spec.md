# Copula extension: joint speeding + hard-braking model

**Status:** spec (not yet built) · **Date:** 2026-10-19 · **Author:** pipeline

## 1. Question

The deployment model predicts finable speeding counts per segment x 15-min
cell. The user asks whether we can extend the search model to also predict
**hard-braking events** on the same cells and link the two margins with a
**copula**, with two hoped-for benefits:

1. **Separation** — a distinct braking margin instead of lumping braking and
   speeding into one risk number (the old `Y = braking + speeding` estimand,
   script 21, deliberately abandoned in script 42).
2. **Driver aggression** — the joint upper tail (cells high on *both*
   speeding and braking) as an aggression signal, and the copula dependence
   itself as a per-fixture aggression diagnostic.

## 2. What exists

- `data/hard_braking_events.csv` — 2.5M event rows: `VehicleID, VehicleType,
  EventStart, EventEnd, PeakDecel_g, PeakJerk_g_s, SpeedLoss_kmh, Duration_s,
  Category, Confidence, SnappedLongitude, SnappedLatitude, RawLongitude,
  RawLatitude`. Snapped to lat/lon, **not** to `road_osmid`.
- `scripts/18_prepare_eb_dataset.py` — snaps events to nearest road segment
  (reusable snapping logic) and builds the EB dataset with `braking` /
  `speeding` columns.
- `results/segment_15min_panel.parquet` — 64M rows, 7,558 segments, 43,729
  bins, months 2025-01..2026-04. **Speeding only** (`speeding_gt5`,
  `vehicles`, `speeding_trips`). No braking column.
- `scripts/59_game_day_search_predict.py` — the speeding search + two-stage
  surface (metacountregressor, Bayesian structural search, BIC).
- `scripts/60_game_day_mip.py` — site-selection MIP on a predicted surface.

## 3. Design (staged, each with a gate)

### Stage 1 — braking margin (data + count model)

1. **Build a braking panel** (new script `58b_build_braking_panel.py` or an
   extension of 58): load `hard_braking_events.csv`, snap to `road_osmid`
   (reuse script 18's snapping), aggregate to segment x 15-min bin counts.
   - Quality filter: keep `Confidence` categories that are defensible
     (e.g. drop `Low` if it is most of the data — check the distribution).
   - Optional intensity weighting: count events, or sum `PeakDecel_g` as a
     severity-weighted braking measure. Default = raw counts.
   - Output: `results/segment_15min_panel_braking.parquet` (or a `braking`
     column added to the main panel).
2. **Fit a braking count model** mirroring the speeding model. Two options:
   - **(a) Parallel metacountregressor search** with a braking response —
     cleanest, reuses the search machinery, but needs a `--target braking`
     (or a response-column flag) in 59. Bigger change.
   - **(b) NB/Poisson GLM** on the same design family (exposure = trips) —
     fast, good enough for a margin that only needs to rank cells.
   - Recommend **(b)** for stage 1, escalate to (a) if the margin underperforms.

**Gate 1:** holdout deviance of the braking model < intercept-only baseline;
calibration slope in a sane range. If the braking margin cannot beat a
baseline, the copula has nothing to link — stop.

### Stage 2 — copula layer + aggression score

1. **Marginal PITs:** per cell, transform observed speeding and braking counts
   to pseudo-observations via their marginal CDFs (Poisson/NB CDF at the
   fitted mean), then to normal scores (inverse normal CDF). Ties/discreteness
   handled by jittering or mid-PIT.
2. **Fit a copula** on the normal scores, per fixture (or per day-type):
   - Families: Gaussian (1 param rho), Clayton (lower tail), Gumbel (upper
     tail), Frank (symmetric). Select by AIC.
   - The **aggression-relevant** dependence is the **upper tail** (both high),
     so Gumbel/Clayton are the a-priori candidates; Gaussian is the null.
3. **Aggression score per segment:** under the fitted copula, the joint
   upper-tail probability
   `P(Y_speed > q90, Y_brake > q90)` (or the conditional
   `E[Y_brake | Y_speed > q90]`), aggregated to segment level. This is the
   "driver aggression" ranking.
4. **Dependence diagnostic:** the copula rho (and tail-dependence coefficient)
   per fixture — is speeding-braking dependence stable across games?

**Gate 2:**
- Copula log-likelihood > independence copula (LR test, rho != 0).
- The aggression score ranks held-out **joint** events (cells where both
  speeding and braking exceed their q90) better than either margin alone
  (Q1 share of joint events; 25% = chance).

### Stage 3 — deployment arm

- New predicted column `pred_aggression` (the joint exceedance probability)
  written to a parquet for a MIP arm (`60d_game_day_mip_aggression.pbs`),
  alongside `pred_speeding`. Lets the deployment target aggressive corridors
  (both) vs speeding-only, and quantifies the overlap.

## 4. Honesty / risks

- **Discreteness:** counts are small integers; PITs are coarse. Mid-PIT +
  jitter, and report sensitivity to the jitter.
- **Confidence filter:** most events are `Low` confidence; the filter choice
  can change the margin. Report the distribution and gate on the filtered
  set.
- **Causality:** the copula captures *dependence*, not causation. Aggression
  score = "this cell has both phenomena", not "speeding causes braking".
- **Estimand discipline:** the deployment target stays finable speeding
  (no risk weighting). The aggression score is a *separate* surface for
  exploration, not a re-weighting of the speeding estimand.
- **Search integration:** a full joint copula inside the Bayesian structural
  search is a large metacountregressor change. The staged approach keeps the
  search on the margins and adds the copula as a post-search layer.

## 5. Open decisions (need user input)

1. Braking margin: **(a)** parallel metacountregressor search vs **(b)** NB
   GLM? (Recommend b for stage 1.)
2. Braking event quality filter: keep all, or drop `Low` confidence?
3. Aggression metric: joint exceedance `P(both > q90)` vs conditional
   `E[brake | speed > q90]`?
4. Copula family: fix Gaussian, or select by AIC among Gaussian/Clayton/
   Gumbel/Frank?

## 6. Files / jobs touched (when built)

- new `scripts/58b_build_braking_panel.py` (+ pbs) — braking panel.
- new `scripts/68_copula_aggression.py` (+ pbs) — copula + aggression score.
- new `60d_game_day_mip_aggression.pbs` — MIP arm on the aggression surface.
- No changes to the shipped speeding pipeline or its outputs.
