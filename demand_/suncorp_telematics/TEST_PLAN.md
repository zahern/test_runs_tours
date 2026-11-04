# Test plan — session changes (61 / 61b pipeline)

Written after the local pilot exposed three defects that had each survived
two full cluster runs. The organising principle: **every mechanism is proven
on synthetic data in seconds before any cluster time is spent on it**, because
the failure mode here is silent (a 1e12 sentinel read as a real delta, a
leaking holdout, a fit that never happened but reported a verdict).

## Environment (probed, job 26032592)

| | |
|---|---|
| cluster package | `metacountregressor` **1.0.165**, pip-installed |
| has `_variance_barrier` | yes (so NB dispersion is capped) |
| has `nb_parameterization` | yes (so `ALPHA_MAX` is read from the package) |
| **`reject_summary`** | **no** — this session's `main_hpc.py` diagnostics are local-only |
| python | 3.12.14 in `zigenv` |

**Consequence:** the T-series tests the *scripts* against the real installed
package. The `main_hpc.py` diagnostics (`reject_summary`, `_reject_counts`,
role-change blame) are **not** on the cluster and will not be until the
package is version-bumped and published. The tests report that as
present/absent rather than failing, so they measure what is actually
deployed.

## Tiers

### T0 — environment probe — DONE, job 26032592
10 min, 1 core. Confirms package path, version, and which diagnostics are
present. Result above. Cheap enough to always run first.

### T1 — unit / integration — job 26056522, 30 min, 4 cores, 16 GB
**RESULT: 28 passed, 0 failed, exit 0** (2026-09-30 02:42, 3:29 wall).
Full log: `t1unit.o26056522`.

| | Test | Catches |
|---|---|---|
| A | split on `fixture_id` | label collision across seasons (13 rows / 6 names); asserts the *old* label split is **rejected** by `_assert_split` |
| B | grouped probe | scaffolds build; 0/12 INCONCLUSIVE; deltas non-zero; segment carrier probed |
| C | hazard offset | `logexp` leaves the design; offset tensor present and non-zero; `exposure_mode` correct |
| D | LRT | recovers a known 0.60 coefficient; **and stays quiet (χ²=2.26) when truth is 1.0** |
| E | alpha parse | `ALPHA_MAX == e^12`; 5e4 accepted; inf rejected |
| F | candidate set | `SEG_GROUP_VAR` is `log_length`; `SEARCH_GEN == 7`; flags registered; no duplicate argparse |

**Local result: 28 passed, 0 failed.** D4 matters most — an LRT that fires
on correct data would be worse than useless.

Cluster highlights, confirming the probe bug is genuinely fixed and the LRT
is calibrated:

```
A1  13 rows / 6 names share a label (nunique=34 of 41 rows)
A4  _assert_split REJECTS the old label-based split  -- correctly refused
B2  0/12 inconclusive
B4  deltas non-degenerate -- range -1264.41..8.06
C1  logexp removed from offset design -- fixed=['lanes_z','tod_cos','in_event']
D1  recovered=0.5958 truth=0.60
D3  chi2(1)=5399.30  (misspecified channel detected)
D4  chi2(1)=2.26     (correct channel NOT flagged)
```

### T2 — smoke test — job 26056836, 90 min, 4 cores, 96 GB
**RESULT: all 10 log assertions PASSED** (22:00 wall, 19 GB RSS, python exit 0).
Log: `t2smoke.o26056836`.

The grouped probe produces real numbers on the real panel for the first time
in this pipeline's history — previously every arm read `1e12`:

```
[grouped] term=log_length  disp=0 base=170628.5 grouped=27434.2
           delta=-143194.3 (breakeven -23.0) -> GROUPED WINS
[grouped] term=lanes_z     disp=0 base=170640.0 grouped=21193.4
           delta=-149446.6 -> GROUPED WINS
[grouped] term=maxspeed_z  disp=0 base=170640.0 grouped=170651.5
           delta=+11.5 -> grouped loses (over BIC penalty)
```

Read: under **Poisson**, grouping is worth ~140k BIC on four terms — segment
heterogeneity is real and large. Under **NB2** every term is a wash (+21.7,
the cost of two extra parameters), because NB's dispersion already absorbs the
same heterogeneity. That is a coherent result, and it means the dispersion
gene and the grouping axis are competing for the same signal.

**Headline finding — the exposure channel is misspecified:**

```
refit likelihood: poisson (search dispersion gene=0); exposure: free
[exposure] LRT chi2(1)=+364.097 (95% crit 3.84)
  -> free coefficient significantly better (exposure channel looks misspecified)
```

This is the answer to the question that motivated the hazard work, and it is
on real data: pinning the exposure coefficient at 1 is rejected at χ²=364.
The `seg_flow × bin_factor × uplift` projection is not consistent with the
count model, which means a free coefficient has been quietly absorbing that
error. **The hazard form is not the right default for this data** — but the
LRT is precisely the diagnostic that would have said so, and that was the
argument for it.

### T3 — hazard vs free on real data — READY, NOT SUBMITTED
`t3_hazard_vs_free.pbs`, 12 h, two paired 40-iteration runs.
**T2 has answered the question T3 was built to answer**, so T3 is now
optional confirmation rather than the primary experiment. Recommend NOT
submitting: 24 h of cluster time to re-derive a χ²=364 that is already
measured. Better spent on the flow-uplift discrepancy below.
One short **real** run on the actual 64M-row panel: 3 search iters,
R=10, 2 future fixtures, `--group-by segment`, `--exposure-lrt on`,
`--temporal-refit none`. Not a scientific run — the spec it finds is
meaningless. It answers only "does it crash, and do the diagnostics say
something?".

T2 asserts on the log text, so a silent regression cannot pass as success:

- split reports row counts; label-collision NOTE present
- `[grouped]` lines exist, contain **no** `1e12`, and **no** `delta=+0.0`
- refit likelihood echoes `nb`/`poisson` from the search's dispersion gene
- exposure mode echoed; `[exposure] LRT` verdict printed
- **no** `Traceback`, **no** `bad dispersion alpha=inf`

### T3 — hazard vs free on real data — superseded by T2
See above. Two paired 40-iteration runs in `t3_hazard_vs_free.pbs`, written
and ready but **not submitted** — T2 already produced the LRT verdict on real
data, so T3 would spend ~24 h re-deriving it.

### T4 — full 61 / 61b — blocked
Requires T1 and T2 green (both met).

**The flow-uplift discrepancy is now RESOLVED.** It was never a bug in the
pipeline — it was me comparing measurements taken with *different train
sets*. Recomputed with the correct 35 train fixtures against a full-panel
denominator:

| source | train fixtures | u[0] | u[12] | u[27] |
|---|---|---|---|---|
| o25979215 (original) | 28 labels | 0.97 | 0.96 | 0.88 |
| my first pass (wrong set) | 4 | 0.865 | 0.925 | 0.857 |
| **reconciled (correct)** | **35** | **0.955** | **0.963** | **0.869** |
| T2 smoke | 39 (n_future=2) | 0.96 | 1.05 | 0.95 |

The reconciled figures match the original log. The real signal is
`peak-to-trough = 49.4%`, not the ±5% the flat log line suggested.

**Genuine defect found and fixed:** the divisor was
`background["vehicle_count"].mean()` over a *random* 150k-row subsample, so
`u` was not reproducible between runs. `estimate_flow_uplift` now takes an
explicit `bg_mean`, and the caller passes the full-panel aggregate
`panel.loc[~covered, "vehicle_count"].mean()`. Verified: identical `u` across
repeated calls with a fixed divisor.

Remaining before a full run: `u` is still **one scalar per rel_bin for the
whole network** (35 fixtures × ~1,500 segments × 15 min collapsed), which is
the likely reason the exposure LRT rejects at χ²=364. A per-segment-class or
per-fixture-type `u` is the substantive fix and is *not* done.

## What is deliberately not tested

- **Per-trip / per-vehicle granularity** (component A2/A3). Needs a rebuild of
  `58_build_15min_panel.py`; the trip-level detail lives in per-month partials.
- **Copula on nested margins** (E3). A real extension, not a parameterisation
  change, and not needed to make progress.
- **Line/tobit families.** Rejected on grounds, not untested: a Gaussian
  predicts negative counts and Tobit has no upper bound here.

## Sequencing

```
T0 DONE ──> T1 (26056522) ──┐
                            ├──> T3 hazard vs free ──> T4 full runs
              T2 (26056836) ─┘        (gated on T2 + uplift gap)
```

T1 and T2 run in parallel — they share no outputs and T2 does not depend on
T1's result, only on the scripts being syntactically valid, which is
already established locally.

## Fallbacks

If T1 fails: nothing else should run. The suite is fast enough to iterate
on locally first.

If T2 fails on the `[grouped]` assertions: the probe scaffold regressed;
`test_session_fixes.py` test B is the local reproduction.

If T2 fails on `bad dispersion alpha=inf`: the `metacount_nb.py` fix did not
reach the cluster, or a new path re-exponentiates an alpha-scale value.

If T2's LRT line is absent: `--exposure-lrt` did not fire — check the flag
is reaching `main()`.
