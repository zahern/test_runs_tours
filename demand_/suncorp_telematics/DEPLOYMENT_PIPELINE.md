# Suncorp deployment pipeline — real telematics, all fixtures

This rebuilds the deployment layer of the paper on the **real** connected-vehicle
telematics (2025-01 → 2026-03), replacing the earlier synthetic
`hard_braking_events.csv` and the auto-fabricated fixture attendances.

## Why this exists

The original `02_master_scan.py` hard-coded a **March-2026-only** window and 6
games, so only 6 fixtures ever got per-event data. The paper's deployment was
then pointed at a separate *synthetic* event file, which auto-extracted 42
high-traffic **days** (not real games) and fabricated their attendances as a
function of the event count. The raw trip data (`data/raw_near_miss_2026/`,
5,000 files, ~88 GB) actually spans **all of 2025 to March 2026**, so every real
Suncorp fixture can be recovered by re-extracting the full period.

## Inputs

- `data/raw_near_miss_2026/trips-*.csv.gz` — 5,000 raw trip files with delimited
  trajectory paths (`SnappedPath`, `TimestampPath`, `SpeedPath_kph`, `XAccPath`,
  `YAccPath`). Copy/symlink these in, or set `RAW_DIR` to the OneDrive location:
  `.../telematics_data/suncorp_deployment_model/data/raw_near_miss_2026`.
- `data/fixtures_all.csv` — real Broncos fixture calendar with crowds (for
  attendance/opponent lookup by date).
- `results/post_selection_mip.csv` — the K=14 candidate posts (Programme P4).

## Steps

| # | script | what it does |
|---|--------|--------------|
| 50 | `50_extract_raw_events.py` | explode raw paths → per-event rows (brake/accel/corner/extreme via XAcc/YAcc thresholds), filter to the Suncorp bbox, stream to `results/event_points_full.csv.gz`. Shardable for PBS arrays; checkpointed. |
| 50b | `50b_merge_shards.py` | merge PBS array shards into one event file. |
| 51 | `51_detect_fixtures.py` | build the fixture calendar. **Not needed** for the default run — `data/fixture_calendar.csv` is already curated (43 real Suncorp events, see below). `--mode known` regenerates the Broncos-only subset; `--mode discover` is a telemetry-surge fallback (unreliable — real games don't cleanly stand out in braking counts). |
| 52 | `52_build_surfaces.py` | build per-fixture space-time surfaces `(N,14,28)` + event cache from the real events + calendar. |
| 28 | `28_post_selection_mip.py` | Programme P4 candidate-post selection (exact MIP). |
| — | `build_benefits_cache.py` | cache junction benefits for the K/weight analyses. |
| 32 | `32_officer_tokens.py` | fig32/fig34 — per-officer token trajectories, aggregate + every fixture. |
| 30 | `30_movement_temporal.py` | per-fixture movement stats, movement CSVs. |
| 33 | `33_choose_K.py` | fig37 — held-out coverage vs K (justifies K=14). |
| 34 | `34_phased_posts.py` | fig38 — phase-resolved candidate posts. |
| 31 | `31_eq1_sensitivity.py` | fig36 — score-weight robustness (coverage-based). |
| 35 | `35_fig_temporal.py` | fig35 — per-fixture variation. |
| 53 | `53_pergame.py` | per-fixture leave-one-out rosters → `deployment_pergame.tex`. |

## Run on HPC (recommended — the extraction is ~88 GB)

```bash
mkdir -p logs
qsub pbs_50_extract.pbs          # 20-way array, re-extracts 2025..2026-03
# wait for all array tasks, then:
python scripts/50b_merge_shards.py
qsub pbs_deploy.pbs              # 51→52→28→32/30/33/34/31/35/53
```

## Run locally (single machine, ~4 h for extraction)

```bash
RAW_DIR=".../suncorp_deployment_model/data/raw_near_miss_2026"
python scripts/50_extract_raw_events.py --raw-dir "$RAW_DIR"
python scripts/51_detect_fixtures.py --fixtures data/fixtures_all.csv --top-n 42
python scripts/52_build_surfaces.py --ordinary 2026-03-18T18:00:00
python scripts/28_post_selection_mip.py --k 14
python scripts/build_benefits_cache.py
for s in 32_officer_tokens 30_movement_temporal 33_choose_K 34_phased_posts 31_eq1_sensitivity 35_fig_temporal 53_pergame; do
  python scripts/$s.py
done
```

## The fixture calendar (`data/fixture_calendar.csv`)

**43 real Suncorp events, 2025-01 → 2026-03**, source-verified from austadiums
and Wikipedia (crowds cross-checked; the 12 regular-season Broncos 2025 games
sum to the official 494,224 aggregate exactly). Breakdown:

- 24 NRL — 16 Broncos home + 3 Magic Round days (2–4 May 2025) + Men's & Women's
  State of Origin G1 + 2 RLWC + Dolphins-Titans (Mar 2026).
- 12 A-League — Brisbane Roar home games (both seasons in range).
- 6 Super Rugby — Queensland Reds home games.
- 1 Football — Socceroos v Brazil.

Crowds span 2,454 → 52,491 (a 21:1 gradient). 38 of 43 carry a verified crowd;
5 (internationals + two 2026 Roar) are left blank — the space-time deployment
doesn't need attendance (only the per-game attendance-scaling in `53` does, which
falls back to the training mean when a crowd is missing).

## Notes
- Event thresholds match the original pipeline: brake `XAcc ≤ −0.30 g`,
  accel `≥ +0.30 g`, corner `|YAcc| ≥ 0.35 g`.
- The 16-fixture matched-control EB/street-risk/temporal layers are unchanged
  (they run on `segment_roads_part0_5000.csv.gz`); this pipeline only rebuilds
  the space-time **deployment** layer so the whole paper is on real data.
- The manuscript lives in `manuscript/` (main.tex + results/ + figures/).
