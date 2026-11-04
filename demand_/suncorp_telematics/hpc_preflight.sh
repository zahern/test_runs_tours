#!/bin/bash
# hpc_preflight.sh - preflight checks before submitting the game-day pipeline.
PROJ=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$PROJ" || exit 1
echo "== env =="
module purge >/dev/null 2>&1
module load Anaconda3/2024.02-1 >/dev/null 2>&1
conda activate zigenv >/dev/null 2>&1
python - <<'PY'
import importlib
for mod in ("metacountregressor", "geopandas", "scipy", "pyarrow", "jax"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {mod}: MISSING ({exc})")
PY
echo "== inputs =="
for f in results/traffic_points_full.csv.gz results/post_selection_mip.csv \
         data/fixture_calendar.csv data/osm_roads_suncorp.geojson \
         data/road_geometry_features.csv; do
  if [ -s "$f" ]; then echo "  OK  $f ($(du -h "$f" | cut -f1))"; else echo "  MISSING $f"; fi
done
echo "== stale outputs (must be absent) =="
ls results/segment_15min_panel*.parquet 2>/dev/null && echo "  WARN: monthly panel exists" || echo "  OK no 15-min panel yet"
ls -d results/_15min_tmp 2>/dev/null && echo "  WARN: tmp exists" || echo "  OK no tmp"
ls results/game_day_search_best.json 2>/dev/null && echo "  WARN: search checkpoint exists (59 will resume)" || echo "  OK no search checkpoint"
echo "== disk =="
df -h /mnt/hpccs01 | tail -1
echo "== queue =="
qstat -u ahernz 2>&1 | head -5
echo "== preflight done =="
