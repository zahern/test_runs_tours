#!/bin/bash
# READ-ONLY audit. Stage breakdown from python entrypoints + pbs scripts.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

echo "=== ga_hypothesis_testing.py HEAD (120) ==="
sed -n '1,120p' "$CODE/ga_hypothesis_testing.py"

echo
echo "=== ga_variant_refits.py HEAD (120) ==="
sed -n '1,120p' "$CODE/ga_variant_refits.py"

echo
echo "=== grep stage defs across py ==="
grep -n -E "STAGE[0-9_]*\s*=|def run_stage|stage_name|STAGES\s*=|['\"]stage[0-9]" "$CODE"/ga_*.py "$CODE"/stage*.py 2>/dev/null | head -120