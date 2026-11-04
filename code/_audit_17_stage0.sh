#!/bin/bash
# READ-ONLY. Stage 0: shard result vs merged -1e9 sentinel.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
D=$CODE/runs/ga_parallel/cdap_s5_refit_20260928

echo "=== stage0 shard ga_summary.csv ==="
cat "$D/stage0/shards/all/ga_reports/ga_summary.csv" 2>/dev/null | cut -c1-300
echo
echo "=== stage0 shard_info.json ==="
cat "$D/stage0/shards/all/shard_info.json" 2>/dev/null
echo
echo "=== stage0 ga_progress.json ==="
cat "$D/stage0/shards/all/ga_progress.json" 2>/dev/null
echo
echo "=== stage0 convergence_trace.csv ==="
cat "$D/stage0/shards/all/convergence_trace.csv" 2>/dev/null
echo
echo "=== stage0 ga_results.pkl best (via python, read-only) ==="
/mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv/bin/python - <<'PY' 2>&1 | head -20
import pickle
p="/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/cdap_s5_refit_20260928/stage0/shards/all/ga_results.pkl"
d=pickle.load(open(p,"rb"))
print("type:", type(d))
for a in ("best_fitness","final_best_so_far","n_generations","selected_features","best_features"):
    print(a, "=", getattr(d, a, "<none>"))
PY
echo
echo "=== merged ga_summary stage0 row (raw) ==="
grep -n '^0,' "$D/merged/ga_reports/ga_summary.csv" 2>/dev/null | cut -c1-400
echo
echo "=== paper_missing stage0 shard files ==="
ls -1 --time-style=long-iso -l "$CODE/runs/ga_parallel/paper_missing_20260929/stage0/shards/all/ga_reports/" 2>/dev/null
cat "$CODE/runs/ga_parallel/paper_missing_20260929/stage0/shards/all/ga_reports/ga_summary.csv" 2>/dev/null | cut -c1-200
echo
echo "=== ga_s0all_paper_missing stdout: fitness ==="
grep -n -E 'best_fitness|sentinel|1e\+09|generations|GA result' "$CODE/ga_s0all_paper_missing_20260929.o26591203" 2>/dev/null | cut -c1-200 | head -15