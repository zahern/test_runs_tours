#!/usr/bin/env bash
# Poll qld_lccm2 shards + merge BY JOB ID (qstat truncates names). Done when
# all four leave the queue; then dump leaderboard + the s5mode rows + lccm err.
cd ~/test_runs_tours/code
IDS='2650171[89]|2650172[0-2]'
for i in $(seq 1 13); do          # ~26 min
  n=$(qstat -u ahernz 2>/dev/null | grep -cE "$IDS")
  [ "$n" = "0" ] && break
  sleep 120
done
n=$(qstat -u ahernz 2>/dev/null | grep -cE "$IDS")
echo "===ACTIVE_COUNT=== $n"
qstat -u ahernz 2>/dev/null | grep -E "$IDS" | awk '{print $1,$4,$10}'
if [ "$n" != "0" ]; then
  echo "===NOT_DONE: convergence tails==="
  for s in work edu nonmand; do
    echo "-- $s --"; tail -1 runs/ga_parallel/qld_lccm2/stage5/shards/$s/convergence_trace.csv 2>/dev/null | tr -d '\r'
  done
  exit 0
fi
echo "===DONE==="
echo "===LEADERBOARD==="
cat runs/ga_parallel/qld_lccm2/merged/shard_leaderboard.csv 2>/dev/null | tr -d '\r' || echo "(no merged leaderboard; per-shard below)"
echo "===PER-SHARD s5mode rows (nl/mnl/lccm)==="
grep -rh 's5mode' runs/ga_parallel/qld_lccm2/stage5/shards/*/variants/leaderboard.csv 2>/dev/null | tr -d '\r'
echo "===LCCM ERRORS in shard stderr (if any)==="
for seg in work edu nonmand; do
  e=$(ls -t ga_s5${seg}_qld_lccm2.e* 2>/dev/null | head -1)
  [ -z "$e" ] && continue
  tr -d '\r' < "$e" | grep -iE 'lccm|LatentClass|AssertionError|mode model build failed|_build_mode_lccm|LCCM built' | tail -4 | sed "s/^/[$seg] /"
done
echo "===END==="
