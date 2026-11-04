#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== stage5 job 26665306 : did the key vars land? ==="
f=$(ls ga_s5work_qld_conv1.o26665306 2>/dev/null || ls ga_s5work_qld_conv1.*26665306 2>/dev/null | head -1)
echo "file: $f"
if [ -n "$f" ]; then
  sed -n '/Submit_arguments/,/^$/p' "$f" | tr ',' '\n' | grep -E "GA_STAGE5_GENS|GA_STAGE5_RESTARTS|GA_STAGE5_STAG|STAGE5_LC_GENES|GA_S5_MODE_VARIANTS|GA_S5_DEST_VARIANTS|STAGE_ID|SHARD_SEGMENT"
fi
echo
echo "=== stage2 job 26665320 : STAGE2_MH landed? ==="
g=$(ls ga_s2start_qld_conv1.*26665320 2>/dev/null | head -1)
if [ -n "$g" ]; then
  sed -n '/Submit_arguments/,/^$/p' "$g" | tr ',' '\n' | grep -E "STAGE2_MH|STAGE_ID|SHARD_SEGMENT"
fi
echo
echo "=== stage4 job 26665325 : GA_STAGE4_MXL_SEARCH landed? ==="
h=$(ls ga_s4work_qld_conv1.*26665325 2>/dev/null | head -1)
if [ -n "$h" ]; then
  sed -n '/Submit_arguments/,/^$/p' "$h" | tr ',' '\n' | grep -E "GA_STAGE4_MXL_SEARCH|GA_STAGE4_GENS|STAGE_ID|SHARD_SEGMENT"
fi