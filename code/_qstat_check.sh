#!/usr/bin/env bash
echo "=== qstat -f stage5 work 26665306 : Variable_List ==="
qstat -f 26665306 2>/dev/null | grep -o "Variable_List.*" | tr ',' '\n' | \
  grep -E "GA_STAGE5_GENS|GA_STAGE5_RESTARTS|GA_STAGE5_STAG|STAGE5_LC_GENES|GA_S5_MODE_VARIANTS|GA_S5_DEST_VARIANTS|STAGE_ID=|SHARD_SEGMENT"
echo
echo "=== qstat -f stage2 start 26665320 ==="
qstat -f 26665320 2>/dev/null | grep -o "Variable_List.*" | tr ',' '\n' | \
  grep -E "STAGE2_MH|STAGE_ID=|SHARD_SEGMENT"
echo
echo "=== qstat -f stage4 work 26665325 ==="
qstat -f 26665325 2>/dev/null | grep -o "Variable_List.*" | tr ',' '\n' | \
  grep -E "GA_STAGE4_MXL_SEARCH|GA_STAGE4_GENS|STAGE_ID=|SHARD_SEGMENT"
echo
echo "=== qstat -f stage51 othdiscr 26665316 ==="
qstat -f 26665316 2>/dev/null | grep -o "Variable_List.*" | tr ',' '\n' | \
  grep -E "GA_STAGE51_GENS|GA_STAGE51_RESTARTS|STAGE_ID=|SHARD_SEGMENT"
echo
echo "=== qstat -f stage0 merge 26665329 ==="
qstat -f 26665329 2>/dev/null | grep -o "Variable_List.*" | tr ',' '\n' | \
  grep -E "RUN_TAG|STAGE_IDS"