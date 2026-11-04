#!/usr/bin/env bash
# Join the wrapped Variable_List then filter.
show_vars() {
  local jid=$1; shift
  qstat -f "$jid" 2>/dev/null \
    | sed -n '/Variable_List/,/^\s*[A-Za-z_]* =/p' \
    | tr -d '\t' | tr '\n' ' ' | tr ',' '\n' \
    | grep -E "$1" | sed 's/^ *//'
}
echo "=== stage5 work 26665306 ==="
show_vars 26665306 'GA_STAGE5_GENS|GA_STAGE5_RESTARTS|GA_STAGE5_STAG|STAGE5_LC_GENES|GA_S5_MODE_VARIANTS|GA_S5_DEST_VARIANTS|STAGE_ID=|SHARD_SEGMENT'
echo "=== stage2 start 26665320 ==="
show_vars 26665320 'STAGE2_MH|STAGE_ID=|SHARD_SEGMENT'
echo "=== stage4 work 26665325 ==="
show_vars 26665325 'GA_STAGE4_MXL_SEARCH|GA_STAGE4_GENS|STAGE_ID=|SHARD_SEGMENT'
echo "=== stage51 othdiscr 26665316 ==="
show_vars 26665316 'GA_STAGE51_GENS|GA_STAGE51_RESTARTS|STAGE_ID=|SHARD_SEGMENT'
echo "=== stage0 merge 26665329 ==="
show_vars 26665329 'RUN_TAG|STAGE_IDS'