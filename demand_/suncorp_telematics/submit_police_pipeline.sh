#!/bin/bash
# submit_police_pipeline.sh - submit the game-day police pipeline in order:
#   58 (15-min panel) -> 59 (MetaCount search + future-game predictions)
#                     -> 60 (site-selection MIP + plots)
# Usage: bash submit_police_pipeline.sh

cd "$(dirname "$0")"

JOB58=$(qsub 58_build_15min_panel.pbs) || exit 1
JOB58=${JOB58%%.*}
echo "58 submitted: $JOB58"

JOB59=$(qsub -W depend=afterok:$JOB58 59_game_day_search_predict.pbs) || exit 1
JOB59=${JOB59%%.*}
echo "59 submitted: $JOB59 (after $JOB58)"

JOB60=$(qsub -W depend=afterok:$JOB59 60_game_day_mip.pbs) || exit 1
JOB60=${JOB60%%.*}
echo "60 submitted: $JOB60 (after $JOB59)"

echo ""
echo "Chain: 58 -> 59 -> 60"
echo "Monitor: qstat -u \$USER"
echo "Logs:    suncorp_15min.o*, suncorp_gameday.o*, suncorp_gameday_mip.o*"
