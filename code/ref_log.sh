#!/bin/bash
F=/mnt/hpccs01/home/ahernz/test_runs_tours/code/ga_s5_destfix_gatefix.o24981671
echo "=== first 60 lines ==="
head -60 "$F"
echo ""
echo "=== pool line context ==="
grep -n -E 'Candidate pools|Preparing HHTS|running [0-9]+ segment|valid segments|mode_cands|HHTS tours' "$F" | head