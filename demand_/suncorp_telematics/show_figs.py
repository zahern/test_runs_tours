#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','r') as f:
    lines=f.readlines()
# Show lines 285-365 for figure context
for i in range(284,365):
    print(f'Line {i+1}: {lines[i][:200]}')