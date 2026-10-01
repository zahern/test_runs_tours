#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','r') as f:
    lines=f.readlines()
for i,l in enumerate(lines):
    if 'fig:kalam' in l or 'speeding_street_eb_event_control' in l or 'fig6' in l.lower():
        print(f'Line {i+1}: {l.rstrip()[:200]}')