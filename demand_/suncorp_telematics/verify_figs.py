#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','r') as f:
    lines=f.readlines()
for i in range(352,362):
    print(f'Line {i+1}: {lines[i][:250]}')