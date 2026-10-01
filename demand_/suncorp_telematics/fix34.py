#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\scripts\\34_phased_posts.py','r') as f:
    lines = f.readlines()

# Update the weighted sum
for i, l in enumerate(lines):
    if '0.45*minmax(riskv)' in l:
        # Replace all four weights in one line
        old = l
        new = l.replace('0.45*minmax(riskv)', '0.30*minmax(riskv)')
        new = new.replace('0.25*minmax(flow)', '0.35*minmax(flow)')
        new = new.replace('0.15*minmax(deg)', '0.20*minmax(deg)')
        new = new.replace('0.15*minmax(betw)', '0.15*minmax(betw)')
        lines[i] = new
        break

with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\scripts\\34_phased_posts.py','w') as f:
    f.writelines(lines)
print('Updated 34_phased_posts.py')