#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','r') as f:
    lines=f.readlines()
# Replace the \includegraphics line (line 355, 0-indexed 354) with speeding_street_eb_event_control.png
lines[354] = r'\includegraphics[width=0.95\textwidth]{figures/speeding_street_eb_event_control.png}\\[6pt]'
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','w') as f:
    f.writelines(lines)
print('Done - Figure 6 image reference updated')