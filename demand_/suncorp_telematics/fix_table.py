#!/usr/bin/env python3
import sys

with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\table.tex','r') as f:
    lines=f.readlines()

# Replace Table 6 caption (line 170, 0-indexed)
new_caption = (
    r'\caption{\textcolor{Highlight}{Out-of-sample method comparison on finable speeding exceedances '
    r'($\ge$10~kph over posted limit), three held-out 2026 fixtures.} Overlap = share of observed top-50 '
    r'segments recovered by the method\'s EB ranking. Variants trained on segment-aggregated finable-speeding '
    r'totals ($n\approx7.5k$; AICs on that scale). Raw-count RMSE improves with complexity; rank quality tied '
    r'($\rho\approx0.96$); EB-count RMSE favours the shrunk base model using $h_{igp}$ direct estimates.\n}'
)
lines[170] = new_caption

with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\table.tex','w') as f:
    f.writelines(lines)

print('Done - Table 6 caption updated')