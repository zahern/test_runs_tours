#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\scripts\\31_eq1_sensitivity.py','r') as f:
    lines = f.readlines()

# Update line 3 - the docstring
old_doc = lines[2]  # approximately line 3
new_doc = 'S_j = 0.30 r_j + 0.35 f_j + 0.20 g_j + 0.15 b_j   (x distance-decay)\n'
lines[2] = new_doc

# Update line 98 - the base weight vector
# Find the line with base_w
for i, l in enumerate(lines):
    if 'base_w' in l and 'np.array' in l:
        lines[i] = 'base_w = np.array([0.30, 0.35, 0.20, 0.15])\n'
        break

with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\scripts\\31_eq1_sensitivity.py','w') as f:
    f.writelines(lines)
print('Updated 31_eq1_sensitivity.py')