#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','r') as f:
    lines=f.readlines()
# Fix: put includegraphics on its own line, caption on next line
# Current line 355 (0-indexed 354) has both; split them
old = lines[354]  # this currently has \includegraphics + \caption merged
# Replace with just the includegraphics
lines[354] = r'\includegraphics[width=0.95\textwidth]{figures/speeding_street_eb_event_control.png}\\[6pt]'
# The caption should be on the next line (355, 0-indexed 355)
# But currently line 355 (0-indexed 355) is \end{figure} - need to insert caption before it
# Let me check what line 355 (0-indexed) currently is
print(f'Line 355 (0-idx 354): {lines[354][:100]}')
print(f'Line 356 (0-idx 355): {lines[355][:100]}')
# Now I need to insert the caption text before \end{figure}
# Find \end{figure} and insert caption before it
for i in range(350,360):
    if '\\end{figure}' in lines[i]:
        print(f'\\end{figure} found at line {i+1}')
        # Insert caption before this line
        caption_line = r'\caption{\textcolor{Highlight}{\textbf{Kalman-smoothed event-phase risk level} (random walk) with $\pm95\%$ band against observed log-rates,} Event (left) and Control (right) windows. Shows per-and-after-kickoff-hour profiles (pre/during/post phases), not 24-hr clock. Horizontal axis: fixture sequence (Mar 2025--Mar 2026), "between games".}'
        lines.insert(i, caption_line + '\n')
        break
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','w') as f:
    f.writelines(lines)
print('Done - Figure 6 structure fixed')