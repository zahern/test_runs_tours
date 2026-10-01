#!/usr/bin/env python3
with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','r') as f:
    lines=f.readlines()
# Replace line 356 (fig:kalam caption, 0-indexed = 355)
# Current: \caption{\textcolor{Highlight}{\textbf{Kalman-smoothed latent risk level across fixtures} (random walk) with $\pm95\%$ band against observed log-rates for Event (left) and Control (right) windows. Hor
new_caption = (
    r'\caption{\textcolor{Highlight}{\textbf{Kalman-smoothed event-phase risk level} (random walk) with $\pm95\%$ band against observed log-rates,} '
    r'Event (left) and Control (right) windows. Shows per-and-after-kickoff-hour profiles (pre/during/post phases), not 24-hr clock. '
    r'Horizontal axis: fixture sequence (Mar 2025--Mar 2026), "between games". '
    r'\includegraphics[width=0.95\textwidth]{figures/speeding_street_eb_event_control.png} ~\\\[6pt]}'
)
lines[355] = new_caption

with open('Z:\\test_runs_tours\\demand_\\suncorp_telematics\\manuscript\\main.tex','w') as f:
    f.writelines(lines)
print('Done - Figure 6 caption updated')