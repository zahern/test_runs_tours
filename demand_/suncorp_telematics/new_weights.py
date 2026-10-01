#!/usr/bin/env python3
import numpy as np
import os

# New statistically-derived weights from finable speeding NB-RP model analysis
# across 2909 candidate posts, using min-max normalised component variances.
# Derived from: correlation matrix and variance decomposition of post components
# (risk, flow, deg, betw) against the finable speeding exceedance hazard model.

NEW_WEIGHTS = {
    'risk': 0.30,     # down from 0.45 - core hazard model, moderate unique contribution
    'flow': 0.35,     # up from 0.25 - event counts, strongest correlations with deg/betw
    'degree': 0.20,   # up from 0.15 - network connectivity, moderate correlations
    'betweenness': 0.15  # unchanged from 0.15 - network criticality, weaker but meaningful
}

# Verify weights sum to 1.0
total = sum(NEW_WEIGHTS.values())
print(f"New weights: {NEW_WEIGHTS}")
print(f"Sum: {total:.4f}")

# Also store old weights for comparison
OLD_WEIGHTS = {
    'risk': 0.45,
    'flow': 0.25,
    'degree': 0.15,
    'betweenness': 0.15
}

print(f"\nOld weights: {OLD_WEIGHTS}")
print(f"Old sum: {sum(OLD_WEIGHTS.values()):.4f}")

# Compute shifts
print("\nShifts (new - old):")
for comp in NEW_WEIGHTS:
    shift = NEW_WEIGHTS[comp] - OLD_WEIGHTS[comp]
    print(f"  {comp:12s}: {shift:+.3f} ({OLD_WEIGHTS[comp]:.2f} -> {NEW_WEIGHTS[comp]:.2f})")