#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

cache_path = r'Z:\test_runs_tours\demand_\suncorp_telematics\results\benefits_cache.npz'
if os.path.exists(cache_path):
    cache = np.load(cache_path, allow_pickle=True)
    risk = cache['risk']
    flow = cache['flow']
    deg = cache['deg']
    betw = cache['betw']
    cand = cache['cand']
    print(f"Cache loaded: {len(cand)} candidates")
    print(f"risk shape: {risk.shape}, flow shape: {flow.shape}")
    print(f"deg shape: {deg.shape}, betw shape: {betw.shape}")
    print(f"risk mean: {risk.mean():.3f}, flow mean: {flow.mean():.3f}")
    print(f"deg mean: {deg.mean():.1f}, betw mean: {betw.mean():.3f}")
    
    # Compute correlations between each component and "outcome"
    # The outcome could be: observed finable speeding per segment
    # Let's check if we have the finable speeding audit
    audit_path = r'Z:\test_runs_tours\demand_\suncorp_telematics\results\finable_speeding_audit.json'
    if os.path.exists(audit_path):
        import json
        with open(audit_path) as f:
            audit = json.load(f)
        obs_finable = audit.get('obs_finable', None)
        print(f"\nObserved finable from audit: {obs_finable}")
    
    # Compute correlations
    print("\n--- Correlations with observed finable speeding ---")
    # We need observed finable per segment - let's check the segments file
    seg_path = r'Z:\test_runs_tours\demand_\suncorp_telematics\data\segment_roads_part0_5000.csv.gz'
    import gzip
    import csv
    
    # Try to load and compute
    print("\nComputing statistically-derived weights...")
    
    # Approach: use the relative contribution of each component
    # to the total benefit S_j (before decay)
    # Current weights: 0.45, 0.25, 0.15, 0.15
    # Let's compute what data-driven weights would look like
    
    # Normalise each component
    risk_norm = (risk - risk.min()) / (risk.max() - risk.min())
    flow_norm = (flow - flow.min()) / (flow.max() - flow.min())
    deg_norm = (deg - deg.min()) / (deg.max() - deg.min())
    betw_norm = (betw - betw.min()) / (betw.max() - betw.min())
    
    # Compute variance explained by each component
    var_risk = np.var(risk_norm)
    var_flow = np.var(flow_norm)
    var_deg = np.var(deg_norm)
    var_betw = np.var(betw_norm)
    
    total_var = var_risk + var_flow + var_deg + var_betw
    
    w_risk = var_risk / total_var
    w_flow = var_flow / total_var
    w_deg = var_deg / total_var
    w_betw = var_betw / total_var
    
    print(f"Variance-based weights:")
    print(f"  risk:  {w_risk:.4f}")
    print(f"  flow:  {w_flow:.4f}")
    print(f"  degree:{w_deg:.4f}")
    print(f"  betweenness: {w_betw:.4f}")
    print(f"  sum: {w_risk + w_flow + w_deg + w_betw:.4f}")
    
    # Also try: use the RP heterogeneity results if available
    rp_path = r'Z:\test_runs_tours\demand_\suncorp_telematics\results\rp_heterogeneity_summary.csv'
    if os.path.exists(rp_path):
        rp_sum = pd.read_csv(rp_path)
        print(f"\nRP heterogeneity summary loaded: {len(rp_sum)} strata")
        # Could use mean alpha or mean weight across strata
    
else:
    print("No benefits cache found - need to build it first")