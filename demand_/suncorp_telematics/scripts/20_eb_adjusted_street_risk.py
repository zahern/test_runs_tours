"""
20_eb_adjusted_street_risk.py
Implement EB-adjusted street risk for the Police paper using outputs from
17_segment_roads.py and the EB framework from 19_eb_analysis.py.

Computes EB-adjusted risk scores for each road segment accounting for:
  - Exposure (trip counts)
  - Event/Control comparison
  - Regression-to-the-mean adjustment via EB

Outputs:
  - results/eb_street_risk.csv       EB-adjusted risk scores per segment
  - results/eb_street_risk.md        Markdown summary for Police paper
  - figures/eb_street_risk_*.png     Visualizations

Usage:
  python 20_eb_adjusted_street_risk.py
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb, predict_metacount

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
EB_PREDICTIONS = os.path.join(RES, "eb_predictions.csv")
ROAD_META = os.path.join(DATA, "road_meta.csv")  # FIX: meta lives in data/, not results/
FEATURES = os.path.join(DATA, "road_geometry_features.csv")

OUTPUT_CSV = os.path.join(RES, "eb_street_risk.csv")
OUTPUT_MD = os.path.join(RES, "eb_street_risk.md")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Numeric geometry covariates, standardised once (z-score) so fit and predict
# see identical scaling; mixed raw scales make the NB alpha diverge.
GEO_NUM = ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
           "dist_to_intersection_m", "node_degree", "length_m"]
_GEOM_CACHE = None


def geom_table():
    """Per-osmid geometry feature table with globally-standardised _z columns."""
    global _GEOM_CACHE
    if _GEOM_CACHE is None or not os.path.exists(FEATURES):
        if not os.path.exists(FEATURES):
            return None
        t = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        for c in ["curvature_deg_km", "sinuosity", "is_link", "oneway",
                  "road_class_rank", "node_degree", "length_m"]:
            if c in t.columns:
                t[c] = t[c].fillna(0)
        for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
            if c in t.columns:
                t[c] = t[c].fillna(t[c].median())
        for c in GEO_NUM:
            if c in t.columns:
                sd = t[c].std()
                t[c + "_z"] = (t[c] - t[c].mean()) / (sd if sd and sd > 0 else 1.0)
        _GEOM_CACHE = t
    return _GEOM_CACHE


def _mc_terms(mc):
    fixed = ["logexp"]
    for c in mc.columns:
        if c.startswith("hw_"):
            fixed.append(c)
    for c in ["maxspeed_kmh_z", "curvature_deg_km_z"]:
        if c in mc.columns and c not in fixed:
            fixed.append(c)
    rdm = [f"{c}:normal" for c in ["lanes_z", "dist_to_intersection_m_z"] if c in mc.columns]
    return fixed, rdm


def fit_spf_by_type(df, event_type):
    """Fit NB random-parameters count model via metacountregressor."""
    df_sub = df[df['type'] == event_type].copy()
    if len(df_sub) == 0:
        return None, None
    agg = df_sub.groupby('road_osmid').agg({
        'braking': 'sum', 'speeding': 'sum', 'trips': 'sum'}).reset_index()
    agg = agg[agg['trips'] > 0].copy()
    agg['total_risk'] = agg['braking'] + agg['speeding']
    if 'highway' in df_sub.columns and 'highway' not in agg.columns:
        hw = df_sub.groupby('road_osmid')['highway'].first().reset_index()
        agg = agg.merge(hw, on='road_osmid', how='left')
    gt = geom_table()
    if gt is not None:
        agg = agg.merge(gt, on='road_osmid', how='left')
    if 'highway' in agg.columns:
        agg['highway'] = agg['highway'].fillna('unknown').astype(str)
    if len(agg) < 10:
        return None, agg
    mc, _ = prepare_metacount_frame(agg, exposure_col="trips", highway_col="highway")
    # attach geometry _z already in gt; ensure present
    for c in list(mc.columns):
        pass
    # merge _z cols from agg if prepare dropped them (it keeps all cols)
    fixed, rdm = _mc_terms(mc)
    try:
        info = fit_metacount_nb(mc, y_col="total_risk", id_col="road_osmid",
                                fixed_terms=fixed, rdm_terms=rdm,
                                model="nb", R=150, de_maxiter=8,
                                de_popsize=8, verbose=True)
        print(f"     AIC: {info['aic']:.1f}, Alpha: {info['alpha']:.4f} [{info['spec_label']}]")
        return info, agg
    except Exception as e:
        print(f"  Warning: MetaCount fit failed for {event_type}: {e}")
        return None, agg


def eb_adjustment(observed, expected, alpha):
    """Empirical Bayes adjustment.

    EB = w * expected + (1 - w) * observed, where w = 1/(1 + alpha * expected).
    Low-exposure segments (small expected) get high weight on the model
    prediction; high-exposure segments retain their own observations.
    """
    weight = 1.0 / (1.0 + alpha * expected)
    eb_estimate = weight * expected + (1.0 - weight) * observed
    return eb_estimate, weight


def compute_eb_risk_scores(df, spf_event, spf_control):
    """Compute EB-adjusted risk scores."""
    
    # Aggregate by road segment and type
    event_agg = df[df['type'] == 'Event'].groupby('road_osmid').agg({
        'braking': 'sum',
        'speeding': 'sum',
        'trips': 'sum'
    }).reset_index()
    event_agg.columns = ['road_osmid', 'event_braking', 'event_speeding', 'event_trips']
    event_agg['event_total'] = event_agg['event_braking'] + event_agg['event_speeding']
    
    control_agg = df[df['type'] == 'Control'].groupby('road_osmid').agg({
        'braking': 'sum',
        'speeding': 'sum',
        'trips': 'sum'
    }).reset_index()
    control_agg.columns = ['road_osmid', 'control_braking', 'control_speeding', 'control_trips']
    control_agg['control_total'] = control_agg['control_braking'] + control_agg['control_speeding']
    
    # Merge
    result = event_agg.merge(control_agg, on='road_osmid', how='outer')
    result = result.fillna(0)
    
    # Attach highway (for C(highway)) and geometry covariates (same
    # standardised table used for fitting) so spf.predict() sees every
    # variable in the augmented formula.
    if 'highway' in df.columns and 'highway' not in result.columns:
        hw = df.groupby('road_osmid')['highway'].first().reset_index()
        result = result.merge(hw, on='road_osmid', how='left')
    gt = geom_table()
    if gt is not None:
        result = result.merge(gt, on='road_osmid', how='left')
        for c in GEO_NUM:
            if c + "_z" in result.columns:
                result[c + "_z"] = result[c + "_z"].fillna(0)
    if 'highway' in result.columns:
        result['highway'] = result['highway'].fillna('unknown').astype(str)
    
    # Compute expected values from MetaCount NB random-parameters models
    if spf_event is not None:
        alpha_event = float(spf_event["alpha"])
        tmp = result.rename(columns={'event_trips': 'trips'}).copy()
        mc_tmp, _ = prepare_metacount_frame(tmp, exposure_col="trips", highway_col="highway")
        for c in spf_event["fixed_terms"] + [r.split(":")[0] for r in spf_event["rdm_terms"]]:
            if c not in mc_tmp.columns:
                mc_tmp[c] = 0.0
        result['expected_event'] = predict_metacount(mc_tmp, spf_event)
        result['eb_event'], result['eb_weight_event'] = zip(*result.apply(
            lambda row: eb_adjustment(row['event_total'], row['expected_event'], alpha_event), axis=1
        ))
    else:
        result['expected_event'] = 0
        result['eb_event'] = result['event_total']
        result['eb_weight_event'] = 1.0

    if spf_control is not None:
        alpha_control = float(spf_control["alpha"])
        tmp = result.rename(columns={'control_trips': 'trips'}).copy()
        mc_tmp, _ = prepare_metacount_frame(tmp, exposure_col="trips", highway_col="highway")
        for c in spf_control["fixed_terms"] + [r.split(":")[0] for r in spf_control["rdm_terms"]]:
            if c not in mc_tmp.columns:
                mc_tmp[c] = 0.0
        result['expected_control'] = predict_metacount(mc_tmp, spf_control)
        result['eb_control'], result['eb_weight_control'] = zip(*result.apply(
            lambda row: eb_adjustment(row['control_total'], row['expected_control'], alpha_control), axis=1
        ))
    else:
        result['expected_control'] = 0
        result['eb_control'] = result['control_total']
        result['eb_weight_control'] = 1.0
    
    # Compute EB-adjusted uplift
    result['eb_uplift'] = result['eb_event'] - result['eb_control']
    
    # Rate per 1000 trips
    result['event_rate_per_1k'] = np.where(
        result['event_trips'] > 0,
        1000.0 * result['event_total'] / result['event_trips'],
        0
    )
    result['control_rate_per_1k'] = np.where(
        result['control_trips'] > 0,
        1000.0 * result['control_total'] / result['control_trips'],
        0
    )
    result['eb_event_rate_per_1k'] = np.where(
        result['event_trips'] > 0,
        1000.0 * result['eb_event'] / result['event_trips'],
        0
    )
    
    return result


def generate_summary_report(df, eb_df, output_path):
    """Generate markdown summary for Police paper."""
    
    with open(output_path, 'w') as f:
        f.write("# EB-Adjusted Street Risk Analysis\n\n")
        f.write("## Summary Statistics\n\n")
        
        f.write(f"- **Total road segments analyzed:** {len(eb_df):,}\n")
        f.write(f"- **Segments with event data:** {(eb_df['event_trips'] > 0).sum():,}\n")
        f.write(f"- **Segments with control data:** {(eb_df['control_trips'] > 0).sum():,}\n\n")
        
        f.write("### Event vs Control (Raw Counts)\n\n")
        f.write(f"- Total event braking: {eb_df['event_braking'].sum():,.0f}\n")
        f.write(f"- Total event speeding: {eb_df['event_speeding'].sum():,.0f}\n")
        f.write(f"- Total control braking: {eb_df['control_braking'].sum():,.0f}\n")
        f.write(f"- Total control speeding: {eb_df['control_speeding'].sum():,.0f}\n\n")
        
        f.write("### EB-Adjusted Estimates\n\n")
        f.write(f"- Total EB event risk: {eb_df['eb_event'].sum():,.1f}\n")
        f.write(f"- Total EB control risk: {eb_df['eb_control'].sum():,.1f}\n")
        f.write(f"- Total EB uplift (event - control): {eb_df['eb_uplift'].sum():,.1f}\n")
        f.write(f"- Mean EB weight (event): {eb_df['eb_weight_event'].mean():.3f}\n")
        f.write(f"- Mean EB weight (control): {eb_df['eb_weight_control'].mean():.3f}\n\n")
        
        f.write("### Top 20 Segments by EB-Adjusted Event Risk\n\n")
        f.write("| Rank | Road OSM ID | Road Name | EB Event Risk | Event Trips | EB Uplift |\n")
        f.write("|------|-------------|-----------|---------------|-------------|----------|\n")
        
        # Use road_name already in eb_df
        eb_ranked = eb_df.copy()
        if 'road_name' in eb_ranked.columns:
            eb_ranked['road_name_display'] = eb_ranked['road_name'].fillna('(unnamed)').astype(str)
        else:
            eb_ranked['road_name_display'] = '(metadata not available)'
        
        top20 = eb_ranked.nlargest(20, 'eb_event')
        
        for i, (_, row) in enumerate(top20.iterrows(), 1):
            f.write(f"| {i} | {int(row['road_osmid'])} | {row['road_name_display'][:30]} | "
                   f"{row['eb_event']:.1f} | {int(row['event_trips'])} | {row['eb_uplift']:.1f} |\n")
        
        f.write("\n### Interpretation\n\n")
        f.write("The EB-adjusted risk scores account for exposure (trip counts) and regression-to-the-mean bias. "
               "Segments with low exposure receive more weight on the model prediction, while high-exposure segments "
               "rely more on observed counts. The EB uplift (event - control) identifies segments where risk is "
               "specifically elevated on event days relative to matched non-event days.\n\n")
        
        f.write("**For the Police paper:** Use `eb_event` for overall event-day deployment ranking, and `eb_uplift` "
               "to identify event-specific corridors that warrant incremental staffing.\n")
    
    print(f"  Saved: {output_path}")


def plot_eb_comparison(eb_df, output_path):
    """Plot Event vs Control EB-adjusted risk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter to segments with both event and control data
    both = eb_df[(eb_df['event_trips'] > 0) & (eb_df['control_trips'] > 0)].copy()
    
    # Left: Raw counts
    ax = axes[0]
    ax.scatter(both['control_total'], both['event_total'], alpha=0.4, s=20)
    lim = max(both['event_total'].max(), both['control_total'].max())
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Control Total Risk (raw)', fontsize=11)
    ax.set_ylabel('Event Total Risk (raw)', fontsize=11)
    ax.set_title('Raw Counts: Event vs Control', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: EB-adjusted
    ax = axes[1]
    ax.scatter(both['eb_control'], both['eb_event'], alpha=0.4, s=20)
    lim = max(both['eb_event'].max(), both['eb_control'].max())
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Control EB Risk', fontsize=11)
    ax.set_ylabel('Event EB Risk', fontsize=11)
    ax.set_title('EB-Adjusted: Event vs Control', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_eb_uplift_distribution(eb_df, output_path):
    """Plot distribution of EB uplift."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    uplift = eb_df[eb_df['eb_uplift'] != 0]['eb_uplift']
    
    ax.hist(uplift, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero uplift')
    ax.set_xlabel('EB Uplift (Event - Control)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of EB-Adjusted Uplift', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("20_eb_adjusted_street_risk.py - EB-Adjusted Street Risk for Police Paper")
    print("=" * 80)
    
    # Load segment roads data
    print("\n1. Loading segment roads data...")
    df = pd.read_csv(SEGMENT_ROADS)
    print(f"   Loaded {len(df):,} segment-phase observations")
    print(f"   Unique segments: {df['road_osmid'].nunique():,}")
    print(f"   Event observations: {(df['type'] == 'Event').sum():,}")
    print(f"   Control observations: {(df['type'] == 'Control').sum():,}")
    
    # Fit NB random-parameters count models via metacountregressor
    print("\n2. Fitting MetaCount NB random-parameters models...")
    print("   Event model:")
    spf_event, event_agg = fit_spf_by_type(df, 'Event')
    if spf_event:
        print(f"     AIC: {spf_event['aic']:.1f}, Alpha: {spf_event['alpha']:.4f}")
    
    print("   Control model:")
    spf_control, control_agg = fit_spf_by_type(df, 'Control')
    if spf_control:
        print(f"     AIC: {spf_control['aic']:.1f}, Alpha: {spf_control['alpha']:.4f}")
    
    # Compute EB-adjusted risk scores
    print("\n3. Computing EB-adjusted risk scores...")
    eb_df = compute_eb_risk_scores(df, spf_event, spf_control)
    print(f"   Computed EB scores for {len(eb_df):,} road segments")
    
    # Merge with road metadata (DATA first, then RES for backwards compat)
    print("\n4. Merging with road metadata...")
    _meta_candidates = [os.path.join(DATA, "road_meta.csv"), ROAD_META]
    _meta_path = next((p for p in _meta_candidates if os.path.exists(p)), None)
    if _meta_path:
        meta = pd.read_csv(_meta_path)
        # meta has columns osmid,name,... - keep SPF highway intact, only add display name
        name_col = "name" if "name" in meta.columns else "road_name"
        eb_df = eb_df.merge(meta[["osmid", name_col]], left_on="road_osmid",
                            right_on="osmid", how="left")
        eb_df["road_name"] = eb_df[name_col].fillna("")
        print(f"   Merged metadata for {len(eb_df):,} segments")
    else:
        eb_df["road_name"] = ""
        # FIX: do NOT wipe highway here - SPF predict already ran, but wiping
        # breaks downstream joins/plots and caused the empty-Road-Name table.
        print("   Warning: road_meta.csv not found")

    # Geometry features are already attached to eb_df via
    # compute_eb_risk_scores() (same standardised table as the SPF fit).
    
    # Save results
    print("\n5. Saving EB-adjusted risk scores...")
    output_cols = ['road_osmid', 'road_name', 'highway', 
                   'event_braking', 'event_speeding', 'event_trips', 'event_total',
                   'control_braking', 'control_speeding', 'control_trips', 'control_total',
                   'expected_event', 'eb_event', 'eb_weight_event',
                   'expected_control', 'eb_control', 'eb_weight_control',
                   'eb_uplift', 'event_rate_per_1k', 'control_rate_per_1k', 'eb_event_rate_per_1k']
    geo_cols = [c for c in ['curvature_deg_km', 'sinuosity', 'maxspeed_kmh', 'lanes',
                            'oneway', 'is_link', 'road_class_rank', 'node_degree',
                            'dist_to_intersection_m', 'length_m']
                if c in eb_df.columns]
    output_cols = [c for c in output_cols if c in eb_df.columns] + geo_cols
    
    eb_df[output_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved: {OUTPUT_CSV}")
    
    # Generate summary report
    print("\n6. Generating summary report...")
    generate_summary_report(df, eb_df, OUTPUT_MD)
    
    # Generate figures
    print("\n7. Generating figures...")
    plot_eb_comparison(eb_df, os.path.join(FIG, "eb_street_risk_comparison.png"))
    plot_eb_uplift_distribution(eb_df, os.path.join(FIG, "eb_street_risk_uplift_dist.png"))
    
    print("\n" + "=" * 80)
    print("DONE! EB-adjusted street risk analysis complete.")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_MD}")
    print(f"  - {os.path.join(FIG, 'eb_street_risk_*.png')}")
    print("\nKey Results:")
    print(f"  - {len(eb_df):,} road segments analyzed")
    print(f"  - Total EB event risk: {eb_df['eb_event'].sum():,.1f}")
    print(f"  - Total EB uplift: {eb_df['eb_uplift'].sum():,.1f}")
    print(f"  - Mean EB weight (event): {eb_df['eb_weight_event'].mean():.3f}")


if __name__ == "__main__":
    main()
