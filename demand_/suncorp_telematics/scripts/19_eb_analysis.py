"""
19_eb_analysis.py
Empirical Bayes analysis of near-miss hard braking events.

Count model is a Negative-Binomial random-parameters regression fitted
with metacountregressor (NOT a statsmodels SPF):
    Y_i ~ NegBinom(lambda_i, alpha)
    log lambda_i = b0 + b1*log(E_i+1) + X_fixed*b + X_rdm*b_i
    b_i,k ~ Normal(mean_k, sd_k^2)

EB step is unchanged: w = 1/(1+alpha*lambda), EB = w*lambda + (1-w)*Y.

Outputs:
  - results/eb_results.tex
  - results/eb_spf_coef.csv
  - results/eb_predictions.csv
  - figures/eb_*.png
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb, predict_metacount

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)

INPUT = os.path.join(DATA, "near_miss_hb_ecf.csv")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")
OUTPUT_TEX = os.path.join(RES, "eb_results.tex")
OUTPUT_COEF = os.path.join(RES, "eb_spf_coef.csv")
OUTPUT_PRED = os.path.join(RES, "eb_predictions.csv")

GEO_NUMERIC = [
    "curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
    "dist_to_intersection_m", "node_degree", "length_m",
]


def _standardise(df):
    for c in GEO_NUMERIC:
        if c in df.columns:
            sd = df[c].std()
            df[c + "_z"] = (df[c] - df[c].mean()) / (sd if sd and sd > 0 else 1.0)
    return df


def load_with_geom_features(path):
    df = pd.read_csv(path)
    if "curvature_deg_km" in df.columns:
        print("   Geometry features already present in table")
    elif os.path.exists(FEATURES):
        feat = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        df = df.merge(feat, on="road_osmid", how="left")
        print(f"   Merged geometry features from {os.path.basename(FEATURES)}")
    else:
        print("   WARNING: no geometry features available -> exposure-only")
    for c in ["curvature_deg_km", "sinuosity", "is_link", "oneway",
              "road_class_rank", "node_degree", "length_m"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    if "highway" in df.columns:
        df["highway"] = df["highway"].fillna("unknown").astype(str)
    df = _standardise(df)
    return df


def build_terms(df):
    """Fixed + random term lists for the NB random-parameters model."""
    mc, dummies = prepare_metacount_frame(df, exposure_col="exposure_trips",
                                          highway_col="highway")
    fixed = ["logexp"] + dummies
    for c in ["maxspeed_kmh_z", "curvature_deg_km_z"]:
        if c in mc.columns:
            fixed.append(c)
    rdm = []
    for c in ["lanes_z", "dist_to_intersection_m_z"]:
        if c in mc.columns:
            rdm.append(f"{c}:normal")
    # if geometry missing entirely, fall back to exposure-only
    if not rdm and "logexp" in mc.columns:
        pass
    return mc, fixed, rdm


def fit_count_model(mc_df, fixed, rdm, y_col="total_hb", R=200, verbose=True):
    """Fit NB random-parameters model via metacountregressor."""
    print(f"\nFitting MetaCount NB random-parameters: fixed={fixed} rdm={rdm}")
    info = fit_metacount_nb(mc_df, y_col=y_col, id_col="road_osmid",
                            fixed_terms=fixed, rdm_terms=rdm,
                            model="nb", R=R, de_maxiter=8,
                            de_popsize=8, verbose=verbose)
    print(f"  spec: {info['spec_label']}, AIC: {info['aic']:.1f}, "
          f"BIC: {info['bic']:.1f}, alpha: {info['alpha']:.4f}")
    return info


def eb_adjustment(observed, expected, phi):
    weight = 1.0 / (1.0 + phi * expected)
    eb_estimate = weight * expected + (1.0 - weight) * observed
    return eb_estimate, weight


def cross_validate_spf(df, fixed, rdm, y_col="total_hb", n_folds=5):
    """K-fold CV with the MetaCount NB random-parameters model (R=100)."""
    print(f"\nCross-validating MetaCount NB with {n_folds} folds...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        try:
            tr = df.iloc[train_idx].reset_index(drop=True)
            te = df.iloc[test_idx].reset_index(drop=True)
            # rebuild fixed/rdm on train dummies (same reference).
            # NOTE: prepare_metacount_frame returns 2 items (frame, dummy_cols),
            # so unpack accordingly (3-unpack here broke every fold in 25218912).
            mc_tr2, fixed_tr, rdm_tr = build_terms(tr)
            mc_te2, _ = prepare_metacount_frame(te, exposure_col="exposure_trips",
                                                highway_col="highway")
            # align dummy columns
            for c in fixed_tr:
                if c not in mc_te2.columns and c != "logexp":
                    mc_te2[c] = 0.0
            info = fit_metacount_nb(mc_tr2, y_col=y_col, id_col="road_osmid",
                                    fixed_terms=fixed_tr, rdm_terms=rdm_tr,
                                    model="nb", R=100, de_maxiter=6,
                                    de_popsize=6, seed=fold, verbose=False)
            y_test = te[y_col].values.astype(float)
            # align test frame to train columns for prediction
            te_aligned = mc_te2.copy()
            for c in mc_tr2.columns:
                if c not in te_aligned.columns:
                    te_aligned[c] = 0.0
            y_pred = predict_metacount(te_aligned, info)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            fold_results.append({'fold': fold, 'rmse': rmse, 'mae': mae,
                                 'n_train': len(tr), 'n_test': len(te)})
            print(f"  Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}")
        except Exception as e:
            print(f"  Fold {fold}: Error - {e}")
            continue
    cv_df = pd.DataFrame(fold_results)
    if len(cv_df):
        print(f"\nCV Summary: RMSE={cv_df['rmse'].mean():.2f} +/- {cv_df['rmse'].std():.2f}, "
              f"MAE={cv_df['mae'].mean():.2f} +/- {cv_df['mae'].std():.2f}")
    return cv_df


def plot_eb_adjustment(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(df['expected_hb'], df['total_hb'], alpha=0.3, s=10)
    lim = max(df['expected_hb'].max(), df['total_hb'].max())
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Expected HB (MetaCount NB-RP)', fontsize=11)
    ax.set_ylabel('Observed HB', fontsize=11)
    ax.set_title('Observed vs Expected', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.scatter(df['eb_hb'], df['total_hb'], alpha=0.3, s=10)
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('EB-Adjusted HB', fontsize=11)
    ax.set_ylabel('Observed HB', fontsize=11)
    ax.set_title('Observed vs EB-Adjusted', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_eb_weights(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['exposure_trips'], df['eb_weight'], alpha=0.4, s=10)
    ax.set_xlabel('Exposure (trips)', fontsize=11)
    ax.set_ylabel('EB Weight', fontsize=11)
    ax.set_title('Empirical Bayes Weight vs Exposure', fontsize=12, fontweight='bold')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_severity_distribution(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_events = df[df['total_hb'] > 0].copy()
    severity_cols = ['hb_candidate', 'hb_critical', 'hb_extreme']
    severity_totals = df[severity_cols].sum()
    ax = axes[0]
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    ax.bar(['Candidate', 'Critical', 'Extreme'], severity_totals, color=colors)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Hard Braking Events by Severity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(severity_totals):
        ax.text(i, v + severity_totals.max() * 0.02, f'{int(v):,}',
                ha='center', va='bottom', fontsize=9)
    ax = axes[1]
    ax.hist(df_events['total_hb'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Hard Braking Events per Segment', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of HB Events (segments with HB > 0)', fontsize=12, fontweight='bold')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def generate_latex_tables(df, fit_info, cv_results, output_path):
    coef = fit_info["coef"]
    with open(output_path, 'w') as f:
        f.write("% eb_results.tex - Empirical Bayes Analysis Results\n")
        f.write("% Auto-generated by 19_eb_analysis.py (MetaCount NB random-parameters)\n\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Near-Miss Hard Braking Dataset Summary}\n\\label{tab:eb_data}\n")
        f.write("\\begin{tabular}{lr}\n\\hline\nMetric & Value \\\\\n\\hline\n")
        f.write(f"Total road segments & {len(df):,} \\\\\n")
        f.write(f"Segments with HB events & {(df['total_hb'] > 0).sum():,} \\\\\n")
        f.write(f"Total HB events & {df['total_hb'].sum():,} \\\\\n")
        f.write(f"\\quad Candidate (Low) & {df['hb_candidate'].sum():,} \\\\\n")
        f.write(f"\\quad Critical-Likely & {df['hb_critical'].sum():,} \\\\\n")
        f.write(f"\\quad Extreme & {df['hb_extreme'].sum():,} \\\\\n")
        f.write(f"Mean HB per segment & {df['total_hb'].mean():.1f} \\\\\n")
        f.write(f"Median HB per segment & {df['total_hb'].median():.0f} \\\\\n")
        f.write(f"Total exposure (trips) & {df['exposure_trips'].sum():,} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Negative-Binomial Random-Parameters Count Model (metacountregressor)}\n")
        f.write("\\label{tab:eb_spf}\n\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Variable & Coef. & Std Err & z & P$>|$z \\\\\n\\hline\n")
        for _, r in coef.iterrows():
            if str(r['variable']).lower() == 'dispersion':
                continue
            param_clean = str(r['variable']).replace('_', '\\_')
            se = r['std_err'] if np.isfinite(r['std_err']) else 0.0
            z = r['z_value'] if np.isfinite(r['z_value']) else 0.0
            p = r['p_value'] if np.isfinite(r['p_value']) else 1.0
            _c = r.get('coefficient', r.get('coef_mle', float('nan')))
            _c = float(_c) if np.isfinite(float(_c)) else 0.0
            f.write(f"{param_clean} & {_c:.4f} & {se:.4f} & {z:.2f} & {p:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"AIC & \\multicolumn{{4}}{{l}}{{{fit_info['aic']:.1f}}} \\\\\n")
        f.write(f"BIC & \\multicolumn{{4}}{{l}}{{{fit_info['bic']:.1f}}} \\\\\n")
        # FIX point-2: report the NB dispersion actually used for EB (not scale=1)
        f.write(f"$\\alpha$ (dispersion) & \\multicolumn{{4}}{{l}}{{{fit_info['alpha']:.4f}}} \\\\\n")
        f.write(f"spec & \\multicolumn{{4}}{{l}}{{{fit_info['spec_label']}}} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Empirical Bayes Adjustment Summary}\n\\label{tab:eb_adjust}\n")
        f.write("\\begin{tabular}{lrr}\n\\hline\nMetric & Before EB & After EB \\\\\n\\hline\n")
        total_obs = df['total_hb'].sum()
        total_eb = df['eb_hb'].sum()
        f.write(f"Total HB events & {total_obs:,.0f} & {total_eb:,.1f} \\\\\n")
        f.write(f"Mean per segment & {df['total_hb'].mean():.2f} & {df['eb_hb'].mean():.2f} \\\\\n")
        f.write(f"Std dev per segment & {df['total_hb'].std():.2f} & {df['eb_hb'].std():.2f} \\\\\n")
        f.write(f"Mean EB weight & \\multicolumn{{2}}{{r}}{{{df['eb_weight'].mean():.3f}}} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        if cv_results is not None and len(cv_results) > 0:
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{5-Fold Cross-Validation Results (MetaCount NB-RP)}\n\\label{tab:eb_cv}\n")
            f.write("\\begin{tabular}{lrr}\n\\hline\nFold & RMSE & MAE \\\\\n\\hline\n")
            for _, row in cv_results.iterrows():
                f.write(f"{int(row['fold'])} & {row['rmse']:.2f} & {row['mae']:.2f} \\\\\n")
            f.write("\\hline\n")
            f.write(f"Mean & {cv_results['rmse'].mean():.2f} & {cv_results['mae'].mean():.2f} \\\\\n")
            f.write(f"Std Dev & {cv_results['rmse'].std():.2f} & {cv_results['mae'].std():.2f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("19_eb_analysis.py - EB with MetaCount NB random-parameters")
    print("=" * 80)
    print("\n1. Loading near_miss_hb_ecf.csv...")
    df = load_with_geom_features(INPUT)
    print(f"   Loaded {len(df):,} road segments")
    print(f"   Total HB events: {df['total_hb'].sum():,}")
    print(f"   Segments with HB > 0: {(df['total_hb'] > 0).sum():,}")
    df = df[df['exposure_trips'] > 0].copy()
    print(f"   Filtered to {len(df):,} segments with exposure > 0")

    print("\n2. Fitting MetaCount NB random-parameters count model...")
    mc, fixed, rdm = build_terms(df)
    # align df index with mc (prepare adds logexp/dummies)
    df = df.reset_index(drop=True)
    mc = mc.reset_index(drop=True)
    fit_info = fit_count_model(mc, fixed, rdm, y_col="total_hb", R=200)

    print("\n3. Computing expected HB counts...")
    df['expected_hb'] = fit_info["expected"]
    alpha = fit_info["alpha"]
    print(f"   Overdispersion parameter (alpha): {alpha:.4f}")

    print("\n4. Computing Empirical Bayes adjustments...")
    w = 1.0 / (1.0 + alpha * df['expected_hb'].values)
    df['eb_weight'] = w
    df['eb_hb'] = w * df['expected_hb'].values + (1.0 - w) * df['total_hb'].values
    print(f"   Mean EB weight: {df['eb_weight'].mean():.3f}")
    print(f"   Total observed HB: {df['total_hb'].sum():,.0f}")
    print(f"   Total expected HB: {df['expected_hb'].sum():,.1f}")
    print(f"   Total EB-adjusted HB: {df['eb_hb'].sum():,.1f}")

    print("\n5. Running cross-validation...")
    cv_results = cross_validate_spf(df, fixed, rdm, n_folds=5)

    print("\n6. Generating figures...")
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plot_eb_adjustment(df, os.path.join(FIG, "eb_obs_vs_pred.png"))
    plot_eb_weights(df, os.path.join(FIG, "eb_weights.png"))
    plot_severity_distribution(df, os.path.join(FIG, "eb_severity_dist.png"))

    print("\n7. Generating LaTeX tables...")
    generate_latex_tables(df, fit_info, cv_results, OUTPUT_TEX)

    print("\n8. Saving EB predictions...")
    extra = [c for c in ['curvature_deg_km', 'maxspeed_kmh', 'lanes',
                         'dist_to_intersection_m', 'node_degree']
             if c in df.columns]
    output_cols = ['road_osmid', 'road_name', 'highway', 'exposure_trips',
                   'total_hb', 'expected_hb', 'eb_hb', 'eb_weight'] + extra
    df[output_cols].to_csv(OUTPUT_PRED, index=False)
    print(f"   Saved: {OUTPUT_PRED}")

    print("\n9. Saving coefficients...")
    try:
        coef_df = pd.DataFrame({
            'variable': fit_info["coef"]['variable'],
            'coefficient': fit_info["coef"]['coefficient'],
            'std_err': fit_info["coef"]['std_err'],
            'z_value': fit_info["coef"]['z_value'],
            'p_value': fit_info["coef"]['p_value'],
        })
        coef_df.to_csv(OUTPUT_COEF, index=False)
        print(f"   Saved: {OUTPUT_COEF}")
    except Exception as e:
        print(f"   WARNING: coefficient export failed ({e}); core outputs already saved.")

    print("\n" + "=" * 80)
    print("DONE! Empirical Bayes analysis complete.")
    print("=" * 80)
    print(f"\nOutputs:\n  - {OUTPUT_TEX}\n  - {OUTPUT_COEF}\n  - {OUTPUT_PRED}")
    print(f"\nKey Results:\n  - {len(df):,} road segments analyzed")
    print(f"  - {df['total_hb'].sum():,} total hard braking events")
    print(f"  - MetaCount {fit_info['spec_label']} AIC: {fit_info['aic']:.1f}, BIC: {fit_info['bic']:.1f}")
    print(f"  - Mean EB weight: {df['eb_weight'].mean():.3f}")
    if len(cv_results):
        print(f"  - CV RMSE: {cv_results['rmse'].mean():.2f} +/- {cv_results['rmse'].std():.2f}")


if __name__ == "__main__":
    main()
