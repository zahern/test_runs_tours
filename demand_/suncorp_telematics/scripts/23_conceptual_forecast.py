"""
23_conceptual_forecast.py
Conceptual forward model for predicting the time-based risk profile of a
*future* Suncorp event that is not yet in the data.

Idea
----
The timewise EB pipeline (21) already gives a phase-resolved, exposure-adjusted
baseline risk for ordinary (Control) days and for event days. The event-day
surge is well described by a single multiplicative "event factor"

    f_e = total_event_risk_e / baseline_control_risk

which varies from fixture to fixture because of kickoff time, day of week,
month and expected exposure. We regress log(f_e) on those calendar/exposure
features and obtain a closed-form predictor:

    log f_future = b0 + b1*kickoff_hour + b2*dow + b3*month + b4*log(exposure)

Given a future fixture's kickoff datetime and expected exposure, we predict
its event factor and scale the exposure-adjusted Control baseline to obtain
the predicted phase-resolved risk profile for every segment. This is the
conceptual model that supports prospective (future) predictions.

Outputs
    results/conceptual_forecast.csv        predicted phase profile (sample future event)
    results/conceptual_forecast_coef.csv    regression coefficients
    results/conceptual_forecast.tex         equations + coefficients for the paper
    figures/conceptual_forecast_profile.png predicted 24h profile vs historical

Usage
    python 23_conceptual_forecast.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
SEG_PHASE = os.path.join(RES, "eb_timewise_segment_phase.csv")
OUT_CSV = os.path.join(RES, "conceptual_forecast.csv")
OUT_COEF = os.path.join(RES, "conceptual_forecast_coef.csv")
OUT_TEX = os.path.join(RES, "conceptual_forecast.tex")
OUT_TABLE = os.path.join(RES, "conceptual_table.tex")
FIG_PROFILE = os.path.join(FIG, "conceptual_forecast_profile.png")

PHASES = ["pre", "during", "post"]
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def main():
    print("=" * 80)
    print("23_conceptual_forecast.py - conceptual forward model")
    print("=" * 80)

    print("\n1. Loading data...")
    df = pd.read_csv(SEGMENT_ROADS)
    df["Y"] = df["braking"] + df["speeding"]
    df["ref_dt"] = pd.to_datetime(df["ref_dt"])

    # Event-day totals and calendar features (matched control = same ref_id)
    print("\n2. Building event-level event factor (vs matched control)...")
    ev = df[df["type"] == "Event"].groupby("ref_id").agg(
        event_risk=("Y", "sum"),
        kickoff_h=("ref_dt", lambda s: (s.dt.hour + s.dt.minute / 60.0).iloc[0]),
        dow=("ref_dt", lambda s: s.dt.dayofweek.iloc[0]),
        month=("ref_dt", lambda s: s.dt.month.iloc[0]),
        exposure=("trips", "sum"),
    ).reset_index()
    ct = df[df["type"] == "Control"].groupby("ref_id").agg(
        control_risk=("Y", "sum"),
        control_exposure=("trips", "sum")).reset_index()
    ev = ev.merge(ct, on="ref_id", how="left")
    ev["control_risk"] = ev["control_risk"].fillna(
        df[df["type"] == "Control"]["Y"].sum() / df[df["type"] == "Control"]["ref_id"].nunique())
    ev["control_exposure"] = ev["control_exposure"].fillna(
        df[df["type"] == "Control"]["trips"].sum() / df[df["type"] == "Control"]["ref_id"].nunique())
    # Exposure-adjusted rate ratio (risk per trip, event vs matched control)
    ev["event_factor"] = (ev["event_risk"] / ev["exposure"]) / (
        ev["control_risk"] / ev["control_exposure"])
    ev["log_ef"] = np.log(ev["event_factor"])
    ev["log_exposure"] = np.log(ev["exposure"] + 1)

    # ---- 3. Regression of log(event factor) on calendar/exposure -----------
    print("\n3. Fitting conceptual model  log f = b0 + b1*hour + b2*dow "
          "+ b3*month + b4*log(exp)...")
    model = smf.ols(
        "log_ef ~ kickoff_h + dow + month + log_exposure", data=ev
    ).fit()
    print(model.summary())
    coef_df = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values,
    })
    coef_df.to_csv(OUT_COEF, index=False)
    print(f"   Wrote {OUT_COEF}")

    # ---- 4. Predict a sample future event -----------------------------------
    print("\n4. Forecasting a sample future fixture (Fri 19:00, month=3)...")
    future = pd.DataFrame([{
        "kickoff_h": 19.0, "dow": 4, "month": 3,
        "log_exposure": ev["log_exposure"].median(),
    }])
    pred_log_ef = model.predict(future).iloc[0]
    pred_factor = float(np.exp(pred_log_ef))
    print(f"   Predicted event factor = {pred_factor:.3f}")

    # Scale the average Control phase profile by the predicted factor
    ctrl_phase = pd.read_csv(SEG_PHASE)
    ctrl_phase = ctrl_phase[ctrl_phase["type"] == "Control"]
    # phase split of control baseline
    ctrl_by_phase = ctrl_phase.groupby(["road_osmid", "phase"])["eb"].sum().unstack().fillna(0)
    pred = ctrl_by_phase.mul(pred_factor).reset_index()
    pred.to_csv(OUT_CSV, index=False)
    print(f"   Wrote {OUT_CSV} ({len(pred):,} segments x phases)")

    # ---- 5. Figure: predicted 24h profile vs historical mean ----------------
    print("\n5. Profile figure...")
    ev_tot = df[df["type"] == "Event"].groupby("road_osmid")["Y"].sum()
    hist_mean = ev_tot.mean()
    pred_mean = pred[["pre", "during", "post"]].sum(axis=1).mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    ax.bar(x - 0.2, [ctrl_by_phase[p].mean() for p in PHASES], 0.4,
           label="Control baseline", color="#5b8fb0")
    ax.bar(x + 0.2, [pred[p].mean() for p in PHASES], 0.4,
           label=f"Predicted future event (f={pred_factor:.2f})", color="#e8633a")
    ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in PHASES])
    ax.set_ylabel("Mean EB risk per segment")
    ax.set_title("Conceptual forecast: future-event phase risk vs baseline")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG_PROFILE, dpi=150); plt.close(fig)
    print(f"   Saved {FIG_PROFILE}")

    # ---- 6. LaTeX (FIX point-5: old code printed "-0.0054\\beta_0 ..." ) ----
    b = {k: float(model.params[k]) for k in model.params.index}
    eq = ("\\log \\hat f_\\text{future} = "
          f"{b.get('Intercept', 0.0):+.4f} "
          f"{b.get('kickoff_h', 0.0):+.4f}\\,h "
          f"{b.get('dow', 0.0):+.4f}\\,d "
          f"{b.get('month', 0.0):+.4f}\\,m "
          f"{b.get('log_exposure', 0.0):+.4f}\\,\\log E")
    L = []
    L.append("% conceptual_forecast.tex - auto-generated by 23_conceptual_forecast.py")
    L.append("\\begin{equation}")
    L.append(eq)
    L.append("\\end{equation}")
    L.append("where $h$ is kickoff hour, $d$ day-of-week, $m$ month and "
             "$E$ expected exposure. The predicted phase-resolved risk is "
             "$\\hat m_{i,p}^{\\text{future}} = f_\\text{future}\\,\\hat m_{i,p}^{\\text{control}}$.")
    L.append("")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Conceptual forward-model coefficients "
             "(event factor regression).}")
    L.append("\\label{tab:concept}")
    L.append("\\begin{tabular}{lrrr}")
    L.append("\\hline")
    L.append("Term & Coef & Std err & p \\\\")
    L.append("\\hline")
    for _, r in coef_df.iterrows():
        L.append("{0} & {1:.4f} & {2:.4f} & {3:.4f} \\\\".format(
            r["variable"], r["coef"], r["std_err"], r["p_value"]))
    L.append("R$^2$ & \\multicolumn{3}{l}{" + "{0:.3f}".format(model.rsquared) + "} \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    print("   Wrote " + OUT_TEX)

    # Table-only fragment for \input from the paper (no duplicate prose)
    T = []
    T.append("\\begin{table}[h]")
    T.append("\\centering")
    T.append("\\caption{Conceptual forward-model coefficients (event-factor "
             "regression, 16 events).}")
    T.append("\\label{tab:concept}")
    T.append("\\begin{tabular}{lrrr}")
    T.append("\\hline")
    T.append("Term & Coef & Std err & $p$ \\\\")
    T.append("\\hline")
    for _, r in coef_df.iterrows():
        T.append("{0} & {1:.4f} & {2:.4f} & {3:.4f} \\\\".format(
            r["variable"], r["coef"], r["std_err"], r["p_value"]))
    T.append("R$^2$ & \\multicolumn{3}{l}{" + "{0:.3f}".format(model.rsquared) + "} \\\\")
    T.append("\\hline")
    T.append("\\end{tabular}")
    T.append("\\end{table}")
    with open(OUT_TABLE, "w") as f:
        f.write("\n".join(T) + "\n")
    print("   Wrote " + OUT_TABLE)
    print("\nDONE! Conceptual forecast complete.")


if __name__ == "__main__":
    main()
