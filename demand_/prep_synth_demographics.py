"""Fix ptype and re-aggregate demographics with all categories."""
import pandas as pd
import numpy as np

SYNTH_POP_CSV = r"Z:\test_runs_tours\new_pop\saved_data\synth_population.csv"
OUT_CSV = r"Z:\test_runs_tours\demand_\synth_pop_demographics_agg.csv"

COLS = ["SA2_MAIN16", "sexp", "agep", "age_group", "age5p",
        "WORKSTAT", "STUDYING", "CARLICENCE", "ptype",
        "income_band", "occp_group", "HHSIZE", "veh_bin"]

print("Reading synth_population.csv ...")
df = pd.read_csv(SYNTH_POP_CSV, usecols=[c for c in COLS if c in pd.read_csv(SYNTH_POP_CSV, nrows=1).columns])
print(f"Loaded {len(df):,} persons, columns: {list(df.columns)}")

# Check ptype values
if "ptype" in df.columns:
    print(f"\nptype value counts:")
    print(df["ptype"].value_counts(dropna=False).head(10))

# Map categories properly
sex_map = {1: "Male", 2: "Female"}
workstat_map = {1: "Employed", 2: "Unemployed", 0: "Not in labour force"}
car_map = {1.0: "Yes", 0.0: "No", 1: "Yes", 0: "No"}
study_map = {1.0: "Studying", 0.0: "Not studying", 1: "Studying", 0: "Not studying"}

print("\nMapping categories ...")
if "sexp" in df.columns:
    df["sex_label"] = df["sexp"].map(sex_map).fillna("Unknown")
if "WORKSTAT" in df.columns:
    df["workstat_label"] = df["WORKSTAT"].map(workstat_map).fillna("Unknown")
if "CARLICENCE" in df.columns:
    df["car_licence_label"] = df["CARLICENCE"].map(car_map).fillna("Unknown")
if "ptype" in df.columns:
    # ptype is already a string column with values like "FullTimeWorker", etc.
    df["ptype_label"] = df["ptype"].fillna("Unknown")
if "STUDYING" in df.columns:
    df["studying_label"] = df["STUDYING"].map(study_map).fillna("Unknown")

# Aggregate by SA2
print("\nAggregating by SA2 ...")
sa2_list = sorted(df["SA2_MAIN16"].unique())
agg_rows = []
all_cat_columns = []

for i, sa2 in enumerate(sa2_list):
    subset = df[df["SA2_MAIN16"] == sa2]
    total = len(subset)
    row = {"sa2_code": str(sa2), "total_persons": total}

    for col, labels in [
        ("sex_label", ["Male", "Female"]),
        ("workstat_label", ["Employed", "Unemployed", "Not in labour force"]),
        ("car_licence_label", ["Yes", "No"]),
        ("ptype_label", None),  # use actual values
        ("studying_label", ["Studying", "Not studying"]),
        ("income_band", None),
        ("age_group", None),
        ("veh_bin", None),
    ]:
        if col not in df.columns:
            continue
        if labels is None:
            labels = sorted(subset[col].dropna().unique())
        for label in labels:
            cnt = len(subset[subset[col] == label])
            key = f"{col.replace('_label', '')}_{str(label).replace(' ', '_').replace('/', '_')}"
            row[key] = cnt
            row[f"{key}_pct"] = round(100 * cnt / total, 1) if total else 0.0
            if f"{key}_pct" not in all_cat_columns:
                all_cat_columns.append(f"{key}_pct")

    agg_rows.append(row)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(sa2_list)} ...")

agg = pd.DataFrame(agg_rows).sort_values("total_persons", ascending=False).reset_index(drop=True)
agg.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"Zones: {len(agg)}, Columns: {len(agg.columns)}")
print(f"Demo pct columns: {len(all_cat_columns)}")

# Print sample ptype values
ptype_pct_cols = [c for c in agg.columns if c.startswith("ptype_") and c.endswith("_pct")]
print(f"ptype pct columns: {ptype_pct_cols}")
