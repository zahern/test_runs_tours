"""
05_crash_matrices.py
Load crash frequency matrices and prepare for OMX integration.
"""
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"Z:\test_runs_tours\demand_")
from omx_utils import write_omx

OUT_DIR = r"Z:\test_runs_tours\demand_"
CRASH_PATH = r"Z:\test_runs_tours\code\SA2_OD_Crash_Matrix.xlsx"

# Load 507 zone IDs
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = sorted([int(x) for x in re.split(r'[\s,]+', content.strip()) if x])
n_sa2 = len(zone_ids_507)
zone_ids_set = set(zone_ids_507)
sa2_to_idx = {code: i for i, code in enumerate(zone_ids_507)}
print(f"Loaded {n_sa2} SA2 zones")

# --- a) Read all 6 sheets ---
SHEETS = {
    'Run_off_Road': 'crash_runoff',
    'Head_on': 'crash_headon',
    'Rear_end': 'crash_rearend',
    'Intersection': 'crash_intersection',
    'Other': 'crash_other',
    'Total': 'crash_total',
}

sheet_dfs = {}
for sheet_name in SHEETS:
    df = pd.read_excel(CRASH_PATH, sheet_name=sheet_name)
    print(f"{sheet_name}: {len(df)} rows, columns: {df.columns.tolist()}")
    df['Origin_SA2'] = df['Origin_SA2'].astype(int)
    df['Destination_SA2'] = df['Destination_SA2'].astype(int)
    sheet_dfs[sheet_name] = df

total_pairs = len(sheet_dfs['Total'])
print(f"\nTotal OD pairs in crash data: {total_pairs:,}")

total_df = sheet_dfs['Total']
total_in_507 = total_df[
    total_df['Origin_SA2'].isin(zone_ids_set) &
    total_df['Destination_SA2'].isin(zone_ids_set)
]
print(f"OD pairs with both SA2s in 507: {len(total_in_507):,}")
print(f"Unique origins in 507: {total_in_507['Origin_SA2'].nunique()}")
print(f"Unique destinations in 507: {total_in_507['Destination_SA2'].nunique()}")

# --- b) Pivot each sheet to 507x507 matrix ---
def build_matrix(df, value_col='Exp'):
    mat = np.zeros((n_sa2, n_sa2), dtype=np.float32)
    filtered = df[
        df['Origin_SA2'].isin(zone_ids_set) &
        df['Destination_SA2'].isin(zone_ids_set)
    ].copy()
    for _, row in filtered.iterrows():
        o_idx = sa2_to_idx.get(int(row['Origin_SA2']), -1)
        d_idx = sa2_to_idx.get(int(row['Destination_SA2']), -1)
        if o_idx >= 0 and d_idx >= 0:
            val = row[value_col]
            if pd.notna(val):
                mat[o_idx, d_idx] = float(val)
    return mat

print("\nBuilding crash matrices...")
crash_matrices = {}
for sheet_name, mat_name in SHEETS.items():
    df = sheet_dfs[sheet_name]
    mat = build_matrix(df, 'Exp')
    crash_matrices[mat_name] = mat
    print(f"  {mat_name}: sum={mat.sum():.1f}, non-zero={np.count_nonzero(mat)}")

# --- c) Crash rate and length ---
length_mat = build_matrix(sheet_dfs['Total'], 'Length_km')
crash_rate_mat = np.where(
    length_mat > 0,
    crash_matrices['crash_total'] / length_mat,
    0.0
).astype(np.float32)
crash_matrices['crash_rate_total'] = crash_rate_mat
crash_matrices['crash_length_km'] = length_mat
print(f"  crash_rate_total: max={crash_rate_mat.max():.4f}, non-zero={np.count_nonzero(crash_rate_mat)}")

# --- d) Save crash_matrices_507.omx ---
out_omx = os.path.join(OUT_DIR, 'crash_matrices_507.omx')
if os.path.exists(out_omx):
    os.remove(out_omx)

zone_mapping = {str(code): i for i, code in enumerate(zone_ids_507)}
write_omx(out_omx, crash_matrices, zone_mapping=zone_mapping, shape=(n_sa2, n_sa2))
print(f"\nSaved {out_omx}")

# --- e) Save crash_od_507.csv ---
total_507 = sheet_dfs['Total'][
    sheet_dfs['Total']['Origin_SA2'].isin(zone_ids_set) &
    sheet_dfs['Total']['Destination_SA2'].isin(zone_ids_set)
].copy()
total_507 = total_507.rename(columns={'Exp': 'crash_total', 'Length_km': 'length_km'})

for sheet_name, mat_name in SHEETS.items():
    if sheet_name == 'Total':
        continue
    sub = sheet_dfs[sheet_name][
        sheet_dfs[sheet_name]['Origin_SA2'].isin(zone_ids_set) &
        sheet_dfs[sheet_name]['Destination_SA2'].isin(zone_ids_set)
    ][['Origin_SA2', 'Destination_SA2', 'Exp']].rename(columns={'Exp': mat_name})
    total_507 = total_507.merge(sub, on=['Origin_SA2', 'Destination_SA2'], how='left')

total_507 = total_507.rename(columns={'Origin_SA2': 'origin_sa2', 'Destination_SA2': 'dest_sa2'})

out_csv = os.path.join(OUT_DIR, 'crash_od_507.csv')
total_507.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(total_507):,} rows)")

# --- f) Print summary ---
print("\n=== CRASH MATRIX SUMMARY ===")
print(f"Total crash expected value: {crash_matrices['crash_total'].sum():.1f}")
print(f"Total OD pairs with crash data: {np.count_nonzero(crash_matrices['crash_total'])}")

top10 = total_507.nlargest(10, 'crash_total')
print(f"\nTop 10 most dangerous OD pairs:")
print(top10[['origin_sa2', 'dest_sa2', 'length_km', 'crash_total']].to_string(index=False))

print(f"\nCrash type breakdown:")
for sheet_name, mat_name in SHEETS.items():
    print(f"  {mat_name}: {crash_matrices[mat_name].sum():.1f}")
