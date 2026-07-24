"""Fill missing population for DTA zones using Paper_Plan per-zone synthetic pop data."""
import os, glob
import pandas as pd
import numpy as np

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
PAPER_ZONES_DIR = r"Z:\Paper_Plan\synth_pop\zones_sa_0"
SA2_TO_ORDER = os.path.join(DEMAND_DIR, "sa2_to_order_crosswalk.csv")
DEMO_AGG = os.path.join(DEMAND_DIR, "synth_pop_demographics_agg.csv")

# Known zones with zero population in DTA
ZERO_POP_ZONES = [
    '307021179', '307031184', '307031185', '307031186',
    '309081260', '314011385',
    '317011445', '317011446', '317011448', '317011449', '317011450',
    '317011451', '317011452', '317011454', '317011455', '317011457', '317011459',
    '319021509', '319031511', '319031512', '319031513', '319031514', '319031515', '319051525',
]

# Load SA2 -> ORDER crosswalk
xw = pd.read_csv(SA2_TO_ORDER)
xw["sa2_code"] = xw["sa2_code"].astype(str)
xw["order"] = xw["order"].astype(int)

# Build order -> SA2 mapping (many SA2s may map to same order)
order_to_sa2s = {}
for _, row in xw.iterrows():
    o = row["order"]
    if o not in order_to_sa2s:
        order_to_sa2s[o] = []
    order_to_sa2s[o].append(row["sa2_code"])

# Check per-zone synthetic population files
zone_files = glob.glob(os.path.join(PAPER_ZONES_DIR, "zone_*_synthetic_population.csv"))
print(f"Found {len(zone_files)} per-zone synthetic population files")

order_pops = {}
for zf in zone_files:
    fname = os.path.basename(zf)
    # Extract ORDER number from filename: zone_<ORDER>_synthetic_population.csv
    try:
        order_num = int(fname.split("_")[1])
    except (IndexError, ValueError):
        continue
    df = pd.read_csv(zf)
    order_pops[order_num] = len(df)

print(f"Loaded population from {len(order_pops)} ORDER zones")
print(f"Population range: {min(order_pops.values())} - {max(order_pops.values())}")

# Now map missing SA2 zones to ORDER population
filled = {}
for code in ZERO_POP_ZONES:
    row = xw[xw["sa2_code"] == code]
    if len(row) > 0:
        order = int(row.iloc[0]["order"])
        if order in order_pops:
            filled[code] = order_pops[order]
        else:
            # Check via 2016 concordance
            conc_path = os.path.join(DEMAND_DIR, "sa2_2016_to_2021_concordance.csv")
            if os.path.exists(conc_path):
                conc = pd.read_csv(conc_path)
                conc["sa2_code21"] = conc["sa2_code21"].astype(str)
                conc_row = conc[conc["sa2_code21"] == code]
                if len(conc_row) > 0:
                    sa2_16 = int(conc_row.iloc[0]["sa2_main16"])
                    # Find ORDER for 2016 code
                    xw_row = xw[xw["sa2_code"] == str(sa2_16)]
                    if len(xw_row) > 0:
                        order = int(xw_row.iloc[0]["order"])
                        if order in order_pops:
                            filled[code] = order_pops[order]
    if code not in filled:
        print(f"  {code}: NOT found in any pop source")

print(f"\nFilled {len(filled)} of {len(ZERO_POP_ZONES)} zero-pop zones:")
for code, pop in sorted(filled.items()):
    print(f"  {code}: {pop:,} persons")

# Check which still missing
still_missing = set(ZERO_POP_ZONES) - set(filled.keys())
if still_missing:
    print(f"\nStill missing ({len(still_missing)}): {sorted(still_missing)}")
    # Use nearest neighbor population as fallback
    print("\nUsing nearest SA2 population for missing...")
    # Get all SA2 centroids
    sa2_crosswalk = pd.read_csv(os.path.join(DEMAND_DIR, "sa2_crosswalk_507.csv"))
    sa2_crosswalk["sa2_code"] = sa2_crosswalk["sa2_code"].astype(int)
    lat_lon = sa2_crosswalk.set_index("sa2_code")[["sa2_lat", "sa2_lon"]]
    
    for code in still_missing:
        icode = int(code)
        if icode in lat_lon.index:
            lat0, lon0 = lat_lon.loc[icode]
        else:
            continue
        # Find nearest SA2 that has population
        best_dist = float("inf")
        best_pop = 0
        for order, pop in order_pops.items():
            sa2s = order_to_sa2s.get(order, [])
            for sa2 in sa2s:
                isa2 = int(sa2)
                if isa2 in lat_lon.index:
                    lat1, lon1 = lat_lon.loc[isa2]
                    dlat = lat1 - lat0
                    dlon = lon1 - lon0
                    dist = dlat*dlat + dlon*dlon
                    if dist < best_dist:
                        best_dist = dist
                        best_pop = pop
        filled[code] = best_pop
        print(f"  {code}: nearest={best_pop:,} persons")

# Save fill results
out_csv = os.path.join(DEMAND_DIR, "zero_pop_fill_from_paper_plan.csv")
fill_df = pd.DataFrame({"sa2_code": list(filled.keys()), "paper_pop": list(filled.values())})
fill_df.to_csv(out_csv, index=False)
print(f"\nSaved fill data: {out_csv}")
print(f"Total filled: {len(filled)}, Total population added: {sum(filled.values()):,}")
