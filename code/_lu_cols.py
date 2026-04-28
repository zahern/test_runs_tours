import pandas as pd, numpy as np, os

lu = pd.read_csv("zone_order_and_landuse.csv")
print("zone_order_and_landuse.csv COLUMNS (%d):" % len(lu.columns))
for c in sorted(lu.columns):
    print("  ", c)

z40 = lu[lu["ORDER"]==40].iloc[0]
edu_like = [c for c in lu.columns if any(k in c.lower() for k in
    ["edu","school","univ","tafe","tert","enrol","study","colleg",
     "postse","higher","hsenrol","adult","primar","second"])]
print("\nEdu-like columns in landuse:", edu_like)
for c in edu_like:
    print("  Zone 40", c, "=", z40.get(c, "N/A"))

# Also check pddopulation file
pop_file = "pddopulation_by_order_with_osm_and_landuse.csv"
if os.path.exists(pop_file):
    pop = pd.read_csv(pop_file, nrows=5)
    print("\n%s columns (%d):" % (pop_file, len(pop.columns)))
    for c in sorted(pop.columns):
        print("  ", c)
    edu_like2 = [c for c in pop.columns if any(k in c.lower() for k in
        ["edu","school","univ","tafe","tert","enrol","study","colleg",
         "postse","higher","hsenrol","adult","primar","second"])]
    print("\nEdu-like columns in pddopulation:", edu_like2)
    pop_full = pd.read_csv(pop_file)
    z40p = pop_full[pop_full["ORDER"]==40]
    if len(z40p):
        for c in edu_like2:
            print("  Zone 40", c, "=", z40p.iloc[0].get(c, "N/A"))

# Check SEQ_SA2 OSM files for education POI counts
for fname in ["SEQ_SA2_2016_OSM_counts.csv", "SEQ_SA2_2016_OSM_category_counts.csv"]:
    if os.path.exists(fname):
        osm = pd.read_csv(fname, nrows=5)
        print("\n%s columns (%d):" % (fname, len(osm.columns)))
        for c in sorted(osm.columns):
            print("  ", c)

print("\nDone.")
