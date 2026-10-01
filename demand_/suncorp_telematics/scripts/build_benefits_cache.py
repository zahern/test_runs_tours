"""One-time: build the P4 network + junction benefit components and cache them
so downstream analyses (K-sweep, phased selection, weight sensitivity) skip the
slow betweenness-centrality computation."""
import numpy as np, pandas as pd, geopandas as gpd
_src = open("scripts/28_post_selection_mip.py").read().replace(
    'if __name__ == "__main__":\n    main()', '')
p4 = {}
exec(compile(_src, "28.py", "exec"), p4)


class A:
    min_len_m = 100.0; bsamples = 500; decay_m = 5000.0
    stadium_lat = p4["STADIUM_LAT"]; stadium_lon = p4["STADIUM_LON"]


segs = gpd.read_file("data/osm_roads_suncorp.geojson")
ecf = pd.read_csv("data/near_miss_hb_ecf.csv")
G, xy, ll, inc, sm = p4["build_network"](segs, segs.to_crs(28356), ecf)
cand, clon, clat, dst, S0, risk, flow, deg, betw = p4["junction_benefits"](G, ll, inc, sm, A)
np.savez_compressed("results/benefits_cache.npz",
                    cand=cand, clon=clon, clat=clat, dst=dst, S0=S0,
                    risk=risk, flow=flow, deg=deg, betw=betw,
                    xy=xy[cand])
print(f"cached {len(cand)} candidate junctions")
