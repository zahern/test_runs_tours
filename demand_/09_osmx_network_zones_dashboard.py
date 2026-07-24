r"""
09_osmx_network_zones_dashboard.py

OSMX map dashboard - DTA network SA2 zones linked to full synthetic
population (new_pop ABM). Shows zones actively modelled in the DTA
network (SEQ_Running_UE) with demographic breakdowns.

Usage:  python Z:/test_runs_tours/demand_/09_osmx_network_zones_dashboard.py
        Then open http://127.0.0.1:8053
"""
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────
DEMAND_DIR     = r"Z:\test_runs_tours\demand_"
PREP_DIR       = r"Z:\test_runs_tours\prepare_network"
NEW_POP_DIR    = r"Z:\test_runs_tours\new_pop\saved_data"
PAPER_POP_DIR  = r"Z:\Paper_Plan\synth_pop"
SA2_SHP        = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"

ZONE_SUMMARY   = os.path.join(DEMAND_DIR, "zone_summary_507.csv")
LANDUSE_CSV    = os.path.join(NEW_POP_DIR, "zone_landuse_507.csv")
LANDUSE_357_CSV = os.path.join(PAPER_POP_DIR, "zone_landuse_357.csv")
EXPANSION_CSV  = os.path.join(DEMAND_DIR, "planned_expansion_38_zones.csv")
DEMO_AGG_CSV   = os.path.join(DEMAND_DIR, "synth_pop_demographics_agg.csv")
FILL_CSV       = os.path.join(DEMAND_DIR, "zero_pop_fill_from_paper_plan.csv")
ACTIVE_SHP     = os.path.join(PREP_DIR, "new", "active_only", "zone_polygons.shp")
NOT_MOD_SHP    = os.path.join(PREP_DIR, "new", "not_modelled.shp")
SA2_TO_ORDER   = os.path.join(DEMAND_DIR, "sa2_to_order_crosswalk.csv")

print("=" * 60)
print("OSMX DTA Network Zones Dashboard - Loading")
print("=" * 60)

# ── 1. Load active zone polygons & build SA2 intersect set ──────────────
print("[1] Active zone polygons ...")
active_gdf = gpd.read_file(ACTIVE_SHP)

# ── 2. SA2 shapefile ──────────────────────────────────────────────────────
print("[2] SA2 shapefile ...")
sa2_all = gpd.read_file(SA2_SHP)
sa2_all = sa2_all.rename(columns={"SA2_CODE21": "sa2_code", "SA2_NAME21": "sa2_name"})
sa2_all["sa2_code"] = sa2_all["sa2_code"].astype(str)
if sa2_all.crs and sa2_all.crs.to_epsg() != 4326:
    sa2_all = sa2_all.to_crs(epsg=4326)

# ── 3. Determine active SA2 codes: all 507-zone SA2s that intersect the
#      DTA active union (includes inner/donut zones absorbed by outer zones
#      in the consolidation process, plus boundary-overlapping zones)
print("[3] Computing SA2 intersect with DTA active area ...")
zone_summary = pd.read_csv(ZONE_SUMMARY)
zone_summary["sa2_code"] = zone_summary["sa2_code"].astype(str)

if active_gdf.crs != sa2_all.crs:
    active_gdf_union = active_gdf.to_crs(sa2_all.crs)
else:
    active_gdf_union = active_gdf
active_union_geom = active_gdf_union.union_all()

# The zone UNIVERSE is the full SEQ study-area list
# (sa2_zones_reference_357.csv - all SEQ SA2s incl. the 2026-07 NSW
# extension), NOT zone_summary_507.csv. The old universe silently dropped
# every SEQ SA2 absent from the 507 summary - e.g. Jimboomba - Glenlogan
# (311041570) and ~25 other in_emme_network=False zones only ever appeared
# as a faint "planned" overlay, which read as "the dashboard is missing
# zones". zone_summary_507 is still used for demand joins below.
_SEQ_REF_CSV = os.path.join(DEMAND_DIR, "sa2_zones_reference_357.csv")
seq_ref = pd.read_csv(_SEQ_REF_CSV, dtype={"sa2_code": str})
all_seq_codes = set(seq_ref["sa2_code"])
print(f"  SEQ reference universe: {len(all_seq_codes)} SA2 zones")

# Every SEQ SA2 IS part of the network - the DTA-intersection test is no
# longer a membership filter. Zones outside today's DTA active polygon are
# the expansion set to be added to the network, not a separate "dormant"
# universe. The intersection is computed only to LABEL current coverage.
_current_dta = set()
for code in sorted(all_seq_codes):
    rows = sa2_all[sa2_all["sa2_code"] == code]
    if len(rows) > 0 and rows.iloc[0].geometry.intersects(active_union_geom):
        _current_dta.add(code)
_current_dta |= (set(active_gdf["sa2_code"].astype(str)) & all_seq_codes)
print(f"  ... of which currently covered by the DTA active polygon: {len(_current_dta)}")

# Full-universe GeoDataFrame: all SEQ zones are network zones
dta_gdf = sa2_all[sa2_all["sa2_code"].isin(all_seq_codes)].copy()
dta_gdf["network_status"] = np.where(
    dta_gdf["sa2_code"].isin(_current_dta),
    "Active in DTA Network",
    "Network zone (DTA coverage pending)",
)
active_sa2_codes = sorted(all_seq_codes)
print(f"  Network zones with geometry: {len(dta_gdf)}")

# Dormant set now empty by construction (kept as an empty frame so the
# downstream layer code needs no structural changes).
dormant_gdf = sa2_all[sa2_all["sa2_code"].isin(set())].copy()
print(f"  Dormant zones: {len(dormant_gdf)} (all SEQ zones are network members)")

# ── 3. Demographics (new_pop) ─────────────────────────────────────────────
print("[3] Synth pop demographics ...")
demo = pd.read_csv(DEMO_AGG_CSV)
demo["sa2_code"] = demo["sa2_code"].astype(str)
demo_pct_cols = sorted([c for c in demo.columns if c.endswith("_pct")])

# ── 4. Landuse from new_pop ───────────────────────────────────────────────
print("[4] new_pop landuse ...")
landuse = pd.read_csv(LANDUSE_CSV)
landuse["SA2_MAIN16"] = landuse["SA2_MAIN16"].astype(str)
pl = landuse.set_index("SA2_MAIN16")
for col, default in [("TOTPOP", 0), ("TOTHH", 0), ("TOTEMP", 0)]:
    dta_gdf[col.lower()] = dta_gdf["sa2_code"].map(
        lambda c: int(pl.loc[c, col]) if c in pl.index and pd.notna(pl.loc[c, col]) else default
    ).fillna(default).astype(int)

# ── 5. Population fallbacks ────────────────────────────────────────────
print("[5] Population fallbacks ...")
FILL_CSV = os.path.join(DEMAND_DIR, "zero_pop_fill_from_paper_plan.csv")
if os.path.exists(FILL_CSV):
    fill_df = pd.read_csv(FILL_CSV)
    fill_df["sa2_code"] = fill_df["sa2_code"].astype(str)
    fill_lookup = fill_df.set_index("sa2_code")["paper_pop"]
    filled = 0
    for idx in dta_gdf.index:
        code = dta_gdf.loc[idx, "sa2_code"]
        if dta_gdf.loc[idx, "totpop"] == 0 and code in fill_lookup.index:
            val = int(fill_lookup.loc[code])
            if val > 0:
                dta_gdf.loc[idx, "totpop"] = val
                dta_gdf.loc[idx, "tothh"] = max(dta_gdf.loc[idx, "tothh"], int(val / 2.5))
                filled += 1
    print(f"  Filled {filled} DTA zones from fill CSV")

# Also try zone_landuse_357 directly for SA2 codes
if os.path.exists(LANDUSE_357_CSV):
    lu357 = pd.read_csv(LANDUSE_357_CSV)
    lu357["SA2_MAIN16"] = lu357["SA2_MAIN16"].astype(str).str.strip()
    lu_lookup = dict(zip(lu357["SA2_MAIN16"], lu357["TOTPOP"]))
    filled2 = 0
    for idx in dta_gdf.index:
        code = dta_gdf.loc[idx, "sa2_code"]
        if dta_gdf.loc[idx, "totpop"] == 0 and code in lu_lookup:
            val = int(lu_lookup[code])
            if val > 0:
                dta_gdf.loc[idx, "totpop"] = val
                dta_gdf.loc[idx, "tothh"] = max(dta_gdf.loc[idx, "tothh"], int(val / 2.5))
                filled2 += 1
    print(f"  Filled {filled2} DTA zones from zone_landuse_357")

# ── 5b. Planned expansion zones (38 new SA2s for next HPC run) ──────
print("[5b] Loading planned expansion zones ...")
if os.path.exists(EXPANSION_CSV):
    exp_df = pd.read_csv(EXPANSION_CSV)
    exp_df["sa2_code"] = exp_df["sa2_code"].astype(str)
    exp_codes = set(exp_df["sa2_code"])
    expansion_gdf = sa2_all[sa2_all["sa2_code"].isin(exp_codes)].copy()
    expansion_gdf["totpop"] = expansion_gdf["sa2_code"].map(
        lambda c: int(exp_df[exp_df["sa2_code"] == c]["planned_pop"].iloc[0])
        if c in exp_codes else 0
    ).fillna(0).astype(int)
    expansion_gdf["tothh"] = (expansion_gdf["totpop"] / 2.5).astype(int)
    expansion_gdf["totemp"] = (expansion_gdf["totpop"] * 0.48).astype(int)
    expansion_gdf["network_status"] = "Planned (Next Run)"
    expansion_gdf["pop_status"] = "Planned"
    # Merge into dta_gdf so they show on the main map
    dta_gdf = pd.concat([dta_gdf, expansion_gdf], ignore_index=True)
    print(f"  Added {len(expansion_gdf)} planned expansion zones to DTA map")
    print(f"  Planned pop: {expansion_gdf['totpop'].sum():,}")

# Add pop_status and fill demand / demo for expansion zones
dta_gdf["pop_status"] = dta_gdf["pop_status"].fillna("Has pop data")


# ── 6. Demand ─────────────────────────────────────────────────────────────
print("[6] Demand data ...")
dl = zone_summary.set_index("sa2_code")
for dcol in ["synth_demand_car", "synth_demand_pt", "synth_demand_walk",
             "synth_demand_bike", "hhts_demand_car", "hhts_demand_pt",
             "hhts_demand_walk", "hhts_demand_bike"]:
    if dcol in zone_summary.columns:
        dta_gdf[dcol] = dta_gdf["sa2_code"].map(
            lambda c: float(dl.loc[c, dcol]) if c in dl.index else 0.0
        ).fillna(0.0)

for prefix in ["synth", "hhts"]:
    mode_cols = [f"{prefix}_demand_{m}" for m in ["car", "pt", "walk", "bike"]]
    existing = [c for c in mode_cols if c in dta_gdf.columns]
    if existing:
        dta_gdf[f"{prefix}_demand_total"] = dta_gdf[existing].sum(axis=1)

# ── 7. Merge demographics ─────────────────────────────────────────────────
print("[7] Demographics merge ...")
demo_dl = demo.set_index("sa2_code")
for dc in demo_pct_cols:
    dta_gdf[dc] = dta_gdf["sa2_code"].map(
        lambda c: float(demo_dl.loc[c, dc]) if c in demo_dl.index and pd.notna(demo_dl.loc[c, dc]) else np.nan
    )
dta_gdf["total_persons"] = dta_gdf["sa2_code"].map(
    lambda c: int(demo_dl.loc[c, "total_persons"])
    if c in demo_dl.index and pd.notna(demo_dl.loc[c, "total_persons"])
    else int(dta_gdf[dta_gdf["sa2_code"] == c]["totpop"].iloc[0])
).fillna(0).astype(int)

print(f"  Final zones on map: {len(dta_gdf)}")
active_count = len(dta_gdf[dta_gdf["network_status"] == "Active in DTA Network"])
planned_count = len(dta_gdf[dta_gdf["network_status"] == "Planned (Next Run)"])
print(f"  Active: {active_count}, Planned: {planned_count}")
print(f"  Total pop (all): {int(dta_gdf['totpop'].sum()):,}")

# ── 8. GeoJSON ────────────────────────────────────────────────────────────
print("[8] GeoJSON ...")
geojson_dta = json.loads(dta_gdf.to_json())
for i, feat in enumerate(geojson_dta["features"]):
    feat["id"] = dta_gdf.iloc[i]["sa2_code"]

sa2_qld = sa2_all[sa2_all["STE_NAME21"] == "Queensland"].copy()
geojson_qld = json.loads(sa2_qld.to_json())
for i, feat in enumerate(geojson_qld["features"]):
    feat["id"] = sa2_qld.iloc[i]["sa2_code"]

# ── 9. Center ─────────────────────────────────────────────────────────────
dta_proj = dta_gdf.to_crs(epsg=7856)
CENTER_LAT = float(dta_proj.geometry.centroid.to_crs(4326).y.mean())
CENTER_LON = float(dta_proj.geometry.centroid.to_crs(4326).x.mean())

# ── 10. Metric options ────────────────────────────────────────────────────
DEMO_CATS = sorted(set(demo_pct_cols) & set(dta_gdf.columns))
METRIC_OPTIONS = [
    {"label": "Total Population", "value": "totpop", "type": "count"},
    {"label": "Total Households", "value": "tothh", "type": "count"},
    {"label": "Total Employment", "value": "totemp", "type": "count"},
    {"label": "Network Status", "value": "network_status", "type": "categorical"},
    {"label": "--- Demographics (%) ---", "value": "_d1", "type": "divider", "disabled": True},
] + [{"label": c.replace("_pct", "").replace("_", " ").title() + " %",
      "value": c, "type": "percent"} for c in DEMO_CATS] + [
    {"label": "--- Demand ---", "value": "_d2", "type": "divider", "disabled": True},
    {"label": "Synth Demand - Car", "value": "synth_demand_car"},
    {"label": "Synth Demand - PT", "value": "synth_demand_pt"},
    {"label": "Synth Demand - Total", "value": "synth_demand_total"},
    {"label": "HHTS Demand - Car", "value": "hhts_demand_car"},
    {"label": "HHTS Demand - Total", "value": "hhts_demand_total"},
]
METRIC_OPTIONS = [m for m in METRIC_OPTIONS
                  if m.get("disabled") or m["value"] in dta_gdf.columns]

# ── 11. Dash App ──────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="DTA Network Zones - SA2 & ABM Population")
app._favicon = None

app.layout = html.Div([
    html.H1(f"SEQ DTA Network Zones ({len(dta_gdf)} SA2 zones) - ABM Population",
            style={"textAlign": "center", "fontFamily": "Segoe UI, Arial",
                   "color": "#2c3e50", "marginBottom": "3px", "fontSize": "19px"}),
    html.Div(
        f"DTA Model: SEQ_Running_UE  |  {int(dta_gdf['totpop'].sum()):,} persons  |  "
        f"{(dta_gdf['network_status'] == 'Active in DTA Network').sum()} active + "
        f"{(dta_gdf['network_status'] == 'Planned (Next Run)').sum()} planned  |  "
        f"{len(dormant_gdf)} dormant",
        style={"textAlign": "center", "color": "#7f8c8d", "marginBottom": "6px",
               "fontSize": "12px"}),
    # ── Controls ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Metric:", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(id="metric-drop", options=METRIC_OPTIONS,
                         value="totpop", clearable=False,
                         style={"width": "300px", "fontSize": "11px"}),
        ], style={"display": "inline-block", "marginRight": "10px"}),
        html.Div([
            html.Label("Colour:", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(id="cscale-drop",
                         options=[{"label": s, "value": s} for s in
                                  ["Viridis", "Blues", "Reds", "YlOrRd",
                                   "Plasma", "Inferno", "Greens"]],
                         value="Viridis", clearable=False,
                         style={"width": "120px", "fontSize": "11px"}),
        ], style={"display": "inline-block", "marginRight": "10px"}),
        html.Div([
            html.Label("Map:", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Dropdown(id="mapstyle-drop",
                         options=[{"label": "Light", "value": "carto-positron"},
                                  {"label": "Dark", "value": "carto-darkmatter"},
                                  {"label": "OSM", "value": "open-street-map"}],
                         value="carto-positron", clearable=False,
                         style={"width": "110px", "fontSize": "11px"}),
        ], style={"display": "inline-block", "marginRight": "10px"}),
        html.Div([
            html.Label("Dormant:", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Checklist(id="show-dormant",
                          options=[{"label": "Show", "value": "dormant"}],
                          value=[], inline=True),
        ], style={"display": "inline-block", "verticalAlign": "middle",
                  "marginRight": "10px"}),
        html.Div([
            html.Label("Opacity:", style={"fontWeight": "bold", "fontSize": "11px"}),
            dcc.Slider(0.3, 1.0, 0.05, value=0.7, id="opacity-slider",
                       marks={i/10: str(i/10) for i in range(3, 11)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"display": "inline-block", "width": "180px",
                  "verticalAlign": "middle"}),
    ], style={"padding": "6px 16px", "backgroundColor": "#ecf0f1",
              "borderBottom": "2px solid #bdc3c7"}),

    # ── Main row ───────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Graph(id="main-map", style={"height": "70vh"},
                      config={"scrollZoom": True}),
            html.Div(id="click-info",
                     style={"padding": "2px 8px", "color": "#666", "fontSize": "10px"}),
        ], style={"width": "66%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.H4("Zone Detail", style={"textAlign": "center", "marginBottom": "3px",
                                          "fontSize": "13px"}),
            html.Div(id="zone-detail", style={
                "padding": "6px", "backgroundColor": "#f8f9fa", "borderRadius": "3px",
                "minHeight": "240px", "fontSize": "11px", "overflowY": "auto",
                "maxHeight": "36vh"}),
            html.Hr(style={"margin": "4px 0"}),
            html.Div(id="demo-breakdown-container", children=[
                dcc.Graph(id="demo-breakdown", style={"height": "260px"}),
            ]),
        ], style={"width": "34%", "display": "inline-block", "verticalAlign": "top",
                  "paddingLeft": "6px"}),
    ]),

    # ── Bottom tabs ────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs([
            dcc.Tab(label="Charts", children=[
                html.Div([
                    html.Div([
                        dcc.Graph(id="work-bar", style={"height": "250px"}),
                    ], style={"width": "33%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(id="top-pop-bar", style={"height": "250px"}),
                    ], style={"width": "33%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(id="scatter-plot", style={"height": "250px"}),
                    ], style={"width": "33%", "display": "inline-block"}),
                ]),
            ]),
            dcc.Tab(label="Data Table", children=[
                dash_table.DataTable(
                    id="data-table",
                    columns=[{"name": c, "id": c}
                            for c in ["sa2_code", "sa2_name", "totpop",
                                      "tothh", "totemp", "sex_Male_pct",
                                      "work_Employed_pct", "carlic_Yes_pct"]
                            if c in dta_gdf.columns],
                    data=dta_gdf.drop(columns="geometry").to_dict("records"),
                    page_size=30, filter_action="native", sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": "10px", "maxWidth": "90px",
                                "overflow": "hidden", "textOverflow": "ellipsis"},
                    style_header={"backgroundColor": "#2c3e50", "color": "white",
                                  "fontWeight": "bold", "fontSize": "10px"},
                ),
            ]),
            dcc.Tab(label="QLD Overview", children=[
                dcc.Graph(id="qld-overview-map", style={"height": "60vh"}),
            ]),
            dcc.Tab(label="Person Day", children=[
                html.Div([
                    html.Div([
                        html.Label("Person type", style={"fontSize": "11px"}),
                        dcc.Dropdown(id="pd-ptype", options=[], value=None,
                                     style={"fontSize": "11px"}),
                    ], style={"width": "24%", "display": "inline-block",
                              "verticalAlign": "top", "paddingRight": "8px"}),
                    html.Div([
                        html.Label("Person", style={"fontSize": "11px"}),
                        dcc.Dropdown(id="pd-person", options=[], value=None,
                                     style={"fontSize": "11px"}),
                    ], style={"width": "24%", "display": "inline-block",
                              "verticalAlign": "top"}),
                    html.Div(id="pd-status", style={
                        "width": "50%", "display": "inline-block",
                        "fontSize": "11px", "color": "#666",
                        "paddingLeft": "12px", "verticalAlign": "bottom"}),
                ], style={"padding": "6px 2px"}),
                html.Div([
                    html.Div([
                        dcc.Graph(id="pd-map-safety", style={"height": "52vh"}),
                    ], style={"width": "50%", "display": "inline-block"}),
                    html.Div([
                        dcc.Graph(id="pd-map-nosafety", style={"height": "52vh"}),
                    ], style={"width": "50%", "display": "inline-block"}),
                ]),
            ]),
        ]),
    ], style={"marginTop": "6px"}),
], style={"fontFamily": "Segoe UI, Arial"})


# ── Person Day tab ──────────────────────────────────────────────────────────
# Samples are pre-extracted from the newest completed pipeline run(s) by
# code/generate_person_day_samples.py (runs automatically at the end of each
# PBS pipeline job) - this tab never touches the multi-GB stage6 CSVs itself,
# so it only shows anything AFTER the full model chain has produced output.
PD_SAMPLES_PATH = os.path.join(DEMAND_DIR, "person_day_samples.json")
PD_SAMPLES = {}
if os.path.exists(PD_SAMPLES_PATH):
    with open(PD_SAMPLES_PATH, encoding="utf-8") as _fh:
        PD_SAMPLES = json.load(_fh)
_pd_centroids = seq_ref.set_index(seq_ref["sa2_code"].astype(int))[
    ["centroid_lat", "centroid_lon", "sa2_name"]]


@app.callback([Output("pd-ptype", "options"), Output("pd-ptype", "value"),
               Output("pd-status", "children")],
              Input("pd-ptype", "id"))
def pd_init(_):
    pts = sorted(PD_SAMPLES.get("safety", {}).keys())
    if not pts:
        return [], None, ("No person-day samples yet - run the pipeline, then "
                          "python code/generate_person_day_samples.py")
    has_cf = bool(PD_SAMPLES.get("nosafety"))
    msg = (f"{sum(len(v) for v in PD_SAMPLES['safety'].values())} sampled persons | "
           + ("counterfactual (no-safety) arm loaded - side-by-side enabled"
              if has_cf else
              "no-safety arm not run yet - right panel will be empty"))
    return [{"label": p, "value": p} for p in pts], pts[0], msg


@app.callback([Output("pd-person", "options"), Output("pd-person", "value")],
              Input("pd-ptype", "value"))
def pd_persons(ptype):
    if not ptype:
        return [], None
    pids = sorted(PD_SAMPLES.get("safety", {}).get(ptype, {}).keys())
    both = set(PD_SAMPLES.get("nosafety", {}).get(ptype, {}).keys())
    opts = [{"label": pid + (" (both arms)" if pid in both else ""), "value": pid}
            for pid in pids]
    default = next((p for p in pids if p in both), pids[0] if pids else None)
    return opts, default


def _pd_day_figure(recs, title, line_color):
    fig = go.Figure()
    if not recs:
        fig.update_layout(title=f"{title} - no data",
                          margin={"r": 0, "t": 35, "l": 0, "b": 0})
        return fig
    trips = [r for r in recs if r.get("row_type") == "trip"
             and r.get("OTAZ") in _pd_centroids.index
             and r.get("DTAZ") in _pd_centroids.index]
    lats, lons, texts = [], [], []
    for i, t in enumerate(trips):
        o = _pd_centroids.loc[t["OTAZ"]]
        d = _pd_centroids.loc[t["DTAZ"]]
        fig.add_trace(go.Scattermapbox(
            lat=[o["centroid_lat"], d["centroid_lat"]],
            lon=[o["centroid_lon"], d["centroid_lon"]],
            mode="lines+markers", showlegend=False,
            line={"width": 3, "color": line_color},
            marker={"size": 9},
            hovertext=(f"trip {i+1}: {t['mode']} {t.get('purpose','')} "
                       f"{t.get('dist','?')}km<br>{o['sa2_name']} → {d['sa2_name']}"),
            hoverinfo="text",
        ))
        lats += [o["centroid_lat"], d["centroid_lat"]]
        lons += [o["centroid_lon"], d["centroid_lon"]]
    acts = [r for r in recs if r.get("row_type") == "activity"]
    summary = " → ".join(a.get("purpose", "?") for a in acts) or "no activities"
    center = ({"lat": float(np.mean(lats)), "lon": float(np.mean(lons))}
              if lats else {"lat": CENTER_LAT, "lon": CENTER_LON})
    fig.update_layout(
        title=f"{title}<br><sub>{len(trips)} trips · {summary}</sub>",
        mapbox={"style": "carto-positron", "zoom": 9, "center": center},
        margin={"r": 0, "t": 55, "l": 0, "b": 0}, uirevision="pd",
    )
    return fig


@app.callback([Output("pd-map-safety", "figure"),
               Output("pd-map-nosafety", "figure")],
              [Input("pd-ptype", "value"), Input("pd-person", "value")])
def pd_maps(ptype, pid):
    if not ptype or not pid:
        return go.Figure(), go.Figure()
    base = PD_SAMPLES.get("safety", {}).get(ptype, {}).get(pid, [])
    cf = PD_SAMPLES.get("nosafety", {}).get(ptype, {}).get(pid, [])
    return (_pd_day_figure(base, f"{ptype} {pid} - crash skims ON", "#C85A14"),
            _pd_day_figure(cf, f"{ptype} {pid} - crash skims OFF (counterfactual)", "#2E7D6B"))


# ── Map callback ────────────────────────────────────────────────────────────
@app.callback(
    Output("main-map", "figure"),
    [Input("metric-drop", "value"), Input("cscale-drop", "value"),
     Input("mapstyle-drop", "value"), Input("show-dormant", "value"),
     Input("opacity-slider", "value")]
)
def update_map(metric, cscale, mapstyle, show_dormant, opacity):
    if not metric or metric.startswith("_"):
        return go.Figure()

    if metric == "pop_status" or metric == "network_status":
        color_col = metric
        cmap = {"Active in DTA Network": "#2ecc71", "Planned (Next Run)": "#f39c12",
                "Has pop data": "#2ecc71", "No pop data": "#e74c3c"}
        fig = px.choropleth_mapbox(
            dta_gdf, geojson=geojson_dta, locations="sa2_code",
            color=color_col,
            color_discrete_map=cmap,
            mapbox_style=mapstyle, zoom=7.5,
            center={"lat": CENTER_LAT, "lon": CENTER_LON},
            opacity=opacity,
            hover_data={"sa2_code": True, "sa2_name": True, "totpop": True},
            labels={color_col: "Status"},
            custom_data=["sa2_code", "sa2_name", "totpop"],
        )
        fig.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0})
        return fig

    is_pct = metric.endswith("_pct")
    label = metric.replace("_pct", "").replace("_", " ").title()
    if is_pct:
        label += " %"

    fig = px.choropleth_mapbox(
        dta_gdf, geojson=geojson_dta, locations="sa2_code",
        color=metric,
        color_continuous_scale=cscale,
        range_color=[0, 100] if is_pct else None,
        mapbox_style=mapstyle,
        zoom=7.5,
        center={"lat": CENTER_LAT, "lon": CENTER_LON},
        opacity=opacity,
        hover_data={"sa2_code": True, "sa2_name": True,
                   metric: ":.1f" if is_pct else True, "totpop": True},
        labels={metric: label},
        custom_data=["sa2_code", "sa2_name", metric, "totpop"],
    )

    if "dormant" in (show_dormant or []):
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson_qld,
            locations=dormant_gdf["sa2_code"],
            z=[1] * len(dormant_gdf),
            colorscale=[[0, "rgba(231,76,60,0.2)"], [1, "rgba(231,76,60,0.35)"]],
            showscale=False, marker_opacity=0.35, marker_line_width=0.1,
            name="Dormant", hoverinfo="skip",
        ))

    # Always overlay planned expansion zones (orange)
    if expansion_gdf is not None and len(expansion_gdf) > 0:
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson_qld,
            locations=expansion_gdf["sa2_code"],
            z=[1] * len(expansion_gdf),
            colorscale=[[0, "rgba(243,156,18,0.45)"], [1, "rgba(243,156,18,0.6)"]],
            showscale=False, marker_opacity=0.55, marker_line_width=0.3,
            marker_line_color="#e67e22",
            name="Planned expansion (next run)",
            hovertemplate="<b>%{location}</b><br>Planned for next run<br>Pop: %{customdata[0]:,}<extra></extra>",
            customdata=expansion_gdf[["totpop"]].values,
        ))

    fig.update_traces(marker_line_width=0.2, marker_line_color="#555",
                      selector={"type": "choroplethmapbox"})
    fig.update_layout(
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
        coloraxis_colorbar={"title": label},
        clickmode="event+select",
        uirevision="map",
    )
    return fig


# ── Zone detail callback ───────────────────────────────────────────────────
@app.callback(
    [Output("click-info", "children"), Output("zone-detail", "children")],
    Input("main-map", "clickData"),
)
def zone_detail(click_data):
    if not click_data or "points" not in click_data:
        return "Click a zone for detail", html.P("Click a zone", style={"color": "#999"})
    try:
        pt = click_data["points"][0]
        code = str(pt.get("location", "") or pt.get("customdata", ["", ""])[0])
    except Exception:
        return "", html.P("N/A")
    row = dta_gdf[dta_gdf["sa2_code"] == code]
    if len(row) == 0:
        return f"Zone {code}", html.P("Not found")
    r = row.iloc[0]
    info = f"Selected: {r['sa2_name']} (SA2: {code})"
    panel = html.Div([
        html.H4(r["sa2_name"]),
        html.P([html.B("Pop: "), f"{int(r['totpop']):,}"], style={"marginBottom": "1px"}),
        html.P([html.B("HH: "), f"{int(r['tothh']):,}  ", html.B("Emp: "),
                f"{int(r['totemp']):,}"], style={"marginBottom": "1px"}),
        html.P([html.B("Car demand: "), f"{r.get('synth_demand_car', 0):,.0f}"],
               style={"marginBottom": "1px", "fontSize": "10px"}),
        html.P([html.B("PT demand: "), f"{r.get('synth_demand_pt', 0):,.0f}"],
               style={"marginBottom": "1px", "fontSize": "10px"}),
        html.Hr(style={"margin": "3px 0"}),
        html.P([html.B("Male: "), f"{r.get('sex_Male_pct', 0):.1f}%  ",
                html.B("Female: "), f"{r.get('sex_Female_pct', 0):.1f}%"],
               style={"fontSize": "10px", "marginBottom": "1px"}),
        html.P([html.B("Employed: "), f"{r.get('work_Employed_pct', 0):.1f}%  ",
                html.B("Unemployed: "), f"{r.get('work_Unemployed_pct', 0):.1f}%"],
               style={"fontSize": "10px", "marginBottom": "1px"}),
        html.P([html.B("Car Licence: "), f"{r.get('carlic_Yes_pct', 0):.1f}%"],
               style={"fontSize": "10px", "marginBottom": "1px"}),
        html.P([html.B("High Income: "), f"{r.get('inc_high_pct', 0):.1f}%"],
               style={"fontSize": "10px"}),
    ])
    return info, panel


# ── Demo breakdown callback ────────────────────────────────────────────────
@app.callback(
    Output("demo-breakdown", "figure"),
    Input("main-map", "clickData"),
)
def demo_breakdown(click_data):
    empty = go.Figure(layout={"title": "Click a zone"})
    if not click_data:
        return empty
    try:
        pt = click_data["points"][0]
        code = str(pt.get("location", "") or pt.get("customdata", ["", ""])[0])
    except Exception:
        return empty
    row = dta_gdf[dta_gdf["sa2_code"] == code]
    if len(row) == 0:
        return empty
    r = row.iloc[0]

    male = r.get("sex_Male_pct", 0) or 0
    female = r.get("sex_Female_pct", 0) or 0
    if male == 0 and female == 0:
        return empty

    fig = go.Figure(go.Pie(
        labels=["Male", "Female"], values=[male, female],
        hole=0.45, marker_colors=["#3498db", "#e74c3c"],
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=f"Sex - {r['sa2_name']}",
        margin={"l": 5, "r": 5, "t": 30, "b": 5}, height=260, showlegend=False,
    )
    return fig


# ── Summary charts ─────────────────────────────────────────────────────────
@app.callback(
    [Output("work-bar", "figure"), Output("top-pop-bar", "figure"),
     Output("scatter-plot", "figure")],
    Input("metric-drop", "value"),
)
def summary(_):
    wcols = {}
    for c, lbl, clr in [("work_Employed_pct", "Employed", "#2ecc71"),
                          ("work_Unemployed_pct", "Unemployed", "#e74c3c"),
                          ("work_Not in labour force_pct", "Not LF", "#95a5a6")]:
        if c in dta_gdf.columns:
            wcols[lbl] = (dta_gdf[c].mean(), clr)

    work_fig = go.Figure(go.Bar(
        x=list(wcols.keys()), y=[v[0] for v in wcols.values()],
        marker_color=[v[1] for v in wcols.values()],
        text=[f"{v[0]:.1f}%" for v in wcols.values()], textposition="outside",
    ))
    work_fig.update_layout(title="Avg Work Status %", margin={"l": 30, "r": 10, "t": 30, "b": 40},
                           height=250, showlegend=False)

    top15 = dta_gdf.nlargest(15, "totpop")
    pop_fig = go.Figure(go.Bar(
        x=top15["totpop"], y=top15["sa2_name"], orientation="h",
        marker_color="#3498db",
        text=[f"{int(v):,}" for v in top15["totpop"]], textposition="outside",
    ))
    pop_fig.update_layout(title="Top 15 Zones", yaxis={"categoryorder": "total ascending"},
                          margin={"l": 80, "r": 10, "t": 30, "b": 10},
                          height=250, showlegend=False)

    scat = go.Figure()
    if "synth_demand_total" in dta_gdf.columns:
        valid = dta_gdf[dta_gdf["totpop"] > 0]
        scat = px.scatter(valid, x="totpop", y="synth_demand_total",
                          hover_name="sa2_name", opacity=0.6,
                          title="Pop vs Synth Demand", trendline="ols")
        scat.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, height=250)
    return work_fig, pop_fig, scat


@app.callback(
    Output("qld-overview-map", "figure"),
    Input("mapstyle-drop", "value"),
)
def qld_overview(mapstyle):
    sa2_all_view = sa2_qld.copy()
    sa2_all_view["status"] = sa2_all_view["sa2_code"].apply(
        lambda c: "Active in DTA" if c in set(active_sa2_codes) else "Outside DTA"
    )
    fig = px.choropleth_mapbox(
        sa2_all_view, geojson=geojson_qld, locations="sa2_code",
        color="status",
        color_discrete_map={"Active in DTA": "#2ecc71", "Outside DTA": "#e74c3c"},
        mapbox_style=mapstyle, zoom=5.5,
        center={"lat": CENTER_LAT, "lon": CENTER_LON},
        opacity=0.6, title="All QLD SA2 Zones - DTA Status",
        hover_data={"sa2_code": True, "sa2_name": True, "status": True},
    )
    fig.update_traces(marker_line_width=0.15, marker_line_color="#555")
    fig.update_layout(margin={"r": 0, "t": 35, "l": 0, "b": 0},
                      uirevision="qld")
    return fig


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nDashboard ready: http://127.0.0.1:8053")
    app.run(debug=False, port=8053, host="127.0.0.1")
