"""
08_demand_dashboard_final.py
Dash dashboard on port 8052 — SA2 polygon choropleth with demand + crash overlay.
"""
import os
import re
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import io

OUT_DIR = r"Z:\test_runs_tours\demand_"

# --- Load data ---
print("Loading data for dashboard...")

# Zone polygons
gpkg_path = os.path.join(OUT_DIR, 'zone_demand_crash_507.gpkg')
gdf = gpd.read_file(gpkg_path, layer='sa2_demand_crash')
if gdf.crs and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)
geojson_data = json.loads(gdf.to_json())
# Add id field matching sa2_code for choropleth
for feat in geojson_data['features']:
    feat['id'] = str(feat['properties']['sa2_code'])

# Zone summary (no geometry)
zone_summary = pd.read_csv(os.path.join(OUT_DIR, 'zone_summary_507.csv'))
zone_summary['sa2_code'] = zone_summary['sa2_code'].astype(int)

# OD data for desire lines
synth_od = pd.read_csv(os.path.join(OUT_DIR, 'synth_pop_od_507.csv'))

# Crosswalk for lat/lon
crosswalk = pd.read_csv(os.path.join(OUT_DIR, 'sa2_crosswalk_507.csv'))
lat_lon = crosswalk.set_index('sa2_code')[['sa2_lat', 'sa2_lon']].to_dict('index')

# Available skim columns
skim_cols_available = [c for c in zone_summary.columns if c.startswith('skim_') or c.startswith('crash_') or
                       c.startswith('synth_') or c.startswith('hhts_')]

# Metric options
METRIC_OPTIONS = [
    {'label': 'Synth Demand - Car', 'value': 'synth_demand_car'},
    {'label': 'Synth Demand - PT', 'value': 'synth_demand_pt'},
    {'label': 'Synth Demand - Walk', 'value': 'synth_demand_walk'},
    {'label': 'Synth Demand - Bike', 'value': 'synth_demand_bike'},
    {'label': 'HHTS Demand - Car', 'value': 'hhts_demand_car'},
    {'label': 'HHTS Demand - PT', 'value': 'hhts_demand_pt'},
    {'label': 'HHTS Demand - Walk', 'value': 'hhts_demand_walk'},
    {'label': 'HHTS Demand - Bike', 'value': 'hhts_demand_bike'},
    {'label': 'Crash Total (Inbound)', 'value': 'crash_inbound_total'},
    {'label': 'Crash Density', 'value': 'crash_density'},
    {'label': 'Crash - Run Off Road', 'value': 'crash_inbound_runoff'},
    {'label': 'Crash - Head On', 'value': 'crash_inbound_headon'},
    {'label': 'Crash - Rear End', 'value': 'crash_inbound_rearend'},
    {'label': 'Crash - Intersection', 'value': 'crash_inbound_intersection'},
]
# Filter to only metrics that exist in zone_summary
METRIC_OPTIONS = [m for m in METRIC_OPTIONS if m['value'] in zone_summary.columns]

PERIOD_OPTIONS = [
    {'label': 'AM Peak', 'value': 'AM'},
    {'label': 'PM Peak', 'value': 'PM'},
    {'label': 'Midday', 'value': 'MD'},
    {'label': 'Early AM', 'value': 'EA'},
    {'label': 'Evening', 'value': 'EV'},
]

# --- App layout ---
try:
    import dash_bootstrap_components as dbc
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    USE_DBC = True
except ImportError:
    app = Dash(__name__)
    USE_DBC = False

app.layout = html.Div([
    html.H2("SEQ SA2 Travel Demand & Crash Risk Dashboard (507 Zones)",
            style={'textAlign': 'center', 'color': '#fff', 'padding': '10px',
                   'backgroundColor': '#1e3a5f'}),

    html.Div([
        # Left panel: controls
        html.Div([
            html.Label("Metric:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=METRIC_OPTIONS,
                value=METRIC_OPTIONS[0]['value'] if METRIC_OPTIONS else 'crash_inbound_total',
                clearable=False,
                style={'color': '#000'}
            ),
            html.Br(),
            html.Label("Time Period (for skims):", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='period-radio',
                options=PERIOD_OPTIONS,
                value='AM',
                inline=True
            ),
            html.Br(),
            html.Label("Selected Zone:", style={'fontWeight': 'bold'}),
            html.Div(id='selected-zone-info', style={'fontSize': '12px', 'minHeight': '60px'}),
            html.Br(),
            html.Button("Export OD CSV", id='export-btn', n_clicks=0,
                        style={'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none',
                               'padding': '8px 16px', 'cursor': 'pointer', 'borderRadius': '4px'}),
            dcc.Download(id='download-od'),
        ], style={'width': '18%', 'padding': '10px', 'backgroundColor': '#1a1a2e',
                  'color': '#eee', 'minHeight': '80vh', 'verticalAlign': 'top',
                  'display': 'inline-block', 'fontSize': '13px'}),

        # Map
        html.Div([
            dcc.Graph(id='choropleth-map', style={'height': '80vh'}),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right panel: charts
        html.Div([
            html.Label("Mode Split (Origin Zone):", style={'fontWeight': 'bold'}),
            dcc.Graph(id='mode-split-bar', style={'height': '250px'}),
            html.Label("Crash Breakdown:", style={'fontWeight': 'bold'}),
            dcc.Graph(id='crash-donut', style={'height': '250px'}),
            html.Label("Top OD Desire Lines:", style={'fontWeight': 'bold'}),
            html.Div(id='desire-lines-table', style={'fontSize': '11px', 'overflowY': 'auto', 'maxHeight': '200px'}),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                  'padding': '10px', 'backgroundColor': '#1a1a2e', 'color': '#eee'}),
    ]),
], style={'backgroundColor': '#0d1117', 'color': '#eee', 'fontFamily': 'Arial'})


@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('metric-dropdown', 'value'), Input('period-radio', 'value')]
)
def update_map(metric, period):
    # Map metric to actual column
    col = metric
    if col not in zone_summary.columns:
        col = zone_summary.columns[-1]

    fig = px.choropleth_mapbox(
        zone_summary,
        geojson=geojson_data,
        locations='sa2_code',
        color=col,
        featureidkey='id',
        mapbox_style='carto-darkmatter',
        zoom=8,
        center={'lat': -27.5, 'lon': 153.0},
        opacity=0.7,
        color_continuous_scale='Viridis',
        labels={col: col.replace('_', ' ').title()},
        hover_data=['sa2_name', col] if 'sa2_name' in zone_summary.columns else [col],
        custom_data=['sa2_code', 'sa2_name'] if 'sa2_name' in zone_summary.columns else ['sa2_code'],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#0d1117',
        font_color='#eee',
        title=dict(text=f"{col.replace('_', ' ').title()} ({period})", x=0.5, font=dict(size=13))
    )
    return fig


@app.callback(
    [Output('selected-zone-info', 'children'),
     Output('mode-split-bar', 'figure'),
     Output('crash-donut', 'figure'),
     Output('desire-lines-table', 'children')],
    [Input('choropleth-map', 'clickData')]
)
def update_zone_details(clickData):
    empty_fig = go.Figure(layout=go.Layout(
        paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e', font_color='#eee',
        margin=dict(l=10, r=10, t=10, b=10)
    ))

    if not clickData:
        return "Click on a zone to see details", empty_fig, empty_fig, "No zone selected"

    try:
        sa2_code = int(clickData['points'][0]['customdata'][0])
        sa2_name = clickData['points'][0]['customdata'][1] if len(clickData['points'][0].get('customdata', [])) > 1 else str(sa2_code)
    except (KeyError, IndexError, ValueError):
        return "Click on a zone to see details", empty_fig, empty_fig, ""

    zone_row = zone_summary[zone_summary['sa2_code'] == sa2_code]
    if len(zone_row) == 0:
        return f"Zone {sa2_code} not found", empty_fig, empty_fig, ""

    zone_row = zone_row.iloc[0]
    info_text = [
        html.B(sa2_name), html.Br(),
        f"SA2 Code: {sa2_code}", html.Br(),
        f"EMME Zones: {zone_row.get('n_emme_zones', 'N/A')}", html.Br(),
    ]

    # Mode split bar chart
    mode_data = {}
    for mode in ['car', 'pt', 'walk', 'bike']:
        col = f'synth_demand_{mode}'
        if col in zone_row:
            mode_data[mode] = float(zone_row[col]) if pd.notna(zone_row[col]) else 0

    if mode_data:
        fig_mode = go.Figure(data=[
            go.Bar(
                x=list(mode_data.keys()),
                y=list(mode_data.values()),
                marker_color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
            )
        ])
        fig_mode.update_layout(
            paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e', font_color='#eee',
            margin=dict(l=30, r=10, t=10, b=30), showlegend=False
        )
    else:
        fig_mode = empty_fig

    # Crash donut
    crash_types = {}
    for ct in ['crash_inbound_runoff', 'crash_inbound_headon', 'crash_inbound_rearend', 'crash_inbound_intersection']:
        if ct in zone_row:
            label = ct.replace('crash_inbound_', '').replace('_', ' ').title()
            crash_types[label] = float(zone_row[ct]) if pd.notna(zone_row[ct]) else 0

    if crash_types and sum(crash_types.values()) > 0:
        fig_crash = go.Figure(data=[go.Pie(
            labels=list(crash_types.keys()),
            values=list(crash_types.values()),
            hole=0.4
        )])
        fig_crash.update_layout(
            paper_bgcolor='#1a1a2e', font_color='#eee',
            margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
            legend=dict(font=dict(size=10))
        )
    else:
        fig_crash = empty_fig

    # Desire lines table
    od_subset = synth_od[synth_od['origin_sa2'] == sa2_code].nlargest(20, 'demand')
    if len(od_subset) > 0:
        table_rows = [html.Tr([html.Th('Dest SA2'), html.Th('Mode'), html.Th('Demand')],
                              style={'borderBottom': '1px solid #444'})]
        for _, row in od_subset.iterrows():
            table_rows.append(html.Tr([
                html.Td(str(int(row['dest_sa2']))),
                html.Td(str(row['mode'])),
                html.Td(f"{row['demand']:.1f}"),
            ]))
        table = html.Table(table_rows, style={'width': '100%', 'fontSize': '11px'})
    else:
        table = html.P("No OD data for this zone")

    return info_text, fig_mode, fig_crash, table


@app.callback(
    Output('download-od', 'data'),
    [Input('export-btn', 'n_clicks')],
    [State('choropleth-map', 'clickData')],
    prevent_initial_call=True
)
def export_od(n_clicks, clickData):
    if not clickData:
        return None
    try:
        sa2_code = int(clickData['points'][0]['customdata'][0])
    except (KeyError, IndexError, ValueError):
        return None
    od_subset = synth_od[synth_od['origin_sa2'] == sa2_code]
    return dcc.send_data_frame(od_subset.to_csv, f"od_zone_{sa2_code}.csv", index=False)


if __name__ == '__main__':
    print("Starting dashboard on port 8052...")
    app.run(debug=False, host='0.0.0.0', port=8052)
