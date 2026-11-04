"""
Script 06: Dash dashboard with SA2 polygon choropleth demand map.
Run standalone: python 06_demand_dashboard_sa2.py
Then open http://127.0.0.1:8050 in your browser.
"""
import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = r"Z:\test_runs_tours\demand_"
GPKG_FILE    = os.path.join(OUTPUT_DIR, 'zone_demand_polygons.gpkg')
SA2_REF_CSV  = os.path.join(OUTPUT_DIR, 'sa2_zones_reference.csv')
SYNTH_OD_CSV = os.path.join(OUTPUT_DIR, 'synth_pop_od_sa2.csv')
HHTS_OD_CSV  = os.path.join(OUTPUT_DIR, 'hhts_od_by_sa2_mode.csv')

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")

gdf = gpd.read_file(GPKG_FILE, layer='sa2_demand')
if gdf.crs and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)
gdf['sa2_code'] = gdf['sa2_code'].astype(str)
print(f"GeoPackage loaded: {len(gdf)} SA2 zones")

# Convert to GeoJSON for plotly
geojson = json.loads(gdf.to_json())
# Add id field to each feature for choropleth matching
for i, feat in enumerate(geojson['features']):
    feat['id'] = str(gdf.iloc[i]['sa2_code'])

# Load OD data for spider lines
synth_od = pd.read_csv(SYNTH_OD_CSV) if os.path.exists(SYNTH_OD_CSV) else pd.DataFrame()
hhts_od  = pd.read_csv(HHTS_OD_CSV)  if os.path.exists(HHTS_OD_CSV)  else pd.DataFrame()
sa2_ref  = pd.read_csv(SA2_REF_CSV)  if os.path.exists(SA2_REF_CSV)  else pd.DataFrame()

# Build centroid lookup for OD spider lines
centroid_lookup = {}
if 'centroid_lon' in sa2_ref.columns:
    for _, row in sa2_ref.iterrows():
        if pd.notna(row.get('centroid_lon')) and pd.notna(row.get('centroid_lat')):
            centroid_lookup[str(row['sa2_code'])] = (float(row['centroid_lon']), float(row['centroid_lat']))

# Also build centroids from gdf geometry for any missing
for _, row in gdf.iterrows():
    code = str(row['sa2_code'])
    if code not in centroid_lookup and row.geometry is not None:
        try:
            c = row.geometry.centroid
            centroid_lookup[code] = (c.x, c.y)
        except Exception:
            pass

# ── Define demand metric options ──────────────────────────────────────────────
demand_cols = [c for c in gdf.columns if 'demand' in c.lower() and c != 'geometry']
metric_options = [{'label': c.replace('_', ' ').title(), 'value': c} for c in demand_cols]

# Mode split for bar chart
modes = ['car', 'pt', 'walk', 'bike', 'other']

# ── Build app ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="SEQ SA2 Demand Dashboard")

CENTER_LAT = gdf.geometry.centroid.y.mean()
CENTER_LON = gdf.geometry.centroid.x.mean()

app.layout = html.Div([
    html.H1("SEQ Travel Demand Dashboard – SA2 Level",
            style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginBottom': '10px'}),

    # ── Control row ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Demand Metric:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=metric_options,
                value='synth_demand_total' if 'synth_demand_total' in demand_cols else
                      (demand_cols[0] if demand_cols else None),
                clearable=False,
                style={'width': '350px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),

        html.Div([
            html.Label("Colour Scale:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='colorscale-dropdown',
                options=[
                    {'label': 'Viridis', 'value': 'Viridis'},
                    {'label': 'Blues',   'value': 'Blues'},
                    {'label': 'Reds',    'value': 'Reds'},
                    {'label': 'YlOrRd',  'value': 'YlOrRd'},
                    {'label': 'Plasma',  'value': 'Plasma'},
                ],
                value='Viridis',
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),

        html.Div([
            html.Label("OD Data Source:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='od-source',
                options=[
                    {'label': 'Synth Pop', 'value': 'synth'},
                    {'label': 'HHTS', 'value': 'hhts'},
                ],
                value='synth',
                inline=True,
            ),
        ], style={'display': 'inline-block'}),
    ], style={'padding': '10px 20px', 'backgroundColor': '#f0f0f0', 'borderBottom': '1px solid #ccc'}),

    # ── Main content ─────────────────────────────────────────────────────────
    dcc.Tabs([
        # Tab 1: Choropleth Map
        dcc.Tab(label='Choropleth Map', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='choropleth-map', style={'height': '70vh'}),
                    html.Div(id='selected-zone-info', style={'padding': '5px 10px', 'color': '#555'}),
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4("Mode Split (Origin)", style={'textAlign': 'center'}),
                    dcc.Graph(id='mode-split-chart', style={'height': '35vh'}),
                    html.H4("OD Spider Lines (top 20 destinations)", style={'textAlign': 'center'}),
                    dcc.Graph(id='spider-lines-map', style={'height': '33vh'}),
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ]),
        ]),

        # Tab 2: Summary Statistics
        dcc.Tab(label='Summary Statistics', children=[
            html.Div([
                html.Div([
                    html.H3("Demand by Mode (All SA2s)"),
                    dcc.Graph(id='total-mode-bar'),
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Top SA2 Origins by Total Demand"),
                    dcc.Graph(id='top-origins-bar'),
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.H3("SA2 Demand Data Table"),
                dash_table.DataTable(
                    id='data-table',
                    data=gdf.drop(columns='geometry').fillna(0).round(1).to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in gdf.columns if c != 'geometry'],
                    page_size=20,
                    filter_action='native',
                    sort_action='native',
                    style_table={'overflowX': 'auto'},
                    style_cell={'fontSize': '11px', 'maxWidth': '120px',
                                'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}
                    ],
                ),
            ], style={'padding': '0 20px'}),
        ]),

        # Tab 3: Skim Comparison
        dcc.Tab(label='SA2 Skims', children=[
            html.Div([
                html.H3("Aggregated SEQSTM Skim Distribution (to be loaded from OMX)"),
                html.P("Select a metric above and a zone in the map to view skim values."),
                dcc.Graph(id='skim-histogram'),
            ], style={'padding': '20px'}),
        ]),
    ]),
], style={'fontFamily': 'Arial'})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('colorscale-dropdown', 'value')]
)
def update_choropleth(metric, colorscale):
    if not metric or metric not in gdf.columns:
        fig = go.Figure()
        fig.update_layout(title="No metric selected")
        return fig

    z_vals = gdf[metric].fillna(0)
    label  = metric.replace('_', ' ').title()

    fig = px.choropleth_mapbox(
        gdf,
        geojson=geojson,
        locations='sa2_code',
        color=metric,
        color_continuous_scale=colorscale,
        mapbox_style='carto-positron',
        zoom=8,
        center={'lat': CENTER_LAT, 'lon': CENTER_LON},
        opacity=0.7,
        labels={metric: label},
        hover_data={c: True for c in ['sa2_name_abs', 'n_emme_zones', metric]
                    if c in gdf.columns},
        title=f"SA2 Demand: {label}",
    )
    fig.update_layout(
        margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
        coloraxis_colorbar={'title': label},
        clickmode='event+select',
    )
    return fig


@app.callback(
    [Output('mode-split-chart', 'figure'),
     Output('spider-lines-map', 'figure'),
     Output('selected-zone-info', 'children')],
    [Input('choropleth-map', 'clickData'),
     Input('od-source', 'value')]
)
def update_on_click(clickData, od_source):
    if clickData is None:
        return go.Figure(), go.Figure(), "Click on a zone to see details."

    # Get selected SA2 code
    point = clickData['points'][0]
    sa2_code = str(point.get('location', ''))
    if not sa2_code:
        return go.Figure(), go.Figure(), "Could not identify zone."

    # Get name
    row_mask = gdf['sa2_code'] == sa2_code
    sa2_name = gdf.loc[row_mask, 'sa2_name_abs'].iloc[0] if row_mask.any() else sa2_code
    info = f"Selected: {sa2_name} (SA2 code: {sa2_code})"

    # ── Mode split bar chart ──────────────────────────────────────────────────
    if od_source == 'synth' and len(synth_od) > 0:
        zone_od = synth_od[synth_od['origin_sa2'].astype(str) == sa2_code]
        mode_vals = zone_od.groupby('mode')['demand'].sum()
        title_suffix = "Synth Pop"
        col_name = 'demand'
    else:
        zone_od = hhts_od[hhts_od['origin_sa2'].astype(str) == sa2_code] if len(hhts_od) > 0 else pd.DataFrame()
        mode_vals = zone_od.groupby('mode')['total_weighted_trips'].sum() if len(zone_od) > 0 else pd.Series()
        title_suffix = "HHTS"
        col_name = 'total_weighted_trips'

    if len(mode_vals) > 0:
        mode_fig = px.bar(
            x=mode_vals.index,
            y=mode_vals.values,
            title=f"Mode Split – {sa2_name} ({title_suffix})",
            labels={'x': 'Mode', 'y': 'Trips'},
            color=mode_vals.index,
            color_discrete_map={'car': '#e74c3c', 'pt': '#3498db', 'walk': '#2ecc71',
                                 'bike': '#f39c12', 'other': '#95a5a6'},
        )
        mode_fig.update_layout(margin={'t': 40, 'b': 20}, showlegend=False)
    else:
        mode_fig = go.Figure()
        mode_fig.update_layout(title=f"No data for {sa2_name}")

    # ── Spider lines map ──────────────────────────────────────────────────────
    if len(zone_od) > 0:
        total_col = col_name
        # Top 20 destinations by demand
        dest_totals = zone_od.groupby('dest_sa2')[total_col].sum().nlargest(20)
        o_lon, o_lat = centroid_lookup.get(sa2_code, (None, None))

        spider_traces = []
        if o_lon is not None:
            for dest_code, demand in dest_totals.items():
                d_lon, d_lat = centroid_lookup.get(str(dest_code), (None, None))
                if d_lon is None:
                    continue
                width = max(0.5, min(5, demand / max(dest_totals.values) * 5))
                spider_traces.append(go.Scattermapbox(
                    lon=[o_lon, d_lon, None],
                    lat=[o_lat, d_lat, None],
                    mode='lines',
                    line={'width': width, 'color': 'rgba(231,76,60,0.6)'},
                    hoverinfo='skip',
                    showlegend=False,
                ))
            # Origin marker
            spider_traces.append(go.Scattermapbox(
                lon=[o_lon], lat=[o_lat], mode='markers',
                marker={'size': 12, 'color': 'red'},
                name=sa2_name, hoverinfo='name',
            ))

        if spider_traces:
            spider_fig = go.Figure(data=spider_traces)
            spider_fig.update_layout(
                mapbox={'style': 'carto-positron', 'zoom': 7,
                        'center': {'lat': o_lat, 'lon': o_lon}},
                margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                showlegend=False,
            )
        else:
            spider_fig = go.Figure()
            spider_fig.update_layout(title="No centroid data for spider lines")
    else:
        spider_fig = go.Figure()
        spider_fig.update_layout(title="No OD data for selected zone")

    return mode_fig, spider_fig, info


@app.callback(
    [Output('total-mode-bar', 'figure'),
     Output('top-origins-bar', 'figure')],
    Input('od-source', 'value')
)
def update_summary_charts(od_source):
    if od_source == 'synth' and len(synth_od) > 0:
        mode_totals = synth_od.groupby('mode')['demand'].sum().reset_index()
        mode_totals.columns = ['mode', 'total']
        mode_fig = px.bar(mode_totals, x='mode', y='total',
                          title='Total Synth Pop Demand by Mode',
                          color='mode',
                          color_discrete_map={'car': '#e74c3c', 'pt': '#3498db',
                                              'walk': '#2ecc71', 'bike': '#f39c12',
                                              'other': '#95a5a6'})

        top_origins = (synth_od.groupby('origin_sa2')['demand'].sum()
                       .nlargest(20).reset_index())
        top_origins.columns = ['origin_sa2', 'total_demand']
    elif len(hhts_od) > 0:
        mode_totals = hhts_od.groupby('mode')['total_weighted_trips'].sum().reset_index()
        mode_totals.columns = ['mode', 'total']
        mode_fig = px.bar(mode_totals, x='mode', y='total',
                          title='Total HHTS Demand by Mode',
                          color='mode')
        top_origins = (hhts_od.groupby('origin_sa2')['total_weighted_trips'].sum()
                       .nlargest(20).reset_index())
        top_origins.columns = ['origin_sa2', 'total_demand']
    else:
        return go.Figure(), go.Figure()

    mode_fig.update_layout(showlegend=False)

    top_fig = px.bar(top_origins, x='origin_sa2', y='total_demand',
                     title='Top 20 SA2 Origins by Total Demand')
    top_fig.update_layout(xaxis_tickangle=-45)

    return mode_fig, top_fig


@app.callback(
    Output('skim-histogram', 'figure'),
    Input('metric-dropdown', 'value')
)
def update_skim_hist(metric):
    if not metric or metric not in gdf.columns:
        return go.Figure()
    vals = gdf[metric].dropna()
    vals = vals[vals > 0]
    if len(vals) == 0:
        return go.Figure()
    fig = px.histogram(vals, nbins=50,
                       title=f"Distribution of {metric.replace('_',' ').title()} across SA2 zones",
                       labels={'value': metric})
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting dashboard on http://127.0.0.1:8051")
    app.run(debug=False, port=8051, host='127.0.0.1')
