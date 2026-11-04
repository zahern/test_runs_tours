"""
PYRAM 3D Visualizer
===================

Web-based 3D visualization for PYRAM pipeline outputs using Three.js.
Visualizes:
- Network (links, nodes, intersections)
- Vehicles/agents with trails
- Activities/locations (home, work, school, etc.)
- Parking facilities (on-street, garage, P&R)
- Simulation outputs (speeds, flows, densities)
- DTA feedback convergence

Run:
    python -m pyram.visualizer --data-dir output --port 8080

Or from Python:
    from pyram.visualizer import VisualizerServer
    server = VisualizerServer(data_dir="output")
    server.run(port=8080)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import geopandas as gpd
from flask import Flask, render_template, jsonify, send_from_directory, request


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer."""
    data_dir: str = "output"
    port: int = 8080
    host: str = "0.0.0.0"
    debug: bool = False
    mapbox_token: str = ""  # Optional: for satellite basemap
    default_camera: Dict = None
    
    def __post_init__(self):
        if self.default_camera is None:
            self.default_camera = {
                "lat": -27.4705,  # Brisbane CBD
                "lon": 153.0260,
                "zoom": 11,
                "pitch": 45,
                "bearing": 0
            }


class VisualizerDataLoader:
    """Loads PYRAM pipeline outputs for visualization."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache = {}
    
    def load_network(self) -> Dict[str, Any]:
        """Load network (links, nodes) from various possible sources."""
        # Try multiple possible network files
        network_files = [
            self.data_dir / "network.geojson",
            self.data_dir / "network_links.geojson",
            self.data_dir / "network_nodes.geojson",
            self.data_dir / "supply" / "network.geojson",
        ]
        
        for f in network_files:
            if f.exists():
                return self._parse_network_geojson(f)
        
        # Try to build from link/node CSVs
        link_file = self.data_dir / "network_links.csv"
        node_file = self.data_dir / "network_nodes.csv"
        if link_file.exists() and node_file.exists():
            return self._build_network_from_csv(link_file, node_file)
        
        # Try MATSim network
        matsim_network = self.data_dir / "matsim" / "network.xml"
        if matsim_network.exists():
            return self._parse_matsim_network(matsim_network)
        
        return {"links": [], "nodes": []}
    
    def _parse_network_geojson(self, path: Path) -> Dict:
        """Parse network from GeoJSON."""
        with open(path) as f:
            data = json.load(f)
        
        features = data.get("features", [])
        links = []
        nodes = []
        
        for feat in features:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            
            if geom.get("type") == "LineString":
                links.append({
                    "id": props.get("id", props.get("link_id", len(links))),
                    "coords": geom["coordinates"],
                    "length": props.get("length", 0),
                    "lanes": props.get("lanes", 1),
                    "freespeed": props.get("freespeed", props.get("speed_limit", 50)),
                    "modes": props.get("modes", ["car"]),
                    "from_node": props.get("from_node", props.get("from")),
                    "to_node": props.get("to_node", props.get("to")),
                })
            elif geom.get("type") == "Point":
                nodes.append({
                    "id": props.get("id", props.get("node_id", len(nodes))),
                    "coords": geom["coordinates"],
                    "type": props.get("type", "junction"),
                })
        
        return {"links": links, "nodes": nodes}
    
    def _build_network_from_csv(self, link_file: Path, node_file: Path) -> Dict:
        """Build network from link/node CSVs."""
        links_df = pd.read_csv(link_file)
        nodes_df = pd.read_csv(node_file)
        
        links = []
        for _, row in links_df.iterrows():
            links.append({
                "id": row.get("id", row.get("link_id", len(links))),
                "coords": [
                    [row.get("from_lon"), row.get("from_lat")],
                    [row.get("to_lon"), row.get("to_lat")]
                ] if all(c in row for c in ["from_lon", "from_lat", "to_lon", "to_lat"]) else [],
                "length": row.get("length", 0),
                "lanes": row.get("lanes", 1),
                "freespeed": row.get("freespeed", row.get("speed_limit", 50)),
                "modes": ["car"],
                "from_node": row.get("from_node", row.get("from")),
                "to_node": row.get("to_node", row.get("to")),
            })
        
        nodes = []
        for _, row in nodes_df.iterrows():
            nodes.append({
                "id": row.get("id", row.get("node_id", len(nodes))),
                "coords": [row.get("lon", row.get("x")), row.get("lat", row.get("y"))],
                "type": row.get("type", "junction"),
            })
        
        return {"links": links, "nodes": nodes}
    
    def _parse_matsim_network(self, path: Path) -> Dict:
        """Parse MATSim network.xml (simplified)."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        
        nodes = []
        for node in root.findall(".//node"):
            nodes.append({
                "id": node.get("id"),
                "coords": [float(node.get("x")), float(node.get("y"))],
                "type": "junction",
            })
        
        links = []
        for link in root.findall(".//link"):
            from_node = link.get("from")
            to_node = link.get("to")
            from_coords = None
            to_coords = None
            
            for n in nodes:
                if n["id"] == from_node:
                    from_coords = n["coords"]
                if n["id"] == to_node:
                    to_coords = n["coords"]
            
            if from_coords and to_coords:
                links.append({
                    "id": link.get("id"),
                    "coords": [from_coords, to_coords],
                    "length": float(link.get("length", 0)),
                    "lanes": int(float(link.get("permlanes", 1))),
                    "freespeed": float(link.get("freespeed", 50)),
                    "modes": ["car"],
                    "from_node": from_node,
                    "to_node": to_node,
                })
        
        return {"links": links, "nodes": nodes}
    
    def load_vehicles(self, timestep: int = 0) -> List[Dict]:
        """Load vehicle positions for a given timestep."""
        # Try multiple possible vehicle files
        vehicle_files = [
            self.data_dir / "vehicles.csv",
            self.data_dir / "vehicle_positions.csv",
            self.data_dir / "simulation" / f"vehicles_{timestep}.csv",
            self.data_dir / "dta_feedback" / f"iter_0" / "vehicles.csv",
        ]
        
        for f in vehicle_files:
            if f.exists():
                df = pd.read_csv(f)
                return self._parse_vehicles(df, timestep)
        
        # Try MATSim events
        events_file = self.data_dir / "matsim" / "events.xml.gz"
        if events_file.exists():
            return self._parse_matsim_events(events_file, timestep)
        
        return []
    
    def _parse_vehicles(self, df: pd.DataFrame, timestep: int) -> List[Dict]:
        """Parse vehicle data."""
        vehicles = []
        
        # Filter by timestep if column exists
        if "time" in df.columns or "timestep" in df.columns:
            time_col = "time" if "time" in df.columns else "timestep"
            df = df[df[time_col] == timestep]
        
        for _, row in df.iterrows():
            vehicles.append({
                "id": row.get("id", row.get("vehicle_id", len(vehicles))),
                "type": row.get("type", "car"),
                "coords": [row.get("lon", row.get("x")), row.get("lat", row.get("y"))],
                "speed": row.get("speed", 0),
                "heading": row.get("heading", 0),
                "route": row.get("route", []),
            })
        
        return vehicles
    
    def _parse_matsim_events(self, path: Path, timestep: int) -> List[Dict]:
        """Parse MATSim events (simplified - just vehicle enters traffic)."""
        # This would need full events parsing - placeholder
        return []
    
    def load_activities(self) -> List[Dict]:
        """Load activity locations from pipeline output."""
        activity_files = [
            self.data_dir / "stage6_output.csv",
            self.data_dir / "stage7_output.csv",
            self.data_dir / "stage7_dta_equilibrated.csv",
            self.data_dir / "synthetic_population.csv",
        ]
        
        for f in activity_files:
            if f.exists():
                df = pd.read_csv(f)
                return self._parse_activities(df)
        
        return []
    
    def _parse_activities(self, df: pd.DataFrame) -> List[Dict]:
        """Parse activity data."""
        activities = []
        
        # Filter activity rows
        if "row_type" in df.columns:
            df = df[df["row_type"] == "activity"]
        elif "type" in df.columns:
            df = df[df["type"] == "activity"]
        
        purpose_colors = {
            "home": "#1f77b4",
            "work": "#ff7f0e",
            "edu": "#2ca02c",
            "school": "#2ca02c",
            "shopping": "#d62728",
            "social": "#9467bd",
            "eat": "#8c564b",
            "escort": "#e377c2",
            "othmaint": "#7f7f7f",
            "othdiscr": "#bcbd22",
        }
        
        for _, row in df.iterrows():
            # Get coordinates
            if "x" in row and "y" in row:
                coords = [row["x"], row["y"]]
            elif "lon" in row and "lat" in row:
                coords = [row["lon"], row["lat"]]
            elif "DTAZ" in row:
                # Would need zone centroids - skip for now
                continue
            else:
                continue
            
            purpose = str(row.get("purpose", "unknown")).lower()
            activities.append({
                "id": row.get("person_id", len(activities)),
                "person_id": row.get("person_id"),
                "purpose": purpose,
                "coords": coords,
                "color": purpose_colors.get(purpose, "#999999"),
                "start_time": row.get("STARTIME", row.get("start_time", 0)),
                "end_time": row.get("ENDTIME", row.get("end_time", 0)),
                "duration": row.get("predicted_duration_min", 0),
            })
        
        return activities
    
    def load_parking(self) -> List[Dict]:
        """Load parking facilities."""
        parking_files = [
            self.data_dir / "parking_facilities.csv",
            self.data_dir / "location_table.csv",
            self.data_dir / "stage7_output.csv",
        ]
        
        for f in parking_files:
            if f.exists():
                df = pd.read_csv(f)
                return self._parse_parking(df)
        
        return []
    
    def _parse_parking(self, df: pd.DataFrame) -> List[Dict]:
        """Parse parking data."""
        parking = []
        
        type_map = {1: "onstreet", 2: "garage", 3: "pnr"}
        type_colors = {
            "onstreet": "#2196F3",
            "garage": "#FF9800",
            "pnr": "#4CAF50",
        }
        
        for _, row in df.iterrows():
            ptype = row.get("parking_type", row.get("type", "garage"))
            if isinstance(ptype, (int, float)):
                ptype = type_map.get(int(ptype), "garage")
            
            if "x" in row and "y" in row:
                coords = [row["x"], row["y"]]
            elif "lon" in row and "lat" in row:
                coords = [row["lon"], row["lat"]]
            else:
                continue
            
            parking.append({
                "id": row.get("location_id", row.get("id", len(parking))),
                "type": ptype,
                "coords": coords,
                "capacity": row.get("capacity", 100),
                "cost": row.get("avg_parking_cost", row.get("cost", 0)),
                "color": type_colors.get(ptype, "#999999"),
            })
        
        return parking
    
    def load_dta_convergence(self) -> Dict:
        """Load DTA feedback convergence metrics."""
        convergence_files = list(self.data_dir.glob("dta_feedback/*/convergence.json"))
        if not convergence_files:
            convergence_files = list(self.data_dir.glob("dta_feedback/convergence.json"))
        
        if convergence_files:
            with open(convergence_files[0]) as f:
                return json.load(f)
        
        return {"iterations": [], "rmse": [], "converged": False}
    
    def load_all(self) -> Dict[str, Any]:
        """Load all visualization data."""
        return {
            "network": self.load_network(),
            "vehicles": self.load_vehicles(),
            "activities": self.load_activities(),
            "parking": self.load_parking(),
            "dta_convergence": self.load_dta_convergence(),
        }


def create_app(config: VisualizerConfig) -> Flask:
    """Create Flask application for the visualizer."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    
    loader = VisualizerDataLoader(config.data_dir)
    
    @app.route("/")
    def index():
        """Main visualizer page."""
        return render_template("visualizer.html", config=config)
    
    @app.route("/api/data")
    def api_data():
        """API endpoint for all visualization data."""
        data = loader.load_all()
        return jsonify(data)
    
    @app.route("/api/network")
    def api_network():
        """API endpoint for network data."""
        return jsonify(loader.load_network())
    
    @app.route("/api/vehicles")
    def api_vehicles():
        """API endpoint for vehicle positions."""
        timestep = request.args.get("timestep", 0, type=int)
        return jsonify(loader.load_vehicles(timestep))
    
    @app.route("/api/activities")
    def api_activities():
        """API endpoint for activities."""
        return jsonify(loader.load_activities())
    
    @app.route("/api/parking")
    def api_parking():
        """API endpoint for parking facilities."""
        return jsonify(loader.load_parking())
    
    @app.route("/api/dta_convergence")
    def api_dta_convergence():
        """API endpoint for DTA convergence."""
        return jsonify(loader.load_dta_convergence())
    
    @app.route("/static/<path:path>")
    def send_static(path):
        """Serve static files."""
        return send_from_directory(app.static_folder, path)
    
    return app


class VisualizerServer:
    """Main visualizer server class."""
    
    def __init__(self, data_dir: str = "output", port: int = 8080, host: str = "0.0.0.0", debug: bool = False):
        self.config = VisualizerConfig(
            data_dir=data_dir,
            port=port,
            host=host,
            debug=debug,
        )
        self.app = create_app(self.config)
    
    def run(self):
        """Run the visualizer server."""
        print(f"Starting PYRAM 3D Visualizer on http://{self.config.host}:{self.config.port}")
        print(f"Data directory: {self.config.data_dir}")
        self.app.run(host=self.config.host, port=self.config.port, debug=self.config.debug)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="PYRAM 3D Visualizer")
    parser.add_argument("--data-dir", default="output", help="Pipeline output directory")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    server = VisualizerServer(
        data_dir=args.data_dir,
        port=args.port,
        host=args.host,
        debug=args.debug,
    )
    server.run()


if __name__ == "__main__":
    main()