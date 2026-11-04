#!/usr/bin/env python
"""
Model Loader and Prediction Script for PYRAM Framework
=======================================================

Loads saved models (.pkl and .html) for prediction and analysis.

Usage:
    python load_and_predict.py --run-dir <run_dir> --predict
    python load_and_predict.py --run-dir <run_dir> --report
    python load_and_predict.py --compare --run-dirs <dir1> <dir2> ...
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def load_pickle_model(path: Path) -> Any:
    """Load a pickled model."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_html_report(path: Path) -> str:
    """Load HTML report content."""
    with open(path, 'r') as f:
        return f.read()


class ModelLoader:
    """Loads and manages PYRAM models for prediction."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.models = {}
        self.reports = {}
        self.data = {}
        self._load_all()
    
    def _load_all(self):
        """Load all models and reports from run directory."""
        # Stage models
        for stage in range(1, 8):
            pkl_path = self.run_dir / f"stage{stage}_models.pkl"
            html_path = self.run_dir / f"stage{stage}_models.html"
            
            if pkl_path.exists():
                try:
                    self.models[f"stage{stage}"] = load_pickle_model(pkl_path)
                    print(f"✓ Loaded stage{stage}_models.pkl")
                except Exception as e:
                    print(f"✗ Failed to load stage{stage}_models.pkl: {e}")
            
            if html_path.exists():
                self.reports[f"stage{stage}"] = load_html_report(html_path)
        
        # Merged GA results
        for name in ["ga_results", "merged/ga_results"]:
            pkl_path = self.run_dir / f"{name}.pkl"
            html_path = self.run_dir / f"{name}.html"
            
            if pkl_path.exists():
                try:
                    self.models[name] = load_pickle_model(pkl_path)
                    print(f"✓ Loaded {name}.pkl")
                except Exception as e:
                    print(f"✗ Failed to load {name}.pkl: {e}")
            
            if html_path.exists():
                self.reports[name] = load_html_report(html_path)
        
        # Activity generation model
        act_pkl = self.run_dir / "activity_generation_model.pkl"
        act_html = self.run_dir / "activity_generation_report.html"
        if act_pkl.exists():
            try:
                self.models['activity_generation'] = load_pickle_model(act_pkl)
                print(f"✓ Loaded activity_generation_model.pkl")
            except Exception as e:
                print(f"✗ Failed to load activity_generation_model.pkl: {e}")
        if act_html.exists():
            self.reports['activity_generation'] = load_html_report(act_html)
        
        # DTA feedback
        dta_pkl = self.run_dir / "dta_feedback" / "convergence.pkl"
        dta_html = self.run_dir / "dta_feedback" / "convergence.html"
        if dta_pkl.exists():
            try:
                self.models['dta_feedback'] = load_pickle_model(dta_pkl)
                print(f"✓ Loaded dta_feedback/convergence.pkl")
            except Exception as e:
                print(f"✗ Failed to load dta_feedback/convergence.pkl: {e}")
        if dta_html.exists():
            self.reports['dta_feedback'] = load_html_report(dta_html)
        
        # Stage 6/7 outputs
        for fname in ["stage6_output.csv", "stage7_output.csv", "stage7_dta_equilibrated.csv"]:
            fpath = self.run_dir / fname
            if fpath.exists():
                try:
                    self.data[fname] = pd.read_csv(fpath)
                    print(f"✓ Loaded {fname}")
                except Exception as e:
                    print(f"✗ Failed to load {fname}: {e}")
    
    def predict_activities(self, persons: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict activities using activity generation model."""
        if 'activity_generation' not in self.models:
            raise ValueError("Activity generation model not loaded")
        
        model = self.models['activity_generation']
        return model.predict_activities(persons, **kwargs)
    
    def predict_destination(self, trips: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict destinations using destination choice model."""
        if 'stage3' not in self.models and 'destination' not in self.models:
            raise ValueError("Destination choice model not loaded")
        
        model = self.models.get('stage3') or self.models.get('destination')
        return model.predict(trips, **kwargs)
    
    def predict_mode(self, trips: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict mode choice."""
        if 'stage4' not in self.models and 'mode' not in self.models:
            raise ValueError("Mode choice model not loaded")
        
        model = self.models.get('stage4') or self.models.get('mode')
        return model.predict(trips, **kwargs)
    
    def predict_parking(self, trips: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict parking choices."""
        if 'stage7' not in self.models and 'parking' not in self.models:
            raise ValueError("Parking model not loaded")
        
        model = self.models.get('stage7') or self.models.get('parking')
        return model.predict(trips, **kwargs)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of loaded models."""
        summary = {
            'models': list(self.models.keys()),
            'reports': list(self.reports.keys()),
            'data_files': list(self.data.keys())
        }
        return summary
    
    def print_summary(self):
        """Print summary of loaded models."""
        print(f"\n{'='*60}")
        print(f"Model Loader Summary: {self.run_dir.name}")
        print(f"{'='*60}")
        print(f"Models loaded: {list(self.models.keys())}")
        print(f"HTML reports: {list(self.reports.keys())}")
        print(f"Data files: {list(self.data.keys())}")
        
        for name, model in self.models.items():
            if hasattr(model, 'loglik'):
                print(f"  {name}: LL={model.loglik():.2f}")
            elif hasattr(model, 'log_likelihood_'):
                print(f"  {name}: LL={model.log_likelihood_:.2f}")


def compare_runs(run_dirs: List[Path]) -> pd.DataFrame:
    """Compare multiple run directories."""
    rows = []
    
    for run_dir in run_dirs:
        loader = ModelLoader(run_dir)
        loader.print_summary()
        
        row = {'run_dir': str(run_dir)}
        
        # Extract key metrics
        for name, model in loader.models.items():
            if hasattr(model, 'loglik'):
                rows.append({**row, 'model': name, 'loglik': model.loglik()})
            elif hasattr(model, 'log_likelihood_'):
                rows.append({**row, 'model': name, 'loglik': model.log_likelihood_})
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Load and use PYRAM models")
    parser.add_argument('--run-dir', type=Path, help='Run directory to load')
    parser.add_argument('--predict', action='store_true', help='Run prediction')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    parser.add_argument('--compare', action='store_true', help='Compare multiple runs')
    parser.add_argument('--run-dirs', nargs='+', type=Path, help='Run directories to compare')
    parser.add_argument('--input-csv', type=Path, help='Input CSV for prediction')
    parser.add_argument('--output-csv', type=Path, help='Output CSV for predictions')
    
    args = parser.parse_args()
    
    if args.compare and args.run_dirs:
        print(f"Comparing {len(args.run_dirs)} runs...")
        df = compare_runs(args.run_dirs)
        if not df.empty:
            print(df.to_string())
            if args.output_csv:
                df.to_csv(args.output_csv, index=False)
                print(f"Saved comparison to {args.output_csv}")
        return
    
    if not args.run_dir:
        parser.error("--run-dir required unless --compare")
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 1
    
    loader = ModelLoader(run_dir)
    loader.print_summary()
    
    if args.predict and args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        
        # Run predictions based on available models
        if 'activity_generation' in loader.models:
            print("Predicting activities...")
            result = loader.predict_activities(input_df)
            if args.output_csv:
                result.to_csv(args.output_csv, index=False)
                print(f"Predictions saved to {args.output_csv}")
        else:
            print("No activity generation model available for prediction")
    
    if args.report:
        print("\nGenerating combined report...")
        # Generate combined HTML report from all loaded reports
        report_path = loader.run_dir / "combined_report.html"
        with open(report_path, 'w') as f:
            f.write("<html><body>\n")
            f.write(f"<h1>Combined Report: {loader.run_dir.name}</h1>\n")
            for name, html in loader.reports.items():
                f.write(f"<h2>{name}</h2>\n")
                f.write(html)
            f.write("</body></html>\n")
        print(f"Combined report saved to {report_path}")


if __name__ == "__main__":
    main()