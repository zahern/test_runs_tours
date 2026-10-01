"""
Regenerate GA stage plots from saved ga_results.pkl using the updated
plot_convergence style (no diversity/mutation rate overlay).

Usage:
    python regenerate_ga_plots.py <path_to_ga_results.pkl> [output_dir]
"""
import sys
import os

# Add the code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_logger import regenerate_ga_plots_from_pkl

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python regenerate_ga_plots.py <path_to_ga_results.pkl> [output_dir]")
        sys.exit(1)
    
    ga_results_pkl = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plots_dir = regenerate_ga_plots_from_pkl(ga_results_pkl, out_dir)
    print(f"Done regenerating plots in {plots_dir}")
