#!/usr/bin/env python3
"""
Standalone runner for HB vs Crash Prediction Model Search.
Designed for PBS job submission via qsub.

Usage:
  qsub -v OUT_DIR=<path> hb_search_runner.pbs
  python run_hb_search.py [--out-dir <path>] [--resume] [--de-only]
"""

import os, sys, json, time
import argparse

# Ensure the parent dir (paper template) is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from hb_vs_crash_prediction_search import main as run_search


def main():
    parser = argparse.ArgumentParser(
        description="HB vs Crash Prediction Model Search")
    parser.add_argument("--out-dir", default=PAPER_DIR,
                        help="Output directory (default: paper template dir)")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints")
    parser.add_argument("--de-only", action="store_true",
                        help="Skip MCR structure search")
    parser.add_argument("--maxiter", type=int, default=60,
                        help="DE max iterations")
    parser.add_argument("--popsize", type=int, default=15,
                        help="DE population size")
    parser.add_argument("--mcr-n-starts", type=int, default=1,
                        help="MCR single start (no-restarts policy)")
    parser.add_argument("--mcr-max-iter", type=int, default=500,
                        help="MCR max iterations per start")
    parser.add_argument("--solver", default="de", choices=["de", "sa"],
                        help="MCR solver")
    args = parser.parse_args()

    print(f"Starting HB vs Crash Prediction Search at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.out_dir}")
    print(f"Resume: {args.resume}, DE-only: {args.de_only}")

    run_search()


if __name__ == "__main__":
    main()