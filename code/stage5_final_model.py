"""Run the final GA-selected Stage 5 prediction workflow and save outputs.

This script loads a GA results pickle, survey data, synthetic population,
and skims, then applies the GA-selected feature sets to estimate Stage 1-5
models and write prediction outputs.

Usage:
  python stage5_final_model.py \
      --ga-results saved_data/ga_results.pkl \
      --synth saved_data/synthetic_population.csv \
      --skims downcasted_matrices_base.omx \
      --save-dir saved_data/final_model
"""

import argparse
import os
import pickle

import pandas as pd

from data_connect import Synth_Connect
from compute_accessibility import run_accessibility
from ga_hypothesis_testing import apply_ga_results_to_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run GA-selected final model prediction and save outputs.'
    )
    parser.add_argument(
        '--ga-results',
        default='saved_data/ga_results.pkl',
        help='Path to GA results pickle file.'
    )
    parser.add_argument(
        '--synth',
        required=True,
        help='Path to synthetic population CSV file.'
    )
    parser.add_argument(
        '--skims',
        default='downcasted_matrices_base.omx',
        help='Path to the OMX skim file.'
    )
    parser.add_argument(
        '--save-dir',
        default='saved_data/final_model',
        help='Directory where final outputs will be written.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for prediction sampling.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    for path in (args.ga_results, args.synth, args.skims):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    with open(args.ga_results, 'rb') as f:
        ga_results = pickle.load(f)

    synth = pd.read_csv(args.synth)

    sc = Synth_Connect()
    out = sc.process_survey(save=False)
    landuse = run_accessibility(
        sc.get_land(),
        skims=args.skims,
        save_path=os.path.join(args.save_dir, 'accessibility.csv'),
    )

    scheduled_dest, s5_models = apply_ga_results_to_pipeline(
        ga_results=ga_results,
        out=out,
        synth=synth,
        landuse=landuse,
        skims_path=args.skims,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    print('\nFinal GA-selected prediction complete:')
    print(f'  Stage 5 output: {args.save_dir}/stage5_output.csv')
    print(f'  Stage 5 models: {args.save_dir}/stage5_models.pkl')
    print(f'  Synthetic population rows: {len(synth):,}')
    print(f'  Trips in predicted plan: {(scheduled_dest[scheduled_dest["row_type"] == "trip"]).shape[0]:,}')


if __name__ == '__main__':
    main()
