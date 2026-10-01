#!/usr/bin/env python3
"""Quick dry-run test of the analysis pipeline."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'Z:\test_runs_tours\demand_\paper\Unofficial_Transportation_Research_Board__TRB__LaTeX_template')
from hb_search.hb_vs_crash_prediction_search import load_data, fit_nb_model

df, geo = load_data()

# Test with a small subset of features
test_feats = geo[:5]
print(f"Testing with features: {test_feats}")
result = fit_nb_model(df, test_feats, 'nm_total', 'log_len', label='TEST')
print(f"TEST model: LL={result['ll']:.2f}, BIC={result['bic']:.2f}, k={result['k']}, n={result['n']}")
print(f"Converged: {result['converged']}")
print("Dry run PASSED!")