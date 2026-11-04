import os
os.makedirs(" /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics/MetaCount/metacount\, exist_ok=True)
with open(\/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics/MetaCount/metacount/__init__.py\, \w\) as f:
 f.write(\from .sparse_bayes import SparseBayesianRegressor\n\)
 f.write(\from .risk_predictor import RiskPredictor build_fixture_dataframe\n\)
 f.write(\from .sparse_solver import SparseSolver quick_sparsity_path\n\)
 f.write(\__version__ = \\0.1.0\\\n\)
 f.write(\__all__ = [\\SparseBayesianRegressor\\ \\RiskPredictor\\ \\SparseSolver\\ \\quick_sparsity_path\\ \\build_fixture_dataframe\\]\n\)
print(\Created __init__.py\)
