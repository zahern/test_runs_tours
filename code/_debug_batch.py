#!/usr/bin/env python
"""
Minimal debug script to test Stage 5 batch prediction with detailed logging.
"""
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Add current dir to path
sys.path.insert(0, '.')

# Import stage5
from stage5_destination_choice import run_stage5

# Quick smoke load
person_sample = pd.read_csv('person_ss.csv', nrows=50)
trip_sample = pd.read_csv('initial_trips_hhts.csv', nrows=100)

logger.info(f"Loaded {len(person_sample)} persons, {len(trip_sample)} trips for debug")

# Load landuse, skims
import pickle
with open('runs/stage6_smoke_models.pkl', 'rb') as f:
    stage6_mdl = pickle.load(f)

landuse_s6 = stage6_mdl.get('landuse', pd.DataFrame())
logger.info(f"Landuse: {landuse_s6.shape}")

# Fake Stage 4 output with purpose already present
s4_sample = pd.DataFrame({
    'person_id': np.arange(1, 11),
    'OTAZ': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    'purpose': ['work', 'work', 'edu', 'nonmand', 'nonmand', 'work', 'edu', 'nonmand', 'work', 'work'],
    'STARTIME': ['07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00'],
    'other_col': np.random.randn(10)
})

logger.info(f"Sample Stage 4 activities:\n{s4_sample}")
logger.info(f"Sample Stage 4 columns: {list(s4_sample.columns)}")

# Call run_stage5 with minimal synth pop
try:
    result = run_stage5(
        s4=s4_sample,
        persons=person_sample,
        landuse=landuse_s6,
        skims=None,  # Will be loaded internally
        seed=42,
        config={}
    )
    logger.info(f"Run_stage5 succeeded! Result keys: {list(result.keys())}")
    if 'plan' in result and result['plan'] is not None:
        logger.info(f"Plan shape: {result['plan'].shape}, columns: {list(result['plan'].columns)}")
except Exception as e:
    import traceback as tb
    logger.error(f"Run_stage5 failed: {e}")
    logger.error(f"Traceback:\n{tb.format_exc()}")
    sys.exit(1)
