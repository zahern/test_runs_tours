#!/usr/bin/env python
"""
Minimal test of Stage 5 batch prediction with DEBUG logging to capture full tracebacks.
"""
import sys
import logging
import os

# Force debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s | %(name)s | %(message)s'
)

# Disable pandas warnings  
os.environ['PANDAS_USE_NUMEXPR'] = '0'

logger = logging.getLogger(__name__)

# Add current dir to path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

# Load a minimal Stage 4 sample (just 20 activities from 2 zones)
logger.info("Loading Stage 4 sample...")
s4_full = pd.read_csv('runs/stage5_smoke_s4_input.csv')  # Will be created by smoke script
if len(s4_full) == 0:
    logger.warning("Stage 5 s4 input not found; using synthetic sample")
    s4_full = pd.DataFrame({
        'person_id': ['p1']*5 + ['p2']*5,
        'OTAZ': [1]*5 + [2]*5,
        'purpose': ['work', 'work', 'edu', 'nonmand', 'nonmand'] * 2,
        'STARTIME': ['07:00', '08:00', '09:00', '10:00', '11:00'] * 2,
        'ENDTIME': ['08:00', '09:00', '10:00', '11:00', '12:00'] * 2,
    })

logger.info(f"S4 sample: {s4_full.shape}, columns={list(s4_full.columns)}")
logger.info(f"Purpose values: {s4_full['purpose'].value_counts().to_dict()}")

# Load models (will reuse from smoke run)
# Use load_models() which patches Larch's __setstate__ before unpickling,
# avoiding "Cannot set 'availability_any'" errors on newer Larch versions.
logger.info("Loading Stage 5 models...")
try:
    from stage5_destination_choice import load_models as _s5_load
    models = _s5_load('runs/stage5_smoke_models.pkl')
    logger.info(f"Loaded models for segments: {list(models.keys())}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    sys.exit(1)

# Load other required data
logger.info("Loading persons, landuse, skims...")
persons = pd.read_csv('person_ss.csv', nrows=100)
with open('runs/skims_omx.pkl', 'rb') as f:
    skims = pickle.load(f)
with open('runs/stage5_smoke_landuse.pkl', 'rb') as f:
    landuse = pickle.load(f)

logger.info(f"Persons: {persons.shape}, Landuse: {landuse.shape}")

# Now run the batch building logic
logger.info("=" * 80)
logger.info("Testing _build_trip_chain on small sample...")
logger.info("=" * 80)

from stage5_destination_choice import _build_trip_chain, _skim_lookup

s4_batch = s4_full.iloc[:10].copy()
logger.info(f"Batch input: {s4_batch.shape}, columns={list(s4_batch.columns)}")
logger.info(f"Purpose column present: {'purpose' in s4_batch.columns}")
logger.info(f"Purpose values in batch: {s4_batch['purpose'].values}")

try:
    # Build skim lookup
    skim_df = pd.DataFrame()  # Dummy; real implementation builds this from skims
    sk_lookup = {}
    
    batch_plan = _build_trip_chain(
        s4_batch,
        models,
        persons,
        landuse,
        skims,
        sk_lookup,
        seed=42
    )
    logger.info(f"SUCCESS! Batch result: {batch_plan.shape}, columns={list(batch_plan.columns)}")
except Exception as e:
    import traceback as tb
    logger.error(f"FAILED: {type(e).__name__}: {e}")
    logger.error(f"Full traceback:\n{tb.format_exc()}")
    sys.exit(1)
