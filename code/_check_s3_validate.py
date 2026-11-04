"""Scratch check: run the reworked Stage 3 validation panel on the real HHTS."""
import logging
import os
import pickle

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

from data_connect import Synth_Connect                       # noqa: E402
from validate_hhts import validate_stage3                     # noqa: E402

RUN = 'runs/agds_default_staged__paper_predict_20261024'
OUT = '_tmp_val_stage3'

out = Synth_Connect().process_survey(save=False)
print('survey frames:', {k: (v.shape if hasattr(v, "shape") else type(v))
                         for k, v in out.items()}, flush=True)

with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as f:
    s3 = pickle.load(f)

os.makedirs(OUT, exist_ok=True)
res = validate_stage3(out, s3, OUT)
print('\nmetrics:', res['metrics'], flush=True)

df = pd.read_csv(os.path.join(OUT, 'val_stage3_duration_summary.csv'))
cols = ['purpose', 'n_obs', 'obs_mean_min', 'raw_mean_min', 'pred_mean_min',
        'raw_bias_pct', 'mean_bias_pct']
print('\n' + df[cols].round(1).to_string(index=False), flush=True)
print('\nmean |bias|  raw=%s  pipeline=%s'
      % (res['metrics'].get('mean_abs_bias_raw_pct'),
         res['metrics'].get('mean_abs_bias_pct')), flush=True)
