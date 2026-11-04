"""Capture one Stage 3 prediction chunk to compare against after a refactor."""
import os
import pickle
import time

import pandas as pd

from stage3_activity_duration import predict_activity_durations

RUN = 'runs/agds_default_staged__paper_predict_20261024'
OUT = '_tmp_s3_baseline_chunk1.parquet'
N = int(os.environ.get('BASELINE_PERSONS', '20000'))

with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
    models = pickle.load(fh)
chunk = next(pd.read_csv(os.path.join(RUN, 'stage2_output.csv'), chunksize=N))

t0 = time.time()
acts = predict_activity_durations(chunk, models, seed=42)
dt = time.time() - t0

cols = [c for c in ('person_id', 'purpose', 'activity_seq', 'planning_order',
                    'predicted_duration_min', 'start_of_day', 'end_of_day', 'B')
        if c in acts.columns]
acts[cols].to_parquet(OUT, index=False)
print(f'{len(chunk)} persons -> {len(acts)} activities in {dt:.1f}s '
      f'({dt / len(chunk) * 1000:.2f} ms/person)')
print(f'wrote {OUT} ({len(cols)} cols)')
print(acts.groupby('purpose')['predicted_duration_min'].mean().round(2).to_string())
