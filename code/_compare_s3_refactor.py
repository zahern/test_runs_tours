"""Compare the narrow-frame Stage 3 prediction against the captured baseline."""
import os
import pickle
import time

import numpy as np
import pandas as pd

from stage3_activity_duration import predict_activity_durations

RUN = 'runs/agds_default_staged__paper_predict_20261024'
BASE = '_tmp_s3_baseline_chunk1.parquet'
N = int(os.environ.get('COMPARE_PERSONS', '20000'))

with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
    models = pickle.load(fh)
chunk = next(pd.read_csv(os.path.join(RUN, 'stage2_output.csv'), chunksize=N))

t0 = time.time()
acts = predict_activity_durations(chunk, models, seed=42)
dt = time.time() - t0
print(f'new: {len(acts)} activities in {dt:.1f}s '
      f'({dt / len(chunk) * 1000:.2f} ms/person)', flush=True)

base = pd.read_parquet(BASE)
print(f'baseline rows {len(base):,} vs new {len(acts):,}')
assert len(base) == len(acts), 'row count changed'

ok = True
for col in ('person_id', 'purpose', 'planning_order', 'activity_seq',
            'predicted_duration_min'):
    if col not in base.columns:
        continue
    a = base[col].to_numpy()
    b = acts[col].to_numpy()
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        diff = np.abs(a.astype(float) - b.astype(float))
        same = np.allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9)
        print(f'  {col:24s} identical={same}  max|diff|={np.nanmax(diff):.3g}')
    else:
        same = bool((a.astype(str) == b.astype(str)).all())
        n_diff = int((a.astype(str) != b.astype(str)).sum())
        print(f'  {col:24s} identical={same}  n_diff={n_diff}')
    ok = ok and same

print('\nper-purpose mean duration (baseline vs new):')
cmp = pd.DataFrame({
    'baseline': base.groupby('purpose')['predicted_duration_min'].mean(),
    'new': acts.groupby('purpose')['predicted_duration_min'].mean(),
}).round(3)
cmp['diff'] = (cmp['new'] - cmp['baseline']).round(3)
print(cmp.to_string())
print('\nRESULT:', 'identical' if ok else 'DIFFERS')
