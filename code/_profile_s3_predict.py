"""Profile one Stage 3 prediction chunk to locate the hot path."""
import cProfile
import io
import os
import pickle
import pstats
import time

import pandas as pd

from stage3_activity_duration import predict_activity_durations

RUN = 'runs/agds_default_staged__paper_predict_20261024'
CHUNK = int(os.environ.get('PROFILE_CHUNK', '20000'))

with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
    models = pickle.load(fh)
chunk = next(pd.read_csv(os.path.join(RUN, 'stage2_output.csv'),
                         chunksize=CHUNK))
print(f'chunk: {len(chunk)} persons, {chunk.shape[1]} columns', flush=True)

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
acts = predict_activity_durations(chunk, models, seed=42)
pr.disable()
dt = time.time() - t0
print(f'\n{len(acts)} activities in {dt:.1f}s '
      f'({dt / len(chunk) * 1000:.2f} ms/person)', flush=True)

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(28)
out = s.getvalue().splitlines()
start = next(i for i, line in enumerate(out) if 'ncalls' in line)
print('\n'.join(out[start:start + 32]))
