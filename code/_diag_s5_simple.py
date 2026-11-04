"""Minimal stage5 diagnostic - write each step to stdout for immediate visibility."""
import traceback
import numpy as np
import pandas as pd

# Step 1: build synthetic data (no imports yet)
SEED = 42; N = 600
rng = np.random.default_rng(SEED)
_age  = rng.integers(5, 80, N).astype(float)
_vehrd = rng.integers(0, 4, N).astype(float)
_is_female = (rng.random(N) < 0.5).astype(float)
_worker = (rng.random(N) < 0.5).astype(float)
_incp = rng.normal(0, 1, N)
_nprd = rng.integers(1, 5, N).astype(float)
_purposes = rng.choice(['work', 'edu', 'social', 'shopping'], N)
MODES = ['DA', 'SR', 'Transit', 'Walk', 'Bike']
_chosen_mode = rng.choice(MODES, N)

synth_persons = pd.DataFrame({
    'person_id': np.arange(N),
    'vehrd': _vehrd, 'sexp': np.where(_is_female == 1, 2, 1),
    'nprd': _nprd, 'WORKSTAT': _worker * 2, 'STUDYING': np.zeros(N),
    'incp': np.clip(_incp * 3 + 8, 0, 20),
    'age5p': np.clip((_age / 5).astype(int), 0, 15),
    'CARLICENCE': (_vehrd > 0).astype(int).astype(float),
    'indp': np.random.default_rng(SEED + 99).integers(1, 20, N).astype(float),
})
synth_trips = pd.DataFrame({
    'person_id': np.arange(N), 'purpose': _purposes,
    'chosen_mode': _chosen_mode, 'mode': _chosen_mode,
})

print(f"synth_trips cols: {list(synth_trips.columns)}", flush=True)
print(f"synth_persons cols: {list(synth_persons.columns)}", flush=True)

# Step 2: call prepare_mode_data
print("\n--- prepare_mode_data ---", flush=True)
try:
    from stage6_mode_choice import prepare_mode_data
    md = prepare_mode_data(synth_trips, synth_persons, pd.DataFrame())
    print(f"OK  shape={md.shape}", flush=True)
    print(f"    cols={list(md.columns)}", flush=True)
    sub = md[md['purpose'] == 'work']
    print(f"    work rows={len(sub)}, modes={sub['chosen_mode'].unique()}", flush=True)
except Exception:
    traceback.print_exc()

# Step 3: call NestedLogitModeModel.fit
print("\n--- NestedLogitModeModel.fit ---", flush=True)
try:
    from stage6_mode_choice import NestedLogitModeModel, prepare_mode_data
    md2 = prepare_mode_data(synth_trips, synth_persons, pd.DataFrame())
    sub2 = md2[md2['purpose'] == 'work'].copy()
    print(f"  rows={len(sub2)}", flush=True)
    m = NestedLogitModeModel()
    m.fit(sub2, 'work')
    print(f"  ll_final_={m.ll_final_}  n_obs_={m.n_obs_}", flush=True)
except Exception:
    traceback.print_exc()

print("\nDONE", flush=True)
