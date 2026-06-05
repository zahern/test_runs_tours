"""Diagnostic: surface the real exception from _stage5_trip_fitness."""
import traceback, logging, sys, os
import numpy as np
import pandas as pd

# Redirect all output to a file so JAX/Larch stderr doesn't drown results
_OUTFILE = open(r"z:\test_runs_tours\code\_diag_stage5_out.txt", "w", encoding="utf-8")
_orig_out = sys.stdout
_orig_err = sys.stderr

def _p(*args, **kwargs):
    print(*args, **kwargs, file=_OUTFILE, flush=True)
    _OUTFILE.flush()

# Capture stage51 debug messages specifically
class _Stage51Handler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if 'Stage51' in msg or 'stage51' in msg.lower() or 'stage5' in msg.lower():
            _p(f"[LOG {record.levelname}] {msg}")

_h = _Stage51Handler()
_h.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_h)
logging.getLogger().setLevel(logging.DEBUG)

from ga_hypothesis_testing import _stage5_trip_fitness

# ── Recreate same synthetic data as the test ──────────────────────────────
SEED = 42; N = 600
rng = np.random.default_rng(SEED)
_age      = rng.integers(5, 80, N).astype(float)
_incp     = rng.normal(0, 1, N)
_vehrd    = rng.integers(0, 4, N).astype(float)
_nprd     = rng.integers(1, 5, N).astype(float)
_worker   = (rng.random(N) < 0.5).astype(float)
_is_female = (rng.random(N) < 0.5).astype(float)
_purposes_raw = rng.choice(['work', 'edu', 'social', 'shopping'], N)
MODES = ['DA', 'SR', 'Transit', 'Walk', 'Bike']
_chosen_mode = rng.choice(MODES, N)

synth_persons = pd.DataFrame({
    'person_id': np.arange(N),
    'vehrd':     _vehrd,
    'sexp':      np.where(_is_female == 1, 2, 1),
    'nprd':      _nprd,
    'WORKSTAT':  _worker * 2,
    'STUDYING':  np.zeros(N),
    'incp':      np.clip(_incp * 3 + 8, 0, 20),
    'age5p':     np.clip((_age / 5).astype(int), 0, 15),
    'CARLICENCE': (_vehrd > 0).astype(int).astype(float),
    'indp':      np.random.default_rng(SEED + 99).integers(1, 20, N).astype(float),
})
synth_trips = pd.DataFrame({
    'person_id':   np.arange(N),
    'purpose':     _purposes_raw,
    'chosen_mode': _chosen_mode,
    'mode':        _chosen_mode,
})

MODE_CANDS_S5 = [
    'has_vehicle', 'no_vehicle', 'vehrd', 'CARLICENCE',
    'incp', 'income_hi', 'income_lo',
    'is_female', 'sexp',
    'age_working', 'age_older',
    'nprd', 'n_workers_hh',
    'mandatory_role',
]
genes = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0])

# ── Step 1: prepare_mode_data ─────────────────────────────────────────────
_p("\n=== prepare_mode_data ===")
try:
    from stage6_mode_choice import prepare_mode_data
    md = prepare_mode_data(synth_trips, synth_persons, pd.DataFrame())
    _p(f"mode_data shape:  {md.shape}")
    _p(f"columns:          {list(md.columns)}")
    work_sub = md[md['purpose'] == 'work']
    _p(f"work rows:        {len(work_sub)}")
    _p(f"chosen_mode uniq: {work_sub['chosen_mode'].unique()}")
except Exception:
    _p(traceback.format_exc())

# ── Step 2: full fitness call with traceback ──────────────────────────────
_p("\n=== _stage5_trip_fitness (purpose=work) ===")
try:
    result = _stage5_trip_fitness(
        mode_genes      = genes,
        dest_genes      = np.array([], dtype=int),
        mode_candidates = MODE_CANDS_S5,
        dest_candidates = [],
        purpose         = 'work',
        trips           = synth_trips,
        persons         = synth_persons,
        landuse         = pd.DataFrame(),
        skim_df         = pd.DataFrame(),
    )
    _p(f"Result: {result}")
except Exception:
    _p(traceback.format_exc())

# ── Step 3: manual inline version showing where it fails ─────────────────
_p("\n=== manual inline ===")
try:
    from stage6_mode_choice import NestedLogitModeModel, prepare_mode_data
    mode_vars = [v for v, g in zip(MODE_CANDS_S5, genes) if g == 1]
    _p(f"mode_vars: {mode_vars}")
    mode_data = prepare_mode_data(synth_trips, synth_persons, pd.DataFrame())
    _p(f"mode_data cols: {list(mode_data.columns)}")
    mode_data_purp = mode_data[mode_data['purpose'] == 'work'].copy()
    _p(f"mode_data_purp len: {len(mode_data_purp)}, modes: {mode_data_purp['chosen_mode'].nunique()}")
    for col in mode_vars:
        if col not in mode_data_purp.columns:
            _p(f"  Adding zero col: {col}")
            mode_data_purp[col] = 0.0
    _p("Fitting NestedLogitModeModel...")
    m = NestedLogitModeModel()
    m.fit(mode_data_purp, 'work')
    _p(f"ll_final_={m.ll_final_}  n_obs_={m.n_obs_}")
except Exception:
    _p(traceback.format_exc())

_p("\nDone.")
_OUTFILE.close()
