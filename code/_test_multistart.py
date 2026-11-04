"""Checks for _multistart_refit.

Models the real failure: E2 [nonmand] converged to -9256.83 although its family
contains E1's fit at -6848.59. Here the stub likelihood has two basins — a global
one (matching E1's) and a local one 780 LL worse — and a start inside the local
basin stays there unless the fit is restarted from elsewhere.
"""
import os

import numpy as np
import pandas as pd

os.environ['STAGE5_ESTIMATE_MU'] = '0'
import stage5_destination_choice as s5

GLOBAL_B, LOCAL_B = -1.24, -2.02      # coefficient at each basin
LL_GLOBAL, LL_LOCAL = -6848.59, -7628.59


class Fake:
    def __init__(self, values):
        self.pf = pd.DataFrame({
            'value': {k: float(v) for k, v in values.items()},
            'std_err': {k: 0.1 for k in values},
            'holdfast': {k: False for k in values},
        })
        self._cached_ll = -1000.0

    def set_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)

    def loglike(self):
        return self._cached_ll


def make_estimator(basins, calls=None):
    """Converges to the basin nearest the starting value, like a local optimiser."""
    def _est(mdl):
        start = float(mdl.pf.loc['c_walk_time', 'value'])
        target, ll = min(basins, key=lambda b: abs(b[0] - start))
        mdl.set_value('c_walk_time', target)
        if calls is not None:
            calls.append(round(start, 4))
        mdl._cached_ll = s5._keep_better_solution(mdl, ll)
        return mdl
    return _est


BIMODAL = [(GLOBAL_B, LL_GLOBAL), (LOCAL_B, LL_LOCAL)]

# 1. A single start parked in the local basin is rescued by extra starts.
m = Fake({'c_walk_time': -1.90, 'c_bike_time': -1.31, 'ASC_Walk': 0.92})
out = s5._multistart_refit(m, make_estimator(BIMODAL), label='nonmand',
                           n_starts=8, jitter=0.5)
assert s5._model_ll(out) == LL_GLOBAL, s5._model_ll(out)
assert abs(float(out.pf.loc['c_walk_time', 'value']) - GLOBAL_B) < 1e-9, \
    out.pf.loc['c_walk_time', 'value']
print(f'rescued: LL={s5._model_ll(out):.2f}, c_walk_time='
      f'{float(out.pf.loc["c_walk_time", "value"]):.3f}')

# 2. If no start improves, the original fit is kept unchanged.
m2 = Fake({'c_walk_time': LOCAL_B, 'c_bike_time': -1.0})
m2._cached_ll = s5._keep_better_solution(m2, LL_LOCAL)
out2 = s5._multistart_refit(m2, make_estimator(BIMODAL), label='work',
                            n_starts=2, jitter=0.01)
assert s5._model_ll(out2) == LL_LOCAL, s5._model_ll(out2)
assert abs(float(out2.pf.loc['c_walk_time', 'value']) - LOCAL_B) < 1e-9
print(f'no improvement: kept LL={s5._model_ll(out2):.2f} '
      f'(not the {LL_GLOBAL:.2f} of another basin)')

# 3. n_starts<=0 is a no-op.
calls = []
m3 = Fake({'c_walk_time': -1.9})
s5._multistart_refit(m3, make_estimator(BIMODAL, calls), n_starts=0)
assert calls == []
print('n_starts=0: no refits')

# 4. Held coefficients are never jittered.
m4 = Fake({'c_walk_time': -1.9, 'c_ivt_auto': 0.0})
m4.pf.loc['c_ivt_auto', 'holdfast'] = True
s5._multistart_refit(m4, make_estimator(BIMODAL), label='held', n_starts=2)
assert float(m4.pf.loc['c_ivt_auto', 'value']) == 0.0
print('held coefficient untouched')

# 5. Reproducible start sequence for the same seed.
def walk(seed):
    mm = Fake({'c_walk_time': -1.9, 'c_bike_time': -1.1})
    seen = []
    s5._multistart_refit(mm, make_estimator(BIMODAL, seen), n_starts=4, seed=seed)
    return seen
assert walk(7) == walk(7) and walk(7) != walk(11)
print('deterministic seed: same starts on re-run, different seed differs')

print('\nALL MULTISTART CHECKS PASSED')
