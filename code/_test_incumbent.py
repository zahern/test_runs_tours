"""Checks for _keep_better_solution: a refit must not lower an arm's LL.

Mirrors run 27520087's E2 [work] chain, where the same 13 free parameters were
refit and came back at -4871.329 after -4871.242 (and 15 free: -4870.49 ->
-4870.58). Accepting that made the reported LL depend on refit order.
"""
import os

import numpy as np
import pandas as pd

os.environ['STAGE5_ESTIMATE_MU'] = '0'
import stage5_destination_choice as s5


class Fake:
    def __init__(self, values):
        self.pf = pd.DataFrame({
            'value': {k: float(v) for k, v in values.items()},
            'std_err': {k: 0.1 for k in values},
            'holdfast': {k: False for k in values},
        })

    def set_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)

    def lock_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)
        self.pf.loc[name, 'holdfast'] = True

    def vals(self):
        return {k: round(float(v), 4) for k, v in self.pf['value'].items()}


# 1. First fit becomes the incumbent; returned unchanged.
m = Fake({'a': -1.0, 'b': -2.0})
assert s5._keep_better_solution(m, -100.0) == -100.0
assert m._incumbent_ll == -100.0
print('first fit recorded:', m.vals())

# 2. A better refit of the same set replaces it.
m.set_value('a', -1.1)
assert s5._keep_better_solution(m, -98.5) == -98.5
assert m._incumbent_ll == -98.5
assert round(m._incumbent_coef['a'], 4) == -1.1
print('better refit accepted:', m.vals())

# 3. A WORSE refit of the same set is rejected and the vector restored.
m.set_value('a', -1.9)                      # optimiser stalled elsewhere
kept = s5._keep_better_solution(m, -110.0)
assert kept == -98.5, kept
assert round(float(m.pf.loc['a', 'value']), 4) == -1.1, m.vals()
assert int(getattr(m, '_refit_regressions', 0)) == 1
print('worse refit rejected, vector restored:', m.vals(), 'kept LL', kept)

# 4. Equal LL is accepted silently (no restore, no counter bump).
m.set_value('a', -1.1)
assert s5._keep_better_solution(m, -98.5) == -98.5
assert int(getattr(m, '_refit_regressions', 0)) == 1
print('equal fit: no-op')

# 5. After the parameter SET changes, a lower LL is legitimate and accepted.
m.lock_value('b', 0.0)                      # prune/collapse locks a term
m.set_value('a', -1.4)
kept = s5._keep_better_solution(m, -120.0)
assert kept == -120.0, kept
assert round(float(m.pf.loc['a', 'value']), 4) == -1.4
assert m._incumbent_sig == ('a',)
print('signature change (b locked): lower LL accepted:', m.vals())

# 6. NaN likelihood never overwrites or restores anything.
assert np.isnan(s5._keep_better_solution(Fake({'a': -1.0}), float('nan')))
print('NaN LL: passthrough')

print('\nALL INCUMBENT CHECKS PASSED')
