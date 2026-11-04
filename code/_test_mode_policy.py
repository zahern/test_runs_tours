"""Checks for the declared coefficient policy (STAGE5_MODE_HOLDFAST).

The point of the policy is narrow and testable: a held coefficient must be
pinned at exactly 0.0, marked holdfast, excluded from the free-parameter
count, and invisible to the sign gate — so it can never be censored and the
E1/E2 arms cannot diverge on it.
"""
import os

import numpy as np
import pandas as pd

os.environ.pop('STAGE5_MODE_HOLDFAST', None)
os.environ['STAGE5_ESTIMATE_MU'] = '0'
import stage5_destination_choice as s5


class Fake:
    def __init__(self, rows):
        self.pf = pd.DataFrame(rows).T
        for c, d in (('holdfast', False), ('minimum', -np.inf), ('maximum', np.inf)):
            if c not in self.pf.columns:
                self.pf[c] = d
        # rows() omits holdfast on some params; NaN must read as False, not True
        self.pf['holdfast'] = self.pf['holdfast'].fillna(False).astype(bool)
        self._cached_ll = -5000.0

    def lock_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)
        self.pf.loc[name, 'holdfast'] = True


def rows(auto_val, transit_val, held=False):
    return {
        'c_ivt_auto':    {'value': auto_val, 'std_err': 0.28, 'holdfast': held},
        'c_ivt_transit': {'value': transit_val, 'std_err': 0.42, 'holdfast': held},
        'c_walk_time':   {'value': -1.0, 'std_err': 0.02},
    }


# ── 1. Policy parsing ──────────────────────────────────────────────────────
os.environ.pop('STAGE5_MODE_HOLDFAST', None)
assert s5._mode_holdfast_policy() == set()
os.environ['STAGE5_MODE_HOLDFAST'] = ' c_ivt_auto , c_ivt_transit ,, '
assert s5._mode_holdfast_policy() == {'c_ivt_auto', 'c_ivt_transit'}
os.environ['STAGE5_MODE_HOLDFAST'] = '   '
assert s5._mode_holdfast_policy() == set()
print('parse: empty / spaced / blank all handled')

# ── 2. The unheld model still censors, as the old runs did ─────────────────
m = Fake(rows(-0.001, -0.208))
cands = [c[0] for c in s5._sign_gate_candidates(m, 'stage5_mode', 2e-3,
                                                s5._sign_gate_tmin())]
assert cands == ['c_ivt_auto'], cands
assert s5._model_n_free(m) == 3
print('unheld: sign gate sees c_ivt_auto, free=3')

# ── 3. Held at 0.0 exactly as the policy applies it ───────────────────────
m = Fake(rows(-4.0, -0.9))
for p in ('c_ivt_auto', 'c_ivt_transit'):
    m.lock_value(p, 0.0)
assert float(m.pf.loc['c_ivt_auto', 'value']) == 0.0
assert bool(m.pf.loc['c_ivt_auto', 'holdfast']) is True
assert s5._sign_gate_candidates(m, 'stage5_mode', 2e-3,
                                s5._sign_gate_tmin()) == []
# 3 params, 2 held -> exactly one free
assert s5._model_n_free(m) == 1
print('held: no sign candidates, free=1 (was 3)')

# ── 4. A held term cannot be moved by the gates (both are no-ops) ─────────
def estimate_fn(mdl):                      # must never be called
    raise AssertionError('held model must not be refit')


out = s5._settle_scales_and_signs(m, 'work', estimate_fn, label='work')
assert float(out.pf.loc['c_ivt_auto', 'value']) == 0.0
assert not getattr(out, '_sign_censored', None)
print('settle on held model: 0 refits, value stays 0.0, nothing censored')

# ── 5. Holding only one term leaves the other free and censorable ─────────
m = Fake(rows(-0.001, -0.9))
m.lock_value('c_ivt_transit', 0.0)
assert [c[0] for c in s5._sign_gate_candidates(m, 'stage5_mode', 2e-3,
                                               s5._sign_gate_tmin())] == ['c_ivt_auto']
assert s5._model_n_free(m) == 2
print('partial hold: only the unheld term is still censored, free=2')

os.environ.pop('STAGE5_MODE_HOLDFAST', None)
print('\nALL MODE-POLICY CHECKS PASSED')
