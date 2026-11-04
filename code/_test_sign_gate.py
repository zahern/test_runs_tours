"""Unit check for _enforce_sign_gate using a stubbed parameter frame.

Covers: drop of a non-protected censored coefficient (locked 0.0 + bound
rebound + one refit), interior passthrough, at-bound-but-significant
passthrough, protected-term rescue by restart, protected-term censored
fallback, ASC exclusion, env-off passthrough, and a multi-term sweep.
"""
import os

import numpy as np
import pandas as pd

os.environ.pop('STAGE5_SIGN_GATE', None)
import stage5_destination_choice as s5


class Fake:
    """Minimal stand-in for a fitted Larch model (pf + lock_value only)."""

    def __init__(self, rows):
        # rows: {name: (value, std_err, sign)}  sign in {'neg','pos',None}
        data = {}
        for n, (v, se, sign) in rows.items():
            data[n] = {
                'value': float(v),
                'std_err': float(se),
                'holdfast': False,
                'minimum': float('-inf') if sign != 'pos' else 0.0,
                'maximum': float('inf') if sign != 'neg' else 0.0,
            }
        self.pf = pd.DataFrame(data).T
        self.fits = 0
        self.locked = []

    def lock_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)
        self.pf.loc[name, 'holdfast'] = True
        self.locked.append(name)

    def set_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)


def refit(target=None):
    """estimate_fn that optionally moves params (to emulate a real refit)."""
    def _f(mdl):
        mdl.fits += 1
        for n, (v, se) in (target or {}).items():
            if n in mdl.pf.index and not bool(mdl.pf.loc[n, 'holdfast']):
                mdl.pf.loc[n, 'value'] = float(v)
                mdl.pf.loc[n, 'std_err'] = float(se)
        return mdl
    return _f


def show(tag, mdl):
    free = {n: (round(float(mdl.pf.loc[n, 'value']), 4),
                bool(mdl.pf.loc[n, 'holdfast']))
            for n in mdl.pf.index}
    cens = getattr(mdl, '_sign_censored', None)
    print(f"{tag:34s} refits={mdl.fits} locked={mdl.locked} "
          f"censored={cens}\n{'':34s} {free}")


# 1. Non-protected 'neg' coefficient on the bound, |t|~0 -> dropped at 0.0.
m = Fake({'c_toll_auto': (-0.001, 0.25, 'neg'),
          'c_ivt_auto': (-5.12, 0.43, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 1, f'expected one refit, got {out.fits}'
assert float(out.pf.loc['c_toll_auto', 'value']) == 0.0
assert bool(out.pf.loc['c_toll_auto', 'holdfast']) is True
assert float(out.pf.loc['c_toll_auto', 'maximum']) == 0.0, \
    'bound must be rebound to 0.0 or _estimate re-clips it to -0.001'
assert float(out.pf.loc['c_ivt_auto', 'value']) == -5.12
assert not bool(out.pf.loc['c_ivt_auto', 'holdfast'])
show('bound, unprotected -> drop', out)

# 2. Interior coefficient -> untouched, no refit.
m = Fake({'c_ivt_auto': (-5.12, 0.43, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 0 and not out.locked
show('interior -> passthrough', out)

# 3. On the bound but significantly different from it (tiny SE) -> kept.
m = Fake({'c_toll_auto': (-0.001, 1e-5, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 0 and not out.locked
show('bound but |t|=100 -> keep', out)

# 4. Protected term (c_ivt_auto is in stage5_mode's backbone): restart refit
#    reveals a genuine interior estimate -> rescued, kept free.
m = Fake({'c_ivt_auto': (-0.001, 0.25, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode',
                            refit({'c_ivt_auto': (-5.12, 0.43)}), label='work')
assert out.fits == 1 and not out.locked, 'protected term must not be dropped'
assert abs(float(out.pf.loc['c_ivt_auto', 'value']) + 5.12) < 1e-9
assert not bool(out.pf.loc['c_ivt_auto', 'holdfast'])
assert getattr(out, '_sign_censored', None) is None, 'rescued term is not censored'
show('protected -> rescued', out)

# 5. Protected term that lands on the bound again -> kept but flagged censored.
m = Fake({'c_ivt_transit': (-0.001, 0.5, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 1 and not out.locked
assert out._sign_censored == ['c_ivt_transit']
show('protected -> censored flag', out)

# 6. ASC_* is share-calibrated, never estimated -> always skipped.
m = Fake({'ASC_Walk': (0.0, 0.0, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 0 and not out.locked
show('ASC_Walk at 0.0 -> skip', out)

# 7. Env off -> passthrough.
os.environ['STAGE5_SIGN_GATE'] = '0'
m = Fake({'c_toll_auto': (-0.001, 0.25, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 0 and not out.locked
show('STAGE5_SIGN_GATE=0', out)
os.environ.pop('STAGE5_SIGN_GATE')

# 8. 'pos' coefficient pinned at its lower bound -> dropped the same way.
m = Fake({'c_aggl': (0.001, 3.0, 'pos')})
out = s5._enforce_sign_gate(m, 'stage5_dest', refit(), label='edu')
assert out.fits == 1 and float(out.pf.loc['c_aggl', 'value']) == 0.0
show('pos bound -> drop', out)

# 9. Two censored terms in one pass -> both dropped, single refit.
m = Fake({'c_toll_auto': (-0.001, 0.25, 'neg'),
          'c_xwait_transit': (-0.001, 0.4, 'neg'),
          'c_ivt_auto': (-5.12, 0.43, 'neg')})
out = s5._enforce_sign_gate(m, 'stage5_mode', refit(), label='work')
assert out.fits == 1, f'expected a single refit, got {out.fits}'
assert sorted(out.locked) == ['c_toll_auto', 'c_xwait_transit']
show('two censored -> one refit', out)

# 10. The real 20261030 mirror pair, end to end: E1 has c_ivt_auto censored
#     (protected -> censored flag), E2 has c_ivt_transit censored.
e1 = Fake({'c_ivt_auto': (-0.001, 0.25, 'neg'),
           'c_ivt_transit': (-0.14, 0.42, 'neg')})
e1 = s5._enforce_sign_gate(e1, 'stage5_mode', refit(), label='E1')
e2 = Fake({'c_ivt_auto': (-5.12, 0.43, 'neg'),
           'c_ivt_transit': (-0.001, 0.5, 'neg')})
e2 = s5._enforce_sign_gate(e2, 'stage5_mode', refit(), label='E2')
assert e1._sign_censored == ['c_ivt_auto']
assert e2._sign_censored == ['c_ivt_transit']
show('E1 mirror pair', e1)
show('E2 mirror pair', e2)

print('\nALL SIGN-GATE CHECKS PASSED')
