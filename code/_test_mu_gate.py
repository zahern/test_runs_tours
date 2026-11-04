"""Unit check for _enforce_mu_gate using a stubbed parameter frame.

Covers: floor/cap rejection + one fixed-scale refit, interior acceptance,
already-holdfast passthrough, insignificant rejection, env-off passthrough,
and the real failure sequence (passes at fit time, drifts to the floor during
a later refit, caught by gate #2).
"""
import os

import numpy as np
import pandas as pd

os.environ.setdefault('STAGE5_ESTIMATE_MU', '1')
import stage5_destination_choice as s5


class Fake:
    def __init__(self, mus, holdfast=False, se=0.1):
        rows = {n: {'value': float(v), 'std_err': float(se), 'holdfast': holdfast}
                for n, v in mus.items()}
        self.pf = pd.DataFrame(rows).T
        self.fits = 0
        self.locked = []

    def lock_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)
        self.pf.loc[name, 'holdfast'] = True
        self.locked.append(name)


def refit_that_pins_to_floor(mdl):
    """A later prune/income refit: free Mus drift to the 0.10 floor."""
    mdl.fits += 1
    for n in mdl.pf.index:
        if n.startswith('Mu_') and n != 'Mu_Root' and not bool(mdl.pf.loc[n, 'holdfast']):
            mdl.pf.loc[n, 'value'] = 0.1000
    return mdl


def refit_respecting_locks(mdl):
    mdl.fits += 1
    return mdl


def show(tag, mdl):
    vals = {n: round(float(mdl.pf.loc[n, 'value']), 4)
            for n in mdl.pf.index if n.startswith('Mu_') and n != 'Mu_Root'}
    hf = {n: bool(mdl.pf.loc[n, 'holdfast'])
          for n in mdl.pf.index if n.startswith('Mu_') and n != 'Mu_Root'}
    print(f"{tag:32s} refits={mdl.fits} Mu={vals} holdfast={hf}")


os.environ['STAGE5_ESTIMATE_MU'] = '1'

# 1. Scale sitting exactly on the floor -> rejected, locked, one refit.
m = Fake({'Mu_Motor': 0.1000, 'Mu_NonMotor': 0.45})
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 1, 'expected one fixed-scale refit'
assert float(out.pf.loc['Mu_Motor', 'value']) == 0.8
assert bool(out.pf.loc['Mu_Motor', 'holdfast']) is True
show('floor 0.1000', out)

# 2. Scale at the MNL cap -> rejected.
m = Fake({'Mu_Motor': 0.55, 'Mu_NonMotor': 1.0000})
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 1 and float(out.pf.loc['Mu_NonMotor', 'value']) == 0.8
show('cap 1.0000', out)

# 3. Interior + significant -> accepted, no refit, untouched.
m = Fake({'Mu_Motor': 0.45, 'Mu_NonMotor': 0.70})
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 0 and float(out.pf.loc['Mu_Motor', 'value']) == 0.45
show('interior 0.45/0.70', out)

# 4. Insignificant (|t| < 1) -> rejected even though interior.
m = Fake({'Mu_Motor': 0.45, 'Mu_NonMotor': 0.60}, se=0.9)
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 1
show('interior but |t|<1', out)

# 5. Already holdfast -> passthrough (gate must not refit a fixed model).
m = Fake({'Mu_Motor': 0.80, 'Mu_NonMotor': 0.80}, holdfast=True)
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 0
show('already holdfast', out)

# 6. Env off -> passthrough even at the floor.
os.environ['STAGE5_ESTIMATE_MU'] = '0'
m = Fake({'Mu_Motor': 0.1000, 'Mu_NonMotor': 0.45})
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 0
show('STAGE5_ESTIMATE_MU=0', out)
os.environ['STAGE5_ESTIMATE_MU'] = '1'

# 7. The real sequence: passes at fit time, a later refit drags it to the
#    floor, gate #2 must catch it.
m = Fake({'Mu_Motor': 0.45, 'Mu_NonMotor': 0.70})
m = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert m.fits == 0, 'initial gate should have accepted'
m = refit_that_pins_to_floor(m)          # prune / income re-estimate
assert float(m.pf.loc['Mu_Motor', 'value']) == 0.1000
m = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'post-refit')
assert m.fits == 2, 'gate #2 should add exactly one refit'
assert float(m.pf.loc['Mu_Motor', 'value']) == 0.8
assert bool(m.pf.loc['Mu_Motor', 'holdfast']) is True
show('pass -> drift -> gate #2', m)

# 8. Flat structure (no Mu params at all) -> passthrough.
m = Fake({'ASC_SR': 0.2})
out = s5._enforce_mu_gate(m, 'work', refit_respecting_locks, 'initial fit')
assert out.fits == 0
show('flat structure', out)

print('\nALL MU-GATE CHECKS PASSED')
