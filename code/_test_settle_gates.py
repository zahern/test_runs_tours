"""Regression test for _settle_scales_and_signs.

Reproduces the real sequence that run 20261102/27508262 exposed in E1:

  round 1  sign gate catches c_ivt_transit at -0.0010 (|t|~0) -> protected,
           restart refit, still censored -> flagged
           Mu gate rejects Mu_Motor=0.1000 -> locks 0.8 -> REFIT
           ... and that refit moves the censoring onto c_ivt_auto, leaving
           c_ivt_transit interior. Nothing used to look again.

The settle loop must (a) re-scan after the Mu refit, (b) catch c_ivt_auto,
(c) end with _sign_censored describing the FINAL vector (auto, not transit).
"""
import os

import numpy as np
import pandas as pd

os.environ['STAGE5_ESTIMATE_MU'] = '1'
os.environ.pop('STAGE5_SIGN_GATE', None)
import stage5_destination_choice as s5

BOUND = -0.001


class Fake:
    """Minimal stand-in for a Larch model: pf frame + set/lock + cached LL."""

    def __init__(self, rows, ll=-5000.0):
        self.pf = pd.DataFrame(rows).T
        for c, d in (('holdfast', False), ('minimum', -np.inf), ('maximum', np.inf)):
            if c not in self.pf.columns:
                self.pf[c] = d
        self.pf['holdfast'] = self.pf['holdfast'].astype(bool)
        self._cached_ll = float(ll)
        self.fits = 0
        self.history = []

    def set_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)

    def lock_value(self, name, value):
        self.pf.loc[name, 'value'] = float(value)
        self.pf.loc[name, 'holdfast'] = True
        self.history.append(('lock', name, float(value)))

    def loglike(self):
        return self._cached_ll

    def snapshot(self):
        return {n: (round(float(self.pf.loc[n, 'value']), 4),
                    bool(self.pf.loc[n, 'holdfast']))
                for n in ('c_ivt_auto', 'c_ivt_transit', 'Mu_Motor', 'Mu_NonMotor')}


def make_model():
    # c_ivt_transit starts ON the bound with |t| ~ 0 (censored after prune);
    # c_ivt_auto interior; Mu_Motor on the 0.10 floor so the Mu gate fires.
    return Fake({
        'c_ivt_auto':    {'value': -4.0, 'std_err': 0.4},
        'c_ivt_transit': {'value': BOUND, 'std_err': 0.3, 'maximum': 0.0},
        'Mu_Motor':      {'value': 0.1000, 'std_err': 0.09},
        'Mu_NonMotor':   {'value': 0.4500, 'std_err': 0.09},
    })


def estimate_fn(mdl):
    """A refit. Once Mu is locked at 0.8 the mirror pair flips: transit goes
    interior and the censoring lands on c_ivt_auto exactly as observed."""
    mdl.fits += 1
    mdl._cached_ll -= 1.0 * mdl.fits
    mu_locked = bool(mdl.pf.loc['Mu_Motor', 'holdfast'])
    if mu_locked:
        mdl.pf.loc['c_ivt_transit', 'value'] = -0.2080
        mdl.pf.loc['c_ivt_transit', 'std_err'] = 0.3258
        if not bool(mdl.pf.loc['c_ivt_auto', 'holdfast']):
            mdl.pf.loc['c_ivt_auto', 'value'] = BOUND
            mdl.pf.loc['c_ivt_auto', 'std_err'] = 0.2806
    mdl.history.append(('refit', mdl.fits, mdl.snapshot()))
    return mdl


# ── The old two-call sequence is NOT settled ────────────────────────────────
m = make_model()
m = s5._enforce_sign_gate(m, 'stage5_mode', estimate_fn, label='nonmand')
m = s5._enforce_mu_gate(m, 'nonmand', estimate_fn, 'post-refit')
print('after one pass each:', m.snapshot(), 'censored =', getattr(m, '_sign_censored', None))
assert float(m.pf.loc['c_ivt_auto', 'value']) == BOUND, 'stub should reproduce the drift'
assert float(m.pf.loc['c_ivt_auto', 'std_err']) > 0
_left = s5._sign_gate_candidates(m, 'stage5_mode', 2e-3, s5._sign_gate_tmin())
assert [c[0] for c in _left] == ['c_ivt_auto'], (
    'the drifted term must be detectable at this point (it was simply never '
    f'looked at); got {[c[0] for c in _left]}')

# ── The settle loop must catch it ───────────────────────────────────────────
import logging


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


cap = Capture()
s5.logger.addHandler(cap)

m = make_model()
m = s5._settle_scales_and_signs(m, 'nonmand', estimate_fn, label='nonmand')
s5.logger.removeHandler(cap)
print('after settle:       ', m.snapshot(), 'censored =', getattr(m, '_sign_censored', None))
print('refits:', m.fits)
_stuck = [r.getMessage() for r in cap.records
          if r.levelno >= logging.ERROR and 'still acting' in r.getMessage()]
assert not _stuck, f'settle loop failed to converge: {_stuck}'
assert m.fits == 3, (
    f'expected 3 refits (sign restart, Mu lock, one sign restart on the newly '
    f'censored term) and none for the already-flagged one; got {m.fits}')
assert float(m.pf.loc['Mu_Motor', 'value']) == 0.8, 'Mu must stay locked'
assert bool(m.pf.loc['Mu_Motor', 'holdfast']) is True
cens = set(getattr(m, '_sign_censored', []) or [])
assert 'c_ivt_auto' in cens, f'censored c_ivt_auto must be flagged, got {cens}'
assert 'c_ivt_transit' not in cens, f'stale flag survived from an intermediate fit: {cens}'
assert float(m.pf.loc['c_ivt_transit', 'value']) == -0.208
_left = s5._sign_gate_candidates(m, 'stage5_mode', 2e-3, s5._sign_gate_tmin())
assert {c[0] for c in _left} == cens, 'flags must match the reported vector exactly'

# ── A clean model must not be refit at all ─────────────────────────────────
FakeRows = {
    'c_ivt_auto':    {'value': -4.2, 'std_err': 0.36},
    'c_ivt_transit': {'value': -1.8, 'std_err': 0.60},
    'Mu_Motor':      {'value': 0.80, 'std_err': 0.05, 'holdfast': True},
    'Mu_NonMotor':   {'value': 0.80, 'std_err': 0.05, 'holdfast': True},
}
m2 = Fake(FakeRows)
out = s5._settle_scales_and_signs(m2, 'nonmand', estimate_fn, label='nonmand')
print('clean model:        ', out.snapshot(), 'refits:', out.fits,
      'censored =', getattr(out, '_sign_censored', None))
assert out.fits == 0, 'a settled model must cost one scan and no refit'
assert not getattr(out, '_sign_censored', None)

print('\nALL SETTLE-GATE CHECKS PASSED')
