"""Patch: carry sign-censoring into the dashboard, and demote the per-fit clip log.

1. Factor the |t| cut into `_sign_gate_tmin()` so `_enforce_sign_gate` and the
   behavioural sign checks cannot drift apart.
2. Demote `_estimate`'s per-fit post-clip message from WARNING to INFO: it fires
   on most search fits, so a warning there is log noise. The recording
   (`model._sign_clipped`) and the gate's own WARNING/ERROR are unchanged.
3. `_check_behavioural_signs` currently passes a censored -0.001 as "Auto IVT
   deters driving" because -0.001 < 0. Mark such a coefficient as FAILED with the
   censoring spelled out, so the dashboard CSV can never present a prior-imposed
   value as evidence for the behavioural sign.

Idempotent; CRLF-aware.
"""
import sys

TARGET = 'stage5_destination_choice.py'

TMIN_HELPER = '''

def _sign_gate_tmin() -> float:
    """The |t| cut that separates "identified" from "at the bound".

    Same knobs the pruning rule uses, so one setting (GA_SIG_PVAL_THRESH, or an
    explicit GA_S5_PRUNE_TMIN) governs both.
    """
    _t_env = os.environ.get('GA_S5_PRUNE_TMIN', '').strip()
    if _t_env:
        try:
            return float(_t_env)
        except Exception:
            return 1.6
    try:
        from scipy.stats import norm as _norm
        return float(_norm.isf(float(os.environ.get('GA_SIG_PVAL_THRESH', '0.10')) / 2.0))
    except Exception:
        return 1.6

'''

INLINE_TMIN = '''    # Same |t| cut the prune rule uses, so one knob (GA_SIG_PVAL_THRESH) governs
    # "identified" across pruning and the sign gate alike.
    _t_env = os.environ.get('GA_S5_PRUNE_TMIN', '').strip()
    if _t_env:
        try:
            _tmin = float(_t_env)
        except Exception:
            _tmin = 1.6
    else:
        try:
            from scipy.stats import norm as _norm
            _tmin = float(_norm.isf(float(os.environ.get('GA_SIG_PVAL_THRESH', '0.10')) / 2.0))
        except Exception:
            _tmin = 1.6
'''

NEW_TMIN = '''    # Same |t| cut the prune rule uses, so one knob (GA_SIG_PVAL_THRESH) governs
    # "identified" across pruning and the sign gate alike.
    _tmin = _sign_gate_tmin()
'''

OLD_POSTCLIP_HEAD = '''                    logger.warning(
                        f"_estimate: post-clip {_sn}={_sv:+.4f} -> -0.001 "'''
NEW_POSTCLIP_HEAD = '''                    # INFO, not WARNING: this fires on most search fits, so a
                    # warning here would bury the gate's real rejections. The
                    # value is recorded on the model and the gate reports it.
                    logger.info(
                        f"_estimate: post-clip {_sn}={_sv:+.4f} -> -0.001 "'''

OLD_COEF = '''            coef = (res.get('dest_coef' if mtype == 'dest_model' else 'mode_coef')
                    or {})
'''
NEW_COEF = '''            coef = (res.get('dest_coef' if mtype == 'dest_model' else 'mode_coef')
                    or {})
            # Coefficients whose value came from the sign prior rather than the
            # sample. _enforce_sign_gate records these; re-detecting them here
            # keeps the check honest for a model reloaded from disk too.
            _censored = set(getattr(mdl, '_sign_censored', []) or [])
            try:
                _censored |= {c[0] for c in _sign_gate_candidates(
                    mdl,
                    'stage5_dest' if mtype == 'dest_model' else 'stage5_mode',
                    float(os.environ.get('STAGE5_SIGN_BOUND_EPS', '2e-3')),
                    _sign_gate_tmin())}
            except Exception:
                pass
'''

OLD_VAL = '''                val = float(coef[pname])
                if np.isnan(val):
'''
NEW_VAL = '''                val = float(coef[pname])
                _cens = pname in _censored
                if np.isnan(val):
'''

OLD_EPS = '''                elif abs(val) < 1e-6:
'''
NEW_EPS = '''                elif _cens:
                    # The reported value sits on the sign bound because the
                    # unconstrained estimate violated the prior. It is a
                    # placeholder, so the behavioural sign is NOT demonstrated.
                    ok = False
                    note = note + ' => CENSORED at sign bound (prior-imposed value, not an estimate)'
                elif abs(val) < 1e-6:
'''

ANCHOR_CANDS = '\r\n\r\ndef _sign_gate_candidates(model, stage: str, eps: float, tmin: float):'


def crlf(s):
    return s.replace('\r\n', '\n').replace('\n', '\r\n')


def main():
    src = open(TARGET, encoding='utf-8', newline='').read()
    changed = []

    if '_sign_gate_tmin' not in src:
        assert src.count(ANCHOR_CANDS) == 1, 'candidate anchor not unique'
        src = src.replace(ANCHOR_CANDS,
                          crlf(TMIN_HELPER).rstrip('\n') + ANCHOR_CANDS)
        changed.append('added _sign_gate_tmin')

    for old, new, name in (
        (INLINE_TMIN, NEW_TMIN, 'gate uses helper'),
        (OLD_POSTCLIP_HEAD, NEW_POSTCLIP_HEAD, 'post-clip to INFO'),
        (OLD_COEF, NEW_COEF, 'censored set in sign check'),
        (OLD_VAL, NEW_VAL, 'censored lookup'),
        (OLD_EPS, NEW_EPS, 'censored branch'),
    ):
        _o, _n = crlf(old), crlf(new)
        if _o not in src:
            print(f'{name}: anchor absent (already patched?)')
            continue
        assert src.count(_o) == 1, f'{name}: anchor not unique'
        src = src.replace(_o, _n)
        changed.append(name)

    if changed:
        open(TARGET, 'w', encoding='utf-8', newline='').write(src)
        print('patched:', ', '.join(changed))
    else:
        print('nothing to do')
    return 0


if __name__ == '__main__':
    sys.exit(main())
