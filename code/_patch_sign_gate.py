"""Patch: reject sign-censored coefficients instead of reporting them.

Adds `_enforce_sign_gate` / `_sign_gate_candidates` to
stage5_destination_choice.py, makes _estimate's post-clip audible, and calls the
gate at the two places a Stage-5 model is finalised (mode, destination).

Idempotent: re-running it is a no-op once the markers are present.
Line endings in the target file are CRLF; the inserted block is normalised.
"""
import sys

TARGET = 'stage5_destination_choice.py'

BLOCK = '''

def _sign_bound_of(model, p: str, sign: str) -> float:
    """The declared sign bound a 'neg'/'pos' parameter is constrained against."""
    try:
        col = 'maximum' if sign == 'neg' else 'minimum'
        _pf = model.pf
        if col in _pf.columns:
            _v = float(_pf.loc[p, col])
            if np.isfinite(_v):
                return _v
    except Exception:
        pass
    return 0.0


def _sign_gate_candidates(model, stage: str, eps: float, tmin: float):
    """Free coefficients that sit ON a declared sign bound and are not
    significantly different from it — i.e. censored, not estimated.

    Detection is the same shape as `_mu_scales_valid`: at bound + insignificant
    means the data did not locate the parameter, so the printed value is set by
    the prior, not by the sample.
    """
    try:
        from ga_constraints import sign_constraints as _sc
        signs = _sc(stage) or {}
    except Exception:
        return []
    try:
        _pf = getattr(model, '_final_pf', None)
        if _pf is None:
            _pf = model.pf
    except Exception:
        return []
    _ts = _param_tstats(model) or {}
    _out = []
    for p, s in signs.items():
        if p not in _pf.index or s not in ('neg', 'pos'):
            continue
        if p.startswith('ASC_'):
            continue                      # share-calibrated, never estimated
        try:
            if bool(_pf.loc[p].get('holdfast', False)):
                continue
            _val = float(_pf.loc[p, 'value'])
        except Exception:
            continue
        if not np.isfinite(_val):
            continue
        _b = _sign_bound_of(model, p, s)
        if abs(_val - _b) > eps:
            continue                      # interior: a real estimate
        _t = _ts.get(p, float('nan'))
        if np.isfinite(_t) and abs(_t) >= tmin:
            continue                      # at the bound but significantly off it
        _out.append((p, s, _val, _b, _t))
    return _out


def _enforce_sign_gate(model, stage: str, estimate_fn, *,
                       label: str = '', max_iters: int = 2):
    """Reject a sign-censored coefficient instead of reporting it as an estimate.

    `_apply_sign_constraints` (and `_estimate`'s post-clip) enforce a sign prior
    by *writing* a placeholder value at the bound (-0.001 for 'neg', 0.001 for
    'pos') whenever the unconstrained estimate violates the prior. Nothing then
    re-checks that value, so it is printed, saved and compared as if the sample
    had produced it. Run 20261030/27485123 exposed the cost of that: E1 and E2
    each reported exactly one of c_ivt_auto / c_ivt_transit at -0.001000 with
    |t| ~ 0.004, in mirror image (E1: auto censored; E2: transit censored), and
    the apparent +1063/+2529 mode gain was largely which term got clamped
    rather than anything the search selected.

    Contract, mirroring `_enforce_mu_gate` — detect, act once, refit, re-check:

      * non-protected term  → lock at 0.0 and refit once. The term leaves the
        utility and the free-parameter count, which is the honest reading of
        "the data says this is not a deterrent".
      * protected/structural term (ga_constraints.protected_backbone, e.g. the
        mode time coefficients) → refit once from a start moved off the bound.
        If a genuine interior estimate exists the optimizer finds it and the
        term is kept; if it lands on the bound again the term is flagged
        `model._sign_censored` and logged at ERROR rather than reported.

    Loop stops as soon as nothing more is censored, so a clean model costs one
    inspection and no refit.
    """
    if os.environ.get('STAGE5_SIGN_GATE', '1').strip() == '0':
        return model
    try:
        _eps = float(os.environ.get('STAGE5_SIGN_BOUND_EPS', '2e-3'))
    except Exception:
        _eps = 2e-3
    # Same |t| cut the prune rule uses, so one knob (GA_SIG_PVAL_THRESH) governs
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
    try:
        from ga_constraints import protected_backbone as _pb
        _protected = set(_pb(stage))
    except Exception:
        _protected = set()
    _restart = os.environ.get('STAGE5_SIGN_RESTART', '1').strip() != '0'
    _tag = f"[sign {label or stage}]"

    for _it in range(max_iters):
        _cands = _sign_gate_candidates(model, stage, _eps, _tmin)
        if not _cands:
            if _it == 0:
                logger.info(f"{_tag} no free coefficient sits on a sign bound")
            return model
        _ll0 = _model_ll(model)
        _dropped, _censored = [], []
        for _p, _s, _val, _b, _t in _cands:
            _tt = 'nan' if not np.isfinite(_t) else f'{_t:+.2f}'
            if _p in _protected and _restart:
                # Structural term: never dropped silently. Try to reveal an
                # interior estimate by restarting away from the bound.
                try:
                    model.set_value(_p, -max(0.05, _eps * 10.0) if _s == 'neg'
                                    else max(0.05, _eps * 10.0))
                    model = estimate_fn(model)
                except Exception as _re:
                    logger.warning(f"{_tag} {_p}: restart refit failed ({_re!r})")
                    _censored.append(_p)
                    continue
                try:
                    _now = float(model.pf.loc[_p, 'value'])
                except Exception:
                    _now = float('nan')
                _t2 = (_param_tstats(model) or {}).get(_p, float('nan'))
                if (np.isfinite(_now) and abs(_now - _b) > _eps
                        and np.isfinite(_t2) and abs(_t2) >= _tmin):
                    logger.warning(
                        f"{_tag} {_p} rescued by restart: {_val:+.4f} "
                        f"(|t|={_tt}) -> {_now:+.4f} (|t|={_t2:+.2f})")
                else:
                    logger.error(
                        f"{_tag} {_p} is unidentified at its sign bound "
                        f"({_val:+.4f}, |t|={_tt}); protected, so kept but "
                        f"marked CENSORED — do not read it as an estimate")
                    # Restore the constrained optimum: the restart nudge is an
                    # arbitrary point and must not masquerade as the estimate.
                    try:
                        model.set_value(_p, _val)
                    except Exception:
                        _pf_set(model, _p, 'value', _val)
                    _censored.append(_p)
            else:
                try:
                    # Bind at exactly 0.0 first: _estimate restores pf maxima and
                    # would otherwise re-clip a 0.0 value back onto a -0.001
                    # bound, resurrecting the placeholder we are removing.
                    if _s == 'neg':
                        _pf_set(model, _p, 'maximum', 0.0)
                    model.lock_value(_p, 0.0)
                    _dropped.append(_p)
                except Exception as _le:
                    logger.warning(f"{_tag} cannot lock {_p} at 0.0: {_le!r}")
                    _censored.append(_p)
        if _dropped:
            logger.warning(
                f"{_tag} iter {_it + 1}: dropping sign-censored {_dropped} "
                f"(locked at 0.0 — term leaves the utility)")
            try:
                model = estimate_fn(model)
            except Exception as _de:
                logger.warning(f"{_tag} refit after sign gate failed: {_de!r}")
                _dropped = []
            if _dropped:
                _d = (_model_ll(model) - _ll0
                      if np.isfinite(_ll0) else float('nan'))
                if np.isfinite(_d):
                    logger.info(f"{_tag} refit after sign gate: dLL={_d:+.3f}")
                else:
                    logger.info(f"{_tag} refit after sign gate complete")
        if _censored:
            try:
                _prev = set(getattr(model, '_sign_censored', []) or [])
                model._sign_censored = sorted(_prev | set(_censored))
            except Exception:
                pass
        if not _dropped:
            return model
    return model

'''

OLD_POSTCLIP = (
    '                    logger.debug(\n'
    '                        f"_estimate: post-clip {_sn}={_sv:.4f} \u2192 -0.001 "\n'
    '                        "(sign constraint enforced after optimization)"\n'
    '                    )\n'
)
NEW_POSTCLIP = (
    '                    # Audible, and recorded: this is a prior-imposed value,\n'
    '                    # not an estimate. _enforce_sign_gate rejects it later\n'
    '                    # instead of letting it be reported.\n'
    '                    try:\n'
    '                        _clip = dict(getattr(model, \'_sign_clipped\', {}) or {})\n'
    '                        _clip[_sn] = _sv\n'
    '                        model._sign_clipped = _clip\n'
    '                    except Exception:\n'
    '                        pass\n'
    '                    logger.warning(\n'
    '                        f"_estimate: post-clip {_sn}={_sv:+.4f} -> -0.001 "\n'
    '                        "(sign constraint enforced after optimization; "\n'
    '                        "value is prior-imposed, not estimated)"\n'
    '                    )\n'
)

OLD_MU_CALL = (
    '    mode_mdl = _enforce_mu_gate(mode_mdl, segment, _estimate, \'post-refit\')\n'
)
NEW_MU_CALL = (
    '    # Reject sign-censored coefficients (a -0.001 placeholder is not an\n'
    '    # estimate) BEFORE the Mu gate, so the Mu gate stays the last refit in\n'
    '    # the chain and its verdict covers anything the sign refit moved.\n'
    '    mode_mdl = _enforce_sign_gate(mode_mdl, \'stage5_mode\', _estimate,\n'
    '                                  label=segment)\n'
    '    mode_mdl = _enforce_mu_gate(mode_mdl, segment, _estimate, \'post-refit\')\n'
)

OLD_DEST = (
    "            dest_mdl = _apply_sign_constraints(dest_mdl, 'stage5_dest')\n"
    '            try:\n'
    '                _ll_after = float(dest_mdl.loglike())\n'
)
NEW_DEST = (
    "            dest_mdl = _apply_sign_constraints(dest_mdl, 'stage5_dest')\n"
    '            # Same contract as the mode model: a sign-censored destination\n'
    '            # coefficient is dropped (or flagged if structural), never\n'
    '            # reported as an estimate.\n'
    "            dest_mdl = _enforce_sign_gate(\n"
    "                dest_mdl, 'stage5_dest',\n"
    '                lambda _m: _estimate(_m, two_phase=True), label=segment)\n'
    '            try:\n'
    '                _ll_after = float(dest_mdl.loglike())\n'
)

ANCHOR_BUILD = '\r\n\r\ndef _build_mode_model(dt, segment: str, custom_mode_vars=None, avail_vars_hint=None,'


def crlf(s):
    return s.replace('\r\n', '\n').replace('\n', '\r\n')


def main():
    src = open(TARGET, encoding='utf-8', newline='').read()
    changed = []

    if '_enforce_sign_gate' in src:
        print('gate already present — block insertion skipped')
    else:
        assert src.count(ANCHOR_BUILD) == 1, 'build anchor not unique'
        blk = crlf(BLOCK)
        src = src.replace(ANCHOR_BUILD, blk.rstrip('\n') + '\r\n\r\n\r\ndef _build_mode_model(dt, segment: str, custom_mode_vars=None, avail_vars_hint=None,')
        changed.append('inserted _enforce_sign_gate')

    for old, new, name in (
        (OLD_POSTCLIP, NEW_POSTCLIP, 'post-clip log'),
        (OLD_MU_CALL, NEW_MU_CALL, 'mode gate call'),
        (OLD_DEST, NEW_DEST, 'dest gate call'),
    ):
        _o, _n = crlf(old), crlf(new)
        if _o not in src:
            print(f'{name}: anchor absent (already patched?)')
            continue
        assert src.count(_o) == 1, f'{name}: anchor not unique'
        src = src.replace(_o, _n if _n.endswith('\r\n') else _n)
        changed.append(name)

    if changed:
        open(TARGET, 'w', encoding='utf-8', newline='').write(src)
        print('patched:', ', '.join(changed))
    else:
        print('nothing to do')
    return 0


if __name__ == '__main__':
    sys.exit(main())
