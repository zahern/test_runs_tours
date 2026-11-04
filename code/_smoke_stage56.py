import traceback
import numpy as np
import pandas as pd

from data_connect import Synth_Connect


def run_stage6(out, land, lines):
    from stage6_mode_choice import estimate_mode_models
    models = estimate_mode_models(out['trips'], out['persons'], landuse=land, min_sample=80)
    rows = []
    for purpose, model in models.items():
        t = np.asarray(model.tstat_ if model.tstat_ is not None else [])
        tf = t[np.isfinite(t)]
        sig = int((np.abs(tf) > 1.96).sum())
        tot = int(tf.size)
        rows.append({
            'purpose': purpose,
            'n_obs': int(model.n_obs_),
            'rho2': float(model.rho2_) if model.rho2_ is not None else np.nan,
            'sig_ratio': (sig / tot) if tot > 0 else np.nan,
            'sig_count': sig,
            'tot_count': tot,
            'engine': model._estimation_engine,
        })
    df = pd.DataFrame(rows).sort_values('purpose')
    lines.append('STAGE6_SMOKE_SUMMARY')
    lines.append(df.to_string(index=False))


def _sig_ratio_from_pf(m):
    pf = m._final_pf if hasattr(m, '_final_pf') and isinstance(getattr(m, '_final_pf'), pd.DataFrame) else m.pf
    tcol = 't_stat' if 't_stat' in pf.columns else ('tstat' if 'tstat' in pf.columns else None)
    if tcol is None:
        return (np.nan, 0, 0)
    t = pd.to_numeric(pf[tcol], errors='coerce')
    free = (~pf['holdfast'].fillna(False)) if 'holdfast' in pf.columns else pd.Series(True, index=pf.index)
    tf = t[free & t.notna()]
    sig = int((tf.abs() > 1.96).sum())
    tot = int(tf.shape[0])
    return ((sig / tot) if tot > 0 else np.nan, sig, tot)


def run_stage5(out, land, lines):
    from stage5_destination_choice import run_stage5

    synth_s4 = out['tours'].sample(n=min(2500, len(out['tours'])), random_state=42).copy()
    plan, models = run_stage5(
        out['trips'],
        out['persons'],
        synth_s4,
        'downcasted_matrices_base.omx',
        land,
        tours=out['tours'],
        seed=42,
        save_path='runs/stage5_smoke_models.pkl',
        segments=['work'],
        verbose=False,
    )

    lines.append('STAGE5_SMOKE_SUMMARY')
    if 'work' not in models:
        lines.append('work segment missing in stage5 models')
        return

    work = models['work']
    for mtype, key in [('mode', 'mode_model'), ('dest', 'dest_model')]:
        mdl = work.get(key)
        if mdl is None:
            lines.append(f'{mtype}: missing model')
            continue
        ratio, sig, tot = _sig_ratio_from_pf(mdl)
        lines.append(f'{mtype}: sig={sig}/{tot} ratio={ratio:.3f}')


if __name__ == '__main__':
    lines = []
    try:
        sc = Synth_Connect()
        out = sc.process_survey(save=False)
        land = sc.get_land()
        run_stage6(out, land, lines)
    except Exception as e:
        lines.append('STAGE6_SMOKE_ERROR')
        lines.append(repr(e))
        lines.append(traceback.format_exc())

    try:
        # reload fresh to avoid any accidental mutation carryover
        sc2 = Synth_Connect()
        out2 = sc2.process_survey(save=False)
        land2 = sc2.get_land()
        run_stage5(out2, land2, lines)
    except Exception as e:
        lines.append('STAGE5_SMOKE_ERROR')
        lines.append(repr(e))
        lines.append(traceback.format_exc())

    with open('runs/stage56_smoke_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('WROTE runs/stage56_smoke_report.txt')
