import os
import numpy as np
import pandas as pd
from data_connect import Synth_Connect
from stage5_destination_choice import run_stage5


def sig_ratio_from_pf(m):
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


def write_report(models, segments, report_path):
    lines = ['STAGE5_SMOKE_SUMMARY']
    for seg in segments:
        if seg not in models:
            lines.append(f'{seg} segment missing in stage5 models')
            continue
        for mtype, key in [('mode', 'mode_model'), ('dest', 'dest_model')]:
            mdl = models[seg].get(key)
            if mdl is None:
                lines.append(f'{seg} {mtype}: missing model')
                continue
            ratio, sig, tot = sig_ratio_from_pf(mdl)
            lines.append(f'{seg} {mtype}: sig={sig}/{tot} ratio={ratio:.3f}')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'WROTE {report_path}')


SMOKE_DATA = os.environ.get('SMOKE_DATA', 'legacy').strip().lower()
REPORT_PATH = os.environ.get('SMOKE_REPORT', 'runs/stage5_smoke_report.txt')

if SMOKE_DATA == 'hhts':
    # 371-zone verification: real HHTS estimation frame (same load_data the
    # GA shards use) + real 371 skims, so the vmap-vs-numpy self-verify runs
    # on production shapes instead of the degenerate 330 legacy fixture.
    from pipeline_logger import load_data
    sc, out, landuse = load_data()
    _n = int(os.environ.get('SMOKE_N_TOURS', '1500'))
    _segs = [s.strip() for s in os.environ.get('SMOKE_SEGMENTS', 'work').split(',')
             if s.strip()]
    _skim = (os.environ.get('PIPELINE_SKIMS_PATH')
             or '../demand_/skims_371.omx')
    _save = os.environ.get('SMOKE_OUT',
                           'runs/vmap_verify_371/stage5_smoke_models.pkl')
    synth_s4 = out['tours'].sample(n=min(_n, len(out['tours'])),
                                   random_state=42).copy()
    plan, models = run_stage5(
        out['trips'],
        out['persons'],
        synth_s4,
        _skim,
        landuse,
        tours=out['tours'],
        seed=42,
        save_path=_save,
        segments=_segs,
        verbose=True,
    )
    write_report(models, _segs, REPORT_PATH)
    # Fail-closed vmap verification: with STAGE5_VMAP_VERIFY=1 the run must
    # prove the vmap path was taken AND self-verified — never pass vacuously.
    if os.environ.get('STAGE5_VMAP_VERIFY', '0').strip() == '1':
        import stage5_destination_choice as _s5
        _info = dict(getattr(_s5, '_VMAP_LAST', {}) or {})
        # vmap logsum path is active only when requested (STAGE5_VMAP_LOGSUM=1)
        # AND not force-disabled by PT submodes (the vmap kernel does not cover
        # the PT_Drive/PT_Bike split). When disabled by config there is nothing
        # to self-verify, so SKIP rather than fail vacuously; only a genuinely
        # requested-but-unused path fails.
        _vmap_enabled = (os.environ.get('STAGE5_VMAP_LOGSUM', '0').strip() == '1'
                         and os.environ.get('STAGE5_PT_MODES', '0').strip() != '1')
        if not _vmap_enabled:
            print("VMAP VERIFY SKIPPED: vmap logsum path not enabled "
                  "(needs STAGE5_VMAP_LOGSUM=1 and STAGE5_PT_MODES!=1); "
                  f"nothing to verify (info={_info})")
        elif not _info.get('used'):
            print('VMAP VERIFY FAILED: vmap path was requested but never taken '
                  f"(info={_info}) — refusing to pass vacuously")
            raise SystemExit(1)
        else:
            print(f"VMAP VERIFY PASSED: vmap path taken, "
                  f"maxdiff={_info.get('maxdiff')}, note={_info.get('note')}")
else:
    sc = Synth_Connect()
    out = sc.process_survey(save=False)
    land = sc.get_land()
    synth_s4 = out['tours'].sample(n=min(2500, len(out['tours'])), random_state=42).copy()
    if len(synth_s4) > 1200:
        synth_s4 = synth_s4.sample(n=1200, random_state=42).copy()
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
    write_report(models, ['work'], REPORT_PATH)
