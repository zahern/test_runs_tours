#!/usr/bin/env python
"""PLC work-segment A/B test (Stage 5, fixed default spec — no GA search).

Runs run_stage5(segments=['work']) twice on the HHTS estimation sample:
  A. STAGE5_ENABLE_PLC = False  (current NL behaviour, control)
  B. STAGE5_ENABLE_PLC = True   (NL + PLC captivity refinement)

371-zone inputs: tour OTAZ/DTAZ are derived as 9-digit SA2s from the tour's
own trips (ORIGSA2_2021/DESTSA2_2021), skims come from demand_/skims_371.omx
(SA2 keys), and landuse defaults to saved_data/zone_landuse_overture.csv
(371 SA2s) via USE_OVERTURE_LANDUSE=1. The retired ORDER-keyed
downcasted_matrices_base.omx must NOT be used here — every SA2 lookup
would miss into mat[0,0].

The module flag is read at runtime (prep + _run_segment), so toggling
in-process between the two runs is exact. Compares predicted vs observed
work mode shares and reports captivity shares + the PLC accept/reject log.

Usage:
    python -u plc_work_test.py [--outdir runs/plc_work_test]

Outputs (under outdir):
    plc_work_ab.json                       — machine-readable comparison
    stage5_models_plc_off.pkl / _on.pkl    — fitted models for each arm
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from data_connect import Synth_Connect
import stage5_destination_choice as s5mod
from stage5_destination_choice import (
    run_stage5,
    _filter_seg,
    _coef_dict_from_model,
    _plc_payload_from_cache,
)

MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
# tours_processed.csv stores tour_mode as STRINGS (DA/Other/Walk/Bike/Transit);
# the pipeline standardizes to 1-5 internally, but share reporting here must
# map first (job 25294943: to_numeric on strings -> all-NaN -> empty shares).
MODE_STR_TO_CODE = {'DA': 1, 'DRIVEALONE': 1, 'SR': 2, 'OTHER': 2,
                    'SHARED': 2, 'WALK': 3, 'BIKE': 4, 'TRANSIT': 5}

PLC_LOG_LINES = []


class _PLCLogCollector(logging.Handler):
    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if 'PLC' in msg or 'captivity' in msg or 'captive' in msg:
            PLC_LOG_LINES.append(msg)


def _load(sc):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(base_dir, 'saved_data')
    persons = pd.read_excel(os.path.join(base_dir, '..', '2_QTS_PERSONS.xlsx'))
    household = pd.read_excel(
        os.path.join(base_dir, '..', '1_QTS_HOUSEHOLDS.xlsx'))
    persons = sc.process_hhts_person(
        persons.merge(household, on='HHID', how='left'))
    if 'person_id' not in persons.columns and 'PERSID' in persons.columns:
        persons = persons.rename(columns={'PERSID': 'person_id'})
    sa1_to_sa2 = pd.read_csv(
        os.path.join(base_dir, '..', 'SA1_SHP', 'sa1_2021.csv'))
    sa1_to_sa2 = (sa1_to_sa2.set_index('SA1_CODE21')['SA2_CODE21']
                  .astype(int).to_dict())
    persons['HOME_SA2_2021'] = persons['HOME_SA1_2021'].map(sa1_to_sa2)
    persons['HOME_SA1'] = pd.to_numeric(
        persons['HOME_SA1_2021'], errors='coerce').fillna(0).astype(np.int64)
    tours = pd.read_csv(os.path.join(saved_dir, 'tours_processed.csv'))
    trips = pd.read_csv(os.path.join(saved_dir, 'trips_processed.csv'))
    # ── 371-zone key alignment (Sep-12 fix) ──────────────────────────
    # tours_processed.csv carries legacy ORDER tour_origin/tour_destination
    # (1-330) but the 371 pipeline (skims_371.omx, SA2-keyed landuse) needs
    # 9-digit SA2 OTAZ/DTAZ. Derive them from the tour's own trips (which
    # carry ORIGSA2_2021/DESTSA2_2021) instead of the retired legacy OMX:
    #   origin = modal ORIGSA2 (home dominates) ;
    #   dest   = modal non-origin DESTSA2 (primary out-of-home destination),
    #            falling back to modal DESTSA2 for genuinely intra-SA2 tours.
    # _prep_hhts_tours prefers OTAZ/DTAZ when present, so setting them here
    # takes precedence over tour_origin/tour_destination (kept untouched).
    # Unmapped (no SA2 trips — 0 cases on current data) -> fringe 999, which
    # quarantine/fringe handling covers by design (never 0 = silent mat[0,0]).
    try:
        _trips = trips.copy()
        _trips['tid'] = (_trips['person_id'].astype(str) + '_' +
                         _trips['tour_id'].astype(str))
        _tours = tours.copy()
        _tours['tid'] = (_tours['person_id'].astype(str) + '_' +
                         _tours['tour_id'].astype(str))

        def _tour_sa2(g):
            _o = pd.to_numeric(g['ORIGSA2_2021'],
                               errors='coerce').dropna()
            _d = pd.to_numeric(g['DESTSA2_2021'],
                               errors='coerce').dropna()
            _o_sa2 = int(_o.mode().iloc[0]) if len(_o) else 999
            if len(_d) == 0:
                return (_o_sa2, 999)
            _dnh = _d[_d != _o_sa2]
            _pool = _dnh if len(_dnh) else _d
            return (_o_sa2, int(_pool.mode().iloc[0]))

        _sa2 = _trips.groupby('tid').apply(_tour_sa2, include_groups=False)
        _tours['OTAZ'] = _tours['tid'].map(
            _sa2.apply(lambda x: x[0])).fillna(999).astype(int)
        _tours['DTAZ'] = _tours['tid'].map(
            _sa2.apply(lambda x: x[1])).fillna(999).astype(int)
        _n_fringe = int((_sa2.apply(lambda x: 999 in x)).sum())
        print(f'tour SA2 alignment: OTAZ 9-digit '
              f'{(_tours["OTAZ"] >= 100000000).mean():.1%}, '
              f'DTAZ 9-digit '
              f'{(_tours["DTAZ"] >= 100000000).mean():.1%}, '
              f'fringe-999 tours={_n_fringe}', flush=True)
        tours = _tours.drop(columns=['tid'])
    except Exception as e:
        print(f'tour SA2 alignment skipped ({e}) — '
              f'falling back to ORDER tour_origin/tour_destination', flush=True)
    return tours, trips, persons


def _shares(series):
    s = pd.Series(series)
    # Map string modes (tours_processed.csv) to 1-5 codes first.
    _mapped = s.astype(str).str.strip().str.upper().map(MODE_STR_TO_CODE)
    s = pd.to_numeric(_mapped.where(_mapped.notna(), s), errors='coerce')
    vc = s.value_counts(normalize=True, dropna=True)
    return {MODE_NAMES.get(int(k), str(k)): float(v) for k, v in vc.items()}


def _run_arm(tag, plc_on, ctx):
    s5mod.STAGE5_ENABLE_PLC = bool(plc_on)
    del PLC_LOG_LINES[:]
    out = {'tag': tag, 'plc_on': bool(plc_on), 'ok': False}
    try:
        plan, models = run_stage5(
            trips=ctx['trips'], persons=ctx['persons'],
            synth_s4=ctx['synth_s4'], skims=ctx['skims'],
            landuse=ctx['land'], tours=ctx['tours'], seed=42,
            save_path=os.path.join(ctx['outdir'],
                                   f'stage5_models_plc_{tag}.pkl'),
            segments=['work'], verbose=False,
            hhts_persons=ctx['persons'],
        )
    except Exception as e:
        import traceback
        out['error'] = f'{type(e).__name__}: {e}'
        out['trace'] = traceback.format_exc()[-2000:]
        out['plc_log'] = list(PLC_LOG_LINES)
        return out
    out['ok'] = True
    try:
        # Prediction plans carry mode/mode_name (per-trip rows), not
        # pred_mode — accept any of them so the comparison never prints
        # nan% on a successful run.
        _pmode = None
        if plan is not None:
            for _cand in ('pred_mode', 'mode_name', 'mode'):
                if _cand in plan.columns:
                    _pmode = plan[_cand]
                    break
        out['pred_shares'] = (_shares(_pmode) if _pmode is not None else {})
        out['n_pred'] = int(len(plan)) if plan is not None else 0
    except Exception as e:
        out['pred_shares'] = {}
        out['pred_error'] = str(e)
    try:
        mres = (models or {}).get('work', {}) or {}
        mdl = mres.get('mode_model')
        out['mode_model_type'] = type(mdl).__name__ if mdl is not None else None
        out['is_plc_proxy'] = (type(mdl).__name__ == '_PLCModelProxy')
        cache = _coef_dict_from_model(mdl, None) if mdl is not None else {}
        blocks = _plc_payload_from_cache(cache)
        out['plc_payload_modes'] = [b[0] for b in blocks] if blocks else []
        out['plc_n_params'] = (sum(len(b[1]) for b in blocks)
                               if blocks else 0)
    except Exception as e:
        out['payload_error'] = str(e)
    out['plc_log'] = list(PLC_LOG_LINES)
    # Validity gate. `_collect` can return ok=True with an EMPTY pred_shares
    # dict (no plan column, or _shares() raised), and with a 0-parameter PLC
    # proxy. Downstream that reads as a successful A/B arm with no predictions.
    # Observed in runs/plc_work_test_371: "ok": true with "pred_shares": {},
    # "plc_n_params": 0 and "is_plc_proxy": true, accepted on a nonsensical
    # ΔBIC. An arm with no predicted shares and no PLC parameters has not
    # demonstrated anything, so mark it invalid rather than passing.
    _n_shares = len(out.get('pred_shares') or {})
    if out.get('ok') and _n_shares == 0:
        out['ok'] = False
        out['invalid_reason'] = ('no predicted mode shares produced '
                                 '(pred_shares empty) - arm proves nothing')
        PLC_LOG_LINES.append(
            f"[work] PLC arm marked INVALID: empty pred_shares "
            f"(plc_n_params={out.get('plc_n_params', 0)})")
    return out


def main():
    outdir = 'runs/plc_work_test'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for i, a in enumerate(sys.argv[1:]):
        if a == '--outdir' and i + 1 < len(sys.argv[1:]):
            outdir = sys.argv[1:][i + 1]
    os.makedirs(outdir, exist_ok=True)

    s5mod.logger.addHandler(_PLCLogCollector())
    s5mod.logger.setLevel(logging.INFO)

    print('=== PLC work-segment A/B: loading data ===', flush=True)
    sc = Synth_Connect(new_pop=True)
    tours, trips, persons = _load(sc)
    land = sc.get_land()
    work_tours = _filter_seg(tours, 'work')
    print(f'HHTS tours={len(tours)} work_tours={len(work_tours)} '
          f'persons={len(persons)}', flush=True)
    obs = _shares(work_tours['tour_mode']) if 'tour_mode' in work_tours else {}
    print('observed work shares: ' + ', '.join(
        f'{k}={v:.1%}' for k, v in sorted(obs.items())), flush=True)

    synth_s4 = tours.sample(n=min(2000, len(tours)), random_state=42).copy()
    # Prediction needs OTAZ/DTAZ. _load sets 9-digit SA2 OTAZ/DTAZ above, so
    # this is a no-op fallback (kept for frames that bypass _load):
    # mirror _prep_hhts_tours resolution; estimation standardizes internally.
    if 'OTAZ' not in synth_s4.columns and 'tour_origin' in synth_s4.columns:
        synth_s4['OTAZ'] = pd.to_numeric(
            synth_s4['tour_origin'], errors='coerce').fillna(0).astype(int)
    if 'DTAZ' not in synth_s4.columns and 'tour_destination' in synth_s4.columns:
        synth_s4['DTAZ'] = pd.to_numeric(
            synth_s4['tour_destination'], errors='coerce').fillna(0).astype(int)
    # ── 371-zone inputs (Sep-12 fix) ─────────────────────────────────
    # Retired downcasted_matrices_base.omx (ORDER 1-330 keys) cannot serve
    # SA2-keyed tours — every lookup would miss into mat[0,0]. Resolve the
    # 371 skim absolutely (CWD-proof: the script may run from code/ or the
    # run dir) and fail loudly if it is absent rather than silently
    # degrading (STAGE5_STRICT_SKIMS hard-fails legacy files elsewhere).
    _skim_path = os.path.normpath(os.path.join(
        base_dir, '..', 'demand_', 'skims_371.omx'))
    if not os.path.isfile(_skim_path):
        raise SystemExit(f'FATAL: 371 skim not found at {_skim_path} — '
                         f'refusing to fall back to the retired legacy OMX')
    # 371-zone landuse (zone_landuse_overture.csv: 371 SA2s, strict superset
    # of the "357"-named baseline which also holds 371 SA2s despite its
    # stale name). setdefault so an explicit env export still wins; get_land
    # falls back to baseline with a printed note if the file is missing.
    os.environ.setdefault('USE_OVERTURE_LANDUSE', '1')
    ctx = {'tours': tours, 'trips': trips, 'persons': persons, 'land': land,
           'synth_s4': synth_s4, 'skims': _skim_path,
           'outdir': outdir}

    res = {'observed_work_shares': obs, 'n_work_tours': int(len(work_tours))}
    for tag, plc_on in (('off', False), ('on', True)):
        print(f'=== arm {tag} (PLC={plc_on}) starting ===', flush=True)
        arm = _run_arm(tag, plc_on, ctx)
        res[tag] = arm
        print(f'=== arm {tag} done ok={arm["ok"]} ===', flush=True)

    with open(os.path.join(outdir, 'plc_work_ab.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)

    print()
    print('MODE-SHARE COMPARISON (work segment)')
    print(f'{"mode":<8}{"observed":>10}{"pred PLC-off":>13}{"pred PLC-on":>12}')
    for m in ['DA', 'SR', 'Walk', 'Bike', 'Transit']:
        o = res['observed_work_shares'].get(m, float('nan'))
        a = res.get('off', {}).get('pred_shares', {}).get(m, float('nan'))
        b = res.get('on', {}).get('pred_shares', {}).get(m, float('nan'))
        print(f'{m:<8}{o:>10.1%}{a:>13.1%}{b:>12.1%}')
    print()
    print('PLC arm detail:', json.dumps(
        {k: v for k, v in res.get('on', {}).items()
         if k in ('ok', 'is_plc_proxy', 'plc_payload_modes',
                  'plc_n_params', 'n_pred', 'error')}, indent=2))
    print('PLC log lines (on-arm):')
    for line in res.get('on', {}).get('plc_log', []):
        print('  [plc] ' + line)
    if not res.get('on', {}).get('ok'):
        sys.exit(1)


if __name__ == '__main__':
    main()
