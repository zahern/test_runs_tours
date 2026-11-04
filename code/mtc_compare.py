#!/usr/bin/env python
"""MTC three-arm comparison (Stage 5, work + edu + nonmand tours, TAZ zones).

Arms (all on identical prototype_mtc data/universe):
  M0  ActivitySim standard-model outputs: regress/final_trips.csv tour
      aggregates (mode shares, destination top-N, mean trip length) as the
      citable reference — no fitting here.
  M1  Pipeline default (ActivitySim-style) spec, no search.
  M2  Searched spec (small GA budget) refit + applied.

Needs the mtc_connect download (raw.githubusercontent.com, egress once).
Artifacts: everything under OUTROOT (default runs/mtc_ab, enforced to
contain 'mtc'); NOTHING is written to saved_data/ or any QLD run tree.
Full MTC prototype population is used (all persons/tours in the files).

Usage:  python -u mtc_compare.py [--outdir runs/mtc_ab]
Writes: mtc_metrics.json, mtc_metrics_table.tex, stage5 models, SOURCE.json,
        mapped CSVs.
"""
# NumPy 2.0 compatibility shim - MUST BE FIRST IMPORT
import compat.numpy_shim

import json
import logging
import os
import sys

# 25-zone transfer system: relax the 371-zone skim guard BEFORE any local
# skim import (read_skims reads ZONE_SYSTEM_N_ZONES at import time). Also
# exported by job_mtc_ab.pbs; setdefault keeps direct `python` launches
# working.
os.environ.setdefault('ZONE_SYSTEM_N_ZONES', '25')

import numpy as np
import pandas as pd

import mtc_connect as mtc

logger = logging.getLogger('mtc_compare')


# ── Extensive-search budgets ─────────────────────────────────────────────
# M2 arms default to the small in-file budgets (fast CI-style runs). Override
# via env for benchmark-grade extensive searches without code edits, e.g.
#   MTC_M2_GENS=150 MTC_M2_S5_GENS=120 python -u mtc_compare.py ...
# MTC fits are seconds each (25 zones), so thorough budgets stay cheap.
def _m2_int(name, default):
    try:
        v = int(os.environ.get(name, '').strip() or 0) or 0
        return v if v > 0 else default
    except (ValueError, TypeError, AttributeError):
        return default


_M2_POP = _m2_int('MTC_M2_POP', 12)
_M2_GENS = _m2_int('MTC_M2_GENS', 8)
_M2_RESTARTS = _m2_int('MTC_M2_RESTARTS', 1)
_M2_S5_GENS = _m2_int('MTC_M2_S5_GENS', 10)


# ── metrics helpers (larch-free; same semantics as exampville_compare) ──
def _mode_share_mae(pred_modes, obs_modes):
    po = pd.Series(pd.to_numeric(pred_modes, errors='coerce').dropna())
    oo = pd.Series(pd.to_numeric(obs_modes, errors='coerce').dropna())
    if len(po) == 0 or len(oo) == 0:
        return float('nan')
    pv = po.value_counts(normalize=True)
    ov = oo.value_counts(normalize=True)
    keys = set(pv.index) | set(ov.index)
    return float(np.mean([abs(pv.get(k, 0.0) - ov.get(k, 0.0)) for k in keys]))


def _dest_share_mae(pred_zones, obs_zones, top_n=10):
    pz = pd.Series(pd.to_numeric(pred_zones, errors='coerce').dropna())
    oz = pd.Series(pd.to_numeric(obs_zones, errors='coerce').dropna())
    if len(pz) == 0 or len(oz) == 0:
        return float('nan'), float('nan')
    pv = pz.value_counts(normalize=True)
    ov = oz.value_counts(normalize=True)
    keys = set(pv.index) | set(ov.index)
    mae = float(np.mean([abs(pv.get(k, 0.0) - ov.get(k, 0.0)) for k in keys]))
    top_pred = set(pv.nlargest(top_n).index)
    top_obs = set(ov.nlargest(top_n).index)
    overlap = float(len(top_pred & top_obs) / max(len(top_pred | top_obs), 1))
    return mae, overlap


def _trip_length_stats(home_zones, dest_zones, autodist):
    """Mean DIST(home, dest) over paired tours (Auld-style inputs)."""
    errs = []
    for h, d in zip(home_zones, dest_zones):
        try:
            v = float(autodist.get((int(h), int(d)), np.nan))
        except Exception:
            v = np.nan
        if np.isfinite(v):
            errs.append(v)
    if not errs:
        return float('nan'), 0
    return float(np.mean(errs)), int(len(errs))


def _trip_length_mae(pred_zones, obs_zones, home_zones, autodist):
    """Paired per-tour |T(home,pred) - T(home,obs)| (Auld-style)."""
    try:
        errs = []
        for h, p, o in zip(home_zones, pred_zones, obs_zones):
            try:
                dp = float(autodist.get((int(h), int(p)), np.nan))
                do = float(autodist.get((int(h), int(o)), np.nan))
            except Exception:
                continue
            if np.isfinite(dp) and np.isfinite(do):
                errs.append(abs(dp - do))
        if not errs:
            return float('nan')
        return float(np.mean(errs))
    except Exception:
        return float('nan')


def _larch_ll_stats(mdl):
    """LL / rho2 / n from a fitted Larch model; nulls on genuine failure.

    Recorded LL first: a fresh mdl.loglike() can blow up (~-1e35) or raise
    even for converged models once elastic-net unpatching / JAX state has
    moved on (the _scored_ll lesson in ga_hypothesis_testing). Finite-gated
    at every step so garbage never lands in the metrics.
    """
    import math
    out = {'ll': None, 'rho2': None, 'n': None}
    if mdl is None:
        return out

    def _finite(x, lo=-1e12):
        try:
            v = float(x)
            return v if (math.isfinite(v) and v > lo) else None
        except Exception:
            return None

    for _src in ('_cached_ll',):
        _v = _finite(getattr(mdl, _src, None))
        if _v is not None:
            out['ll'] = _v
            break
    if out['ll'] is None:
        try:
            out['ll'] = _finite(mdl.most_recent_estimation_result.loglike)
        except Exception:
            pass
    if out['ll'] is None:
        try:
            out['ll'] = _finite(mdl.loglike())
        except Exception:
            pass
    try:
        out['rho2'] = _finite(mdl.rho_sq_null(), lo=-1e9)
    except Exception:
        pass
    for _attr in ('datatree',):
        try:
            out['n'] = int(mdl.datatree.n_cases)
            break
        except Exception:
            continue
    if out['n'] is None:
        for _attr in ('n_obs_', 'n_', 'nb_samples'):
            try:
                out['n'] = int(getattr(mdl, _attr))
                break
            except Exception:
                continue
    return out


def _plan_metrics(plan, tours_mapped, autodist):
    out = {}
    try:
        pm = plan['pred_mode'] if 'pred_mode' in plan.columns else None
        tm = tours_mapped['tour_mode']
        out['mode_mae'] = _mode_share_mae(pm, tm) if pm is not None else None
    except Exception:
        out['mode_mae'] = None
    try:
        pz = plan['DTAZ'] if 'DTAZ' in plan.columns else None
        oz = tours_mapped['DTAZ']
        mae, ov = _dest_share_mae(pz, oz) if pz is not None else (None, None)
        out['dest_mae'], out['dest_top10'] = mae, ov
    except Exception:
        out['dest_mae'], out['dest_top10'] = None, None
    try:
        out['n_pred'] = int(len(plan)) if plan is not None else 0
    except Exception:
        out['n_pred'] = 0
    return out


def _fmt(x, digits=4):
    try:
        v = float(x)
        return f'{v:.{digits}f}' if np.isfinite(v) else '---'
    except Exception:
        return '---'


class _SkipStage5(Exception):
    """Raised when the mapped sample is too small for estimation (SKIP, not FAIL)."""


def write_metrics_tex(metrics: dict, path: str):
    """Auto-generated paper table (overwritten on every run)."""
    arms = (metrics.get('arms') or {})
    m0 = arms.get('M0', {})
    with open(path, 'w') as f:
        f.write('% Auto-generated by mtc_compare.py — DO NOT EDIT BY HAND.\n')
        f.write('\\begin{table}[!ht]\n')
        f.write('  \\caption{Prototype-MTC transfer test: ActivitySim standard-model '
                'outputs (M0) vs.\\ pipeline default (M1) and searched (M2) specs}'
                '\\label{tab:mtc_transfer}\n')
        f.write('  \\begin{center}\n')
        f.write('    \\begin{tabular}{lrrrrr}\n')
        f.write('      \\toprule\n')
        f.write('      Arm & Tours & Mode MAE & Dest MAE & Dest top-10 & Trip-len MAE \\\\\n')
        f.write('      \\midrule\n')
        f.write(f"      M0 (ActivitySim regress) & {m0.get('n_tours', '---')} & "
                f"--- & --- & --- & {_fmt(m0.get('mean_trip_len_km'))} km (ref) \\\\\n")
        for _a in ('M1', 'M2'):
            _m = arms.get(_a, {})
            f.write(f"      {_a} ({'default' if _a == 'M1' else 'searched'}) & "
                    f"{_m.get('n_pred', '---')} & {_fmt(_m.get('mode_mae'))} & "
                    f"{_fmt(_m.get('dest_mae'))} & {_fmt(_m.get('dest_top10'))} & "
                    f"{_fmt(_m.get('trip_len_mae'))} \\\\\n")
        f.write('      \\bottomrule\n')
        f.write('    \\end{tabular}\n')
        f.write('  \\end{center}\n')
        _skip = '; '.join(sorted(
            {arms.get(_a, {}).get('reason', '') for _a in ('M1', 'M2')
             if arms.get(_a, {}).get('skipped')}))
        _skip_note = f" Estimation SKIPPED: {(_skip + '.') if _skip else ''}"
        f.write('  \\footnotesize{{MAE = mean absolute error of predicted vs.\\ M0 '
                'shares; top-10 = Jaccard overlap of the ten most-chosen destination '
                'zones; trip-len = mean network distance error. '
                f"M0 observed DA share = {_fmt(m0.get('obs_da_share'))}; "
                f"work-mode LL: M1={_fmt((arms.get('M1', {}).get('work_mode_model') or {}).get('ll'), 1)}, "
                f"M2={_fmt((arms.get('M2', {}).get('work_mode_model') or {}).get('ll'), 1)}.{_skip_note}}}\n")
        f.write('\\end{table}\n')
    print(f'wrote {path}', flush=True)


def main():
    outdir = 'runs/mtc_ab'
    for i, a in enumerate(sys.argv[1:]):
        if a == '--outdir' and i + 1 < len(sys.argv[1:]):
            outdir = sys.argv[1:][i + 1]
    root = mtc.mtc_root(outdir)
    logging.basicConfig(level=logging.INFO)
    # ── Artifact isolation (never touch QLD / Exampville outputs) ──────────
    # GA checkpoints + best_*.npz default to GA_CHECKPOINT_DIR or CWD (the code
    # dir!); batch checkpoints/plots default to PIPELINE_OUT_DIR or the model
    # save dir; _run_segment appends ACDC rows to saved_data/acdc_terms.csv.
    # Pin all three inside this run's root so MTC reruns resume from MTC state
    # only and QLD/Exampville files are never written.
    os.environ.setdefault('GA_CHECKPOINT_DIR', os.path.join(root, 'checkpoints'))
    os.environ.setdefault('PIPELINE_OUT_DIR', root)
    os.environ.setdefault('STAGE5_SAVE_ACDC', '0')
    # No safety skims on transfer fixtures: run every arm crash-free
    # (drops crash params from UECs AND disables the worker CRASH fix).
    os.environ.setdefault('SAFETY_EXPERIMENT_MODE', 'nosafety')
    os.makedirs(os.environ['GA_CHECKPOINT_DIR'], exist_ok=True)
    print(f'=== MTC 3-arm comparison -> {root} ===', flush=True)
    open(mtc.guard_path(root, 'SOURCE.tag'), 'w').write('mtc\n')

    import time
    t_all = time.time()
    metrics = {'source': 'mtc', 'root': root, 'arms': {}}

    # ── download + map ──
    src = mtc.download_mtc_data(root)
    hh = mtc.map_households(pd.read_csv(src['households.csv'], low_memory=False))
    pp = mtc.map_persons(pd.read_csv(src['persons.csv'], low_memory=False), hh)
    lu = mtc.map_landuse(pd.read_csv(src['land_use.csv'], low_memory=False))
    # M0 trips source: regress fixture by default; full ActivitySim model
    # outputs via MTC_TRIPS_CSV (see job_asim_mtc_full.pbs). The full file has
    # the same schema (CI asserts substantive equality with the fixture).
    _trips_override = os.environ.get('MTC_TRIPS_CSV', '')
    if _trips_override and not os.path.isfile(_trips_override):
        raise RuntimeError(f'MTC_TRIPS_CSV points at missing file: {_trips_override}')
    _trips_path = _trips_override or src['regress_final_trips.csv']
    print(f'M0 trips source: {_trips_path}', flush=True)
    metrics['trips_source'] = _trips_path
    regress = pd.read_csv(_trips_path, low_memory=False)

    # OMX (openmatrix preferred, larch.OMX fallback) + slot discovery.
    skim_path = os.path.join(root, 'source', 'skims.omx')
    import shutil
    if not os.path.exists(skim_path):
        shutil.copy2(src['skims.omx'], skim_path)
    omx = None
    for _opener in ('openmatrix', 'larch'):
        try:
            if _opener == 'openmatrix':
                import openmatrix as _om
                omx = _om.open_file(skim_path)
            else:
                import larch as _lx
                omx = _lx.OMX(skim_path, mode='r')
            break
        except Exception:
            omx = None
    if omx is None:
        raise RuntimeError('Cannot open MTC OMX (tried openmatrix, larch.OMX)')
    _names = mtc.discover_skim_matrices(omx)
    print(f'MTC OMX matrices: {_names}', flush=True)
    _dist_exact = 'DIST' if 'DIST' in _names else None
    if _dist_exact is None:
        _cands = [_n for _n in _names
                  if 'DIST' in mtc._norm_name(_n)
                  and 'WALK' not in mtc._norm_name(_n)
                  and 'BIKE' not in mtc._norm_name(_n)]
        if not _cands:
            raise RuntimeError(f'MTC skims: no DIST slot (available={_names})')
        _dist_exact = _cands[0]
    picked = {
        'SOV_TIME': mtc.pick_matrix(_names, ['SOV', 'TIME'], ['DA', 'TIME'],
                                    ['AUTO', 'TIME'], ['SOVTOLL', 'TIME'],
                                    slot='SOV_TIME'),
        'PM_TIME': mtc.pick_matrix(_names, ['SOV', 'TIME', 'PM'],
                                   ['AUTO', 'TIME', 'PM'], ['TIME', 'PM'],
                                   slot='PM_TIME', required=False),
        'DIST': _dist_exact,
        'DISTWALK': mtc.pick_matrix(_names, ['WALK', 'DIST'],
                                    ['WALK', 'DISTANCE'], slot='DISTWALK',
                                    required=False),
        'DISTBIKE': mtc.pick_matrix(_names, ['BIKE', 'DIST'],
                                    ['BIKE', 'DISTANCE'], slot='DISTBIKE',
                                    required=False),
        'IVTT': mtc.pick_matrix(_names, ['TRANSIT', 'IVTT'], ['IVTT'],
                                ['TRANSIT', 'INVEHICLE'], slot='IVTT',
                                required=False),
        'FARE': mtc.pick_matrix(_names, ['FARE'], ['COST'], ['TOLL'],
                                slot='FARE', required=False),
    }
    print(f'MTC skim slot map: {picked}', flush=True)
    mats = mtc.read_omx_matrices(omx, [v for v in picked.values() if v])
    # shape check against the DIST matrix
    _d0 = next(iter(mats.values()))
    _n = _d0.shape[0]
    taz_ids = mtc.zone_ids_from_omx(omx, _n)
    skim_df = mtc.build_skim_df(mats, taz_ids, picked)

    tours = mtc.build_tours_from_regress(regress, pp)
    tours = mtc.add_tour_distances(tours, skim_df)
    pp = mtc.attach_tour_counts(pp, tours)
    s4 = mtc.map_synth_s4(tours)
    # persist mapped frames inside the root (rerunnable, inspectable)
    tours.to_csv(mtc.guard_path(root, 'mtc_tours.csv'), index=False)
    pp.to_csv(mtc.guard_path(root, 'mtc_persons.csv'), index=False)
    lu.to_csv(mtc.guard_path(root, 'mtc_landuse.csv'))
    autodist = {(int(o), int(d)): float(v)
                for (o, d), v in
                skim_df.set_index(['OTAZ', 'DTAZ'])['DIST'].items()
                if np.isfinite(v)}
    mtc.write_manifest(root, {
        'n_hh': int(len(hh)), 'n_persons': int(len(pp)),
        'n_tours': int(len(tours)), 'n_zones': int(len(taz_ids)),
        'skim_slots': dict(picked),
        'n_unmapped_modes': int(tours.attrs.get('n_unmapped_modes', 0)),
    })
    n_work = int((tours['main_purpose'] == 'work').sum())
    n_edu = int((tours['main_purpose'] == 'edu').sum())
    n_non = int((tours['main_purpose'] == 'nonmand').sum())
    print(f'mapped: {len(tours)} tours (work={n_work}, edu={n_edu}, nonmand={n_non}), '
          f'{len(pp)} persons, {len(taz_ids)} zones', flush=True)
    metrics.update({'n_tours': int(len(tours)), 'n_work': n_work,
                    'n_edu': n_edu, 'n_nonmand': n_non,
                    'n_persons': int(len(pp)), 'n_zones': int(len(taz_ids))})

    # ── M0: ActivitySim standard-model reference (no fitting) ──
    try:
        print('=== arm M0 (ActivitySim regress reference) ===', flush=True)
        _tm = tours['tour_mode']
        _pv = pd.Series(pd.to_numeric(_tm, errors='coerce').dropna()).value_counts(normalize=True)
        _homes = tours['OTAZ'].tolist()
        _mean_len, _n_len = _trip_length_stats(
            _homes, tours['DTAZ'].tolist(), autodist)
        m0 = {'arm': 'M0_regress_reference', 'ok': True,
              'n_tours': int(len(tours)),
              'obs_mode_shares': {str(int(k)): float(v) for k, v in _pv.items()},
              'obs_da_share': float(_pv.get(1, 0.0)),
              'mean_trip_len_km': _mean_len, 'n_len': _n_len,
              'dest_top10': [int(z) for z in
                             pd.Series(pd.to_numeric(tours['DTAZ'], errors='coerce')
                                       .dropna()).value_counts().nlargest(10).index]}
    except Exception as e:
        import traceback
        m0 = {'arm': 'M0_regress_reference', 'ok': False,
              'error': f'{type(e).__name__}: {e}',
              'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M0'] = m0
    print('M0: ' + str({k: m0.get(k) for k in
                        ('ok', 'n_tours', 'obs_da_share', 'mean_trip_len_km')}),
          flush=True)

    # ── Stage 0 CDAP: M1 defaults + M2 searched ──
    # Mirrors exampville_compare (same estimators, same guarded best-effort
    # pattern: failures record traces, never abort the comparison). MTC pp
    # already carries N_TOURS/N_WORK_TOURS/N_OTHER_TOURS/ANYSTOPS/MAINACT.
    # Kept OUT of _need below so a stage-0/1 failure cannot fail the job.
    def _pick_scalar(d, keys):
        try:
            for k in keys:
                v = (d or {}).get(k)
                try:
                    if v is not None and np.isfinite(float(v)):
                        return float(v)
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def _require_valid_search_fitness(ga, label):
        try:
            fitness = float(getattr(ga, 'best_fitness', float('nan')))
        except Exception:
            fitness = float('nan')
        if not np.isfinite(fitness) or fitness <= -1e8:
            raise RuntimeError(
                f'{label} failed: invalid best_fitness={fitness!r}')
        return fitness

    try:
        print('=== arm M1_s0 (CDAP defaults) ===', flush=True)
        from cdap import estimate_cdap
        _m1s0 = estimate_cdap(
            pp, save_path=mtc.guard_path(root, 'stage0_models_M1.pkl'))
        m1s0 = {'arm': 'M1_s0_default', 'ok': True,
                'll': _pick_scalar(_m1s0, ['ll', 'loglike']),
                'rho2': _pick_scalar(_m1s0, ['rho2', 'rho_sq']),
                'features': list((_m1s0 or {}).get('features', []))}
    except Exception as e:
        import traceback
        m1s0 = {'arm': 'M1_s0_default', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M1_s0'] = m1s0
    print('M1_s0 ok=' + str(m1s0.get('ok')), flush=True)
    try:
        print('=== arm M2_s0 (CDAP searched) ===', flush=True)
        from ga_hypothesis_testing import run_ga_stage
        _ga0 = run_ga_stage(stage=0, persons=pp, pop_size=_M2_POP,
                            n_generations=_M2_GENS, n_restarts=_M2_RESTARTS,
                            verbose=True)
        _fit0 = _require_valid_search_fitness(_ga0, 'M2_s0 search')
        _best0 = list(_ga0.best_features or [])
        _m2s0 = estimate_cdap(
            pp, save_path=mtc.guard_path(root, 'stage0_models_M2.pkl'),
            feature_override=_best0 or None)
        m2s0 = {'arm': 'M2_s0_searched', 'ok': True,
                'll': _pick_scalar(_m2s0, ['ll', 'loglike']),
                'rho2': _pick_scalar(_m2s0, ['rho2', 'rho_sq']),
                'features': list((_m2s0 or {}).get('features', [])),
                'n_features': len(_best0)}
        try:
            m2s0['best_fitness'] = _fit0
        except Exception:
            m2s0['best_fitness'] = None
    except Exception as e:
        import traceback
        m2s0 = {'arm': 'M2_s0_searched', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M2_s0'] = m2s0
    print('M2_s0 ok=' + str(m2s0.get('ok')), flush=True)

    # ── Stage 1 frequency: M1 defaults + M2 searched ──
    def _pred_means(synth_pred, tags):
        out = {}
        try:
            cols = list(synth_pred.columns)
            for tag, keys in tags.items():
                hit = [c for c in cols
                       if any(k in str(c).lower() for k in keys)]
                vals = []
                for c in hit:
                    try:
                        vals += pd.to_numeric(
                            synth_pred[c], errors='coerce').dropna().tolist()
                    except Exception:
                        pass
                out[tag] = float(np.mean(vals)) if vals else None
        except Exception:
            pass
        return out

    try:
        _nt = pd.to_numeric(pp['N_TOURS'], errors='coerce').fillna(0)
        _nw = pd.to_numeric(pp['N_WORK_TOURS'], errors='coerce').fillna(0)
        _obs_means = {'work': float(_nw.mean()),
                      'nonmand': float(pd.to_numeric(
                          pp['N_OTHER_TOURS'], errors='coerce').fillna(0).mean())}
    except Exception:
        _obs_means = {'work': None, 'nonmand': None}
    metrics['freq_observed_means'] = _obs_means
    try:
        print('=== arm M1_s1 (frequency defaults) ===', flush=True)
        from stage1_activity_frequency import run_stage1
        _p1, _m1 = run_stage1(
            tours, pp, synth=pp, sample=False, seed=42,
            save_path=mtc.guard_path(root, 'stage1_models_M1.pkl'),
            landuse=lu)
        _pm1 = _pred_means(_p1, {'work': ['work'], 'nonmand': ['nonmand']})
        m1s1 = {'arm': 'M1_s1_default', 'ok': True,
                'pred_means': _pm1,
                'work_mean_mae': (abs(_pm1.get('work', np.nan)
                                      - _obs_means['work'])
                                  if _pm1.get('work') is not None
                                  and _obs_means['work'] is not None else None),
                'nonmand_mean_mae': (abs(_pm1.get('nonmand', np.nan)
                                         - _obs_means['nonmand'])
                                     if _pm1.get('nonmand') is not None
                                     and _obs_means['nonmand'] is not None
                                     else None)}
        try:
            m1s1['cells'] = sorted(list((_m1 or {}).keys()))
        except Exception:
            m1s1['cells'] = []
    except Exception as e:
        import traceback
        m1s1 = {'arm': 'M1_s1_default', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M1_s1'] = m1s1
    print('M1_s1 ok=' + str(m1s1.get('ok')), flush=True)
    try:
        print('=== arm M2_s1 (frequency searched) ===', flush=True)
        from ga_hypothesis_testing import run_ga_stage as _rgs1
        from stage1_activity_frequency import run_stage1 as _rs1
        _ga1 = _rgs1(stage=1, tours=tours, persons=pp, landuse=lu,
                     pop_size=_M2_POP, n_generations=_M2_GENS,
                     n_restarts=_M2_RESTARTS, verbose=True)
        _fit1 = _require_valid_search_fitness(_ga1, 'M2_s1 search')
        _bf1 = _ga1.best_features
        _ov1 = None
        if isinstance(_bf1, dict):
            _u = []
            for _pv in _bf1.values():
                _u += list(_pv or [])
            _ov1 = sorted(set(_u)) or None
        elif _bf1:
            _ov1 = list(_bf1)
        _p2, _m2 = _rs1(
            tours, pp, synth=pp, sample=False, seed=42,
            save_path=mtc.guard_path(root, 'stage1_models_M2.pkl'),
            landuse=lu, feature_override=_ov1)
        _pm2 = _pred_means(_p2, {'work': ['work'], 'nonmand': ['nonmand']})
        m2s1 = {'arm': 'M2_s1_searched', 'ok': True,
                'pred_means': _pm2,
                'work_mean_mae': (abs(_pm2.get('work', np.nan)
                                      - _obs_means['work'])
                                  if _pm2.get('work') is not None
                                  and _obs_means['work'] is not None else None),
                'nonmand_mean_mae': (abs(_pm2.get('nonmand', np.nan)
                                         - _obs_means['nonmand'])
                                     if _pm2.get('nonmand') is not None
                                     and _obs_means['nonmand'] is not None
                                     else None)}
        try:
            m2s1['best_fitness'] = _fit1
        except Exception:
            m2s1['best_fitness'] = None
    except Exception as e:
        import traceback
        m2s1 = {'arm': 'M2_s1_searched', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M2_s1'] = m2s1
    print('M2_s1 ok=' + str(m2s1.get('ok')), flush=True)

    # ── estimation feasibility gate ─────────────────────────────────────
    # The regress reference sample is tiny (79 trips → 33 tours); per-segment
    # estimation needs STAGE5_MIN_TOURS (default 30). Below that, M1/M2 are
    # SKIPPED with a reason (not FAILED): fitting a 5-mode NL + destination
    # choice on ~10 tours would print junk LLs. M0 reference stats still stand.
    _min_tours = int(os.environ.get('STAGE5_MIN_TOURS', '30'))
    _seg_counts = tours['main_purpose'].value_counts().to_dict() if \
        'main_purpose' in tours.columns else {}
    _seg_small = {s: int(_seg_counts.get(s, 0)) for s in ('work', 'edu', 'nonmand')
                  if int(_seg_counts.get(s, 0)) < _min_tours}
    _skip_stage5 = bool(_seg_small)
    _skip_reason = (f"regress sample too small for estimation: segment counts "
                    f"{_seg_small} < STAGE5_MIN_TOURS={_min_tours}; M1/M2 need "
                    f"full MTC model outputs") if _skip_stage5 else ''
    if _skip_stage5:
        print(f'SKIP M1/M2 estimation: {_skip_reason}', flush=True)

    # ── pipeline arms need the stage-5 machinery ──
    try:
        from stage5_destination_choice import run_stage5
    except ImportError as _ie:
        # No larch/stage5 here (e.g. local laptop): same SKIP treatment —
        # M0 reference + table still get written.
        run_stage5 = None
        _skip_stage5 = True
        _skip_reason = (f'stage-5 machinery unavailable ({_ie}); ' +
                        (_skip_reason or 'estimation impossible in this env'))
        print(f'SKIP M1/M2 estimation: {_skip_reason}', flush=True)
    trips_legs = tours.copy()
    common = dict(trips=trips_legs, persons=pp, synth_s4=s4, skims=skim_path,
                  landuse=lu, tours=tours, seed=42, verbose=False,
                  hhts_persons=pp, segments=['work', 'edu', 'nonmand'])

    # ── M1: pipeline default spec, no search ──
    try:
        print('=== arm M1 (pipeline default, no search) ===', flush=True)
        if _skip_stage5:
            raise _SkipStage5(_skip_reason)
        plan1, models1 = run_stage5(
            save_path=mtc.guard_path(root, 'stage5_models_M1.pkl'), **common)
        e1 = {'arm': 'M1_pipeline_default', 'ok': True}
        for seg in ('work', 'edu', 'nonmand'):
            sres = (models1 or {}).get(seg, {}) or {}
            for mtype in ('mode_model', 'dest_model'):
                e1[f'{seg}_{mtype}'] = _larch_ll_stats(sres.get(mtype))
        e1.update(_plan_metrics(plan1, tours, autodist))
        try:
            _homes = tours['OTAZ'].tolist()
            _g = min(len(plan1), len(tours))
            e1['trip_len_mae'] = _trip_length_mae(
                plan1['DTAZ'].tolist()[:_g] if 'DTAZ' in plan1.columns else [],
                tours['DTAZ'].tolist()[:_g], _homes[:_g], autodist)
        except Exception:
            e1['trip_len_mae'] = None
    except _SkipStage5 as _s:
        e1 = {'arm': 'M1_pipeline_default', 'ok': False, 'skipped': True,
              'reason': str(_s)}
    except Exception as e:
        import traceback
        e1 = {'arm': 'M1_pipeline_default', 'ok': False,
              'error': f'{type(e).__name__}: {e}',
              'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M1'] = e1
    print('M1 ok=' + str(e1.get('ok')), flush=True)

    # ── M2: searched spec (small GA budget) + refit ──
    try:
        print('=== arm M2 (GA-searched spec) ===', flush=True)
        if _skip_stage5:
            raise _SkipStage5(_skip_reason)
        os.environ.setdefault('GA_ENGINE', 'agds')
        os.environ.setdefault('GA_MAX_WORKERS', '1')
        os.environ.setdefault('SAFETY_EXPERIMENT_MODE', 'nosafety')
        from ga_hypothesis_testing import run_ga_stage
        ga = run_ga_stage(
            stage=5, tours=tours, trips=trips_legs, persons=pp,
            skims=skim_path, landuse=lu, pop_size=_M2_POP,
            n_generations=_M2_S5_GENS,
            mutation_rate=0.10, n_restarts=_M2_RESTARTS, verbose=True,
            stage_overrides={5: {'pop_size': _M2_POP,
                                 'n_generations': _M2_S5_GENS,
                                 'n_restarts': _M2_RESTARTS}},
        )
        e2fit = _require_valid_search_fitness(ga, 'M2 stage-5 search')
        best_mode = (ga.best_features or {}).get('mode', None)
        best_dest = (ga.best_features or {}).get('dest', None)
        plan2, models2 = run_stage5(
            save_path=mtc.guard_path(root, 'stage5_models_M2.pkl'),
            mode_feature_override=best_mode,
            dest_feature_override=best_dest, **common)
        e2 = {'arm': 'M2_searched', 'ok': True, 'best_fitness': e2fit,
              'n_mode_features': len(best_mode) if best_mode else 0,
              'n_dest_features': len(best_dest) if best_dest else 0}
        for seg in ('work', 'edu', 'nonmand'):
            sres = (models2 or {}).get(seg, {}) or {}
            for mtype in ('mode_model', 'dest_model'):
                e2[f'{seg}_{mtype}'] = _larch_ll_stats(sres.get(mtype))
        e2.update(_plan_metrics(plan2, tours, autodist))
        try:
            _homes = tours['OTAZ'].tolist()
            _g = min(len(plan2), len(tours))
            e2['trip_len_mae'] = _trip_length_mae(
                plan2['DTAZ'].tolist()[:_g] if 'DTAZ' in plan2.columns else [],
                tours['DTAZ'].tolist()[:_g], _homes[:_g], autodist)
        except Exception:
            e2['trip_len_mae'] = None
    except _SkipStage5 as _s:
        e2 = {'arm': 'M2_searched', 'ok': False, 'skipped': True,
              'reason': str(_s)}
    except Exception as e:
        import traceback
        e2 = {'arm': 'M2_searched', 'ok': False,
              'error': f'{type(e).__name__}: {e}',
              'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['M2'] = e2
    print('M2 ok=' + str(e2.get('ok')), flush=True)

    metrics['walltime_min'] = round((time.time() - t_all) / 60.0, 1)
    with open(mtc.guard_path(root, 'mtc_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    write_metrics_tex(metrics, mtc.guard_path(root, 'mtc_metrics_table.tex'))
    print('=== comparison table ===')
    for arm in ('M0', 'M1', 'M2'):
        a = metrics['arms'].get(arm, {})
        print(f"[{arm}] ok={a.get('ok')} mode_mae={a.get('mode_mae')} "
              f"dest_mae={a.get('dest_mae')} top10={a.get('dest_top10')} "
              f"trip_len_mae={a.get('trip_len_mae')}")
    print(f'wrote {root}/mtc_metrics.json')
    _need = ['M0', 'M1', 'M2']
    _skipped = [k for k in _need if metrics['arms'].get(k, {}).get('skipped')]
    _bad = [k for k in _need
            if not metrics['arms'].get(k, {}).get('ok')
            and not metrics['arms'].get(k, {}).get('skipped')]
    if _skipped:
        print('ARMS SKIPPED (sample too small for estimation): ' + str(_skipped))
    if _bad:
        print('ARMS FAILED: ' + str(_bad))
        sys.exit(1)


if __name__ == '__main__':
    main()
