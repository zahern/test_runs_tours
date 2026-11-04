#!/usr/bin/env python
"""Exampville three-arm comparison (Stage 5, work + nonmand tours, 40 zones).

Arms (all on identical Exampville data/universe):
  E0  Larch-docs example replication: example-201 3-level NL work mode choice
      re-estimated here verbatim (utilities, nests, availability, set_cap(20));
      destination compared against published example-203 statistics
      (LL -25157.72, rho2 0.109, n=7564) as the citable benchmark — 203 needs
      the 202 logsum artifact chain, so re-estimation is not attempted.
  E1  P0 pipeline default (ActivitySim-style) spec, no search.
  E2  P2 searched spec (small GA budget) refit + applied.

Needs larch (cluster zigenv). Artifacts: everything under OUTROOT
(default runs/exampville_ab, enforced to contain 'exampville'); NOTHING is
written to saved_data/ or any QLD run tree. Full synthetic population is
used (all 12,349 persons / 20,739 tours; prediction over all tours).

Usage:  python -u exampville_compare.py [--outdir runs/exampville_ab]
Writes: exampville_metrics.json, stage5 models, SOURCE.json, mapped CSVs.
"""

# NumPy 2.0 compatibility shim - MUST BE FIRST IMPORT
import compat.numpy_shim

import json
import logging
import os
import sys

# 40-zone transfer system: relax the 371-zone skim guard BEFORE any local skim
# import (read_skims reads ZONE_SYSTEM_N_ZONES at import time). Also exported
# by job_exampville_ab.pbs; setdefault keeps direct `python` launches working.
os.environ.setdefault('ZONE_SYSTEM_N_ZONES', '40')

import numpy as np
import pandas as pd

import exampville_connect as evc

logger = logging.getLogger('exampville_compare')

WORK = 'work'


# ── Extensive-search budgets ─────────────────────────────────────────────
# E2 arms default to the small in-file budgets (fast CI-style runs). Override
# via env for benchmark-grade extensive searches without code edits, e.g.
#   EXV_E2_GENS=150 EXV_E2_S5_GENS=120 python -u exampville_compare.py ...
# Exampville fits are seconds each (40 zones), so thorough budgets stay cheap.
def _e2_int(name, default):
    try:
        v = int(os.environ.get(name, '').strip() or 0) or 0
        return v if v > 0 else default
    except (ValueError, TypeError, AttributeError):
        return default


_E2_POP = _e2_int('EXV_E2_POP', 12)
_E2_GENS = _e2_int('EXV_E2_GENS', 8)
_E2_RESTARTS = _e2_int('EXV_E2_RESTARTS', 1)
_E2_S5_GENS = _e2_int('EXV_E2_S5_GENS', 10)


# ── metrics helpers (larch-free) ──────────────────────────────────────────
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
            return float('nan'), float('nan'), float('nan')
        errs = np.asarray(errs)
        return float(errs.mean()), float(np.median(errs)),
    except Exception:
        return float('nan'), float('nan')
    return float('nan'), float('nan')


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
    try:
        out['n'] = int(mdl.datatree.n_cases)
    except Exception:
        pass
    if out['n'] is None:
        for _attr in ('n_obs_', 'n_', 'nb_samples'):
            try:
                out['n'] = int(getattr(mdl, _attr))
                break
            except Exception:
                continue
    return out


# ── E0: docs example-201 replication ──────────────────────────────────────
def run_e0(tour_raw, hh_raw, pp_raw, skims_omx, outroot):
    """Re-estimate Larch-docs 201 work mode NL verbatim. Returns metrics dict."""
    import larch as lx
    from larch import P, X
    res = {'arm': 'E0_docs_201', 'ok': False}
    work = tour_raw[pd.to_numeric(tour_raw['TOURPURP'],
                                  errors='coerce') == 1].copy()
    tour_ds = lx.Dataset.construct.from_idco(
        work.set_index('TOURID'), alts=[1, 2, 3, 4, 5])
    od = lx.Dataset.construct.from_omx(skims_omx)
    dt = lx.DataTree(
        tour=tour_ds,
        hh=hh_raw.set_index('HHID'),
        person=pp_raw.set_index('PERSONID'),
        od=od, do=od,
        relationships=(
            "tours.HHID @ hh.HHID",
            "tours.PERSONID @ person.PERSONID",
            "hh.HOMETAZ @ od.otaz",
            "tours.DTAZ @ od.dtaz",
            "hh.HOMETAZ @ do.dtaz",
            "tours.DTAZ @ do.otaz",
        ),
    )
    dt_work = dt.query_cases("TOURPURP == 1")
    m = lx.Model(datatree=dt_work)
    m.title = "Exampville Work Tour Mode Choice v1 (replication)"
    m.utility_co[1] = (+ P.InVehTime * X.AUTO_TIME
                       + P.Cost * X.AUTO_COST)
    m.utility_co[2] = (+ P.ASC_SR
                       + P.InVehTime * X.AUTO_TIME
                       + P.Cost * (X.AUTO_COST * 0.5)
                       + P("LogIncome:SR") * X("log(INCOME)"))
    m.utility_co[3] = (+ P.ASC_Walk
                       + P.NonMotorTime * X.WALK_TIME
                       + P("LogIncome:Walk") * X("log(INCOME)"))
    m.utility_co[4] = (+ P.ASC_Bike
                       + P.NonMotorTime * X.BIKE_TIME
                       + P("LogIncome:Bike") * X("log(INCOME)"))
    m.utility_co[5] = (+ P.ASC_Transit
                       + P.InVehTime * X.TRANSIT_IVTT
                       + P.OutVehTime * X.TRANSIT_OVTT
                       + P.Cost * X.TRANSIT_FARE
                       + P("LogIncome:Transit") * X('log(INCOME)'))
    _car = m.graph.new_node(parameter='Mu:Car', children=[1, 2], name='Car')
    _nm = m.graph.new_node(parameter='Mu:NonMotor', children=[3, 4],
                           name='NonMotor')
    m.graph.new_node(parameter='Mu:Motor', children=[_car, 5], name='Motor')
    m.choice_co_code = 'TOURMODE'
    m.availability_co_vars = {
        1: 'AGE >= 16', 2: 1, 3: 'WALK_TIME < 60',
        4: 'BIKE_TIME < 60', 5: 'TRANSIT_FARE>0',
    }
    m.set_cap(20)
    r = m.maximize_loglike()
    try:
        m.calculate_parameter_covariance()
    except Exception:
        pass
    stats = _larch_ll_stats(m)
    res.update(stats)
    try:
        res['n'] = int(len(work))
    except Exception:
        pass
    try:
        _p = np.asarray(m.probability(), dtype=float)
        _pred = (_p.argmax(axis=1) + 1).tolist()
        res['mode_mae'] = _mode_share_mae(_pred, work['TOURMODE'].tolist())
    except Exception as e:
        res['mode_mae'] = None
        res['shares_note'] = f'probability() unavailable ({e})'
    try:
        res['coef'] = {str(k): float(v) for k, v in
                       dict(m.pf['value']).items()}
    except Exception:
        res['coef'] = {}
    res['ok'] = True
    res['docs_reference'] = {'ll': -3493.04, 'rho2': 0.672, 'n': 7564,
                             'source': 'Larch v5.7.0 example 201'}
    return res


# ── pipeline arms ─────────────────────────────────────────────────────────
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
        homes = tours_mapped['OTAZ'].tolist()
        g = min(len(plan), len(tours_mapped))
        out['trip_len_mae'], out['trip_len_med'] = _trip_length_mae(
            plan['DTAZ'].tolist()[:g] if 'DTAZ' in plan.columns else [],
            tours_mapped['DTAZ'].tolist()[:g], homes[:g], autodist)
    except Exception:
        out['trip_len_mae'], out['trip_len_med'] = None, None
    try:
        out['n_pred'] = int(len(plan)) if plan is not None else 0
    except Exception:
        out['n_pred'] = 0
    return out


def main():
    outdir = 'runs/exampville_ab'
    for i, a in enumerate(sys.argv[1:]):
        if a == '--outdir' and i + 1 < len(sys.argv[1:]):
            outdir = sys.argv[1:][i + 1]
    root = evc.exampville_root(outdir)
    logging.basicConfig(level=logging.INFO)
    # ── Artifact isolation (never touch QLD / MTC outputs) ─────────────────
    # GA checkpoints + best_*.npz default to GA_CHECKPOINT_DIR or CWD (the code
    # dir!); batch checkpoints/plots default to PIPELINE_OUT_DIR or the model
    # save dir; _run_segment appends ACDC rows to saved_data/acdc_terms.csv.
    # Pin all three inside this run's root so reruns resume from local state
    # only and foreign run trees are never written.
    os.environ.setdefault('GA_CHECKPOINT_DIR', os.path.join(root, 'checkpoints'))
    os.environ.setdefault('PIPELINE_OUT_DIR', root)
    os.environ.setdefault('STAGE5_SAVE_ACDC', '0')
    # No safety skims on transfer fixtures: run every arm crash-free
    # (drops crash params from UECs AND disables the worker CRASH fix).
    os.environ.setdefault('SAFETY_EXPERIMENT_MODE', 'nosafety')
    os.makedirs(os.environ['GA_CHECKPOINT_DIR'], exist_ok=True)
    print(f'=== Exampville 3-arm comparison -> {root} ===', flush=True)
    open(evc.guard_path(root, 'SOURCE.tag'), 'w').write('exampville\n')

    import time
    t_all = time.time()
    metrics = {'source': 'exampville', 'root': root, 'arms': {}}

    # ── load + map (larch needed from here) ──
    raw = evc.load_exampville_raw()
    skim_path = evc.copy_package_skims(raw.get('skim_src'), root)
    hh = evc.map_households(raw['hh'])
    pp = evc.map_persons(raw['pp'], hh)
    lu = evc.map_landuse(raw['emp'])
    # OMX matrices (openmatrix preferred, larch.OMX fallback)
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
        raise RuntimeError('Cannot open Exampville OMX (tried openmatrix, larch.OMX)')
    mats, taz_ids = evc.read_omx_matrices(
        omx, ['AUTO_TIME', 'AUTO_DIST', 'WALK_DIST', 'BIKE_TIME',
              'TRANSIT_IVTT', 'TRANSIT_FARE', 'AUTO_COST'])
    skim_df = evc.build_skim_df(mats, taz_ids)
    tours = evc.map_tours(raw['tour'], pp, mats, taz_ids)
    s4 = evc.map_synth_s4(tours)
    # persist mapped frames inside the root (rerunnable, inspectable)
    tours.to_csv(evc.guard_path(root, 'exampville_tours.csv'), index=False)
    pp.to_csv(evc.guard_path(root, 'exampville_persons.csv'), index=False)
    lu.to_csv(evc.guard_path(root, 'exampville_landuse.csv'))
    _dist_col = next(
        (c for c in ('DIST', 'AUTO_DIST') if c in skim_df.columns), None)
    if _dist_col is None:
        raise RuntimeError(
            "Exampville skim_df has neither DIST nor AUTO_DIST "
            f"(columns={list(skim_df.columns)}) — check OMX matrices")
    autodist = {(int(o), int(d)): float(v)
                for (o, d), v in
                skim_df.set_index(['OTAZ', 'DTAZ'])[_dist_col].items()
                if np.isfinite(v)}
    evc.write_manifest(root, {
        'n_hh': int(len(hh)), 'n_persons': int(len(pp)),
        'n_tours': int(len(tours)), 'n_zones': int(len(taz_ids)),
        'tourpurp_counts': {str(k): int(v) for k, v in
                            pd.to_numeric(raw['tour']['TOURPURP'],
                                          errors='coerce').value_counts().items()},
    })
    n_work = int((tours['main_purpose'] == 'work').sum())
    n_non = int((tours['main_purpose'] == 'othdiscr').sum())
    print(f'mapped: {len(tours)} tours (work={n_work}, nonmand={n_non}), '
          f'{len(pp)} persons, {len(taz_ids)} zones', flush=True)
    metrics.update({'n_tours': int(len(tours)), 'n_work': n_work,
                    'n_nonmand': n_non, 'n_persons': int(len(pp)),
                    'n_zones': int(len(taz_ids))})

    # ── E0: docs replication ──
    try:
        print('=== arm E0 (docs 201 replication) ===', flush=True)
        e0 = run_e0(raw['tour'], raw['hh'], raw['pp'], omx, root)
    except Exception as e:
        e0 = {'arm': 'E0_docs_201', 'ok': False, 'error': f'{type(e).__name__}: {e}'}
    metrics['arms']['E0'] = e0
    print('E0: ' + str({k: e0.get(k) for k in
                        ('ok', 'll', 'rho2', 'n', 'mode_mae')}), flush=True)

    # ── Stage 0 CDAP: E1 defaults + E2 searched ──
    # Labels derive in-pipeline from adapter ANYSTOPS/MAINACT; observed H/N/M
    # distribution is deterministic from tour counts (computed here for the
    # record). LL/rho2 come from estimate_cdap's return dict (best-effort).
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

    _nt = pd.to_numeric(pp['N_TOURS'], errors='coerce').fillna(0)
    _nw = pd.to_numeric(pp['N_WORK_TOURS'], errors='coerce').fillna(0)
    _hnm_obs = {'H': int(((_nt == 0)).sum()),
                'N': int(((_nt > 0) & (_nw == 0)).sum()),
                'M': int((_nw > 0).sum())}
    metrics['cdap_observed'] = _hnm_obs
    try:
        print('=== arm E1_s0 (CDAP defaults) ===', flush=True)
        from cdap import estimate_cdap
        _e1s0 = estimate_cdap(
            pp, save_path=evc.guard_path(root, 'stage0_models_E1.pkl'))
        m0e1 = {'arm': 'E1_s0_default', 'ok': True,
                'll': _pick_scalar(_e1s0, ['ll', 'loglike']),
                'rho2': _pick_scalar(_e1s0, ['rho2', 'rho_sq']),
                'features': list((_e1s0 or {}).get('features', []))}
    except Exception as e:
        import traceback
        m0e1 = {'arm': 'E1_s0_default', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['E1_s0'] = m0e1
    print('E1_s0 ok=' + str(m0e1.get('ok')), flush=True)
    try:
        print('=== arm E2_s0 (CDAP searched) ===', flush=True)
        from ga_hypothesis_testing import run_ga_stage
        _ga0 = run_ga_stage(stage=0, persons=pp, pop_size=_E2_POP,
                            n_generations=_E2_GENS, n_restarts=_E2_RESTARTS,
                            verbose=True)
        _best0 = list(_ga0.best_features or [])
        _e2s0 = estimate_cdap(
            pp, save_path=evc.guard_path(root, 'stage0_models_E2.pkl'),
            feature_override=_best0 or None)
        m0e2 = {'arm': 'E2_s0_searched', 'ok': True,
                'll': _pick_scalar(_e2s0, ['ll', 'loglike']),
                'rho2': _pick_scalar(_e2s0, ['rho2', 'rho_sq']),
                'features': list((_e2s0 or {}).get('features', [])),
                'n_features': len(_best0)}
        try:
            m0e2['best_fitness'] = float(getattr(_ga0, 'best_fitness',
                                                 float('nan')))
        except Exception:
            m0e2['best_fitness'] = None
    except Exception as e:
        import traceback
        m0e2 = {'arm': 'E2_s0_searched', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['E2_s0'] = m0e2
    print('E2_s0 ok=' + str(m0e2.get('ok')), flush=True)

    # ── Stage 1 frequency: E1 defaults + E2 searched ──
    # Counts derive in-pipeline from mapped tours (n_work_out, n_othdiscr_out,
    # consolidated n_nonmand_out). Observed means from person tour counts;
    # predicted means best-effort from run_stage1's synth predictions.
    def _pred_means(synth_pred, tags):
        # run_stage1 returns per-person predicted counts as n_<purpose>_out.
        # Take that column EXACTLY: a substring match on 'work' also picks up
        # employment totals and household counts, which are not per-person
        # tour counts and inflated the mean by ~3 orders of magnitude (the
        # 2852.94 'mean MAE' in jobs 27074291 / 20260924).
        out = {}
        used = {}
        try:
            cols = list(synth_pred.columns)
            for tag in tags:
                cand = [f'n_{tag}_out', f'n_{tag}']
                col = next((c for c in cand if c in cols), None)
                if col is None:
                    used[tag] = None
                    out[tag] = None
                    continue
                vals = pd.to_numeric(
                    synth_pred[col], errors='coerce').dropna()
                out[tag] = float(vals.mean()) if len(vals) else None
                used[tag] = col
        except Exception:
            pass
        out['_columns'] = used
        return out

    _obs_means = {'work': float(_nw.mean()),
                  'nonmand': float(pd.to_numeric(
                      pp['N_OTHER_TOURS'], errors='coerce').fillna(0).mean())}
    metrics['freq_observed_means'] = _obs_means
    try:
        print('=== arm E1_s1 (frequency defaults) ===', flush=True)
        from stage1_activity_frequency import run_stage1
        _p1, _m1 = run_stage1(
            tours, pp, synth=pp, sample=False, seed=42,
            save_path=evc.guard_path(root, 'stage1_models_E1.pkl'),
            landuse=lu)
        _pm1 = _pred_means(_p1, {'work': ['work'], 'nonmand': ['nonmand']})
        m1e1 = {'arm': 'E1_s1_default', 'ok': True,
                'pred_means': _pm1,
                'work_mean_mae': (abs(_pm1.get('work', np.nan)
                                      - _obs_means['work'])
                                  if _pm1.get('work') is not None else None),
                'nonmand_mean_mae': (abs(_pm1.get('nonmand', np.nan)
                                         - _obs_means['nonmand'])
                                     if _pm1.get('nonmand') is not None
                                     else None)}
        try:
            m1e1['cells'] = sorted(list((_m1 or {}).keys()))
        except Exception:
            m1e1['cells'] = []
    except Exception as e:
        import traceback
        m1e1 = {'arm': 'E1_s1_default', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['E1_s1'] = m1e1
    print('E1_s1 ok=' + str(m1e1.get('ok')), flush=True)
    try:
        print('=== arm E2_s1 (frequency searched) ===', flush=True)
        from ga_hypothesis_testing import run_ga_stage as _rgs1
        from stage1_activity_frequency import run_stage1 as _rs1
        _ga1 = _rgs1(stage=1, tours=tours, persons=pp, landuse=lu,
                     pop_size=_E2_POP, n_generations=_E2_GENS,
                     n_restarts=_E2_RESTARTS, verbose=True)
        _bf1 = _ga1.best_features
        # run_stage1 takes a single override list: union work+nonmand feats
        # (documented; JSON records both the per-purpose feats and the union).
        _ov1 = None
        if isinstance(_bf1, dict):
            _u = []
            for _pv in _bf1.values():
                _u += list(_pv or [])
            _ov1 = sorted(set(_u)) or None
        elif isinstance(_bf1, list):
            _ov1 = _bf1 or None
        _p2, _m2 = _rs1(
            tours, pp, synth=pp, sample=False, seed=42,
            save_path=evc.guard_path(root, 'stage1_models_E2.pkl'),
            feature_override=_ov1, landuse=lu)
        _pm2 = _pred_means(_p2, {'work': ['work'], 'nonmand': ['nonmand']})
        m1e2 = {'arm': 'E2_s1_searched', 'ok': True,
                'pred_means': _pm2,
                'work_mean_mae': (abs(_pm2.get('work', np.nan)
                                      - _obs_means['work'])
                                  if _pm2.get('work') is not None else None),
                'nonmand_mean_mae': (abs(_pm2.get('nonmand', np.nan)
                                         - _obs_means['nonmand'])
                                     if _pm2.get('nonmand') is not None
                                     else None),
                'override_union': _ov1}
        try:
            m1e2['best_fitness'] = float(getattr(_ga1, 'best_fitness',
                                                 float('nan')))
        except Exception:
            m1e2['best_fitness'] = None
        try:
            m1e2['cells'] = sorted(list((_m2 or {}).keys()))
        except Exception:
            m1e2['cells'] = []
    except Exception as e:
        import traceback
        m1e2 = {'arm': 'E2_s1_searched', 'ok': False,
                'error': f'{type(e).__name__}: {e}',
                'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['E2_s1'] = m1e2
    print('E2_s1 ok=' + str(m1e2.get('ok')), flush=True)

    # ── pipeline arms need the stage-5 machinery ──
    from stage5_destination_choice import run_stage5
    trips_legs = tours.copy()  # single-leg equivalent (no leg detail in Exampville)
    common = dict(trips=trips_legs, persons=pp, synth_s4=s4, skims=skim_path,
                  landuse=lu, tours=tours, seed=42, verbose=False,
                  hhts_persons=pp, segments=['work', 'nonmand'])

    # ── E1: pipeline default spec, no search ──
    try:
        print('=== arm E1 (pipeline default, no search) ===', flush=True)
        plan1, models1 = run_stage5(
            save_path=evc.guard_path(root, 'stage5_models_E1.pkl'), **common)
        e1 = {'arm': 'E1_pipeline_default', 'ok': True}
        for seg in ('work', 'nonmand'):
            sres = (models1 or {}).get(seg, {}) or {}
            for mtype in ('mode_model', 'dest_model'):
                e1[f'{seg}_{mtype}'] = _larch_ll_stats(sres.get(mtype))
        e1.update(_plan_metrics(plan1, tours, autodist))
    except Exception as e:
        import traceback
        e1 = {'arm': 'E1_pipeline_default', 'ok': False,
              'error': f'{type(e).__name__}: {e}',
              'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['E1'] = e1
    print('E1 ok=' + str(e1.get('ok')), flush=True)

    # ── E2: searched spec (small GA budget) + refit ──
    try:
        print('=== arm E2 (GA-searched spec) ===', flush=True)
        os.environ.setdefault('GA_ENGINE', 'agds')
        os.environ.setdefault('GA_MAX_WORKERS', '1')
        os.environ.setdefault('SAFETY_EXPERIMENT_MODE', 'nosafety')
        from ga_hypothesis_testing import run_ga_stage
        ga = run_ga_stage(
            stage=5, tours=tours, trips=trips_legs, persons=pp,
            skims=skim_path, landuse=lu, pop_size=_E2_POP,
            n_generations=_E2_S5_GENS,
            mutation_rate=0.10, n_restarts=_E2_RESTARTS, verbose=True,
            stage_overrides={5: {'pop_size': _E2_POP,
                                 'n_generations': _E2_S5_GENS,
                                 'n_restarts': _E2_RESTARTS}},
        )
        best_mode = (ga.best_features or {}).get('mode', None)
        best_dest = (ga.best_features or {}).get('dest', None)
        try:
            e2fit = float(getattr(ga, 'best_fitness', float('nan')))
        except Exception:
            e2fit = None
        plan2, models2 = run_stage5(
            save_path=evc.guard_path(root, 'stage5_models_E2.pkl'),
            mode_feature_override=best_mode,
            dest_feature_override=best_dest, **common)
        e2 = {'arm': 'E2_searched', 'ok': True, 'best_fitness': e2fit,
              'n_mode_features': len(best_mode) if best_mode else 0,
              'n_dest_features': len(best_dest) if best_dest else 0}
        for seg in ('work', 'nonmand'):
            sres = (models2 or {}).get(seg, {}) or {}
            for mtype in ('mode_model', 'dest_model'):
                e2[f'{seg}_{mtype}'] = _larch_ll_stats(sres.get(mtype))
        e2.update(_plan_metrics(plan2, tours, autodist))
    except Exception as e:
        import traceback
        e2 = {'arm': 'E2_searched', 'ok': False,
              'error': f'{type(e).__name__}: {e}',
              'trace': traceback.format_exc()[-2000:]}
    metrics['arms']['E2'] = e2
    print('E2 ok=' + str(e2.get('ok')), flush=True)

    metrics['walltime_min'] = round((time.time() - t_all) / 60.0, 1)
    with open(evc.guard_path(root, 'exampville_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print('=== comparison table ===')
    for arm in ('E0', 'E1', 'E2'):
        a = metrics['arms'].get(arm, {})
        print(f"[{arm}] ok={a.get('ok')} ll_mode(work)="
              f"{(a.get('work_mode_model') or {}).get('ll')} "
              f"rho2={((a.get('work_mode_model') or {}).get('rho2'))} "
              f"mode_mae={a.get('mode_mae')} dest_mae={a.get('dest_mae')} "
              f"trip_len_mae={a.get('trip_len_mae')}")
    for arm in ('E1_s0', 'E2_s0'):
        a = metrics['arms'].get(arm, {})
        print(f"[{arm}] ok={a.get('ok')} ll={a.get('ll')} "
              f"rho2={a.get('rho2')} features={a.get('features')}")
    print(f"CDAP observed H/N/M: {metrics.get('cdap_observed')}")
    for arm in ('E1_s1', 'E2_s1'):
        a = metrics['arms'].get(arm, {})
        print(f"[{arm}] ok={a.get('ok')} work_mean_mae={a.get('work_mean_mae')} "
              f"nonmand_mean_mae={a.get('nonmand_mean_mae')} "
              f"cells={a.get('cells')}")
    print(f'wrote {root}/exampville_metrics.json')
    _need = ['E0', 'E1', 'E2', 'E1_s0', 'E2_s0', 'E1_s1', 'E2_s1']
    if not all(metrics['arms'].get(k, {}).get('ok') for k in _need):
        print('ARMS FAILED: '
              + str([k for k in _need
                     if not metrics['arms'].get(k, {}).get('ok')]))
        sys.exit(1)


if __name__ == '__main__':
    main()
