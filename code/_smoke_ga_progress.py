#!/usr/bin/env python
"""Quick GA progress smoke test.

Checks whether mini GA runs are actually exploring search space (not flat):
- max diversity > 0
- best-so-far changes at least once
- crash terms retained where required

Runs a tiny budget for speed.
"""
from __future__ import annotations

import os
import traceback
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from data_connect import Synth_Connect
from ga_hypothesis_testing import run_ga_all_stages


def _sample_inputs(
    tours: pd.DataFrame,
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    n_persons: int = 2200,
    n_tours: int = 12000,
    n_trips: int = 25000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    p = persons.copy()
    if len(p) > n_persons and 'person_id' in p.columns:
        keep = p.sample(n=n_persons, random_state=seed)['person_id']
        keep_set = set(keep.tolist())
        p = p[p['person_id'].isin(keep_set)].copy()
        t = tours[tours['person_id'].isin(keep_set)].copy() if 'person_id' in tours.columns else tours.copy()
        r = trips[trips['person_id'].isin(keep_set)].copy() if 'person_id' in trips.columns else trips.copy()
    else:
        t = tours.copy()
        r = trips.copy()

    if len(t) > n_tours:
        t = t.sample(n=n_tours, random_state=seed).copy()
    if len(r) > n_trips:
        r = r.sample(n=n_trips, random_state=seed).copy()

    return t, r, p


def _history_progress_metrics(hist_df: pd.DataFrame) -> Dict[str, Any]:
    if hist_df is None or hist_df.empty:
        return {
            'n_generations': 0,
            'max_diversity': np.nan,
            'best_changed': False,
            'delta_best_so_far': np.nan,
        }

    best = pd.to_numeric(hist_df.get('best_so_far', hist_df.get('best_fitness')), errors='coerce')
    div = pd.to_numeric(hist_df.get('diversity', np.nan), errors='coerce')
    uniq_best = pd.Series(best).dropna().round(8).nunique()
    delta = float(best.iloc[-1] - best.iloc[0]) if len(best) >= 2 else 0.0

    return {
        'n_generations': int(len(hist_df)),
        'max_diversity': float(np.nanmax(div.values)) if np.isfinite(np.nanmax(div.values)) else np.nan,
        'best_changed': bool(uniq_best > 1),
        'delta_best_so_far': delta,
    }


def _check_stage_result(stage: int, res) -> Dict[str, Any]:
    out: Dict[str, Any] = {'stage': stage, 'ok': True, 'messages': []}

    seg_results = res.metadata.get('segment_results', {}) if isinstance(getattr(res, 'metadata', {}), dict) else {}
    if seg_results:
        bad = []
        for seg_name, seg_res in seg_results.items():
            h = seg_res.history_summary()
            m = _history_progress_metrics(h)
            seg_ok = bool((m['n_generations'] > 0) and (m['best_changed'] or (m['max_diversity'] > 0.0)))
            out['messages'].append(
                f"  {seg_name}: gens={m['n_generations']} max_div={m['max_diversity']:.4f} "
                f"best_changed={m['best_changed']} delta={m['delta_best_so_far']:.6f}"
            )
            if not seg_ok:
                bad.append(seg_name)
        if bad:
            out['ok'] = False
            out['messages'].append(f"  FAIL segments flat/no-progress: {bad}")
    else:
        h = res.history_summary()
        m = _history_progress_metrics(h)
        ok = bool((m['n_generations'] > 0) and (m['best_changed'] or (m['max_diversity'] > 0.0)))
        out['messages'].append(
            f"  stage={stage}: gens={m['n_generations']} max_div={m['max_diversity']:.4f} "
            f"best_changed={m['best_changed']} delta={m['delta_best_so_far']:.6f}"
        )
        out['ok'] = ok

    return out


def _check_crash_presence(results, lines: list[str]) -> bool:
    ok = True

    s5 = results.results.get(5)
    if s5 is not None and isinstance(s5.best_features, dict):
        mode_feats = set(s5.best_features.get('mode', []))
        dest_feats = set(s5.best_features.get('dest', []))
        has_mode = 'CRASH' in mode_feats
        has_dest = 'c_crash' in dest_feats
        lines.append(f"Stage5 crash terms: mode_CRASH={has_mode} dest_c_crash={has_dest}")
        ok = ok and has_mode and has_dest

    s51 = results.results.get(51)
    if s51 is not None and isinstance(s51.best_features, dict):
        mode_feats = set(s51.best_features.get('mode', []))
        dest_feats = set(s51.best_features.get('dest', []))
        has_mode = 'CRASH' in mode_feats
        has_dest = 'c_crash' in dest_feats
        lines.append(f"Stage51 crash terms: mode_CRASH={has_mode} dest_c_crash={has_dest}")
        ok = ok and has_mode and has_dest

    s6 = results.results.get(6)
    if s6 is not None and not isinstance(s6.best_features, dict):
        has_crash = 'crash' in set(s6.best_features)
        lines.append(f"Stage6 crash term: crash={has_crash}")
        ok = ok and has_crash

    return ok


def run_smoke() -> int:
    os.makedirs('runs', exist_ok=True)
    lines: list[str] = []
    overall_ok = True

    # Keep runtime short for smoke
    ga_small = {
        'pop_size': 12,
        'n_generations': 8,
        'mutation_rate': 0.12,
        'n_restarts': 0,
        'seed': 42,
    }

    try:
        sc = Synth_Connect()
        out = sc.process_survey(save=False)
        land = sc.get_land()

        tours_s, trips_s, persons_s = _sample_inputs(
            out['tours'], out['trips'], out['persons'],
            n_persons=2200, n_tours=12000, n_trips=25000, seed=42,
        )

        lines.append(
            f"INPUT sample sizes: tours={len(tours_s)} trips={len(trips_s)} persons={len(persons_s)}"
        )

        # A) default estimator: stage 5/51/6 (the most relevant to recent fixes)
        os.environ['GA_ESTIMATOR'] = 'default'
        lines.append("\n=== RUN A: GA_ESTIMATOR=default | stages=[5,51,6] ===")
        res_a = run_ga_all_stages(
            tours=tours_s,
            trips=trips_s,
            persons=persons_s,
            skims='downcasted_matrices_base.omx',
            landuse=land,
            stages=[5, 51, 6],
            save_path='runs/ga_smoke_default.pkl',
            **ga_small,
        )
        for stg in [5, 51, 6]:
            chk = _check_stage_result(stg, res_a.results[stg]) if stg in res_a.results else {'ok': False, 'messages': [f"  Stage {stg} missing"]}
            lines.extend(chk['messages'])
            overall_ok = overall_ok and bool(chk['ok'])
        overall_ok = overall_ok and _check_crash_presence(res_a, lines)

        # B) metacount estimator: stage 1 quick check (external estimator path)
        os.environ['GA_ESTIMATOR'] = 'metacount'
        lines.append("\n=== RUN B: GA_ESTIMATOR=metacount | stages=[1] ===")
        res_b = run_ga_all_stages(
            tours=tours_s,
            trips=trips_s,
            persons=persons_s,
            skims=None,
            landuse=None,
            stages=[1],
            save_path='runs/ga_smoke_metacount.pkl',
            **ga_small,
        )
        if 1 in res_b.results:
            chk = _check_stage_result(1, res_b.results[1])
            lines.extend(chk['messages'])
            overall_ok = overall_ok and bool(chk['ok'])
        else:
            lines.append("  Stage 1 missing")
            overall_ok = False

        # C) searchlibrium estimator: stage 4 quick check (external estimator path)
        os.environ['GA_ESTIMATOR'] = 'searchlibrium'
        lines.append("\n=== RUN C: GA_ESTIMATOR=searchlibrium | stages=[4] ===")
        res_c = run_ga_all_stages(
            tours=tours_s,
            trips=trips_s,
            persons=persons_s,
            skims=None,
            landuse=None,
            stages=[4],
            save_path='runs/ga_smoke_searchlibrium.pkl',
            **ga_small,
        )
        if 4 in res_c.results:
            chk = _check_stage_result(4, res_c.results[4])
            lines.extend(chk['messages'])
            overall_ok = overall_ok and bool(chk['ok'])
        else:
            lines.append("  Stage 4 missing")
            overall_ok = False

    except Exception as exc:
        overall_ok = False
        lines.append("\nSMOKE_EXCEPTION")
        lines.append(repr(exc))
        lines.append(traceback.format_exc())

    lines.append("\nRESULT=" + ("PASS" if overall_ok else "FAIL"))

    out_path = 'runs/ga_progress_smoke_report.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"WROTE {out_path}")
    print("PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == '__main__':
    raise SystemExit(run_smoke())
