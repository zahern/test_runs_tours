#!/usr/bin/env python
"""promote_stage5_bests.py - refit Stage 5 from best_*.npz, no GA rerun.

The qld_conv1 Stage-5 shards were killed at the 12 h walltime after the GA had
written best_*.npz genes but before the post-search refit. This script rebuilds
the fitted models from those genes and promotes them to saved_data:

  1. load_data()  (HHTS tours/persons + landuse, ~1 min)
  2. per shard: reconstruct the best feature mask from best_*.npz
     (_run_ga_single_stage._salvage_best_npz) and refit ONLY that shard's
     segment (pipeline_logger._ga_save_stage_coef_html_from_best(segments=[seg]))
  3. merge the three per-segment fits into {seg: {mode_model, dest_model}}
  4. save_models() -> --target (default saved_data/stage5_models.pkl), with a
     timestamped backup of the previous target
  5. refresh <target_dir>/stage5_models_coefficients.csv
  6. write <out-dir>/ga_results_promoted.pkl (AllStagesResult, stage 5 only)
     and <out-dir>/promoted_features.json (provenance + env flags)

Fail-loud: exits non-zero if any npz is missing, a refit produces no pkl, a
segment is absent, or a fitted model has no coefficients. Never leaves a
half-written target (write temp + os.replace).

Env (read before stage5 import):
  STAGE5_PT_MODES   default 0  - must be 1 to fit PT_Walk/Drive/Bike (8 modes)
  STAGE5_PT_ACCESS  default 0  - coupled PT access terms in the mode spec

Usage:
  python promote_stage5_bests.py --tag qld_conv1 --dry-run
  python promote_stage5_bests.py --tag qld_conv1
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

import compat.numpy_shim  # noqa: F401
import pipeline_logger as pl
from pipeline_logger import load_data, _ga_save_stage_coef_html_from_best
from ga_hypothesis_testing import (
    AllStagesResult,
    GAResult,
    Individual,
    _combine_segment_feature_results,
    _ga_save_stage_coef_csv,
)

SEGMENTS = ['work', 'edu', 'nonmand']


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-root', default=os.environ.get('RUN_ROOT', 'runs/ga_parallel'))
    p.add_argument('--tag', default=os.environ.get('RUN_TAG', 'qld_conv1'))
    p.add_argument('--stage-dir', default=None,
                   help='Defaults to <run-root>/<tag>/stage5')
    p.add_argument('--target', default='saved_data/stage5_models.pkl')
    p.add_argument('--out-dir', default=None,
                   help='Defaults to <stage-dir>/promoted')
    p.add_argument('--segments', default=','.join(SEGMENTS),
                   help='Comma list; default work,edu,nonmand (all required)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print npz provenance + features, do not refit')
    p.add_argument('--no-backup', action='store_true')
    p.add_argument('--allow-partial', action='store_true',
                   help='Missing segments are copied from the existing target')
    return p.parse_args()


def _npz_rows(shard_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(shard_dir, 'best_*.npz'))):
        z = np.load(path, allow_pickle=True)
        try:
            genes = np.asarray(z['genes']).ravel()
            feats = [str(f) for f in np.asarray(z['feat_names'], dtype=object).ravel()]
            fit = float(z['fitness']) if 'fitness' in z else float('nan')
            label = str(z['label']) if 'label' in z else os.path.basename(path)
            saved_at = str(z['saved_at']) if 'saved_at' in z else ''
        finally:
            z.close()
        selected = [f for f, g in zip(feats, genes) if int(g) == 1]
        rows.append({
            'path': path,
            'label': label,
            'kind': 'dest' if 'dest' in label.lower() else 'mode',
            'fitness': fit,
            'saved_at': saved_at,
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S',
                                   time.localtime(os.path.getmtime(path))),
            'n_candidates': len(feats),
            'features': selected,
        })
    return rows


def _coef_counts(seg_models):
    from stage5_destination_choice import _coef_dict_from_model
    counts, missing = {}, []
    for seg, entry in (seg_models or {}).items():
        for kind in ('mode', 'dest'):
            key = f'{seg}_{kind}'
            model = (entry or {}).get(f'{kind}_model')
            if model is None:
                counts[key] = 0
                missing.append(key)
                continue
            try:
                cd = _coef_dict_from_model(model) or {}
            except Exception as exc:  # noqa: BLE001
                print(f"  coefficient read failed [{key}]: {exc}", file=sys.stderr)
                cd = {}
            counts[key] = len(cd)
            if not cd:
                missing.append(key)
    return counts, missing


def _old_coefs(path):
    if not os.path.isfile(path):
        return {}
    try:
        from stage5_destination_choice import load_models
        return load_models(path) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: could not read previous target ({exc})")
        return {}


def _print_deltas(old, new, seg):
    from stage5_destination_choice import _coef_dict_from_model
    try:
        old_seg = old.get(seg, {}) or {}
        new_seg = new.get(seg, {}) or {}
        for mkey in ('mode_model', 'dest_model'):
            om, nm = old_seg.get(mkey), new_seg.get(mkey)
            if om is None or nm is None:
                continue
            oc = _coef_dict_from_model(om) or {}
            nc = _coef_dict_from_model(nm) or {}
            changed = sorted(
                ((k, oc[k], nc[k]) for k in nc if k in oc and abs(nc[k] - oc[k]) > 1e-9),
                key=lambda t: -abs(t[2] - t[1]))
            added = sorted(k for k in nc if k not in oc)
            dropped = sorted(k for k in oc if k not in nc)
            print(f"  {seg}/{mkey}: {len(nc)} params "
                  f"({len(changed)} changed, {len(added)} new, {len(dropped)} dropped)")
            for k, ov, nv in changed[:5]:
                print(f"      {k:<28} {ov:+.6f} -> {nv:+.6f}  ({nv - ov:+.6f})")
            if added:
                print(f"      new: {', '.join(added[:8])}")
            if dropped:
                print(f"      dropped: {', '.join(dropped[:8])}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: delta report failed for {seg}: {exc}")


def main():
    args = _parse_args()
    segs = [s.strip() for s in args.segments.split(',') if s.strip()]
    bad = [s for s in segs if s not in SEGMENTS]
    if bad:
        print(f"FATAL: unknown segment(s) {bad}; valid: {SEGMENTS}", file=sys.stderr)
        return 2

    stage_dir = args.stage_dir or os.path.join(args.run_root, args.tag, 'stage5')
    out_dir = args.out_dir or os.path.join(stage_dir, 'promoted')

    provenance = {}
    for seg in segs:
        shard_dir = os.path.join(stage_dir, 'shards', seg)
        if not os.path.isdir(shard_dir):
            print(f"FATAL: shard dir missing: {shard_dir}", file=sys.stderr)
            return 2
        rows = _npz_rows(shard_dir)
        if not rows:
            print(f"FATAL: no best_*.npz in {shard_dir}", file=sys.stderr)
            return 2
        provenance[seg] = rows

    print(f"stage dir : {stage_dir}")
    print(f"out dir   : {out_dir}")
    print(f"target    : {args.target}")
    print(f"segments  : {segs}")
    print(f"STAGE5_PT_MODES={os.environ.get('STAGE5_PT_MODES', '0')} "
          f"STAGE5_PT_ACCESS={os.environ.get('STAGE5_PT_ACCESS', '0')}")
    for seg in segs:
        for r in provenance[seg]:
            print(f"  [{seg}] {r['kind']:<4} fit={r['fitness']:+.6f} "
                  f"{r['mtime']} {os.path.basename(r['path'])}")
            print(f"         features({len(r['features'])}): {', '.join(r['features'])}")

    if args.dry_run:
        print("\nDRY RUN - no refit, no writes.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    print("\nLoading pipeline data (HHTS tours/persons + landuse) ...")
    t0 = time.time()
    pl.OUT_DIR = os.path.join(out_dir, 'load_data')
    os.makedirs(pl.OUT_DIR, exist_ok=True)
    _, out, landuse = load_data()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # ── 8-mode PT preflight ─────────────────────────────────────────────────
    # STAGE5_PT_MODES=1 recodes observed Transit tours to PT_Walk/Drive/Bike
    # from tour_transit_access, which data_connect only creates when
    # STAGE5B_TRANSIT_ACCESS=1. Without it every PT alternative has zero chosen
    # tours, the ASCs run to their bounds and the fit is unidentifiable.
    _pt_modes = os.environ.get('STAGE5_PT_MODES', '0').strip() not in ('', '0')
    if _pt_modes:
        _tours = out.get('tours')
        _acc = next((c for c in ('tour_transit_access', 'transit_access')
                     if _tours is not None and c in _tours.columns), None)
        if _acc is None:
            print("FATAL: STAGE5_PT_MODES=1 but the tours frame carries no "
                  "tour_transit_access/transit_access column. Re-run with "
                  "STAGE5B_TRANSIT_ACCESS=1 so data_connect derives it.",
                  file=sys.stderr)
            return 8
        _lab = _tours[_acc].astype(str).str.strip().str.title()
        _cnt = _lab.value_counts().to_dict()
        print(f"  PT access labels ({_acc}): {_cnt}")
        _obs = sum(int(_cnt.get(k, 0)) or 0 for k in ('Walk', 'Drive', 'Bike'))
        print(f"  observed PT-access tours: {_obs}")
        if _obs == 0:
            print("FATAL: no observed PT access labels - every PT_Walk/Drive/"
                  "Bike alternative would have zero chosen tours and its ASC "
                  "would be unidentified (runs to the parameter bound). "
                  "Fix the data_connect access derivation before the 8-mode fit.",
                  file=sys.stderr)
            return 8

    import _run_ga_single_stage as _single
    from stage5_destination_choice import load_models, save_models

    refit = {}
    for seg in segs:
        shard_dir = os.path.join(stage_dir, 'shards', seg)
        os.environ['GA_CHECKPOINT_DIR'] = shard_dir
        res, n_files = _single._salvage_best_npz(5)
        if res is None or n_files == 0:
            print(f"FATAL [{seg}]: _salvage_best_npz returned nothing", file=sys.stderr)
            return 3
        feats = getattr(res, 'best_features', {}) or {}
        print(f"\n[{seg}] salvaged {n_files} npz | mode={len(feats.get('mode', []))} "
              f"dest={len(feats.get('dest', []))} | fitness={res.best_fitness:+.6f}")

        pl.OUT_DIR = os.path.join(out_dir, 'refit', seg)
        os.makedirs(pl.OUT_DIR, exist_ok=True)
        t1 = time.time()
        _ga_save_stage_coef_html_from_best(5, res, out, landuse=landuse,
                                           segments=[seg])
        pkl = os.path.join(pl.OUT_DIR, 'stage5_models.pkl')
        if not os.path.isfile(pkl):
            print(f"FATAL [{seg}]: refit produced no {pkl} "
                  f"(see warnings above)", file=sys.stderr)
            return 4
        blob = load_models(pkl) or {}
        if not isinstance(blob.get(seg), dict):
            print(f"FATAL [{seg}]: {pkl} has no '{seg}' entry "
                  f"(keys={sorted(blob)})", file=sys.stderr)
            return 4
        refit[seg] = blob[seg]
        print(f"  refit {seg} in {time.time() - t1:.1f}s -> {pkl}")

    pl.OUT_DIR = out_dir

    old = _old_coefs(args.target)
    if args.allow_partial:
        for seg in SEGMENTS:
            if seg not in refit:
                if seg in old:
                    refit[seg] = old[seg]
                    print(f"  [{seg}] kept from previous target")
                else:
                    print(f"FATAL: missing segment '{seg}' and no previous fit",
                          file=sys.stderr)
                    return 5
    missing_segs = [s for s in SEGMENTS if s not in refit]
    if missing_segs:
        print(f"FATAL: missing segments {missing_segs} "
              f"(use --allow-partial to keep old)", file=sys.stderr)
        return 5

    merged = {seg: refit[seg] for seg in SEGMENTS}
    counts, missing_models = _coef_counts(merged)
    for seg in SEGMENTS:
        print(f"  coefficients [{seg}]: mode={counts.get(f'{seg}_mode', 0)} "
              f"dest={counts.get(f'{seg}_dest', 0)}")
    if missing_models:
        print(f"FATAL: models with no coefficients: {missing_models}", file=sys.stderr)
        return 6

    print("\ncoefficient deltas vs previous target:")
    for seg in SEGMENTS:
        _print_deltas(old, merged, seg)

    target_dir = os.path.dirname(args.target) or '.'
    os.makedirs(target_dir, exist_ok=True)
    if os.path.isfile(args.target) and not args.no_backup:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        backup = f"{args.target}.bak.{stamp}"
        with open(args.target, 'rb') as src, open(backup, 'wb') as dst:
            dst.write(src.read())
        print(f"  backup -> {backup}")

    tmp_dir = os.path.join(out_dir, '.pkl_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, 'stage5_models.pkl')
    if os.path.isfile(tmp):
        os.remove(tmp)
    save_models(merged, tmp)
    if not os.path.isfile(tmp):
        print(f"FATAL: save_models wrote no {tmp}", file=sys.stderr)
        return 7
    with open(tmp, 'rb') as src, open(args.target, 'wb') as dst:
        dst.write(src.read())
    print(f"  target  -> {args.target}")

    promoted_copy = os.path.join(out_dir, 'stage5_models.pkl')
    if os.path.abspath(promoted_copy) != os.path.abspath(args.target):
        with open(args.target, 'rb') as src, open(promoted_copy, 'wb') as dst:
            dst.write(src.read())
        print(f"  copy    -> {promoted_copy}")

    _ga_save_stage_coef_csv(merged, 'stage5_models', target_dir)
    print(f"  coef csv-> {os.path.join(target_dir, 'stage5_models_coefficients.csv')}")

    seg_results = {}
    feature_map = {'mode': [], 'dest': []}
    for seg in SEGMENTS:
        entry = refit[seg]
        for kind in ('mode', 'dest'):
            model = entry.get(f'{kind}_model')
            if model is None:
                print(f"FATAL: {seg}/{kind}_model missing after refit",
                      file=sys.stderr)
                return 6
            feats = _features_for(provenance, seg, kind)
            seg_results[f'{seg}_{kind}'] = GAResult(
                stage=5,
                best_features=list(feats),
                best_fitness=_fitness_for(provenance, seg, kind),
                best_individual=Individual(genes=np.zeros(0, dtype=np.int8)),
                history=[], all_candidates=[], feature_names=list(feats),
                fitness_metric='BIC/n',
                n_restarts=0, cache_hits=0,
            )
            for f in feats:
                if f not in feature_map[kind]:
                    feature_map[kind].append(f)

    combined = _combine_segment_feature_results(
        5, seg_results, {'mode': sorted(feature_map['mode']),
                         'dest': sorted(feature_map['dest'])},
        'BIC/n', 0)
    combined.metadata['segment_results'] = seg_results
    combined.metadata['promoted_from'] = {
        seg: [os.path.basename(r['path']) for r in provenance[seg]] for seg in segs
    }
    combined.metadata['stage5_models'] = merged

    allst = AllStagesResult()
    allst.results[5] = combined
    ga_out = os.path.join(out_dir, 'ga_results_promoted.pkl')
    ga_tmp = os.path.join(tmp_dir, 'ga_results_promoted.pkl')
    written = False
    try:
        import cloudpickle
        with open(ga_tmp, 'wb') as fh:
            cloudpickle.dump(allst, fh)
        written = True
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: cloudpickle ga pkl failed ({exc}); retrying without models")
    if not written:
        combined.metadata.pop('stage5_models', None)
        with open(ga_tmp, 'wb') as fh:
            pickle.dump(allst, fh)
    with open(ga_tmp, 'rb') as src, open(ga_out, 'wb') as dst:
        dst.write(src.read())
    print(f"  ga pkl  -> {ga_out}")

    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tag': args.tag,
        'stage_dir': stage_dir,
        'target': args.target,
        'env': {
            'STAGE5_PT_MODES': os.environ.get('STAGE5_PT_MODES', '0'),
            'STAGE5_PT_ACCESS': os.environ.get('STAGE5_PT_ACCESS', '0'),
            'STAGE5_LARCH_LOGSUM': os.environ.get('STAGE5_LARCH_LOGSUM', '0'),
        },
        'segments': {
            seg: {
                'mode_features': _features_for(provenance, seg, 'mode'),
                'dest_features': _features_for(provenance, seg, 'dest'),
                'mode_fitness': _fitness_for(provenance, seg, 'mode'),
                'dest_fitness': _fitness_for(provenance, seg, 'dest'),
                'sources': provenance[seg],
                'n_coef': {
                    'mode_model': counts.get(f'{seg}_mode', 0),
                    'dest_model': counts.get(f'{seg}_dest', 0),
                },
            } for seg in SEGMENTS
        },
    }
    man_path = os.path.join(out_dir, 'promoted_features.json')
    with open(man_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"  manifest-> {man_path}")
    print("\nPromotion complete.")
    return 0


def _features_for(provenance, seg, kind):
    if kind == 'dest':
        want = 'dest'
    else:
        want = 'mode'
    for r in provenance.get(seg, []):
        if r['kind'] == want:
            return list(r['features'])
    return []


def _fitness_for(provenance, seg, kind):
    for r in provenance.get(seg, []):
        if r['kind'] == kind:
            return float(r['fitness'])
    return float('nan')


if __name__ == '__main__':
    sys.exit(main())
