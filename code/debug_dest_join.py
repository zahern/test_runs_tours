#!/usr/bin/env python
"""debug_dest_join.py — Stage-5 destination OD-join key diagnosis.

The Sep-12 plc_work_test run logged Walk=nan% Bike=nan% (DA=93.8%) from
_mode_probs_numpy on 6909 HHTS work tours: the Walk/Bike utilities were NaN
while DA/SR/Transit stayed finite, which means a Walk/Bike-only input
(dist_walk/dist_bike) missed its skim join. This script reproduces the
zone-key chain from the real input files and names the broken link — no
estimation, no JAX/Larch (pandas + numpy; OMX keys via openmatrix, with an
h5py fallback since OMX is plain HDF5).

Key-space map (do NOT mix them):
  * ORDER codes 1-330/1-371  -> legacy downcasted_matrices_base.omx (1-330)
    and HHTS tour_origin/tour_destination columns.
  * 9-digit SA2 codes        -> skims_371.omx mapping, zone_landuse_overture
    SA2_MAIN16, trips DESTSA2_2021/ORIGSA2_2021, 371 crosswalk.
The 371 SA2->ORDER crosswalk bridges them (99.9% of trip rows).

Run on the login node (zigenv has pandas; h5py covers the skim leg):
    cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
    python debug_dest_join.py

Exit 0 HEALTHY / 1 BROKEN (names the link) / 2 INCONCLUSIVE.
"""
import os
import sys

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(CODE, 'saved_data')
TRIPS = os.path.join(SAVED, 'trips_processed.csv')
SEQ_LEGACY = os.path.join(CODE, 'seq_zone_ids_.csv')          # SA2->ORDER 1-330
SEQ_371 = os.path.join(SAVED, 'seq_zone_ids_371.csv')         # SA2->ORDER 1-371
OMX = os.path.abspath(os.path.join(CODE, '..', 'demand_', 'skims_371.omx'))
LANDUSE = os.path.join(SAVED, 'zone_landuse_overture.csv')


def _fmt(s, n=6):
    v = pd.to_numeric(s, errors='coerce').dropna().astype(np.int64)
    if len(v) == 0:
        return {'n': 0}
    return {'n_unique': int(v.nunique()), 'min': int(v.min()),
            'max': int(v.max()),
            'nine_digit_frac': float(((v >= 100000000)).mean()),
            'sample': v.drop_duplicates().head(n).tolist()}


def _show(name, s):
    f = _fmt(s)
    if 'n_unique' not in f:
        print(f'  {name}: EMPTY/unreadable')
        return set()
    print(f"  {name}: nunique={f['n_unique']} range=[{f['min']},{f['max']}] "
          f"9-digit_frac={f['nine_digit_frac']:.3f} sample={f['sample']}")
    return set(pd.to_numeric(s, errors='coerce').dropna().astype(int).tolist())


def _omx_keys_h5py(path):
    """Read OMX zone keys with h5py (OMX is plain HDF5). Returns
    (keys, note) or (None, reason)."""
    try:
        import h5py
    except Exception as e:
        return None, f'h5py unavailable ({e})'
    if not os.path.exists(path):
        return None, 'file missing'
    try:
        h = h5py.File(path, 'r')
    except Exception as e:
        return None, f'open failed ({e})'
    try:
        if 'lookup' not in h:
            return None, 'no lookup group'
        lk = h['lookup']
        if not hasattr(lk, 'keys'):
            ids = np.asarray(lk).ravel()
            return sorted(int(v) for v in ids), 'lookup dataset'
        for m in lk.keys():
            g = lk[m]
            if hasattr(g, 'keys') and 'ID' in g:
                ids = np.asarray(g['ID']).ravel()
                return sorted(int(v) for v in ids), f'lookup/{m}/ID'
        # legacy layout: lookup/ids dataset
        for m in lk.keys():
            g = lk[m]
            if not hasattr(g, 'keys'):
                try:
                    ids = np.asarray(g).ravel()
                    if len(ids) > 10:
                        return (sorted(int(v) for v in ids),
                                f'lookup/{m}')
                except Exception:
                    continue
        return None, 'no mapping readable'
    except Exception as e:
        return None, f'read failed ({e})'
    finally:
        try:
            h.close()
        except Exception:
            pass


def _omx_keys(path):
    keys, note = None, 'not attempted'
    try:
        import openmatrix as omx
    except Exception as e:
        keys, note = None, f'openmatrix unavailable ({e})'
    else:
        if not os.path.exists(path):
            return None, 'file missing'
        h = None
        try:
            h = omx.open_file(path, 'r')
            mappings = h.listMappings() if hasattr(h, 'listMappings') else []
            for m in mappings:
                try:
                    keys = sorted(int(k) for k in dict(h.mapping(m)).keys())
                    note = f'mapping {m}'
                    break
                except Exception:
                    continue
            if keys is None:
                note = 'no mapping readable'
        except Exception as e:
            note = f'open failed ({e})'
        finally:
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass
    if keys is not None:
        return keys, note + ' (openmatrix)'
    # fallback: h5py reads the same HDF5 containers directly
    keys2, note2 = _omx_keys_h5py(path)
    if keys2 is not None:
        return keys2, note2 + ' (h5py fallback)'
    return None, note + '; ' + note2


def _omx_nan_report(path):
    """{matrix: (nan_frac, all_nan_rows, all_nan_cols)}; {} if unreadable.
    Tries openmatrix first, h5py second (same HDF5 layout)."""
    mats = ('AUTO_TIME', 'SOVTOLL_TIME__AM', 'WLK_LOC_WLK_TOTIVT__AM',
            'DISTWALK', 'DISTBIKE', 'CRASH')
    out = {}
    h = None
    try:
        import openmatrix as omx
        h = omx.open_file(path, 'r')
        try:
            names = list(h.list_matrices())
        except Exception:
            names = list(h.listMatrices())
        for m in mats:
            if m not in names:
                continue
            try:
                try:
                    a = np.asarray(h[m][:, :], dtype=float)
                except Exception:
                    a = np.asarray(h[m], dtype=float)
                if a.ndim != 2:
                    continue
                nan = np.isnan(a)
                out[m] = (float(nan.mean()), int(nan.all(axis=1).sum()),
                          int(nan.all(axis=0).sum()))
            except Exception:
                continue
    except Exception:
        pass
    finally:
        try:
            if h is not None:
                h.close()
        except Exception:
            pass
    if out:
        return out
    # h5py fallback over /data/<matrix>
    try:
        import h5py
        h2 = h5py.File(path, 'r')
    except Exception:
        return out
    try:
        if 'data' not in h2:
            return out
        for m in mats:
            if m not in h2['data']:
                continue
            try:
                a = np.asarray(h2['data'][m], dtype=float)
                if a.ndim != 2:
                    continue
                nan = np.isnan(a)
                out[m] = (float(nan.mean()), int(nan.all(axis=1).sum()),
                          int(nan.all(axis=0).sum()))
            except Exception:
                continue
    finally:
        try:
            h2.close()
        except Exception:
            pass
    return out


def _h5_nan_report(path, label):
    """NaN audit for fine walk/bike h5 skims (lvl4/sa1 x SA2).
    Returns {dataset: (nan_frac, shape)}; {} if unreadable."""
    out = {}
    try:
        import h5py
    except Exception:
        return out
    if not os.path.exists(path):
        return out
    try:
        h = h5py.File(path, 'r')
    except Exception:
        return out
    try:
        def _walk(obj, prefix=''):
            for k in obj.keys():
                p = prefix + '/' + k
                if hasattr(obj[k], 'keys'):
                    yield from _walk(obj[k], p)
                else:
                    yield p
        for ds in _walk(h):
            base = ds.split('/')[-1]
            if base not in ('DISTWALK', 'DISTBIKE'):
                continue
            try:
                a = np.asarray(h[ds], dtype=float)
            except Exception:
                continue
            if a.ndim != 2:
                continue
            out[f'{label}:{base}'] = (float(np.isnan(a).mean()), a.shape)
    finally:
        try:
            h.close()
        except Exception:
            pass
    return out


def main():
    broken, inconclusive = [], []
    print('== A. trip zone columns (as the pipeline reads them) ==')
    try:
        trips = pd.read_csv(TRIPS, usecols=lambda c: c in (
            'OTAZ', 'DTAZ', 'DESTSA2_2021', 'ORIGSA2_2021',
            'tour_origin', 'tour_destination'))
    except Exception as e:
        print(f'INCONCLUSIVE: trips unreadable ({e})')
        return 2
    tour_keys = set()
    for c in ('OTAZ', 'DTAZ', 'tour_origin', 'tour_destination',
              'DESTSA2_2021', 'ORIGSA2_2021'):
        if c in trips.columns:
            s = _show(f'trips.{c}', trips[c])
            if c in ('DTAZ', 'tour_destination', 'DESTSA2_2021'):
                tour_keys |= s
    try:
        desa = pd.to_numeric(trips['DESTSA2_2021'], errors='coerce').dropna().astype(int)
    except Exception:
        desa = pd.Series(dtype=float)

    print('== B. crosswalk mappings applied to trip SA2s ==')
    xw371 = {}
    for label, path in (('legacy SA2->ORDER(1-330)', SEQ_LEGACY),
                        ('371 SA2->ORDER(1-371)', SEQ_371)):
        try:
            xw = pd.read_csv(path, usecols=lambda c: c in ('SA2_MAIN16', 'ORDER'))
            m = dict(zip(pd.to_numeric(xw['SA2_MAIN16'], errors='coerce').dropna().astype(int),
                         pd.to_numeric(xw['ORDER'], errors='coerce').dropna().astype(int)))
            if '371' in label:
                xw371 = m
            mapped = desa.map(m)
            hit = float(mapped.notna().mean()) if len(desa) else 0.0
            vals = mapped.dropna().astype(int)
            rng = f'[{vals.min()},{vals.max()}]' if len(vals) else 'n/a'
            print(f'  {label}: {len(m)} entries; trip-DESTSA2 match={hit:.1%} -> codes {rng}')
            if 'legacy' in label:
                # The legacy 330 system is retired; a miss here only matters
                # if the 371 crosswalk ALSO misses (checked below). Record
                # for the verdict, do not flag yet.
                legacy_hit = hit
            else:
                if hit < 0.99:
                    broken.append(f'{label} covers only {hit:.1%}')
        except Exception as e:
            print(f'  {label}: unreadable ({e})')
            inconclusive.append(label)

    print('== C. skim OMX keys ==')
    okeys, note = _omx_keys(OMX)
    if okeys is None:
        print(f'  INCONCLUSIVE on skim leg ({note})')
        inconclusive.append('skim keys: ' + note)
        oset = set()
    else:
        oset = set(okeys)
        print(f'  skim keys: n={len(oset)} range=[{min(oset)},{max(oset)}] '
              f'sample={sorted(oset)[:6]} ({note})')
        # Like-for-like: 9-digit trip SA2s vs skim keys (same key space).
        # ORDER tour codes are bridged via the 371 crosswalk first.
        desa_set = set(pd.to_numeric(
            trips['DESTSA2_2021'], errors='coerce'
        ).dropna().astype(int).tolist()) if 'DESTSA2_2021' in trips.columns else set()
        if desa_set:
            hit = len(desa_set & oset) / len(desa_set)
            n_rows_miss = int(trips['DESTSA2_2021'].isin(
                sorted(desa_set - oset)).sum()) if desa_set - oset else 0
            print(f'  trip DESTSA2 vs skim keys: '
                  f'{len(desa_set & oset)}/{len(desa_set)} ({hit:.1%}); '
                  f'rows affected={n_rows_miss}/{len(trips)}')
            if hit < 0.99 and n_rows_miss > 0.01 * len(trips):
                broken.append(f'trip DESTSA2 vs skim keys only {hit:.1%} '
                              f'({n_rows_miss} rows)')
            elif desa_set - oset:
                print(f'  note: {len(desa_set - oset)} out-of-scope SA2 codes '
                      f'(fringe/quarantine, {n_rows_miss} rows)')
        _nanrep = _omx_nan_report(OMX)
        if _nanrep:
            print('  skim NaN audit (missing = off-network/fringe pairs):')
            for _mn, (_nf, _nr, _nc) in _nanrep.items():
                print(f'    {_mn}: NaN={_nf:.1%} all-NaN rows={_nr} cols={_nc}')
                if _mn in ('DISTWALK', 'DISTBIKE') and _nf > 0.01:
                    broken.append(
                        f'skim {_mn} has {_nf:.1%} NaN — dist miss -> '
                        f'V_WK/V_BK NaN -> Walk=nan% Bike=nan%')

    print('== D. landuse keys (like-for-like key spaces) ==')
    try:
        lu = pd.read_csv(LANDUSE, nrows=3)
        key_col = next((c for c in ('ORDER', 'SA2_MAIN16', 'TAZ', 'zone_id')
                        if c in lu.columns), None)
        print(f'  landuse key column: {key_col}')
        if key_col is None:
            broken.append('landuse has no recognisable zone key')
        else:
            lk = set(pd.to_numeric(pd.read_csv(
                LANDUSE, usecols=[key_col])[key_col],
                errors='coerce').dropna().astype(int).tolist())
            print(f'  landuse keys: n={len(lk)} sample={sorted(lk)[:6]}')
            # Leg 1: 9-digit trip SA2s vs landuse SA2s (same key space).
            desa_set = set(pd.to_numeric(
                trips['DESTSA2_2021'], errors='coerce'
            ).dropna().astype(int).tolist()) if 'DESTSA2_2021' in trips.columns else set()
            if desa_set:
                hit_d = len(desa_set & lk) / len(desa_set)
                n_rows_miss = int(trips['DESTSA2_2021'].isin(
                    sorted(desa_set - lk)).sum()) if desa_set - lk else 0
                print(f'  trip DESTSA2 vs landuse SA2: '
                      f'{len(desa_set & lk)}/{len(desa_set)} distinct '
                      f'({hit_d:.1%}); rows affected={n_rows_miss}/{len(trips)}')
                if hit_d < 0.99 and n_rows_miss > 0.01 * len(trips):
                    broken.append(
                        f'trip DESTSA2 vs landuse only {hit_d:.1%} '
                        f'({n_rows_miss} rows)')
                elif desa_set - lk:
                    print(f'  note: {len(desa_set - lk)} out-of-scope SA2 codes '
                          f'(fringe/quarantine, {n_rows_miss} rows): '
                          f'{sorted(desa_set - lk)[:8]}')
            # Leg 2: ORDER tour codes -> SA2 via inverse 371 crosswalk, then
            # vs landuse (bridges the ORDER leg without mixing key spaces).
            order_set = set()
            for c in ('DTAZ', 'tour_destination'):
                if c in trips.columns:
                    order_set |= set(pd.to_numeric(
                        trips[c], errors='coerce').dropna().astype(int).tolist())
            order_set = {v for v in order_set if v < 100000000}
            if order_set and xw371:
                inv = {}
                for sa2, o in xw371.items():
                    inv.setdefault(int(o), int(sa2))
                bridged = {inv[v] for v in order_set if v in inv}
                hit_o = len(bridged & lk) / max(len(bridged), 1)
                print(f'  tour ORDER codes -> SA2 via 371-xw vs landuse: '
                      f'{len(bridged & lk)}/{len(bridged)} ({hit_o:.1%})')
                if bridged and hit_o < 0.99:
                    broken.append(f'tour ORDER->SA2 vs landuse only {hit_o:.1%}')
            elif order_set and not xw371:
                inconclusive.append('ORDER->SA2 bridge (371 crosswalk unreadable)')
    except Exception as e:
        print(f'  INCONCLUSIVE: landuse unreadable ({e})')
        inconclusive.append('landuse')

    print('== E. fine walk/bike h5 skims (lvl4/SA1 x SA2 NaN audit) ==')
    _fine_paths = [
        ('lvl4', os.path.abspath(os.path.join(
            CODE, '..', 'demand_', 'walk_bike_lvl4_to_sa2_skims.h5'))),
        ('sa1', os.path.abspath(os.path.join(
            CODE, '..', 'demand_', 'walk_bike_sa1_to_sa2_skims.h5'))),
    ]
    for _tag, _fp in _fine_paths:
        _rep = _h5_nan_report(_fp, _tag)
        if not _rep:
            print(f'  {_tag}: unreadable/missing ({_fp})')
            continue
        for _ds, (_nf, _sh) in _rep.items():
            print(f'  {_ds}: shape={_sh} NaN={_nf:.2%}')
            if _nf > 0.01:
                broken.append(
                    f'{_ds} has {_nf:.1%} NaN — dist_walk/dist_bike miss '
                    f'-> V_WK/V_BK NaN -> Walk=nan% Bike=nan%')

    print()
    try:
        _legacy_note = (f'legacy 330 crosswalk covers {legacy_hit:.1%} '
                        f'(retired system; 371 crosswalk is authoritative)')
    except NameError:
        _legacy_note = 'legacy 330 crosswalk unreadable'
    print(f'  note: {_legacy_note}')
    if broken and hit < 0.99:
        print('JOIN BROKEN:')
        for b in broken:
            print(f'  - {b}')
        print('Fix the named mapping (crosswalk/order-system), then resubmit '
              'stage 5. Do NOT mask it with a DIST-free spec (hides data rot).')
        return 1
    if inconclusive and not oset:
        print('INCONCLUSIVE (skim keys unreadable; landuse/tour legs above still apply):')
        for b in inconclusive:
            print(f'  - {b}')
        return 2
    print('JOIN HEALTHY: key-spaces agree (>=99%). If dest still fails, look '
          'downstream (sampled-alts universe, crash join, logsum coords).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
