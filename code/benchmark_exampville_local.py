#!/usr/bin/env python
"""Local Exampville benchmark — full-pipeline dry run WITHOUT the HPC stack.

Runs end-to-end on this machine using ONLY stdlib + pandas/numpy/scipy/
statsmodels/matplotlib (whatever is installed): download-once of the Larch
example data, Exampville->pipeline mapping (reuses exampville_connect's pure
pandas helpers), then benchmark-grade estimators:

  B0  CDAP home/nonmand/mandatory shares  (statsmodels MNLogit)
  B1  work-tour frequency                 (statsmodels Poisson)
  B2  work tour mode choice               (statsmodels MNLogit; FLAT MNL —
      no nesting, so LL is not comparable to the nested docs-201 fit)
  B3  destination choice, 20 sampled alts (scipy L-BFGS-B MNL:
      log-dist + linear dist + log size + mode-logsum-free)

This is a BENCHMARK of data flow + model plumbing, not the Larch/JAX
estimators the cluster jobs use (no larch, openmatrix, jax, or SearchLibrium
imports anywhere below). Metrics mirror exampville_compare.py (mode_mae,
dest_mae/top10, trip_len_mae) plus walltimes, written to
runs/benchmark_exampville/benchmark_metrics.json. NOTHING is written to
saved_data/ or any QLD/HPC run tree.

Usage (Windows or Linux, plain python):
    python benchmark_exampville_local.py [--outdir runs/benchmark_exampville]
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import exampville_connect as evc  # pure pandas/numpy — no larch import

RAW_BASE = ("https://raw.githubusercontent.com/driftlesslabs/larch"
            "/main/src/larch/data_warehouse")
CANDIDATES = {
    "hh": ["exampville_households.csv.gz"],
    "pp": ["exampville_persons.csv.gz"],
    "tour": ["exampville_tours.csv.gz"],
    "emp": ["exampville_employment.csv.gz"],
    "skims": ["exampville_skims.omx"],
}
SKIM_MATS = ["AUTO_TIME", "AUTO_DIST", "WALK_DIST", "BIKE_TIME",
             "TRANSIT_IVTT", "TRANSIT_OVTT", "TRANSIT_FARE", "AUTO_COST"]


def _root(outdir=None):
    root = os.path.abspath(outdir or os.environ.get(
        "BENCH_ROOT", os.path.join("runs", "benchmark_exampville")))
    if "benchmark" not in os.path.basename(root).lower():
        raise RuntimeError(f"refusing non-benchmark root {root}")
    os.makedirs(root, exist_ok=True)
    return root


def _download(root):
    src = os.path.join(root, "source")
    os.makedirs(src, exist_ok=True)
    paths = {}
    for key, names in CANDIDATES.items():
        dst = None
        for nm in names:
            cand = os.path.join(src, nm)
            if os.path.exists(cand) and os.path.getsize(cand) > 0:
                dst = cand
                break
        if dst is None:
            errs = []
            for nm in names:
                url = f"{RAW_BASE}/{nm}"
                cand = os.path.join(src, nm)
                try:
                    urllib.request.urlretrieve(url, cand)
                    if os.path.getsize(cand) > 0:
                        dst = cand
                        break
                except Exception as e:
                    errs.append(f"{nm}: {e}")
                    try:
                        os.remove(cand)
                    except OSError:
                        pass
            if dst is None:
                raise RuntimeError(
                    f"could not fetch '{key}' data file (tried {names}): "
                    f"{errs}. Check network/raw base {RAW_BASE}.")
        print(f"  source {key}: {os.path.basename(dst)} "
              f"({os.path.getsize(dst) / 1e6:.1f} MB)", flush=True)
        paths[key] = dst
    return paths


def _read_omx_h5py(path, names):
    """Minimal OMX reader (h5py only): {matrix: 2D array} + TAZ id list."""
    import h5py
    mats, taz_ids = {}, None
    with h5py.File(path, "r") as f:
        data = f["data"] if "data" in f else f
        for nm in names:
            if nm in data:
                mats[nm] = np.asarray(data[nm], dtype=float)
        for key in ("TAZ_ID", "taz_id", "TAZ", "taz", "zones", "ZONE"):
            for grp in (f.get("lookup"), data):
                try:
                    if grp is not None and key in grp:
                        taz_ids = np.asarray(grp[key]).astype(int).ravel().tolist()
                        break
                except Exception:
                    continue
            if taz_ids is not None:
                break
    missing = [nm for nm in names if nm not in mats]
    if missing:
        print(f"  OMX missing (NaN-filled): {missing}", flush=True)
    if taz_ids is None:
        n = next(iter(mats.values())).shape[0]
        taz_ids = list(range(1, n + 1))
    return mats, taz_ids


def _numdrop(x):
    # pd.to_numeric returns ndarray (no .dropna) on old pandas for list-likes.
    return pd.Series(np.asarray(
        pd.to_numeric(x, errors="coerce"), dtype=float)).dropna()


def _mode_share_mae(pred, obs):
    po, oo = _numdrop(pred), _numdrop(obs)
    if len(po) == 0 or len(oo) == 0:
        return float("nan")
    pv, ov = po.value_counts(normalize=True), oo.value_counts(normalize=True)
    keys = set(pv.index) | set(ov.index)
    return float(np.mean([abs(pv.get(k, 0.0) - ov.get(k, 0.0)) for k in keys]))


def _dest_mae(pred_z, obs_z, top_n=10):
    pz, oz = _numdrop(pred_z), _numdrop(obs_z)
    if len(pz) == 0 or len(oz) == 0:
        return float("nan"), float("nan")
    pv, ov = pz.value_counts(normalize=True), oz.value_counts(normalize=True)
    keys = set(pv.index) | set(ov.index)
    mae = float(np.mean([abs(pv.get(k, 0.0) - ov.get(k, 0.0)) for k in keys]))
    tp, to = set(pv.nlargest(top_n).index), set(ov.nlargest(top_n).index)
    return mae, float(len(tp & to) / max(len(tp | to), 1))


def main():
    outdir = "runs/benchmark_exampville"
    for i, a in enumerate(sys.argv[1:]):
        if a == "--outdir" and i + 1 < len(sys.argv[1:]):
            outdir = sys.argv[1:][i + 1]
    root = _root(outdir)
    t_all = time.time()
    metrics = {"benchmark": "local-scipy-statsmodels-NO-larch", "arms": {}}
    print(f"=== Local Exampville benchmark -> {root} ===", flush=True)

    src = _download(root)
    hh_raw = pd.read_csv(src["hh"])
    pp_raw = pd.read_csv(src["pp"])
    tour_raw = pd.read_csv(src["tour"])
    try:
        emp_raw = pd.read_csv(src["emp"])
    except Exception:
        emp_raw = None
    print(f"raw: hh={len(hh_raw)} pp={len(pp_raw)} tours={len(tour_raw)}", flush=True)

    mats, taz_ids = _read_omx_h5py(src["skims"], SKIM_MATS)
    if "WALK_TIME" not in mats and "WALK_DIST" in mats:
        # v6 OMX ships WALK_DIST but no WALK_TIME: walk time at 5 km/h,
        # same conversion the pipeline applies to DISTWALK.
        mats["WALK_TIME"] = mats["WALK_DIST"] * 12.0
        print("  derived WALK_TIME = WALK_DIST*12 (5 km/h)", flush=True)
    skim_df = evc.build_skim_df(mats, taz_ids)
    hh = evc.map_households(hh_raw)
    pp = evc.map_persons(pp_raw, hh)
    lu = evc.map_landuse(emp_raw) if emp_raw is not None else None
    tours = evc.map_tours(tour_raw, pp, mats, taz_ids)
    tours.to_csv(os.path.join(root, "bench_tours.csv"), index=False)
    pp.to_csv(os.path.join(root, "bench_persons.csv"), index=False)
    autodist = {(int(o), int(d)): float(v)
                for (o, d), v in
                skim_df.set_index(["OTAZ", "DTAZ"])["DIST"].items()
                if np.isfinite(v)}
    metrics.update({"n_tours": int(len(tours)), "n_persons": int(len(pp)),
                    "n_zones": int(len(taz_ids))})

    # ── B0: CDAP shares (MNLogit H/N/M) ──
    try:
        import statsmodels.api as sm
        t0 = time.time()
        _nt = pd.to_numeric(pp["N_TOURS"], errors="coerce").fillna(0)
        _nw = pd.to_numeric(pp["N_WORK_TOURS"], errors="coerce").fillna(0)
        y = np.where(_nt.values == 0, 0, np.where(_nw.values > 0, 2, 1))
        X = pd.DataFrame({
            "const": 1.0,
            "age_working": pd.to_numeric(pp.get("age_working", 0),
                                         errors="coerce").fillna(0),
            "vehrd": pd.to_numeric(pp.get("vehrd", 1),
                                   errors="coerce").fillna(1),
        }).values
        res = sm.MNLogit(y, X).fit(disp=0, maxiter=200)
        pred = res.predict(X).argmax(axis=1)
        metrics["arms"]["B0"] = {"ok": True, "ll": float(res.llf),
                                 "n": int(len(y)),
                                 "mae": _mode_share_mae(pred, y),
                                 "sec": round(time.time() - t0, 1)}
    except Exception as e:
        metrics["arms"]["B0"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print("B0:", metrics["arms"]["B0"], flush=True)

    # ── B1: work-tour frequency (Poisson) ──
    try:
        import statsmodels.api as sm
        t0 = time.time()
        yw = pd.to_numeric(pp["N_WORK_TOURS"], errors="coerce").fillna(0)
        X = pd.DataFrame({
            "const": 1.0,
            "vehrd": pd.to_numeric(pp.get("vehrd", 1),
                                   errors="coerce").fillna(1),
            "n_workers_hh": pd.to_numeric(pp.get("n_workers_hh", 1),
                                          errors="coerce").fillna(1),
        })
        res = sm.Poisson(yw.values, X.values).fit(disp=0, maxiter=200)
        pred = res.predict(X.values)
        metrics["arms"]["B1"] = {
            "ok": True, "ll": float(res.llf), "n": int(len(yw)),
            "obs_mean": float(yw.mean()), "pred_mean": float(np.mean(pred)),
            "mean_mae": float(abs(float(yw.mean()) - float(np.mean(pred)))),
            "sec": round(time.time() - t0, 1)}
    except Exception as e:
        metrics["arms"]["B1"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print("B1:", metrics["arms"]["B1"], flush=True)

    # ── B2: work tour mode choice (flat MNL mimicking docs-201 utility) ──
    try:
        import statsmodels.api as sm
        t0 = time.time()
        work = tours[pd.to_numeric(tours["tour_mode"],
                                   errors="coerce").isin([1, 2, 3, 4, 5])].copy()
        _home = dict(zip(pp["person_id"].astype(str),
                         pd.to_numeric(pp["OTAZ"], errors="coerce").astype("Int64").astype(str)))
        _inc = dict(zip(pp["person_id"].astype(str),
                        pd.to_numeric(pp["incp"], errors="coerce")))
        _o = work["person_id"].astype(str).map(_home)
        _key = {(int(o), int(d)): i for i, (o, d) in
                enumerate(zip(skim_df["OTAZ"].astype(int), skim_df["DTAZ"].astype(int)))}
        _ix = [ _key.get((int(float(o)), int(float(d))), -1)
                for o, d in zip(_o, work["DTAZ"])]
        for _col, _arr in (("AT", "AUTO_TIME"), ("CT", "AUTO_COST"),
                           ("WT", "WALK_TIME"), ("BT", "BIKE_TIME"),
                           ("IV", "TRANSIT_IVTT"), ("OV", "TRANSIT_OVTT"),
                           ("FR", "TRANSIT_FARE")):
            _v = np.asarray(mats.get(_arr, np.full((len(taz_ids),) * 2, np.nan)),
                            dtype=float).ravel()
            work[_col] = [_v[i] if 0 <= i < len(_v) else np.nan for i in _ix]
        work["LI"] = np.log(pd.to_numeric(
            work["person_id"].astype(str).map(_inc), errors="coerce").fillna(8.0))
        keep = work.dropna(subset=["AT", "CT", "WT", "BT", "IV", "OV", "FR", "LI",
                                   "tour_mode"]).copy()
        # case-specific MNL via alternative-specific constants + generic slopes:
        # stack alternative rows (long format) — DA base
        _alts = {1: ("DA", ("AT", "CT")), 2: ("SR", ("AT", "CT")),
                 3: ("Walk", ("WT",)), 4: ("Bike", ("BT",)),
                 5: ("Transit", ("IV", "OV", "FR"))}
        import itertools
        rows = []
        for _, r in keep.iterrows():
            for _a, (_nm, _cc) in _alts.items():
                _row = {"case": r["ptour_id"], "alt": _a,
                        "chosen": int(r["tour_mode"] == _a), "LI": r["LI"]}
                for _c in ("AT", "CT", "WT", "BT", "IV", "OV", "FR"):
                    _row[_c] = float(r[_c]) if _c in _cc else 0.0
                rows.append(_row)
        long = pd.DataFrame(rows)
        Xd = pd.get_dummies(long["alt"], prefix="ASC", drop_first=True)
        for _c in ("AT", "CT", "WT", "BT", "IV", "OV", "FR", "LI"):
            Xd[_c] = long[_c].values
        Xd.insert(0, "const", 1.0)
        # pandas>=2 get_dummies returns bool columns -> object-dtype .values,
        # which statsmodels' np.isfinite rejects (HPC: pandas 2.3/numpy 2.3).
        Xd = Xd.astype(float)
        import statsmodels.discrete.discrete_model as _dm
        res = _dm.Logit(long["chosen"].values, Xd.values).fit(
            disp=0, maxiter=200)
        # argmax predicted mode per case
        long["p"] = res.predict(Xd.values)
        pred = long.loc[long.groupby("case")["p"].idxmax()][
            ["case", "alt"]].set_index("case")["alt"]
        obs = keep.set_index("ptour_id")["tour_mode"].astype(int)
        common = obs.index.intersection(pred.index)
        metrics["arms"]["B2"] = {
            "ok": True, "ll": float(res.llf), "n": int(len(keep)),
            "mode_mae": _mode_share_mae(pred.loc[common], obs.loc[common]),
            "sec": round(time.time() - t0, 1)}
    except Exception as e:
        import traceback
        metrics["arms"]["B2"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-1500:]}
    print("B2:", {k: v for k, v in metrics["arms"]["B2"].items()
                  if k != "trace"}, flush=True)

    # ── B3: destination MNL, 20 sampled alts (scipy) ──
    try:
        from scipy.optimize import minimize
        t0 = time.time()
        rng = np.random.default_rng(42)
        _size = None
        if lu is not None:
            _s = pd.to_numeric(lu.get("TOTEMP", lu.get("TOTAL_EMP", 0)),
                               errors="coerce").fillna(0).values.astype(float)
            _size = {int(z): float(v) for z, v in zip(lu.index.tolist(), _s)}
        _zones = [int(z) for z in taz_ids]
        _cases, _alts_list, _n_skip = [], [], 0
        for _, r in tours.iterrows():
            try:
                _o, _c = int(float(r["OTAZ"])), int(float(r["DTAZ"]))
            except Exception:
                continue
            _pool = [a for a in rng.choice(
                _zones, size=min(19, len(_zones)), replace=False).tolist()
                if a != _c]
            _alts = [_c] + _pool
            _rows, _zl = [], []
            for _z in _alts:
                _d = autodist.get((_o, int(_z)), np.nan)
                if not np.isfinite(_d):
                    continue
                _rows.append((np.log(max(float(_d), 0.0) + 1.0),
                              float(max(_d, 0.0)),
                              np.log(float((_size or {}).get(int(_z), 0.0)) + 1.0),
                              1 if int(_z) == _c else 0))
                _zl.append(int(_z))
            if len(_rows) >= 2 and sum(_r[3] for _r in _rows) == 1:
                _a = np.asarray(_rows, dtype=float)
                _cases.append((_a[:, 0], _a[:, 1], _a[:, 2], _a[:, 3]))
                _alts_list.append(_zl)
            else:
                _n_skip += 1

        def _nll(th):
            _tot = 0.0
            for _ld, _dl, _st, _ch in _cases:
                _u = th[0] * _ld + th[1] * _dl + th[2] * _st
                _u -= _u.max()
                _e = np.exp(np.clip(_u, -500, 0))
                _p = _e / _e.sum()
                _tot += float((_ch * np.log(np.clip(_p, 1e-300, 1))).sum())
            return -_tot

        _res = minimize(_nll, np.array([-0.6, -0.02, 0.8]), method="L-BFGS-B",
                        options={"maxiter": 200})
        _th = _res.x
        # in-sample dest MAE / top-10 on predicted argmax ZONES
        _pzones, _ozones = [], []
        for (_ld, _dl, _st, _ch), _zl in zip(_cases, _alts_list):
            _u = _th[0] * _ld + _th[1] * _dl + _th[2] * _st
            _pzones.append(_zl[int(np.argmax(_u))])
            _ozones.append(_zl[int(np.argmax(_ch))])
        _dmae, _dtop = _dest_mae(_pzones, _ozones)
        metrics["arms"]["B3"] = {
            "ok": bool(_res.success), "ll": float(-_res.fun),
            "n_cases": int(len(_cases)), "n_skipped": int(_n_skip),
            "coef": {"c_log_dist": float(_th[0]), "c_dist": float(_th[1]),
                     "c_size": float(_th[2])},
            "dest_mae": float(_dmae), "dest_top10": float(_dtop),
            "sec": round(time.time() - t0, 1)}
    except Exception as e:
        import traceback
        metrics["arms"]["B3"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-1500:]}
    print("B3:", {k: v for k, v in metrics["arms"]["B3"].items()
                  if k != "trace"}, flush=True)

    metrics["walltime_min"] = round((time.time() - t_all) / 60.0, 2)
    with open(os.path.join(root, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("=== benchmark table ===")
    for _a in ("B0", "B1", "B2", "B3"):
        _m = metrics["arms"].get(_a, {})
        print(f"[{_a}] ok={_m.get('ok')} {str({k: _m.get(k) for k in ('ll', 'mae', 'mode_mae', 'coef', 'sec')})}")
    print(f"wrote {root}/benchmark_metrics.json")
    if not all(metrics["arms"].get(k, {}).get("ok") for k in ("B0", "B1", "B2", "B3")):
        sys.exit(1)


if __name__ == "__main__":
    main()
