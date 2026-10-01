"""
schedule_optimizer.py — optional OR-Tools CP-SAT repair pass for Stage 4.
==========================================================================
Stage 4 (stage4_schedule_placement.predict_schedule) places greedily:
sequential period-MNL + uniform in-period sample, one-way link
schedule → dest → mode. That can leave overlaps, window overruns and
schedules that ignore realized travel time.

This module adds an optional per-person CP-SAT repair:

* keeps the MNL order/utilities (period choice stays the demand signal),
* enforces global feasibility: day window, durations, no-overlap +
  travel buffers, mandatory anchors,
* optionally pulls realized travel times (stage5 dest + skims / stage6
  mode) back into the schedule — the backward linkage.

Gated: everything is a no-op unless SCHEDULE_OPTIMIZE=1 AND ortools is
importable. HPC never breaks when ortools is missing.

Env:
    SCHEDULE_OPTIMIZE=1   enable (default 0)
    SCHEDULE_OPT_MAX_PERSONS  cap persons repaired per call (default all)
    SCHEDULE_OPT_TIME_S       solver seconds per person (default 2.0)
    SCHEDULE_OPT_SLOT         minutes per slot (default 5)
    SCHEDULE_OPT_GAP          travel buffer minutes (default 10)

Usage (pass 1, pre-dest — inside pipeline_logger.predict_stage4):
    from schedule_optimizer import maybe_repair_schedules
    result = maybe_repair_schedules(result)

Usage (pass 2, post dest/mode — caller supplies realized travel minutes
in a per-activity column, e.g. 'travel_min'):
    result = maybe_repair_schedules(result, travel_col='travel_min')
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from ortools.sat.python import cp_model
    ORTOOLS_OK = True
except Exception:  # pragma: no cover — HPC without ortools stays green
    cp_model = None
    ORTOOLS_OK = False


def _env_flag(name, default='0'):
    return os.environ.get(name, default).strip().lower() not in ('', '0', 'no', 'off', 'false')


def _solve_one(starts, ends, durs, period_lo, period_hi, sod, eod,
               gap, utils, time_s, slot):
    """Solve one person's schedule. All inputs in slot units (ints) except utils."""
    n = len(starts)
    model = cp_model.CpModel()
    s = [model.NewIntVar(sod, eod, f's_{i}') for i in range(n)]
    e = [model.NewIntVar(sod, eod, f'e_{i}') for i in range(n)]
    for i in range(n):
        model.Add(e[i] == s[i] + int(durs[i]))
        # anchor inside the MNL-chosen period window (clipped to day)
        lo = max(int(period_lo[i]), sod)
        hi = min(int(period_hi[i]) - int(durs[i]), eod - int(durs[i]))
        if hi >= lo:
            model.Add(s[i] >= lo)
            model.Add(s[i] <= hi)
        else:  # infeasible period window — fit anywhere in day
            model.Add(s[i] >= sod)
            model.Add(s[i] <= max(sod, eod - int(durs[i])))
        model.Add(s[i] >= sod)
        model.Add(e[i] <= eod)
        if i > 0:
            model.Add(s[i] >= e[i - 1] + int(gap))
    # objective: stay near MNL-desired starts (utils = -|slot - desired| scaled
    # by caller) — linear surrogate for period utility + overlap/travel penalty.
    # Maximize sum(utils[i] at chosen s[i]): piecewise via boolean expansion is
    # overkill for a repair pass; instead minimise L1 distance to greedy starts.
    obj = []
    for i in range(n):
        d = model.NewIntVar(-1440, 1440, f'd_{i}')
        model.Add(d == s[i] - int(starts[i]))
        ad = model.NewIntVar(0, 1440, f'ad_{i}')
        model.AddAbsEquality(ad, d)
        obj.append(ad * -1)
    model.Maximize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_s)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [int(solver.Value(v)) for v in s]
    return None


def repair_schedules(df, travel_col=None, time_s=None, slot_min=None,
                     gap_min=None, max_persons=None):
    """Repair per-person feasibility with CP-SAT. Returns a new DataFrame.

    Keeps MNL order; adjusts STARTIME/ENDTIME to remove overlaps and window
    overruns. If travel_col names a per-activity realized-travel-minutes
    column (pass 2), it is added as inter-activity buffer.
    Never raises — returns `df` unchanged on any failure / missing ortools.
    """
    if not ORTOOLS_OK:
        logger.warning('schedule_optimizer: ortools missing — skipping repair')
        return df
    if df is None or len(df) == 0:
        return df
    slot = int(slot_min or os.environ.get('SCHEDULE_OPT_SLOT', 5))
    gap_default = int(gap_min or os.environ.get('SCHEDULE_OPT_GAP', 10))
    tlim = float(time_s or os.environ.get('SCHEDULE_OPT_TIME_S', 2.0))
    cap = max_persons or os.environ.get('SCHEDULE_OPT_MAX_PERSONS', '')
    cap = int(cap) if str(cap).strip() else None

    work = df.copy()
    if 'person_id' not in work.columns and 'PERSID' in work.columns:
        work = work.rename(columns={'PERSID': 'person_id'})
    need = {'person_id', 'STARTIME', 'ENDTIME', 'predicted_duration_min',
            'start_of_day', 'end_of_day'}
    if not need.issubset(work.columns):
        logger.warning('schedule_optimizer: missing %s — skipping',
                       sorted(need - set(work.columns)))
        return df

    from stage4_schedule_placement import PERIOD_STARTS, PERIOD_ENDS  # noqa: E402

    def _period_bounds(tp):
        try:
            i = list(['early', 'am_peak', 'midday', 'pm_peak', 'evening', 'late']).index(str(tp))
            return int(PERIOD_STARTS[i] // slot), int(PERIOD_ENDS[i] // slot)
        except Exception:
            return 0, 288

    out_starts = work['STARTIME'].to_numpy(dtype=float).copy()
    out_ends = work['ENDTIME'].to_numpy(dtype=float).copy()
    travel = None
    if travel_col and travel_col in work.columns:
        travel = pd.to_numeric(work[travel_col], errors='coerce').fillna(0).clip(lower=0).to_numpy(dtype=float)

    pids = work['person_id'].to_numpy()
    order = np.argsort(pids, kind='stable')
    uniq, idx0 = np.unique(pids[order], return_index=True)
    if cap:
        uniq = uniq[:cap]
    n_fixed = 0
    for pi, pid in enumerate(uniq):
        lo = idx0[pi]
        hi = idx0[pi + 1] if pi + 1 < len(idx0) else len(order)
        # cap applies to persons considered; still need rows beyond cap untouched
        if cap and pi >= cap:
            break
        rows = order[lo:hi]
        # keep MNL execution order (already sorted by planning_order upstream)
        seq = rows[np.argsort(work['STARTIME'].to_numpy()[rows], kind='stable')]
        n = len(seq)
        if n < 2:
            continue
        # quick check: already feasible → skip solver
        st = work['STARTIME'].to_numpy(dtype=float)[seq]
        en = work['ENDTIME'].to_numpy(dtype=float)[seq]
        sod = float(work['start_of_day'].to_numpy(dtype=float)[seq[0]])
        eod = float(work['end_of_day'].to_numpy(dtype=float)[seq[0]])
        dur = np.maximum(work['predicted_duration_min'].to_numpy(dtype=float)[seq], 1.0)
        buf = np.full(n, gap_default, dtype=float)
        if travel is not None:
            buf[1:] = np.maximum(buf[1:], travel[seq[1:]])
        feasible = bool(np.all(st[:-1] + 0 <= st[1:] - 0) and np.all(en <= eod) and np.all(st >= sod))
        # overlap test with buffers
        overlap = bool(np.any(st[1:] < en[:-1] + buf[1:]))
        if feasible and not overlap:
            continue
        bounds = [_period_bounds(work['time_period'].to_numpy()[j]) if 'time_period' in work.columns else (0, 288)
                  for j in seq]
        plo = np.array([b[0] for b in bounds])
        phi = np.array([b[1] for b in bounds])
        sol = _solve_one((st // slot).astype(int), (en // slot).astype(int),
                         np.maximum((dur // slot).astype(int), 1),
                         plo, phi, int(sod // slot), int(eod // slot),
                         int(np.median(buf) // slot), None, tlim, slot)
        if sol is None:
            continue
        new_st = np.array(sol) * slot
        new_en = new_st + dur
        out_starts[seq] = new_st
        out_ends[seq] = new_en
        n_fixed += 1

    work['STARTIME'] = np.round(out_starts).astype(int)
    work['ENDTIME'] = np.round(out_ends).astype(int)
    # Joint re-sync: co-travellers repaired per-person may split — restore one
    # shared start per joint_tour_id (group median). Safe no-op untagged.
    try:
        if 'joint_tour_id' in work.columns:
            _jt = work['joint_tour_id'].astype(str)
            _m = _jt.ne('') & _jt.ne('nan') & _jt.ne('None')
            if _m.any():
                _med = work.loc[_m].groupby(_jt[_m])['STARTIME'].transform('median')
                _dur = (work.loc[_m, 'ENDTIME'] - work.loc[_m, 'STARTIME']).clip(lower=1)
                work.loc[_m, 'STARTIME'] = _med.astype(int).values
                work.loc[_m, 'ENDTIME'] = (_med.astype(int) + _dur.astype(int)).values
    except Exception as e:
        logger.warning('schedule_optimizer joint re-sync skipped: %s', e)
    work['sched_opt'] = 0
    # mark repaired persons (cheap: anyone whose start moved)
    moved = np.abs(work['STARTIME'].to_numpy() - df['STARTIME'].to_numpy()) > 0.5
    work.loc[moved, 'sched_opt'] = 1
    logger.info('schedule_optimizer: repaired %d/%d persons (travel_col=%s)',
                n_fixed, len(uniq), travel_col)
    return work


def maybe_repair_schedules(df, travel_col=None):
    """Env-gated wrapper: no-op unless SCHEDULE_OPTIMIZE=1."""
    if not _env_flag('SCHEDULE_OPTIMIZE', '0'):
        return df
    try:
        return repair_schedules(df, travel_col=travel_col)
    except Exception as e:  # never break the pipeline
        logger.warning('schedule_optimizer failed (non-fatal): %s', e)
        return df
