#!/usr/bin/env python
"""
micro_location_placement.py — POLARIS-style sub-zonal micro-location assignment.
================================================================================
Keeps the SA2-level demand models (destination choice stays at the zone) but
places every activity at a specific MICRO-location inside its chosen SA2, in
accordance with land use — so the simulation has detailed x/y endpoints.

POLARIS rule: land_use flags gate which activities a location can host
    home  activities → is_home       locations (residential)
    work  activities → is_work        locations (non-residential)
    edu   activities → is_school      locations (educational)
    discretionary    → is_discretionary locations (shop/eat/social/…)
Within the eligible pool a location is drawn with probability ∝ lu_area (size),
i.e. a size-term location choice one level below the zone.

Efficiency: locations are pre-indexed once into per-(zone, flag) pools with a
cumulative-weight vector, so each activity is placed by a single vectorised
searchsorted over uniforms — O(N log P) for N activities, not a per-activity MNL.
ANCHORS (home / usual workplace / usual school) are drawn ONCE per person and
reused across all their tours (a person has one home, one workplace); only
discretionary activities are resampled per visit.

API
    loc = MicroLocator('saved_data/location_table.csv')
    persons = loc.assign_anchor_locations(persons)     # home/work/school loc + x/y
    acts    = loc.place_activities(acts, persons)       # location_id, loc_x, loc_y
"""
import os
import logging

import numpy as np
import pandas as pd

log = logging.getLogger('micro_loc')

# activity purpose → (flag column, optional preferred land_use subset).
# The preferred subset narrows discretionary purposes to the most relevant land
# use (shopping→retail); it falls back to the whole flag pool if the zone has
# none of the preferred type.
_PURPOSE_RULE = {
    'home':     ('is_home', None),
    'work':     ('is_work', None),
    'edu':      ('is_school', None),
    'shopping': ('is_discretionary', {'retail'}),
    'eat':      ('is_discretionary', {'food'}),
    'social':   ('is_discretionary', {'recreation', 'food', 'civic'}),
    'othdiscr': ('is_discretionary', {'recreation', 'retail', 'civic'}),
    'othmaint': ('is_discretionary', {'retail', 'health', 'civic', 'office'}),
    'escort':   ('is_discretionary', {'education', 'retail', 'health'}),
}
_FLAG_COLS = ('is_home', 'is_work', 'is_school', 'is_discretionary')


class MicroLocator:
    def __init__(self, location_table='saved_data/location_table.csv'):
        if isinstance(location_table, pd.DataFrame):
            loc = location_table.copy()
        else:
            loc = pd.read_csv(location_table)
        loc = loc.reset_index(drop=True)
        for _c in _FLAG_COLS:
            if _c not in loc.columns:
                loc[_c] = 0
        self.loc = loc
        self.loc_id = loc['location'].to_numpy()
        self.x = loc['x'].to_numpy(float)
        self.y = loc['y'].to_numpy(float)
        self.area = np.clip(pd.to_numeric(loc['lu_area'], errors='coerce').fillna(1.0).to_numpy(float), 1e-6, None)
        self.zone = pd.to_numeric(loc['zone'], errors='coerce').fillna(-1).astype('int64').to_numpy()
        self.land_use = loc['land_use'].astype(str).to_numpy()
        self.flags = {c: loc[c].to_numpy().astype(bool) for c in _FLAG_COLS}
        # zone → row indices (built once)
        self._zone_rows = {int(z): np.where(self.zone == int(z))[0]
                           for z in np.unique(self.zone) if z >= 0}
        self._pool_cache = {}
        log.info("MicroLocator: %d locations across %d zones", len(loc), len(self._zone_rows))

    def _pool(self, zone, flag, land_use_subset=None):
        """(row_idx, cum_weights) for eligible locations in a zone. Cached."""
        key = (int(zone), flag, tuple(sorted(land_use_subset)) if land_use_subset else None)
        if key in self._pool_cache:
            return self._pool_cache[key]
        rows = self._zone_rows.get(int(zone))
        if rows is None or len(rows) == 0:
            self._pool_cache[key] = (None, None)
            return None, None
        m = self.flags[flag][rows]
        idx = rows[m]
        if land_use_subset and len(idx):
            sub = np.isin(self.land_use[idx], list(land_use_subset))
            if sub.any():
                idx = idx[sub]
        if len(idx) == 0:                          # fall back to any location in zone
            idx = rows
        w = self.area[idx]
        cum = np.cumsum(w)
        cum = cum / cum[-1]
        self._pool_cache[key] = (idx, cum)
        return idx, cum

    def _sample(self, zones, flag, rng, land_use_subset=None):
        """Vectorised: one location row-index per zone entry (∝ lu_area)."""
        zones = np.asarray(zones)
        out = np.full(len(zones), -1, dtype=int)
        for z in pd.unique(zones):
            if z is None or (isinstance(z, float) and not np.isfinite(z)):
                continue
            sel_mask = zones == z
            k = int(sel_mask.sum())
            idx, cum = self._pool(int(z), flag, land_use_subset)
            if idx is None:
                continue
            u = rng.random(k)
            out[sel_mask] = idx[np.searchsorted(cum, u, side='left').clip(0, len(idx) - 1)]
        return out

    # ── anchors: one home / workplace / school per person ────────────────────
    def assign_anchor_locations(self, persons: pd.DataFrame,
                                home_col='home_zone', work_col='WORK_TAZ',
                                school_col='SCHOOL_TAZ', seed=42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        p = persons.copy()
        _plans = [('home_loc', home_col, 'is_home', None),
                  ('work_loc', work_col, 'is_work', None),
                  ('school_loc', school_col, 'is_school', None)]
        for out_col, zcol, flag, _ in _plans:
            if zcol not in p.columns:
                p[out_col] = -1
                continue
            z = pd.to_numeric(p[zcol], errors='coerce').fillna(-1).astype('int64').to_numpy()
            ridx = self._sample(z, flag, rng)
            p[out_col] = np.where(ridx >= 0, self.loc_id[ridx.clip(0)], -1)
            p[out_col + '_x'] = np.where(ridx >= 0, self.x[ridx.clip(0)], np.nan)
            p[out_col + '_y'] = np.where(ridx >= 0, self.y[ridx.clip(0)], np.nan)
        return p

    # ── activities: anchor purposes reuse the person's anchor; others sampled ─
    def place_activities(self, activities: pd.DataFrame, persons: pd.DataFrame = None,
                         purpose_col='purpose', zone_col='DTAZ',
                         person_col='person_id', seed=7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        a = activities.copy()
        n = len(a)
        loc_id = np.full(n, -1, dtype=np.int64)
        # anchor lookup (home/work/school reuse the person's single location)
        anchor_map = {'home': 'home_loc', 'work': 'work_loc', 'edu': 'school_loc'}
        if persons is not None and person_col in a.columns and person_col in persons.columns:
            _pl = persons.set_index(person_col)
            for pur, acol in anchor_map.items():
                if acol not in _pl.columns:
                    continue
                m = (a[purpose_col] == pur).to_numpy()
                if m.any():
                    loc_id[m] = pd.to_numeric(
                        a.loc[m, person_col].map(_pl[acol]), errors='coerce').fillna(-1).to_numpy()
        # discretionary + any anchor that didn't resolve → sample in dest zone
        z = pd.to_numeric(a[zone_col], errors='coerce').fillna(-1).astype('int64').to_numpy()
        pur = a[purpose_col].astype(str).to_numpy()
        need = loc_id < 0
        for _pur in pd.unique(pur[need]):
            flag, subset = _PURPOSE_RULE.get(_pur, ('is_discretionary', None))
            m = need & (pur == _pur)
            ridx = self._sample(z[m], flag, rng, subset)
            _lid = np.where(ridx >= 0, self.loc_id[ridx.clip(0)], -1)
            loc_id[m] = _lid
        a['location_id'] = loc_id
        # attach x/y for the resolved locations
        _pos = pd.Series(np.arange(len(self.loc_id)), index=self.loc_id)
        _ri = _pos.reindex(loc_id).to_numpy()
        _ok = np.isfinite(_ri)
        a['loc_x'] = np.where(_ok, self.x[np.where(_ok, _ri, 0).astype(int)], np.nan)
        a['loc_y'] = np.where(_ok, self.y[np.where(_ok, _ri, 0).astype(int)], np.nan)
        _n_placed = int((loc_id > 0).sum())
        log.info("placed %d/%d activities at micro-locations (%.1f%%)",
                 _n_placed, n, 100.0 * _n_placed / max(n, 1))
        return a


def run_micro_placement(persons: pd.DataFrame, activities: pd.DataFrame,
                        location_table='saved_data/location_table.csv',
                        out_dir=None) -> tuple:
    """Convenience: assign anchors + place activities; optionally save."""
    loc = MicroLocator(location_table)
    persons = loc.assign_anchor_locations(persons)
    activities = loc.place_activities(activities, persons)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        activities.to_csv(os.path.join(out_dir, 'activities_micro_located.csv'), index=False)
    return persons, activities
