"""Shared exposure-offset builder for the HB/crash count models.

Target scale: yearly rate per traffic exposure, i.e. the NB offset becomes
    log_expo = log_len + log(AADT) + log(obs_years)
           = log(seg_km * AADT * years) = log(VMT-years),
so fitted means are rates per VMT-year and segments are on a one-to-one
yearly scale regardless of how the HB panel is spread over time.

- AADT comes from the segment row (median-filled, clipped at 1.0 so the
  log is always finite; note AADT also stays a *covariate*, which then
  estimates the deviation from unit elasticity -- standard HSM practice).
- obs_years comes from the monthly HB panel (distinct ym per segment_id
  divided by 12, clipped below at 1/12). If the panel is absent, years=1
  (constant -> absorbed in the intercept, harmless).
- Falls back to plain log_len with a warning if AADT is missing.

Usage (in any paper script, run from PROJ_DIR so the sibling import works):
    from exposure import OFF, add_exposure_offsets
    df = add_exposure_offsets(df)   # panel auto-located next to the script
    m = fit_model(df, feats, y, OFF, label=label, ...)
"""
import os

import numpy as np
import pandas as pd

OFF = "log_expo"
PANEL_NAME = "near_miss_hb_panel_year.csv"


def add_exposure_offsets(df, panel_path=None, aadt_col="AADT",
                         len_col="log_len"):
    df = df.copy()
    if aadt_col not in df.columns or len_col not in df.columns:
        print(f"WARNING: exposure cols absent; offset falls back to "
              f"{len_col}", flush=True)
        df[OFF] = pd.to_numeric(df[len_col], errors="coerce").fillna(0.0) \
            if len_col in df.columns else 0.0
        return df

    aadt = pd.to_numeric(df[aadt_col], errors="coerce")
    aadt = aadt.fillna(aadt.median()).clip(lower=1.0)

    if panel_path is None:
        panel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  PANEL_NAME)
    years = None
    if os.path.isfile(panel_path):
        try:
            p = pd.read_csv(panel_path, usecols=["segment_id", "ym"],
                            low_memory=False)
            nmo = p.groupby(p["segment_id"].astype(str))["ym"].nunique()
            years = df["segment_id"].astype(str).map(nmo) / 12.0
        except Exception as e:
            print(f"WARNING: panel read failed ({e}); years=1", flush=True)
    if years is None:
        years = pd.Series(1.0, index=df.index)
    years = pd.to_numeric(years, errors="coerce").fillna(1.0 / 12).clip(
        lower=1.0 / 12)

    base = pd.to_numeric(df[len_col], errors="coerce").fillna(0.0)
    df[OFF] = base + np.log(aadt.values) + np.log(years.values)
    print(f"Exposure offset {OFF}: median={df[OFF].median():.3f}, "
          f"years median={years.median():.3f} "
          f"(log_len + log AADT + log years)", flush=True)
    return df
