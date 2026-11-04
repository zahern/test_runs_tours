"""
person_latent.py  —  shared per-person latent that links the pipeline stages
=============================================================================
Layer-1 backbone for cross-stage coordination (see the CDAP household work for
the within-household analogue). A single standardised person latent — a "travel
style" index — is attached to every person and fed into stages 1-5, so the SAME
individual's choices are correlated across activity frequency → scheduling →
duration → schedule placement → mode/destination, instead of each stage drawing
independently.

Construction (identical in estimation and prediction, so the loading each stage
estimates on HHTS transfers to the synth population unchanged):

    person_latent = z( w_attr · travel_propensity            # stable attributes
                     + w_cdap · cdap_score                    # realized DAP (M/N/H)
                     + w_act  · z(total_activity_count) )     # realized tour count

Only the components present in the frame contribute (attribute-only if CDAP /
counts aren't available yet), and the result is re-standardised to ~N(0,1) so
its scale is stable regardless of which components are in play.

`ensure_person_latent(df)` is idempotent and safe to call from any stage's
data-prep: it computes the column if missing (or `force=True`) and returns df.
Gated at the call sites by PIPELINE_PERSON_LATENT so existing runs are unchanged.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

PERSON_LATENT_COL = 'person_latent'

# CDAP pattern → mobility contribution (Mandatory travellers are most mobile,
# Home-all-day the least). Accepts the string labels used across the pipeline.
_CDAP_SCORE = {'M': 1.0, 'N': 0.0, 'H': -1.0,
               'Mandatory': 1.0, 'NonMandatory': 0.0, 'Home': -1.0}


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').astype(float)
    mu = float(s.mean())
    sd = float(s.std())
    if not np.isfinite(sd) or sd < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s.fillna(mu) - mu) / sd


def _attribute_base(persons: pd.DataFrame) -> pd.Series:
    """Attribute travel-propensity index, reusing stage 5's definition so the
    latent is consistent with the c_propensity_dist term already in the dest
    model. Falls back to a light local proxy if that import isn't available."""
    try:
        from stage5_destination_choice import _travel_propensity_score
        return pd.to_numeric(_travel_propensity_score(persons),
                             errors='coerce').fillna(0.0)
    except Exception:
        # Light fallback: standardised sum of mobility-positive attributes.
        def _c(name, default=0.0):
            return pd.to_numeric(persons.get(name, pd.Series(default, index=persons.index)),
                                 errors='coerce').fillna(default)
        base = (0.5 * (_c('WORKSTAT') > 0).astype(float)
                + 0.3 * (_c('CARLICENCE') > 0).astype(float)
                + 0.2 * (_c('vehrd') > 0).astype(float)
                + 0.1 * _zscore(_c('incp', 7.0)))
        return _zscore(base)


def _activity_count(persons: pd.DataFrame,
                    activity_cols: Optional[Sequence[str]]) -> Optional[pd.Series]:
    """Total realized activity/tour count across the stage-1 n_<purpose>_out
    columns (observed in HHTS, predicted in synth). None if none present."""
    if activity_cols is None:
        activity_cols = [c for c in persons.columns
                         if c.startswith('n_') and c.endswith('_out')]
    cols = [c for c in activity_cols if c in persons.columns]
    if not cols:
        return None
    tot = pd.to_numeric(persons[cols].apply(pd.to_numeric, errors='coerce')
                        .fillna(0.0).sum(axis=1), errors='coerce').fillna(0.0)
    return tot


def compute_person_latent(
    persons: pd.DataFrame,
    cdap_col: Optional[str] = 'CDAP',
    activity_cols: Optional[Sequence[str]] = None,
    w_attr: float = 1.0,
    w_cdap: float = 0.8,
    w_act: float = 0.6,
) -> pd.Series:
    """Return the standardised (~N(0,1)) person latent, indexed like `persons`.

    Uses only the components available in `persons`; re-standardises the blend
    so downstream loadings are on a stable scale.
    """
    total = w_attr * _attribute_base(persons).reindex(persons.index).fillna(0.0)

    if cdap_col and cdap_col in persons.columns:
        cdap_score = persons[cdap_col].map(_CDAP_SCORE)
        cdap_score = pd.to_numeric(cdap_score, errors='coerce').fillna(0.0)
        total = total + w_cdap * cdap_score

    act = _activity_count(persons, activity_cols)
    if act is not None and float(act.std()) > 1e-8:
        total = total + w_act * _zscore(act)

    return _zscore(total).rename(PERSON_LATENT_COL)


def ensure_person_latent(
    df: pd.DataFrame,
    cdap_col: Optional[str] = 'CDAP',
    activity_cols: Optional[Sequence[str]] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Attach PERSON_LATENT_COL to `df` if absent (or force). Idempotent, never
    raises — on any failure it leaves the frame unchanged (the column simply
    won't become a candidate feature). Returns `df` (same object, column added)."""
    if not force and PERSON_LATENT_COL in df.columns:
        return df
    try:
        df[PERSON_LATENT_COL] = compute_person_latent(
            df, cdap_col=cdap_col, activity_cols=activity_cols).values
    except Exception:
        pass
    return df


if __name__ == '__main__':
    # Tiny self-check: attribute-only vs enriched, both standardised.
    rng = np.random.default_rng(0)
    n = 500
    demo = pd.DataFrame({
        'WORKSTAT': rng.integers(0, 3, n),
        'CARLICENCE': rng.integers(0, 2, n),
        'vehrd': rng.integers(0, 3, n),
        'incp': rng.integers(1, 14, n),
        'CDAP': rng.choice(['M', 'N', 'H'], n),
        'n_work_out': rng.integers(0, 3, n),
        'n_shop_out': rng.integers(0, 4, n),
    })
    base = compute_person_latent(demo, cdap_col=None, activity_cols=[])
    full = compute_person_latent(demo)
    print(f"attr-only latent: mean={base.mean():.3f} std={base.std():.3f}")
    print(f"enriched latent : mean={full.mean():.3f} std={full.std():.3f}")
    print(f"corr(attr, enriched) = {np.corrcoef(base, full)[0,1]:.3f}")
