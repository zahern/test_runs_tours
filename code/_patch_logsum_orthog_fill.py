"""Gate the logsum-perp-distance step to rows that actually carry a surface value.

The bug: rows whose tour/zone has no entry on the mode-logsum surface are filled
with the constant 0.0.  The orthogonalization block then residualizes EVERY row
(the comment claimed the fill "contributes nothing").  For a case whose rows are
all 0.0 the within-case demeaned value is 0, so

    _resid = _Lw - X @ beta   ->   -X @ beta

i.e. the fill is rewritten as -(beta_1*dw + beta_2*ldw), a synthetic profile
along within-case distance (+0.49 * log-distance, median R^2 0.886, measured on
the 79 affected work tours).  A distance mirror, not a dead column.

Fix: record which CA rows took a real surface value and residualize only those;
the rest keep the constant fill.

Run:  python _patch_logsum_orthog_fill.py [--check]
"""
import io
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "stage5_destination_choice.py")

OLD_LOOP = """        ca_rows = []
        for _, row in sampled_alts.iterrows():
"""

NEW_LOOP = """        ca_rows = []
        _have   = []      # per-row: did this tour/zone hit the logsum surface?
        for _, row in sampled_alts.iterrows():
"""

OLD_SAMPLE = """            r    = case_to_row.get(pid)
            c    = zone_to_col.get(zone)
            ls_val = float(ls_vals[r, c]) if (r is not None and c is not None) else 0.0
            ca_rows.append({
                'ptour_id':       pid,
                '_altid_':        zone,
                'logsum':         ls_val,
                'log_correction': float(row['log_correction']),
            })
        ca_df = pd.DataFrame(ca_rows)
"""

NEW_SAMPLE = """            r    = case_to_row.get(pid)
            c    = zone_to_col.get(zone)
            _hit = (r is not None and c is not None)
            ls_val = float(ls_vals[r, c]) if _hit else 0.0
            _have.append(_hit)
            ca_rows.append({
                'ptour_id':       pid,
                '_altid_':        zone,
                'logsum':         ls_val,
                'log_correction': float(row['log_correction']),
            })
        ca_df = pd.DataFrame(ca_rows)
        # _ls_have marks rows that carry a REAL surface value. The orthogonalization
        # below must not touch the 0.0 fills (see the note at that block); keeping
        # the mask here is what makes the two distinguishable after the fact.
        _ls_have = np.asarray(_have, dtype=bool)
"""

OLD_ALLZONES = """        ca_df = pd.DataFrame({
            'ptour_id':       ptour_rep,
            '_altid_':        altid_rep,
            'logsum':         ls_flat,
            'log_correction': 0.0,
        })
"""

NEW_ALLZONES = """        ca_df = pd.DataFrame({
            'ptour_id':       ptour_rep,
            '_altid_':        altid_rep,
            'logsum':         ls_flat,
            'log_correction': 0.0,
        })
        # Every row here is a direct surface lookup, so only non-finite entries
        # (if the surface itself has holes) are treated as unfilled.
        _ls_have = np.isfinite(np.asarray(ls_flat, dtype=float))
"""

OLD_ORTH = """    if os.environ.get('STAGE5_LOGSUM_ORTHOG', '1') == '1' and 'logsum' in ca_df.columns:
        try:
            _g   = ca_df.groupby('ptour_id', sort=False)
            _L   = ca_df['logsum'].astype(float).values
            _Lm  = _g['logsum'].transform('mean').values          # per-case mean
            _dw  = ca_df['DIST'].astype(float).values      - _g['DIST'].transform('mean').values
            _ldw = ca_df['log_DIST_1'].astype(float).values - _g['log_DIST_1'].transform('mean').values
            _Lw  = _L - _Lm                                        # within-case demeaned logsum
            _X    = np.column_stack([_dw, _ldw])
            _mask = np.isfinite(_X).all(axis=1) & np.isfinite(_Lw)
            if int(_mask.sum()) > 10 and np.linalg.matrix_rank(_X[_mask]) == 2:
                _beta, *_ = np.linalg.lstsq(_X[_mask], _Lw[_mask], rcond=None)
                _resid = _Lw - _X @ _beta                          # logsum ⊥ distance (within case)
                _new   = _Lm + _resid
                _v0, _v1 = float(np.nanvar(_L)), float(np.nanvar(_new))
                ca_df['logsum'] = _new.astype(np.float32)
                logger.info(
                    f"_dest_tree: logsum ⊥ distance orthogonalized (within-case, "
                    f"β={np.round(_beta, 4).tolist()}); total var {_v0:.4g}→{_v1:.4g}"
                )
            else:
                logger.warning("_dest_tree: logsum orthogonalization skipped "
                               "(rank-deficient distance basis or too few rows)")
"""

NEW_ORTH = """    if os.environ.get('STAGE5_LOGSUM_ORTHOG', '1') == '1' and 'logsum' in ca_df.columns:
        try:
            # Rows whose tour/zone has no entry on the mode-logsum surface were
            # filled with the constant 0.0 above. They must be EXCLUDED from the
            # residualization, not merely left zero-weighted: for a case whose
            # rows are all 0.0 the within-case demeaned value is 0, and
            # 0 - (β_dist·dw + β_ldist·ldw) is a synthetic profile along
            # within-case distance (measured +0.49·log-distance, median R² 0.886
            # on the 79 affected work tours). A distance mirror wearing the
            # accessibility column's name biases c_logsum instead of identifying
            # it. Only rows carrying a real surface value are residualized; the
            # rest keep their constant fill.
            _ok = pd.Series(np.asarray(_ls_have, dtype=bool),
                            index=ca_df.index).fillna(False).astype(bool).values
            _n_skip = int((~_ok).sum())
            if _n_skip:
                _n_skip_cases = int(ca_df.loc[~_ok, 'ptour_id'].nunique())
                logger.warning(
                    f"_dest_tree: logsum orthogonalization restricted to "
                    f"{int(_ok.sum())}/{len(ca_df)} rows with a surface value — "
                    f"{_n_skip_cases} case(s) have no logsum row and keep the "
                    f"constant fill (they are NOT distance-mirrored)")
            _fit  = ca_df.loc[_ok, ['ptour_id', 'logsum', 'DIST', 'log_DIST_1']]
            _g   = _fit.groupby('ptour_id', sort=False)
            _L   = _fit['logsum'].astype(float).values
            _Lm  = _g['logsum'].transform('mean').values          # per-case mean
            _dw  = _fit['DIST'].astype(float).values      - _g['DIST'].transform('mean').values
            _ldw = _fit['log_DIST_1'].astype(float).values - _g['log_DIST_1'].transform('mean').values
            _Lw  = _L - _Lm                                        # within-case demeaned logsum
            _X    = np.column_stack([_dw, _ldw])
            _mask = np.isfinite(_X).all(axis=1) & np.isfinite(_Lw)
            if int(_mask.sum()) > 10 and np.linalg.matrix_rank(_X[_mask]) == 2:
                _beta, *_ = np.linalg.lstsq(_X[_mask], _Lw[_mask], rcond=None)
                _resid = _Lw - _X @ _beta                          # logsum ⊥ distance (within case)
                _new   = _Lm + _resid
                _v0, _v1 = float(np.nanvar(_L)), float(np.nanvar(_new))
                _col = ca_df['logsum'].to_numpy(dtype=np.float32, copy=True)
                _col[_ok] = _new.astype(np.float32)
                ca_df['logsum'] = _col
                logger.info(
                    f"_dest_tree: logsum ⊥ distance orthogonalized (within-case, "
                    f"β={np.round(_beta, 4).tolist()}); total var {_v0:.4g}→{_v1:.4g}"
                    f" on {int(_ok.sum())} surface rows"
                )
            else:
                logger.warning("_dest_tree: logsum orthogonalization skipped "
                               "(rank-deficient distance basis or too few rows)")
"""


def main():
    data = open(PATH, "rb").read()
    text = data.decode("utf-8")
    crlf = "\r\n" in text

    def conv(s):
        return s.replace("\n", "\r\n") if crlf else s

    pairs = [("loop", OLD_LOOP, NEW_LOOP),
             ("sample", OLD_SAMPLE, NEW_SAMPLE),
             ("allzones", OLD_ALLZONES, NEW_ALLZONES),
             ("orthog", OLD_ORTH, NEW_ORTH)]

    if "--check" in sys.argv:
        for name, old, _ in pairs:
            print(f"{name:9s} present={text.count(conv(old))}")
        return 0

    for name, old, new in pairs:
        o, n = conv(old), conv(new)
        c = text.count(o)
        if c != 1:
            print(f"FAIL {name}: found {c} matches, expected 1")
            return 1
        text = text.replace(o, n)
        print(f"patched {name}")

    open(PATH, "wb").write(text.encode("utf-8"))
    print(f"wrote {PATH} (crlf={crlf})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
