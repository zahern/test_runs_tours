"""Diagnose Stage 4 SearchLibrium MNL fitness failure."""
import warnings
import sys
import io
import numpy as np
import pandas as pd
import traceback

_old = sys.stdout; sys.stdout = io.StringIO()
try:
    from SearchLibrium import MultinomialLogit
finally:
    cap = sys.stdout.getvalue(); sys.stdout = _old
    if cap.strip():
        print("[SL import stdout]", cap[:200])

print("SearchLibrium imported OK")

np.random.seed(42)
n = 100
period = np.random.randint(0, 6, n)
feat = np.random.randn(n)
data = pd.DataFrame({'period': period, 'feat': feat, 'purpose': 'work'})

from ga_external_estimators import _to_long_format, _fit_searchlib_model, _result_success

n_alts = 6
long = _to_long_format(data.copy(), choice_col='period', feat_cols=['feat'], n_alts=n_alts)
print("long shape:", long.shape)
chosen_sum = long.groupby('case_id')['chosen'].sum()
print("chosen per obs min/max:", chosen_sum.min(), chosen_sum.max())
print("n unique ids:", long['case_id'].nunique())
print("alts in long['alt_id']:", sorted(long['alt_id'].unique()))
print("alts in long['chosen'==1]['alt_id']:", sorted(long[long['chosen']==1]['alt_id'].unique()))

alts_global = np.arange(n_alts)
alts_perrow = long['alt_id'].values.astype(int)

print("\n--- Test 1: alts=global np.arange(n_alts) ---")
mnl = MultinomialLogit()
try:
    mnl.setup(
        X=long[['feat']].values.astype(float),
        y=long['chosen'].values.astype(int),
        varnames=['feat'],
        alts=alts_global,
        ids=long['case_id'].values.astype(int),
        base_alt=int(alts_global[0]),
    )
    print("setup OK")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = _fit_searchlib_model(mnl)
    print("success:", getattr(result, 'success', None), "fun:", getattr(result, 'fun', None))
    x = getattr(result, 'x', None)
    print("x:", x[:5] if x is not None and len(x) > 0 else x)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:400])
    traceback.print_exc()

print("\n--- Test 2: alts=per-row long['alt_id'] ---")
mnl2 = MultinomialLogit()
try:
    mnl2.setup(
        X=long[['feat']].values.astype(float),
        y=long['chosen'].values.astype(int),
        varnames=['feat'],
        alts=alts_perrow,
        ids=long['case_id'].values.astype(int),
        base_alt=0,
    )
    print("setup OK")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result2 = _fit_searchlib_model(mnl2)
    print("success:", getattr(result2, 'success', None), "fun:", getattr(result2, 'fun', None))
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:400])
    traceback.print_exc()

print("\n--- Test 3: check setup signature ---")
import inspect
sig = inspect.signature(MultinomialLogit.setup)
print("MultinomialLogit.setup params:", list(sig.parameters.keys()))

print("\nDone.")
