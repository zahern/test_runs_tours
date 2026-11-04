"""Quick SearchLibrium setup test - no fitting."""
import sys, io, numpy as np, pandas as pd, inspect

_old = sys.stdout; sys.stdout = io.StringIO()
try:
    from SearchLibrium import MultinomialLogit
finally:
    sys.stdout = _old
print("SL import OK")

sig = inspect.signature(MultinomialLogit.setup)
print("setup params:", list(sig.parameters.keys()))

# minimal data
n = 20
np.random.seed(0)
period = np.random.randint(0, 6, n)
feat = np.random.randn(n)
data = pd.DataFrame({'period': period, 'feat': feat})

from ga_external_estimators import _to_long_format
long = _to_long_format(data, choice_col='period', feat_cols=['feat'], n_alts=6)
print("long shape:", long.shape)
print("long head:")
print(long.head(12).to_string())

alts_global = np.arange(6)
alts_perrow = long['alt_id'].values.astype(int)

mnl = MultinomialLogit()
print("\nTrying setup with global alts array...")
try:
    mnl.setup(
        X=long[['feat']].values.astype(float),
        y=long['chosen'].values.astype(int),
        varnames=['feat'],
        alts=alts_global,
        ids=long['case_id'].values.astype(int),
        base_alt=0,
    )
    print("OK - X shape:", mnl.X.shape if hasattr(mnl,'X') else 'N/A')
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])

mnl2 = MultinomialLogit()
print("\nTrying setup with per-row alts array...")
try:
    mnl2.setup(
        X=long[['feat']].values.astype(float),
        y=long['chosen'].values.astype(int),
        varnames=['feat'],
        alts=alts_perrow,
        ids=long['case_id'].values.astype(int),
        base_alt=0,
    )
    print("OK - X shape:", mnl2.X.shape if hasattr(mnl2,'X') else 'N/A')
    print("nb_samples:", getattr(mnl2, 'nb_samples', 'N/A'))
    print("nb_alts:", getattr(mnl2, 'nb_alts', 'N/A'))
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])

print("Done.")
