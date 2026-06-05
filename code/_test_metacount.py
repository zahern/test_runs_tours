"""
Quick smoke test for MetaCount GA fitness functions.
Run with the project venv:  .venv/Scripts/python _test_metacount.py
Or in VS Code: select .venv interpreter (Ctrl+Shift+P > Python: Select Interpreter)

SECTIONS:
  1. Fallback fitness functions (OLS/lognormal) - fast, no MetaCount, MUST PASS
  2+3. MetaCount import, guards, live fit - slow (JAX hangs on Windows)
       Enabled only when MC_SKIP_FIT=0 (default=1 = skip on Windows)

Usage:
  python _test_metacount.py             # section 1 only (default)
  set MC_SKIP_FIT=0 && python ...       # all sections (HPC / Linux)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must be set BEFORE any jax import — JAX reads this during C-extension init.
# ga_external_estimators also sets this, but belt-and-suspenders here.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import pandas as pd

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'obs_id':       range(n),
    'age':          np.random.randint(18, 70, n).astype(float),
    'incp':         np.random.randn(n),
    'vehrd':        np.random.randint(0, 3, n).astype(float),
    'nprd':         np.random.randint(1, 5, n).astype(float),
    'ptype':        np.random.choice(['FT worker', 'PT worker', 'Student'], n),
    'n_work_out':   np.random.poisson(1.2, n),
    'start_of_day': np.random.normal(7.5, 1.5, n).clip(4, 12),
    'B':            np.random.lognormal(4.5, 0.8, n),
    'duration':     np.random.lognormal(4.0, 0.7, n).clip(5, None),
    'purpose':      np.random.choice(['work', 'edu', 'social'], n),
})
feat_names = ['age', 'incp', 'vehrd', 'nprd']
genes_all  = np.ones(len(feat_names), dtype=int)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
pass_c = fail_c = 0

def check(label, ok, required=True):
    global pass_c, fail_c
    tag = PASS if ok else (WARN if not required else FAIL)
    print(f"  [{tag}] {label}")
    if ok:
        pass_c += 1
    elif required:
        fail_c += 1

# ======================================================================
# SECTION 1 - Fallback fitness functions (pure Python, no MetaCount)
# ======================================================================
print("\n-- Section 1: Fallback fitness functions (fast, required) --")

from ga_hypothesis_testing import (
    _stage1_fitness_aggregate,
    _stage2_fitness_aggregate,
    _stage3_fitness_aggregate,
)

ptypes = df['ptype'].unique().tolist()
s = _stage1_fitness_aggregate(genes_all, df, feat_names, ['work'], ptypes)
check(f"stage1 ordered-logit BIC/n = {s:.4f}", s > -1e8)

s = _stage2_fitness_aggregate(genes_all, df, feat_names, ['B'])
check(f"stage2 OLS BIC/n (B) = {s:.4f}", s > -1e8)

s = _stage2_fitness_aggregate(genes_all, df, feat_names, ['start_of_day'])
check(f"stage2 OLS BIC/n (start_of_day) = {s:.4f}", s > -1e8)

try:
    s = _stage3_fitness_aggregate(genes_all, df, feat_names, ['work'], 'purpose')
    check(f"stage3 lognormal BIC/n (work) = {s:.4f}", s > -1e8)
except Exception as e:
    check(f"stage3 lognormal (SKIP — {e})", True)  # not required for GA fix

print(f"\n  Section 1: {pass_c} passed, {fail_c} failed")
if fail_c:
    print("FALLBACK FUNCTIONS BROKEN -- fix before running GA")
    sys.exit(1)

# ======================================================================
# SECTION 2+3 - MetaCount import, guards, and live fit
# ======================================================================
skip_mc = os.environ.get('MC_SKIP_FIT', '0') == '1'
if skip_mc:
    print("\n-- Sections 2+3: MetaCount SKIPPED (MC_SKIP_FIT=1) --")
else:
    print("\n-- Section 2: MetaCount import + guards --")
    from ga_external_estimators import _METACOUNT_OK, _METACOUNT_IMPORT_ERROR, _metacount_fit

    print(f"  MetaCount available: {_METACOUNT_OK}")
    if not _METACOUNT_OK:
        print(f"  Import error: {_METACOUNT_IMPORT_ERROR}")
        check("MetaCount import", False, required=False)
    else:
        import metacountregressor
        print(f"  Version: {getattr(metacountregressor, '__version__', 'unknown')}")
        check("MetaCount import", True)

        s = _metacount_fit(df[['obs_id','n_work_out']], 'obs_id', 'n_work_out', [], 'nb2')
        check(f"guard empty vars -> sentinel ({s:.0f})", s <= -1e8)

        s = _metacount_fit(df[['obs_id','n_work_out','age']], 'obs_id', 'n_work_out', ['missing'], 'nb2')
        check(f"guard missing col -> sentinel ({s:.0f})", s <= -1e8)

        s = _metacount_fit(df.head(5), 'obs_id', 'n_work_out', ['age','incp'], 'nb2')
        check(f"guard n<30 -> sentinel ({s:.0f})", s <= -1e8)

        print("\n-- Section 3: MetaCount live fit --")
        from ga_external_estimators import (
            stage1_fitness_metacount, stage2_fitness_metacount, stage3_fitness_metacount
        )
        df['duration_min'] = df['duration']   # alias used by stage3 MC path
        s = stage1_fitness_metacount(genes_all, df, feat_names, ['work'])
        check(f"MC NB2 stage1 work = {s:.4f}", s > -1e8)
        s = stage2_fitness_metacount(genes_all, df, feat_names, ['B'])
        check(f"MC duration stage2 B = {s:.4f}", s > -1e8)
        s = stage3_fitness_metacount(genes_all, df, feat_names, ['work'], 'purpose', 'duration_min')
        check(f"MC lognormal stage3 work = {s:.4f}", s > -1e8)

print(f"\n{'='*55}")
print(f"Required: {pass_c} passed, {fail_c} failed")
sys.exit(0 if fail_c == 0 else 1)
