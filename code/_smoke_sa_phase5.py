"""Run _test_sa_smoke.py with GA_SMOKE_PHASE5_ONLY=1 (phases 1-4 skipped)."""
import os, runpy, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['GA_SMOKE_PHASE5_ONLY'] = '1'
# Ensure stdout/stderr can emit Unicode regardless of the terminal's default codec.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
runpy.run_path('_test_sa_smoke.py', run_name='__main__')
