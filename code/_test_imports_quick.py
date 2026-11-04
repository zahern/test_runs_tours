import sys, os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('A: starting', flush=True)
from ga_hypothesis_testing import BanditFeatureSearcher, Individual, GAResult
print('B: ga_hypothesis_testing imported', flush=True)

try:
    import larch as lx
    from larch import P, X
    print('C: larch imported OK', flush=True)
except ImportError as e:
    print(f'C: larch NOT available: {e}', flush=True)

try:
    from stage5_destination_choice import (
        _build_mode_model, _build_od_dataset, _estimate as _lx_estimate,
        MODE_MAP, DEST_UEC_FLAT,
    )
    print('D: stage5_destination_choice imported OK', flush=True)
except Exception as e:
    print(f'D: stage5 import FAILED: {type(e).__name__}: {e}', flush=True)

print('E: all imports done', flush=True)
