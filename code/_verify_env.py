"""Import-verify an estimation environment. Usage: python _verify_env.py"""
import importlib, sys

MODS = [
    "numpy", "scipy", "numba", "pandas", "xarray", "pyarrow",
    "jax", "jaxlib", "jaxopt",
    "larch", "sharrow.dataset", "sharrow.flows",
    "openmatrix", "tables", "pyproj", "geopandas", "statsmodels", "matplotlib",
    "metacountregressor", "SearchLibrium",
    "metacountregressor.bivariate_copula",
    "metacountregressor.multivariate_count_regressor",
]
fails = 0
for m in MODS:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "")
        print(f"OK   {m} {v}")
    except Exception as e:
        fails += 1
        print(f"FAIL {m}: {type(e).__name__}: {str(e)[:100]}")

# functional smoke: a tiny larch MNL + a metacount copula logcdf
try:
    import numpy as np, jax.numpy as jnp
    from metacountregressor.bivariate_copula import joe_logcdf
    val = float(joe_logcdf(jnp.array(0.4), jnp.array(0.6), jnp.array(2.0)))
    print(f"OK   joe_logcdf functional -> {val:.4f}")
except Exception as e:
    fails += 1
    print(f"FAIL joe_logcdf functional: {type(e).__name__}: {str(e)[:100]}")

print(f"\nRESULT: {'PASS' if fails == 0 else 'FAIL ('+str(fails)+' import errors)'}")
sys.exit(1 if fails else 0)
