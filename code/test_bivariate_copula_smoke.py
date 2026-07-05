"""Quick smoke test for the bivariate copula NB module."""
import sys
sys.path.insert(0, r"C:\Users\ahernz\source\MetaCount\metacountregressor")

import numpy as np
np.random.seed(7)

from bivariate_copula import (
    frank_logpdf, normal_logpdf, kimeldorf_sampson_logpdf,
    fit_copula_bivariate_nb, fit_famoye_nb, fit_marshall_olkin_nb,
    compare_bivariate_copulas, bivariate_copula_loglik,
)
import jax
import jax.numpy as jnp

# ---- 1. Copula values are reasonable ----
for name, fn in [("frank", frank_logpdf), ("normal", normal_logpdf),
                 ("kimeldorf", kimeldorf_sampson_logpdf)]:
    u = jnp.array([0.3, 0.5, 0.7])
    v = jnp.array([0.4, 0.5, 0.6])
    rho = jnp.array(0.5)
    val = float(fn(u, v, rho).sum())
    print(f"  {name:9s} logpdf at rho=0.5: {val:.4f}")
    assert np.isfinite(val), f"{name} produced non-finite"
print("[OK] All three copula log-pdfs return finite values")

# ---- 2. Synthesise a correlated count dataset ----
n = 1500
x1 = np.column_stack([np.ones(n), np.random.randn(n)])
x2 = np.column_stack([np.ones(n), np.random.randn(n)])
off1 = np.zeros(n)
off2 = np.zeros(n)
b1 = np.array([-1.0, 0.5])
b2 = np.array([-1.5, 0.7])
mu1 = np.exp(x1 @ b1 + off1)
mu2 = np.exp(x2 @ b2 + off2)
g = np.random.gamma(shape=1.0, scale=1.0, size=n)
y1 = np.random.poisson(mu1 * g)
y2 = np.random.poisson(mu2 * g)
print(f"  synth: y1 mean={y1.mean():.3f}, y2 mean={y2.mean():.3f}, "
      f"corr={np.corrcoef(y1, y2)[0,1]:.3f}")

# ---- 3. Fit each model ----
print("\n  --- Frank ---")
f = fit_copula_bivariate_nb(y1, y2, x1, x2, off1, off2, copula="frank",
                            feature_names_1=["Intercept", "X1"],
                            feature_names_2=["Intercept", "X2"])
print(f.summary())
print(f"  converged={f.converged}  rho={f.rho:+.3f}  AIC={f.aic:.1f}")
assert f.converged
assert 0 < f.rho < 1.0, f"rho out of expected range (positive cor): got {f.rho}"

print("\n  --- Normal ---")
f = fit_copula_bivariate_nb(y1, y2, x1, x2, off1, off2, copula="normal",
                            feature_names_1=["Intercept", "X1"],
                            feature_names_2=["Intercept", "X2"])
print(f.summary())
print(f"  converged={f.converged}  rho={f.rho:+.3f}")
assert f.converged

print("\n  --- Kimeldorf-Sampson ---")
f = fit_copula_bivariate_nb(y1, y2, x1, x2, off1, off2, copula="kimeldorf",
                            feature_names_1=["Intercept", "X1"],
                            feature_names_2=["Intercept", "X2"])
print(f.summary())
print(f"  converged={f.converged}  rho={f.rho:+.3f}")
assert f.converged

print("\n  --- Famoye (no copula) ---")
f = fit_famoye_nb(y1, y2, x1, x2, off1, off2,
                   feature_names_1=["Intercept", "X1"],
                   feature_names_2=["Intercept", "X2"])
print(f.summary())

print("\n  --- Marshall-Olkin ---")
f = fit_marshall_olkin_nb(y1, y2, x1, x2, off1, off2,
                            feature_names_1=["Intercept", "X1"],
                            feature_names_2=["Intercept", "X2"])
print(f.summary())

# ---- 4. Comparison table ----
print("\n  --- compare_bivariate_copulas ---")
df, fits = compare_bivariate_copulas(y1, y2, x1, x2, off1, off2,
                                      feature_names_1=["Intercept", "X1"],
                                      feature_names_2=["Intercept", "X2"])
print(df.to_string(index=False))

# ---- 5. Gradients flow ----
grad_ll = jax.grad(lambda p: -bivariate_copula_loglik(
    jnp.asarray(y1), jnp.asarray(y2),
    jnp.asarray(x1), jnp.asarray(x2),
    jnp.asarray(off1), jnp.asarray(off2), p, copula="frank"))
g_val = np.asarray(grad_ll(jnp.zeros(7)))
assert np.all(np.isfinite(g_val))
print(f"\n[OK] Frank gradient finite, norm={np.linalg.norm(g_val):.4f}")
print("\nAll smoke tests passed.")
