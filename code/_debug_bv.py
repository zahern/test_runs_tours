"""Debug the bivariate copula likelihood NaN."""
import sys
sys.path.insert(0, r"C:\Users\ahernz\source\MetaCount\metacountregressor")

import numpy as np
np.random.seed(7)

from bivariate_copula import (
    frank_logpdf, normal_logpdf, kimeldorf_sampson_logpdf,
    nb_log_pmf, nb_log_cdf,
    fit_univariate_nb,
)
import jax
import jax.numpy as jnp

# Synth data
n = 1500
x1 = np.column_stack([np.ones(n), np.random.randn(n)])
x2 = np.column_stack([np.ones(n), np.random.randn(n)])
off = np.zeros(n)
b1 = np.array([-1.0, 0.5])
b2 = np.array([-1.5, 0.7])
mu1 = np.exp(x1 @ b1 + off)
mu2 = np.exp(x2 @ b2 + off)
g = np.random.gamma(shape=1.0, scale=1.0, size=n)
y1 = np.random.poisson(mu1 * g)
y2 = np.random.poisson(mu2 * g)

print("Univariate fit on y1:")
u1 = fit_univariate_nb(y1, x1, off)
print(f"  alpha_1 = {u1['alpha']:.4f}, beta = {u1['coef']}")
print("Univariate fit on y2:")
u2 = fit_univariate_nb(y2, x2, off)
print(f"  alpha_2 = {u2['alpha']:.4f}, beta = {u2['coef']}")

# Re-evaluate the bivariate loglik at various rho
from bivariate_copula import bivariate_copula_loglik
b1_jax = jnp.zeros(2)
b2_jax = jnp.zeros(2)
alpha_1 = jnp.asarray(u1['alpha'])
alpha_2 = jnp.asarray(u2['alpha'])

for rho in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]:
    params = jnp.array([0.0, 0.0, 0.0, 0.0, rho])
    ll = -float(bivariate_copula_loglik(
        jnp.asarray(y1), jnp.asarray(y2),
        jnp.asarray(x1), jnp.asarray(x2),
        jnp.asarray(off), jnp.asarray(off),
        params, copula='frank',
        alpha_1=alpha_1, alpha_2=alpha_2,
    ))
    print(f"  rho={rho:+.2f}  -LL={ll:.4f}  (nan={np.isnan(ll)})")

# Now try with proper warm-start betas
for rho in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    params = jnp.array([np.log(mu1.mean()), 0.5,
                        np.log(mu2.mean()), 0.7, rho])
    ll = -float(bivariate_copula_loglik(
        jnp.asarray(y1), jnp.asarray(y2),
        jnp.asarray(x1), jnp.asarray(x2),
        jnp.asarray(off), jnp.asarray(off),
        params, copula='frank',
        alpha_1=alpha_1, alpha_2=alpha_2,
    ))
    print(f"  WARMSTART rho={rho:+.2f}  -LL={ll:.4f}  (nan={np.isnan(ll)})")
