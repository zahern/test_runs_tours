"""
Test implementation of JAX nested logit log-likelihood.

This is a prototype to verify JAX nested logit gradients work correctly
before integrating into Larch.
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Enable 64-bit precision
jax.config.update('jax_enable_x64', True)

# ── Nested Logit Structure ──────────────────────────────────────────────
# Root (μ=1 fixed)
#   ├─ Motorised (μ_Motor) 
#   │   ├── DA (alt 0)
#   │   ├── SR (alt 1)  
#   │   └── Transit (alt 2)
#   └─ NonMotorised (μ_NonMotor)
#       ├── Walk (alt 3)
#       └─ Bike (alt 4)

NESTS = {
    'Motorised': [0, 1, 2],    # DA, SR, Transit
    'NonMotorised': [3, 4],    # Walk, Bike
}
NEST_NAMES = ['Motorised', 'NonMotorised']
ALT_TO_NEST = {alt: nest for nest, alts in NESTS.items() for alt in alts}
N_NESTS = len(NEST_NAMES)
N_ALTS = 5

def jax_nested_loglike(params, X, y, nest_scales=None):
    """
    JAX nested logit log-likelihood.
    
    Parameters
    ----------
    params : array
        Coefficients for utility V = X @ params
    X : array (n_cases, n_alts, n_params)
        Utility covariates
    y : array (n_cases,)
        Chosen alternative indices (0-4)
    nest_scales : dict or None
        Fixed nest scales (μ). If None, estimated from params.
        
    Returns
    -------
    Total log-likelihood (scalar)
    """
    # If nest_scales provided, they're fixed; otherwise extract from params
    if nest_scales is None:
        # Last N_NESTS-1 params are nest scales (Mu_Root fixed=1)
        beta = params[:-N_NESTS+1]
        mu_motor = params[-2]
        mu_nonmotor = params[-1]
        mu = jnp.array([1.0, mu_motor, mu_nonmotor])  # Root, Motor, NonMotor
    else:
        beta = params
        mu_motor = nest_scales.get('Motorised', 0.8)
        mu_nonmotor = nest_scales.get('NonMotorised', 0.8)
        mu = jnp.array([1.0, mu_motor, mu_nonmotor])
    
    # Utility: V = X @ beta  (n_cases, n_alts)
    V = jnp.einsum('cai,i->ca', X, beta)
    
    # Inclusive values (logsums) per nest
    # IV_nest = log(sum_{j in nest} exp(V_j))
    iv_motor = jax.nn.logsumexp(V[:, NESTS['Motorised']], axis=1)
    iv_nonmotor = jax.nn.logsumexp(V[:, NESTS['NonMotorised']], axis=1)
    IV = jnp.stack([iv_motor, iv_nonmotor], axis=1)  # (n_cases, 2)
    
    # Nest probabilities (conditional on root)
    # log P(nest) = mu_nest * IV_nest - logsumexp(mu * IV)
    mu = jnp.array([mu_motor, mu_nonmotor])
    log_nest_num = mu * IV
    log_nest_den = jax.nn.logsumexp(mu * IV, axis=1)
    log_nest_prob = log_nest_num - log_nest_den[:, None]  # (n_cases, 2)
    
    # Alternative probabilities within nest
    # log P(alt|nest) = V_alt - logsumexp(V_nest)
    V_motor = V[:, NESTS['Motorised']]
    V_nonmotor = V[:, NESTS['NonMotorised']]
    log_alt_given_motor = V_motor - jax.nn.logsumexp(V_motor, axis=1)[:, None]
    log_alt_given_nonmotor = V_nonmotor - jax.nn.logsumexp(V_nonmotor, axis=1)[:, None]
    
    # Combine: log P(alt) = log P(nest) + log P(alt|nest)
    log_prob = jnp.zeros_like(V)
    log_prob = log_prob.at[:, NESTS['Motorised']].set(log_nest_prob[:, 0:1] + log_alt_given_motor)
    log_prob = log_prob.at[:, NESTS['NonMotorised']].set(log_nest_prob[:, 1:2] + log_alt_given_nonmotor)
    
    # Log-likelihood for chosen alternatives
    chosen_logprob = log_prob[jnp.arange(len(y)), y]
    return jnp.sum(chosen_logprob)

# ── Test with synthetic data ─────────────────────────────────────────────
def test_jax_nested_logit():
    np.random.seed(42)
    n_cases = 200
    
    # Generate synthetic data with known parameters
    true_beta = np.array([-0.5, 0.3, -0.2, 1.0, -0.4])  # 5 params
    true_mu_motor = 0.7
    true_mu_nonmotor = 0.6
    
    # Covariates: (n_cases, n_alts, n_params)
    X = np.random.randn(n_cases, N_ALTS, len(true_beta))
    # Add some structure: alt-specific constants
    X[:, :, 0] += np.array([1.0, 0.5, 0.0, -0.5, -1.0])  # ASCs
    
    # Compute true utilities
    V_true = np.einsum('cai,i->ca', X, true_beta)
    
    # Generate choices from true nested logit
    iv_motor = np.log(np.sum(np.exp(V_true[:, NESTS['Motorised']]), axis=1))
    iv_nonmotor = np.log(np.sum(np.exp(V_true[:, NESTS['NonMotorised']]), axis=1))
    IV = np.stack([iv_motor, iv_nonmotor], axis=1)
    mu = np.array([true_mu_motor, true_mu_nonmotor])
    nest_prob = np.exp(mu * IV) / np.sum(np.exp(mu * IV), axis=1, keepdims=True)
    
    choices = np.zeros(n_cases, dtype=int)
    for i in range(n_cases):
        nest_choice = np.random.choice(2, p=nest_prob[i])
        if nest_choice == 0:  # Motorised
            V_nest = V_true[i, NESTS['Motorised']]
            p = np.exp(V_nest) / np.sum(np.exp(V_nest))
            choices[i] = NESTS['Motorised'][np.random.choice(len(NESTS['Motorised']), p=p)]
        else:  # NonMotorised
            V_nest = V_true[i, NESTS['NonMotorised']]
            p = np.exp(V_nest) / np.sum(np.exp(V_nest))
            choices[i] = NESTS['NonMotorised'][np.random.choice(len(NESTS['NonMotorised']), p=p)]
    
    # ── JAX optimization ────────────────────────────────────────────────
    print("Testing JAX nested logit...")
    
    # Wrap for scipy.optimize
    def neg_loglike(params):
        return -float(jax_nested_loglike(params, X, choices))
    
    # Initial guess
    init_params = np.zeros(len(true_beta) + N_NESTS - 1)
    init_params[:len(true_beta)] = true_beta * 0.5  # rough start
    init_params[-2:] = [0.5, 0.5]  # mu start
    
    print("Initial params:", init_params)
    print("True beta:", true_beta)
    print("True mu:", true_mu_motor, true_mu_nonmotor)
    
    # Check gradient at start
    grad_fn = jax.grad(lambda p: jax_nested_loglike(p, X, choices))
    grad = grad_fn(init_params)
    print("Gradient at init:", grad)
    print("Gradient norm:", np.linalg.norm(grad))
    
    # Finite-difference check
    eps = 1e-6
    fd_grad = np.zeros_like(init_params)
    for i in range(len(init_params)):
        p_plus = init_params.copy()
        p_minus = init_params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        fd_grad[i] = (jax_nested_loglike(p_plus, X, choices) - jax_nested_loglike(p_minus, X, choices)) / (2*eps)
    
    print("FD gradient:", fd_grad)
    print("JAX gradient:", grad)
    print("Max diff:", np.max(np.abs(grad - fd_grad)))
    
    # Optimize
    print("\nOptimizing...")
    result = minimize(neg_loglike, init_params, method='L-BFGS-B', 
                      options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    
    print("\nOptimization result:")
    print("  Success:", result.success)
    print("  Message:", result.message)
    print("  Estimated beta:", result.x[:len(true_beta)])
    print("  True beta:", true_beta)
    print("  Estimated mu:", result.x[-2:])
    print("  True mu:", true_mu_motor, true_mu_nonmotor)
    print("  LL:", -result.fun)
    
    return result

if __name__ == '__main__':
    test_jax_nested_logit()