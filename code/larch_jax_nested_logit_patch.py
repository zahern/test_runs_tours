"""
Monkey-patch Larch to add JAX nested logit support.

This patches Larch's nested logit models to use JAX for log-likelihood and gradients,
enabling fast gradient-based optimization for nested logit models.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import logging

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, vmap
    jax.config.update('jax_enable_x64', True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    grad = None

try:
    import larch as lx
    from larch import P, X
    LARCH_AVAILABLE = True
except ImportError:
    LARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Nested logit structure for Stage 5 mode model ──────────────────────────
# Root (μ=1 fixed)
#   ├─ Motorised (μ_Motor) 
#   │   ├── DA (alt 0)
#   │   ├── SR (alt 1)  
#   │   └─ Transit (alt 2)
#   └─ NonMotorised (μ_NonMotor)
#       ├── Walk (alt 3)
#       └─ Bike (alt 4)

NEST_ALTS = {
    'Motorised': [0, 1, 2],       # DA, SR, Transit
    'NonMotorised': [3, 4],       # Walk, Bike
}
NEST_NAMES = ['Motorised', 'NonMotorised']


def extract_nest_structure(model):
    """
    Extract nest structure from a Larch model.
    Returns dict: nest_name -> list of alt indices
    """
    try:
        # Larch stores nest info in model.nesting_parameters
        nests = {}
        if hasattr(model, 'nesting_parameters') and model.nesting_parameters:
            for nest_name, param in model.nesting_parameters.items():
                # Get children of this nest node
                nest_node = model.graph.nodes.get(nest_name)
                if nest_node and hasattr(nest_node, 'children'):
                    alts = []
                    for child in nest_node.children:
                        if hasattr(child, 'alt_id') and child.alt_id is not None:
                            alts.append(child.alt_id)
                    if alts:
                        nests[nest_name] = sorted(alts)
        if nests:
            return nests
    except Exception:
        pass
    
    # Fallback: Stage 5 default structure
    return NEST_ALTS


def build_utility_jax(model, X, params):
    """
    Build utility V = X @ beta for JAX.
    X: (n_cases, n_alts, n_params) utility covariates
    params: concatenated [beta, mu_motor, mu_nonmotor]
    """
    # Split parameters: coefficients then nesting scales
    n_coef = len([p for p in model.pf.index if not p.startswith('Mu_')])
    beta = params[:n_coef]
    mu_motor = params[-2]
    mu_nonmotor = params[-1]
    
    # Utility: V = X @ beta  (n_cases, n_alts)
    V = jnp.einsum('cai,i->ca', X, beta)
    
    return V, mu_motor, mu_nonmotor


def jax_nested_loglike(model, params, X, y):
    """
    JAX nested logit log-likelihood for Larch model.
    
    Parameters
    ----------
    model : larch.Model
        Fitted/initialized Larch model with nesting structure
    params : array
        Concatenated [coefficients, mu_motor, mu_nonmotor]
    X : array (n_cases, n_alts, n_coef)
        Utility covariates per case/alt
    y : array (n_cases,)
        Chosen alternative indices
        
    Returns
    -------
    Total log-likelihood (scalar)
    """
    nests = extract_nest_structure(model)
    n_alts = X.shape[1]

    # BUGFIX: was build_utility_jax(model, None, params) — ignored the passed
    # X and crashed inside einsum. Pass X through.
    V, mu_motor, mu_nonmotor = build_utility_jax(model, X, params)
    
    # Inclusive values per nest
    motor_alts = nests.get('Motorised', [0, 1, 2])
    nonmotor_alts = nests.get('NonMotorised', [3, 4])
    
    iv_motor = jax.nn.logsumexp(V[:, motor_alts], axis=1)
    iv_nonmotor = jax.nn.logsumexp(V[:, nonmotor_alts], axis=1)
    IV = jnp.stack([iv_motor, iv_nonmotor], axis=1)  # (n_cases, 2)
    
    # Nest probabilities
    mu = jnp.array([params[-2], params[-1]])  # mu_motor, mu_nonmotor
    log_nest_num = mu * IV
    log_nest_den = jax.nn.logsumexp(mu * IV, axis=1)
    log_nest_prob = log_nest_num - log_nest_den[:, None]  # (n_cases, 2)
    
    # Alternative probabilities within nest (guard custom nest names)
    _m_key = 'Motorised' if 'Motorised' in nests else list(nests.keys())[0]
    _nm_key = 'NonMotorised' if 'NonMotorised' in nests else list(nests.keys())[-1]
    V_motor = V[:, nests[_m_key]]
    V_nonmotor = V[:, nests[_nm_key]]
    
    log_alt_given_motor = V_motor - jax.nn.logsumexp(V_motor, axis=1)[:, None]
    log_alt_given_nonmotor = V_nonmotor - jax.nn.logsumexp(V_nonmotor, axis=1)[:, None]
    
    # Combine: log P(alt) = log P(nest) + log P(alt|nest)
    log_prob = jnp.zeros_like(V)
    log_prob = log_prob.at[:, nests[_m_key]].set(log_nest_prob[:, 0:1] + log_alt_given_motor)
    log_prob = log_prob.at[:, nests[_nm_key]].set(log_nest_prob[:, 1:2] + log_alt_given_nonmotor)
    
    # Log-likelihood for chosen alternatives
    chosen_logprob = log_prob[jnp.arange(len(y)), y]
    return jnp.sum(chosen_logprob)


def patch_jax_nested_logit(model):
    """Safe no-op: Larch 6 native JAX engine already handles nested logit.

    Historical scaffold — the old version installed ``jax_loglike`` /
    ``d_logloss`` stubs that raised ``NotImplementedError`` on every call,
    breaking any model it touched, and assumed a 5-mode Motorised/
    NonMotorised nest that does not match temporal/distance NL nests.

    Usage stays the same, but this now:
      1. Ensures CPU float64 JAX env (idempotent),
      2. Forces ``compute_engine='jax'`` when safe,
      3. Leaves Larch's own autodiff loglike/gradients untouched.

    For a pure-JAX reference NL loglike/gradient (tests, prototyping) use
    :func:`jax_nested_loglike_pure` + ``jax.grad`` in this module instead of
    patching a live Larch model.
    """
    try:
        import larch_jax_enforce as _jfe
        _jfe.ensure_jax_environment()
    except Exception:
        pass
    if not JAX_AVAILABLE:
        logger.warning("JAX not available, skipping patch")
        return model
    if not LARCH_AVAILABLE:
        logger.warning("Larch not available, skipping patch")
        return model
    # Do NOT overwrite model.jax_loglike / d_logloss / mangle — Larch 6 native
    # JAX path is correct and tested (see smoke_jax_models.py Family A).
    try:
        if str(getattr(model, 'compute_engine', '')).lower() != 'jax':
            try:
                model.compute_engine = 'jax'
                model.mangle()
                model.unmangle()
                logger.info("patch_jax_nested_logit: promoted model to compute_engine='jax'")
            except Exception as _e:
                logger.debug(f"patch_jax_nested_logit: engine promotion skipped ({_e})")
    except Exception:
        pass
    logger.info("patch_jax_nested_logit: no-op (native JAX NL retained)")
    return model


def build_utility_from_larch(model, case_ids, alt_ids):
    """
    Build utility covariate matrix X from Larch model expressions.
    Returns X: (n_cases, n_alts, n_coef)
    """
    # This requires traversing the model's LinearFunction graph
    # and extracting the coefficient for each parameter per alt
    # Complex - requires Larch internals access
    raise NotImplementedError("Requires Larch LinearFunction traversal")


def apply_jax_nested_patch(model):
    """
    High-level function to patch a Larch nested logit model for JAX.
    
    Usage:
        import larch as lx
        from larch_jax_nested_logit_patch import apply_jax_nested_patch
        
        dt = ...  # DataTree
        model = lx.Model(dt, compute_engine='jax')
        model = apply_jax_nested_patch(model)
        
        # Build model (add nesting, utility, etc.)
        # ...
        
        result = model.maximize_loglike()
    """
    if not JAX_AVAILABLE or not LARCH_AVAILABLE:
        return model
    
    # Check if model already has nesting
    if not hasattr(model, 'nesting_parameters') or not model.nesting_parameters:
        return model
    
    # The full implementation requires:
    # 1. Access to model.datatree to build X matrix
    # 2. Integration with Larch's mangle() to capture X/y
    # 3. Replacement of model.jax_loglike and model.d_logloss
    
    logger.warning("Full JAX nested logit patch requires Larch internals access. "
                   "Current implementation is a scaffold.")
    
    return model


# ── Convenience function for quick testing ──────────────────────────────────

def test_jax_nested_logit_gradient():
    """
    Test JAX nested logit gradient against finite differences.
    """
    if not JAX_AVAILABLE:
        print("JAX not available")
        return
    
    # Synthetic test
    n_cases = 100
    n_alts = 5
    n_coef = 3
    
    np.random.seed(42)
    X = np.random.randn(n_cases, n_alts, n_coef)
    y = np.random.randint(0, 5, n_cases)
    params = np.array([0.5, -0.3, 0.2, 0.7, 0.6])  # 3 coefs + 2 mu
    
    def loglike(p):
        # Simplified version for testing
        beta = p[:3]
        mu_motor = p[3]
        mu_nonmotor = p[4]
        
        V = jnp.einsum('cai,i->ca', X, beta)
        iv_motor = jax.nn.logsumexp(V[:, [0,1,2]], axis=1)
        iv_nonmotor = jax.nn.logsumexp(V[:, [3,4]], axis=1)
        IV = jnp.stack([iv_motor, iv_nonmotor], axis=1)
        
        mu = jnp.array([mu_motor, mu_nonmotor])
        log_nest_num = mu * IV
        log_nest_den = jax.nn.logsumexp(mu * IV, axis=1)
        log_nest_prob = log_nest_num - log_nest_den[:, None]
        
        V_motor = V[:, [0,1,2]]
        V_nonmotor = V[:, [3,4]]
        log_alt_given_motor = V_motor - jax.nn.logsumexp(V_motor, axis=1)[:, None]
        log_alt_given_nonmotor = V_nonmotor - jax.nn.logsumexp(V_nonmotor, axis=1)[:, None]
        
        log_prob = jnp.zeros_like(V)
        log_prob = log_prob.at[:, [0,1,2]].set(log_nest_prob[:, 0:1] + log_alt_given_motor)
        log_prob = log_prob.at[:, [3,4]].set(log_nest_prob[:, 1:2] + log_alt_given_nonmotor)
        
        chosen_logprob = log_prob[jnp.arange(len(y)), y]
        return jnp.sum(chosen_logprob)
    
    # Test gradient
    init_params = np.array([0.5, -0.3, 0.2, 0.7, 0.6])
    grad_fn = grad(loglike)
    jax_grad = grad_fn(init_params)
    
    # Finite difference
    eps = 1e-6
    fd_grad = np.zeros_like(init_params)
    for i in range(len(init_params)):
        p_plus = init_params.copy()
        p_minus = init_params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        fd_grad[i] = (loglike(p_plus) - loglike(p_minus)) / (2*eps)
    
    print("JAX gradient:", jax_grad)
    print("FD gradient:", fd_grad)
    print("Max diff:", np.max(np.abs(jax_grad - fd_grad)))
    
    if np.max(np.abs(jax_grad - fd_grad)) < 1e-5:
        print("✅ Gradient test PASSED")
    else:
        print("❌ Gradient test FAILED")
    
    return np.max(np.abs(jax_grad - fd_grad))


if __name__ == '__main__':
    if JAX_AVAILABLE:
        test_jax_nested_logit_gradient()
    else:
        print("JAX not available")