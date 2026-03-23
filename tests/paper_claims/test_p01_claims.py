"""
P01 claim verification tests.
Paper: A Clifford-Thomas Runtime with Inter-Grade Transport and Split-Time Relaxation
DOI: 10.5281/zenodo.18995428

Each test corresponds to a specific claim in the paper.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np

from ods_unified_v2.jax_runtime.energy import free_energy_frozen, grad_free_energy_frozen
from ods_unified_v2.jax_runtime.integrator import default_params, step_frozen
from ods_unified_v2.jax_runtime.metrics import effective_dimension
from ods_unified_v2.jax_runtime.transport import diss_transport_12, rev_transport_12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_state():
    """Minimal 4x4 field for fast claim tests."""
    key = jax.random.PRNGKey(42)
    return 0.15 * jax.random.normal(key, (1, 4, 4, 8), dtype=jnp.float32)


@pytest.fixture
def params():
    return default_params()


# ---------------------------------------------------------------------------
# C1: F = E - TH is monotone decreasing during tau sub-steps
# Paper claim: "the tau sub-loop performs gradient descent on F"
# ---------------------------------------------------------------------------

def test_p01_c1_tau_monotone_free_energy(small_state, params):
    """P01-C1: Free energy F must not increase during tau relaxation."""
    state = small_state
    mem_target = jnp.zeros_like(state)

    F_values = []
    for _ in range(8):
        F = float(free_energy_frozen(
            state,
            mem_target=mem_target,
            nu=params["nu"],
            temperature=params["temperature"],
            lam_mem=0.0,
        ))
        F_values.append(F)
        # one tau sub-step
        grad = grad_free_energy_frozen(
            state,
            mem_target=mem_target,
            nu=params["nu"],
            temperature=params["temperature"],
            lam_mem=0.0,
        )
        state = state - params["tau_dt"] * grad

    deltas = [F_values[i+1] - F_values[i] for i in range(len(F_values)-1)]
    assert all(d <= 1e-6 for d in deltas), (
        f"P01-C1 FAIL: F not monotone. Deltas: {deltas}"
    )


# ---------------------------------------------------------------------------
# C2: Reversible transport conserves E12 = ||v||^2 + ||B||^2
# Paper claim: "the reversible block conserves E12"
# ---------------------------------------------------------------------------

def test_p01_c2_reversible_transport_conserves_E12(small_state):
    """P01-C2: Reversible transport flow must have zero inner product with state gradient of E12."""
    state = small_state
    flow = rev_transport_12(state, omega12=0.5)
    v = state[..., 1:4]
    b = state[..., 4:7]
    dv = flow[..., 1:4]
    db = flow[..., 4:7]
    dE_dt = float(2.0 * jnp.sum(v * dv + b * db))
    assert abs(dE_dt) < 1e-5, f"P01-C2 FAIL: dE12/dt = {dE_dt}, expected ~0"


# ---------------------------------------------------------------------------
# C3: Dissipative transport reduces vector-bivector misalignment
# Paper claim: "the dissipative block reduces ||v - beta||^2"
# ---------------------------------------------------------------------------

def test_p01_c3_dissipative_transport_reduces_misalignment(small_state):
    """P01-C3: One dissipative step must reduce ||v - B||^2."""
    state = small_state
    flow = diss_transport_12(state, gamma12=0.5)
    next_state = state + 0.05 * flow
    before = float(jnp.mean((state[..., 1:4] - state[..., 4:7]) ** 2))
    after = float(jnp.mean((next_state[..., 1:4] - next_state[..., 4:7]) ** 2))
    assert after <= before + 1e-7, (
        f"P01-C3 FAIL: misalignment increased: {before:.6f} -> {after:.6f}"
    )


# ---------------------------------------------------------------------------
# C4: Manual gradient of F matches autodiff
# Paper claim: "analytic gradient is equivalent to JAX autodiff"
# ---------------------------------------------------------------------------

def test_p01_c4_grad_free_energy_matches_autodiff(small_state):
    """P01-C4: Manual grad_free_energy_frozen must match jax.grad to 1e-4."""
    state = small_state
    mem_target = jnp.zeros_like(state)
    nu = 0.05
    temperature = 0.10

    manual = grad_free_energy_frozen(
        state, mem_target=mem_target, nu=nu, temperature=temperature, lam_mem=0.0,
    )

    def F_scalar(s):
        return free_energy_frozen(
            s, mem_target=mem_target, nu=nu, temperature=temperature, lam_mem=0.0,
        )

    auto = jax.grad(F_scalar)(state)
    max_err = float(jnp.max(jnp.abs(manual - auto)))
    assert max_err < 1e-4, f"P01-C4 FAIL: max grad error = {max_err}"


# ---------------------------------------------------------------------------
# C5: d_eff is bounded in [1, 8]
# Paper claim: "d_eff = exp(H) with H bounded by log(8)"
# ---------------------------------------------------------------------------

def test_p01_c5_deff_bounded(small_state):
    """P01-C5: d_eff must satisfy 1 <= d_eff <= 8 for Cl(3,0) with 8 channels."""
    state = small_state
    d = float(effective_dimension(state))
    assert 1.0 <= d <= 8.0 + 1e-6, f"P01-C5 FAIL: d_eff = {d} outside [1, 8]"
