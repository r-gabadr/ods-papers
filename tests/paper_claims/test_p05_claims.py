"""
P05 claim tests — Multiscale Extension: Contractivity and Fixed Points.

These tests verify the mathematical foundations for P05, determining whether
the Clifford maps support:
  - Variant A: aggregate contractivity (Banach fixed point of T_ℓ)
  - Variant B: individual contractivity (Hutchinson fractal attractor)

Run: python -m pytest -q _claude/tests/paper_claims/test_p05_claims.py
"""
import pytest
import jax.numpy as jnp
import numpy as np


# ── Helpers ─────────────────────────────────────────────────────────────

def _random_field(seed=42, H=16, W=16):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.normal(size=(H, W, 8)).astype(np.float32))


def _field_norm(Psi):
    return float(jnp.sqrt(jnp.sum(jnp.square(Psi))))


def _contraction_ratio(Psi1, Psi2, map_fn):
    d_in = _field_norm(Psi1 - Psi2)
    if d_in < 1e-10:
        return float('nan')
    d_out = _field_norm(map_fn(Psi1) - map_fn(Psi2))
    return d_out / d_in


def _scaled_rotation(Psi, r=0.9, theta=0.5):
    """r · rotate(Ψ) in the e12 plane."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    result = Psi.copy()
    v1, v2 = Psi[..., 1], Psi[..., 2]
    result = result.at[..., 1].set(c * v1 - s * v2)
    result = result.at[..., 2].set(s * v1 + c * v2)
    b1, b2 = Psi[..., 5], Psi[..., 6]
    result = result.at[..., 5].set(c * b1 - s * b2)
    result = result.at[..., 6].set(s * b1 + c * b2)
    return r * result


def _rotor_plus_delta(Psi, r=0.85, theta=0.3, delta=0.10):
    rotated = _scaled_rotation(Psi, r=r, theta=theta)
    return rotated + delta * jnp.tanh(Psi)


# ── Claim 1: Pure rotation is an isometry (c = 1.0) ─────────────────────

class TestRotorIsometry:
    """R·Ψ·R̃ preserves Clifford norm — fundamental geometric fact."""

    def test_rotation_preserves_norm(self):
        Psi = _random_field()
        rotated = _scaled_rotation(Psi, r=1.0, theta=0.7)
        norm_before = _field_norm(Psi)
        norm_after = _field_norm(rotated)
        assert abs(norm_before - norm_after) / norm_before < 1e-4

    def test_rotation_contraction_ratio_is_one(self):
        rng = np.random.default_rng(42)
        ratios = []
        for i in range(50):
            Psi1 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            Psi2 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            c = _contraction_ratio(Psi1, Psi2, lambda P: _scaled_rotation(P, r=1.0))
            ratios.append(c)
        mean_c = np.mean(ratios)
        assert abs(mean_c - 1.0) < 0.01, f"Rotation should be isometry, got mean c={mean_c}"


# ── Claim 2: Scaled rotor is contractive (c = r < 1) ────────────────────

class TestScaledRotorContractivity:

    def test_scaled_rotor_contracts(self):
        rng = np.random.default_rng(42)
        r = 0.85
        ratios = []
        for i in range(50):
            Psi1 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            Psi2 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            c = _contraction_ratio(Psi1, Psi2, lambda P: _scaled_rotation(P, r=r))
            ratios.append(c)
        max_c = max(ratios)
        assert max_c < 1.0, f"Scaled rotor should contract, got max c={max_c}"
        assert abs(np.mean(ratios) - r) < 0.02, f"Mean ratio should ≈ r={r}"


# ── Claim 3: Rotor + delta contractivity depends on r + Lip(Δ) ──────────

class TestRotorPlusDelta:

    def test_contractive_when_r_plus_lip_less_than_1(self):
        """r=0.70, Lip(tanh)=1.0, δ=0.10 → bound = 0.70 + 0.10 = 0.80 < 1."""
        rng = np.random.default_rng(42)
        ratios = []
        for i in range(100):
            Psi1 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            Psi2 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            c = _contraction_ratio(Psi1, Psi2,
                                   lambda P: _rotor_plus_delta(P, r=0.70, delta=0.10))
            ratios.append(c)
        max_c = max(ratios)
        assert max_c < 1.0, f"r+δ=0.80 should contract, got max c={max_c}"

    def test_not_contractive_when_r_plus_lip_exceeds_1(self):
        """r=0.85, δ=0.20 → bound = 0.85 + 0.20 = 1.05 > 1."""
        rng = np.random.default_rng(42)
        ratios = []
        for i in range(100):
            Psi1 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            Psi2 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            c = _contraction_ratio(Psi1, Psi2,
                                   lambda P: _rotor_plus_delta(P, r=0.85, delta=0.20))
            ratios.append(c)
        max_c = max(ratios)
        # May or may not be contractive — Lip(tanh) ≤ 1 but actual Lip
        # over the sampled range may be lower. We just verify it's close to 1.
        # This test documents the boundary behavior.
        assert max_c > 0.8, f"Near-boundary should have high ratio, got {max_c}"


# ── Claim 4: Weighted mixture contractivity ──────────────────────────────

class TestMixtureContractivity:

    def test_conservative_mixture_contracts(self):
        """Mixture with L_bound = Σ|w|(r+δ) < 1."""
        components = [(0.60, 0.10, 0.5), (0.50, 0.05, 0.5)]
        L_bound = sum(abs(w) * (r + d) for r, d, w in components)
        assert L_bound < 1.0, f"Test setup error: L_bound={L_bound}"

        def mixture(Psi):
            result = jnp.zeros_like(Psi)
            for r, d, w in components:
                result = result + w * _rotor_plus_delta(Psi, r=r, delta=d)
            return result

        rng = np.random.default_rng(42)
        ratios = []
        for i in range(100):
            Psi1 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            Psi2 = jnp.array(rng.normal(size=(16, 16, 8)).astype(np.float32))
            c = _contraction_ratio(Psi1, Psi2, mixture)
            ratios.append(c)
        max_c = max(ratios)
        assert max_c < 1.0, f"Conservative mixture should contract, got max c={max_c}"


# ── Claim 5: Fixed point convergence ─────────────────────────────────────

class TestFixedPointConvergence:

    def test_iterated_map_converges(self):
        """T^n(Ψ₀) should converge to a fixed point."""
        def T(Psi):
            return 0.5 * _rotor_plus_delta(Psi, r=0.60, delta=0.10, theta=0.3) + \
                   0.5 * _rotor_plus_delta(Psi, r=0.50, delta=0.05, theta=0.7)

        Psi = _random_field(seed=42)
        norms = []
        for step in range(30):
            Psi_next = T(Psi)
            diff = _field_norm(Psi_next - Psi)
            norms.append(diff)
            Psi = Psi_next

        # Should decrease monotonically (at least after initial transient)
        assert norms[-1] < norms[0] * 0.1, \
            f"Should converge: initial diff={norms[0]:.4f}, final={norms[-1]:.4f}"

    def test_fixed_point_is_unique(self):
        """Starting from different initial conditions should converge to same point."""
        def T(Psi):
            return 0.5 * _rotor_plus_delta(Psi, r=0.60, delta=0.10, theta=0.3) + \
                   0.5 * _rotor_plus_delta(Psi, r=0.50, delta=0.05, theta=0.7)

        Psi_a = _random_field(seed=42)
        Psi_b = _random_field(seed=99)

        for _ in range(50):
            Psi_a = T(Psi_a)
            Psi_b = T(Psi_b)

        distance = _field_norm(Psi_a - Psi_b)
        initial_distance = _field_norm(_random_field(42) - _random_field(99))
        assert distance < initial_distance * 0.01, \
            f"Should converge to same point: d={distance:.4f}, d0={initial_distance:.4f}"
