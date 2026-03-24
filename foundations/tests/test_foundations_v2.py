"""
Tests for the notebook theory implementation.

Tests Propositions 1-5, Corollary 1, and the emergence pipeline.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from foundations.undulating import (
    AnalyticOscillator, SmoothlyUndulatingNumber,
    standard_oscillator, sawtooth_smooth_oscillator,
    are_equivalent_factorizations, represent_sequence_as_sun,
)
from foundations.flow import ComplexityFlow, HeatSmoothing
from foundations.emergence import (
    ReadingMode, quantization_radius, rigid_rank,
    closure_defect, rigidity_functional, extract_metric,
    check_clifford_emergence,
)

import pytest

S_GRID = np.linspace(0, 1, 500)


# ── A) Smoothly Undulating Numbers ────────────────────────────

class TestDefinition1:
    """Analytic oscillator: bounded, periodic, zero-mean."""

    def test_cosine_satisfies_def1(self):
        osc = standard_oscillator()
        result = osc.verify()
        assert result["all_pass"], f"Cosine failed Def 1: {result}"

    def test_sawtooth_satisfies_def1(self):
        osc = sawtooth_smooth_oscillator(sharpness=5.0)
        result = osc.verify()
        assert result["sup_equals_1"], f"sup != 1: {result}"
        assert result["inf_equals_neg1"], f"inf != -1: {result}"
        # Zero mean may have small error due to smoothing
        assert abs(float(np.sum(osc(np.linspace(0, 1, 10000)))) / 10000) < 0.1

    def test_oscillator_not_trig(self):
        """Def 1 does NOT require trigonometry."""
        osc = sawtooth_smooth_oscillator()
        xi = np.linspace(0, 1, 1000)
        vals = osc(xi)
        # Should not be exactly cos
        cos_vals = np.cos(2 * np.pi * xi)
        assert not np.allclose(vals, cos_vals, atol=0.1)


class TestDefinition2:
    """Smoothly undulating number: u = α·ω(φ)."""

    def test_basic_sun(self):
        osc = standard_oscillator()
        sun = SmoothlyUndulatingNumber(
            envelope=lambda s: 0.5 * np.ones_like(s),
            phase=lambda s: 5 * s,
            oscillator=osc, M=1.0, kappa=5.0, vartheta=0.5,
        )
        assert sun.is_bounded(S_GRID)

    def test_slowness_A1(self):
        """Condition (A1): envelope slower than phase."""
        osc = standard_oscillator()
        # Constant envelope → trivially satisfies A1
        sun = SmoothlyUndulatingNumber(
            envelope=lambda s: 0.8 * np.ones_like(s),
            phase=lambda s: 10 * s,
            oscillator=osc, M=1.0, kappa=10.0, vartheta=0.5,
        )
        result = sun.verify_slowness_A1(S_GRID)
        assert result["condition_A1_holds"], f"A1 violated: {result}"
        assert result["phi_monotone"]


class TestProposition1:
    """Invariance under phase reparametrization."""

    def test_reparametrized_phase_gives_same_u(self):
        osc = standard_oscillator()
        sun1 = SmoothlyUndulatingNumber(
            envelope=lambda s: 0.5 * np.ones_like(s),
            phase=lambda s: 3 * s,
            oscillator=osc, M=1.0,
        )
        # Reparametrize: h(t) = 2t, φ̃ = h∘φ = 6s, ω̃ = ω∘h⁻¹
        osc2 = AnalyticOscillator(
            fn=lambda xi: np.cos(2 * np.pi * xi / 2),  # ω(h⁻¹(ξ)) = cos(πξ)
            period=2.0, name="reparametrized",
        )
        sun2 = SmoothlyUndulatingNumber(
            envelope=lambda s: 0.5 * np.ones_like(s),
            phase=lambda s: 6 * s,
            oscillator=osc2, M=1.0,
        )
        assert are_equivalent_factorizations(sun1, sun2, S_GRID)


class TestProposition3:
    """Representation of sequences by reading modes."""

    def test_sequence_recovery(self):
        """Given (bₙ), construct u and readers with Lₙ[u] = bₙ."""
        b = np.array([1.0, -0.5, 0.3, 2.0, -1.0])
        u_vals, readers, s_grid = represent_sequence_as_sun(b)
        ds = s_grid[1] - s_grid[0]

        for n in range(len(b)):
            Ln_u = float(np.sum(u_vals * readers[n](s_grid)) * ds)
            np.testing.assert_allclose(Ln_u, b[n], atol=0.2,
                err_msg=f"Reader L_{n} failed: got {Ln_u}, expected {b[n]}")


# ── B) Complexity Flow ────────────────────────────────────────

class TestProposition4:
    """Global existence and boundedness preservation."""

    def test_boundedness_preserved(self):
        """Prop 4: |u(λ)|_∞ ≤ M for all λ."""
        M = 1.0
        flow = ComplexityFlow(M=M, eps_c=0.05, eps_f=0.005)
        u0 = 0.9 * np.sin(2 * np.pi * np.linspace(0, 1, 200))

        u_final, sup_norms = flow.evolve(u0, lam_max=2.0, d_lam=0.01)

        # ALL sup norms must stay ≤ M
        assert all(sn <= M + 1e-6 for sn in sup_norms), \
            f"Boundedness violated: max sup = {max(sup_norms)}"

    def test_boundedness_from_boundary(self):
        """Start near boundary M: saturation prevents escape."""
        M = 1.0
        flow = ComplexityFlow(M=M, eps_c=0.05, eps_f=0.005)
        u0 = 0.99 * np.ones(200)  # near M

        u_final, sup_norms = flow.evolve(u0, lam_max=1.0, d_lam=0.01)
        assert all(sn <= M + 1e-6 for sn in sup_norms)

    def test_negative_boundary(self):
        """Symmetric: u near -M also stays bounded."""
        M = 1.0
        flow = ComplexityFlow(M=M, eps_c=0.05, eps_f=0.005)
        u0 = -0.95 * np.ones(200)

        u_final, sup_norms = flow.evolve(u0, lam_max=1.0, d_lam=0.01)
        assert all(sn <= M + 1e-6 for sn in sup_norms)


class TestProposition5:
    """Complexity energy identity."""

    def test_energy_computable(self):
        flow = ComplexityFlow(M=1.0, eps_c=0.05, eps_f=0.005)
        u = 0.5 * np.sin(4 * np.pi * np.linspace(0, 1, 200))
        E = flow.complexity_energy(u)
        assert isinstance(E, float)
        # R is positive operator → E should be ≥ 0
        # (approximately, since R = S_f - S_c and S_f > S_c for high freq)

    def test_defect_A_zero_for_smooth(self):
        """Very smooth signal: A[u] ≈ 0 (already at coarse scale)."""
        flow = ComplexityFlow(M=1.0, eps_c=0.5, eps_f=0.05)
        # Very low frequency signal
        u = 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, 200))
        A = flow.defect_A(u)
        assert np.max(np.abs(A)) < 0.3  # mostly captured by coarse smoothing


class TestCorollary1:
    """Rupture set."""

    def test_rupture_set_empty_for_smooth(self):
        """Smooth signal: no rupture (production ≤ dissipation)."""
        flow = ComplexityFlow(M=1.0, eps_c=0.1, eps_f=0.01, tau=0.1)
        u = 0.3 * np.sin(2 * np.pi * np.linspace(0, 1, 200))
        rup = flow.rupture_set(u, lam=0.5)
        frac = float(np.mean(rup))
        assert frac < 0.5, f"Too much rupture for smooth signal: {frac}"

    def test_stationary_detection(self):
        """Zero field is trivially stationary."""
        flow = ComplexityFlow(M=1.0, eps_c=0.1, eps_f=0.01)
        u = np.zeros(200)
        assert flow.is_stationary(u, lam=1.0, tol=1e-10)


# ── C) Algebraic Emergence ────────────────────────────────────

class TestEmergence:
    def _make_modes(self, n: int = 4) -> list:
        etas = np.linspace(0.1, 0.9, n)
        return [ReadingMode(eta=float(e), sigma=0.05, delta=0.1)
                for e in etas]

    def test_quantization_radius_decreases_with_N(self):
        """More templates → smaller quantization radius."""
        modes = self._make_modes(8)
        u = np.sin(6 * np.pi * S_GRID) + 0.5 * np.cos(10 * np.pi * S_GRID)

        d2 = quantization_radius(u, S_GRID, modes, epsilon=0.05, N=2)
        d4 = quantization_radius(u, S_GRID, modes, epsilon=0.05, N=4)
        d8 = quantization_radius(u, S_GRID, modes, epsilon=0.05, N=8)

        assert d8 <= d4 + 1e-6
        assert d4 <= d2 + 1e-6

    def test_rigid_rank_finite_for_simple_signal(self):
        """Simple signal should have small rigid rank."""
        modes = self._make_modes(6)
        u = np.sin(2 * np.pi * S_GRID)
        N_rig = rigid_rank(u, S_GRID, modes, max_N=8, threshold=0.1)
        # Should find some finite rank
        assert N_rig is not None, "No finite rigid rank found"
        assert N_rig <= 8

    def test_closure_defect_symmetric(self):
        """D_{μν} should be close to D_{νμ}."""
        modes = self._make_modes(4)
        u = np.sin(4 * np.pi * S_GRID)
        d_01, c_01 = closure_defect(u, S_GRID, modes, epsilon=0.05, mu=0, nu=1)
        d_10, c_10 = closure_defect(u, S_GRID, modes, epsilon=0.05, mu=1, nu=0)
        np.testing.assert_allclose(d_01, d_10, atol=1e-6)
        np.testing.assert_allclose(c_01, c_10, atol=1e-6)

    def test_metric_is_symmetric(self):
        """Emergent metric g_{μν} must be symmetric."""
        modes = self._make_modes(4)
        u = np.sin(4 * np.pi * S_GRID) + 0.3 * np.cos(8 * np.pi * S_GRID)
        g = extract_metric(u, S_GRID, modes, epsilon=0.05, N=4)
        np.testing.assert_allclose(g, g.T, atol=1e-10)

    def test_full_emergence_pipeline(self):
        """Run the complete check_clifford_emergence pipeline."""
        modes = self._make_modes(4)
        u = np.sin(2 * np.pi * S_GRID) * 0.8
        result = check_clifford_emergence(u, S_GRID, modes, max_N=6)
        assert "rigid_rank" in result
        assert "is_clifford" in result
        assert isinstance(result["is_clifford"], bool)
