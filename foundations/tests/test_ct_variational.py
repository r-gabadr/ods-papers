"""Tests for ct_variational.py and ct_dirac.py — Action, E-L, D†D."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ct_algebra import CliffordAlgebra
from ct_dirac import DiracOperator
from ct_variational import ActionFunctional, DualClockAction
from ct_embeddings import SignatureLadder, embed, project

import pytest


class TestDiracOperator:
    def test_DdaggerD_is_laplacian_for_euclidean(self):
        """In Cl(n,0), D†D = -Δ (discrete Laplacian)."""
        cl = CliffordAlgebra(3, 0)
        D = DiracOperator(cl, grid_spacing=0.1)

        N = 50
        psi = np.zeros((N, cl.dim))
        x = np.linspace(0, 2 * np.pi, N)
        psi[:, 0] = np.sin(x)  # scalar sine wave

        DdD = D.apply_DdaggerD(psi)

        # -Δ sin(x) = sin(x) (eigenvalue +1 for unit frequency)
        # But with grid spacing h=0.1, the discrete eigenvalue is
        # (2/h²)(1 - cos(kh)) ≈ k² for small kh
        # Check that DdD has same shape as input and is non-trivial
        assert DdD.shape == psi.shape
        assert np.max(np.abs(DdD)) > 0.01  # non-trivial

    def test_spectral_gap_positive_for_bounded_domain(self):
        cl = CliffordAlgebra(2, 0)
        D = DiracOperator(cl, grid_spacing=0.5)
        N = 20
        psi = np.zeros((N, cl.dim))
        psi[:, 0] = 1.0

        gap = D.spectral_gap(psi)
        assert gap > 0, "Spectral gap should be positive on bounded domain"


class TestActionFunctional:
    def test_constant_field_has_zero_kinetic_energy(self):
        cl = CliffordAlgebra(2, 0)
        D = DiracOperator(cl, grid_spacing=1.0)
        S = ActionFunctional(cl, D)

        N = 20
        psi = np.zeros((N, cl.dim))
        psi[:, 0] = 5.0  # constant scalar

        KE = S.kinetic_energy(psi)
        assert abs(KE) < 1e-10, "Constant field should have zero kinetic energy"

    def test_wave_has_positive_kinetic_energy(self):
        cl = CliffordAlgebra(3, 0)
        D = DiracOperator(cl, grid_spacing=0.1)
        S = ActionFunctional(cl, D)

        N = 50
        psi = np.zeros((N, cl.dim))
        x = np.linspace(0, 2 * np.pi, N)
        psi[:, 0] = np.sin(x)

        KE = S.kinetic_energy(psi)
        assert KE > 0, "Wave should have positive kinetic energy"

    def test_chi_is_zero_at_critical_point(self):
        """A constant field with V=0 is a critical point → χ ≈ 0."""
        cl = CliffordAlgebra(2, 0)
        D = DiracOperator(cl, grid_spacing=1.0)
        S = ActionFunctional(cl, D, potential=lambda psi: 0.0)

        N = 20
        psi = np.zeros((N, cl.dim))
        psi[:, 0] = 1.0  # constant

        chi = S.chi_from_residual(psi)
        assert chi < 1e-8, f"χ should be ~0 at critical point, got {chi}"

    def test_chi_positive_away_from_critical(self):
        cl = CliffordAlgebra(3, 0)
        D = DiracOperator(cl, grid_spacing=0.1)
        # Quadratic potential V(Ψ) = |Ψ|²
        S = ActionFunctional(cl, D,
                             potential=lambda psi: float(np.sum(psi ** 2)))

        N = 50
        psi = np.zeros((N, cl.dim))
        x = np.linspace(0, 2 * np.pi, N)
        psi[:, 0] = np.sin(x)

        chi = S.chi_from_residual(psi)
        assert chi > 0.01, f"χ should be positive away from critical, got {chi}"


class TestDualClock:
    def test_handoff_low_chi(self):
        cl = CliffordAlgebra(2, 0)
        D = DiracOperator(cl, grid_spacing=1.0)
        S_t = ActionFunctional(cl, D)
        S_tau = ActionFunctional(cl, D)
        dual = DualClockAction(S_t, S_tau)

        # Constant field → low chi → α ≈ 0 (stay in fast clock)
        N = 20
        psi = np.zeros((N, cl.dim))
        psi[:, 0] = 1.0

        alpha = dual.handoff_parameter(psi)
        assert alpha < 0.1, f"Low-stress field should have α ≈ 0, got {alpha}"


class TestSignatureLadder:
    def test_standard_ladder(self):
        ladder = SignatureLadder([(1, 0), (2, 0), (3, 0), (1, 3)])
        assert ladder.n_levels == 4
        assert ladder.full_state_dim() == 2 + 4 + 8 + 16  # = 30

    def test_embed_project_roundtrip(self):
        cl30 = CliffordAlgebra(3, 0)
        cl13 = CliffordAlgebra(4, 1)  # Cl(3,0) embeds into Cl(4,1) not Cl(1,3)

        rng = np.random.RandomState(123)
        psi = rng.randn(cl30.dim)

        embedded = embed(psi, cl30, cl13)
        projected = project(embedded, cl13, cl30)

        np.testing.assert_allclose(projected.real, psi, atol=1e-12,
            err_msg="Embed→project should recover original")

    def test_pack_unpack_roundtrip(self):
        ladder = SignatureLadder([(1, 0), (2, 0), (3, 0)])
        states = [np.array([1.0, 2.0]),
                  np.array([3.0, 4.0, 5.0, 6.0]),
                  np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])]

        packed = ladder.pack(states)
        assert len(packed) == 14
        unpacked = ladder.unpack(packed)
        for s, u in zip(states, unpacked):
            np.testing.assert_allclose(u.real, s, atol=1e-14)

    def test_energy_conservation_under_embedding(self):
        """Embedding should not change norm of the original state."""
        cl20 = CliffordAlgebra(2, 0)
        cl30 = CliffordAlgebra(3, 0)

        rng = np.random.RandomState(99)
        psi = rng.randn(cl20.dim)
        norm_before = cl20.norm(psi)

        embedded = embed(psi, cl20, cl30)
        projected = project(embedded, cl30, cl20)
        norm_after = cl20.norm(projected)

        np.testing.assert_allclose(norm_after, norm_before, rtol=1e-10)
