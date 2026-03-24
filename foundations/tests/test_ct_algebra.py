"""Tests for ct_algebra.py — Cl(p,q) for all CT signatures."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ct_algebra import CliffordAlgebra

import pytest


# ── Signature Ladder ──────────────────────────────────────────

SIGNATURES = [(1, 0), (2, 0), (3, 0), (1, 3), (4, 1), (2, 4)]


@pytest.mark.parametrize("p,q", SIGNATURES)
class TestCliffordRelations:
    """Verify γ_μ γ_ν + γ_ν γ_μ = 2g_{μν}I for every signature."""

    def test_anticommutation(self, p, q):
        cl = CliffordAlgebra(p, q)
        assert cl.verify_clifford_relations(), f"Cl({p},{q}) failed anticommutation"

    def test_dimension(self, p, q):
        cl = CliffordAlgebra(p, q)
        assert cl.dim == 2 ** (p + q)

    def test_blade_count(self, p, q):
        cl = CliffordAlgebra(p, q)
        assert len(cl.blade_basis()) == cl.dim

    def test_gamma_squares(self, p, q):
        cl = CliffordAlgebra(p, q)
        I_mat = np.eye(cl.mat_dim, dtype=np.complex128)
        for mu in range(cl.n):
            sq = cl.gammas[mu] @ cl.gammas[mu]
            expected = cl.metric[mu, mu] * I_mat
            np.testing.assert_allclose(sq, expected, atol=1e-12,
                err_msg=f"γ_{mu}² failed in Cl({p},{q})")


@pytest.mark.parametrize("p,q", SIGNATURES)
class TestOperations:
    """Test reversion, conjugation, norm for all signatures."""

    def test_scalar_norm(self, p, q):
        cl = CliffordAlgebra(p, q)
        s = np.zeros(cl.dim)
        s[0] = 3.0  # scalar = 3
        assert abs(cl.norm(s) - 3.0) < 1e-12

    def test_reversion_scalar(self, p, q):
        cl = CliffordAlgebra(p, q)
        s = np.zeros(cl.dim)
        s[0] = 5.0
        rev = cl.reversion(s)
        np.testing.assert_allclose(rev, s, atol=1e-14)

    def test_reversion_vector(self, p, q):
        """Reversion of a vector is itself (grade 1: sign = +1)."""
        cl = CliffordAlgebra(p, q)
        if cl.n == 0:
            return
        v = np.zeros(cl.dim)
        v[1] = 2.0  # first vector component
        rev = cl.reversion(v)
        np.testing.assert_allclose(rev, v, atol=1e-14)

    def test_reversion_bivector(self, p, q):
        """Reversion of bivector flips sign: (-1)^{1} = -1."""
        cl = CliffordAlgebra(p, q)
        if cl.n < 2:
            return
        bv = np.zeros(cl.dim)
        # Find first bivector index
        for i, idx in enumerate(cl._blade_indices):
            if len(idx) == 2:
                bv[i] = 1.0
                break
        rev = cl.reversion(bv)
        np.testing.assert_allclose(rev, -bv, atol=1e-14)

    def test_geometric_product_associativity(self, p, q):
        cl = CliffordAlgebra(p, q)
        rng = np.random.RandomState(42)
        a = rng.randn(cl.dim)
        b = rng.randn(cl.dim)
        c = rng.randn(cl.dim)
        ab_c = cl.geometric_product(cl.geometric_product(a, b), c)
        a_bc = cl.geometric_product(a, cl.geometric_product(b, c))
        np.testing.assert_allclose(ab_c.real, a_bc.real, atol=1e-8,
            err_msg=f"Associativity failed in Cl({p},{q})")


class TestCl30Specific:
    """Regression tests for the current ODS runtime algebra."""

    def test_matches_known_structure(self):
        cl = CliffordAlgebra(3, 0)
        assert cl.dim == 8
        assert cl.mat_dim in (2, 4, 8)  # recursive construction gives 2^ceil(n/2)  # depends on construction
        assert cl.n == 3
        assert cl.verify_clifford_relations()

    def test_all_gammas_anticommute(self):
        cl = CliffordAlgebra(3, 0)
        I_mat = np.eye(cl.mat_dim, dtype=np.complex128)
        for i in range(3):
            for j in range(i + 1, 3):
                anticomm = cl.gammas[i] @ cl.gammas[j] + cl.gammas[j] @ cl.gammas[i]
                np.testing.assert_allclose(anticomm, np.zeros_like(I_mat), atol=1e-12)
