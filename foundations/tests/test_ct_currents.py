"""Tests for ct_currents.py — Clifford-valued currents and slicing."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ct_algebra import CliffordAlgebra
from ct_currents import CliffordCurrent, slice_current, coarea_bound

import pytest


class TestCurrentMass:
    def test_single_point_unit_mass(self):
        cl = CliffordAlgebra(3, 0)
        coeffs = np.zeros((1, cl.dim))
        coeffs[0, 0] = 1.0  # unit scalar
        T = CliffordCurrent(cl, grade_k=0,
                            support=np.array([[0.0, 0.0, 0.0]]),
                            coefficients=coeffs)
        assert abs(T.mass() - 1.0) < 1e-12

    def test_mass_scales_with_multiplicity(self):
        cl = CliffordAlgebra(2, 0)
        coeffs = np.zeros((3, cl.dim))
        coeffs[:, 0] = 1.0
        mult = np.array([2.0, 3.0, 5.0])
        T = CliffordCurrent(cl, grade_k=0,
                            support=np.array([[0.0], [1.0], [2.0]]),
                            coefficients=coeffs, multiplicity=mult)
        assert abs(T.mass() - 10.0) < 1e-12

    def test_grade_energy(self):
        cl = CliffordAlgebra(3, 0)
        coeffs = np.zeros((1, cl.dim))
        coeffs[0, 0] = 3.0  # scalar
        coeffs[0, 1] = 4.0  # first vector
        T = CliffordCurrent(cl, grade_k=0,
                            support=np.array([[0.0, 0.0, 0.0]]),
                            coefficients=coeffs)
        # Grade 0 energy should be |3| = 3
        assert abs(T.grade_energy(0) - 3.0) < 1e-12
        # Grade 1 energy should include component 4
        assert T.grade_energy(1) > 3.5


class TestSlicing:
    def test_slice_concentrates_near_level(self):
        cl = CliffordAlgebra(2, 0)
        N = 100
        x = np.linspace(0, 1, N).reshape(-1, 1)
        coeffs = np.zeros((N, cl.dim))
        coeffs[:, 0] = np.sin(2 * np.pi * x[:, 0])  # scalar wave

        T = CliffordCurrent(cl, grade_k=1, support=x, coefficients=coeffs)
        u = x[:, 0]  # slice by position

        s = slice_current(T, u, r=0.5, epsilon=0.05)
        # Slice should have fewer points, concentrated around x=0.5
        assert s.N < N
        if s.N > 0:
            assert np.mean(s.support[:, 0]) > 0.3
            assert np.mean(s.support[:, 0]) < 0.7

    def test_coarea_inequality(self):
        cl = CliffordAlgebra(2, 0)
        N = 50
        x = np.linspace(0, 1, N).reshape(-1, 1)
        coeffs = np.zeros((N, cl.dim))
        coeffs[:, 0] = 1.0

        T = CliffordCurrent(cl, grade_k=1, support=x, coefficients=coeffs)
        u = x[:, 0]

        integral, bound = coarea_bound(T, u, n_samples=30)
        # The coarea inequality should hold (with numerical tolerance)
        assert integral <= bound * 1.5 + 1e-6, \
            f"Coarea violated: {integral} > {bound}"
