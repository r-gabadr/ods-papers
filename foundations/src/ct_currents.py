"""
ct_currents.py — Clifford-valued currents on discrete grids.

A current T is a Clifford-valued measure with:
  - support: a set of grid points (rectifiable set)
  - orientation: a k-vector at each point
  - multiplicity: a scalar weight at each point

Mass: M(T) = Σ_x |θ(x)| · |ξ(x)|

Slicing: ⟨T, u, r⟩ via the coarea construction.
"""
import numpy as np
from typing import Optional, Tuple, List
from ct_algebra import CliffordAlgebra


class CliffordCurrent:
    """
    A k-current on a discrete grid with values in Cl(p,q).

    Attributes:
        algebra: the underlying CliffordAlgebra
        grade_k: the grade of the current (dimension of support)
        support: (N, spatial_dim) array of grid positions
        coefficients: (N, algebra.dim) Clifford coefficients at each point
        multiplicity: (N,) scalar weights
    """

    def __init__(self, algebra: CliffordAlgebra, grade_k: int,
                 support: np.ndarray, coefficients: np.ndarray,
                 multiplicity: Optional[np.ndarray] = None):
        self.algebra = algebra
        self.grade_k = grade_k
        self.support = np.asarray(support, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.complex128)
        self.N = self.support.shape[0]

        if multiplicity is None:
            self.multiplicity = np.ones(self.N, dtype=np.float64)
        else:
            self.multiplicity = np.asarray(multiplicity, dtype=np.float64)

    def mass(self) -> float:
        """
        M(T) = Σ_x |θ(x)| · ‖ξ(x)‖

        Total geometric mass of the current.
        """
        total = 0.0
        for i in range(self.N):
            norm_xi = self.algebra.norm(self.coefficients[i])
            total += abs(self.multiplicity[i]) * norm_xi
        return total

    def grade_energy(self, k: int) -> float:
        """Energy concentrated in grade k."""
        proj = self.algebra.grade_projector(k)
        total = 0.0
        for i in range(self.N):
            projected = proj @ self.coefficients[i]
            total += abs(self.multiplicity[i]) * np.sqrt(
                np.sum(np.abs(projected) ** 2).real
            )
        return total

    def evaluate(self, omega: np.ndarray) -> complex:
        """
        Evaluate current on a "form" omega (coefficient vector at each point).

        T(ω) = Σ_x θ(x) ⟨ω(x), ξ(x)⟩
        """
        result = 0.0 + 0.0j
        for i in range(self.N):
            ip = self.algebra.inner_product(omega[i], self.coefficients[i])
            result += self.multiplicity[i] * ip
        return result

    def restrict(self, mask: np.ndarray) -> 'CliffordCurrent':
        """Restrict current to subset defined by boolean mask."""
        idx = np.where(mask)[0]
        return CliffordCurrent(
            algebra=self.algebra,
            grade_k=self.grade_k,
            support=self.support[idx],
            coefficients=self.coefficients[idx],
            multiplicity=self.multiplicity[idx],
        )


def slice_current(T: CliffordCurrent, u: np.ndarray,
                  r: float, epsilon: float = 0.1) -> CliffordCurrent:
    """
    Approximate the GMT slice ⟨T, u, r⟩.

    Uses a smoothed delta: δ_ε(u(x) - r) = exp(-(u-r)²/2ε²) / (ε√2π)

    Parameters:
        T: the current to slice
        u: (N,) function values at support points
        r: slice level
        epsilon: smoothing width

    Returns:
        A new current representing the slice.
    """
    # Smoothed delta weights
    delta_weights = np.exp(-0.5 * ((u - r) / epsilon) ** 2) / (epsilon * np.sqrt(2 * np.pi))

    # New multiplicity = old multiplicity × delta weight
    new_mult = T.multiplicity * delta_weights

    # Filter near-zero entries
    mask = np.abs(new_mult) > 1e-14 * np.max(np.abs(new_mult) + 1e-30)

    return CliffordCurrent(
        algebra=T.algebra,
        grade_k=max(T.grade_k - 1, 0),
        support=T.support[mask],
        coefficients=T.coefficients[mask],
        multiplicity=new_mult[mask],
    )


def coarea_bound(T: CliffordCurrent, u: np.ndarray,
                 n_samples: int = 50) -> Tuple[float, float]:
    """
    Verify coarea inequality: ∫ M(⟨T,u,r⟩) dr ≤ Lip(u) · M(T)

    Returns (integral_of_slice_masses, lip_u * mass_T)
    """
    r_min, r_max = np.min(u), np.max(u)
    if r_max - r_min < 1e-14:
        return 0.0, 0.0

    rs = np.linspace(r_min, r_max, n_samples)
    dr = rs[1] - rs[0]
    epsilon = 2 * dr  # smoothing ~ sample spacing

    integral = 0.0
    for r in rs:
        s = slice_current(T, u, r, epsilon=epsilon)
        integral += s.mass() * dr

    # Estimate Lipschitz constant of u
    if T.N > 1:
        diffs_u = np.abs(u[1:] - u[:-1])
        diffs_x = np.linalg.norm(T.support[1:] - T.support[:-1], axis=-1)
        diffs_x = np.maximum(diffs_x, 1e-14)
        lip_u = np.max(diffs_u / diffs_x)
    else:
        lip_u = 0.0

    return integral, lip_u * T.mass()
