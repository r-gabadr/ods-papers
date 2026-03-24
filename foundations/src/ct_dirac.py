"""
ct_dirac.py — Dirac operator and its adjoint composition D†D.

D = Σ_μ γ^μ ∂_μ

D†D is the principal part of the Euler-Lagrange operator
for the Dirichlet energy on Cl(p,q)-valued fields.

For Cl(n,0): D†D = -Δ (negative Laplacian) — elliptic
For Cl(p,q) with q>0: D†D has mixed signature — hyperbolic part
"""
import numpy as np
from typing import Optional
from ct_algebra import CliffordAlgebra


class DiracOperator:
    """
    Discrete Dirac operator on a regular grid.

    D Ψ(x) = Σ_μ γ^μ · (Ψ(x+e_μ) - Ψ(x-e_μ)) / (2h)

    where h is the grid spacing and e_μ is the unit vector in direction μ.
    """

    def __init__(self, algebra: CliffordAlgebra,
                 grid_spacing: float = 1.0,
                 spatial_dims: Optional[int] = None):
        self.algebra = algebra
        self.h = grid_spacing
        self.spatial_dims = spatial_dims or min(algebra.n, 3)
        # Use only spatial_dims generators for spatial derivatives
        # (for Cl(1,3), γ₀ is temporal, γ₁,γ₂,γ₃ are spatial)

    def apply(self, psi_field: np.ndarray) -> np.ndarray:
        """
        Apply D to a 1D field: (N, algebra.dim) → (N, algebra.dim)

        Uses central differences along the single spatial axis.
        For multi-D fields, this generalizes to sum over spatial dims.
        """
        N = len(psi_field)
        result = np.zeros_like(psi_field)

        # For 1D: only use first generator γ₀ or γ₁
        gamma_idx = 0  # first spatial generator

        for i in range(N):
            # Central difference
            ip1 = min(i + 1, N - 1)
            im1 = max(i - 1, 0)
            dpsi = (psi_field[ip1] - psi_field[im1]) / (2 * self.h)

            # Apply γ_μ · dpsi via geometric product
            result[i] = self.algebra.geometric_product(
                self._gamma_coeffs(gamma_idx), dpsi
            )

        return result

    def apply_adjoint(self, psi_field: np.ndarray) -> np.ndarray:
        """
        Apply D† to a field.

        D† = -Σ_μ γ^μ† ∂_μ = -Σ_μ g^{μμ} γ^μ ∂_μ  (for diagonal metric)

        In Euclidean signature (p,0): D† = -D
        In Lorentzian signature: D† picks up metric signs
        """
        N = len(psi_field)
        result = np.zeros_like(psi_field)
        gamma_idx = 0

        metric_sign = self.algebra.metric[gamma_idx, gamma_idx]

        for i in range(N):
            ip1 = min(i + 1, N - 1)
            im1 = max(i - 1, 0)
            dpsi = (psi_field[ip1] - psi_field[im1]) / (2 * self.h)

            gamma_dpsi = self.algebra.geometric_product(
                self._gamma_coeffs(gamma_idx), dpsi
            )
            result[i] = -metric_sign * gamma_dpsi

        return result

    def apply_DdaggerD(self, psi_field: np.ndarray) -> np.ndarray:
        """
        Apply D†D to a field.

        For Cl(n,0): D†D = -Δ (discrete Laplacian with sign)
        For Cl(p,q): D†D = -Σ g^{μμ} ∂²_μ  (d'Alembertian-like)
        """
        N = len(psi_field)
        result = np.zeros_like(psi_field)
        gamma_idx = 0
        metric_sign = self.algebra.metric[gamma_idx, gamma_idx]

        for i in range(N):
            ip1 = min(i + 1, N - 1)
            im1 = max(i - 1, 0)

            # Second derivative: (Ψ(x+h) - 2Ψ(x) + Ψ(x-h)) / h²
            d2psi = (psi_field[ip1] - 2 * psi_field[i] + psi_field[im1]) / (self.h ** 2)

            # D†D Ψ = -g^{μμ} ∂²_μ Ψ  (for each spatial direction)
            result[i] = -metric_sign * d2psi

        return result

    def spectrum_DdaggerD(self, psi_field: np.ndarray,
                          n_modes: int = 10) -> np.ndarray:
        """
        Estimate spectrum of D†D by applying it to Fourier-like modes.

        Returns approximate eigenvalues (ascending).
        """
        N = len(psi_field)
        eigenvalues = []

        for k in range(min(n_modes, N)):
            # Fourier mode as test vector
            mode = np.zeros_like(psi_field)
            freq = np.pi * (k + 1) / N
            for i in range(N):
                mode[i, 0] = np.sin(freq * i)  # scalar component

            DdD_mode = self.apply_DdaggerD(mode)

            # Rayleigh quotient
            num = sum(np.sum(DdD_mode[i] * mode[i]).real for i in range(N))
            den = sum(np.sum(mode[i] * mode[i]).real for i in range(N))

            if abs(den) > 1e-14:
                eigenvalues.append(num / den)

        return np.sort(eigenvalues)

    def _gamma_coeffs(self, mu: int) -> np.ndarray:
        """Get coefficient vector for γ_μ."""
        coeffs = np.zeros(self.algebra.dim, dtype=np.complex128)
        # γ_μ is the (mu+1)-th blade (index 0 is scalar)
        coeffs[mu + 1] = 1.0
        return coeffs

    def spectral_gap(self, psi_field: np.ndarray) -> float:
        """Estimate spectral gap of D†D (smallest nonzero eigenvalue)."""
        eigs = self.spectrum_DdaggerD(psi_field, n_modes=5)
        positive = [e for e in eigs if e > 1e-10]
        return min(positive) if positive else 0.0
