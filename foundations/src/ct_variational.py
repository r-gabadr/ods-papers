"""
ct_variational.py — Variational calculus on Clifford-valued fields.

Defines:
  - Action functional S[Ψ]
  - Euler-Lagrange residual δS/δΨ
  - Energy (Dirichlet + potential)
  - Dual-clock decomposition S = S_t + S_τ + S_coupling
"""
import numpy as np
from typing import Callable, Optional, Tuple
from ct_algebra import CliffordAlgebra
from ct_dirac import DiracOperator


class ActionFunctional:
    """
    S[Ψ] = ∫ [ ½⟨DΨ, DΨ⟩ + V(Ψ) ] dμ

    Discretized on a 1D grid for now. Extends to nD.
    """

    def __init__(self, algebra: CliffordAlgebra,
                 dirac: 'DiracOperator',
                 potential: Optional[Callable] = None,
                 grid_spacing: float = 1.0):
        self.algebra = algebra
        self.dirac = dirac
        self.potential = potential or (lambda psi: 0.0)
        self.dx = grid_spacing

    def kinetic_energy(self, psi_field: np.ndarray) -> float:
        """
        ½ Σ_x ‖DΨ(x)‖² dx

        psi_field: (N_points, algebra.dim) array
        """
        D_psi = self.dirac.apply(psi_field)
        energy = 0.0
        for i in range(len(psi_field)):
            energy += 0.5 * self.algebra.norm_squared(D_psi[i])
        return energy * self.dx

    def potential_energy(self, psi_field: np.ndarray) -> float:
        """Σ_x V(Ψ(x)) dx"""
        energy = 0.0
        for i in range(len(psi_field)):
            energy += self.potential(psi_field[i])
        return energy * self.dx

    def action(self, psi_field: np.ndarray) -> float:
        """S[Ψ] = kinetic + potential"""
        return self.kinetic_energy(psi_field) + self.potential_energy(psi_field)

    def euler_lagrange_residual(self, psi_field: np.ndarray) -> np.ndarray:
        """
        δS/δΨ = D†DΨ + V'(Ψ)

        Returns residual field. When this is zero, Ψ is a critical point.
        """
        DdD_psi = self.dirac.apply_DdaggerD(psi_field)

        # Numerical gradient of potential
        eps = 1e-7
        V_grad = np.zeros_like(psi_field)
        for i in range(len(psi_field)):
            for j in range(self.algebra.dim):
                psi_plus = psi_field[i].copy()
                psi_plus[j] += eps
                psi_minus = psi_field[i].copy()
                psi_minus[j] -= eps
                V_grad[i, j] = (self.potential(psi_plus) -
                                self.potential(psi_minus)) / (2 * eps)

        return DdD_psi + V_grad

    def chi_from_residual(self, psi_field: np.ndarray) -> float:
        """
        χ = ‖δS/δΨ‖ / ‖Ψ‖

        This is what P03's χ_Cl SHOULD be: the normalized
        Euler-Lagrange residual. When χ→0, the field is at
        a critical point of the action.
        """
        residual = self.euler_lagrange_residual(psi_field)
        res_norm = np.sqrt(sum(
            self.algebra.norm_squared(residual[i])
            for i in range(len(psi_field))
        ))
        psi_norm = np.sqrt(sum(
            self.algebra.norm_squared(psi_field[i])
            for i in range(len(psi_field))
        )) + 1e-30
        return res_norm / psi_norm


class DualClockAction:
    """
    S[Ψ] = S_t[Ψ] + S_τ[Ψ] + S_coupling[Ψ]

    where α(χ) is the handoff weight, derived as a Lagrange multiplier.
    """

    def __init__(self, action_t: ActionFunctional,
                 action_tau: ActionFunctional,
                 coupling_weight: Callable = None):
        self.S_t = action_t
        self.S_tau = action_tau
        self.alpha = coupling_weight or (lambda chi: 1.0 / (1.0 + np.exp(-10 * (chi - 0.5))))

    def total_action(self, psi_field: np.ndarray) -> float:
        chi = self.S_t.chi_from_residual(psi_field)
        a = self.alpha(chi)
        return ((1 - a) * self.S_t.action(psi_field) +
                a * self.S_tau.action(psi_field))

    def handoff_parameter(self, psi_field: np.ndarray) -> float:
        """The handoff α(χ) for current field state."""
        chi = self.S_t.chi_from_residual(psi_field)
        return self.alpha(chi)
