"""
flow.py — Complexity Flow ∂_λ u = G_λ[u]

Definitions 4-6, Propositions 4-5, Corollary 1.

The flow equation:
    ∂_λ u = -α(λ)·A[u] + p_λ[u]·(1 - u²/M²)·R[u]

where:
    S_c = exp(ε_c ∂_ss)   coarse smoothing (heat semigroup)
    S_f = exp(ε_f ∂_ss)   fine smoothing
    A[u] = u - S_c u       defect from coarse scale
    R[u] = S_f u - S_c u   fine return between resolutions
    Q[u] = |A[u]|          complexity density
    Π[u] = |R[u]|          production proxy
    p_λ[u] = β(λ) Π[u] / (Π[u] + μ(λ)Q[u] + τ)   permeability

Prop 4: Global existence + boundedness preservation.
Prop 5: Exact complexity production identity.
"""
import numpy as np
from typing import Callable, Optional, Tuple
from scipy.ndimage import gaussian_filter1d


class HeatSmoothing:
    """
    S_ε = exp(ε ∂_ss) — heat semigroup on a 1D grid.
    Implemented as Gaussian smoothing with σ = √(2ε).
    """

    def __init__(self, epsilon: float):
        assert epsilon > 0, "ε must be positive"
        self.epsilon = epsilon
        self.sigma = np.sqrt(2 * epsilon)

    def apply(self, u: np.ndarray) -> np.ndarray:
        """S_ε[u] via Gaussian filter."""
        # sigma in grid units — assume unit spacing, scale by sigma
        return gaussian_filter1d(u, sigma=self.sigma, mode='wrap')


class ComplexityFlow:
    """
    The complexity flow ∂_λ u = G_λ[u] from Def 5.

    Parameters:
        M: global bound
        eps_c: coarse smoothing scale (ε_c)
        eps_f: fine smoothing scale (ε_f), must be < eps_c
        alpha_fn: α(λ) — absorption rate
        beta_fn: β(λ) — permeability ceiling
        mu_fn: μ(λ) — dissipation in permeability
        tau: τ > 0 — regularization in permeability denominator
    """

    def __init__(self, M: float = 1.0,
                 eps_c: float = 0.1, eps_f: float = 0.01,
                 alpha_fn: Optional[Callable] = None,
                 beta_fn: Optional[Callable] = None,
                 mu_fn: Optional[Callable] = None,
                 tau: float = 0.01):
        assert eps_f < eps_c, "ε_f must be < ε_c"
        self.M = M
        self.S_c = HeatSmoothing(eps_c)
        self.S_f = HeatSmoothing(eps_f)
        self.alpha = alpha_fn or (lambda lam: 1.0)
        self.beta = beta_fn or (lambda lam: 1.0)
        self.mu = mu_fn or (lambda lam: 1.0)
        self.tau = tau

    def defect_A(self, u: np.ndarray) -> np.ndarray:
        """A[u] = u - S_c u  (defect from coarse scale)."""
        return u - self.S_c.apply(u)

    def return_R(self, u: np.ndarray) -> np.ndarray:
        """R[u] = S_f u - S_c u  (fine return between resolutions)."""
        return self.S_f.apply(u) - self.S_c.apply(u)

    def complexity_Q(self, u: np.ndarray) -> np.ndarray:
        """Q[u] = |A[u]|  (complexity density)."""
        return np.abs(self.defect_A(u))

    def production_Pi(self, u: np.ndarray) -> np.ndarray:
        """Π[u] = |R[u]|  (production proxy)."""
        return np.abs(self.return_R(u))

    def permeability(self, u: np.ndarray, lam: float) -> np.ndarray:
        """
        Def 4. p_λ[u] = β(λ) · Π[u] / (Π[u] + μ(λ)·Q[u] + τ)

        Satisfies: 0 ≤ p_λ[u] ≤ β(λ).
        Rupture when: Π[u] > μ(λ)·Q[u] + τ  (then p > β/2).
        """
        Pi = self.production_Pi(u)
        Q = self.complexity_Q(u)
        return self.beta(lam) * Pi / (Pi + self.mu(lam) * Q + self.tau)

    def G(self, u: np.ndarray, lam: float) -> np.ndarray:
        """
        Def 5. G_λ[u] = -α(λ)·A[u] + p_λ[u]·(1-u²/M²)·R[u]

        The flow equation ∂_λ u = G_λ[u].
        """
        A = self.defect_A(u)
        R = self.return_R(u)
        p = self.permeability(u, lam)
        saturation = 1.0 - (u / self.M) ** 2

        return -self.alpha(lam) * A + p * saturation * R

    def step(self, u: np.ndarray, lam: float,
             d_lam: float = 0.01) -> np.ndarray:
        """One Euler step of the flow."""
        return u + d_lam * self.G(u, lam)

    def evolve(self, u0: np.ndarray, lam_max: float = 1.0,
               d_lam: float = 0.01) -> Tuple[np.ndarray, list]:
        """
        Evolve from λ=0 to λ=lam_max.

        Prop 4 guarantees: |u(λ)|_∞ ≤ M for all λ.

        Returns (u_final, trajectory_of_sup_norms).
        """
        u = u0.copy()
        lam = 0.0
        sup_norms = [float(np.max(np.abs(u)))]

        while lam < lam_max:
            u = self.step(u, lam, d_lam)
            lam += d_lam
            sup_norms.append(float(np.max(np.abs(u))))

        return u, sup_norms

    def complexity_energy(self, u: np.ndarray) -> float:
        """
        E[u] = ½ ∫ u(s) R[u](s) ds

        Prop 5: the energy whose production identity we track.
        """
        R = self.return_R(u)
        return 0.5 * float(np.sum(u * R)) / len(u)

    def rupture_set(self, u: np.ndarray, lam: float) -> np.ndarray:
        """
        Cor 1. Ω_rup(λ) = {s : Π[u] > μ(λ)·Q[u] + τ}

        Returns boolean mask.
        """
        Pi = self.production_Pi(u)
        Q = self.complexity_Q(u)
        return Pi > self.mu(lam) * Q + self.tau

    def is_stationary(self, u: np.ndarray, lam: float,
                      tol: float = 1e-6) -> bool:
        """
        Def 6. Check if u is close to a limit attractor:
        G_∞[u*] ≈ 0.
        """
        residual = self.G(u, lam)
        return float(np.max(np.abs(residual))) < tol
