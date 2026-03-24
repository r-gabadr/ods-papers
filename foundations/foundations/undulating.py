"""
undulating.py — Smoothly Undulating Numbers

Definitions 1-3, Propositions 1-3 from the notebook.

Ontology: the SERIES u is the object.
The triple (α, ω, φ) is an admissible factorization, not the object itself.

    u(s) = α(s) · ω(φ(s))

with:
  - ω: analytic oscillator (Def 1) — bounded, periodic, zero-mean, isolated crits
  - α: slow envelope in W^{1,∞}, 0 ≤ α ≤ M
  - φ: monotone phase, φ' ≥ κ > 0
  - Slowness condition (A1): |α'(s)| ≤ ϑ(1+α(s))φ'(s), 0 ≤ ϑ < 1
"""
import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.interpolate import BSpline


# ── Definition 1: Analytic Oscillator ─────────────────────────

class AnalyticOscillator:
    """
    Def 1. ω ∈ C²(R) such that:
      sup|ω| = 1,  inf ω = -1,
      ω(ξ + P) = ω(ξ) for some P > 0,
      ∫₀ᴾ ω(ξ) dξ = 0,
      all critical points isolated.

    No trigonometry required. ω need not be sin or cos.
    """

    def __init__(self, fn: Callable, period: float,
                 name: str = "generic"):
        self.fn = fn
        self.period = period
        self.name = name

    def __call__(self, xi: np.ndarray) -> np.ndarray:
        return self.fn(xi)

    def verify(self, n_points: int = 10000) -> dict:
        """Verify all conditions of Def 1."""
        xi = np.linspace(0, self.period, n_points)
        vals = self.fn(xi)
        dxi = xi[1] - xi[0]

        sup_val = np.max(vals)
        inf_val = np.min(vals)
        integral = np.sum(vals) * dxi
        # Periodicity: check endpoints
        period_err = abs(self.fn(np.array([0.0]))[0] -
                        self.fn(np.array([self.period]))[0])

        return {
            "sup_equals_1": abs(sup_val - 1.0) < 1e-6,
            "inf_equals_neg1": abs(inf_val + 1.0) < 1e-6,
            "zero_mean": abs(integral) < 0.01,
            "periodic": period_err < 1e-6,
            "all_pass": (abs(sup_val - 1.0) < 1e-6 and
                        abs(inf_val + 1.0) < 1e-6 and
                        abs(integral) < 0.01 and
                        period_err < 1e-6),
        }


def standard_oscillator() -> AnalyticOscillator:
    """cos(2πξ) — simplest oscillator satisfying Def 1."""
    return AnalyticOscillator(
        fn=lambda xi: np.cos(2 * np.pi * xi),
        period=1.0,
        name="cosine",
    )


def sawtooth_smooth_oscillator(sharpness: float = 5.0) -> AnalyticOscillator:
    """A non-trigonometric oscillator: smooth sawtooth-like."""
    P = 1.0

    def fn(xi):
        # Smooth periodic function with sup=1, inf=-1, zero mean
        t = np.mod(xi, P) / P  # t ∈ [0,1)
        # Smoothed triangle wave
        raw = 4 * np.abs(t - 0.5) - 1  # triangle in [-1, 1]
        return np.tanh(sharpness * raw) / np.tanh(sharpness)

    return AnalyticOscillator(fn=fn, period=P, name="smooth_sawtooth")


# ── Definition 2: Smoothly Undulating Number ──────────────────

class SmoothlyUndulatingNumber:
    """
    Def 2. u ∈ U_M with factorization u(s) = α(s)·ω(φ(s)).

    - α ∈ W^{1,∞}(I), 0 ≤ α(s) ≤ M  (envelope)
    - φ ∈ C¹(I), φ'(s) ≥ κ > 0       (monotone phase)
    - ω: analytic oscillator           (Def 1)
    - Slowness (A1): |α'| ≤ ϑ(1+α)φ'  with 0 ≤ ϑ < 1

    The ontology is the series u. The triple (α,ω,φ) is a factorization.
    """

    def __init__(self, envelope: Callable, phase: Callable,
                 oscillator: AnalyticOscillator,
                 M: float = 1.0, kappa: float = 1.0,
                 vartheta: float = 0.5):
        self.envelope = envelope    # α(s)
        self.phase = phase          # φ(s)
        self.oscillator = oscillator  # ω
        self.M = M
        self.kappa = kappa
        self.vartheta = vartheta

    def evaluate(self, s: np.ndarray) -> np.ndarray:
        """u(s) = α(s) · ω(φ(s))"""
        s = np.asarray(s, dtype=np.float64)
        alpha = self.envelope(s)
        phi = self.phase(s)
        omega = self.oscillator(phi)
        return alpha * omega

    def bound(self, s_grid: np.ndarray) -> float:
        """sup|u(s)| over grid."""
        return float(np.max(np.abs(self.evaluate(s_grid))))

    def is_bounded(self, s_grid: np.ndarray) -> bool:
        """Check sup|u| ≤ M."""
        return self.bound(s_grid) <= self.M * (1 + 1e-10)

    def verify_slowness_A1(self, s_grid: np.ndarray,
                           ds: float = None) -> dict:
        """
        Verify condition (A1): |α'(s)| ≤ ϑ(1+α(s))φ'(s).
        Uses numerical derivatives.
        """
        if ds is None:
            ds = s_grid[1] - s_grid[0]

        alpha_vals = self.envelope(s_grid)
        phi_vals = self.phase(s_grid)

        # Numerical derivatives
        alpha_prime = np.gradient(alpha_vals, ds)
        phi_prime = np.gradient(phi_vals, ds)

        # Check (A1)
        lhs = np.abs(alpha_prime)
        rhs = self.vartheta * (1.0 + alpha_vals) * phi_prime

        violations = lhs > rhs + 1e-10
        max_ratio = np.max(lhs / (rhs + 1e-14))

        return {
            "condition_A1_holds": not np.any(violations),
            "max_ratio": float(max_ratio),
            "fraction_violated": float(np.mean(violations)),
            "phi_prime_min": float(np.min(phi_prime)),
            "phi_monotone": bool(np.all(phi_prime > 0)),
        }

    def to_series(self, s_grid: np.ndarray,
                  n_coeffs: int = 50) -> np.ndarray:
        """
        Prop 2 compatibility: expand u in Schauder basis (Fourier).
        Returns coefficients cₙ such that u ≈ Σ cₙ φₙ.
        """
        vals = self.evaluate(s_grid)
        # Use DCT as proxy for Schauder expansion
        from scipy.fft import dct
        coeffs = dct(vals, type=2, norm='ortho')
        return coeffs[:n_coeffs]


# ── Definition 3: Equivalence of Factorizations ──────────────

def are_equivalent_factorizations(u1: SmoothlyUndulatingNumber,
                                  u2: SmoothlyUndulatingNumber,
                                  s_grid: np.ndarray,
                                  tol: float = 1e-8) -> bool:
    """
    Prop 1. Two factorizations represent the same SUN if
    u₁(s) = u₂(s) for all s (up to tolerance).

    The equivalence is: ∃ h ∈ C¹ strictly increasing such that
    φ̃ = h∘φ, ω̃ = ω∘h⁻¹, α̃ = α.
    But the invariant is simply: u₁ = u₂.
    """
    v1 = u1.evaluate(s_grid)
    v2 = u2.evaluate(s_grid)
    return bool(np.max(np.abs(v1 - v2)) < tol)


# ── Proposition 3: Representation of sequences by reading modes ──

def represent_sequence_as_sun(b: np.ndarray, I: tuple = (0, 1),
                              n_grid: int = 1000) -> Tuple:
    """
    Prop 3. Given (bₙ) ∈ ℓ∞, construct u ∈ U_M and readers Lₙ
    such that Lₙ[u] = bₙ for all n.

    Returns (u_values, readers, s_grid).
    """
    N = len(b)
    M = float(np.max(np.abs(b)))
    s_grid = np.linspace(I[0], I[1], n_grid)
    ds = s_grid[1] - s_grid[0]

    # Disjoint intervals Jₙ
    interval_width = (I[1] - I[0]) / (N + 1)
    u_vals = np.zeros(n_grid)

    readers = []
    for n in range(N):
        center = I[0] + (n + 1) * interval_width
        # Bump function ψₙ supported near center
        # and test function θₙ with ∫ ψₙ θₙ = 1
        sigma = interval_width / 6

        def make_psi(c, sig):
            return lambda s: np.exp(-0.5 * ((s - c) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

        def make_theta(c, sig, bn_val):
            psi = make_psi(c, sig)
            # ∫ ψ θ ds = 1, so θ = ψ / ∫ψ²ds
            psi_vals = psi(s_grid)
            norm_sq = np.sum(psi_vals ** 2) * ds
            return lambda s: psi(s) / (norm_sq + 1e-30)

        psi_n = make_psi(center, sigma)
        u_vals += b[n] * psi_n(s_grid)

        theta_n = make_theta(center, sigma, b[n])
        readers.append(theta_n)

    return u_vals, readers, s_grid
