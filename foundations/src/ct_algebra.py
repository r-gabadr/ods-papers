"""
ct_algebra.py — Generic Clifford Algebra Cl(p,q) constructor.

Builds gamma matrices, blade bases, product tables, grade projectors,
reversion, and conjugation for ANY signature (p,q) via recursive
tensor product construction.

Mathematical foundation:
  γ_μ γ_ν + γ_ν γ_μ = 2 g_{μν} I
  g = diag(+1,...,+1, -1,...,-1)  (p pluses, q minuses)
"""
import numpy as np
from functools import lru_cache
from itertools import combinations
from typing import Tuple, List, Dict, Optional

# Pauli matrices (building blocks)
sigma_1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def build_gamma_matrices(p: int, q: int) -> List[np.ndarray]:
    """
    Build gamma matrices for Cl(p,q) via recursive tensor product.

    Algorithm (standard induction):
      - Start from Cl(0,0) = R (1×1 identity)
      - To add a positive generator: γ_new^(j) = γ_old^(j) ⊗ σ₃,
        γ_new^(n+1) = I ⊗ σ₁
      - To add a negative generator: γ_new^(j) = γ_old^(j) ⊗ σ₃,
        γ_new^(n+1) = I ⊗ σ₂

    Returns list of (2^ceil(n/2) × 2^ceil(n/2)) complex matrices.
    """
    n = p + q
    if n == 0:
        return []

    gammas: List[np.ndarray] = []

    # Build generators one by one
    for i in range(n):
        if i == 0:
            # First generator
            if i < p:
                gammas = [sigma_1]  # positive: σ₁² = +I
            else:
                gammas = [sigma_2]  # negative: (iσ₂)... we use σ₂ → σ₂²=-I? No.
                # σ₁² = I, σ₂² = I, σ₃² = I (all Pauli square to I)
                # For negative signature we need γ² = -I
                # Use iσ₁ → (iσ₁)² = -I ✓
                gammas = [1j * sigma_1]
        else:
            # Extend existing gammas
            dim = gammas[0].shape[0]
            I_old = np.eye(dim, dtype=np.complex128)

            new_gammas = []
            for g in gammas:
                new_gammas.append(np.kron(g, sigma_3))

            if i < p:
                # Positive generator: squares to +I
                new_gammas.append(np.kron(I_old, sigma_1))
            else:
                # Negative generator: squares to -I
                new_gammas.append(np.kron(I_old, 1j * sigma_1))

            gammas = new_gammas

    return gammas


class CliffordAlgebra:
    """
    Complete Clifford algebra Cl(p,q) with all operations.

    Attributes:
        p, q: signature
        n: dimension of vector space (p+q)
        dim: dimension of algebra (2^n)
        mat_dim: dimension of matrix representation
        gammas: list of n gamma matrices
        metric: diagonal metric tensor g_{μν}
    """

    def __init__(self, p: int, q: int):
        self.p = p
        self.q = q
        self.n = p + q
        self.dim = 2 ** self.n
        self.gammas = build_gamma_matrices(p, q)
        self.mat_dim = self.gammas[0].shape[0] if self.gammas else 1
        self.metric = np.diag(
            [1.0] * p + [-1.0] * q
        )

        # Build blade basis
        self._blade_indices = self._compute_blade_indices()
        self._blade_matrices = self._compute_blade_matrices()
        self._product_table = None  # lazy

    def _compute_blade_indices(self) -> List[Tuple[int, ...]]:
        """All multi-indices for blades, ordered by grade."""
        indices = [()]  # scalar
        for k in range(1, self.n + 1):
            for combo in combinations(range(self.n), k):
                indices.append(combo)
        return indices

    def _compute_blade_matrices(self) -> List[np.ndarray]:
        """Compute matrix representation of each blade."""
        I_mat = np.eye(self.mat_dim, dtype=np.complex128)
        matrices = [I_mat]  # scalar blade = identity

        for k in range(1, self.n + 1):
            for combo in combinations(range(self.n), k):
                mat = I_mat.copy()
                for idx in combo:
                    mat = mat @ self.gammas[idx]
                matrices.append(mat)

        return matrices

    def blade_basis(self) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
        """Return list of (multi-index, matrix) for all 2^n blades."""
        return list(zip(self._blade_indices, self._blade_matrices))

    def grade(self, k: int) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
        """Return all blades of grade k."""
        return [(idx, mat) for idx, mat in self.blade_basis()
                if len(idx) == k]

    def grade_projector(self, k: int) -> np.ndarray:
        """
        Projector onto grade-k subspace.
        Acts on the coefficient vector (not on matrices).
        Returns a (dim × dim) diagonal-like matrix.
        """
        proj = np.zeros(self.dim)
        for i, idx in enumerate(self._blade_indices):
            if len(idx) == k:
                proj[i] = 1.0
        return np.diag(proj)

    def multivector_to_matrix(self, coeffs: np.ndarray) -> np.ndarray:
        """Convert coefficient vector to matrix representation."""
        assert len(coeffs) == self.dim, f"Expected {self.dim} coeffs, got {len(coeffs)}"
        result = np.zeros((self.mat_dim, self.mat_dim), dtype=np.complex128)
        for c, mat in zip(coeffs, self._blade_matrices):
            result += c * mat
        return result

    def matrix_to_multivector(self, M: np.ndarray) -> np.ndarray:
        """Extract coefficient vector from matrix representation."""
        coeffs = np.zeros(self.dim, dtype=np.complex128)
        for i, mat in enumerate(self._blade_matrices):
            # Use trace orthogonality: c_I = tr(Γ_I† M) / tr(Γ_I† Γ_I)
            denom = np.trace(mat.conj().T @ mat).real
            if abs(denom) > 1e-14:
                coeffs[i] = np.trace(mat.conj().T @ M) / denom
        return coeffs

    def geometric_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geometric product of two multivectors (coefficient vectors)."""
        Ma = self.multivector_to_matrix(a)
        Mb = self.multivector_to_matrix(b)
        return self.matrix_to_multivector(Ma @ Mb)

    def inner_product(self, a: np.ndarray, b: np.ndarray) -> complex:
        """Scalar part of geometric product = Clifford inner product."""
        prod = self.geometric_product(a, b)
        return prod[0]  # scalar component

    def outer_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Outer (wedge) product — highest-grade part of product."""
        # For vectors: a∧b = ½(ab - ba)
        Ma = self.multivector_to_matrix(a)
        Mb = self.multivector_to_matrix(b)
        wedge_mat = 0.5 * (Ma @ Mb - Mb @ Ma)
        return self.matrix_to_multivector(wedge_mat)

    def left_contraction(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Left contraction a⌋b.
        For grade-r a and grade-s b: grade of result is s-r (if s≥r, else 0).
        """
        result = np.zeros(self.dim, dtype=np.complex128)
        for i, (idx_a, mat_a) in enumerate(self.blade_basis()):
            for j, (idx_b, mat_b) in enumerate(self.blade_basis()):
                r = len(idx_a)
                s = len(idx_b)
                if s >= r:
                    # Component of a⌋b at grade (s-r)
                    prod = self.matrix_to_multivector(
                        mat_a.conj().T @ mat_b  # using adjoint for contraction
                    )
                    for k, idx_c in enumerate(self._blade_indices):
                        if len(idx_c) == s - r:
                            result[k] += a[i] * b[j] * prod[k]
        return result

    def reversion(self, a: np.ndarray) -> np.ndarray:
        """
        Reversion (tilde): reverses order of vectors in each blade.
        ã_k = (-1)^{k(k-1)/2} a_k  for grade-k component.
        """
        result = np.zeros_like(a)
        for i, idx in enumerate(self._blade_indices):
            k = len(idx)
            sign = (-1) ** (k * (k - 1) // 2)
            result[i] = sign * a[i]
        return result

    def conjugation(self, a: np.ndarray) -> np.ndarray:
        """
        Clifford conjugation: combines reversion and grade involution.
        ā_k = (-1)^{k(k+1)/2} a_k
        """
        result = np.zeros_like(a)
        for i, idx in enumerate(self._blade_indices):
            k = len(idx)
            sign = (-1) ** (k * (k + 1) // 2)
            result[i] = sign * a[i]
        return result

    def grade_involution(self, a: np.ndarray) -> np.ndarray:
        """
        Grade involution (hat): â_k = (-1)^k a_k
        """
        result = np.zeros_like(a)
        for i, idx in enumerate(self._blade_indices):
            k = len(idx)
            result[i] = ((-1) ** k) * a[i]
        return result

    def norm_squared(self, a: np.ndarray) -> float:
        """⟨a ã⟩_0 — scalar part of a times its reverse."""
        rev = self.reversion(a)
        return self.inner_product(a, rev).real

    def norm(self, a: np.ndarray) -> float:
        """√|⟨a ã⟩_0|"""
        ns = self.norm_squared(a)
        return np.sqrt(abs(ns))

    def verify_clifford_relations(self, tol: float = 1e-12) -> bool:
        """Verify γ_μ γ_ν + γ_ν γ_μ = 2 g_{μν} I for all μ,ν."""
        I_mat = np.eye(self.mat_dim, dtype=np.complex128)
        for mu in range(self.n):
            for nu in range(self.n):
                anticomm = self.gammas[mu] @ self.gammas[nu] + \
                           self.gammas[nu] @ self.gammas[mu]
                expected = 2 * self.metric[mu, nu] * I_mat
                if np.max(np.abs(anticomm - expected)) > tol:
                    return False
        return True

    def __repr__(self) -> str:
        return (f"CliffordAlgebra(p={self.p}, q={self.q}, "
                f"dim={self.dim}, mat_dim={self.mat_dim})")
