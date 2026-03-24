"""
emergence.py — Algebraic Emergence (Defs 7-14, Theorem 1, Corollaries 2-3)

Clifford is NOT assumed. It EMERGES when:
  1. N_rig(u) < ∞  (reading modes collapse to finite classes)
  2. Rig_N(ε; u) → 0  (rigidity functional vanishes)

Then: γ_μ γ_ν + γ_ν γ_μ = 2g_{μν} Id   (Theorem 1)

This module implements the quantization radius, rigid rank,
closure defect, rigidity functional, and algebraic detection.
"""
import numpy as np
from typing import Callable, Optional, List, Tuple
from scipy.ndimage import gaussian_filter1d


# ── Definition 7: Reading Modes as Integral Operators ─────────

class ReadingMode:
    """
    Def 7. L_ξ^ε: integral operator with smooth kernel K_ξ^ε(s,r).

    Parameterized by ξ = (η, σ, δ):
      - η: focus center
      - σ: scale
      - δ: penumbra width

    These are NOT ontological layers. They are readers.
    """

    def __init__(self, eta: float, sigma: float, delta: float):
        self.eta = eta
        self.sigma = sigma
        self.delta = delta

    def kernel(self, s: np.ndarray, r: np.ndarray,
               epsilon: float) -> np.ndarray:
        """
        K_ξ^ε(s,r) — smooth kernel.
        Localized around η with width controlled by δ and ε.
        """
        # Gaussian kernel localized at η
        spatial = np.exp(-0.5 * ((s[:, None] - self.eta) / self.delta) ** 2)
        coupling = np.exp(-0.5 * ((s[:, None] - r[None, :]) / (epsilon + self.sigma)) ** 2)
        K = spatial * coupling
        # Normalize rows
        row_sums = np.sum(K, axis=1, keepdims=True)
        return K / (row_sums + 1e-30)

    def apply(self, u: np.ndarray, s_grid: np.ndarray,
              epsilon: float) -> np.ndarray:
        """L_ξ^ε[u](s) = ∫ K_ξ^ε(s,r) u(r) dr"""
        K = self.kernel(s_grid, s_grid, epsilon)
        ds = s_grid[1] - s_grid[0] if len(s_grid) > 1 else 1.0
        return K @ u * ds


# ── Definition 9: Quantization Radius ─────────────────────────

def quantization_radius(u: np.ndarray, s_grid: np.ndarray,
                        modes: List[ReadingMode],
                        epsilon: float,
                        N: int,
                        n_candidates: int = 50) -> float:
    """
    Def 9. Δ_N(ε; u) = inf_{v₁,...,v_N} sup_ξ min_j |L_ξ^ε u - v_j|_∞

    How well can N templates approximate ALL readings?
    Small Δ_N = readings cluster into N classes.

    Approximated by k-means-like procedure.
    """
    # Compute all readings
    readings = []
    for mode in modes:
        reading = mode.apply(u, s_grid, epsilon)
        readings.append(reading)
    readings = np.array(readings)  # (n_modes, n_grid)

    if N >= len(modes):
        return 0.0

    # Simple k-means approximation
    from scipy.cluster.vq import kmeans2
    centroids, labels = kmeans2(readings, N, minit='points', iter=20)

    # Compute sup_ξ min_j |reading - centroid_j|_∞
    max_min_dist = 0.0
    for i, reading in enumerate(readings):
        dists = [np.max(np.abs(reading - centroids[j]))
                 for j in range(N)]
        max_min_dist = max(max_min_dist, min(dists))

    return float(max_min_dist)


# ── Definition 10: Rigid Rank ─────────────────────────────────

def rigid_rank(u: np.ndarray, s_grid: np.ndarray,
               modes: List[ReadingMode],
               epsilons: np.ndarray = None,
               max_N: int = 16,
               threshold: float = 0.05) -> Optional[int]:
    """
    Def 10. N_rig(u) = min{N : lim_{ε→0} Δ_N(ε; u) = 0}

    Returns None if no finite rigid rank found within max_N.
    """
    if epsilons is None:
        epsilons = np.array([0.1, 0.05, 0.02, 0.01])

    for N in range(1, max_N + 1):
        # Check if Δ_N decreases toward 0 as ε → 0
        deltas = []
        for eps in epsilons:
            d = quantization_radius(u, s_grid, modes, eps, N)
            deltas.append(d)

        # Check decreasing trend and final value small
        if deltas[-1] < threshold and all(
            deltas[i] >= deltas[i+1] - 1e-6 for i in range(len(deltas)-1)
        ):
            return N

    return None


# ── Definition 12: Symmetrized Closure Defect ─────────────────

def closure_defect(u: np.ndarray, s_grid: np.ndarray,
                   modes: List[ReadingMode],
                   epsilon: float,
                   mu: int, nu: int) -> Tuple[float, float]:
    """
    Def 12. D_{μν}^ε[u; c] and minimizer g_{μν}^ε[u].

    Measures: |L_μ L_ν v + L_ν L_μ v - 2c·v|_∞
    minimized over c ∈ R, for v in the generated family.

    Returns (defect, optimal_c).
    """
    L_mu = modes[mu]
    L_nu = modes[nu]

    # Test on u itself and a few derived vectors
    test_vectors = [u / (np.max(np.abs(u)) + 1e-14)]
    # Add L_μ u, L_ν u as test vectors
    for m in [mu, nu]:
        v = modes[m].apply(u, s_grid, epsilon)
        norm = np.max(np.abs(v)) + 1e-14
        if norm > 1e-10:
            test_vectors.append(v / norm)

    best_defect = np.inf
    best_c = 0.0

    # For each test vector, compute L_μ L_ν v + L_ν L_μ v
    for v in test_vectors:
        L_mu_v = L_mu.apply(v, s_grid, epsilon)
        L_nu_v = L_nu.apply(v, s_grid, epsilon)
        L_mu_L_nu_v = L_mu.apply(L_nu_v, s_grid, epsilon)
        L_nu_L_mu_v = L_nu.apply(L_mu_v, s_grid, epsilon)

        anticomm_v = L_mu_L_nu_v + L_nu_L_mu_v

        # Optimal c: minimize |anticomm_v - 2c·v|_∞
        # Approximate: c ≈ ⟨anticomm_v, v⟩ / (2⟨v,v⟩)
        v_norm_sq = np.sum(v ** 2) + 1e-30
        c_opt = float(np.sum(anticomm_v * v) / (2 * v_norm_sq))

        defect = float(np.max(np.abs(anticomm_v - 2 * c_opt * v)))

        if defect < best_defect:
            best_defect = defect
            best_c = c_opt

    return best_defect, best_c


# ── Definition 13: Rigidity Functional ────────────────────────

def rigidity_functional(u: np.ndarray, s_grid: np.ndarray,
                        modes: List[ReadingMode],
                        epsilon: float, N: int) -> float:
    """
    Def 13. Rig_N(ε; u) = Δ_N(ε; u) + max_{μ,ν} D_{μν}^ε[u]

    First term: quantization (discretization of readings).
    Second term: closure (algebraic compatibility).
    """
    delta_N = quantization_radius(u, s_grid, modes, epsilon, N)

    max_defect = 0.0
    n_modes = min(N, len(modes))
    for mu in range(n_modes):
        for nu in range(mu, n_modes):
            defect, _ = closure_defect(u, s_grid, modes, epsilon, mu, nu)
            max_defect = max(max_defect, defect)

    return delta_N + max_defect


# ── Theorem 1: Recovery of Clifford Relations ─────────────────

def extract_metric(u: np.ndarray, s_grid: np.ndarray,
                   modes: List[ReadingMode],
                   epsilon: float, N: int) -> np.ndarray:
    """
    From Theorem 1: extract the emergent metric g_{μν}.

    If Rig_N → 0, then L_μ L_ν + L_ν L_μ → 2g_{μν}·Id.

    Returns (N × N) symmetric matrix g_{μν}.
    """
    n = min(N, len(modes))
    g = np.zeros((n, n))

    for mu in range(n):
        for nu in range(mu, n):
            _, c_opt = closure_defect(u, s_grid, modes, epsilon, mu, nu)
            g[mu, nu] = c_opt
            g[nu, mu] = c_opt

    return g


def check_clifford_emergence(u: np.ndarray, s_grid: np.ndarray,
                              modes: List[ReadingMode],
                              epsilons: np.ndarray = None,
                              max_N: int = 8) -> dict:
    """
    Full emergence check: Defs 9-14 + Theorem 1.

    Returns comprehensive report:
      - rigid_rank: N_rig(u) or None
      - rigidity_values: Rig_N(ε; u) for each ε
      - emergent_metric: g_{μν} at finest ε
      - signature: signs of diagonal g_{μμ}
      - is_clifford: whether emergence criterion (C7) is approximately met
    """
    if epsilons is None:
        epsilons = np.array([0.1, 0.05, 0.02, 0.01])

    N_rig = rigid_rank(u, s_grid, modes, epsilons, max_N)

    if N_rig is None:
        return {
            "rigid_rank": None,
            "is_clifford": False,
            "note": "No finite rigid rank found",
        }

    # Compute rigidity at each epsilon
    rig_values = []
    for eps in epsilons:
        rig = rigidity_functional(u, s_grid, modes, eps, N_rig)
        rig_values.append(float(rig))

    # Extract metric at finest epsilon
    g = extract_metric(u, s_grid, modes, epsilons[-1], N_rig)

    # Signature (Cor 2): signs of diagonal
    diag = np.diag(g)
    signature = np.sign(diag)

    # Check Cor 3: is there a sub-closure of type Cl(3,0)?
    pos_count = int(np.sum(signature > 0))
    neg_count = int(np.sum(signature < 0))

    return {
        "rigid_rank": N_rig,
        "rigidity_values": rig_values,
        "rigidity_decreasing": all(
            rig_values[i] >= rig_values[i+1] - 1e-6
            for i in range(len(rig_values) - 1)
        ),
        "emergent_metric": g,
        "diagonal": diag.tolist(),
        "signature": signature.tolist(),
        "positive_generators": pos_count,
        "negative_generators": neg_count,
        "is_clifford": (rig_values[-1] < 0.1 and N_rig is not None),
        "possible_algebra": f"Cl({pos_count},{neg_count})" if N_rig else None,
    }
