"""
ct_embeddings.py — Embedding and projection between Cl(p,q) levels.

Embedding:  Cl(p,q) ↪ Cl(p',q')  for p≤p', q≤q'
  ι(Ψ) embeds the lower algebra into the higher one.

Projection: Cl(p',q') → Cl(p,q)
  π(Ψ') projects by tracing out extra dimensions.

These are the morphisms of the CT signature ladder.
"""
import numpy as np
from typing import Tuple
from ct_algebra import CliffordAlgebra


def embed(psi: np.ndarray, source: CliffordAlgebra,
          target: CliffordAlgebra) -> np.ndarray:
    """
    Embed multivector from Cl(p,q) into Cl(p',q').

    Strategy: match blades by generator indices.
    A blade e_{i₁...i_k} in source maps to the same-indexed blade in target.
    Extra dimensions get coefficient 0.

    Parameters:
        psi: (source.dim,) coefficient vector
        source: the smaller algebra
        target: the larger algebra

    Returns:
        (target.dim,) coefficient vector in the larger algebra
    """
    assert source.n <= target.n, \
        f"Cannot embed Cl({source.p},{source.q}) into Cl({target.p},{target.q})"

    result = np.zeros(target.dim, dtype=np.complex128)

    # Build index mapping: source blade indices → target blade indices
    source_blades = source._blade_indices
    target_blades = target._blade_indices

    # Create lookup for target
    target_lookup = {idx: i for i, idx in enumerate(target_blades)}

    for src_i, src_idx in enumerate(source_blades):
        if src_idx in target_lookup:
            result[target_lookup[src_idx]] = psi[src_i]

    return result


def project(psi: np.ndarray, source: CliffordAlgebra,
            target: CliffordAlgebra) -> np.ndarray:
    """
    Project multivector from Cl(p',q') down to Cl(p,q).

    Strategy: keep only blades whose generator indices are all < target.n.
    Discard blades involving higher generators.

    Parameters:
        psi: (source.dim,) coefficient vector in larger algebra
        source: the larger algebra
        target: the smaller algebra

    Returns:
        (target.dim,) coefficient vector in the smaller algebra
    """
    assert target.n <= source.n, \
        f"Cannot project Cl({source.p},{source.q}) to Cl({target.p},{target.q})"

    result = np.zeros(target.dim, dtype=np.complex128)

    source_blades = source._blade_indices
    target_blades = target._blade_indices
    target_lookup = {idx: i for i, idx in enumerate(target_blades)}

    for src_i, src_idx in enumerate(source_blades):
        # Check if all generator indices are within target range
        if all(g < target.n for g in src_idx):
            if src_idx in target_lookup:
                result[target_lookup[src_idx]] = psi[src_i]

    return result


def embedding_energy_ratio(psi: np.ndarray, source: CliffordAlgebra,
                           target: CliffordAlgebra) -> float:
    """
    How much of the embedded state's energy lives in the source subspace?

    Returns ratio ∈ [0, 1]. Ratio = 1 means no energy leaked to extra dims.
    """
    embedded = embed(psi, source, target)
    projected_back = project(embedded, target, source)

    norm_original = source.norm(psi)
    norm_projected = source.norm(projected_back)

    if norm_original < 1e-14:
        return 1.0
    return (norm_projected / norm_original) ** 2


class SignatureLadder:
    """
    The CT signature ladder: an ordered sequence of Cl(p_k, q_k).

    Provides:
    - Algebra at each level
    - Embedding/projection between adjacent levels
    - Full CT state as direct sum
    """

    def __init__(self, signatures: list):
        """
        Parameters:
            signatures: list of (p, q) tuples, ordered by level
        """
        self.signatures = signatures
        self.algebras = [CliffordAlgebra(p, q) for p, q in signatures]
        self.n_levels = len(signatures)

    def embed_up(self, psi: np.ndarray, level: int) -> np.ndarray:
        """Embed from level to level+1."""
        assert level < self.n_levels - 1
        return embed(psi, self.algebras[level], self.algebras[level + 1])

    def project_down(self, psi: np.ndarray, level: int) -> np.ndarray:
        """Project from level to level-1."""
        assert level > 0
        return project(psi, self.algebras[level], self.algebras[level - 1])

    def full_state_dim(self) -> int:
        """Total dimension of the direct sum Ψ_CT."""
        return sum(a.dim for a in self.algebras)

    def pack(self, states: list) -> np.ndarray:
        """Pack list of per-level states into single Ψ_CT vector."""
        return np.concatenate([np.asarray(s, dtype=np.complex128)
                               for s in states])

    def unpack(self, psi_ct: np.ndarray) -> list:
        """Unpack Ψ_CT into per-level states."""
        states = []
        offset = 0
        for a in self.algebras:
            states.append(psi_ct[offset:offset + a.dim])
            offset += a.dim
        return states

    def level_energy(self, psi_ct: np.ndarray) -> list:
        """Energy at each level of the ladder."""
        states = self.unpack(psi_ct)
        return [a.norm_squared(s) for a, s in zip(self.algebras, states)]

    def __repr__(self) -> str:
        sigs = " → ".join(f"Cl({p},{q})" for p, q in self.signatures)
        return f"SignatureLadder({sigs})"
