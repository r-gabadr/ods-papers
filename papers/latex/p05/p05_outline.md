# P05 — Multiscale Extension of the CT Runtime: Contractive Families and Fractal Structure

**Status**: Outline draft — pending empirical test of contractivity
**Depends on**: P01 (runtime), P02 (regimes), P03 (observability)
**Target**: Zenodo preprint

---

## Core question

Given the family of Clifford maps

    Φ_{α,ℓ}(Ψ) = r_{α,ℓ} · R_{α,ℓ} Ψ R̃_{α,ℓ} + Δ_{α,ℓ}(Ψ)

where R is a unit rotor (isometry), r < 1 is a scaling factor, and Δ is a
nonlinear perturbation with Lipschitz constant L_{α,ℓ}:

**Are the individual maps contractive (r + L < 1)?**

- If YES → Hutchinson operator H_ℓ(K) = ∪ Φ_{α,ℓ}(K) has a unique compact
  attractor K*_ℓ in Hausdorff metric → **true fractal extension**
- If NO (only contractive in aggregate) → T_ℓ(Ψ) = Σ w_α Φ_α(Ψ) has a unique
  fixed point A*_ℓ by Banach → **multiscale contractive extension**

Both are mathematically legitimate. The difference is structural:
- Fractal: the attractor is a *set* with self-similar structure
- Contractive: the attractor is a *point* (single field configuration)

---

## Variant A: Multiscale Contractive Extension (minimal, safe)

**Theorem**: If L_ℓ = Σ |w_{α,ℓ}|(r_{α,ℓ} + L_{α,ℓ}) < 1, then T_ℓ is
contractive and has a unique fixed point A*_ℓ = T_ℓ(A*_ℓ).

No Hausdorff metric needed. No "fractal" claim. Just Banach on a weighted
average of rotor-scaled maps.

**Pros**: Honest, simple, easy to verify empirically.
**Cons**: Less exciting. "Contractive extension" doesn't capture the
self-similar structure that the maps visually produce.

---

## Variant B: Fractal Extension via Hutchinson (strong, beautiful)

**Definition** (Set-valued fractal slice operator):

    H_ℓ(K) = ∪_{α=1}^{m_ℓ} Φ_{α,ℓ}(K),    K ∈ K(X)

where K(X) is the space of non-empty compact subsets of X with Hausdorff
distance d_H.

**Theorem** (Hutchinson '81): If every Φ_{α,ℓ} is a contraction with ratio
c_{α,ℓ} < 1, then H_ℓ is contractive in (K(X), d_H) with ratio
max_α c_{α,ℓ}, and there exists a unique compact invariant set K*_ℓ such that

    K*_ℓ = H_ℓ(K*_ℓ)

**Corollary**: The averaged operator T_ℓ produces a representative point
(barycenter) of K*_ℓ. The fractal structure of K*_ℓ is the "true" attractor;
T_ℓ(A*) is its compressed operational summary.

**Pros**: Mathematically beautiful. Separates fractal structure from
operational compression. Justifies "fractal" in the title.
**Cons**: Requires proving individual contractivity of each Φ_{α,ℓ}.

---

## Key requirement for Variant B

Each Φ_{α,ℓ} must satisfy:

    ||Φ_{α,ℓ}(Ψ₁) - Φ_{α,ℓ}(Ψ₂)|| ≤ c_{α,ℓ} · ||Ψ₁ - Ψ₂||

with c_{α,ℓ} < 1 for ALL α, ℓ.

Since R·Ψ·R̃ is an isometry, we need:

    c_{α,ℓ} = r_{α,ℓ} + Lip(Δ_{α,ℓ}) < 1

This is the empirical question that the test must answer.

---

## Additional elements (both variants)

### Rotor-guided mutation

    M_{j→k}^{(ℓ)}(Ψ) = U_{j→k}^{(ℓ)} Ψ Ũ_{j→k}^{(ℓ)}

Since U is a unit rotor, mutation preserves the Clifford norm and acts by
isometry before acceptance filtering. This is a geometric fact, not a claim.

### Acceptance criterion

A mutated candidate is accepted if it improves pinning coherence, spectral
connectivity (λ₂), or focus score. This is a greedy filter, not an optimization
guarantee.

### Multiscale descriptor (truncated)

    D_d(Ψ) = Σ_{k=0}^{d} w_k · obs(T^k(Ψ))

Computable in O(d) time for fixed observable cost. NOT "constant time" — that
was incorrect in the previous draft.

---

## Tests needed

See `_claude/tests/paper_claims/test_p05_claims.py` and
`_claude/experiments/contractivity_test.py`.

---

## Figures (planned)

1. **Contractivity scatter**: ||Φ(Ψ₁)-Φ(Ψ₂)|| vs ||Ψ₁-Ψ₂|| for individual maps
2. **Hausdorff convergence**: d_H(H^n(K₀), K*) over iterations (if Variant B works)
3. **Fixed point convergence**: ||T^n(Ψ₀) - A*|| over iterations
4. **Multiscale descriptor**: D_d values at different depths d
