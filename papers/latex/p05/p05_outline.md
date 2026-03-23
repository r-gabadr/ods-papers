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

## Overnight results (2026-03-23)

### Contractivity: ALL maps individually contractive
- rotor_r090: c_max = 0.900 (exact, all grids)
- rotor_delta_085_010: c_max ≈ 0.910 (all grids, 100%)
- rotor_delta_085_020: c_max ≈ 0.970 (all grids, 100%)
- gamma_flow (dt=0.01-0.02): c_max ≈ 0.9997 (100%)
- defect_stab (dt=0.001-0.01): c_max ≈ 0.9992 (100%)
- laplacian (α=0.005-0.02): c_max ≈ 0.48-0.82 (100%)

### Hutchinson convergence: CONDITIONAL
| Config | r | δ | Converges? | Final ratio |
|--------|---|---|-----------|-------------|
| conservative (r=0.70, δ=0.10) | ✅ | | 0.79 |
| moderate (r=0.80, δ=0.15) | ✅ slow | | 0.95 |
| aggressive (r=0.85, δ=0.20) | ❌ saturates ~44 | | ≈1.0 |
| boundary (r=0.90, δ=0.15) | ❌ saturates ~55 | | ≈1.0 |

### Fixed point uniqueness: CONFIRMED
20 random starts × 80 iterations → pairwise distance = 0.000000

### Texture stability
Single-scale 3.1-3.5× more stable than multiscale under perturbation.

### Regime separability
Cohen's d: focus=15, entropy=123, tension=50, d_eff=415 (Thomas vs Spinor).

---

## New pieces from review (2026-03-24)

### Lifting-evolution-projection cycle (formalize)

    x_{n+1} = Ψ ∘ T ∘ Φ(x_n)

where Φ = FFT/Clifford lifting, T = dynamics in slices/rotors, Ψ = projection.
This is already implicit in the runtime (FFT → filter → IFFT).

### Smooth interpolation α = σ(χ)

    Ψ_{n+1} = (1-α)·F_t(Ψ_n) + α·S_λ ∘ F_τ(Ψ_n)
    α = σ((χ(Ψ_n) - χ₀) / ε)

Required: ∂_τ F(Ψ) ≤ 0 (τ regime always dissipates).
This turns α from blending into a dissipative regime selector.

### Explicit contractivity condition

    |Ψ_{n+1} - Ψ*| ≤ ρ|Ψ_n - Ψ*|,  ρ < 1

Or weaker: F(Ψ_{n+1}) ≤ F(Ψ_n). Connects with Hutchinson + free energy.

### Inter-slice consistency (formalize)

    E_slice = Σ_{ℓ≠ℓ'} |Π_ℓ Ψ - T_{ℓ→ℓ'}(Π_{ℓ'} Ψ)|²  ≤  ε

Closes multislice + transport + global coherence.

### Remark (Future): Γ_virt state-dependent

    Γ_virt^μ = Σ_a β_a(Ψ) · Γ_a^μ

Not a claim. Remark/Future extension only. Already has Clifford structure;
connects with state-dependent Dirac operator. Not testable now.

---

## Decision for paper title

**Main claim**: Multiscale Contractive Extension
**Conditional claim**: Hutchinson Fractal Variant (r + Lip(Δ) < ~0.85)
**Honest note**: "Individual contractivity is supported empirically over the
sampled operational domain. A uniform global proof on the full state space
is left for future work."

---

## Figures (planned)

1. **Contractivity scatter**: ||Φ(Ψ₁)-Φ(Ψ₂)|| vs ||Ψ₁-Ψ₂|| for individual maps
2. **Hausdorff convergence**: d_H(H^n(K₀), K*) over iterations (conservative + moderate)
3. **Fixed point convergence**: ||T^n(Ψ₀) - A*|| over iterations (20 starts)
4. **Multiscale descriptor**: D_d values at different depths d
5. **Regime boundary**: contractivity ratio vs (r, δ) parameter space
