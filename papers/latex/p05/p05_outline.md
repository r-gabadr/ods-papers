# P05 — Multiscale Extension of the CT Runtime: Contractive Families and Fractal Structure

**Status**: Outline draft — empirical contractivity supported on the sampled
operational domain; uniform global proof still open
**Depends on**: P01 (runtime), P02 (regimes), P03 (observability)
**Target**: Zenodo preprint

---

## Scope

`P05` is the multiscale / contractive paper. It owns:

- Banach-style contractive extensions;
- Hutchinson-style fractal structure on strictly contractive subsets;
- compressed operational representatives on top of that structure.

It does **not** own closure claims, singularity language, or raw-gap
instability narratives; those belong to `P06` and only through the shifted
low-mode surrogate.

---

## Formal

### Core question

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

### State space and admissibility

To apply Banach or Hutchinson cleanly, the ambient space must be fixed
explicitly. The intended setting is

    X = ⊕_{ℓ∈Λ} C(Ω, Cl_{3,0}^{(ℓ)})

with Ω compact and the sup norm

    ||Ψ||_∞ = sup_{x∈Ω} ||Ψ(x)||.

This makes X a complete metric space. Operationally, one usually works on an
admissible subset X_adm ⊆ X selected by the runtime checker / projection
Π_h. The contractive/fractal claims should therefore be read as applying
either:
- directly on X_adm when the generators preserve admissibility, or
- on X with each iteration followed by Π_h / acceptance filtering.

This point is not cosmetic: contractivity alone is not enough if the maps can
leave the admissible manifold used by the runtime.

### Variant A: Multiscale Contractive Extension (minimal, safe)

**Theorem**: If L_ℓ = Σ |w_{α,ℓ}|(r_{α,ℓ} + L_{α,ℓ}) < 1, then T_ℓ is
contractive and has a unique fixed point A*_ℓ = T_ℓ(A*_ℓ).

No Hausdorff metric needed. No "fractal" claim. Just Banach on a weighted
average of rotor-scaled maps.

**Pros**: Honest, simple, easy to verify empirically.
**Cons**: Less exciting. "Contractive extension" doesn't capture the
self-similar structure that the maps visually produce.

### Variant B: Fractal Extension via Hutchinson (strong, beautiful)

**Definition** (Set-valued fractal slice operator):

    H_ℓ(K) = ∪_{α=1}^{m_ℓ} Φ_{α,ℓ}(K),    K ∈ K(X)

where K(X) is the space of non-empty compact subsets of X with Hausdorff
distance d_H.

**Theorem** (Hutchinson '81): If every Φ_{α,ℓ} is a contraction with ratio
c_{α,ℓ} < 1, then H_ℓ is contractive in (K(X), d_H) with ratio
max_α c_{α,ℓ}, and there exists a unique compact invariant set K*_ℓ such that

    K*_ℓ = H_ℓ(K*_ℓ)

**Corollary / operational remark**: The averaged operator T_ℓ may provide a
fixed-point surrogate or compressed operational representative associated with
K*_ℓ. This representative is **not automatically a barycenter** of the
Hutchinson attractor unless an invariant measure μ_ℓ on K*_ℓ is specified and
one defines

    Ā_ℓ = ∫_X x dμ_ℓ(x).

The fractal structure of K*_ℓ is the "true" attractor; the averaged operator
only supplies an operational compression of that structure.

**Pros**: Mathematically beautiful. Separates fractal structure from
operational compression. Justifies "fractal" in the title.
**Cons**: Requires proving individual contractivity of each Φ_{α,ℓ}.

### Key requirement for Variant B

Each Φ_{α,ℓ} must satisfy:

    ||Φ_{α,ℓ}(Ψ₁) - Φ_{α,ℓ}(Ψ₂)|| ≤ c_{α,ℓ} · ||Ψ₁ - Ψ₂||

with c_{α,ℓ} < 1 for ALL α, ℓ.

Since R·Ψ·R̃ is an isometry, we need:

    c_{α,ℓ} = r_{α,ℓ} + Lip(Δ_{α,ℓ}) < 1

This is the empirical question that the test must answer.

**Important correction**: a pure rotor with r = 1 and Δ = 0 is an isometry,
not a strict contraction. Such maps must be treated as transport symmetries,
not as generators of the strong Hutchinson IFS, unless they are combined with
explicit damping/projection so that c_{α,ℓ} < 1.

### Contractivity margin

Passing/failing the test c_{α,ℓ} < 1 is not enough. A useful operational
quantity is the contractivity margin

    m_{α,ℓ} := 1 - c_{α,ℓ}.

This suggests the following taxonomy:
- **strictly contractive and useful**: m_{α,ℓ} is non-trivial
- **marginally contractive**: m_{α,ℓ} ≈ 0
- **isometric**: m_{α,ℓ} = 0

This distinction matters because c_max = 0.9997 still implies contractivity in
the formal sense, but with extremely weak finite-time separation and very slow
convergence.

### Admissibility preservation

Besides contractivity, one wants admissibility preservation:

    Φ_{α,ℓ}(X_adm) ⊆ X_adm

or at least

    x ∈ X_adm  ⇒  Π_h ∘ Φ_{α,ℓ}(x) ∈ X_adm.

This is the correct bridge between the paper-level IFS story and the runtime,
where checker / projection / acceptance filtering already play a central role.

## Operational V1

### Implemented now

- `regime_alpha(...)`, `spectral_smoothing(...)`, and `regime_blend(...)` in
  `ods_unified_v2.jax_runtime.multiscale`
- `slice_consistency(...)` in the same module as the runtime proxy for
  inter-slice transport mismatch
- empirical contractivity experiments for rotor, rotor-plus-delta, gamma flow,
  defect stabilizer, and Laplacian/Wilson-like maps

The operational reading remains:

    lift / evolve / project

with runtime-level blend and consistency checks. The paper should not pretend
that these operational helpers already amount to a full global proof over all
of `X`.

### Additional elements (both variants)

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

### Surrogates / proxies only

- `slice_consistency(...)` is a runtime mismatch proxy, not a transport theorem
- runtime-level averaged fixed points are compressed representatives, not
  automatic barycenters
- contractivity evidence is sampled operational evidence, not a global uniform
  proof on the full ambient space

## Tests and Evidence

See `_claude/tests/paper_claims/test_p05_claims.py` and
`_claude/experiments/contractivity_test.py`.

---

## Overnight results (2026-03-23)

### Contractivity: all tested runtime maps individually contractive
- rotor_r090: c_max = 0.900 (exact, all grids)
- rotor_delta_085_010: c_max ≈ 0.910 (all grids, 100%)
- rotor_delta_085_020: c_max ≈ 0.970 (all grids, 100%)
- gamma_flow (dt=0.01-0.02): c_max ≈ 0.9997 (100%)
- defect_stab (dt=0.001-0.01): c_max ≈ 0.9992 (100%)
- laplacian (α=0.005-0.02): c_max ≈ 0.48-0.82 (100%)

**Interpretation**:
- rotor/laplacian maps are comfortably contractive
- gamma_flow and defect_stab are only marginally contractive (m ≈ 10^-3)
- pure rotor transport with c = 1 would sit outside the strong Hutchinson
  subset unless damped

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

## Future

### New pieces from review (2026-03-24)

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

### Code status update (2026-03-23)

- `regime_alpha(...)`, `spectral_smoothing(...)`, and `regime_blend(...)` are
  now implemented in `ods_unified_v2.jax_runtime.multiscale`.
- `slice_consistency(...)` is now implemented in the same module as a runtime
  proxy for inter-slice transport mismatch.

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
