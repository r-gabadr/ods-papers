# ODS Paper Series — Global Notation Table (P01→P06)

| Symbol | Paper | Meaning | Domain |
|--------|-------|---------|--------|
| Ψ | All | Cl(3,0) multivector field, shape (..., 8) | Cl(3,0) |
| E_k | P01+ | Grade-k energy: ‖Π_k Ψ‖² | ℝ≥0 |
| H_grade | P01+ | Grade entropy: -Σ p_k log p_k | [0, log4] |
| d_eff | P01+ | Effective dimension: exp(H_grade) | [1, 4] |
| F | P01+ | Free energy (full field) | ℝ |
| t | P01+ | Fast physical clock | continuous |
| τ | P01+ | Slow structural clock | discrete steps |
| χ_Cl | P03 | Structural stress functional on Cl(3,0) observables | [0, 1] |
| χ* | P03 | Threshold for t→τ handoff | 0.5 default |
| α(χ) | P03 | Blending weight: σ((χ-χ*)/ε) | [0, 1] |
| C_spec | P03 | Spectral coherence: Σλ²/(Σ|λ|)² | (0, 1] |
| H_spec | P03 | Spectral entropy: -Σ p_k log p_k on eigenvalues | [0, log n] |
| δ_D | P03 | Dirac defect | ℝ≥0 |
| R(x) | P03 | Rotational tension | ℝ≥0 |
| F_Γ | P03 | Gamma-focus score | ℝ |
| δ_comm | P03 | Commutator defect ‖S_τ∘S_t - S_t∘S_τ‖ | ℝ≥0 |
| E_t | P03 | Fast amplitude energy | ℝ≥0 |
| M_τ | P03 | Structural mass | ℝ≥0 |
| c_eff | P03 | Resolution limit (Nyquist, NOT speed of light) | ℝ>0 |
| E₁₂ | P04 | Combined grade 1-2 energy: E₁+E₂ | ℝ≥0 |
| F₁₂ | P04 | Grade 1-2 free energy | ℝ |
| q | P04 | Grade ratio: E₁/(E₁+E₂) | (0, 1) |
| c₁₂ | P04 | Grade correlation: ⟨v,β⟩/√(E₁E₂) | [-1, 1] |
| κ | P04 | Memory curvature proxy | ℝ≥0 |
| Φ_{α,ℓ} | P05 | Clifford map: r·RΨR̃ + Δ(Ψ) | Cl(3,0)→Cl(3,0) |
| T_ℓ | P05 | Averaged slice operator: Σ w_α Φ_{α,ℓ} | contraction |
| H_ℓ | P05 | Hutchinson operator: ∪ Φ_{α,ℓ}(K) | K(X)→K(X) |
| K*_ℓ | P05 | IFS attractor (compact invariant set) | K(X) |
| χ_closure | P06 | Continuum closure-breakdown trigger | [0, 1] |
| η | P06 | Kolmogorov scale: (ν³/ε)^{1/4} | ℝ>0 |
| Kn | P06 | Local Knudsen number: λ/L | ℝ≥0 |

## Key distinction

- **χ_Cl** (P03): defined on Cl(3,0) field observables (tension, focus, d_eff, δ_D)
- **χ_closure** (P06): defined on continuum-resolution diagnostics (η, Kn, enstrophy, curvature, focus)
- A projection linking both is left for future work.
