# P03 — Observability, Spectral Coherence, and Topological Control in a Clifford--Thomas Runtime

**Status**: Outline draft
**Depends on**: P01 (runtime), P02 (regimes)
**Target**: Zenodo preprint

---

## Abstract (draft)

We establish an observability framework for the CT runtime in Cl(3,0).
Three independent measurement channels — commutator defect, spectral graph
connectivity (Fiedler λ₂), and rotational tension maps — together provide a
complete observability panel for runtime state without full state reconstruction.
We show that (1) the t/τ clock commutator defect quantifies non-commutativity
of the split-time architecture, (2) algebraic connectivity λ₂ tracks field
coherence across slices via a spectral graph construction, and (3) rotational
tension localizes topological activity in the bivector sector.
A gamma_focus_score derived from grade entropy and bivector coherence serves as
an effective observability-to-action bridge.

---

## Claims and Evidence

### Claim 1: Commutator defect measures clock non-commutativity

**Statement**: `||Step_τ ∘ Step_t(Ψ) - Step_t ∘ Step_τ(Ψ)||₂ > 0` when both
clocks are active, quantifying the non-Abelian structure of the split-time
architecture.

**Code**: `observability.compute_commutator_defect(Psi, params)`
- Applies `step_frozen` in both orders: t-then-τ vs τ-then-t
- Returns RMS of the difference

**Test needed**:
- Defect > 0 when ω₁₂ > 0 and k_τ > 0
- Defect = 0 when either clock is disabled
- Defect scales monotonically with ω₁₂ (at fixed k_τ)
- Defect scales monotonically with k_τ (at fixed ω₁₂)

### Claim 2: Algebraic connectivity λ₂ tracks field coherence

**Statement**: The Fiedler eigenvalue λ₂ of a Gaussian-kernel graph over
slice feature vectors measures inter-slice coherence. λ₂ → 0 means
disconnected slices; λ₂ large means uniform coherence.

**Code**: `tcl_spectral.build_tcl_spectral_report(field)`
- Feature matrix: [scalar_E, vector_E, bivector_E, pseudo_E, coherence, disorder, capacity, attractor_residual] per slice
- Gaussian adjacency: `A_ij = exp(-||f_i - f_j||² / 2σ²)`
- Graph Laplacian: `L = D - A`
- λ₂ = second smallest eigenvalue of L

**Test needed**:
- λ₂ > 0 for non-trivial fields
- λ₂ increases when slices are made more similar
- λ₂ correlates with OAM group coherence
- t₀_spec = 1/λ₂ as characteristic mixing time

### Claim 3: Rotational tension localizes topological activity

**Statement**: The rotational tension map `R(x) = √(Δrow² + Δcol²)` where
`Δ = 1 - |⟨b̂(x), b̂(neighbor)⟩|` localizes regions of high bivector
orientation change — topological activity hotspots.

**Code**: `amr.rotational_tension_map(Psi)`
- Normalizes bivector sector
- Computes alignment with row/col neighbors
- Tension = misalignment magnitude

**Test needed**:
- Tension map correlates with vortex locations (from P02 soliton detector)
- Activity mask at quantile q=0.9 captures > 50% of vortex centers
- Tension map is zero for spatially uniform bivector fields
- Higher attractor dimension → higher mean tension

### Claim 4: gamma_focus_score as observability-to-action bridge

**Statement**: `gamma_focus = b_ratio + 0.24·φ_ratio + 0.22·coherence - 0.28·H_grade`
provides a single scalar that bridges observability (what the field is doing)
to action (whether to activate gamma flow).

**Code**: `gamma_virtual.gamma_focus_score(field_mv)`
- b_ratio = bivector_energy / total_energy
- φ_ratio = pseudoscalar_energy / total_energy
- coherence = ||mean(b̂)||/√3
- H_grade = normalized grade entropy

**Test needed**:
- Score > 0 for bivector-dominated fields
- Score decreases as grade entropy increases (more uniform grade distribution)
- Score correlates with vortex persistence (from P02)

### Claim 5: Defect stabilizer preserves spin index

**Statement**: The defect stabilizing flow `g·(α·∇²Ψ - β·D_filtered)` gated
by `σ(gain·(D - threshold))` reduces Dirac defect while preserving the spin
grade energy index via projection.

**Code**: `defect_stabilizer.apply_defect_stabilizer(field_mv, dt_tau=...)`
- Gate activates when dirac_defect > threshold
- Flow = Laplacian smoothing minus filtered Dirac term
- Projection: removes component along spin_index gradient
- Selection: picks mixing parameter η that doesn't increase defect

**Test needed**:
- Dirac defect decreases or stays constant after application
- Spin grade energy index is preserved (within tolerance)
- Gate = 0 when defect < threshold

---

## Figures (planned)

1. **Commutator defect heatmap**: defect as function of (ω₁₂, k_τ)
2. **λ₂ temporal evolution**: λ₂ over time for Thomas3D vs Spinor8D
3. **Rotational tension map**: spatial map with vortex centers overlaid
4. **Observability panel**: composite figure showing all 3 channels + focus_score

---

## New pieces from review (2026-03-24)

### Claim 6 (candidate): χ trigger as formalized handoff criterion

**Statement**: Define a dimensionless, regular, monotonic validity functional:

    χ(x,t) = w₁·(η_crit/η) + w₂·Kn_loc + w₃·|ω|/ω_ref + w₄·B(t)

where B(t) = ∫_{t-Δt}^{t} ||ω(·,s)||_{L∞} ds (windowed BKM sentinel).

Properties:
- χ ∈ C⁰(Ψ), ideally C¹ (no jumps in handoff)
- ∂_Ψ χ · δΨ ≥ 0 in critical regime (monotonicity = interpretability)
- All terms dimensionless

When χ > χ*, dynamics transfer from fast clock t to structural clock τ.

**Paper-ready statement**: "We do not claim to predict finite-time singularities.
We define an operational breakdown criterion for the continuum closure. When
exceeded, dynamics are transferred to the structural clock τ."

**Test needed**:
- χ increases when tension/enstrophy increases
- χ = 0 for uniform fields
- Handoff α = σ(χ) is smooth (C¹ in Ψ)

### Spectral coherence (new metric)

    C_spec = Σ_k λ_k² / (Σ_k |λ_k|)²

Measures spectral dispersion. Intermediate between gap (local) and torsion (global).

### Spectral entropy (explicit)

    H_spec = -Σ_k p_k log p_k,   p_k = |λ_k| / Σ_j |λ_j|

Connects with d_eff, τ, and χ. Already implicit in code, needs to be explicit.

### Remark: Σ_par = Ψ_par·Ψ̃_par

Observable of the even sector (scalar + bivector). Gives scalar magnitude +
pseudoscalar chirality. Only valid for even subalgebra — does NOT work for
general multivectors (Gemini overclaims this).

### Future: Spectral torsion T(D)

Refined invariant for nonminimal Hodge–Dirac operators (arXiv 2025).
Candidate for appendix or future extension.

---

## Overnight data available (2026-03-23)

- `03_commutator_sweep.json` — 126 configurations (42 × 3 seeds)
- `05_regime_separability.json` — Cohen's d: focus=15, entropy=123, tension=50, d_eff=415

---

## What's missing in the codebase

1. **tcl_offline.py** (persistent homology β₀, β₁, β₂) — referenced in plans but not yet verified as a standalone claim
2. **Spectral monitor** — appears to be a convenience wrapper, not new math
3. **Pinning metrics** — unclear if independently testable
4. χ trigger not yet implemented — needs new module or extension of existing focus_score
5. C_spec and H_spec not yet computed — straightforward from existing eigenvalue computation

---

## Test file

`tests/paper_claims/test_p03_claims.py` (in ods-papers repo)
