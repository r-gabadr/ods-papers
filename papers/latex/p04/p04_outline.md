# P04 — Memory Coupling, Tau Cascade, and Virtual Population Dynamics in a Clifford--Thomas Runtime

**Status**: Outline draft
**Depends on**: P01 (runtime), P02 (regimes), P03 (observability)
**Target**: Zenodo preprint

---

## Abstract (draft)

We demonstrate that the CT runtime in Cl(3,0) supports document-to-field memory
coupling through spinor fingerprints, and that a hierarchical tau cascade with
coherence gating produces associative dynamics over a virtual population of
motif replicas. Document injection via spatially-anchored Gaussian patches with
phase-signed oscillation measurably alters field observables. The tau cascade
gates associative processing via vector coherence threshold, and virtual
population spinor dynamics maintain coherent evolution across motif groups.

---

## Claims and Evidence

### Claim 1: Document injection measurably alters field dynamics

**Statement**: Injecting a DocumentChunk via `inject_chunk(Ψ, chunk, strength)`
using the phase_signed mode produces a measurable change in field observables
(free energy, d_eff, vortex count).

**Code**: `memory.field_adapter.inject_chunk(Psi, chunk, strength, mode)`
- Spinor fingerprint → `uniform_drive_from_spinor(spinor, grid_shape)` (base drive)
- Embedding → spatial anchor `(cy, cx, r)` via hashing
- Gaussian patch: `exp(-0.5·d²/σ²)` centered at anchor
- Phase-signed oscillation: `cos(fy·π·y) + sin(fx·π·x)` modulated by `sign(spinor)`
- Drive = 0.5 · patch · base · sign_channels · oscillation

**Test needed**:
- `||Ψ_after - Ψ_before||₂ > 0` after injection
- ΔF (free energy change) scales with `strength`
- Different chunks produce different drive patterns (measure cosine distance of drives)
- Observables change: d_eff, vortex count shift

### Claim 2: Assignment energy provides principled source-to-group coupling

**Statement**: `E[n,m] = α|c_n - κ_m| + β||B_n - B_m||² + χ·d_m - γ·r_n - δ·q_m`
assigns document sources to OAM groups by matching complexity, bivector alignment,
disorder, resonance, and coherence.

**Code**: `tau_assimilation.assignment_energy(meta, groups, cfg)`
- TauSourceMeta: (complexity, bivector, resonance, source_kind) per source
- OAMGroupState: (capacity κ, disorder d, coherence q, centroid_bivector B) per group
- Soft assignment: `softmax(-E/τ)` over groups

**Test needed**:
- Sources with high complexity are assigned to high-capacity groups
- Sources with similar bivector are assigned to aligned groups
- Assignment is differentiable (verify gradient exists)
- Soft assignment sums to 1 per source

### Claim 3: Tau cascade gates associative processing via coherence

**Statement**: The tau cascade has two stages — core (always active) and
associative (gated by vector coherence > threshold). The gate prevents
premature associative binding when the field lacks sufficient structure.

**Code**: `tau_cascade.coherence_gates(vec_coherence, biv_coherence, config)`
- `assoc_open = vec_coherence >= config.coherence_vec_threshold`
- `bivector_stable = biv_coherence >= config.coherence_biv_threshold`
- `tau_core_step()` always runs (readback + field_spinor + doc_spinor → motif_state, tau_state)
- `tau_assoc_step()` only runs when `assoc_open=True`

**Test needed**:
- Below threshold: assoc gate closed, only core dynamics
- Above threshold: assoc gate opens, associative layer adds to effective channels
- Gate transition is sharp at threshold (verify boolean behavior)
- Core step alone produces coherent field spinor projection

### Claim 4: Virtual population maintains coherent spinor dynamics

**Statement**: A virtual population of motif replicas, organized by
(motif, replica, member) indices across spectral bands, maintains coherent
spinor evolution through phase offsets and gain modulation.

**Code**: `virtual_population.build_virtual_population_layout(motifs, ...)`
- Layout: source_index, replica_index, member_index, band_index, phase_offset, gain
- Band = pair_index × |grades| + grade_index (e.g., 3 pairs × 4 grades = 12 bands)
- Spinor seeds: base signature + small perturbations per (rep, member, band)
- Channel basis: one-hot primary + 0.35 secondary + 0.01 noise
- Groups bucketed for collective coherence tracking

**Test needed**:
- Population size = n_motifs × replicas × members
- Phase offsets are unique per member (no duplicates within motif)
- Spinor seeds are non-degenerate (pairwise cosine < 1)
- Channel basis vectors are approximately orthogonal across bands
- Group labels match expected format "motif@b{bucket_idx}"

### Claim 5: CT relaxation in grade-1/2 space conserves E₁₂

**Statement**: The CT free energy `F₁₂ = ½k₁E₁ + ½k₂E₂ + ½λ||v-β||² - TH₁₂`
with `H₁₂ = -q log q - (1-q) log(1-q)`, `q = E₁/(E₁+E₂)` provides a
thermodynamically consistent relaxation for the vector-bivector sector.

**Code**: `runtime.ct_relaxation.free_energy_12(v, beta, ...)`
- Chemical potentials: `μ₁ = ½k₁ - T·∂H/∂q · ∂q/∂E₁`
- Gradient: `∇v F = 2μ₁v + λ(v-β)`, `∇β F = 2μ₂β - λ(v-β)`
- Relaxation step descends F₁₂
- mobility_scale modulated by audio/memory/doc biases

**Test needed**:
- F₁₂ decreases monotonically during relaxation steps
- E₁₂ = E₁ + E₂ is conserved (within tolerance) during transport-only steps
- μ₁ = μ₂ at equilibrium (chemical potential equality)
- Correlation c₁₂ = ⟨v,β⟩/√(E₁E₂) bounded in [-1,1]

---

## Figures (planned)

1. **Document injection**: field state before/after with drive pattern overlay
2. **Assignment energy matrix**: heatmap of E[n,m] for multiple sources × groups
3. **Tau cascade timeline**: core activation → coherence gate → associative binding
4. **Virtual population bands**: spectral band distribution of replicas

---

## What's missing in the codebase

1. **Memory curvature κ** — mentioned in plans but not found as explicit function
2. **Overnight consolidation** — `neuro_field/overnight.py` exists but not reviewed
3. **End-to-end memory pipeline** — need a test that goes chunk → inject → relax → measure
4. **Motif graph construction** — `motifs.py` defines MotifSpec but graph building not yet reviewed
5. No benchmark data yet (unlike P02 which has summary.json)

---

## Test file

`_claude/tests/paper_claims/test_p04_claims.py`
