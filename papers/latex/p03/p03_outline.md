# P03 — Observability, Spectral Coherence, and Admissibility Triggers in a Clifford Runtime

**Status**: Outline draft aligned with the unified formulation
**Depends on**: P01 (runtime), P02 (regimes)
**Target**: Zenodo preprint

---

## Scope

`P03` is the observability paper. Its job is to define and validate the
runtime observables that make later decisions possible. It should **not**
absorb:

- fractal / contractive structure claims from `P05`;
- closure or fluid-singularity language from `P06`;
- exact generalized slice-monogenic or spectral torsion implementations.

The correct narrative is:

    observability -> admissibility trigger -> downstream action

where the downstream action is implemented in the runtime and formalized more
fully elsewhere.

---

## Abstract (draft)

We define an observability layer for a Clifford-valued runtime in `Cl(3,0)`
using independent channels that can be measured without full state
reconstruction. The formal layer consists of a split-clock commutator defect,
slice spectral connectivity, rotational activity, and grade-sensitive focus
scores. The operational layer packages these observables into a smooth
admissibility trigger `chi`, which is used by the runtime to gate control and
multiscale responses. The paper is deliberately modest: it documents what is
implemented, distinguishes runtime surrogates from stronger mathematical
objects, and leaves closure and handoff claims to `P06`.

---

## Formal

### Formal object 1: split-clock commutator defect

For fast and structural updates `Step_t` and `Step_tau`, define

    Delta_comm(Psi) = ||Step_tau o Step_t(Psi) - Step_t o Step_tau(Psi)||_2.

This is the clean observable of non-commutativity in the split-time
architecture. It is an observability object, not yet a closure theorem.

### Formal object 2: slice spectral coherence

Given slice feature vectors `f_ell`, define a Gaussian-kernel adjacency and
graph Laplacian

    A_ij = exp(-||f_i - f_j||^2 / 2 sigma^2),   L = D - A.

The Fiedler value `lambda_2(L)` is the minimal formal connectivity signal.
Two auxiliary slice-spectrum statistics are also useful:

    C_spec = sum_k lambda_k^2 / (sum_k |lambda_k|)^2
    H_spec = -sum_k p_k log p_k,   p_k = |lambda_k| / sum_j |lambda_j|.

These quantify concentration and dispersion of the slice spectrum.

### Formal object 3: rotational activity

Let `b_hat(x)` be the normalized bivector sector. The local rotational tension
map is a nearest-neighbor misalignment observable:

    R(x) = sqrt(Delta_row(x)^2 + Delta_col(x)^2),

with `Delta = 1 - |<b_hat(x), b_hat(neighbor)>|`.

This localizes rapid bivector reorientation without claiming a full topological
classification theorem.

### Formal object 4: observability-to-admissibility trigger

The paper-level object is a smooth, dimensionless validity functional

    chi = chi(eta, Kn_loc, |omega|, B, ...)

whose role is operational: it marks when the current continuum-style update is
becoming unreliable enough that the runtime should tighten control envelopes or
hand off to another regime.

In `P03`, `chi` is only an admissibility trigger. The closure/handoff reading
belongs to `P06`.

---

## Operational V1

### Implemented now

1. **Commutator defect**
   - `ods_unified_v2.jax_runtime.observability.compute_commutator_defect(...)`
   - tests already cover positive defect with both clocks active and collapse
     to zero when one branch is disabled.

2. **Slice spectral report**
   - `ods_unified_v2.jax_runtime.tcl_spectral.build_tcl_spectral_report(...)`
   - runtime quantities now explicit:
     - `lambda2`
     - `spectral_coherence(...)`
     - `spectral_entropy(...)`
     - `t0_spec`, `t1_spec`, `t_spec`

3. **Rotational activity**
   - `ods_unified_v2.jax_runtime.amr.rotational_tension_map(...)`
   - `activity_mask(...)`, `activity_ratio(...)`

4. **Focus / grade bridge**
   - `ods_unified_v2.jax_runtime.gamma_virtual.gamma_focus_score(...)`
   - `gamma_grade_entropy(...)`

5. **Admissibility trigger**
   - `ods_unified_v2.jax_runtime.observability.compute_chi(...)`
   - `handoff_alpha(...)`

6. **Operational topology proxies**
   - `ods_unified_v2.jax_runtime.spectral_monitor`
   - `betti_numbers`, `topological_drift`, and stability flags are accepted as
     runtime proxies, not as paper-level persistent-homology theorems.

### Surrogates / proxies only

- `D_Psi^dagger D_Psi` as an exact state-dependent geometric operator
- exact generalized slice-monogenic admissible decomposition in the sense of
  Hu--Ding
- spectral torsion in the sense of Wang--Wang
- q-Dirac double-clock torsion as a literature-backed invariant
- full persistent homology as a primary claim

### Future

- refined admissible slice classes motivated by Hu--Ding
- stronger spectral metrology motivated by Wang--Wang
- tighter integration between `chi` and declarative control envelopes
- appendix-level comparison between runtime topology proxies and offline
  persistent homology

---

## Public Claim Set

### Claim A: commutator defect measures split-clock non-commutativity

**Paper-safe statement**:

    Delta_comm(Psi) > 0

when both clocks are active, and collapses when one branch is disabled.

This is implemented and tested.

### Claim B: slice spectral quantities measure coherence, not full geometry

**Paper-safe statement**:

`lambda_2`, `C_spec`, and `H_spec` provide an operational panel for inter-slice
coherence and spectral concentration/dispersion.

This is implemented and tested as a runtime spectral panel. It is **not** an
exact spectral torsion claim.

### Claim C: rotational tension localizes activity hotspots

**Paper-safe statement**:

The rotational tension map detects localized bivector reorientation and yields
stable activity masks suitable for AMR and control.

This is implemented and tested.

### Claim D: `chi` is an admissibility trigger, not a singularity predictor

**Paper-safe statement**:

`chi` is a smooth, dimensionless operational validity functional used to gate
runtime responses.

This is implemented. `P03` does **not** claim that `chi` predicts singularity
formation or proves continuum breakdown.

---

## Figures (planned)

1. Commutator defect heatmap over `(omega_12, k_tau)`
2. Slice spectral panel: `lambda_2`, `C_spec`, `H_spec`
3. Rotational tension map with activity mask overlay
4. Composite observability panel: defect + spectrum + tension + `chi`

---

## Tests and Artefacts

### Claim tests

- `tests/paper_claims/test_p03_claims.py`

Current tests already cover:

- commutator defect activation / collapse;
- positivity and coherence behavior of `lambda_2`;
- zero tension for uniform bivector fields;
- `gamma_focus_score` and grade-entropy sanity;
- non-increase of defect under the stabilizer.

### Runtime artefacts already available

- `03_commutator_sweep.json`
- `05_regime_separability.json`

These stay as evidence artefacts, not as the paper's mathematical backbone.

---

## Editorial Rules

- Do say: **observability**, **spectral coherence**, **activity localization**,
  **admissibility trigger**.
- Do not say: **exact spectral torsion**, **exact q-Dirac runtime**,
  **generalized slice-monogenic implementation**, **finite-time singularity
  prediction**.
- If closure/handoff language appears, route it to `P06`.

