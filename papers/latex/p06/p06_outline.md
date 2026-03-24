# P06 — Closure Handoff and Declarative Control Envelopes in the CT Runtime

**Status**: Outline draft
**Depends on**: P03 (observability), P05 (multiscale)
**Target**: Post-P05 note / thesis chapter / later preprint

---

## Scope

`P06` is deliberately modest. It should cover:

- the smooth handoff from fast clock `t` to structural clock `tau`;
- declarative control envelopes driven by observables;
- the operational meaning of closure failure in the runtime.

It should **not** claim:

- a proof about Navier--Stokes singularities;
- an exact fluid closure theorem;
- an exact q-Dirac double-clock torsion invariant;
- that the raw spectral gap is a physical negative-mass observable.

---

## Formal

### Two-clock envelope

The intended split is

    partial_t Psi = F_t(Psi, input, stream)
    partial_tau Psi = F_tau(Psi)

with `F_t` the fast visible update and `F_tau` the slower structural /
relaxation flow.

### Smooth handoff

The paper-level handoff is expressed through a smooth trigger

    alpha(Psi) = sigma((chi(Psi) - chi0) / epsilon),

and a blended update of the form

    Psi^+ = (1 - alpha) F_t(Psi) + alpha F_tau(Psi).

`chi` is inherited from `P03` as an admissibility / validity trigger. `P06`
owns the interpretation of `alpha` as a closure handoff variable.

### Instability language: only via shifted low-mode surrogate

If one wants to use instability-driven language, the only accepted route is

    L_tilde(Psi) = L_Psi - mu_c^2 I
    m_eff^2(Psi) = lambda_min(L_tilde(Psi)).

Then:

- `m_eff^2 >= 0` means no low-mode instability detected by the surrogate;
- `m_eff^2 < 0` means the shifted operator has developed an unstable low-mode
  direction.

This is an **operational shifted low-mode surrogate**, not a literal
field-theoretic mass observable.

### Closure reading

The paper-safe claim is:

> We define an operational breakdown criterion for the continuum closure. When
> exceeded, dynamics are transferred from the fast update to the structural
> clock `tau`.

That is the strongest acceptable phrasing.

---

## Operational V1

### Implemented now

- `ods_unified_v2.jax_runtime.observability.compute_chi(...)`
- `handoff_alpha(...)`
- `effective_mass_squared(...)`
- `tachyonic_instability_score(...)`
- `tachyonic_instability_from_spectrum(...)`
- `ods_unified_v2.jax_runtime.multiscale.regime_alpha(...)`
- `regime_blend(...)`
- `slice_consistency(...)`
- runtime control response through:
  - `spectral_contracts.py`
  - `mode_synth.py`
  - `runtime/control_dsl.py`
  - `tick_runtime.py`

### Control semantics

The operational rule is:

    observability -> contract / admissibility -> synth / AMR / control response

The control layer acts on envelopes and operator settings, not by asserting a
mathematical theorem about the continuum model.

### Runtime interpretation

Current runtime objects already support:

- smooth `chi -> alpha` gating;
- regime blending;
- spectral and topological stability flags;
- declarative reactive control based on observable thresholds.

This is enough for an operational paper. It is not enough for a strong closure
theorem, and the paper should say so explicitly.

### Surrogates / proxies only

- shifted low-mode instability proxy
- `D_Psi^dagger D_Psi` as an exact geometric operator
- q-Dirac double-clock torsion
- Wang--Wang torsion in the strong spectral-geometric sense
- any claim that `chi` predicts singularity formation

---

## Public Claim Set

### Claim A: `chi` provides a smooth runtime validity trigger

Implemented and already testable through `compute_chi(...)` and
`handoff_alpha(...)`.

### Claim B: the handoff variable `alpha` is a smooth regime selector

Implemented and already testable through `regime_alpha(...)` and
`regime_blend(...)`.

### Claim C: instability wording is allowed only through the shifted low-mode surrogate

Implemented as a proxy. The paper must explicitly call it a surrogate.

### Claim D: control DSL envelopes can react to spectral / reuse observables

Implemented operationally in runtime and benchmark artefacts. This is a
runtime-control claim, not a theorem about PDE closure.

---

## Tests and Runtime Validation

### Minimum regression gate

- shifted-instability subset in `ods-unified-v2`:
  - `effective_mass_squared`
  - `tachyonic_instability_score`
  - `tachyonic_instability_from_spectrum`
  - `compute_chi` integration

### Runtime validation expected after sync touching exported observables

- spectral monitor subset
- spectral contracts subset
- multiscale subset
- tick-runtime control subset

---

## Figures (planned)

1. `chi -> alpha` smooth gating curve
2. shifted low-mode surrogate crossing zero
3. control-envelope example with reactive AMR/render changes
4. closure handoff cartoon: `t` branch vs `tau` branch

---

## Future

- compiler-first coupling where `chi` also selects compiled control envelopes
- stronger comparison between runtime handoff and continuum closure heuristics
- q-Dirac style temporal commutators as future observables
- exact spectral torsion only if a proper operator-theoretic implementation
  exists

---

## Editorial Rules

- Do say: **closure handoff**, **operational breakdown criterion**,
  **smooth regime selector**, **declarative control envelope**.
- Do not say: **NS3D solved**, **finite-time singularity detected**,
  **exact tachyonic field phase**, **exact q-Dirac torsion computed**.

