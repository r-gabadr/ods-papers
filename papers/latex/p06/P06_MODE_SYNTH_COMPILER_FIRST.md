# P06 Mode Synth v0 - Why Synth Before Hardware

Working note. Not a published claim.
Last update: 2026-03-23

---

## Decision

Implement the modal synthesizer first, then profile hardware/backend behavior.

Reason:

- hardware tuning only matters once the modal kernel is stable
- a compiler-first JAX path is easier to reason about than a mixed Taichi+JAX hot path
- Taichi is still useful, but better as a trajectory generator / PDE baseline than as the first integration target

The correct order is:

1. fix the modal analysis/synthesis interface
2. define monitors and stability budgets
3. verify JAX/XLA-friendly execution
4. only then compare backends and external simulators

---

## What the v0 synthesizer is

`src/ods_unified_v2/jax_runtime/mode_synth.py` provides:

- FFT/IFFT on channels-last Clifford states
- per-mode energy
- per-mode bivector phase coherence
- spectral-curvature map
- spectral gap proxy
- deterministic gain-field synthesis
- a closed-loop `mode_synth_step`

This is intentionally not a trained Clifford-FNO yet.

It is the compiler-first scaffold that a future learned policy can plug into.

---

## Why not Taichi first

Mixing Taichi simulation with JAX control in the hot path would add:

- host/device orchestration overhead
- extra state handoff complexity
- weaker compiler visibility
- less direct evidence for XLA-fusibility and shape stability

So the current policy is:

- JAX in the runtime core
- Taichi as an optional external generator, baseline, or teacher

---

## Relation to existing code

The v0 synth is aligned with code already present in the workspace:

- `genesis_jax/core/fno.py`
- `genesis_jax/core/afno.py`
- `src/ods_unified_v2/jax_runtime/spectral_pde.py`
- `src/ods_unified_v2/jax_runtime/field_clifford_bridge.py`

The synthesizer does not replace those pieces.

It defines a lean operational interface that future learned spectral blocks can use.

---

## Relation to AMR

The synth is naturally upstream of AMR.

AMR answers:

- where should we spend local spatial budget?

The synth answers:

- which modal bands are dangerous or coherent enough to deserve intervention?

So the intended flow is:

1. modal monitors produce:
   - mode energy
   - phase coherence
   - spectral curvature
   - gap proxy
2. those are collapsed into a spatial indicator
3. AMR turns that indicator into a mask / budget allocation

This is why `mode_synth.py` now exposes:

- `mode_synth_indicator(...)`
- `mode_synth_activity_mask(...)`

The synth should therefore be seen as a *spectral front-end for AMR*, not as a
replacement for AMR.

In the current runtime, the first live integration is already in place:

- `tick_runtime` can apply the mode synth step when the runtime control overlay
  sets a positive `synth.alpha`
- the synth reports activity ratio, mean gain, gap proxy, and coherence summary
- these metrics are exposed through runtime status for inspection

This means the synth is no longer just a sandbox kernel; it is an optional
runtime actuator driven by the existing control surface.

---

## Relation to the control DSL

The synth is also a control target.

The DSL is the right place to steer:

- blend strength
- damping aggressiveness
- modal energy budget

without hard-coding those schedules into the runtime.

The current minimal control surface is:

- `synth.alpha`
- `synth.damping`
- `synth.energy_budget`

and `spec.gap` is a valid observable for conditions.

That means scripts like:

    when spec.gap < 0.2: synth.damping(0.9)
    synth.alpha(0.35) for 3 tau

can modulate the synthesizer without touching the compiled modal kernel.

---

## Observe -> decide -> respond

The intended control loop is now explicit:

1. `spectral_monitor` observes
   - eigenvalues
   - gap
   - radius
   - entropy
   - spectral curvature

2. `spectral_contracts` decides
   - whether the current update is admissible
   - whether the runtime should warn, accept, or reject-with-damping

3. runtime consumers respond
   - `mode_synth` changes blend/damping behavior
   - `AMR` changes where recomputation budget is spent
   - `control_targets` / DSL adjust the control surface

This keeps the runtime architecture legible:

- observables are gathered in one place
- admissibility is evaluated in one place
- actuation remains distributed but traceable

The first implementation is intentionally conservative.

`src/ods_unified_v2/jax_runtime/spectral_contracts.py` currently enforces:

- gap floor
- radius growth cap
- spectral-curvature cap
- windowed total-variation caps on gap/radius
- optional eigenvalue-window total variation

This is a spectral contract only.
No homological or full operator-level invariants are claimed yet.

The current runtime coupling is deliberately conservative:

- the contract is evaluated after the reduced render-graph monitor is refreshed
- if it only warns, the runtime promotes `amr.recompute`
- if it hard-fails, the runtime arms `synth.damping`, injects a minimal
  `synth.alpha`, and marks `amr.recompute`

So the first closed loop is:

`render_graph -> spectral_monitor -> spectral_contracts -> synth/AMR overlay`

This is operational, but still a v1 policy layer rather than a final theorem
about global stability.

---

## Tests that matter

The initial test layer should demonstrate:

1. exact analysis/synthesis roundtrip
2. identity under neutral gains
3. high-frequency damping reduces high-band energy
4. no obvious energy creation under damping-only settings
5. bounded phase coherence
6. compiler-friendly shape stability under `jax.jit`

These are implemented in:

- `tests/test_jax_runtime_mode_synth.py`
- `tests/test_jax_runtime_spectral_contracts.py`

---

## Next layer

Once the v0 kernel is stable, the next useful upgrades are:

1. learned gain policy
   - AFNO/FNO-inspired policy behind the same gain interface

2. budget-aware sparse control
   - integrate with ROI / delta streams from P06

3. backend/perf bench
   - CPU/GPU/TPU parity
   - compile time vs steady-state
   - jitter and recompilation checks

4. Taichi teacher loop
   - use Taichi trajectories to supervise or stress-test the gain policy
