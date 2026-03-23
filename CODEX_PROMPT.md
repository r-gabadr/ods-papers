# Prompt para Codex — Continuación ODS 2026-03-24

## Contexto

Eres un agente trabajando en `ods-unified-v2`, la librería JAX canon del programa ODS.
El repo de papers con toda la documentación actualizada está en:
**https://github.com/r-gabadr/ods-papers**

Lee estos ficheros PRIMERO (están actualizados a hoy):
- `papers/MASTER_PLAN_2026.md` — plan completo con estado, datos overnight, piezas nuevas
- `papers/latex/p03/p03_outline.md` — outline P03 con claims + datos + nuevas métricas
- `papers/latex/p04/p04_outline.md` — outline P04
- `papers/latex/p05/p05_outline.md` — outline P05 con resultados overnight + decisiones

## Estado del repo ods-unified-v2

- **Tests pasando**: ~100+ en `tests/` (repo principal) + 47 claim tests en `ods-papers/tests/`
- **Último trabajo de Codex**: tick_runtime con synth modal, contratos espectrales, topología β₀/β₁/β₂, trazas en disco. 27/27 tests de tick_runtime pasando.
- **Uncommitted**: 11 ficheros relevantes en `src/` y `tests/` (spectral_monitor, tick_runtime, control_dsl, launchers)
- **991 ficheros changed total** — la mayoría es ruido fuera de `src/` y `tests/`

## Tu TODO list (en orden de prioridad)

### 1. Cerrar trazas y telemetría (runtime)
- [ ] Meter `spectral_contract` y `mode_synth` en traza estructurada (control_trace o trace nuevo)
- [ ] Decidir si `topological_drift` entra como observable DSL (`topo.drift`)
- [ ] Tests de estabilidad duros:
  - No spurious energy creation del synth
  - Monotonicidad de damping
  - No flapping del contrato

### 2. Git cleanup (IMPORTANTE)
- [ ] Separar los 11 ficheros relevantes (`src/`, `tests/`) del ruido
- [ ] Commitear por bloques: runtime, tests, docs
- [ ] NO tocar `_claude/` — eso es territorio de Claude Code, solo lectura para ti

### 3. Formalización para papers
Lee los outlines en ods-papers. Lo que necesitan los papers del código:

**P03 necesita** (de `src/ods_unified_v2/jax_runtime/`):
- `observability.compute_commutator_defect` — ya existe, verificado
- `tcl_spectral.build_tcl_spectral_report` — ya existe, verificado
- `amr.rotational_tension_map` — ya existe, verificado
- `gamma_virtual.gamma_focus_score` — ya existe, verificado
- **NUEVO**: χ trigger composite (dimensionless validity functional)
  ```python
  def compute_chi(state, params) -> float:
      """Dimensionless validity functional for t→τ handoff."""
      # Components: η_ratio, Kn_local, vorticity_ratio, BKM_sentinel
  ```
- **NUEVO**: `spectral_coherence(eigenvalues)` y `spectral_entropy(eigenvalues)`
  ```python
  def spectral_coherence(lambdas):
      return jnp.sum(lambdas**2) / jnp.sum(jnp.abs(lambdas))**2

  def spectral_entropy(lambdas):
      p = jnp.abs(lambdas) / jnp.sum(jnp.abs(lambdas))
      return -jnp.sum(p * jnp.log(p + 1e-12))
  ```

**P05 necesita**:
- Contractivity verified empirically (data in `ods-papers/experiments/overnight_results/`)
- **NUEVO**: α interpolation between regimes
  ```python
  def regime_blend(state, chi, chi0=0.5, eps=0.1):
      alpha = jax.nn.sigmoid((chi - chi0) / eps)
      return (1 - alpha) * evolve_t(state) + alpha * evolve_tau(state)
  ```
- **NUEVO**: inter-slice consistency metric
  ```python
  def slice_consistency(field_slices):
      """E_slice = Σ_{ℓ≠ℓ'} |Π_ℓ Ψ - T_{ℓ→ℓ'}(Π_{ℓ'} Ψ)|²"""
  ```

### 4. Documentación
- [ ] Actualizar CLAUDE.md con nuevos módulos (spectral_contracts, mode_synth, launchers)
- [ ] Tabla de módulos: observa / decide / actúa / audita
- [ ] NO crear nuevos docs en `docs/theory/` — ya hay 21 ficheros, no más

### 5. AGENTS.md
- [ ] Revisar y actualizar con synth, contratos y topología

## Reglas CRÍTICAS

1. **Shape canon: Psi[..., 8]** — Clifford en último eje. SIEMPRE.
2. **No tocar `_claude/`** — es de Claude Code, solo lectura
3. **No crear ficheros .md nuevos en docs/** salvo actualizar los existentes
4. **Commitear por bloques** — no meter todo en un snapshot gigante
5. **Los tests deben pasar** — ejecutar `pytest tests/ -q` antes de cada commit
6. **47 claim tests en ods-papers** deben seguir pasando — son el contrato de los papers

## Overnight data disponible (en ods-papers repo)

Si necesitas datos para implementar algo:
- Contractivity: `experiments/overnight_results/07_runtime_maps_contractivity.json`
- Hutchinson: `experiments/overnight_results/02_hutchinson_convergence.json`
- Commutator sweep: `experiments/overnight_results/03_commutator_sweep.json`
- Regime separability: `experiments/overnight_results/05_regime_separability.json`
- Fixed point: `experiments/overnight_results/06_fixed_point_uniqueness.json`
- Texture stability: `experiments/overnight_results/04_texture_stability.json`
