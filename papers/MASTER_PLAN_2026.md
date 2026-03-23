# Plan Maestro de Papers — 2026

Documento de trabajo con Claude Code.
Última revisión: 2026-03-22

---

## Visión general: dos programas, una base matemática

```
genesis_jax (agente rico)          ods-unified-v2 (core limpio)
─────────────────────────          ────────────────────────────
P00 Genesis-D ✅ Zenodo            P01 CT runtime ✅ Zenodo
P01 Field substrate (abierto)      P02 Regímenes  ← PRÓXIMO
P02 Attractor family (abierto)     P03 Observabilidad
P03 Tokenless repr. (abierto)      P04 Memoria/neuro_field
P04 Memory/RAG (abierto)           P05 Benchmark
P05 Eval benchmarks (futuro)       P06 Síntesis
P06 Runtime API (futuro)
```

**Base matemática compartida:** `Cl(3,0)`, estado `Psi[..., 8]`, `F = E - TH`, doble reloj `t/τ`.

**Diferencia operativa:**
- `genesis_jax`: LLM, MCP, RAG, chaos bio, FNO, loop interno rico
- `ods-unified-v2`: core verificable, JAX puro, Wilson-Dirac, OAM groups, ablaciones limpias

---

## 1. Revisión de papers ya publicados en Zenodo

### P00 — Genesis-D (Zenodo 10.5281/zenodo.18929805, 2026-03-09)

**Estado:** Publicado. Ancla histórica del programa.

**Claims que necesitan verificación cruzada con código:**

| Claim | Código asociado | Verificado |
|-------|----------------|------------|
| Atractor hipercaótico con λ₁=+0.2635, λ₂=+0.1583, λ₃=+0.0161 | `genesis_jax/core/chaos.py` | ⚠️ pendiente |
| D_KY = 5.44 (Kaplan-Yorke) | `genesis_jax/core/chaos.py` | ⚠️ pendiente |
| Memoria topológica con Q=0 (temporal) y Q=±1 (estable) | `genesis_jax/memory/` | ⚠️ pendiente |
| FNO con aprendizaje espectral por grade | `genesis_jax/core/fno.py` | ⚠️ pendiente |
| Campo espinorial de 8 planos en Cl(3,0) | `genesis_jax/core/clifford.py`, `spinor.py` | ⚠️ pendiente |

**Issues matemáticos a revisar:**
- Los exponentes de Lyapunov son afirmaciones numéricas muy específicas — ¿se generan reproduciblemente desde el código? ¿Hay script/notebook que los calcule?
- La dimensión KY=5.44 implica sistema 6D efectivo — ¿cuál es la conexión exacta con el campo Cl(3,0) de 8 componentes?
- "8-plane spinor field" en genesis_jax usa shape `[B, S=8, 2, H, W]`, que es distinto al `Psi[..., 8]` de ods-unified-v2. El paper debe ser explícito sobre cuál convención usa.

**Acción:** Crear `docs/papers/P00_REVIEW.md` en genesis_jax con verificación específica.

---

### P01 — CT Runtime (Zenodo 10.5281/zenodo.18995428, 2026-03-13)

**Estado:** Publicado. Base sólida.

**Claims verificados:**
- ✅ F = E - TH con monotonía en τ (tests pasan)
- ✅ Transporte 1↔2 reversible y disipativo (test conservación E12)
- ✅ Suite 55 tests (32 subset P01 estructural)
- ✅ Implementación JAX en Cl(3,0), estado `Psi[..., 8]`

**Issues menores a corregir en versión actualizada:**
- `dirac_from_scalar_channel_only` en `clifford_operator.py:222` — implementación incorrecta (stacks mismo gradiente 3 veces, luego promedia). No afecta los claims de P01 pero debe corregirse antes de citar ese método.
- Pesos de grado `(1, 1, 0.5, 1/6)` sin motivación documentada. Agregar nota en texto.

---

## 2. Orden de producción de papers restantes

### Criterio de prioridad

```
P02 (ods) → P03 (ods) → P01-gj (genesis) → P02-gj (genesis) → P04 (ods) → P05 (ods)
```

La lógica: el core limpio de ods-unified-v2 es la base verificable. genesis_jax papers vienen después porque dependen de ese core.

---

## 3. P02 (ods-unified-v2) — INMEDIATO

**Título:** Regimes, Soliton Structure, and Inter-Grade Modulation in a Clifford-Thomas Runtime

**Estado:** Todos los datos listos. Solo falta narrativa + figuras finales.

### Datos disponibles

- Perfil ganador: `genesis_phase_signed_64_seed7_spectral_96`
- Gaps reproducibles: `structured_vortex_step_gap = 0.2917`, `baseline_vortex_step_gap = 0.2604`
- Replica consistente: `genesis_phase_signed_64_seed17_spectral_96`
- Artefactos: `artifacts/p02_benchmarks_v2/summary.json` + figuras en `docs/site/assets/`

### Tabla de resultados congelada (de P02_RESULTS_V0.md)

| Caso | F final | d_eff | Alignment | Vortex fraction | Mean complexity |
|------|---------|-------|-----------|----------------|-----------------|
| thomas_baseline | -309.03 | 2.47 | 0.0111 | 0.583 | 5.97 |
| thomas_structured | -355.78 | 2.65 | 0.0130 | 0.542 | 6.00 |
| spinor8d_baseline | -316.33 | 2.51 | 0.0120 | 0.844 | 6.06 |
| spinor8d_structured | -362.37 | 2.69 | 0.0139 | 0.833 | 6.10 |

**Señal principal:** Spinor8D mantiene persistencia vorticial temporal mayor que Thomas.
La separación es robusta en dos seeds con el observable `(3,6)`.

### Estructura de paper propuesta

```
1. Introduction (½ pág)
   - Motivación: runtime de campo multivectorial computacionalmente testable
   - Puente con P01: F=E-TH, transporte 1↔2, τ ya establecidos

2. Framework (1 pág)
   - Cl(3,0), Psi[H,W,8], Thomas3D, SpinorAttractor8D
   - Transporte 1↔2 (reversible + disipativo)
   - Detector de solitones (blob vs vortex, complejidad)

3. Experimental Setup (½ pág)
   - 4 configuraciones: baseline/structured × Thomas/Spinor8D
   - Backend: spectral (paper) vs laplacian (smoke)
   - Observables: F, d_eff, alignment, vortex_count(t), complexity

4. Results (1½ pág)
   - Tabla comparativa completa
   - Figura temporal: F(step), d_eff(step), vortex_count(step)
   - Gap estructural: structured_vortex_step_gap = 0.2917

5. Discussion / Limits (½ pág)
   - Genesis checkpoint = prior estructurado empírico (no prior físico verificado)
   - Observable (3,6) > snapshot final como discriminador
   - LO QUE NO AFIRMA: no cuantización, no Lindblad, no gamma completa, no atlas universal

6. Conclusion (¼ pág)
```

### Checklist de cierre P02

- [ ] Ejecutar benchmark final en tamaño estable (spectral, 96 pasos)
- [ ] Generar tabla comparativa limpia (LaTeX)
- [ ] Verificar figura temporal `p02_temporal_winner.png` tiene leyenda correcta
- [ ] Redactar secciones 1-6
- [ ] Tabla de claims y límites explícita
- [ ] Verificar que los 100 tests siguen pasando
- [ ] Compilar PDF
- [ ] Submit Zenodo

---

## 4. P03 (ods-unified-v2)

**Depende de:** P02 enviado.

**Pregunta:** ¿Puede el runtime extender la observabilidad básica y acoplar memoria documental sin romper los gates del core?

**Módulos listos:** `commutator_defect`, `tcl_spectral.py`, `amr.py`, `rotational_metrics.py`, `pinning_metrics.py`.

**Figuras ya generadas:** `p03_capture_score.svg`, `p03_commutator_defect.svg`.

**Pendiente:** Probes A/B con mismo drive y distinto paisaje documental. Narrativa.

---

## 5. P01-gj (genesis_jax) — Field Substrate

**Depende de:** P01 ods publicado (ya está).

**Pregunta:** ¿Cuál es el sustrato de campo canon de genesis_jax y sus operadores mínimos?

**Módulos:** `core/clifford.py`, `core/spinor.py`, `core/pde.py`, `core/entropy.py`, `core/dirac.py`.

**Issue principal:** La convención de shape `[B, S=8, 2, H, W]` de genesis_jax debe reconciliarse formalmente con la notación `Psi[...,8]` de ods. El paper debe aclarar que son dos implementaciones del mismo álgebra, no dos álgebras distintas.

---

## 6. P02-gj (genesis_jax) — Attractor Family

**Depende de:** P01-gj cerrado.

**Pregunta:** ¿Cómo se organiza la familia de atractores de genesis_jax y qué regímenes produce?

**Módulos:** `core/chaos.py`, `core/brainwave.py`, `core/koopman.py`.

**Nota:** Los exponentes de Lyapunov de Genesis-D (P00) deben verificarse aquí formalmente.

---

## 7. Mapa de consolidación genesis_jax ↔ ods-unified-v2

### Lo que ods-unified-v2 tiene que genesis_jax no tiene (para portar si se necesita)

| Pieza | ods módulo | Impacto en genesis |
|-------|-----------|-------------------|
| Wilson-Dirac S_K dependiente del estado | `wilson.py`, `clifford_operator.py` | Mejora el loop interno |
| OAM groups + coherencia | `oam_groups.py` | Observable nuevo para P02-gj |
| Tau assimilation completo | `tau_assimilation.py` | Reemplaza `step_entropic` de genesis |
| Commutator defect | `observability.py` | Health check del loop |
| Soliton tracker temporal | `soliton_tracker.py` | Mejora memory/soliton.py |
| AMR topológico | `amr.py` | Actividad mask + tension map |

### Lo que genesis_jax tiene que ods no tiene (valor diferencial)

| Pieza | genesis módulo | Valor |
|-------|---------------|-------|
| Bio-attractors | `core/chaos.py` | Hodgkin-Huxley, Ramachandran, etc. |
| Brainwave modes | `core/brainwave.py` | Modos neurales explícitos |
| Koopman/DMD | `core/koopman.py` | Análisis espectral del campo |
| FNO/AFNO | `core/fno.py`, `afno.py` | Aprendizaje espectral por grade |
| RAG + spinor embed | `rag/` | Memoria semántica |
| API OpenAI-compat | `api/server.py` | Exposición como servicio |
| Tokenless route | `core/route0.py`, `spectral_codec.py` | Representación sin tokens |

### Regla de integración

**No fusionar los repos.** Usar ods-unified-v2 como "núcleo de librería" y genesis_jax como "capa de aplicación". Las mejoras de core van primero a ods, luego se portan a genesis vía adapter.

Cuando un módulo de ods está estable + testeado → se porta a genesis en `core/` via PR controlada.

---

## 8. Issues matemáticos globales a resolver

### I1. Convención de shape (CRÍTICO)

| Repo | Shape | Convención |
|------|-------|-----------|
| ods-unified-v2 | `Psi[..., 8]` | Clifford en último eje |
| genesis_jax | `field[B, S=8, 2, H, W]` | S=planos spinoriales, 2=real/imag |

**Acción en papers:** Cada paper debe declarar explícitamente cuál convención usa. No asumir que el lector conoce ambas.

### I2. Exponentes de Lyapunov de Genesis-D (VERIFICAR)

Las afirmaciones λ₁=+0.2635, λ₂=+0.1583, λ₃=+0.0161, D_KY=5.44 son cuantitativas y deben ser reproducibles desde código. Localizar el script que los generó.

### I3. Pesos de grado (1, 1, 0.5, 1/6) en clifford_operator.py

Sin motivación formal en ningún documento. Antes de citar ods-P02 o siguientes, agregar justificación (dimensional, empírica o convencional).

### I4. Genesis checkpoint como prior

P02 resulta ya lo dice correctamente: Genesis checkpoint = prior estructurado empírico, no prior físicamente verificado. Asegurar que Genesis-D (P00) tampoco hace afirmaciones de "convergencia física" que no estén respaldadas.

### I5. OAM efectiva y dependencia de gauge

La coherencia OAM usa `atan2(B₁₃, B₁₂)` como fase bivectorial. Esta elección de componentes rompe simetría. Debe documentarse como convención operativa, no como invariante.

---

## 9. Timeline propuesto

| Semana | Tarea |
|--------|-------|
| 2026-03-22 → 29 | Redactar narrativa P02 (ods) + figuras |
| 2026-03-29 → 04 | Submit P02 (ods) a Zenodo |
| 2026-04 | Verificar claims numéricos Genesis-D con código |
| 2026-04 | Empezar P03 (ods) — memoria documental |
| 2026-04/05 | P01-gj (genesis) — Field substrate paper |

---

## Historial

| Fecha | Acción |
|-------|--------|
| 2026-03-09 | Genesis-D enviado a Zenodo (P00) |
| 2026-03-13 | P01 CT runtime enviado a Zenodo |
| 2026-03-22 | Creación de este plan, revisión completa con Claude Code |
