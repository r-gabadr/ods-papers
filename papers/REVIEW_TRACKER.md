# Paper Review Tracker

Documento de seguimiento mantenido con Claude Code.
Última revisión: 2026-03-22

---

## Estado global

| Paper | Título | Código | Tests | Datos | Narrativa | Figuras | Claims | Estado |
|-------|--------|--------|-------|-------|-----------|---------|--------|--------|
| P01 | CT runtime Cl(3,0) | ✅ | ✅ | ✅ | ✅ | ? | ✅ | Cerrado/publicado |
| **P02** | **Regímenes, solitones, modulación inter-grade** | ✅ | ✅ | ✅ | ❌ | ⚠️ | ⚠️ | **PUBLICABLE — pendiente redacción** |
| P03 | Observabilidad, TCL, AMR | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ | Abierto |
| P04 | Memoria documental + neuro_field | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | Abierto |
| P05 | Benchmark de sistema | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Futuro |
| P06 | Síntesis / tesis | — | — | — | — | — | — | Futuro |

Leyenda: ✅ completo · ⚠️ parcial · ❌ pendiente

---

## P02 — Prioridad inmediata

**Título:** Regimes, Soliton Structure, and Inter-Grade Modulation in a Clifford-Thomas Runtime

**Claim central:** El runtime CT puede expresar regímenes dinámicos distinguibles y medir estructuras localizadas útiles dentro del slice práctico Cl(3,0).

### Claims específicos

| # | Claim | Evidencia en código | Verificado |
|---|-------|---------------------|------------|
| C1 | Thomas3D y SpinorAttractor8D son familias distinguibles | `attractors.py`, `artifacts/p02_benchmarks_v2/summary.json` | ✅ |
| C2 | El bloque 1↔2 modifica mediblemente la distribución interna | `transport.py`, benchmark structured vs baseline | ✅ |
| C3 | El detector de solitones distingue blob de vortex con complejidad estructural | `solitons.py`, `soliton_tracker.py` | ✅ |
| C4 | genesis + phase_signed + spectral + 96 = configuración ganadora | `artifacts/p02_benchmarks_v2/summary.json` (structured_vortex_step_gap=0.2917) | ✅ |

### Pendiente para release

- [ ] **Narrativa del paper** — 4-6 secciones: intro, framework, experiments, results, limits
- [ ] **Figuras limpias:**
  - [ ] Tabla Thomas3D vs SpinorAttractor8D (d_eff, energías por grado, alineación, F final)
  - [ ] Tabla transport off vs on (blob/vortex counts, complejidad)
  - [ ] Figura temporal: F(step), d_eff(step) para la configuración ganadora
  - [ ] Tabla de audio: caso → wav → observables finales
- [ ] **Tabla de claims y límites** — qué NO afirma este paper
- [ ] **Sección de límites explícitos:**
  - No cuantización
  - No gamma continua completa
  - No Lindblad físico
  - No Tomita-Takesaki como fundamento operativo
  - Genesis como prior estructurado empírico, no prior físicamente verificado

### Issues técnicos identificados (revisión Claude Code 2026-03-22)

Ver `docs/papers/P02_TECHNICAL_REVIEW.md` para detalles.

Resumen:
- `dirac_from_scalar_channel_only` en `clifford_operator.py:222` — implementación incorrecta (no afecta P02 que usa el operador 2D `wilson.py`)
- `lambda_sigmoid` sin normalización de K — bug latente, documentar rango esperado
- Pesos de grado `(1, 1, 0.5, 1/6)` sin motivación en documentación

### Artefactos disponibles

- `artifacts/p02_benchmarks_v2/summary.json` — resultados completos
- `docs/site/data/p02_summary.json` — versión site
- `docs/site/data/p02_winner_spinor_structured_metrics.json`
- `docs/site/data/p02_winner_thomas_structured_metrics.json`
- `docs/site/assets/p02_leaderboard_gap.svg` — figura leaderboard
- `docs/site/assets/p02_temporal_winner.png` — figura temporal
- `docs/site/assets/p02_winner_d_eff.svg`
- `docs/site/assets/p02_winner_free_energy.svg`

### Próximos pasos concretos

1. Leer `artifacts/p02_benchmarks_v2/summary.json` y extraer los números clave
2. Redactar narrativa mínima (puede ser en español primero, luego EN)
3. Generar figuras faltantes (tablas comparativas)
4. Escribir tabla de claims + límites
5. Submit

---

## P03 — Estado

Abierto. No iniciar hasta que P02 esté en submit.

Módulos disponibles: `commutator_defect`, `tcl_spectral.py`, `amr.py`, `rotational_metrics.py`, `pinning_metrics.py`.

Figuras generadas: `docs/site/assets/p03_capture_score.svg`, `p03_commutator_defect.svg`.

---

## P04 — Estado

Abierto. Línea `lab/neuro_field` en curso. No iniciar hasta P03 tenga módulo estable.

---

## Documentos relacionados

- [MASTER_PLAN_2026.md](MASTER_PLAN_2026.md) — plan completo con ambos repos y timeline

---

## Historial de revisiones

| Fecha | Quién | Qué |
|-------|-------|-----|
| 2026-03-22 | Claude Code | Creación del tracker, revisión técnica P02, issues identificados en clifford_operator.py, plan maestro con genesis_jax |
