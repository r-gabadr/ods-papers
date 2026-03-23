# Library Cleanup Plan — ods-unified-v2

Hoja de ruta para pasar la librería a limpio **sin tocar genesis_jax**.
Fecha: 2026-03-22

---

## Principio

ods-unified-v2 es la librería canon verificable. El criterio de "limpio" es:
cada módulo tiene claims claros, tests que los prueban, y sin código muerto.

genesis_jax queda intacto. Las mejoras fluyen: ods → genesis (nunca al revés sin gate).

---

## Estado actual

- ✅ 100 tests pasando
- ✅ Estructura `jax_runtime/` modular
- ✅ `clifford_operator.py` — núcleo 3D
- ✅ `wilson.py` — operador canónico 2D
- ✅ `oam_groups.py`, `tau_assimilation.py`, `slice_attractor.py`
- ⚠️ `dirac_from_scalar_channel_only` — implementación incorrecta (fix pendiente)
- ⚠️ Pesos de grado `(1, 1, 0.5, 1/6)` sin documentación
- ⚠️ `lambda_sigmoid` — K sin normalización

---

## Fases de limpieza

### Fase 0 — Preparación (no rompe nada)

**Objetivo:** documentar y marcar sin cambiar comportamiento.

| Tarea | Módulo | Acción |
|-------|--------|--------|
| Documentar pesos de grado | `clifford_operator.py:143` | Añadir docstring con justificación |
| Marcar `dirac_from_scalar_channel_only` | `clifford_operator.py:222` | `# TEST-ONLY: not used in production` + nota en docstring |
| Documentar rango esperado de K | `clifford_operator.py:249` | Añadir nota sobre normalización |
| Crear `tests/paper_claims/` | `tests/` | ✅ Ya creado |
| Crear CSVs de benchmarks | `scripts/export_p02_csv.py` | ✅ Ya creado |

---

### Fase 1 — Core fixes (antes de submit P02)

**Objetivo:** corregir bugs que afectan citas en papers.

| Fix | Módulo | Descripción | Impacto en papers |
|-----|--------|-------------|-------------------|
| F1.1 | `clifford_operator.py:222` | Corregir `dirac_from_scalar_channel_only` (stacks mismo gradiente 3 veces) | No afecta P01/P02 pero evita confusión si se cita |
| F1.2 | `clifford_operator.py:249` | Añadir normalización o documentar rangos para `lambda_sigmoid` | Rigor matemático P02+ |
| F1.3 | `clifford_operator.py:143` | Documentar origen de `(1, 1, 0.5, 1/6)` | Honestidad en P02 |

**Proceso para cada fix:**
1. Escribir test que falla con el bug actual (si aplica)
2. Corregir
3. Verificar que los 100 tests siguen pasando
4. Actualizar docstring con justificación

---

### Fase 2 — Tests de claims (paralelo a P02)

**Objetivo:** cada claim de cada paper tiene un test que lo verifica directamente.

```
tests/paper_claims/
├── test_p01_claims.py   ✅ creado (5 tests, P01 Zenodo)
├── test_p02_claims.py   ✅ creado (5 tests, P02 pendiente)
├── test_p03_claims.py   — cuando P03 tenga claims cerrados
└── README.md            ✅ creado
```

**Regla:** un claim no está "probado" hasta que hay un test con ese número de claim en el nombre y pasa en CI.

---

### Fase 3 — Reorganización de módulos experimentales

**Objetivo:** separar claramente core estable de experimental.

Situación actual:
```
jax_runtime/
├── [CORE ESTABLE]          — no tocar
│   ├── algebra.py
│   ├── state.py
│   ├── operators.py
│   ├── spectral_pde.py
│   ├── wilson.py
│   ├── energy.py
│   ├── transport.py
│   ├── integrator.py
│   ├── metrics.py
│   ├── attractors.py
│   ├── oscillator.py
│   ├── solitons.py
│   └── soliton_tracker.py
│
├── [ESTABLE CON CAVEATS]   — documentar limitaciones
│   ├── clifford_operator.py  (fix F1.1-F1.3 pendiente)
│   ├── oam_groups.py         (OAM gauge-dependiente, documentar)
│   ├── tau_assimilation.py
│   └── slice_attractor.py
│
└── [EXPERIMENTAL]          — no citar en papers sin ablación
    ├── cmcf.py
    ├── gamma_virtual.py
    ├── amr.py
    ├── pulse_stabilizer.py
    ├── defect_stabilizer.py
    ├── tcl_spectral.py
    └── tcl_offline.py
```

**Acción:** añadir header en cada módulo experimental:
```python
# STATUS: EXPERIMENTAL — not yet cited in any paper.
# Do not include in paper claims without completed ablation.
```

---

### Fase 4 — Documentación matemática inline

**Objetivo:** cada función matemáticamente no-trivial tiene su fórmula en el docstring.

Prioridad (por orden de aparición en papers):

| Función | Módulo | Fórmula a documentar |
|---------|--------|----------------------|
| `free_energy_frozen` | `energy.py` | F = E - TH |
| `rev_transport_12` | `transport.py` | ∂_t(v,β) = ω₁₂[[0,I],[-I,0]](v,β) |
| `diss_transport_12` | `transport.py` | ∂_t(v,β) = -γ₁₂[[I,-I],[-I,I]](v,β) |
| `S_K` | `clifford_operator.py` | S_K(ψ) = ν·Δ_h·ψ - λ(K)·F(D_h·ψ) |
| `extract_oam_groups` | `oam_groups.py` | q_m = |mean(e^{iφ_B})| con φ_B = atan2(B₁₃, B₁₂) |
| `dirac` | `clifford_operator.py` | D_h ψ = Σ_a γ_a ∂_{h,a} ψ |

---

### Fase 5 — Integración con genesis_jax (post-P02)

**Regla estricta:** solo después de que P02 esté enviado a Zenodo.

Módulos de ods listos para portar a genesis (en orden):

| Módulo ods | Destino genesis | Valor |
|-----------|----------------|-------|
| `observability.py` (`commutator_defect`) | `core/observability.py` | Health check del loop |
| `soliton_tracker.py` | `memory/soliton.py` | Tracking temporal |
| `oam_groups.py` | `core/oam_groups.py` | Observable nuevo |
| `tau_assimilation.py` | reemplaza `free_energy.step_entropic` | Tau real con OAM |

**Protocolo de port:**
1. Copiar módulo de ods a genesis
2. Adaptar shapes: `Psi[...,8]` → `field[B,S=8,2,H,W]` donde necesario
3. Tests rápidos en genesis: `test_chaos.py`, `test_spinor.py`, `test_pde_parity.py`
4. Verificar que server arranca limpio

---

## Checklist de "librería limpia"

Para cada módulo del core estable, marca cuando se cumpla:

- [ ] Docstrings con fórmulas matemáticas
- [ ] Tests en `paper_claims/` si tiene claim en paper
- [ ] Header de status (CORE ESTABLE / ESTABLE CON CAVEATS / EXPERIMENTAL)
- [ ] Sin código comentado / dead code
- [ ] `__all__` actualizado

---

## Qué NO hacer

- No refactorizar sin test que lo motive
- No mover módulos experimentales a core solo porque funcionan
- No mergear genesis_jax y ods-unified-v2
- No tocar genesis_jax desde este repo
- No abrir P03 hasta que P02 esté enviado
