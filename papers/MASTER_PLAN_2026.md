# Plan Maestro de Papers — 2026

Documento de trabajo. Última revisión: 2026-03-24

---

## Estado actual

```
P00  Genesis-D                    ✅ Zenodo DOI 10.5281/zenodo.18929805
P01  CT Runtime Cl(3,0)           ✅ Zenodo DOI 10.5281/zenodo.18995428
P02  Regímenes y Vórtices         ✅ Zenodo (subido 2026-03-23)
P03  Observabilidad               📝 Outline + 16 tests ✓
P04  Memoria documental           📝 Outline + 13 tests ✓
P05  Extensión multiescala        📝 Outline + 8 tests ✓ + overnight data
P06  Control DSL                  📋 Documentación Codex lista
```

**47 claim tests pasando** en total. Repo: https://github.com/r-gabadr/ods-papers

---

## Cadena de dependencias

```
P01 → P02 → P03 → P04 → P05 → P06
              ↓
         P03 observabilidad = prerequisito para P05 (χ trigger)
```

---

## P03 — Observability, Spectral Coherence, and Topological Control

### Claims (5 verificados con tests)

1. **Commutator defect** mide no-conmutatividad t/τ
2. **λ₂ Fiedler** rastrea coherencia inter-slice
3. **Tensión rotacional** localiza actividad topológica
4. **gamma_focus_score** como puente observabilidad→acción
5. **Defect stabilizer** preserva spin index

### Datos overnight disponibles

- `03_commutator_sweep.json` — 126 configs (42 × 3 seeds)
- `05_regime_separability.json` — Cohen's d: 15-415 entre Thomas/Spinor

### Piezas nuevas a incorporar (review GPT 2026-03-24)

**χ como trigger formalizado (dimensionless + regular + monotónico):**
```
χ = w₁·(η_crit/η) + w₂·Kn_loc + w₃·|ω|/ω_ref + w₄·B(t)
```
Propiedades requeridas:
- χ ∈ C⁰(Ψ), idealmente C¹
- ∂_Ψ χ · δΨ ≥ 0 en régimen crítico (monotonicidad)
- Todos los términos adimensionales

**Coherencia espectral (nueva métrica intermedia):**
```
C_spec = Σ_k λ_k² / (Σ_k |λ_k|)²
```
Mide dispersión del espectro. Conecta gap (local) con torsión (global).

**Entropía espectral explícita:**
```
H_spec = -Σ_k p_k log p_k,   p_k = |λ_k| / Σ_j |λ_j|
```
Conecta con d_eff, τ, y χ. Ya implícito en el código, falta explicitar.

**Σ_par = Ψ_par·Ψ̃_par** (Remark, no claim):
Observable comprimido del sector par. Magnitud escalar + quiralidad pseudoescalar.
Solo para sector par (escalar + bivector). Gemini overclaims esto para multivectores generales.

**Torsión espectral T(D)** (Future work):
Invariante refinado de operadores Hodge-Dirac no minimales (arXiv 2025).
Candidato para P03 appendix o futuro.

### Paper-ready claim sobre handoff t→τ
```
"We define an operational breakdown criterion for the continuum closure,
based on vorticity concentration, local dissipation scale, and validity loss.
When exceeded, dynamics transfer from the fast clock t to the structural
clock τ, where unresolved complexity is absorbed by slice-structured
Clifford states."
```
NO decimos "NS3D explota". Decimos "el modelo continuo sale de su dominio de validez".

---

## P04 — Document Memory and τ Cascade

### Claims (5 verificados con tests)

1. Document injection via spinor fingerprint altera campo mediblemente
2. Assignment energy E[n,m] da acoplamiento principiado source→group
3. Tau cascade gates asociativo vía umbral de coherencia vectorial
4. Virtual population mantiene dinámica spinorial coherente
5. Memory curvature κ rastrea calidad de asimilación

### Sin cambios por el review de hoy. P04 está limpio.

---

## P05 — Multiscale Contractive Extension (Hutchinson variant)

### Claims (3 verificados con tests + overnight)

1. Mapas individuales contractivos empíricamente (c_max < 1, 100%)
2. Operador promediado T_ℓ tiene punto fijo único (20 starts → 0 distancia)
3. Hutchinson converge para configs conservadoras (r≤0.80)

### Datos overnight disponibles

- `01_extended_contractivity.json` — 5 grids × 4 configs × 500 pares
- `02_hutchinson_convergence.json` — 4 configs × 50 iters
- `04_texture_stability.json` — 7 niveles de perturbación
- `06_fixed_point_uniqueness.json` — 20 starts convergiendo a 0
- `07_runtime_maps_contractivity.json` — gamma_flow, defect_stab, laplacian

### Resultado clave del overnight

| Config | r | δ | ¿Hutchinson converge? |
|--------|---|---|----------------------|
| conservative (r=0.70, δ=0.10) | ✅ ratio=0.79 | |
| moderate (r=0.80, δ=0.15) | ✅ lento, ratio=0.95 | |
| aggressive (r=0.85, δ=0.20) | ❌ satura en ~44, ratio≈1.0 | |
| boundary (r=0.90, δ=0.15) | ❌ satura en ~55, ratio≈1.0 | |

**Decisión:** Main claim = multiscale contractive. Hutchinson = claim condicional (r+L < ~0.85).

### Piezas nuevas a incorporar (review GPT 2026-03-24)

**Ciclo lifting-evolución-proyección (formalizar):**
```
x_{n+1} = Ψ ∘ T ∘ Φ(x_n)
```
Φ = FFT/lifting Clifford, T = dinámica en slices/rotors, Ψ = proyección.

**Interpolación suave α = σ(χ) entre regímenes:**
```
Ψ_{n+1} = (1-α)·F_t(Ψ_n) + α·S_λ ∘ F_τ(Ψ_n)
α = σ((χ(Ψ_n) - χ₀) / ε)
```
Propiedad requerida: ∂_τ F(Ψ) ≤ 0 (parte τ siempre disipa).

**Condición de contractividad explícita:**
```
|Ψ_{n+1} - Ψ*| ≤ ρ|Ψ_n - Ψ*|,  ρ < 1
```
O más débil: F(Ψ_{n+1}) ≤ F(Ψ_n).

**Consistencia entre slices (formalizar):**
```
E_slice = Σ_{ℓ≠ℓ'} |Π_ℓ Ψ - T_{ℓ→ℓ'}(Π_{ℓ'} Ψ)|²  ≤  ε
```
Cierra multislice + transporte + coherencia global.

**Textura multiescala:**
```
Θ_ms(Ψ) = Σ_{k=0}^{d} ρ^k · Θ(T^k(Ψ)),  0 < ρ < 1
```
Resultado overnight: single-scale 3.1-3.5× más estable que multiescala.
Separabilidad de regímenes: Cohen's d = 15-415 (brutal).

**Remark: Γ_virt dependiente del estado (Future extension):**
```
Γ_virt^μ = Σ_a β_a(Ψ) · Γ_a^μ
```
No incluir como claim. Sí como Remark/Future porque:
- Ya existe estructura Clifford
- Encaja con Dirac dependiente del estado
- No es testeable ahora

---

## P06 — Control DSL (futuro, post-P05)

### Fuente: documentación Codex (revisada 2026-03-24)

Cuatro documentos en `docs/theory/` y `docs/runtime/` definen:
- Capa de control declarativo: tick→tau→render
- Control no toca Ψ directamente, solo modula operadores
- Grammar: `every N ticks, when Cond, for M tau`
- 5 targets: attractor.gain, coupling.strength, budget.cells, render.rate, fluid.eta
- Decision gate: ≤20% latency penalty

### Piezas de arXiv relevantes (review 2026-03-24)

- **XLA SSM compiler-first** (arXiv 2603.09555): cache O(1) on-device, sin Python loops
- **PolyBlocks MLIR** (arXiv 2603.06731): tiling + fusion + mapping analítico

Conexión: χ como trigger de régimen de compilación, no solo runtime logic.

---

## Ideas futuras (post-papers, NO abrir ahora)

### Decoder Ψ→visual (d-lidar, Genie 3)
Proyecto separado. Requiere pares (campo, imagen) para entrenar decoder.
Golf insights útiles: spectral mixer, quantization int8+zlib, tiny models <16MB.

### Analogía Navier-Stokes ↔ CT runtime
Si encontramos excepciones de contractividad (c≥1), comparar con singularidades NS.
Misma estructura: campo vectorial en grid periódica, splitting advección/difusión,
conservación enstrofía↔E₁₂, vórtices topológicos, vorticidad↔tensión rotacional.
Diferencia: NS sale de primeros principios, CT de energía libre construida.
NO decir "hemos resuelto NS3D". SÍ decir "handoff cuando el closure falla".

### Clifford wavelet frame
```
Ψ(x) = Σ_{ℓ,r} Σ_{j,ξ} c_{j,ξ}^{(ℓ,r)} · φ_{j,ξ}^{(ℓ,r)}(x)
```
Base en literatura (Brackx, Sommen). Extensión natural del FFT spectral mixer.

### Multiespectro dinámico con alineamiento spinorial
```
Ψ = Σ c_{ℓmk} · R_ℓ · φ_{ℓmk}(Γ_virt, ω_ℓm) · R̃_ℓ
```
Bonito formalmente. No testeable ahora. P07+ material.

### arXiv leads (Pulse GPT 2026-03-23)
- Slice-monogenic structures: clases admisibles para campos Clifford ⭐
- Spectral torsion T(D): invariante refinado ⭐
- Dirac on graphs/trees: modelo discreto con espectro exacto ✅
- q-Dirac deformado: especulativo ⚠️
- Bohmian-Clifford: puente conceptual ⚠️

---

## Issues matemáticos globales

| ID | Issue | Estado |
|----|-------|--------|
| I1 | Shape convention Psi[...,8] vs [B,S=8,2,H,W] | Documentado, cada paper lo declara |
| I2 | Lyapunov exponents Genesis-D (λ₁=+0.2635...) | ⚠️ Pendiente verificación |
| I3 | Grade weights (1, 1, 0.5, 1/6) motivation | ⚠️ Pendiente documentar |
| I4 | Genesis checkpoint = prior empírico, no físico | ✅ P02 lo dice correctamente |
| I5 | OAM gauge dependence atan2(B₁₃, B₁₂) | ✅ Documentado como convención |

---

## Errores de Gemini/GPT filtrados (NO incluir en papers)

- "ΨΨ̃ elimina vectores" → FALSO para multivectores generales (solo sector par)
- "Hemos sentenciado a muerte a la física clásica" → narrativa
- "a=f(a) es autoconsciencia del universo" → filosofía, no math
- "Loop Quantum Gravity connection" → sin puente formal
- "Los espinores son partículas VFX del universo" → metáfora
- "Podemos calcular dónde peta NS3D exactamente" → NO (problema Millennium abierto)

---

## Timeline

| Fecha | Tarea |
|-------|-------|
| 2026-03-23 | P02 subido a Zenodo ✅ |
| 2026-03-23 | Repo ods-papers creado + Zenodo conectado ✅ |
| 2026-03-23 | Overnight tests (7 experiments) completados ✅ |
| 2026-03-24 | Review GPT + arXiv leads + documentación actualizada ✅ |
| 2026-03-24+ | Escribir .tex P03 con datos overnight |
| Siguiente | Escribir .tex P04 |
| Siguiente | Escribir .tex P05 con Hutchinson condicional |
| Futuro | P06 Control DSL (post-P05) |

---

## Historial

| Fecha | Acción |
|-------|--------|
| 2026-03-09 | Genesis-D (P00) → Zenodo |
| 2026-03-13 | P01 CT runtime → Zenodo |
| 2026-03-22 | Creación plan maestro |
| 2026-03-23 | P02 → Zenodo. 47 tests. Repo ods-papers. Overnight tests. |
| 2026-03-24 | Review integral: docs Codex + GPT referee + arXiv leads. Sin contradicciones. |
