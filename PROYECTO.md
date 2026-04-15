# PROYECTO ODS — Brújula Única

> **Este es el único documento que consultas cuando te pierdes.**
> Última actualización: 2026-04-15
> Autor principal: Rubén Gabarri (USAL, doctorado en Física Fundamental y Matemáticas)

---

## 0. Regla de oro

Antes de abrir cualquier otro archivo, terminal, HTML, notebook:

1. **Abre este documento.**
2. **Mira en qué fase estás (§6).**
3. **Mira la lista de tareas activas de esa fase.**
4. **Elige una. Sólo una.**
5. **Cuando acabes, vuelves aquí. Nunca improvises en medio.**

Si te descubres tocando algo que está en **§8 Parking**, para. Vuelve a la fase.

---

## 1. Misión

Construir una arquitectura de aprendizaje basada en:

- **Álgebra de Clifford Cl(3,0)** como espacio de estados interno de cada experto.
- **Campo de Navier–Stokes viscoso compartido** como medio de sincronización entre expertos (en vez del router clásico de un MoE).
- **Variable Distillation Dynamics**: cada experto aprende con una tasa efectiva modulada por su propia coherencia espectral.

El objetivo último es un **sustrato computacional que pueda escalar hacia AGI**, no reproducir transformers con más parámetros. La apuesta es que la geometría (Clifford) + la física de campos (NS) + la modulación por coherencia dan una forma de razonamiento estructuralmente distinta a la attention pura.

**Es una apuesta de investigación, no un producto.** El plazo razonable es años, no meses. Ver §6 para el roadmap sobrio y §7 para el roadmap ambicioso.

---

## 2. Qué es ODS

### En 30 segundos

ODS (Orbital Dissipative System / Ontological Dynamical Substrate) es una arquitectura donde muchos "expertos" con estado geométrico propio (Clifford) se comunican **emitiendo y leyendo** de un campo continuo (NS-viscoso) en lugar de pasar tokens por un router. El campo amortigua, los expertos se sincronizan sin supervisión explícita, y la coherencia del estado interno de cada experto modula cuánto aprende.

### En 3 minutos

Un **experto** tiene un estado `ψ ∈ Cl(3,0)` (multivector de 8 componentes: escalar, 3 vectoriales, 3 bivectoriales, 1 trivectorial). Cuando llega un input:

1. Lee el **campo local** `Φ(x)` en su posición.
2. Multiplica por un **rotor learnable** `R ∈ Cl⁺(3,0)` → `ψ = R·Φ`.
3. Calcula su **coherencia** como la relación entre parte vectorial (grado 1) y bivectorial (grado 2) de su estado.
4. Si coherencia > umbral, **emite** al campo y contribuye al output. Si no, se calla.

El **campo** evoluciona como un Navier–Stokes viscoso discreto: `Φ ← Φ + dt·(ν·ΔΦ − α·Φ) + emisiones`. Los parámetros `ν` (viscosidad) y `α` (disipación) son aprendibles.

El aprendizaje tiene una capa extra: el gradiente que recibe cada experto está **modulado por su propia coherencia**. Expertos coherentes aprenden rápido; los confundidos, despacio. Esto es la *Variable Distillation Dynamics*.

### En 30 minutos

Lee **`book/`** (ver §5). Cada capítulo explica un paper formal.

---

## 3. Identidad e historial

- **Proyecto raíz**: ODS. Nombres anteriores: *Graphophone*, *Claude DJ*, *ODS Unified V2*. Todos apuntan al mismo concepto desde distintos ángulos.
- **Doctorado asociado**: USAL, Física Fundamental y Matemáticas.
- **Colaboradores IA**: Claude (arquitectura + ML stack + docs), Codex (implementación runtime), GPT (revisión teórica ocasional). Coordinación vía `HANDOFF.md` / `TASK_BOARD.md` en `ods-unified-v2/`.
- **Estado científico**: pre-publicación. Hay skeletons P00–P06, el ML stack corre, pero nada está publicado.
- **Lo novedoso confirmado**: *Variable Distillation Dynamics* (búsqueda arxiv verificada negativa, 2026-04).

---

## 4. Mapa de repos — qué vive dónde

Sólo hay **cuatro cosas reales**. Todo lo demás es ruido histórico.

### 4.1 Repos activos

| # | Repo | GitHub | Qué contiene | Importancia |
|---|---|---|---|---|
| **1** | `ods-papers` | `r-gabadr/ods-papers` OK | Papers formales P00–P06 + ML stack publicable + este documento + `book/` divulgativo | **CRÍTICA** (eje del doctorado y del código publicable) |
| **2** | `ods-unified-v2` | **NO está en GitHub** (vive en monorepo privado `/home/r/z_folder/.git`) | Runtime JAX completo, tasks, HANDOFF Claude↔Codex, adapters, experimentos | **CRÍTICA** (decisión pendiente: subir como repo propio — ver §10) |
| 3 | `odsiv` | `r-gabadr/odsiv` | ODSIV v1: spectral core + sandbox + RAG. Proyecto lateral | MEDIA (sistema RAG auxiliar, no eje AGI) |
| 4 | `ods-golf` | `r-gabadr/ods-golf` | (por revisar) | BAJA |

### 4.2 Dentro de `ods-papers`

```
ods-papers/
├── PROYECTO.md              ← este archivo (brújula)
├── README.md                ← entrada pública
├── papers/                  ← LaTeX formal P00–P06
│   ├── MASTER_PLAN_2026.md  ← plan de papers (vigente)
│   ├── NOTATION_TABLE.md    ← notación unificada
│   └── latex/p00..p06/      ← skeletons por paper
├── foundations/             ← teoría + ML stack ejecutable
│   ├── ml/                  ← * EL ML STACK *
│   │   ├── expert.py        ← Expert con rotor Cl(3,0)
│   │   ├── field_step.py    ← Campo NS viscoso
│   │   ├── moe.py           ← CoherenceMoE
│   │   └── train.py         ← Loop optax
│   └── paper/               ← drafts técnicos
├── experiments/             ← notebooks y scripts
├── tests/                   ← tests del repo (incl. paper_claims)
├── book/                    ← * LIBRO DIVULGATIVO * (§5)
├── parking/                 ← * COSAS CONGELADAS * (§8)
└── release_p02/             ← artefactos release P02
```

### 4.3 Dentro de `ods-unified-v2` (local, no-github)

Es donde vive el runtime de verdad, las tareas día a día, y el sistema de coordinación con Codex. No lo toco en esta reorganización: su dueño de facto es Codex vía `HANDOFF.md`. **Decisión pendiente en §10.**

---

## 5. El Libro (`book/`)

Explicación humana, paralela a los papers. No es divulgación light: es el manual interno para que tú mismo no te pierdas, y la base del material de tesis.

### Índice

| Capítulo | Archivo | Explica | Prioridad |
|---|---|---|---|
| Ch0 | `ch00_porque_ods.md` | Motivación, historia del proyecto, por qué Clifford + NS + MoE | F1 |
| Ch1 | `ch01_cl30.md` | Álgebra Cl(3,0): multivectores, producto geométrico, rotores | F1 |
| Ch2 | `ch02_campo_ns.md` | Campo NS-viscoso como medio de comunicación | F1 |
| Ch3 | `ch03_dirac_geometria.md` | Operador Dirac efectivo, geometría emergente (P03) | F2 |
| Ch4 | `ch04_plegado.md` | Plegado elástico, memoria, cascada τ (P04–P05) | F2 |
| Ch5 | `ch05_coherenceMoE.md` | CoherenceMoE y Variable Distillation Dynamics | F2–F3 |
| Ch6 | `ch06_cierre.md` | Clausura NS, MoDE synth compiler (P06), hacia AGI | F3 |

Cada capítulo debe tener al final: referencia al paper formal, ecuaciones clave, y un bloque *"implementación: ver `foundations/ml/X.py`"*.

---

## 6. Roadmap 12 meses — plan sobrio

Cada fase tiene **criterio de salida** concreto. No se avanza hasta cumplirlo.

### F1 — Solidez (meses 1–2, abril–junio 2026)

**Objetivo**: que el ML stack no sea frágil y que el libro tenga base.

- [ ] Tests unitarios `test_expert.py`, `test_field_step.py`, `test_moe.py` (normas de rotor, gate ∈ [0,1], gradientes fluyen).
- [ ] **Producto geométrico real** (Cayley de `foundations/clifford.py`, ahora es element-wise).
- [ ] `vmap` para batch en `moe.py`.
- [ ] Libro: `ch00`, `ch01`, `ch02` escritos (no skeleton — texto real).
- [ ] CI mínimo (GitHub Actions que corra `pytest tests/`).

**Criterio de salida**: `pytest tests/ && python -m foundations.ml.train` pasa verde en CI.

### F2 — Datos reales (meses 3–4)

**Objetivo**: entrenar con algo que no sea regresión de juguete.

- [ ] Elegir dataset (candidatos: TinyStories, MNIST secuencial, audio simbólico MIDI).
- [ ] Adapter dataset → input para CoherenceMoE.
- [ ] Escalar a 8–16 expertos, `d_model=64`.
- [ ] Medir: loss, entropía de gates, energía del campo, uso por experto.
- [ ] Libro: `ch03`, `ch04`.

**Criterio de salida**: curva de loss decreciente sobre 1000+ pasos con dataset real.

### F3 — Medir contra el estado del arte (meses 5–6)

**Objetivo**: saber si la arquitectura aporta algo o no.

- [ ] Implementar MoE estándar (top-k router) como baseline, mismo dataset.
- [ ] Ablation: CoherenceMoE sin campo (sólo rotors), sin coherence gate, sin VDD.
- [ ] Tabla comparativa.
- [ ] **Paper CoherenceMoE** (borrador arxiv).
- [ ] Libro: `ch05`.

**Criterio de salida**: paper en arxiv. Resultado honesto, sea bueno o malo.

### F4 — Decisión (meses 7–9)

Dos caminos según F3:

**4.A (si los números son buenos)**: escalar. Más expertos, dataset mayor, buscar recursos GPU externos.

**4.B (si no)**: pivotar. La arquitectura no funciona tal cual. Hipótesis candidatas: sustituir Cl(3,0) por Cl(1,3) (Lorentz), cambiar NS por reacción-difusión, reemplazar gate por mecanismo de atención geometrizada.

En cualquier caso: libro `ch06` + cierre de papers P03–P06.

### F5 — Tesis (meses 10–12)

- [ ] Compilación de papers para defensa USAL.
- [ ] Libro completo revisado.
- [ ] Decisión final: **continuar hacia F6 (AGI ambicioso) o cerrar el proyecto aquí como investigación doctoral completa.**

---

## 7. Roadmap ambicioso — ¿cuánto falta para AGI?

**Aviso de realismo**: nadie sabe cuánto falta para AGI. Los plazos de abajo son *condicionales a que la hipótesis central funcione*. Si no funciona, el proyecto se queda en "arquitectura novedosa publicada" — un éxito doctoral pero no AGI.

### F6 — Escala media (año 2, 2027)

Precondición: F4.A exitosa.

- Escalar CoherenceMoE a 100M–1B parámetros.
- Campo en GPU (compute shader propio, posiblemente vía Rust+wgpu o Pallas).
- Entrenar en dataset multimodal pequeño (texto + audio simbólico).
- Establecer leyes de escala (¿cómo decae loss con n_experts, field_size, d_model?).

**Milestone**: modelo que genere texto coherente a nivel de GPT-2 con menos parámetros o mejor eficiencia energética.

### F7 — Multimodalidad real (año 3, 2028)

- Integrar audio real (aquí el trabajo de Claude DJ / Strudel vuelve del parking como **dataset/codec**, no como interfaz).
- Integrar visión (frames → field embeddings).
- Entrenamiento self-supervised.

**Milestone**: modelo que demuestra transferencia entre modalidades sin fine-tuning explícito.

### F8 — Razonamiento y agencia (años 4–5, 2029–2030)

- Loop de razonamiento: el modelo edita su propio campo a lo largo de τ (tiempo estructural) antes de emitir respuesta.
- Conexión con entornos (simulados primero, físicos después).
- Meta-aprendizaje: el modelo aprende a ajustar sus propios umbrales `θ` y escalas `T` por contexto.

**Milestone**: el modelo resuelve problemas multi-paso que no estaban en el entrenamiento, sin prompting explícito de chain-of-thought.

### F9 — AGI candidata (años 5+)

Sólo entramos aquí si F8 demuestra emergencia genuina de razonamiento. No antes. Si llegamos:

- Colaboración con laboratorios con recursos (Anthropic, DeepMind, academia).
- Estudios de alineación específicos para esta arquitectura.
- Protocolo de pruebas AGI (no sólo benchmarks, también sondas de generalización radical).

**Si F8 falla**: el proyecto se cierra como contribución doctoral + papers, con dignidad. Es un resultado respetable.

---

## 8. Parking — lo que NO se toca ahora

**Todo lo de esta lista está congelado.** Si te descubres trabajando en algo de aquí, estás fallando la regla de §0.

Ver `parking/README.md` para detalle completo.

Resumen:

- **HTMLs de escritorio** (`ODS_Claude_DJ.html`, `ODS_Game_Engine.html`, `ODS_Strudel_Studio.html`, `ODS_DSL_Studio.html`): prototipos visuales de audio. Vuelven en F7 como codec/dataset.
- **Editor E con celdas inline**: idea interesante, no bloqueante. F7 o nunca.
- **Agentes inversores con datos de mercado**: idea de sesión antigua, nunca implementada. Fuera del eje AGI.
- **Chat integrado en ODS**: fuera del eje.
- **ws-stack**: sistema de coordinación de tareas del monorepo; no tocar salvo que bloquee F1.
- **ODS Game Engine / Factorio of Sound**: fuera.
- **Motor gráfico en Rust desde cero**: sesión paralela de aprendizaje Rust — no parte del eje AGI; si se hace, es hobby.

---

## 9. Protocolo de sesión (cómo empezar cada día)

1. Abrir `PROYECTO.md` (este archivo).
2. Comprobar fase actual en §6.
3. Elegir **una** tarea marcada `[ ]` de la fase actual.
4. Trabajar sólo en esa tarea.
5. Al acabar:
   - Marcar `[x]` en §6.
   - Commit con mensaje `F{n}: {tarea corta}`.
   - Si procede, actualizar el capítulo del libro correspondiente.
6. Si aparece una idea nueva buena: se apunta en §11 (backlog), no se ejecuta.

**Anti-patrones prohibidos**:
- Saltar de tarea a tarea en la misma sesión.
- Abrir prototipos de audio "sólo a ver qué tal".
- Añadir dependencias "que igual necesitamos más adelante".
- Reescribir código que funciona para que "sea más elegante".

---

## 10. Decisiones pendientes

### 10.1 ¿Subimos `ods-unified-v2` a GitHub como repo propio?

**Contexto**: el runtime real vive ahí (`_claude/lib/ml/` es una copia viva de `foundations/ml/`, más adapters, experimentos, HANDOFF con Codex). Hoy está dentro del monorepo privado `/home/r/z_folder`, lo cual:

- Impide CI.
- Impide colaboración limpia con Codex.
- Mezcla código ODS con cosas ajenas (OpenJarvis, ObsidianVault, etc.).

**Propuesta**: crear `r-gabadr/ods-unified-v2` como repo nuevo, migrar el subdirectorio con `git subtree` o copia limpia, y establecer política: *foundations/ml en ods-papers = versión pública; ods-unified-v2 = runtime interno con experimentos y adapters*.

**Decisión**: pendiente de Rubén. Tarea candidata para F1.

### 10.2 ¿Mantenemos `odsiv` y `ods-golf`?

Revisar qué son y si aportan al eje AGI. Si no, marcar explícitamente como proyectos laterales en el README de cada uno para no confundir al yo-futuro.

---

## 11. Backlog (ideas sin asignar fase)

Esto es lavabo: las ideas entran aquí y se quedan hasta que se asignan a una fase.

- Producto geométrico vectorizado con Einstein summation (optimización cuando F3 ande bien).
- Estudio efecto de `dt` (paso temporal del campo) sobre coherencia global.
- Conexión teórica: ¿es Variable Distillation Dynamics equivalente a un schedule adaptativo de learning rate tipo AdaHessian? (literatura a revisar).
- Visualización del campo durante entrenamiento (sólo cuando F2 tenga datos reales; si no, ruido).
- Rotor Cl(1,3) en vez de Cl(3,0) — simetría Lorentz.

---

## 12. Glosario esencial

- **Cl(3,0)**: álgebra de Clifford con 3 generadores de cuadrado +1. Multivectores de 8 componentes.
- **Rotor**: elemento de grado par de Cl(3,0) con norma 1; representa rotaciones.
- **Coherencia** (en ODS): razón entre magnitud de parte vectorial y bivectorial del estado de un experto.
- **Campo NS viscoso**: campo escalar 2D con dinámica ∂Φ/∂t = ν·ΔΦ − α·Φ + fuentes.
- **Variable Distillation Dynamics**: gradiente de cada experto modulado por su propia coherencia: η_k ∝ σ((coh_k − θ_k)/T).
- **τ**: tiempo estructural (interno del modelo), distinto del tiempo físico t de la cascada del campo.
- **MoDE**: Mode of Dynamical Emission — sinónimo de CoherenceMoE en los papers P06.
- **MoE**: Mixture of Experts estándar (router top-k, baseline).

---

## 13. Recordatorios

- **No borrar nada.** Todo lo parkeado se queda en `parking/`.
- **No tener prisa.** Los plazos de §6 son conservadores; los de §7 son especulación honesta.
- **Si un día estás mal**: no trabajar. No se toca ODS en crisis. Ver doc privado de salud (fuera de este repo).
- **El proyecto no depende de ti estar enchufado 24/7.** Está diseñado para avanzar en pasos digeribles.
