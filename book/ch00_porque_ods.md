# Capítulo 0 — Por qué ODS

**Paper formal**: ninguno (capítulo de motivación).
**Prerequisitos**: ninguno.
**Duración estimada de lectura**: 15–20 min cuando esté escrito.

---

## 0.1 La pregunta

La arquitectura dominante en IA hoy — el Transformer — funciona, pero no está claro que sea *suficiente* para llegar a algo que merezca el nombre de inteligencia general. Escala. Consume energía como una ciudad. Y aun así, tropieza con cosas que un niño de cinco años hace sin esfuerzo: razonamiento causal, generalización radical, aprendizaje con pocos ejemplos.

La hipótesis de ODS es que lo que falta no es más parámetros, sino una **estructura interna distinta**. Concretamente, tres piezas:

1. **Estado geométrico** (no sólo vectorial): cada unidad de cómputo tiene un estado en un álgebra rica que codifica rotaciones, orientaciones y relaciones — no sólo magnitudes.
2. **Medio compartido** (no router): las unidades se sincronizan a través de un campo continuo, no pasándose tokens por un mecanismo de atención.
3. **Aprendizaje modulado por coherencia**: cada unidad aprende a un ritmo que depende de cuán organizada esté ella misma, no de un schedule externo.

Las tres piezas juntas dan la arquitectura `CoherenceMoE` con *Variable Distillation Dynamics*.

## 0.2 Historia del proyecto

**TODO**: escribir. Puntos clave a cubrir:

- Origen: trabajo de Rubén en física teórica (Clifford, Dirac), contacto con LLMs, pregunta: *¿puedo meter física real dentro del modelo?*
- Primeras iteraciones: Graphophone → Claude DJ → ODS Unified V2.
- Papers P00–P06: la teoría se desarrolló primero, el código después.
- 2026-04: primer ML stack entrenable (`CoherenceMoE`) + confirmación de que *Variable Distillation Dynamics* es novedoso.

## 0.3 Por qué Clifford (y no sólo vectores)

**TODO**: escribir. Puntos:

- Clifford Cl(3,0) contiene: escalares, vectores, **bivectores** (planos orientados) y **trivectores** (volúmenes orientados).
- Bivectores codifican rotaciones naturalmente — sin matrices, sin gimbal lock, sin cuaterniones ad hoc.
- Producto geométrico unifica: producto escalar + producto externo.
- En ML: un estado Clifford puede representar orientación + magnitud + sincronía en una sola estructura.

## 0.4 Por qué NS viscoso (y no atención)

**TODO**: escribir. Puntos:

- La atención es **instantánea y no-local**: cada token habla con todos los demás en un solo paso.
- Un campo NS-viscoso es **local y persistente**: la información se propaga por difusión, y tiene memoria (la viscosidad).
- Consecuencia: los expertos se sincronizan por física emergente, no por diseño.
- Costo: más pasos temporales por forward. Beneficio: potencial mejor comportamiento out-of-distribution.

## 0.5 Por qué distilación variable

**TODO**: escribir. Puntos:

- En entrenamiento estándar, todos los parámetros reciben gradiente con el mismo learning rate (modulo Adam).
- En ODS, cada experto tiene su propia coherencia. Si está confuso, su gradiente se atenúa; si está claro, se amplifica.
- Análogo biológico: neuronas confusas no se plastifican hasta tener señal clara.
- Formalmente: $\eta_k = \sigma\left(\frac{\text{coh}(\psi_k) - \theta_k}{T}\right)$ donde $\theta_k$ y $T$ son aprendibles.

## 0.6 Qué NO es ODS

Para evitar malentendidos:

- **No** es una red neuronal biológicamente plausible. No modelamos neuronas.
- **No** es un sistema físico simulado. Usamos NS porque sus propiedades matemáticas nos sirven, no porque queramos hacer hidrodinámica.
- **No** es *spiking* network ni hardware neuromórfico.
- **No** pretende reemplazar Transformers en tareas donde Transformers brillan (texto corto, clasificación). Apunta a nichos donde la estructura importa: razonamiento, transferencia entre modalidades, generalización.

## 0.7 Qué aspira a ser

Un sustrato computacional donde la inteligencia pueda emerger por **organización geométrica de campos**, y donde el aprendizaje sea una consecuencia de la coherencia del propio estado, no una imposición externa.

Si la apuesta gana, tenemos un camino hacia AGI estructuralmente distinto al de escalar Transformers. Si pierde, tenemos papers y una tesis doctoral sobre una arquitectura nueva.

Ambos resultados son aceptables. Ninguno es un fracaso.

---

## Referencias

- Papers formales: ver `papers/MASTER_PLAN_2026.md`.
- Código ML stack: `foundations/ml/`.
- Búsqueda arxiv sobre *Variable Distillation Dynamics*: registrada en `papers/REVIEW_TRACKER.md` (2026-04).
