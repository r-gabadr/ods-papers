# Parking — cosas congeladas

**Regla**: todo lo que está en esta lista NO se toca hasta que su fase lo descongele (ver `PROYECTO.md` §7).

Esto no es una papelera. Es un congelador. Las ideas aquí pueden ser buenas; simplemente **no están en el eje crítico hacia AGI**, y trabajar en ellas ahora es distracción.

---

## Congelados actuales

### 1. Prototipos de audio (HTMLs en Desktop)

**Archivos**:
- `C:\Users\ruben\Desktop\ODS_Claude_DJ.html` — graphophone con audio real fusionado, sliders, editor
- `C:\Users\ruben\Desktop\ODS_Game_Engine.html` — "Factorio of Sound"
- `C:\Users\ruben\Desktop\ODS_Strudel_Studio.html` — motor audio que se fusionó con Claude DJ
- `C:\Users\ruben\Desktop\ODS_DSL_Studio.html` — panel glassmorphism con sliders

**Estado**: funcionan, son bonitos, no aportan a AGI.

**Vuelven en**: F7 (multimodalidad real, 2028), como **dataset / codec de audio**, no como interfaz. La interfaz en sí no es el objetivo.

**No tocar para**: "probar una cosa", "añadir un botón", "ver si suena bien". Si pica, cerrar el navegador.

### 2. Editor E con celdas inline (Claude DJ)

**Descripción**: rediseño del panel del editor para que sea cell-based (Jupyter-like) con waveforms inline por celda, en vez de split-panel.

**Estado**: iniciado, inacabado, rechazado por el usuario en sesión previa.

**Vuelve en**: nunca, salvo que en F7 se decida que la interfaz de audio es prioritaria (improbable).

### 3. Agentes inversores con datos de mercado

**Descripción**: idea de sesión anterior (GPT/Gemini) sobre 3 agentes LLM recibiendo datos de mercado e invirtiendo.

**Estado**: no implementado. Sólo código suelto de `market_shadow` encontrado en el monorepo.

**Vuelve en**: nunca en este proyecto. Si vuelve la idea, es **otro proyecto** distinto, no parte de ODS.

### 4. Chat integrado en ODS

**Descripción**: interfaz de chat sobre el stack ODS.

**Estado**: idea flotante, no implementada.

**Vuelve en**: F8 como mucho, si llegamos. Antes no aporta nada.

### 5. ws-stack (sistema de tasks del monorepo)

**Descripción**: sistema de lanes remotos y tasks coordinadas en el monorepo personal `z_folder`.

**Estado**: existe, parcialmente cableado, MCP no del todo conectado.

**No tocar salvo que**: bloquee una tarea de F1 concreta. En ese caso, tarea-mínima: desbloquear y salir.

### 6. Motor gráfico en Rust desde cero (Proyecto R)

**Descripción**: aprender Rust construyendo un rasterizer a lo TinyRenderer, siguiendo un plan de 0 a N con Kimi.

**Estado**: sesión paralela de aprendizaje.

**Relación con ODS**: **ninguna directa**. Es hobby. Puede ser útil en F6 si decidimos hacer el campo en GPU con compute shaders propios (`wgpu`), pero eso es año 2+.

**Regla**: está bien programarlo para aprender Rust. NO está bien meter esas ganas dentro del flujo ODS.

### 7. ODS Game Engine / Factorio of Sound (concepto)

**Descripción**: la idea de que ODS sea "jugable" como un Factorio.

**Estado**: hay un HTML prototipo (ver #1).

**Vuelve en**: podría ser un vehículo divulgativo en F9 si llegamos a AGI candidato. Por ahora no.

### 8. Papers P03–P06 — texto completo en LaTeX

**Descripción**: los skeletons están; el texto completo no.

**Estado**: skeletons en `papers/latex/p03..p06/`.

**Vuelve en**:
- P03 completo: F2 (a medida que se implementa Dirac efectivo).
- P04, P05 completos: F2–F4.
- P06 completo: F4.

**Antes de escribir cualquier paper completo**: comprobar que el libro correspondiente está escrito. El libro es más fácil; si no puedes escribirlo, no entiendes bien el tema como para el paper formal.

---

## Cómo desarchivar algo

1. Abrir `PROYECTO.md`.
2. Comprobar que la fase actual permite descongelar el ítem (ver §7).
3. Editar `parking/README.md`: mover el ítem a la sección "Recién descongelados" con fecha.
4. Añadirlo como tarea explícita en §6 de `PROYECTO.md`.

No se descongela "sobre la marcha".

---

## Recién descongelados

(vacío por ahora)
