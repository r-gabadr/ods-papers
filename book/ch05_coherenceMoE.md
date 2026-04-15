# Capítulo 5 — CoherenceMoE y Variable Distillation Dynamics

**Paper formal**: aún sin escribir (será el paper de F3, nuevo, no parte de P00–P06 original).
**Prerequisitos**: capítulos 1–2. Capítulo 3 ayuda pero no es bloqueante.
**Estado**: **skeleton — desarrollar en F2–F3**.

---

## 5.1 La arquitectura completa

Juntamos capítulos 1 y 2 en una sola unidad entrenable.

**Componentes**:
- $N$ expertos, cada uno con posición $x_k \in \Omega$, rotor learnable $R_k$, umbral $\theta_k$, escala $T_k$, sigma de emisión $\sigma_k$, proyector de salida $W_k \in \mathbb{R}^{8 \to d_{\text{out}}}$.
- Un campo compartido $\Phi$ con $\nu, \alpha$ aprendibles.
- Proyección de entrada $W_{\text{in}} \in \mathbb{R}^{d_{\text{model}} \to 8}$.
- Proyección de merge $W_{\text{merge}} \in \mathbb{R}^{d_{\text{model}} \to d_{\text{model}}}$.

## 5.2 Forward pass (paso a paso)

Dado un token $x \in \mathbb{R}^{d_{\text{model}}}$:

1. **Proyecta a Clifford**: $\phi_{\text{in}} = W_{\text{in}} x \in \mathbb{R}^8$.
2. **Emite al campo** (en posición arbitraria, por ejemplo el centro): $\Phi \mathrel{+}= G(\cdot; \sigma) \phi_{\text{in}}$.
3. **Expertos leen, piensan, emiten**:

   Para cada experto $k$:
   - Lee campo local: $\phi_k = \Phi(x_k)$.
   - Rotor: $\psi_k = R_k \odot \phi_k$ (element-wise por ahora; producto geométrico real en F1).
   - Normaliza: $\psi_k \leftarrow \psi_k / \|\psi_k\|$.
   - Coherencia: $c_k = \text{coh}(\psi_k)$ según (ch1.4).
   - Gate: $g_k = \sigma\left(\frac{c_k - \theta_k}{T_k}\right)$.
   - Output local: $y_k = g_k \cdot (W_k \psi_k)$.
   - Emite al campo: $S_k = g_k \cdot G(x_k; \sigma_k) \cdot \psi_k[0]$ (grado 0).

4. **Un paso de campo**: $\Phi \leftarrow \text{step}(\Phi, \{S_k\})$.
5. **Merge outputs**: $y = W_{\text{merge}} \sum_k y_k$.

Hoy: un solo paso de campo por forward. Futuro: cascada τ (cap. 4).

## 5.3 Variable Distillation Dynamics

**Clave del capítulo**: el gradiente no es uniforme.

Cuando calculamos $\partial L / \partial R_k$, el gate $g_k$ ya multiplica la salida. Por la regla de la cadena, el gradiente efectivo que llega al rotor está escalado por $g_k$:

$$\frac{\partial L}{\partial R_k} = g_k \cdot \frac{\partial L}{\partial (y_k / g_k)} \cdot \frac{\partial (W_k \psi_k)}{\partial R_k} \tag{ch5.1}$$

Esto es la **distilación variable**: expertos con gate bajo casi no reciben gradiente; expertos con gate alto lo reciben todo. Y como $g_k$ depende de la coherencia *aprendida* del experto, el schedule de learning rate es **autoajustado por cada experto**.

**Por qué es novedoso**: busqué en arxiv (2026-04) y no hay trabajo con esta formulación exacta. Las arquitecturas existentes con "gated learning rate" usan heurísticas externas, no coherencia intrínseca del estado geométrico.

## 5.4 Entrenamiento

Loop estándar `optax.adam` con `eqx.filter_value_and_grad`:

```python
@eqx.filter_jit
def step(model, opt_state, x, y):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val
```

**Métricas a seguir** (mencionadas en `train.py`):

- `loss`: obvio.
- `mean_gate`: promedio de $g_k$ sobre expertos y batch. Si → 0: colapso total (nadie emite). Si → 1: no hay selectividad.
- `gate_entropy`: $-\sum_k g_k \log g_k$. Alta = buena distribución. Baja = pocos expertos dominan.
- `field_energy`: $\|\Phi\|^2$. Debe crecer y estabilizarse, no explotar ni irse a cero.

## 5.5 Resultados iniciales (2026-04)

Toy regression, 2 expertos, $d_{\text{model}}=8$, 40 pasos:

- Loss 0.82 → 0.81 (muy poco movimiento — tarea pequeña, tuning pendiente).
- 278 parámetros totales.
- Sin NaN.
- Thetas apenas se mueven (sospecha: necesitan lr mayor o inicialización distinta) — tarea F1.

**Interpretación**: el stack **corre** y es **entrenable**, pero los resultados no son concluyentes a esta escala. La validación real viene en F2 (datos reales) y F3 (ablation vs MoE estándar).

## 5.6 Ablations pendientes (F3)

1. **CoherenceMoE completo** (baseline del paper).
2. **Sin campo**: expertos independientes, mismo gate. ¿Cuánto perdemos?
3. **Sin gate de coherencia**: gate = 1 siempre. ¿Cuánto perdemos?
4. **Sin VDD**: gradiente uniforme (no escalado por $g_k$). ¿Cuánto perdemos?
5. **vs MoE top-k estándar**: mismo $N$, $d_{\text{model}}$, número de parámetros comparable.

## 5.7 Relación con el código

Ver `foundations/ml/moe.py` y `foundations/ml/train.py`.

---

## Referencias

- **Paper formal**: TBD en F3 (working title: *CoherenceMoE: Field-Mediated Mixture of Experts with Variable Distillation Dynamics*).
- **Código**: `foundations/ml/moe.py`, `foundations/ml/train.py`, `foundations/ml/expert.py`.
- **Literatura relacionada**:
  - Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (2017).
  - Fedus, Zoph, Shazeer. "Switch Transformer" (2021).
  - Brandfonbrener et al. "Mixture of Experts Meets In-Context Learning" (2024).
  - Búsqueda negativa sobre VDD: `papers/REVIEW_TRACKER.md`, 2026-04.
