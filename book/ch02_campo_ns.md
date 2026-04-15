# Capítulo 2 — El campo NS-viscoso como medio de sincronización

**Papers formales**: P01, P02.
**Prerequisitos**: capítulo 1 (Cl(3,0)), cálculo multivariable, idea de PDE.
**Duración estimada**: 30–45 min.

---

## 2.1 El problema del router

En un MoE clásico (Mixture of Experts), un **router** decide qué expertos procesan cada token:

$$\text{router}(x) \to \{k_1, \ldots, k_{\text{top}}\} \tag{ch2.1}$$

Esto tiene problemas conocidos:

- **Colapso**: algunos expertos se usan siempre, otros nunca.
- **Discontinuidad**: el top-k es no-diferenciable, requiere trucos (Gumbel, auxiliary loss).
- **No hay estructura entre expertos**: cada uno opera aislado. No hay sincronía ni intercambio.

¿Y si en vez de un router, todos los expertos **viven en un medio continuo** y se afectan mutuamente por física emergente?

## 2.2 El campo $\Phi$

Definimos un **campo escalar 2D** $\Phi : \Omega \times t \to \mathbb{R}$ (en el futuro, Clifford-valuado; por ahora escalar por simplicidad) con dinámica viscosa:

$$\frac{\partial \Phi}{\partial t} = \nu \Delta \Phi - \alpha \Phi + \sum_k S_k(x, t) \tag{ch2.2}$$

donde:

- $\nu \geq 0$: viscosidad (difusión).
- $\alpha \geq 0$: disipación (amortiguamiento).
- $S_k$: fuente del experto $k$ — lo que emite al campo cuando está coherente.

**TODO**: derivación de (ch2.2) a partir de primeros principios (por qué Laplaciano y por qué término lineal de decay).

## 2.3 Discretización

Para el código, discretizamos $\Omega$ como una rejilla $H \times W$ y $t$ en pasos $\Delta t$:

$$\Phi_{t+1} = \Phi_t + \Delta t \left(\nu \, \text{lap}(\Phi_t) - \alpha \Phi_t\right) + \sum_k S_k^t \tag{ch2.3}$$

El Laplaciano discreto con condiciones de contorno periódicas:

$$\text{lap}(\Phi)_{ij} = \Phi_{i+1,j} + \Phi_{i-1,j} + \Phi_{i,j+1} + \Phi_{i,j-1} - 4\Phi_{ij} \tag{ch2.4}$$

Implementación directa en `foundations/ml/field_step.py`.

## 2.4 Emisión: cómo un experto habla al campo

Cuando un experto $k$ en posición $x_k$ quiere emitir con intensidad $a_k$, depositamos una Gaussiana:

$$S_k(x, t) = a_k \cdot G(x - x_k; \sigma_k) \tag{ch2.5}$$

donde $\sigma_k$ es aprendible (un experto puede aprender a emitir más localmente o más difuso).

## 2.5 Lectura: cómo un experto escucha el campo

Simétrico: el experto $k$ lee el valor del campo en su posición:

$$\Phi_k^{\text{local}} = \Phi(x_k, t) \tag{ch2.6}$$

Por ahora, lectura puntual. En el futuro podría ser ponderada por la misma Gaussiana que usa para emitir.

## 2.6 Por qué esto sincroniza

**TODO**: desarrollar. Argumento informal:

- Si dos expertos emiten en regiones cercanas, sus emisiones se superponen por difusión.
- El experto que lee ve la suma → su estado interno (después del rotor) incorpora información del otro.
- Si ambos están coherentes, sus emisiones se refuerzan; si uno está confuso, el otro no se ve arrastrado (porque la confusión se atenúa antes de llegar).
- Resultado: emergencia de *modos globales* coherentes, análogos a modos normales en física.

## 2.7 Parámetros aprendibles

El campo tiene dos hiperparámetros que pasan a ser parámetros:

- `log_nu`: se optimiza, determina qué tan rápido se difunde la información.
- `log_alpha`: se optimiza, determina cuánto persiste la información (memoria).

La parametrización logarítmica garantiza $\nu, \alpha > 0$.

## 2.8 Implementación

```python
# foundations/ml/field_step.py (esquema)
class SharedField(eqx.Module):
    log_nu: jax.Array
    log_alpha: jax.Array
    dt: float  # static

    def step(self, Phi, sources):
        nu = jnp.exp(self.log_nu)
        alpha = jnp.exp(self.log_alpha)
        return Phi + self.dt * (nu * laplacian(Phi) - alpha * Phi) + sources
```

Los detalles de `emit_to_field` (Gaussiana con $\sigma$ JAX-traceable) y `sample` están en el archivo.

## 2.9 Ejercicios

**TODO**:

1. Demostrar que la versión discreta (ch2.3) preserva masa si $\alpha = 0$ y no hay fuentes.
2. ¿Qué pasa si $\nu = 0$? ¿Y si $\alpha = 0$?
3. Simular dos emisores Gaussianos a distancia $d$. Graficar el campo a $t$ creciente.
4. Calcular el tiempo característico de decay en función de $\alpha$.

---

## Referencias

- **Papers formales**: `papers/latex/p01/`, `papers/latex/p02/`.
- **Código**: `foundations/ml/field_step.py`.
- **Bibliografía externa**:
  - Evans, L.C. "Partial Differential Equations" (2010) — Laplaciano, equivalencia heat equation.
  - Doering & Gibbon. "Applied Analysis of the Navier-Stokes Equations" (1995).
