# Capítulo 3 — Operador Dirac efectivo y geometría emergente

**Paper formal**: P03 — Observability, Spectral Coherence, Admissibility Triggers.
**Prerequisitos**: capítulos 1–2.
**Estado**: **skeleton — desarrollar en F2**.

---

## 3.1 Qué es un operador Dirac y por qué nos importa

**TODO**: escribir.

- Dirac original: $D = i\gamma^\mu \partial_\mu$ en QFT.
- En geometría: $D$ captura la estructura diferencial de una variedad spin.
- En ODS: versión *efectiva*, actúa sobre secciones del "haz de estados" de los expertos.

## 3.2 Geometría inducida por el campo

**TODO**: escribir.

- El campo $\Phi$ en reposo tiene métrica trivial.
- Cuando los expertos emiten, aparece una deformación $h(x,t)$ sobre la que emergen coeficientes métricos:

$$g_{ij} = \delta_{ij} + \partial_i h \, \partial_j h \tag{ch3.1}$$

- Esta geometría dice al operador Dirac efectivo cómo diferenciar, y al Laplaciano cómo difundir.

## 3.3 Coherencia espectral y observabilidad

**TODO**: escribir.

- Un estado es *observable* en ODS si su proyección sobre los modos propios del Dirac efectivo pasa un umbral de admisibilidad.
- Trigger espectral $\chi(x,t)$: función booleana diferenciable (vía sigmoid) que decide si un estado contribuye al output.
- Relación con la coherencia de (ch1.4).

## 3.4 Resolvente y tensión

**TODO**: escribir.

- Resolvente $\rho_S = (D - zI)^{-1}$: mide qué tan cerca está $z$ del espectro.
- Tensión espectral: picos del resolvente indican modos colectivos emergentes.

## 3.5 Relación con el código

Hoy: el ML stack NO implementa Dirac efectivo ni tensión espectral. Son piezas *teóricas* de P03 que se añadirán a `foundations/ml/` cuando sea relevante (posiblemente F4 según resultados).

---

## Referencias

- **Paper formal**: `papers/latex/p03/p03_skeleton.tex`.
- **Bibliografía externa**:
  - Connes, A. "Noncommutative Geometry" (1994).
  - Bär, Ginoux. "Classical and Quantum Fields on Lorentzian Manifolds" (2009).
