# Capítulo 4 — Plegado elástico, memoria y cascada en τ

**Papers formales**: P04 (Memory Coupling, Tau Cascade, Virtual Population), P05 (Hutchinson IFS, Contractivity, Fixed Point).
**Prerequisitos**: capítulos 1–3.
**Estado**: **skeleton — desarrollar en F2**.

---

## 4.1 Dos tiempos: $t$ físico y $\tau$ estructural

**TODO**: escribir.

- $t$: tiempo del campo NS. La difusión ocurre en $t$.
- $\tau$: tiempo *interno* del razonamiento. En cada paso de $\tau$, el modelo puede refinar su estado leyendo y emitiendo sucesivamente.
- El modelo pasa por varios pasos de $\tau$ antes de emitir output. Esto es la **cascada τ**.

## 4.2 Plegado elástico de la geometría

**TODO**: escribir.

- La función $h(x,t,\tau)$ codifica cuánto se pliega localmente la variedad.
- En regiones coherentes, $h$ es pequeño → métrica casi plana → computación barata.
- En regiones activas, $h$ crece → métrica curvada → coste mayor pero expresividad mayor.

## 4.3 Acoplamiento con memoria

**TODO**: escribir.

- El campo $\Phi$ tiene memoria por el término de disipación (decay exponencial con constante $1/\alpha$).
- Memorias más largas → $\alpha$ menor → el modelo recuerda más contexto.

## 4.4 Población virtual

**TODO**: escribir.

- Un solo experto físico puede "actuar como" varios virtuales si su rotor aprendible decompone naturalmente en subrotores.
- Análogo: un head de attention puede capturar varios patrones.

## 4.5 Contractividad y punto fijo (P05)

**TODO**: escribir.

- Para que la cascada $\tau$ converja, la iteración debe ser contractiva.
- Hutchinson IFS: herramienta matemática para estudiar convergencia de sistemas iterados.
- Resultado: condición suficiente sobre $\nu, \alpha, \sigma_k$ que garantiza punto fijo estable.

## 4.6 Relación con el código

Hoy: la cascada $\tau$ no está implementada. Un forward de `CoherenceMoE` hace *un* paso de campo. Para cascada completa habría que iterar $N_\tau$ pasos antes de leer el output — tarea F4 según resultados previos.

---

## Referencias

- **Papers formales**: `papers/latex/p04/`, `papers/latex/p05/`.
- **Bibliografía externa**:
  - Hutchinson, J.E. "Fractals and Self-Similarity" (1981).
  - Banach fixed-point theorem; cualquier libro de análisis funcional.
