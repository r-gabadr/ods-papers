# El Libro de ODS

Manual humano paralelo a los papers formales. Cada capítulo explica un paper y apunta a su implementación en `foundations/ml/`.

**No es divulgación light.** Es el manual interno para que tú mismo no te pierdas, la base del material de tesis (USAL), y — en su día — material para terceros que quieran entender la apuesta.

## Cómo leerlo

1. **Primera vez**: ch00 → ch01 → ch02 en orden. Son fundacionales.
2. **Saber qué implementa el código**: ch05 (CoherenceMoE).
3. **Entender la teoría detrás**: ch01–ch04.
4. **Ver hacia dónde va**: ch06.

## Estado

| Cap | Archivo | Paper(s) que explica | Estado |
|---|---|---|---|
| 0 | `ch00_porque_ods.md` | — (motivación) | skeleton |
| 1 | `ch01_cl30.md` | P00 | skeleton |
| 2 | `ch02_campo_ns.md` | P01, P02 | skeleton |
| 3 | `ch03_dirac_geometria.md` | P03 | skeleton |
| 4 | `ch04_plegado.md` | P04, P05 | skeleton |
| 5 | `ch05_coherenceMoE.md` | (nuevo, no hay paper aún — será F3) | skeleton |
| 6 | `ch06_cierre.md` | P06 | skeleton |

Plan de escritura (ver `PROYECTO.md` §6):

- F1 (meses 1–2): escribir ch00, ch01, ch02.
- F2 (meses 3–4): ch03, ch04.
- F3 (meses 5–6): ch05.
- F4 (meses 7–9): ch06.

## Convenciones

- Markdown plano, ecuaciones en LaTeX inline (`$...$`) y en bloque (`$$...$$`).
- Al final de cada capítulo, un bloque **Referencias** con: paper formal correspondiente, bibliografía externa, archivos de código implicados.
- Cuando sea útil, añadir una **caja de "metáfora física"** para ayudar a la intuición sin sustituir la formalidad.
- Las ecuaciones clave se numeran dentro del capítulo: `(ch1.3)`, `(ch2.1)`, etc.
