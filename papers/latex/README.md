# LaTeX Paper Sources

Fuentes LaTeX del programa modular de papers ODS.

## Estructura

```
latex/
├── shared/
│   ├── ods_macros.sty    — macros compartidos (Clifford, operadores, observables)
│   └── references.bib    — bibliografía compartida
├── p01/                  — P01: CT Runtime (fuente en Zenodo 10.5281/zenodo.18995428)
└── p02/
    ├── p02_regimes.tex   — P02: Regímenes, solitones, modulación inter-grade
    └── tables/           — tablas LaTeX generadas por scripts/export_p02_csv.py
```

## Compilar P02

```bash
cd docs/papers/latex/p02
pdflatex p02_regimes.tex
bibtex p02_regimes
pdflatex p02_regimes.tex
pdflatex p02_regimes.tex
```

## Tests que prueban los claims

```bash
python -m pytest tests/paper_claims/ -v
```

## Convenciones de macros

| Macro | Significado |
|-------|------------|
| `\Cl` | Clifford algebra Cl |
| `\Psi` | campo multivectorial |
| `\slice{\ell}` | slice ℓ del campo |
| `\grade{k}` | proyector de grado k |
| `\freeE` | energía libre F |
| `\deff` | dimensión efectiva d_eff |
| `\SK` | operador S_K |
| `\vortexfrac` | fracción de pasos con vórtice |
| `\complexity` | complejidad de solitón |
