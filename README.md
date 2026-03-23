# ODS Papers

Research papers, claim verification tests and experimental results for the **ODS (Observable Dynamical Systems)** program — a Clifford algebra Cl(3,0) cognitive runtime.

## Structure

```
tests/paper_claims/     ← pytest claim verification (run against ods-unified-v2)
experiments/            ← reproducible experimental results + overnight data
papers/latex/           ← LaTeX source for P02-P05
release_p02/            ← P02 final submission (Zenodo DOI pending)
papers/MASTER_PLAN_2026.md
papers/REVIEW_TRACKER.md
```

## Paper program

| Paper | Title | Status | Zenodo |
|-------|-------|--------|--------|
| P00 | Genesis-D: SpinorAttractor8D | Published | [10.5281/zenodo.18929805](https://doi.org/10.5281/zenodo.18929805) |
| P01 | CT Runtime Cl(3,0) | Published | [10.5281/zenodo.18995428](https://doi.org/10.5281/zenodo.18995428) |
| P02 | Attractor Regimes & Vortex Persistence | Submitted | — |
| P03 | Observability: commutator defect, λ₂, tension | In progress | — |
| P04 | Document memory & τ cascade | In progress | — |
| P05 | Multiscale contractive extension (Hutchinson) | In progress | — |

## Running claim tests

```bash
# Install the runtime library first
pip install -e /path/to/ods-unified-v2

# Run all claim tests
pytest tests/paper_claims/ -v

# Single paper
pytest tests/paper_claims/test_p03_claims.py -v
```

## Runtime dependency

Tests depend on [`ods-unified-v2`](https://github.com/r-gabadr/ods-unified-v2) — the canonical JAX Clifford runtime library.

## License

MIT
