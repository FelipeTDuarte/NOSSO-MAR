# NOSSO-MAR

**Neural Operator Scalable System for Ocean Modelling, Analysis, and Renewables**

A research-grade, modular AI framework for fast ocean and coastal physics simulation,
built on Physics-Informed Neural Operators (PIMR-NO) and open metocean data.

```
## Vision — Phases 0 to 7


Phase 0  Foundational architecture          ✓ complete
Phase 1  WSI + Wave field operators         → in progress
Phase 2  Spectral waves + data assimilation
Phase 3  Hydrodynamics and tides
Phase 4  Morphodynamics
Phase 5  Tracer transport
Phase 6  Advanced FSI / CFD
Phase 7  Digital twin + MARL optimisation
  
```

## Architecture layers

| Layer | Components |
|-------|------------|
| **L0 — Observations** | NDBC/COOPS buoys, altimeters, OISST SST |
| **L1 — Spectral state** | Sea-state statistics, 1D/2D spectra |
| **L2 — Phase-resolved fields** | η(x,y,t), velocities, nearshore propagation |
| **L3 — WEC response / WSI** | Hydrodynamic coefficients, power, arrays |
| **L4 — High-fidelity CFD/FSI** | Reference for calibration and validation |
| **Neural Operators** | FNO2d/3d, F-FNO, Geo-FNO, WNO, GNO, DeepONet, RINO |
| **Embedded physics** | Navier-Stokes residuals, multi-fidelity, UQ |
| **Data pipeline** | Open data → analytic → BEM (Capytaine) → solvers |

## Phase 1 — Core scientific contribution

Replaces the phase-resolving wave solver (OceanWave3D) + BEM (Capytaine)
with coupled neural operator surrogates.

```
F1A (DeepONet — WSI) ↔ F1B (FNO/WNO — wave field) → F1C (bidirectional coupling)
```

Expected speedup: **100×–1000×** per evaluation vs. classical solvers.

## Quickstart

```bash
# Install
pip install -e .

# Generate Phase 1 analytic dataset
python scripts/generate_phase1_dataset.py

# Train WEC operator (DeepONet)
python scripts/train_phase1.py

# Run tests
pytest tests/ -v

# Check available open data
python scripts/build_open_database.py
```

## Repository structure

```
NOSSO-MAR/
├── README.md
├── specs/          ← Technical specifications (01–10)
├── docs/           ← Architecture, CONTRIBUTING, FAQ, Roadmap
├── src/nossomar/   ← Phase 1 code (operators, physics, data)
├── scripts/        ← Entry points
├── configs/        ← Dataset and training configs
├── tests/          ← pytest tests (TDD)
├── data/           ← Generated data + open database
├── checkpoints/    ← Saved models and metrics
└── _archive/       ← Previous versions (reference only)
```

## Documentation

- `docs/ARCHITECTURE.md` — system overview and fidelity layers
- `specs/01_IO_CONTRACTS.md` — data contracts and tensor formats
- `docs/PHASE_1_ROADMAP.md` — current progress (7-task checklist)
- `docs/ROADMAP.md` — full roadmap, Phases 0–7
- `docs/FAQ.md` — common questions
- `docs/CONTRIBUTING.md` — TDD workflow + subagent dispatch

## Citation

```bibtex
@software{nosso_mar_2025,
  title   = {NOSSO-MAR: Neural Operator Scalable System for Ocean Modelling, Analysis, and Renewables},
  author  = {Duarte, Felipe T.},
  year    = {2025},
  license = {MIT},
  url     = {https://github.com/FelipeTDuarte/NOSSO-MAR}
}
```
