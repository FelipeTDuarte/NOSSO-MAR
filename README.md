# NOSSO-MAR

**Neural Operator Scalable System for Ocean Modelling, Analysis, and Renewables**

A research-grade, modular AI framework that replaces classical ocean and coastal
physics solvers with physics-informed neural operators — from wave propagation and
wave energy converter response to data assimilation, morphodynamics, and real-time
digital twin operation with multi-agent reinforcement learning optimisation.

---

## What this project builds

NOSSO-MAR is structured as eight progressive phases. Each phase adds a physical
process and a neural operator family to handle it. The end state is an operational
digital twin for offshore wave energy farms: real-time state estimation, device
response prediction, and reinforcement learning layout and control optimisation.

```
Observations (buoys · altimeters · SCADA)
        ↓
Data Assimilation — EnKF / 4D-Var (Phase 2)
        ↓
State estimate: η(x,y,t), Hs, Tp, currents, tides
        ↓               ↓
  Wave field       WEC response
  operator F1B     operator F1A
  (FNO / WNO)      (DeepONet / GNO)
        ↕  F1C coupling  ↕
   WEC farm simulation
        ↓
   MARL optimisation
   Layout (MADDPG) + PTO control (MAPPO)
        ↓
   Optimised farm → Power output
```

---

## Phases

### Phase 0 — Foundational Architecture ✓
IO contracts, five neural operator families (FNO, WNO, GNO, DeepONet, RINO),
physics residual modules (Navier-Stokes, shallow-water, wave-action, Exner,
WEC equation of motion), multi-fidelity bridges, open data pipeline, test suite.

### Phase 1 — WEC Farm Simulation → *in progress*
Three coupled operators replace classical BEM and phase-resolving wave solvers:

**F1A — Wave-Structure Interaction**: DeepONet and GNO learn hydrodynamic
coefficients (added mass, damping, excitation force), device motions (surge,
heave, pitch), forces, and absorbed power from Capytaine BEM sweeps.
Expected speedup: **100×–1000×** vs. Capytaine at inference.

**F1B — Wave field**: FNO2d or WNO learns η(x,y,t), u, v from bathymetry
and boundary spectrum. Replaces OceanWave3D / SWASH at inference.

**F1C — Iterative coupling**: F1B and F1A exchange wave conditions and
radiation effects until convergence (typically ≤ 5 iterations).

The Phase 1 deliverable is a fully configurable WEC farm simulation:
wave input (parametric, spectrum, or time series) + bathymetry + tides +
wind + current + farm layout → power per device + wave field + uncertainty
intervals, written to NetCDF or Zarr. Outputs are directly comparable to
SWAN, Delft3D, WEC-Sim, XBeach, and OpenFOAM. A Streamlit GUI and full
documentation (user manual, technical manual, 7 tutorials, API reference)
complete the phase.

First study site: **Viana do Castelo, Portugal** — 50-WEC layout 1 km offshore.

### Phase 2 — Spectral Waves + Data Assimilation
Neural operators become the forecast model inside a data assimilation cycle.
Ensemble Kalman Filter (EnKF) and 4D-Var with automatic differentiation
assimilate observations from wave buoys, COOPS tide gauges, and satellite
altimeters. Replaces the classical numerical model in the analysis-forecast
cycle at a fraction of the cost. Supports WAVEWATCH III and ERA5 as spectral
boundary input.

### Phase 3 — Hydrodynamics and Tides
Barotropic tidal dynamics, harmonic constituent fitting (M2, S2, N2, K1, O1),
depth-averaged currents, wave-current interaction, wave setup, and runup.
Enables nearshore and estuarine sites where tidal range is significant.

### Phase 4 — Morphodynamics
Bedload and suspended sediment transport, shoreline evolution, bed level change.
Governed by the Exner equation (already implemented in `residuals_torch.py`).
Bidirectional coupling with wave propagation — morphology changes the wave field,
wave field drives the morphology.

### Phase 5 — Tracer Transport
Temperature, salinity, suspended particulate matter, and contaminants advected
by the velocity field from Phase 3. Relevant for environmental impact assessment
of offshore renewable installations.

### Phase 6 — Advanced FSI / CFD
Full incompressible Navier-Stokes for the near-device flow field. Large
deformations, non-linear Froude-Krylov, pressure distributions. Used as a
high-fidelity reference to calibrate Phase 1 surrogates and to validate results
for extreme sea states.

### Phase 7 — Digital Twin + MARL + Unified Platform
Real-time sensor ingestion feeds a Phase 2 state estimator, which drives the
Phase 1 farm simulation in a continuous prediction loop.

**Layout optimisation**: MADDPG — each WEC is an agent, action = Δposition,
reward = total farm power. Learns optimal spacing and heading alignment.

**PTO control**: MAPPO — each WEC is an agent, action = Δdamping,
reward = individual + cooperative absorbed power.

REST API for integration with SCADA, grid operators, and external tools.
HPC deployment on LUMI, MareNostrum 5, Leonardo, and Deucalion.

---

## Fidelity ladder

NOSSO-MAR never compares outputs from different fidelity levels without an
explicit bridge. The `multifidelity.py` module provides the translations.

| Level | Content | Neural operator role |
|-------|---------|---------------------|
| L0 Observations | Buoy, altimeter, gauge data | Training context, assimilation targets |
| L1 Spectral | Hs, Tp, S(f,θ), ERA5, WW3 | Boundary conditions, spectral consistency loss |
| L2 Phase-resolved | η(x,y,t), u, v | F1B training target |
| L3 WEC response | A(ω), B(ω), Fex, motion, power | F1A training target |
| L4 CFD / FSI | Full Navier-Stokes fields | Sparse calibration reference |

---

## Operator families

| Family | Best for | Phases |
|--------|---------|--------|
| FNO2d / F-FNO | Regular grids, structured spatiotemporal fields | 1–3 |
| WNO | Multi-scale spatial structure, shoaling, localised gradients | 1–3 |
| GNO | Irregular geometry, device arrays, coastal meshes | 1, 7 |
| DeepONet / PI-DeepONet | Frequency-domain response surfaces, transfer maps | 1, 6 |
| RINO | Resolution transfer, super-resolution, fidelity bridging | 1–2 |

---

## First study site — Viana do Castelo, Portugal

```bash
python scripts/prepare_viana_dataset.py
# → configs/viana_50_layout.json     (50 WEC positions, 1 km offshore)
# → data/datasets/viana_50.json      (analytic WEC dataset)
# → data/open_database/viana/        (metocean metadata)
```

---

## Quickstart

```bash
# Install
pip install -e .

# Generate Phase 1 analytic dataset
python scripts/generate_phase1_dataset.py

# Prepare Viana do Castelo site
python scripts/prepare_viana_dataset.py

# Architecture smoke sweep (all operator families)
python scripts/smoke_operator_sweep.py

# Train WEC operator
python scripts/train_phase1.py

# Build open metocean database
python scripts/build_open_database.py

# Run tests
pytest tests/ -v
```

---

## Repository structure

```
NOSSO-MAR/
├── specs/              ← Technical specifications (01–10, growing)
├── docs/               ← Architecture, roadmaps, contributing, FAQ
├── src/nossomar/
│   ├── core/           ← IO contracts and field schemas
│   ├── data/           ← Analytic generator, open data pipeline, datasets
│   ├── operators/      ← FNO, WNO, GNO, DeepONet, RINO, factory
│   ├── physics/        ← Residuals, multifidelity bridges
│   ├── loss/           ← Physics loss module (damping, EOM, curriculum)
│   ├── training/       ← Training loops
│   ├── experiments/    ← Architecture sweep
│   ├── assimilation/   ← EnKF, 4D-Var (Phase 2 skeleton)
│   ├── digital_twin/   ← Sensor ingest, state estimation (Phase 7 skeleton)
│   └── reinforcement/  ← MADDPG, MAPPO, farm env (Phase 7 skeleton)
├── configs/            ← Training, HPC, site, scenario, output configs
├── scripts/            ← Entry points
├── tests/              ← pytest test suite (TDD)
├── data/               ← Generated data and open database
├── checkpoints/        ← Model weights and metrics
└── _archive/           ← Previous versions (reference)
```

---

## Documentation

| Document | Content |
|----------|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, fidelity layers, digital twin diagram |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Full roadmap, Phases 0–7 |
| [`docs/PHASE_1_ROADMAP.md`](docs/PHASE_1_ROADMAP.md) | Phase 1 implementation checklist (21 tasks) |
| [`docs/DATA_ASSIMILATION.md`](docs/DATA_ASSIMILATION.md) | EnKF and 4D-Var design (Phase 2) |
| [`docs/HPC_GUIDE.md`](docs/HPC_GUIDE.md) | LUMI, MareNostrum 5, Leonardo, Deucalion |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | TDD workflow, subagent dispatch, PR template |
| [`docs/FAQ.md`](docs/FAQ.md) | Common questions |
| `specs/01–10` | Technical specifications per component |

---

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
