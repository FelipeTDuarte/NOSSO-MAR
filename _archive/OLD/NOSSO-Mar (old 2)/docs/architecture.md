# NOSSO-MAR — System Architecture

## Overview

NOSSO-MAR is a modular physics-informed neural operator framework for coastal and offshore renewable energy applications. The core scientific strategy is to replace expensive numerical solvers with learned operators trained primarily on analytic solutions and public datasets, with solvers used only for high-fidelity validation cases.

---

## Guiding Principles

- **Analytic first.** Closed-form solutions (linear wave theory, Mei 1989; Faltinsen 1990) cover ~80% of the relevant parameter space for point-absorber WECs. Solvers enter only when analytic coverage fails.
- **Contracts before code.** Every module has explicit IO contracts (`WECState`, `WaveField`) with shape and NaN validation before any physics is implemented.
- **TDD throughout.** No module is considered done without a RED-GREEN test cycle.
- **Independent modules.** Each module produces a testable and potentially publishable artefact before the next module is started.
- **Uncertainty from the start.** Even Phase 1 outputs are designed to host uncertainty estimates later. No architectural refactoring needed in Phase 2.

---

## System Layers

```
┌────────────────────────────────────────────────────────┐
│  Layer 0 — IO Contracts                                │
│  WECState, WaveField, validate_wec_state               │
└────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────┐
│  Layer 1 — Data Generation                            │
│  analytic_wec.py  │  wec_dataset.py  │  public_benchmarks │
└────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────┐
│  Layer 2 — Operator Core                              │
│  deeponet_wec.py        (Phase 1)                      │
│  rino_encoder.py        (Phase 2 — stub)               │
│  full_ocean_pino.py     (Phase 3 — planned)            │
└────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────┐
│  Layer 3 — Coupling and FSI                           │
│  multi_object_fsi.py  │  coupled_pino.py               │
│  multi_object_interaction.py  (Phase 2 — stub)         │
└────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────┐
│  Layer 4 — Training and Evaluation                    │
│  train_wec.py  │  validate_phase1.py                    │
└────────────────────────────────────────────────────────┘
```

---

## Tensor Conventions

| Symbol | Shape | Unit | Description |
|--------|-------|------|-------------|
| `omega` | `(N_ω,)` | rad/s | Angular frequency vector |
| `props` | `(B, 4)` | m, m, m, N·s/m | `[radius, draft, depth, B_pto]` |
| `A(ω)` | `(B, N_ω)` | kg | Added mass in heave |
| `B_rad(ω)` | `(B, N_ω)` | kg/s | Radiation damping in heave |
| `eta` | `(B, Nx, Ny, T)` | m | Free surface elevation |
| `pos` | `(N_obj, 2)` | m | WEC x,y positions |
| `vel` | `(N_obj, 6)` | m/s, rad/s | 6-DOF velocity |
| `force` | `(N_obj, 6)` | N, N·m | 6-DOF force/moment |

---

## Data Strategy

Ordered from cheapest to most expensive. Move to the next level only when error exceeds threshold.

1. **Level 0 — Analytic solutions** (zero cost). Linear wave theory, Yeung 1981, Mei 1989. Covers cylinders and simple geometries. Generates thousands of cases in seconds.
2. **Level 1 — Public datasets** (zero cost). CMEMS, WEC-Sim benchmark cases (NREL GitHub), ERA5, ETOPO. Already generated, free, realistic conditions.
3. **Level 2 — Capytaine** (low cost). 500–2000 cases with LHS focused on high-gradient regions. Only after Level 0/1 validation exceeds 15% RMSE threshold.
4. **Level 3 — OpenFOAM / OceanWave3D** (high cost). 10–20 cases only. Exclusively for paper validation figures, never for training in bulk.

---

## Coupling Order

```
WaveField  →  FullOceanPINO (free-surface propagation)
           →  MultiObjectFSI (per-object force extraction)
           →  WECDeepONet (frequency-domain hydrodynamics)
           →  CoupledPINO.step() (coupled time march)
```

---

## Uncertainty Strategy

Phase 1 outputs deterministic predictions. The architecture is uncertainty-ready:
- `WECState.force` can carry an `uncertainty` field in Phase 2 without breaking contracts.
- `validate_phase1.py` already exports RMSE metrics that become the calibration baseline.
- Phase 2 adds MC dropout or ensemble heads as a drop-in inside `WECDeepONet`.
- Phase 3 adds conformal calibration on held-out benchmark data.

---

## Phase Roadmap

| Phase | Scope | Key new files |
|-------|-------|---------------|
| 1 (done) | Single WEC, frequency domain, analytic data | `deeponet_wec.py`, `train_wec.py` |
| 2 | Multi-WEC, resolution-independent encoder, WEC-Sim data | `rino_encoder.py`, `wec_sim_adapter.py`, `multi_object_interaction.py` |
| 3 | Full ocean field, atmospheric forcing, wetting/drying | `full_ocean_pino.py`, `atmospheric_forcing.py` |
| 4 | Farm optimization with GA + surrogate | `layout_optimizer.py`, `genetic_algorithm.py` |
| 5 | Site generalization, transfer learning | `domain_adapter.py`, `curriculum.py` |
| 6 | Data assimilation, 4DVar-PINO | `data_assimilation.py`, `observation_operator.py` |
| 7 | Operational digital twin, real-time dashboard | `digital_twin.py`, `dashboard/` |
