# Phase 1 — Implementation Roadmap

## Goal: simulate a WEC farm with trained neural operators

**Status legend**: ✓ Complete · → In progress · ○ Not started · ⚠ Partially done (see notes)

---

## Overview

Phase 1 replaces classical solvers (BEM + phase-resolving wave solver) with
coupled neural operator surrogates. The deliverable is a runnable WEC farm
simulation that takes a wave spectrum and layout and returns power per device
with uncertainty estimates.

```text
F1A (DeepONet/GNO — WSI)  ↔  F1B (FNO2d/WNO — wave field)
          ↕
     F1C (iterative coupling)
          ↕
   WEC Farm simulation
```

Expected speedup over classical solvers: **100×–1000×** per evaluation.

---

## Task map and dependencies

```text
T00 (IO Contracts)           ✓
T01 (Analytic data)          ✓
T02 (Operator library)       ✓
T03 (Physics residuals)      ✓
T04 (Open data pipeline)     ✓
T05 (Architecture sweep)     ✓
─────────────────────────────────────────────────────────────
T06 (Wire PyTorch DeepONet)  →  depends on: T00–T05
T07 (Physics loss module)    →  depends on: T06
T08 (Capytaine dataset)      ○  depends on: T00
T09 (Train F1A — full)       ○  depends on: T06, T07, T08
T10 (Wave solver dataset)    ○  depends on: T00
T11 (Train F1B)              ○  depends on: T05, T10
T12 (GNO multi-body F1A2)    ○  depends on: T08, T09
T13 (Coupling layer F1C)     ○  depends on: T09, T11, T12
T14 (Conformal UQ)           ○  depends on: T09
T15 (WEC farm simulation)    ○  depends on: T13, T14
```

**Critical path**: T06 → T07 → T08 → T09 → T13 → T15
**Parallel track**: T10 → T11 (can run alongside T06–T09)
**Can defer**: T14 (UQ) — farm runs without it, add after T13

---

## COMPLETED

---

### T00 — IO Contracts and Types ✓

**Completed**: Phase 0

**What was built**:

- `src/nossomar/core/contracts.py` — `WECState`, `WaveField`, `OceanState`
  with coordinate validation, physical bounds checking, JSON persistence
- `src/nossomar/core/field_schema.py` — `GridDefinition`, `MultiFidelitySample`
- `tests/test_contracts.py` — 10 test cases

**Verified**:

- ✓ All xarray classes serialize/deserialize to JSON
- ✓ Coordinate validation passes
- ✓ Physical bounds checking works (`damping ≥ 0` enforced in `WECState`)

---

### T01 — Analytic WEC Data Generator ✓

**Completed**: Phase 1, Week 1

**What was built**:

- `src/nossomar/data/analytic_wec.py` — closed-form hydrodynamic responses
  (Hulme 1982 cylinder), Airy wave diagnostics, `BRANCH_DIM = 10`,
  `TRUNK_DIM = 7`, `make_frequency_grid()`
- `src/nossomar/data/wec_dataset.py` — LHS sampling, train/val/test splitting,
  JSON serialization, `WECDataset` PyTorch Dataset API
- `tests/test_analytic_wec.py` — 5 canonical cases, <1 second runtime
- `tests/test_wec_dataset.py` — shape tests, regression arrays

**Verified**:

- ✓ 1000+ analytic cases generated in <1 second
- ✓ Dataset split: 70/15/15 train/val/test
- ✓ RMSE vs. Hulme 1982 < 1%

**Current dataset**: `data/phase1_wec_database.json`

- 48 analytic samples (34 train / 7 val / 7 test)
- This is intentionally small — the real dataset is built in T08 (Capytaine)

---

### T02 — Neural Operator Library ✓

**Completed**: Phase 1, Week 1

**What was built** (all in `src/nossomar/operators/`):

- `deeponet/deeponet.py` — full PyTorch DeepONet (branch + trunk MLP)
- `deeponet/physics_deeponet.py` — PI-DeepONet with pluggable PDE residual
- `deeponet/pod_deeponet.py` — POD-DeepONet variant
- `fno/fno2d.py`, `fno/fno3d.py` — Fourier Neural Operator 2D/3D
- `fno/ffno.py` — Factorized FNO
- `fno/geo_fno.py` — Geometry-aware FNO
- `fno/spectral_conv.py` — shared spectral convolution layer
- `wno/wavelet_neural_operator.py`, `wno/wavelet_conv.py` — WNO
- `gno/graph_neural_operator.py` — GNO (3 classes)
- `gno/mesh_utils.py`, `gno/message_passing.py`
- `rino.py` — Resolution-Invariant Neural Operator
- `base.py` — `BaseOperator` with `count_parameters()`
- `factory.py` — `build_operator(name, cfg)` registry

**Verified**:

- ✓ All operators instantiate and run forward pass (smoke sweep)
- ✓ `factory.py` covers: `fno2d`, `fno3d`, `ffno2d`, `geo_fno`, `wno`,
  `gno`, `deeponet`, `physics_deeponet`, `rino2d`

---

### T03 — Physics Residuals ✓

**Completed**: Phase 1, Week 2

**What was built**:

- `src/nossomar/physics/residuals_torch.py`:
  - `navier_stokes_2d_residual`
  - `navier_stokes_3d_residual`
  - `shallow_water_residual`
  - `wave_action_balance_residual`
  - `exner_residual`
  - `wec_frequency_domain_residual`
  - `residual_mse`
- `src/nossomar/physics/multifidelity.py`:
  - `spectral_moments`
  - `bulk_wave_statistics`
  - `phase_series_to_spectrum`
  - `reconstruct_irregular_wave`
  - `cfd_snapshot_to_phase_fields`
  - `summarize_frequency_response`
- `tests/test_multifidelity.py`

**Verified**:

- ✓ All residual functions return finite tensors on valid inputs
- ✓ Multifidelity bridges tested

---

### T04 — Open Data Pipeline ✓

**Completed**: Phase 1, Week 2

**What was built**:

- `src/nossomar/data/open_data_catalog.py`
- `src/nossomar/data/open_data_fetchers.py`
- `src/nossomar/data/database_builder.py`
- `scripts/build_open_database.py`

**Data downloaded** to `data/open_database/downloads/`:

- `ndbc/41009.txt` — offshore wave buoy (Hs, Tp, direction time series)
- `coops/9414290_water_level.json` — SF Bay water level
- `coops/9414290_wind.json` — wind
- `coops/9414290_water_temperature.json` — SST
- `oisst/oisst-avhrr-v02r01.20250428.nc` — satellite SST

**Metadata only** (download required before use):

- `metadata/gebco_grid.html` — GEBCO bathymetry
- `metadata/emodnet_physics.html` — EMODnet physics
- `metadata/rm3_geometry.html` — RM3 WEC geometry
- `metadata/wecsim_rm3_tutorial.html` — WEC-Sim RM3 reference

**Verified**:

- ✓ `tests/test_open_data_catalog.py` passes
- ✓ NDBC and COOPS data parseable

---

### T05 — Architecture Sweep ✓

**Completed**: Phase 1, Week 2

**What was built**:

- `src/nossomar/experiments/architecture_sweep.py` — `run_smoke_sweep()`
- `scripts/smoke_operator_sweep.py`
- `tests/test_operator_factory.py`
- `configs/operator_sweep.yaml`

**Verified**:

- ✓ All 5 operator families pass smoke sweep: `fno2d`, `wno`, `gno`,
  `deeponet`, `rino2d`
- ✓ Forward pass shapes correct for each family
- ✓ Parameter counts non-zero

---

## IN PROGRESS

---

### T06 — Wire PyTorch DeepONet to F1A Training Loop →

**What the current code does**: `train_wec.py` calls `DeepONetWECRegressor`
which is **NumPy ridge regression**, not the PyTorch `DeepONet` from T02.
The checkpoint `deeponet_wec_local.json` is a weight matrix (70×4), not a
PyTorch state dict.

**What needs to change**:

- **Modify** `src/nossomar/training/train_wec.py`:
  - Replace `DeepONetWECRegressor` with `build_operator("deeponet", cfg)`
  - Add PyTorch training loop: `DataLoader`, `optimizer.step()`, `scheduler.step()`
  - Add validation loop with per-channel RMSE reporting
  - Add checkpointing: save `model.state_dict()` (`.pt`) not JSON
  - Keep `--stage {a,b,c,d,all}` and `--resume-from` arguments
- **Modify** `configs/training.yaml`:
  - Replace `ridge` field with full network config
  - Reference: `configs/training/deeponet_wec_full.yaml`
- **Modify** `scripts/train_phase1.py`:
  - Add `--stage` and `--resume-from` CLI arguments

**Done when**:

```bash
python scripts/train_phase1.py --config configs/training/deeponet_wec_full.yaml --stage a
# → checkpoints/stage_a_deeponet.pt exists (PyTorch state dict)
# → checkpoints/stage_a_metrics.json: val_relative_A < 0.10, B_violations = 0
pytest tests/test_deeponet_wec.py  # still passes
```

---

## NOT STARTED

---

### T07 — Physics Loss Module ○

**Why it doesn't exist**: `residuals_torch.py` (T03) has the math but nothing
connects it to the training loop.

**Files to create**:

- `src/nossomar/loss/__init__.py`
- `src/nossomar/loss/physics_losses.py`:
  - `damping_nonneg_loss(B_pred)` — soft relu penalty for B < 0
  - `wec_eom_loss(A, B, Fex_real, Fex_imag, freq, mass, bpto, stiffness)`
    wrapping `wec_frequency_domain_residual`
  - `total_loss(supervised, physics, cross_fidelity, weights)`
  - `CurriculumWeight(start_epoch, end_epoch, start_val, end_val)`
- `tests/test_physics_losses.py`

**Done when**:

```bash
pytest tests/test_physics_losses.py
# B < 0 → positive loss, B ≥ 0 → zero penalty
# EOM residual finite on valid inputs
# CurriculumWeight ramps correctly over epochs
```

**Depends on**: T06

---

### T08 — Capytaine BEM Dataset (1000 cases) ○

**Why it matters**: the current dataset has 48 analytic samples. The trained
model generalises to synthetic responses only. Capytaine provides real BEM
coefficients for arbitrary geometries.

**Files to create**:

- `src/nossomar/data/capytaine_runner.py`:
  - `CapytaineRunner` class
  - `run_single(radius, draft, depth, freq_array)` → `WECState`
  - `run_lhs_sweep(n_samples, param_bounds, freq_array, seed)` → list of `WECState`
  - Parallel via `concurrent.futures.ProcessPoolExecutor`
- **Modify** `src/nossomar/data/wec_dataset.py`:
  - Add `from_zarr()` and `to_zarr()` alongside existing JSON path
- `scripts/generate_f1a_dataset.py` — CLI entry point
- `tests/test_capytaine_runner.py` — 3 cases, B ≥ 0, A > 0 at mid-frequency

**Done when**:

```bash
pip install capytaine
python scripts/generate_f1a_dataset.py \
  --config configs/scenarios/phase1_full_f1a.yaml \
  --output data/phase1_wec_f1a1.zarr
# → 1000 WECState records, split by radius tercile
pytest tests/test_capytaine_runner.py
```

**Depends on**: T00

---

### T09 — Train F1A — Full 4-stage Pipeline ○

**Config**: `configs/training/deeponet_wec_full.yaml`

| Stage | Data | Physics weight | Target | Time |
| ------- | ------ | --------------- | -------- | ------ |
| A — Analytic harness | `analytic_wec.py` | 0.0 | Loss finite, B_violations = 0 | ~10 min |
| B — Capytaine conditioning | `phase1_wec_f1a1.zarr` | 0.05 | val RMSE < 7% | ~2h |
| C — Physics-informed | `phase1_wec_f1a1.zarr` | 0.1 (ramped) | val RMSE < 5%, B_violations = 0 | ~6h |
| D — HAMS calibration | 10% of F1A1 cross-validated | 0.2 | power error < 10% on RM3 | ~1h |

**Files to create**:

- `scripts/validate_f1a.py` — runs all benchmarks, writes `checkpoints/f1a_metrics.json`

**Done when**:

```bash
python scripts/train_phase1.py --config configs/training/deeponet_wec_full.yaml --stage all
python scripts/validate_f1a.py
# checkpoints/f1a_metrics.json:
#   val_A_rmse_pct < 5.0
#   val_B_rmse_pct < 5.0
#   val_Fex_rmse_pct < 8.0
#   B_violations = 0
#   rm3_power_error_pct < 10.0
#   inference_ms < 1.0
```

**Depends on**: T06, T07, T08

---

### T10 — Wave Field Dataset (500 cases) ○

**Solver options** — pick one:

| Path | Solver | Status | Notes |
| ------ | -------- | -------- | ------- |
| A | OceanWave3D | Commercial license required | Highest fidelity |
| B | SWASH | Open-source, setup required | Good fidelity |
| C | Synthetic FD | No external dependency | Use `shallow_water_residual` from T03 |

**Recommendation**: start Path C immediately so T10 does not block T11.
Regenerate with OceanWave3D or SWASH if available later — the training
pipeline (T11) does not change.

**Files to create**:

- `src/nossomar/data/{solver}_runner.py` (whichever path)
- `src/nossomar/data/wave_dataset.py` — `WaveFieldDataset` with Zarr I/O,
  PyTorch Dataset API, bridge calls to `multifidelity.py`
- `scripts/generate_f1b_dataset.py` — CLI entry point
- `tests/test_wave_dataset.py`

**Done when**:

```bash
python scripts/generate_f1b_dataset.py \
  --config configs/scenarios/phase1_full_f1b.yaml \
  --output data/phase1_wave_f1b.zarr
pytest tests/test_wave_dataset.py
# 500 cases: eta(128,128,T), bulk stats (Hs, Tp, Te) per case
```

**Depends on**: T00

---

### T11 — Train F1B — Wave Field Operator ○

**Architecture selection** (run before full training):

```bash
python scripts/smoke_operator_sweep.py  # must pass
# Train FNO2d and WNO on 20% of data — pick lower validation eta RMSE
```

**Config**: `configs/training/f1b_operators_full.yaml`

**Files to create**:

- `src/nossomar/training/train_wave.py` — training loop for F1B: handles
  3D tensor inputs (batch, channels, x, y), full loss stack
- `scripts/train_f1b.py` — CLI entry point
- `scripts/validate_f1b.py` — field benchmarks, writes `checkpoints/f1b_metrics.json`
- `tests/test_wave_operator.py` — forward pass shape, energy balance, causality

**Done when**:

```bash
python scripts/train_f1b.py --config configs/training/f1b_operators_full.yaml --stage all
python scripts/validate_f1b.py
# checkpoints/f1b_metrics.json:
#   eta_rmse_m < 0.10
#   Hs_error_pct < 10.0
#   dispersion_error_pct < 5.0
#   causality_violations = 0
```

**Depends on**: T05, T10

---

### T12 — GNO Multi-body Operator (F1A2 — 2-body array) ○

**Why**: the single-WEC DeepONet cannot model inter-device hydrodynamic
interactions. The GNO (T02) is implemented but not trained on WEC data.

**Files to create**:

- **Modify** `src/nossomar/data/capytaine_runner.py` — add `run_array_lhs_sweep()`
- `scripts/generate_f1a2_dataset.py` — 100 2-body cases
- `src/nossomar/training/train_gno_wec.py` — GNO training loop with graph
  construction (each WEC = node, edges = separation distance)
- `scripts/train_f1a2.py` — CLI entry point
- `tests/test_gno_wec.py` — 2-body forward pass, interaction effect in [5%, 15%]

**Done when**:

```bash
python scripts/generate_f1a2_dataset.py --n-samples 100 --output data/phase1_wec_f1a2.zarr
python scripts/train_f1a2.py
pytest tests/test_gno_wec.py
# interaction_effect_pct in [5, 15]
# B_violations = 0
```

**Depends on**: T08, T09

---

### T13 — Coupling Layer (F1C — iterative F1B ↔ F1A) ○

**Coupling loop per iteration**:

1. F1B predicts η(x,y,t) from incident spectrum
2. Extract local wave conditions at each WEC position
3. F1A predicts device response and absorbed power
4. Radiation field estimated analytically (Haskind relation — Phase 1)
5. Modify incident field, check convergence `|η_k - η_{k-1}|_L2 < tol`
6. Repeat up to `max_iter = 5`

**Files to create**:

- `src/nossomar/coupling/__init__.py`
- `src/nossomar/coupling/iterative_coupler.py`:
  - `IterativeCoupler(f1b_operator, f1a_operator, radiation_model, config)`
  - `run(wave_spectrum, wec_layout, max_iter, tol)` → `CoupledResult`
  - `CoupledResult`: eta_field, power_per_device, n_iterations, converged
- `src/nossomar/coupling/radiation_model.py`:
  - `AnalyticRadiationModel` (Haskind — Phase 1)
  - `LearnedRadiationModel` placeholder (Phase 2)
- `scripts/validate_coupling.py` — 3 benchmarks from `phase1_full_f1c.yaml`
- `tests/test_coupling.py`

**Done when**:

```bash
python scripts/validate_coupling.py --config configs/scenarios/phase1_full_f1c.yaml
pytest tests/test_coupling.py
# checkpoints/f1c_metrics.json:
#   single_wec_power_error_pct < 10.0
#   two_body_interaction_pct in [5, 15]
#   convergence_iterations_max = 5
#   energy_balance_error_pct < 5.0
```

**Depends on**: T09, T11, T12

---

### T14 — Conformal Uncertainty Quantification ○

**Files to create**:

- `src/nossomar/uncertainty/__init__.py`
- `src/nossomar/uncertainty/conformal.py`:
  - `ConformalPredictor(model, calibration_dataset, alpha=0.05)`
  - `calibrate()` — nonconformity scores on calibration set
  - `predict_interval(x)` → `(lower, upper, score)` per channel
  - `coverage_test(test_dataset)` → actual coverage % (target ≥ 95%)
  - `extrapolation_flag(x, training_bounds)` → bool
  - `fallback_trigger(x)` → bool (uncertainty > 30% of signal)
- `scripts/calibrate_uncertainty.py`
- `tests/test_uncertainty.py`

**Done when**:

```bash
python scripts/calibrate_uncertainty.py \
  --checkpoint checkpoints/deeponet_wec_f1a1_best.pt
pytest tests/test_uncertainty.py
# coverage ≥ 90% on test set
# extrapolation flag fires on out-of-range params
# checkpoints/f1a_conformal_quantiles.json exists
```

**Depends on**: T09

---

### T15 — WEC Farm Simulation ○

**The Phase 1 deliverable.** Takes a layout + spectrum, returns power per
device, total farm power, wave field, and uncertainty intervals.

**Files to create**:

- `src/nossomar/farm/__init__.py`
- `src/nossomar/farm/wec_farm.py`:
  - `WECFarm(layout, device_params, f1b_checkpoint, f1a_checkpoint, f1a2_checkpoint)`
  - `simulate(wave_spectrum, n_iter=5)` → `FarmResult`
  - `FarmResult`: power_per_device, total_power, eta_field,
    uncertainty_per_device, converged, n_iterations
- `scripts/simulate_wec_farm.py` — CLI entry point
- `configs/farm_layouts/peniche_3wec.yaml` — 3-WEC triangle, Peniche site
- `configs/farm_layouts/north_sea_10wec.yaml` — 10-WEC grid (Phase 7 baseline)
- `tests/test_wec_farm.py`

**Done when**:

```bash
python scripts/simulate_wec_farm.py \
  --layout configs/farm_layouts/peniche_3wec.yaml \
  --spectrum data/open_database/downloads/ndbc/41009.txt \
  --output results/peniche_farm_result.json

pytest tests/test_wec_farm.py

# results/peniche_farm_result.json:
#   power_per_device: [P1, P2, P3]  (all > 0 kW)
#   total_power: > 0 kW
#   converged: true
#   n_iterations: ≤ 5
#   uncertainty: intervals per device
```

**Usage example**:

```python
from nossomar.farm.wec_farm import WECFarm

farm = WECFarm(
    layout=[(0, 0), (150, 0), (75, 130)],
    device_params={"radius": 5.0, "draft": 3.5, "depth": 50.0, "bpto": 50.0},
    f1b_checkpoint="checkpoints/fno2d_wave_f1b_best.pt",
    f1a_checkpoint="checkpoints/deeponet_wec_f1a1_best.pt",
    f1a2_checkpoint="checkpoints/gno_wec_f1a2_best.pt",
)
result = farm.simulate("data/open_database/downloads/ndbc/41009.txt")
print(result.power_per_device)  # [P1, P2, P3] kW
print(result.total_power)       # sum kW
```

**Depends on**: T13, T14

---

## Progress summary

| Task | Status | Key artifact |
| ------ | -------- | ------------- |
| T00 IO Contracts | ✓ | `core/contracts.py`, `test_contracts.py` |
| T01 Analytic data | ✓ | `analytic_wec.py`, 48-sample dataset |
| T02 Operator library | ✓ | FNO, WNO, GNO, DeepONet, RINO all implemented |
| T03 Physics residuals | ✓ | `residuals_torch.py`, `multifidelity.py` |
| T04 Open data pipeline | ✓ | NDBC 41009, COOPS downloaded |
| T05 Architecture sweep | ✓ | `smoke_operator_sweep.py` passes |
| T06 Wire PyTorch DeepONet | → | `train_wec.py` uses ridge regression — needs replacement |
| T07 Physics loss module | ○ | file does not exist |
| T08 Capytaine dataset | ○ | `capytaine_runner.py` does not exist |
| T09 Train F1A full | ○ | waiting on T06, T07, T08 |
| T10 Wave field dataset | ○ | solver choice pending |
| T11 Train F1B | ○ | waiting on T10 |
| T12 GNO multi-body | ○ | waiting on T08, T09 |
| T13 Coupling F1C | ○ | `coupling/` does not exist |
| T14 Conformal UQ | ○ | `uncertainty/` does not exist |
| T15 WEC farm | ○ | `farm/` does not exist |

**6 of 15 tasks complete.**

---

## Handoff to Phase 2

At Phase 1 completion, Phase 2 receives:

- Trained F1A (DeepONet + GNO) and F1B (FNO2d or WNO) operators
- Iterative coupling layer (F1C) — ready for end-to-end joint training
- WEC farm simulation with conformal UQ
- Open data pipeline with NDBC, COOPS, OISST
- All 10 spec files with implementation status
- Test suite passing: ≥ 12 test files

Phase 2 entry points: `src/nossomar/assimilation/`, `src/nossomar/digital_twin/`,
`src/nossomar/reinforcement/` — skeletons exist, implementation starts here.
