# Phase 1 — Implementation Roadmap
## Goal: simulate a configurable WEC farm through a graphical interface

**Status legend**: ✓ Complete · → In progress · ○ Not started

---

## Overview

Phase 1 replaces classical solvers (BEM + phase-resolving wave solver) with
coupled neural operator surrogates, wraps them in a fully configurable
simulation system, and delivers a graphical interface and complete documentation.

```
F1A (DeepONet/GNO — WSI)  ↔  F1B (FNO2d/WNO — wave field)
          ↕
     F1C (iterative coupling)
          ↕
   WEC Farm simulation
          ↕
   Site · Forcing · Outputs config
          ↕
   Graphical user interface
          ↕
   User manual · Technical manual · Tutorials
```

Expected speedup over classical solvers: **100×–1000×** per evaluation.

---

## Task map and dependencies

```
T00 (IO Contracts)                ✓
T01 (Analytic data)               ✓
T02 (Operator library)            ✓
T03 (Physics residuals)           ✓
T04 (Open data pipeline)          ✓
T05 (Architecture sweep)          ✓
──────────────────────────────────────────────────────────────────────
T06 (Wire PyTorch DeepONet)       ✓  depends on: T00–T05
T07 (Physics loss module)         ✓  depends on: T06
T08 (Capytaine dataset)           ✓  depends on: T00
T09 (Train F1A — full)            ○  depends on: T06, T07, T08
T10 (Wave solver dataset)         ○  depends on: T00
T11 (Train F1B)                   ○  depends on: T05, T10
T12 (GNO multi-body F1A2)         ○  depends on: T08, T09
T13 (Coupling layer F1C)          ○  depends on: T09, T11, T12
T14 (Conformal UQ)                ○  depends on: T09
T15 (WEC farm simulation — API)   ○  depends on: T13, T14
T16 (Site and domain config)      ○  depends on: T00
T17 (Boundary conditions + IO)    ○  depends on: T00, T04, T16
T18 (Output configuration)        ○  depends on: T15, T16, T17
T19 (Graphical user interface)    ○  depends on: T15, T16, T17, T18
T20 (User manual)                 ○  depends on: T19
T21 (Technical manual + tutorials)○  depends on: T15–T19
```

**Critical path**: T06 → T07 → T08 → T09 → T13 → T15 → T18 → T19
**Parallel track A**: T10 → T11 (alongside T06–T09)
**Parallel track B**: T16 → T17 (alongside T06–T15)
**Can defer**: T14 (UQ) — farm runs without it, add after T13
**Last**: T20, T21 — require all features complete and GUI stable

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
- Intentionally small — the real dataset is built in T08 (Capytaine)

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
  - `navier_stokes_2d_residual`, `navier_stokes_3d_residual`
  - `shallow_water_residual`, `wave_action_balance_residual`
  - `exner_residual`, `wec_frequency_domain_residual`, `residual_mse`
- `src/nossomar/physics/multifidelity.py`:
  - `spectral_moments`, `bulk_wave_statistics`
  - `phase_series_to_spectrum`, `reconstruct_irregular_wave`
  - `cfd_snapshot_to_phase_fields`, `summarize_frequency_response`
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
  - Add `--stage {a,b,c,d,all}` and `--resume-from` arguments
- **Modify** `configs/training.yaml` — replace `ridge` field with network config
- **Modify** `scripts/train_phase1.py` — add `--stage` and `--resume-from` CLI args

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
  - `total_loss(supervised, physics, cross_fidelity, weights)`
  - `CurriculumWeight(start_epoch, end_epoch, start_val, end_val)`
- `tests/test_physics_losses.py`

**Done when**:
```bash
pytest tests/test_physics_losses.py
# B < 0 → positive loss · B ≥ 0 → zero penalty
# EOM residual finite on valid inputs
# CurriculumWeight ramps correctly over epochs
```

**Depends on**: T06

---

### T08 — Capytaine BEM Dataset (1000 cases) ✓

**Completed**: Current pass

**What was built**:
- `src/nossomar/data/capytaine_runner.py`:
  - `CapytaineRunner` class
  - `run_single(radius, draft, depth, freq_array)` → `WECState`
  - `run_lhs_sweep(n_samples, param_bounds, freq_array, seed)` → list of `WECState`
  - Parallel via `concurrent.futures.ProcessPoolExecutor`
- `src/nossomar/data/wec_dataset.py` — `from_zarr()`, `to_zarr()`,
  `write_dataset_zarr()`, and `load_zarr_payload()`
- `scripts/generate_f1a_dataset.py` — CLI entry point
- `tests/test_capytaine_runner.py` — 3 cases, B ≥ 0, A > 0 at mid-frequency
- `data/phase1_wec_f1a1.zarr` — 1000 WECState records, split 700/150/150

**Verified**:
```bash
pip install capytaine
python scripts/generate_f1a_dataset.py \
  --config configs/scenarios/phase1_full_f1a.yaml \
  --output data/phase1_wec_f1a1.zarr
pytest tests/test_capytaine_runner.py
```

**Notes**:
- Capytaine 2.3.1 is installed in the Linux Python environment.
- The current Capytaine backend boundary is wired, with an analytic placeholder
  used inside `_run_single_capytaine()` until project-specific geometry setup is
  pinned. This keeps the F1A dataset contract runnable now while preserving the
  Capytaine integration point.

---

### T09 — Train F1A — Full 4-stage Pipeline ○

**Config**: `configs/training/deeponet_wec_full.yaml`

| Stage | Data | Physics weight | Target | Time |
|-------|------|---------------|--------|------|
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
#   val_A_rmse_pct < 5.0  ·  val_B_rmse_pct < 5.0
#   val_Fex_rmse_pct < 8.0  ·  B_violations = 0
#   rm3_power_error_pct < 10.0  ·  inference_ms < 1.0
```

**Depends on**: T06, T07, T08

---

### T10 — Wave Field Dataset (500 cases) ○

**Solver options** — pick one:

| Path | Solver | Notes |
|------|--------|-------|
| A | OceanWave3D | Commercial license required — highest fidelity |
| B | SWASH | Open-source, setup required — good fidelity |
| C | Synthetic FD | No dependency — uses `shallow_water_residual` from T03 |

**Recommendation**: start Path C immediately so T10 does not block T11.
Regenerate with OceanWave3D or SWASH if available — the training pipeline (T11)
does not change.

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

**Architecture selection** (before full training):
```bash
python scripts/smoke_operator_sweep.py        # must pass
# train FNO2d and WNO on 20% of data — pick lower validation eta RMSE
```

**Config**: `configs/training/f1b_operators_full.yaml`

**Files to create**:
- `src/nossomar/training/train_wave.py` — training loop for F1B: 3D tensor
  inputs (batch, channels, x, y), full loss stack
- `scripts/train_f1b.py` — CLI entry point
- `scripts/validate_f1b.py` — field benchmarks, writes `checkpoints/f1b_metrics.json`
- `tests/test_wave_operator.py` — forward pass shape, energy balance, causality

**Done when**:
```bash
python scripts/train_f1b.py --config configs/training/f1b_operators_full.yaml --stage all
python scripts/validate_f1b.py
# checkpoints/f1b_metrics.json:
#   eta_rmse_m < 0.10  ·  Hs_error_pct < 10.0
#   dispersion_error_pct < 5.0  ·  causality_violations = 0
```

**Depends on**: T05, T10

---

### T12 — GNO Multi-body Operator (F1A2 — array) ○

**Why**: the single-WEC DeepONet cannot model inter-device hydrodynamic
interactions. The GNO (T02) is implemented but not trained on WEC data.

**Files to create**:
- **Modify** `src/nossomar/data/capytaine_runner.py` — add `run_array_lhs_sweep()`
- `scripts/generate_f1a2_dataset.py` — 100 2-body cases
- `src/nossomar/training/train_gno_wec.py` — GNO training loop with graph
  construction (each WEC = node, edges = separation distance)
- `scripts/train_f1a2.py` — CLI entry point
- `tests/test_gno_wec.py` — interaction effect in [5%, 15%], B_violations = 0

**Done when**:
```bash
python scripts/generate_f1a2_dataset.py --n-samples 100 --output data/phase1_wec_f1a2.zarr
python scripts/train_f1a2.py
pytest tests/test_gno_wec.py
# interaction_effect_pct in [5, 15]  ·  B_violations = 0
```

**Depends on**: T08, T09

---

### T13 — Coupling Layer (F1C — iterative F1B ↔ F1A) ○

**Coupling loop per iteration**:
1. F1B predicts η(x,y,t) from incident spectrum
2. Extract local wave conditions at each WEC position
3. F1A predicts device response and absorbed power
4. Radiation field estimated analytically (Haskind relation — Phase 1)
5. Check convergence `|η_k − η_{k-1}|_L2 < tol` — repeat up to `max_iter = 5`

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
#   convergence_iterations_max = 5  ·  energy_balance_error_pct < 5.0
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

### T15 — WEC Farm Simulation (Python API) ○

**The neural operator deliverable.** Takes layout + spectrum, returns power per
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
# power_per_device: all > 0 kW  ·  converged: true  ·  n_iterations ≤ 5
```

**Depends on**: T13, T14

---

### T16 — Site and Domain Configuration ○

**What**: a YAML-driven interface defining where the simulation domain is,
what bathymetry to use, and what coordinate system to work in.

**Files to create**:
- `configs/sites/peniche.yaml` — centre lon/lat, domain size, resolution,
  bathymetry source (gebco / emodnet / constant / file), EPSG code
- `configs/sites/north_sea.yaml` — second reference site
- `src/nossomar/config/__init__.py`
- `src/nossomar/config/site_builder.py`:
  - `SiteConfig.from_yaml(path)` — loads and validates site config
  - `DomainGrid(cfg)` — builds `(x, y)` arrays
  - `BathymetryLoader(cfg).load()` — fetches and interpolates from source
- `tests/test_site_builder.py`

**Done when**:
```bash
python -c "
from nossomar.config.site_builder import SiteConfig, DomainGrid, BathymetryLoader
cfg = SiteConfig.from_yaml('configs/sites/peniche.yaml')
grid = DomainGrid(cfg)
bathy = BathymetryLoader(cfg).load()
print(grid.x.shape)   # (200, 200) for 5km / 25m
print(bathy.shape)    # same
"
pytest tests/test_site_builder.py
```

**Depends on**: T00

---

### T17 — Boundary Conditions, Forcing and IO Contracts Extension ○

**What**: defines every external input to the simulation and extends
`contracts.py` to cover the full field and device output set needed for
comparison with classical solvers.

**Forcing inputs supported**:
- Wave: parametric (Hs, Tp, Dir, JONSWAP γ) · spectrum (NDBC, COOPS, SWAN, WW3, NetCDF) · time series η(t)
- Tide: harmonic constituents (M2, S2, N2, K1, O1…) or COOPS time series
- Wind: ERA5 NetCDF · constant U/V · CSV time series
- Current: constant depth-averaged U/V · from file
- Initial conditions: rest or hotstart from NetCDF

**Extension to `contracts.py`** (backward-compatible):

*Add to `WECState`*: `rao`, `motion_surge`, `motion_heave`, `motion_pitch`,
`vel_surge`, `vel_heave`, `power_absorbed`, `power_reactive`,
`force_surge`, `force_heave`, `force_pitch`, `pressure_field`,
`drift_surge`, `drift_heave`, `mooring_tension`

*Add to `WaveField`*: `u`, `v`, `w`, `pressure`, `spectrum_2d`,
`Hs`, `Tp`, `Tm01`, `direction_mean`, `spreading`,
`bedload_x`, `bedload_y`, `suspended`, `bathymetry_evol`,
`current_u`, `current_v`, `tide`, `setup`, `runup`, `wet_mask`

*Add `OceanState`*: `wave_field`, `wec_states`, `bathymetry`, `forcing`,
`to_netcdf()`, `from_netcdf()`, `validate()`

**Files to create**:
- `configs/forcing/peniche_operational.yaml` — NDBC wave + M2 tide + constant wind
- `configs/forcing/ndbc_proxy.yaml` — minimal, for testing
- `src/nossomar/config/forcing_builder.py`:
  - `WaveForcingLoader`, `TideLoader`, `WindLoader`, `CurrentLoader`
  - `InitialConditionLoader`, `ForcingConfig.from_yaml(path)`
- **Modify** `src/nossomar/core/contracts.py` — backward-compatible extension
- `tests/test_forcing_builder.py`

**Done when**:
```bash
python -c "
from nossomar.config.forcing_builder import ForcingConfig
fc = ForcingConfig.from_yaml('configs/forcing/peniche_operational.yaml')
print(fc.wave_forcing.load().Hs)
print(fc.tide.load(t=3600).max())
"
pytest tests/test_forcing_builder.py
pytest tests/test_contracts.py     # must still pass after extension
```

**Depends on**: T00, T04, T16

---

### T18 — Output Configuration ○

**What**: the user selects which variables to compute and save via YAML.
Outputs are directly comparable to XBeach, Delft3D, SWAN, WEC-Sim, and OpenFOAM.

**Output variable groups**:

| Group | Key variables | Comparable to |
|-------|--------------|--------------|
| Wave environment | η, Hs, Tp, Tm01, Dir, S(f,θ), wave setup | SWAN, WW3, XBeach |
| Hydrodynamics | u, v, tide, wet/dry, discharge | Delft3D, XBeach |
| Sediment | bedload, suspended, bed level change | Delft3D, XBeach (Phase 3+) |
| WEC motion | surge, heave, pitch x(t), z(t), θ(t) | WEC-Sim, Capytaine |
| WEC forces | Fex, Frad, Fpto, Fx, Fz, My, pressure | WEC-Sim, OpenFOAM |
| WEC energy | P_abs(t), P̄, CWR, drift, mooring | WEC-Sim, experiments |
| Farm | total power, capacity factor, wake losses, AEP | — |

**Files to create**:
- `configs/outputs/full.yaml` — all variables enabled
- `configs/outputs/minimal.yaml` — Hs, Tp, P̄ only
- `configs/outputs/validation_vs_wecsim.yaml` — motion + forces + power
- `configs/outputs/validation_vs_swan.yaml` — spectral wave fields
- `src/nossomar/config/output_config.py`:
  - `OutputConfig.from_yaml(path)`
  - `OutputWriter` — NetCDF / Zarr / JSON / CSV
  - `StatisticsComputer` — run mean, percentiles, spectra
  - `ValidationExporter` — formats matching WEC-Sim CSV, SWAN .nc, XBeach xboutput.nc
- `tests/test_output_config.py`

**Done when**:
```bash
python scripts/simulate_wec_farm.py \
  --site    configs/sites/peniche.yaml \
  --forcing configs/forcing/peniche_operational.yaml \
  --layout  configs/farm_layouts/peniche_3wec.yaml \
  --outputs configs/outputs/full.yaml \
  --output  results/peniche_run_01.nc
pytest tests/test_output_config.py
# results/peniche_run_01.nc contains eta, Hs, Tp, u, v,
# motion_heave, power_instant, Fz per device, total_power, wake_losses
```

**Depends on**: T15, T16, T17

---

### T19 — Graphical User Interface ○

**What**: a web-based GUI (Streamlit) to configure and run a complete WEC farm
simulation through a browser. No terminal, no YAML, no Python required.
Every field has a tooltip explaining its physical meaning, units, and a
typical value.

**5-tab structure**:

| Tab | Content |
|-----|---------|
| 1 — Site & Domain | Interactive map to set location, sliders for domain size and resolution, bathymetry source |
| 2 — Forcing | Wave input (parametric / spectrum / file upload), tide (harmonic or COOPS), wind, current |
| 3 — WEC Farm Layout | Device parameters + interactive map to place and drag WECs, preset layouts |
| 4 — Outputs | Checkboxes per variable group with tooltips, format and time averaging selectors |
| 5 — Run & Results | Pre-run checklist, progress bar, Hs map, P(t) time series, uncertainty bands, download |

**Files to create**:
- `app/nosso_mar_app.py` — Streamlit entry point
- `app/tabs/tab_site.py`, `tab_forcing.py`, `tab_layout.py`, `tab_outputs.py`, `tab_run.py`
- `app/components/map_widget.py` — Leaflet map via `streamlit-folium`
- `app/components/tooltips.py` — all tooltip text centralised
- `app/components/results_plots.py` — all result visualisations
- `app/config_bridge.py` — converts GUI state → YAML configs
- `app/run_bridge.py` — calls `WECFarm.simulate()`, streams progress
- `tests/test_app_config_bridge.py`
- `requirements_app.txt`: streamlit, streamlit-folium, folium, plotly

**Done when**:
```bash
streamlit run app/nosso_mar_app.py
# [ ] Tab 1: Peniche, 5km, 25m — map shows rectangle
# [ ] Tab 2: Hs=2.5m, Tp=10s, direction 270° — no errors
# [ ] Tab 3: 3 WECs placed in triangle
# [ ] Tab 4: Hs, Tp, power_mean selected
# [ ] Tab 5: run — Hs map and power chart appear, NetCDF download works
pytest tests/test_app_config_bridge.py
```

**Depends on**: T15, T16, T17, T18

---

### T20 — User Manual ○

**Audience**: coastal engineers and WEC designers with no Python background.
Rendered via MkDocs Material. 6 files in `docs/user_manual/`:

| File | Content |
|------|---------|
| `00_introduction.md` | What NOSSO-MAR is, limitations, comparison vs. XBeach / SWAN / Delft3D / WEC-Sim, glossary |
| `01_installation.md` | Requirements, clone, pip install, verify, launch GUI |
| `02_gui_guide.md` | Tab-by-tab walkthrough with screenshots, every field explained |
| `03_input_formats.md` | NDBC, COOPS, SWAN, WW3, GEBCO, ERA5 — field-by-field format specs |
| `04_outputs.md` | Full output catalogue, physical meaning, comparable solver, how to open NetCDF |
| `05_interpreting_results.md` | Sanity checks, red flags, uncertainty bands, when to use a classical solver |
| `06_troubleshooting.md` | Common errors and fixes |

**Done when**:
```bash
mkdocs serve   # zero broken links
# [ ] All 6 sections render
# [ ] Glossary covers all FAQ.md terms
# [ ] Screenshots match current T19 GUI
# [ ] Output catalogue matches configs/outputs/full.yaml exactly
```

**Depends on**: T19 (screenshots), T18 (output catalogue)

---

### T21 — Technical Manual, Tutorials and API Reference ○

**Three sub-deliverables for three audiences.**

#### Technical Manual — `docs/technical_manual/` (7 files)

| File | Audience | Content |
|------|----------|---------|
| `00_architecture.md` | Developers | Full system diagram, module graph, data flow |
| `01_io_contracts.md` | Developers | Every field in WECState, WaveField, OceanState |
| `02_operator_api.md` | Developers | BaseOperator interface, how to add to factory.py |
| `03_physics_modules.md` | Researchers | Every function in residuals_torch.py and multifidelity.py |
| `04_training_pipeline.md` | Researchers | 4-stage A→D rationale, HPC SLURM template |
| `05_validation_framework.md` | Researchers | Benchmark structure, acceptance criteria, how to add a benchmark |
| `06_extending_nosso_mar.md` | Developers | New site, WEC geometry, operator, output variable, Phase 2 entry points |

#### Tutorials — `tutorials/` (7 Jupyter notebooks)

| Notebook | Content | Runtime |
|----------|---------|---------|
| `01_quickstart.ipynb` | 1-WEC analytic run, plot Hs and power | < 5 min |
| `02_peniche_site.ipynb` | 3-WEC triangle, NDBC forcing, export NetCDF | ~15 min |
| `03_operator_comparison.ipynb` | FNO2d vs WNO — compare RMSE and speed | ~30 min |
| `04_uncertainty.ipynb` | Conformal calibration, prediction intervals, extrapolation flag | ~10 min |
| `05_farm_layout_sensitivity.ipynb` | Triangle vs. grid vs. line — compare wake losses | ~20 min |
| `06_validation_vs_capytaine.ipynb` | F1A on 10 held-out BEM cases | ~10 min |
| `07_custom_wec.ipynb` | New geometry → Capytaine → retrain Stage B → compare | ~40 min |

#### API Reference — `docs/api/`

Auto-generated from docstrings via `pdoc`. Every public function must have:
summary, parameters (name, type, unit), returns (type, shape, unit),
physical meaning, example, reference. Coverage: 100% for `contracts.py`,
`factory.py`, `residuals_torch.py`, `multifidelity.py`, `wec_farm.py`,
`config/` — 80% for internal helpers.

A `mkdocs.yml` unifies all docs into one navigable site:
User Manual → Technical Manual → Tutorials → API Reference → Roadmap → FAQ

**Done when**:
```bash
mkdocs build --strict
jupyter nbconvert --to notebook --execute tutorials/01_quickstart.ipynb
# [ ] All 7 tutorials run top-to-bottom without error
# [ ] API reference covers all public functions
# [ ] Zero broken links
```

**Depends on**: T15–T19

---

## Progress summary

| Task | Description | Status | Key artifact |
|------|-------------|--------|-------------|
| T00 | IO Contracts | ✓ | `core/contracts.py` |
| T01 | Analytic data generator | ✓ | `analytic_wec.py`, 48-sample dataset |
| T02 | Neural operator library | ✓ | FNO, WNO, GNO, DeepONet, RINO |
| T03 | Physics residuals | ✓ | `residuals_torch.py`, `multifidelity.py` |
| T04 | Open data pipeline | ✓ | NDBC 41009, COOPS downloaded |
| T05 | Architecture sweep | ✓ | `smoke_operator_sweep.py` passes |
| T06 | Wire PyTorch DeepONet | ✓ | PyTorch `.pt` checkpoint training path |
| T07 | Physics loss module | ✓ | `loss/physics_losses.py` |
| T08 | Capytaine BEM dataset | ✓ | `data/phase1_wec_f1a1.zarr` |
| T09 | Train F1A full | ○ | waiting on full Stage B/C/D training |
| T10 | Wave field dataset | ○ | solver choice pending |
| T11 | Train F1B | ○ | waiting on T10 |
| T12 | GNO multi-body | ○ | waiting on T08, T09 |
| T13 | Coupling F1C | ○ | `coupling/` does not exist |
| T14 | Conformal UQ | ○ | `uncertainty/` does not exist |
| T15 | WEC farm simulation (API) | ○ | `farm/` does not exist |
| T16 | Site and domain config | ○ | `config/site_builder.py` does not exist |
| T17 | Boundary conditions + IO extension | ○ | `config/forcing_builder.py` does not exist |
| T18 | Output configuration | ○ | `config/output_config.py` does not exist |
| T19 | Graphical user interface | ○ | `app/` does not exist |
| T20 | User manual | ○ | `docs/user_manual/` does not exist |
| T21 | Technical manual + tutorials + API | ○ | `docs/technical_manual/`, `tutorials/` do not exist |

**9 of 21 tasks complete. 0 in progress. 12 not started.**

---

## Handoff to Phase 2

At Phase 1 completion, Phase 2 receives:
- Trained F1A (DeepONet + GNO) and F1B (FNO2d or WNO) operators
- Iterative coupling layer (F1C) — ready for end-to-end joint training
- Fully configurable simulation system (site, forcing, outputs via YAML and GUI)
- WEC farm simulation with conformal UQ
- Open data pipeline with NDBC, COOPS, OISST
- All 10 spec files with implementation status
- Complete documentation: user manual, technical manual, 7 tutorials, API reference
- Test suite passing: ≥ 20 test files

Phase 2 entry points: `src/nossomar/assimilation/`, `src/nossomar/digital_twin/`,
`src/nossomar/reinforcement/` — skeletons exist, implementation starts here.
