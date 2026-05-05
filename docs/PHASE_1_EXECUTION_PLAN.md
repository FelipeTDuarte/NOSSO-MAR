# Execution Plan — NOSSO-MAR Phase 1
## Goal: simulate a WEC farm with trained neural operators

---

## State of the repo right now

Before the plan, what actually exists vs what the roadmap claims:

| Component | Roadmap says | Reality |
|-----------|-------------|---------|
| `deeponet_wec.py` | ✓ Complete | **Ridge regression in NumPy, not PyTorch DeepONet** |
| `deeponet/deeponet.py` | — | PyTorch DeepONet exists but not wired to F1A training |
| Dataset | 500 Capytaine cases | **48 analytic samples (34 train / 7 val / 7 test)** |
| `capytaine_runner.py` | ✓ Week 2 | **Does not exist** |
| `loss/physics_losses.py` | ✓ Complete | **Does not exist** |
| `coupling/` | Optional | **Does not exist** |
| `uncertainty/` | Optional | **Does not exist** |
| F1B wave operator | In progress | **Architecture not selected, no data, no file** |
| F1C coupling | Optional | **Does not exist** |
| WEC farm simulation | (Phase 7) | **Does not exist** |

Tasks 0, 1, and 4 are genuinely complete.
Tasks 2, 3, 5, 6, 7 are partially complete or not started.

---

## Execution tasks

Numbered sequentially. Each task has:
- **What to build** — the actual files and code
- **Done when** — a binary test you can run
- **Depends on** — what must exist first

---

### T01 — Wire PyTorch DeepONet to the F1A training loop

**What**: The full PyTorch `DeepONet` (`src/nossomar/operators/deeponet/deeponet.py`)
exists but the training loop (`train_wec.py`) uses only the ridge regression
surrogate (`deeponet_wec.py`). This task replaces the ridge regressor with
the real network.

**Files to create / modify**:
- **Modify** `src/nossomar/training/train_wec.py` — replace `DeepONetWECRegressor`
  with a training loop that calls `build_operator("deeponet", cfg)`, runs forward
  passes, computes MSE loss, calls `optimizer.step()`
- **Modify** `configs/training.yaml` — replace ridge config with network config
  (branch_input_dim, trunk_input_dim, hidden_dim, n_hidden, output_dim, p,
  epochs, lr, batch_size, scheduler)

**Done when**:
```bash
python scripts/train_phase1.py --config configs/training.yaml
# → checkpoints/deeponet_wec_best.pt created (PyTorch state dict, not JSON)
pytest tests/test_deeponet_wec.py  # still passes
```

**Depends on**: nothing new — all components exist

---

### T02 — Add physics loss module (damping non-negativity + EOM residual)

**What**: `src/nossomar/loss/physics_losses.py` is listed as complete in the
roadmap but does not exist. `residuals_torch.py` has `wec_frequency_domain_residual`
but it is not connected to any training loop.

**Files to create**:
- **Create** `src/nossomar/loss/__init__.py`
- **Create** `src/nossomar/loss/physics_losses.py`:
  - `damping_nonneg_loss(B_pred)` — penalises B < 0 (soft relu penalty)
  - `wec_eom_loss(A, B, Fex_real, Fex_imag, freq, mass, bpto, stiffness)` — wraps
    `wec_frequency_domain_residual` from `residuals_torch.py`
  - `total_loss(supervised, physics, cross_fidelity, weights)` — weighted sum
  - `CurriculumWeight(start, end, start_val, end_val)` — ramps physics weight
    from 0 to target over N epochs
- **Modify** `src/nossomar/training/train_wec.py` — import and apply physics loss
  after T01 is done
- **Create** `tests/test_physics_losses.py` — verify B < 0 → positive loss,
  B ≥ 0 → zero penalty, EOM residual finite on valid inputs

**Done when**:
```bash
pytest tests/test_physics_losses.py
# → all pass, B violations = 0 on training run
```

**Depends on**: T01

---

### T03 — Build Capytaine dataset runner (1000 BEM cases)

**What**: The entire dataset is 48 analytic samples. Capytaine is the open-source
BEM solver that generates realistic hydrodynamic coefficients. This task
creates the sweep runner and generates the full dataset.

**Files to create**:
- **Create** `src/nossomar/data/capytaine_runner.py`:
  - `CapytaineRunner` class wrapping the Capytaine Python API
  - `run_single(radius, draft, depth, freq_array)` → returns `WECState`
  - `run_lhs_sweep(n_samples, param_bounds, freq_array, seed)` → list of `WECState`
  - Uses Latin Hypercube Sampling (already in `configs/lhs_phase1.yaml`)
  - Parallel execution via `concurrent.futures.ProcessPoolExecutor`
  - Skips and logs failed cases (non-convergence)
- **Modify** `src/nossomar/data/wec_dataset.py` — add `from_zarr()` and `to_zarr()`
  methods alongside the existing JSON path (Zarr handles 1000 cases efficiently)
- **Create** `scripts/generate_f1a_dataset.py` — CLI entry point that reads
  `configs/scenarios/phase1_full_f1a.yaml` and calls `CapytaineRunner`
- **Create** `tests/test_capytaine_runner.py` — run 3 cases, verify WECState
  valid, B ≥ 0, A > 0 at mid-frequency

**Done when**:
```bash
pip install capytaine
python scripts/generate_f1a_dataset.py --n-samples 1000 --output data/phase1_wec_f1a1.zarr
# → zarr archive with 1000 WECState records, split by radius tercile
pytest tests/test_capytaine_runner.py
```

**Depends on**: nothing new — but Capytaine must be installed

---

### T04 — Train PyTorch DeepONet on Capytaine data (Stage B → C)

**What**: Run the full 4-stage training from `configs/training/deeponet_wec_full.yaml`.
This is primarily execution, not development — but requires T01, T02, T03.

**Stage A** (analytic, ~10 min): verify loss is finite, B violations = 0
**Stage B** (Capytaine 1000 cases, ~2h): val RMSE < 7% all channels
**Stage C** (PI-DeepONet with physics curriculum, ~6h): val RMSE < 5%, B violations = 0
**Stage D** (HAMS cross-validation calibration, ~1h): power error < 10% on RM3

**Files to modify**:
- **Modify** `scripts/train_phase1.py` — add `--stage {a,b,c,d,all}` argument
  and `--resume-from` argument for checkpoint chaining between stages
- **Create** `scripts/validate_f1a.py` — loads final checkpoint, runs all
  benchmarks from `phase1_full_f1a.yaml` (RM3, cylinder analytic, HAMS crossval),
  writes `checkpoints/f1a_metrics.json`

**Done when**:
```bash
python scripts/train_phase1.py --config configs/training/deeponet_wec_full.yaml --stage all
python scripts/validate_f1a.py
# checkpoints/f1a_metrics.json:
#   val_A_rmse_pct < 5.0
#   val_B_rmse_pct < 5.0
#   B_violations = 0
#   rm3_power_error_pct < 10.0
```

**Depends on**: T01, T02, T03

---

### T05 — Generate F1B wave field dataset (500 OceanWave3D / SWASH cases)

**What**: F1B needs phase-resolved wave field data. Two paths depending on
what you have access to:

**Path A — OceanWave3D** (preferred, if licensed):
- Create `src/nossomar/data/oceanwave3d_runner.py` wrapping the CLI
- Sweep: flat, slope, shoal, nearshore_bar bathymetries × JONSWAP/PM/bimodal spectra

**Path B — SWASH** (open-source alternative):
- Create `src/nossomar/data/swash_runner.py` wrapping the SWASH CLI
- Same sweep structure as Path A

**Path C — synthetic shallow-water** (fallback if neither solver available):
- Create `src/nossomar/data/synthetic_wave_runner.py`
- Integrate the shallow-water equations numerically (finite difference)
  using the existing `shallow_water_residual` in `residuals_torch.py` as the
  governing equation — self-consistent but lower fidelity

**Files to create** (whichever path):
- `src/nossomar/data/{solver}_runner.py`
- `src/nossomar/data/wave_dataset.py` — `WaveFieldDataset` class with Zarr I/O,
  PyTorch Dataset API, and bridge calls to `multifidelity.py`
- `scripts/generate_f1b_dataset.py` — CLI entry point
- `tests/test_wave_dataset.py` — load 3 cases, verify eta shape, bulk stats

**Done when**:
```bash
python scripts/generate_f1b_dataset.py --n-samples 500 --output data/phase1_wave_f1b.zarr
pytest tests/test_wave_dataset.py
# → 500 cases, each: eta(128,128,T), bulk stats (Hs, Tp, Te)
```

**Depends on**: solver installation (OceanWave3D, SWASH, or nothing for Path C)

---

### T06 — Train F1B wave field operator (FNO2d / WNO)

**What**: Select and train the wave field operator using the 4-stage pipeline
from `configs/training/f1b_operators_full.yaml`. Run architecture sweep first.

**Files to create**:
- **Create** `src/nossomar/training/train_wave.py` — training loop for F1B
  operators; mirrors `train_wec.py` structure but handles 3D tensor inputs
  (batch, channels, x, y) and the full loss stack (supervised + shallow-water
  residual + spectral consistency)
- **Create** `scripts/train_f1b.py` — CLI entry point
- **Create** `scripts/validate_f1b.py` — runs field benchmarks
  (flat bottom vs Airy, shoaling vs Green's Law, NDBC spectral comparison)
  and writes `checkpoints/f1b_metrics.json`
- **Create** `tests/test_wave_operator.py` — forward pass shape, energy balance,
  causality check (output before forcing must be zero)

**Architecture selection logic** (from spec 04):
- Run `python scripts/smoke_operator_sweep.py` — all must pass
- Train FNO2d and WNO on 20% of data — pick lower validation RMSE
- Proceed with winner for full training

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

**Depends on**: T05

---

### T07 — Build GNO multi-body operator (F1A2 — 2-body array)

**What**: The single-WEC DeepONet (T04) cannot model inter-device hydrodynamic
interactions. The GNO (`src/nossomar/operators/gno/graph_neural_operator.py`)
is already implemented but not trained on WEC data. This task generates the
2-body dataset and trains the GNO.

**Files to create**:
- **Modify** `src/nossomar/data/capytaine_runner.py` (from T03) — add
  `run_array_lhs_sweep(n_bodies, separation_range, ...)` method for multi-body
  Capytaine runs
- **Create** `scripts/generate_f1a2_dataset.py` — 100 2-body cases
- **Create** `src/nossomar/training/train_gno_wec.py` — GNO training loop with
  graph construction (each WEC = node, edges = separation distance), per-device
  loss, interaction kernel loss
- **Create** `scripts/train_f1a2.py` — CLI entry point
- **Create** `tests/test_gno_wec.py` — 2-body forward pass, interaction effect
  within spec (5–15% power reduction on second WEC)

**Done when**:
```bash
python scripts/generate_f1a2_dataset.py --n-samples 100 --output data/phase1_wec_f1a2.zarr
python scripts/train_f1a2.py --config configs/training/deeponet_wec_full.yaml
pytest tests/test_gno_wec.py
# → interaction_effect_pct in [5, 15]
```

**Depends on**: T03, T04

---

### T08 — Build coupling layer (F1C — F1B → F1A → radiation → iterate)

**What**: The coupling layer orchestrates F1B and F1A in a loop. Each iteration:
1. F1B predicts wave field η(x,y,t) given incident spectrum
2. F1A extracts local wave conditions at each WEC location
3. F1A predicts device response and absorbed power
4. Radiation field estimated analytically (Phase 1) or via learned operator (Phase 2)
5. Modify incident field and repeat until convergence

**Files to create**:
- **Create** `src/nossomar/coupling/__init__.py`
- **Create** `src/nossomar/coupling/iterative_coupler.py`:
  - `IterativeCoupler(f1b_operator, f1a_operator, radiation_model, config)`
  - `run(wave_spectrum, wec_layout, max_iter, tol)` → `CoupledResult`
  - `CoupledResult`: eta_field, power_per_device, n_iterations, converged
  - Convergence check: `|η_k - η_{k-1}|_L2 < tol`
- **Create** `src/nossomar/coupling/radiation_model.py`:
  - `AnalyticRadiationModel` — Haskind relation for radiation damping (Phase 1)
  - `LearnedRadiationModel` placeholder (Phase 2)
- **Create** `scripts/validate_coupling.py` — runs the 3 benchmarks from
  `phase1_full_f1c.yaml` (single WEC, 2-body, Portuguese coast)
- **Create** `tests/test_coupling.py` — single WEC convergence ≤ 5 iterations,
  energy balance holds (∇·flux + P_absorbed ≈ 0)

**Done when**:
```bash
python scripts/validate_coupling.py --config configs/scenarios/phase1_full_f1c.yaml
# checkpoints/f1c_metrics.json:
#   single_wec_power_error_pct < 10.0
#   two_body_interaction_pct in [5, 15]
#   convergence_iterations_max = 5
pytest tests/test_coupling.py
```

**Depends on**: T04, T06, T07

---

### T09 — Build conformal uncertainty quantification

**What**: Every operator prediction needs a calibrated uncertainty estimate.
Conformal prediction is distribution-free and requires only a held-out
calibration set (already available from the train/val/test split).

**Files to create**:
- **Create** `src/nossomar/uncertainty/__init__.py`
- **Create** `src/nossomar/uncertainty/conformal.py`:
  - `ConformalPredictor(model, calibration_dataset, alpha=0.05)`
  - `calibrate()` — computes nonconformity scores on calibration set
  - `predict_interval(x)` → `(lower, upper, score)` per output channel
  - `coverage_test(test_dataset)` → actual coverage % (must be ≥ 95%)
  - `extrapolation_flag(x, training_bounds)` → bool (params outside training range)
  - `fallback_trigger(x)` → bool (uncertainty > 30% of signal → use BEM solver)
- **Create** `tests/test_uncertainty.py` — coverage ≥ 90% on test set,
  extrapolation flag fires on out-of-range params
- **Create** `scripts/calibrate_uncertainty.py` — loads checkpoint, calibrates
  conformal predictor, saves quantiles to `checkpoints/f1a_conformal_quantiles.json`

**Done when**:
```bash
python scripts/calibrate_uncertainty.py --checkpoint checkpoints/deeponet_wec_f1a1_best.pt
pytest tests/test_uncertainty.py
# checkpoints/f1a_conformal_quantiles.json exists
# coverage ≥ 90% on test set
```

**Depends on**: T04

---

### T10 — Build WEC farm simulation entry point

**What**: This is the deliverable — a single script that takes a farm layout
(WEC positions + wave spectrum) and returns power per device, total farm power,
wave field, and uncertainty estimates. This is the first usable version of
the NOSSO-MAR digital twin.

**Files to create**:
- **Create** `src/nossomar/farm/__init__.py`
- **Create** `src/nossomar/farm/wec_farm.py`:
  - `WECFarm(layout, device_params, f1b_checkpoint, f1a_checkpoint, f1a2_checkpoint)`
  - `simulate(wave_spectrum, n_iter=5)` → `FarmResult`
  - `FarmResult`: power_per_device, total_power, eta_field, uncertainty_per_device,
    converged, n_iterations
  - Internally uses `IterativeCoupler` (T08) + `ConformalPredictor` (T09)
  - Layout: list of (x, y) positions in metres + device params per WEC
- **Create** `scripts/simulate_wec_farm.py` — CLI entry point:
  ```bash
  python scripts/simulate_wec_farm.py \
    --layout configs/farm_layouts/peniche_3wec.yaml \
    --spectrum data/open_database/downloads/ndbc/41009.txt \
    --output results/peniche_farm_result.json
  ```
- **Create** `configs/farm_layouts/peniche_3wec.yaml` — 3-WEC triangle layout,
  Peniche site, using NDBC 41009 as spectral proxy
- **Create** `configs/farm_layouts/north_sea_10wec.yaml` — 10-WEC grid layout
  for Phase 7 MARL baseline
- **Create** `tests/test_wec_farm.py` — 2-WEC layout, verify FarmResult shape,
  total power > 0, uncertainty intervals non-zero, convergence ≤ 5 iterations

**Done when**:
```bash
python scripts/simulate_wec_farm.py \
  --layout configs/farm_layouts/peniche_3wec.yaml \
  --spectrum data/open_database/downloads/ndbc/41009.txt \
  --output results/peniche_farm_result.json
# results/peniche_farm_result.json:
#   power_per_device: [P1, P2, P3]  (all > 0 kW)
#   total_power: sum (> 0 kW)
#   converged: true
#   uncertainty: intervals per device
pytest tests/test_wec_farm.py
```

**Depends on**: T08, T09

---

## Dependency graph

```
T01 (wire PyTorch DeepONet)
 └── T02 (physics loss module)
      └── T03 (Capytaine dataset)
           └── T04 (train F1A — DeepONet)
                ├── T07 (train F1A2 — GNO multi-body)   ← T03 also
                └── T09 (conformal UQ)
                     └─────────────────────────────────────────┐
T05 (wave field dataset)                                        │
 └── T06 (train F1B — FNO2d/WNO)                               │
      └── T08 (coupling layer F1C)  ← T07 also                 │
           └── T10 (WEC farm simulation) ←─────────────────────┘
```

**Critical path**: T01 → T02 → T03 → T04 → T08 → T10
**Parallel**: T05 → T06 can run in parallel with T01 → T04 (different data, different code)
**Can defer**: T09 (UQ) — farm works without it, add after T08

---

## Estimated effort per task

| Task | Core work | Estimated time |
|------|-----------|----------------|
| T01 | Modify training loop | 0.5 day |
| T02 | Write loss module + tests | 1 day |
| T03 | Capytaine runner + dataset | 2 days |
| T04 | Run training (Stages A–D) | 1 day setup + compute time |
| T05 | Wave solver runner + dataset | 3 days (solver setup is the hard part) |
| T06 | F1B training loop + run | 1 day setup + compute time |
| T07 | GNO multi-body + training | 2 days |
| T08 | Coupling layer | 2 days |
| T09 | Conformal UQ | 1 day |
| T10 | Farm simulation + CLI | 1 day |

**Total**: ~14 days of development + compute time (T04, T06)

---

## The one genuine blocker

**T05 requires a phase-resolving wave solver.**

OceanWave3D is commercial. SWASH is open-source but requires setup.
The Path C fallback (synthetic shallow-water data from a finite-difference
integrator) avoids this blocker entirely and is consistent with the physics
already in `residuals_torch.py`.

Recommendation: start Path C immediately so T05 does not block T06.
If OceanWave3D or SWASH becomes available, regenerate the dataset and retrain.
The training pipeline (T06) does not change.

---

## What "WEC farm simulation" looks like at T10 complete

```python
from nossomar.farm.wec_farm import WECFarm

farm = WECFarm(
    layout=[(0, 0), (150, 0), (75, 130)],       # 3 WECs, triangle, metres
    device_params={"radius": 5.0, "draft": 3.5, "depth": 50.0, "bpto": 50.0},
    f1b_checkpoint="checkpoints/fno2d_wave_f1b_best.pt",
    f1a_checkpoint="checkpoints/deeponet_wec_f1a1_best.pt",
    f1a2_checkpoint="checkpoints/gno_wec_f1a2_best.pt",
)

result = farm.simulate(
    wave_spectrum="data/open_database/downloads/ndbc/41009.txt",
)

print(result.power_per_device)   # [P1, P2, P3] in kW
print(result.total_power)        # sum in kW
print(result.converged)          # True if < 5 iterations
print(result.uncertainty)        # conformal intervals per device
```

This is the minimum viable WEC farm simulation the Phase 1 roadmap leads to.
Phase 7 (MARL layout optimisation) starts from this exact interface.
