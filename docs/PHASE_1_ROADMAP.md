# Phase 1 Implementation Roadmap

## Overview

Phase 1 implements the first complete cycle: **Neural Operator replacement for phase-resolved wave propagation (F1B) and wave-structure interaction (F1A)** with optional bidirectional coupling (F1C).

**Timeline**: 4–6 weeks
**Resources**: 1 lead developer + 1 ML engineer + 1 data engineer
**Deliverables**: 7 completed tasks, all tests passing, paper-ready code

---

## Task-by-Task Checklist

### Task 0: IO Contracts & Types (Phase 0) ✓ COMPLETE

**Status**: ✓ Implemented
**Completion Date**: End of Phase 0

**Artifacts**:
- `src/nossomar/core/contracts.py` — WECState, WaveField, OceanState classes
- `tests/test_contracts.py` — 10 test cases (all passing)

**Success Criteria**:
- ✓ All xarray classes serialize/deserialize to NetCDF
- ✓ Coordinate validation passes
- ✓ Physical bounds checking works

---

### Task 1: Analytical Data Generation (Nível 0) ✓ COMPLETE

**Status**: ✓ Implemented
**Completion Date**: Phase 1, Week 1

**Artifacts**:
- `src/nossomar/data/analytic_wec.py` — Airy waves, cylinder hydrodynamics (Hulme 1982)
- `tests/test_analytic_wec.py` — 5 canonical cases, <1 second runtime

**Success Criteria**:
- ✓ 1000+ analytical WEC cases generated in <1 second
- ✓ RMSE vs. literature values < 1%
- ✓ Validation: depth range [10, 1000] m, frequency [0.05, 2.5] Hz

---

### Task 2: Dataset Pipeline & Capytaine Integration → IN PROGRESS

**Status**: ✓ Mostly complete (GitHub repo has working pipeline)
**Target Date**: Phase 1, Week 2

**Artifacts**:
- `src/nossomar/data/capytaine_runner.py` — BEM sweep with LHS sampling
- `src/nossomar/data/wec_dataset.py` — xarray → Zarr serialization + PyTorch Dataset API
- `data/phase1_wec_database.zarr/` — 500 WEC cases train/val/test split
- `tests/test_wec_dataset.py` — Integration tests

**Success Criteria**:
- [ ] 500 unique WEC cases generated (LHS parameter sweep)
- [ ] Capytaine validation: 10% cases checked vs. HAMS (if available)
- [ ] Zarr dataset: 300 MB, readable in parallel (10 workers at >100 samples/s)
- [ ] PyTorch DataLoader works: batch_size=32, shuffle=True
- [ ] All metadata complete (solver ref, timestamps, data versions)

**Blockers**: None currently

---

### Task 3: DeepONet WSI Operator (F1A1 — Single WEC) ✓ COMPLETE

**Status**: ✓ Implemented & validated
**Completion Date**: Phase 1, Week 2

**Artifacts**:
- `src/nossomar/operators/deeponet_wec.py` — DeepONet architecture (branch + trunk)
- `tests/test_deeponet_wec.py` — Shape tests, forward pass, benchmark (RM3)
- `checkpoints/deeponet_wec_best.pt` — Trained model

**Success Criteria**:
- ✓ RMSE_A < 5%, RMSE_B < 5%, RMSE_Fex < 8%
- ✓ Test on RM3 WEC: power prediction within 10% of WEC-Sim
- ✓ Physics: zero cases with B < 0 (damping always ≥ 0)
- ✓ Inference: < 1 ms per query (real-time capable)

---

### Task 4: Training Pipeline & Loss Functions ✓ COMPLETE

**Status**: ✓ Implemented
**Completion Date**: Phase 1, Week 3

**Artifacts**:
- `src/nossomar/training/train_wec.py` — Training loop (supervised + physics loss)
- `src/nossomar/loss/physics_losses.py` — L_supervised, L_physics, L_total
- `configs/training.yaml` — Hyperparams, scheduler, early stopping
- `scripts/train_phase1.py` — Entry point (CLI)
- `runs/tensorboard/` — Training curves, validation metrics

**Success Criteria**:
- ✓ F1A training converges: val RMSE < 5% after 2 hours
- ✓ Generalization gap < 20% (train vs. val loss ratio)
- ✓ Curriculum learning: physics loss ramps up epochs 50–100%
- ✓ Checkpointing: best model saved, loadable for inference
- ✓ Reproducibility: fixed seed, same results ±0.1%

**Current Status**: All training scripts working; F1A reaches target RMSE

---

### Task 5: Wave Field Operator (F1B — Phase-Resolved Propagation) → IN PROGRESS

**Status**: → Design & architecture selection phase
**Target Date**: Phase 1, Week 3–4

**Artifacts**:
- `src/nossomar/operators/gino_wave.py` (or fno_wave.py) — Neural operator
- `tests/test_wave_operator.py` — Domain validation, energy balance
- `scripts/generate_f1b_training_data.py` — OceanWave3D/SWASH wrapper
- `configs/f1b_gino.yaml` — Architecture & training hyperparams
- `data/phase1b_wave_dataset.zarr/` — 500 training cases

**Success Criteria**:
- [ ] Architecture selected (GINO recommended for irregular bathymetry)
- [ ] Training data: 500 shallow water cases generated (OceanWave3D / SWASH)
- [ ] Operator trained: η RMSE < 10 cm (absolute) on test set
- [ ] Validation: energy spectrum agreement within 15%
- [ ] Physics: causality (no energy before forcing), dispersion relation ±5%

**Blockers**: Need OceanWave3D or SWASH setup for data generation (in progress)

---

### Task 6: Multi-Body WSI & Coupling (F1A2 + F1C) → OPTIONAL, Phase 1 Late

**Status**: → Optional for Phase 1 (can defer to Phase 2)
**Target Date**: Phase 1, Week 4+ (if time)

**Artifacts**:
- `src/nossomar/modules/multi_object_fsi.py` — Multi-WEC interaction
- `src/nossomar/operators/gno_wec.py` (optional) — Graph Neural Operator
- `src/nossomar/coupling/coupled_pino.py` — Iterative or end-to-end coupling
- `tests/test_coupled_operator.py` — 2-body array validation
- `scripts/validate_coupling.py` — Benchmark against SWAN+WEC-Sim

**Success Criteria**:
- [ ] GNO trained on 2-body array (100 training cases)
- [ ] Interaction kernel captured: power reduction in array 5–15%
- [ ] Iterative coupling prototype: F1B → F1A → radiation → iterate
- [ ] Convergence: <5 iterations typical for phase-averaged coupling

**Blockers**: Defer if time is tight; F1A + F1B are higher priority

---

### Task 7: Uncertainty Quantification & Validation → OPTIONAL, Phase 1 Final

**Status**: → Final integration week
**Target Date**: Phase 1, Week 4 (end)

**Artifacts**:
- `src/nossomar/uncertainty/conformal.py` — Conformal prediction
- `src/nossomar/uncertainty/ensemble.py` — 5-member ensemble (optional)
- `tests/test_uncertainty.py` — Calibration metrics (coverage, interval width)
- `scripts/evaluate_uncertainty.py` — Generate uncertainty report

**Success Criteria**:
- [ ] Conformal intervals: >90% coverage on test set
- [ ] Interval width < 15% of signal amplitude
- [ ] Extrapolation flags: catch out-of-range parameter combinations
- [ ] Operational readiness: uncertainty-aware solver fallback logic defined

**Blockers**: None (last task, highest flexibility)

---

## Dependencies & Sequencing

```
Task 0 (Contracts)
  ↓
├─→ Task 1 (Analytical Data)
│     ↓
├─→ Task 2 (Data Pipeline + Capytaine)
│     ↓
├─→ Task 3 (F1A DeepONet)
│     ↓
├─→ Task 4 (Training Loop)
│     ↓ (F1A training executes)
│     ↓
├─→ Task 5 (F1B Wave Operator)       [parallel with Task 4]
│     ↓
└─→ Task 6 (Multi-body + Coupling)   [optional, depends on 3 + 5]
     ↓
     └─→ Task 7 (Uncertainty)        [optional, depends on all]
```

**Critical Path**: Tasks 0 → 1 → 2 → 3 → 4 (F1A complete)
**Parallel**: Task 5 (F1B) can start during Task 4 (training loop development)
**Optional**: Tasks 6–7 deferred to Phase 2 if timeline tight

---

## Weekly Breakdown (Ideal Timeline)

### Week 1
- ✓ Task 0 complete (IO contracts)
- ✓ Task 1 complete (analytical data)
- → Task 2 start (data pipeline design + Capytaine setup)

### Week 2
- ✓ Task 2 complete (dataset ready, 500 cases)
- ✓ Task 3 complete (F1A operator trained)
- → Task 4 start (formalize training loop, validation metrics)

### Week 3
- ✓ Task 4 complete (F1A converged, paper-ready)
- → Task 5 start (F1B operator architecture + data generation)
- ← Task 5 parallel data generation (OceanWave3D/SWASH)

### Week 4
- → Task 5 training (F1B operator)
- → Task 6 optional (multi-body, coupling prototype)
- → Task 7 optional (uncertainty integration)

### Week 5–6 (Buffer)
- Final validation on test set
- Paper writing + code cleanup
- Documentation updates

---

## Success Metrics (Phase 1 Complete)

### Code Quality
- ✓ All 9 tests pass (pytest)
- ✓ No physics violations (B ≥ 0, energy balance checked)
- ✓ Code coverage > 80%
- ✓ Reproducible (fixed seed, ±0.1% variance)

### Operator Accuracy
- ✓ F1A RMSE < 5% (hydrodynamic coefficients)
- ✓ F1B RMSE < 10% (wave field elevation)
- ✓ RM3 benchmark: power within 10% of WEC-Sim
- ✓ Shallow water benchmark: energy balance within 5%

### Documentation
- ✓ 9 specification files complete
- ✓ Architecture.md coherent and linked
- ✓ Each operator has usage example in docstring
- ✓ Paper draft with results, plots, discussion

### Operational Readiness
- ✓ Uncertainty estimates provided (conformal intervals)
- ✓ Extrapolation flags working
- ✓ Solver fallback logic sketched
- ✓ Digital twin compatible (future assimilation path clear)

---

## Known Risks & Mitigations

| Risk | Mitigation | Owner |
|------|-----------|-------|
| Data scarcity (limited Capytaine budget) | Use analytical + public data heavily; LHS sampling efficient | Data engineer |
| F1B generalization to new domains | Train on multiple bathymetries; transfer learning if needed | Researcher |
| Physics loss destabilizes training | Start without physics loss; add gradually (curriculum) | ML engineer |
| Overfitting on small dataset | Ensemble, dropout, early stopping, data augmentation | ML engineer |
| Operational integration mismatch | Iterate with digital twin team early (weekly sync) | Lead |

---

## Handoff to Phase 2

**At Phase 1 completion, Phase 2 receives**:
- ✓ Fully trained F1A and F1B operators
- ✓ All 9 spec files with implementation status
- ✓ GitHub repository with 7 merged tasks, all tests passing
- ✓ Uncertainty estimates + operational flags
- ✓ Benchmark validation (RM3, shallow water, real domain)
- → Ready for: digital twin integration, layout optimization, data assimilation

---

## How to Update This Roadmap

- **Weekly**: Update task status (✓ complete, → in progress, ⏸️ blocked)
- **After milestone**: Document completion date + artifacts
- **If blocked**: Note blocker in task details
- **Links**: Keep specs linked (update if spec moves/renames)

**Last updated**: [To be filled in by project coordinator]
