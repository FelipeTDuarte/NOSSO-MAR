# NOSSO-MAR Phase 1 — Scope and Baseline

## Scientific Objective

Phase 1 delivers the first end-to-end trainable and evaluable surrogate for wave-structure interaction of a single point-absorber WEC. The operator maps geometry and PTO parameters `[radius, draft, depth, B_pto]` to frequency-domain hydrodynamic coefficients `A(ω)` and `B_rad(ω)` using a DeepONet architecture trained entirely on analytic data.

This is the smallest publishable and testable scientific core that proves the operator-learning strategy on a real wave problem. No expensive solvers are used.

---

## What Is Implemented

| Module | File | Status |
|--------|------|--------|
| IO contracts | `src/nossomar/core/contracts.py` | ✅ done |
| Analytic generator | `src/nossomar/data/analytic_wec.py` | ✅ done |
| Dataset pipeline | `src/nossomar/data/wec_dataset.py` | ✅ done |
| DeepONet surrogate | `src/nossomar/operators/deeponet_wec.py` | ✅ done |
| Training loop | `src/nossomar/training/train_wec.py` | ✅ done |
| Benchmark validation | `src/nossomar/data/public_benchmarks.py` | ✅ done |
| Validation CLI | `scripts/validate_phase1.py` | ✅ done |
| Multi-object FSI stub | `src/nossomar/modules/multi_object_fsi.py` | ✅ contract stub |
| Coupled PINO stub | `src/nossomar/coupling/coupled_pino.py` | ✅ contract stub |

**Test coverage:** 9 tests passing on Python 3.11 / Windows (verified 2026-04-20).

---

## What Is Explicitly Not Implemented Yet

- Real hydrodynamic coupling between multiple WECs (Phase 2)
- Resolution-independent RINO encoder (Phase 2)
- WEC-Sim / Capytaine data integration (Phase 2)
- Trained model checkpoint (requires a training run)
- RMSE validation against a real RM3 benchmark file (requires checkpoint + benchmark data)
- Uncertainty quantification (Phase 2/3)
- Atmospheric forcing, wetting/drying, morphodynamics (Phase 3+)

---

## Install

```bash
git clone https://github.com/FelipeTDuarte/NOSSO-Mar.git
cd NOSSO-Mar
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate
pip install -e .
```

---

## Run Tests

```bash
pytest tests/ -v
```

Expected output: `9 passed` with 1 Zarr metadata warning (harmless).

---

## Run Validation (smoke mode)

```bash
python scripts/validate_phase1.py
```

Runs the analytic surrogate against a synthetic benchmark without any external files. With a trained checkpoint:

```bash
python scripts/validate_phase1.py --checkpoint checkpoints/wec_surrogate.pt
```

---

## Reference Benchmark

The Phase 1 success criterion is relative RMSE < 15% on added mass `A(ω)` against a public WEC-Sim RM3 benchmark case. This has not yet been run because it requires:
1. A trained model checkpoint (run `python -m nossomar.training.train_wec` or `scripts/train_phase1.py`)
2. A benchmark JSON file in the format expected by `load_public_benchmark()`

Once both are available, the verification command is:

```bash
python scripts/validate_phase1.py --checkpoint checkpoints/wec_surrogate.pt --benchmark data/benchmarks/rm3_heave.json
```

---

## Phase 1 Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| All 9 unit tests pass | 100% pass | ✅ verified |
| No NaN in any forward pass | Zero NaN | ✅ verified |
| Loss decreases over 5 training epochs | Yes | ✅ verified |
| Relative RMSE A(ω) vs RM3 benchmark | < 15% | ⏳ pending checkpoint |
| CI passes on every push | GitHub Actions | ✅ configured |

---

## Next: Phase 2 Entry Points

The three files that unlock Phase 2 are:
- `src/nossomar/operators/rino_encoder.py` — resolution-independent encoder
- `src/nossomar/data/wec_sim_adapter.py` — WEC-Sim HDF5 data loader
- `src/nossomar/modules/multi_object_interaction.py` — real pairwise hydrodynamic interaction

These replace the current stubs and are the first three tasks of the Phase 2 subagent plan.
