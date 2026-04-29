# Local Phase 1 Baseline

## Why this exists

The workspace had strong specs but no local executable package. This baseline closes that gap with a small, runnable implementation that follows the Phase 1 order from the notes:

1. Freeze contracts.
2. Generate analytical WEC data.
3. Build a trainable WSI surrogate.
4. Save reproducible artifacts.

## What is now local

- `src/nossomar/core/contracts.py`
  Minimal `WECState`, `WaveField`, and `OceanState` contracts with validation and JSON persistence.
- `src/nossomar/data/analytic_wec.py`
  Closed-form synthetic hydrodynamic responses plus Airy-wave diagnostics.
- `src/nossomar/data/wec_dataset.py`
  Latin-hypercube sampling, train/val/test splitting, and dataset serialization.
- `src/nossomar/operators/deeponet_wec.py`
  A factorized DeepONet-style regressor that keeps branch/trunk structure without needing a heavy ML stack.
- `src/nossomar/training/train_wec.py`
  End-to-end training helper that writes a model file and metric report.

## Deliberate simplifications

This is a local execution baseline, not the final scientific stack.

- Storage is `JSON + YAML` instead of `xarray + Zarr`.
- The operator is `NumPy` ridge-regression on branch/trunk features instead of `PyTorch`.
- The hydrodynamics are smooth, bounded synthetic responses aligned with the spec contracts, not a full Capytaine or Hulme implementation.

These choices keep the workspace runnable on a clean machine while preserving the interfaces needed for the later upgrade path.

## Recommended next moves

1. Replace dataset persistence with `xarray`/`Zarr` once the local code repo is stabilized.
2. Swap the factorized regressor for the real `PyTorch` DeepONet when the environment is ready.
3. Add public benchmark ingestion and a Capytaine adapter behind the same dataset contract.
4. Start the first `F1B` wave-field baseline in parallel, using the same contract style.

**Last updated**: April 28, 2026
