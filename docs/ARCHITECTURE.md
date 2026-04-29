# NOSSO-MAR Architecture

## Mission

NOSSO-MAR is evolving into a multi-fidelity ocean and WEC learning stack, not a single monolithic surrogate.
The long-term target is a physically consistent operator ecosystem that can:

1. ingest open metocean, bathymetry, geology and device data;
2. align observations, spectral models, phase-resolved models and CFD outputs;
3. train different neural-operator families for the sub-problems they are best at;
4. expose a reviewable path from low-cost screening to high-fidelity validation.

The repository now reflects that direction in code, not only in notes.

## Fidelity Ladder

### L0 - Observations
- Station and buoy observations
- Tide gauges and coastal stations
- SST grids and other assimilated products

### L1 - Spectral / reduced-order ocean state
- Bulk sea-state statistics
- 1D or 2D spectra
- Wave-action and operational forecast products

### L2 - Phase-resolved fields
- Surface elevation eta(x, y, t)
- Depth-averaged or resolved velocities
- Nearshore propagation and diffraction fields

### L3 - WEC response and WSI
- Frequency-domain hydrodynamic coefficients
- Device motions, PTO damping, absorbed power
- Array interaction graphs

### L4 - High-fidelity CFD / FSI
- Full Navier-Stokes fields
- Pressure, vorticity, viscous and nonlinear effects
- Highest-cost reference used for calibration and validation, not as the default training target everywhere

The key design choice is that NOSSO-MAR does not compare L4 CFD outputs directly against L1 spectral products without an intermediate bridge. The project now includes explicit downgrade and upgrade utilities in `src/nossomar/physics/multifidelity.py`.

## Architectural Principle

Use the most rigorous physics available for each fidelity, but do not force one equation set to do every job.

- L4 supervision and local residual penalties can use incompressible Navier-Stokes.
- L2 coastal propagation can use shallow-water, mild-slope, wave-action or other reduced equations depending on regime.
- L3 WEC response remains frequency-domain hydrodynamics plus equation-of-motion constraints.
- Cross-fidelity consistency is enforced with bridge operators and derived quantities such as spectra, bulk moments, response summaries and depth-averaged fields.

This is the only realistic way to combine CFD, spectral models and open data without pretending they are directly commensurate.

## Operator Families

The repository now supports a broader operator family instead of a single local surrogate.

### FNO
- Best for regular grids and structured spatiotemporal fields
- Implemented in `src/nossomar/operators/fno/`
- Includes `FNO2d`, `FNO3d`, `FFNO2d` and `GeoFNO2d`

### WNO
- Best for multi-scale spatial structure and localized transformations
- Implemented in `src/nossomar/operators/wno/`
- Useful for shoaling, localized gradients and nonuniform spatial frequency content

### GNO
- Best for irregular geometry, meshes, device arrays and graph coupling
- Implemented in `src/nossomar/operators/gno/`
- Natural candidate for WEC farms and complex coastlines

### DeepONet / PI-DeepONet
- Best for operator queries over frequency, coordinates or parameterized response surfaces
- Implemented in `src/nossomar/operators/deeponet/`
- Strong fit for WEC hydrodynamic response and transfer maps

### RINO-style decoder
- Implemented locally in `src/nossomar/operators/rino.py`
- Used as a lightweight coordinate-conditioned resolution-transfer baseline
- Useful for smoke-testing query-based super-resolution and fidelity bridging

### Factory and sweep
- `src/nossomar/operators/factory.py` provides unified construction
- `src/nossomar/experiments/architecture_sweep.py` smoke-tests FNO, WNO, GNO, DeepONet and RINO in one place

## Physics Stack

### Residual layer

`src/nossomar/physics/residuals_torch.py` now exposes:

- 2D Navier-Stokes residuals
- 3D Navier-Stokes residuals
- shallow-water residuals
- reduced wave-action balance residuals
- Exner morphodynamic residuals
- WEC frequency-domain equation-of-motion residuals

These residuals are the current mechanism for "full-physics where possible" without claiming that the repository already contains a production CFD solver.

### Multi-fidelity bridge layer

`src/nossomar/physics/multifidelity.py` now exposes:

- spectrum moments and bulk wave statistics
- phase-series to spectrum conversion
- irregular wave reconstruction from spectra
- CFD snapshot downgrade to phase-resolved fields
- hydrodynamic frequency-response summaries

This bridge layer is essential because CFD outputs, spectral models and WEC response data live in different state spaces.

## Data Backbone

The data layer is now split into two complementary modes.

### Local analytic baseline
- Existing Phase 1 baseline remains available
- Useful for deterministic tests, training-loop checks and interface hardening

### Open-data bootstrap database
- `src/nossomar/data/open_data_catalog.py` curates accessible sources
- `src/nossomar/data/open_data_fetchers.py` downloads lightweight artifacts
- `src/nossomar/data/database_builder.py` materializes a local manifest under `data/open_database`

Current catalogue categories:

- waves and wind observations
- tides and coastal stations
- sea-surface temperature
- bathymetry
- geology and granulometry proxies
- European physics portals
- WEC benchmark repositories
- wind-resource atlases

## Database Philosophy

The repository does not yet pretend to have the final complete global database.
Instead it now has a reproducible bootstrap path:

1. materialize small, direct-download artifacts immediately;
2. capture landing-page metadata for larger catalogues;
3. keep benchmark metadata and source URLs in machine-readable form;
4. expand each source into richer regional subsets later.

This keeps the project executable now while preserving a clean path to scale.

## Current Module Map

### Core
- `src/nossomar/core/contracts.py`
- `src/nossomar/core/field_schema.py`

### Data
- `src/nossomar/data/analytic_wec.py`
- `src/nossomar/data/wec_dataset.py`
- `src/nossomar/data/open_data_catalog.py`
- `src/nossomar/data/open_data_fetchers.py`
- `src/nossomar/data/database_builder.py`

### Operators
- `src/nossomar/operators/base.py`
- `src/nossomar/operators/factory.py`
- `src/nossomar/operators/rino.py`
- `src/nossomar/operators/fno/`
- `src/nossomar/operators/wno/`
- `src/nossomar/operators/gno/`
- `src/nossomar/operators/deeponet/`

### Experiments
- `src/nossomar/experiments/architecture_sweep.py`

### Physics
- `src/nossomar/physics/multifidelity.py`
- `src/nossomar/physics/residuals_torch.py`

## What Is Real Today

Implemented and tested locally:

- operator family imports and forward-pass sweep;
- multi-fidelity helper functions;
- physics residual modules;
- curated open-data catalogue;
- bootstrap database builder;
- baseline analytic WEC dataset and training path.

Not yet implemented as a full production stack:

- full CFD generation pipelines;
- Portuguese regional data subsetting at scale;
- joint end-to-end training over all fidelities and all scenarios;
- production data assimilation and control loops.

Those remain roadmap items, but the architecture now matches that reality instead of hiding it.

## Design Decision Summary

1. Do not force a single operator family on every sub-problem.
2. Do not compare CFD and spectral outputs without explicit state translation.
3. Use full Navier-Stokes where it helps supervision or residual regularization, not where it destroys tractability.
4. Keep open data, benchmark metadata and physics constraints first-class citizens in the codebase.
5. Make every new spec point to modules that already exist or have a clear implementation slot.
