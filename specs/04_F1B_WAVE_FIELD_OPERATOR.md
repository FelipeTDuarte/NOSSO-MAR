# Specification 04: F1B Wave-Field Operator

## Purpose

Learn operators that map forcing, bathymetry and boundary conditions to wave fields while staying physically aligned across multiple fidelity levels.

This module should not be framed as "one model replaces all solvers".
Instead, it should provide the wave-field branch of the fidelity ladder:

- L1 spectral guidance
- L2 phase-resolved field prediction
- L4 CFD-consistent calibration targets on selected cases

## Governing Physics

Different regimes require different equation sets.

### Highest-fidelity reference
- incompressible Navier-Stokes
- pressure and velocity fields
- free-surface and FSI effects where available

### Training and operational reduced physics
- shallow-water equations
- wave-action balance
- mild-slope style propagation constraints
- boundary and energy-consistency terms

NOSSO-MAR should therefore use Navier-Stokes-informed supervision where possible, but not insist on pure full-CFD learning for every scenario.

## Input / Output Contract

### Inputs
- bathymetry or seabed topography
- coast / mask geometry
- incident wave descriptors
- wind and current context when available
- optional structures or WEC array descriptors

### Outputs
- eta(x, y, t) or eta(x, y) depending on task
- velocity surrogates at the required fidelity
- optional reduced summaries: spectra, fluxes, energy densities

## Operator Matrix

### FNO

Use when:
- grid is regular
- domain is rectangular or can be regularized
- inference speed matters

Strength:
- strong baseline for structured spatiotemporal fields

### WNO

Use when:
- spatial scales are strongly mixed
- localized transformations matter
- breaking or shoreline-adjacent structure must be represented more locally

Strength:
- multi-resolution spatial handling

### GNO

Use when:
- geometry is irregular
- coastline and obstacles matter
- device arrays are embedded in the field

Strength:
- natural representation for irregular meshes and coupled object interactions

### RINO-style decoder

Use when:
- resolution transfer is central
- one wants query-based decoding over arbitrary target points

Strength:
- helpful bridge baseline between coarse and fine grids

## Current Local Implementation

The repository now includes the operator families required for this spec:

- `src/nossomar/operators/fno/`
- `src/nossomar/operators/wno/`
- `src/nossomar/operators/gno/`
- `src/nossomar/operators/rino.py`
- `src/nossomar/operators/factory.py`

The comparison entry point is `src/nossomar/experiments/architecture_sweep.py`.

## Loss Design

The F1B operator should use a stacked loss, not a single MSE.

### Supervised field loss
- eta mismatch
- velocity mismatch where available

### Physics residual loss
- shallow-water residuals
- wave-action residuals
- optionally Navier-Stokes residuals on selected collocation slices

### Boundary / energy loss
- open-boundary consistency
- wall or coastline constraints
- energy or wave-action balance checks

### Cross-fidelity bridge loss
- L2 field to L1 spectrum consistency
- L4 CFD downgrade to L2 field consistency

## CFD vs Spectral Comparison Rule

CFD outputs must be downgraded before comparison.

Required bridge examples:

1. CFD velocities -> depth-averaged velocities
2. CFD free-surface series -> spectra and bulk statistics
3. phase-resolved eta(t) -> spectral moments
4. spectral product -> reconstructed irregular wave realization

These operations are now first-class utilities in `src/nossomar/physics/multifidelity.py`.

## Validation Targets

The module should be validated on at least three views:

### Field view
- eta RMSE
- velocity RMSE

### Spectral view
- Hs
- Tp or Tm01
- low-order spectral moments

### Physics view
- residual magnitude
- energy drift
- boundary-condition violation rate

## Near-Term Plan

1. keep the operator sweep lightweight and tested locally;
2. add xarray-backed field datasets for real regional cases;
3. train FNO/WNO/GNO variants on aligned subsets;
4. reserve CFD-heavy cases for sparse but high-value bridge calibration.
