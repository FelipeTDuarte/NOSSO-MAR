# Specification 10: Multi-Fidelity Alignment

## Purpose

Define the bridge between incompatible state spaces:

- observations
- spectral products
- phase-resolved fields
- WEC response operators
- CFD / FSI fields

Without this layer, NOSSO-MAR would compare quantities that do not live in the same representation.

## Alignment Rules

### Rule 1
Never compare raw CFD state tensors directly to spectral products.

### Rule 2
Always compare through a shared representation:
- downgraded field
- reconstructed series
- bulk statistics
- frequency-response summaries

### Rule 3
Keep the downgrade and upgrade operations explicit and testable.

## Required Bridges

### CFD -> phase-resolved

Examples:
- depth-average velocities
- infer eta from free surface or hydrostatic pressure proxy

Current implementation:
- `cfd_snapshot_to_phase_fields` in `src/nossomar/physics/multifidelity.py`

### Phase-resolved -> spectral

Examples:
- eta(t) to one-sided spectrum
- spectral moments and bulk statistics

Current implementation:
- `phase_series_to_spectrum`
- `spectral_moments`
- `bulk_wave_statistics`

### Spectral -> phase-resolved

Examples:
- reconstruct irregular wave realizations from a target spectrum

Current implementation:
- `reconstruct_irregular_wave`

### WEC response summarization

Examples:
- summarize added mass, damping and excitation curves
- compare response peaks instead of raw mismatched curves when needed

Current implementation:
- `summarize_frequency_response`

## Physics Alignment

The bridge layer must work with the residual layer, not separately.

For example:

- a downgraded CFD field can be checked with shallow-water residuals;
- a reconstructed irregular wave can be checked against spectral moments;
- a predicted WEC response can be checked with the frequency-domain equation of motion.

Current residual entry points:
- `navier_stokes_2d_residual`
- `navier_stokes_3d_residual`
- `shallow_water_residual`
- `wave_action_balance_residual`
- `exner_residual`
- `wec_frequency_domain_residual`

## Training Use

Typical bridge-aware loss combinations:

1. supervised L2 field loss + spectral consistency loss;
2. CFD downgrade loss + shallow-water residual loss;
3. WEC response loss + equation-of-motion residual loss.

## Current Scope

Implemented:
- core bridge utilities
- residual modules
- tests covering spectrum reconstruction and WEC balance consistency

Still needed:
- richer xarray-native bridge datasets;
- domain-aware bathymetry and morphology transforms;
- automated pairing of solver outputs across fidelities.
