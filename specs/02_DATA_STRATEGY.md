# Specification 02: Multi-Fidelity Data Strategy

## Purpose

Define how NOSSO-MAR acquires, organizes and validates the data needed for a physically informed ocean-and-WEC operator stack.

The project now assumes from the start that no single source is sufficient:

- observations are sparse but real;
- spectral models are broad and cheap;
- phase-resolved models capture local structure;
- CFD is the most rigorous but also the most expensive;
- WEC benchmarks provide device truth that generic ocean products do not contain.

## Data Tiers

### Tier A - Open observations and reference products

Used for forcing, calibration and external validation.

- NDBC buoy records for waves, wind and metocean context
- NOAA CO-OPS products for water level, tides and station meteorology
- OISST for daily sea-surface temperature
- GEBCO for coarse bathymetry
- EMODnet Physics for European wave, current, sea-level and wind access
- EMODnet Geology for seabed substrate, morphology and geological context
- NEWA for offshore wind resource context

### Tier B - Open WEC benchmark repositories

Used for device metadata, validation scenarios and control-aware response studies.

- RM3 geometry and tutorial data
- MBARI WEC deployment data
- AquaHarmonics tank-test data
- LUPA WEC-Sim and MoorDyn models

### Tier C - Low-cost synthetic and analytic generators

Used for interface hardening, regression tests and local development speed.

- analytic WEC response generator in `src/nossomar/data/analytic_wec.py`
- local Latin-hypercube dataset pipeline in `src/nossomar/data/wec_dataset.py`

### Tier D - Numerical solvers and CFD

Used for the highest-fidelity supervision and for bridge calibration.

- spectral and nearshore models
- phase-resolved wave solvers
- CFD / FSI solvers for selected cases

These are not yet fully automated in the repository, but the architecture now reserves them explicitly.

## Required Variable Families

The target database should eventually cover the following fields, even when early local builds only materialize a subset.

### Ocean forcing
- significant wave height
- wave period and directional content
- wind speed and wind direction
- currents
- tides and sea level
- sea-surface temperature

### Geospatial seabed context
- bathymetry
- seabed morphology
- geology
- substrate class
- granulometry or compatible proxies

### Device and WSI context
- device dimensions
- mass and inertia
- center of gravity
- hydrodynamic added mass and damping
- excitation forces
- PTO damping and control metadata
- mooring metadata where available

## Data Model

NOSSO-MAR now uses a multi-fidelity sample concept defined in `src/nossomar/core/field_schema.py`.

Each sample can carry aligned information for:

- `l0_observations`
- `l1_spectral`
- `l2_phase_resolved`
- `l3_wec_response`
- `l4_cfd`

This avoids forcing all cases into the same state representation.

## Open Data Implementation

The current implementation is intentionally pragmatic.

### Catalogue
- `src/nossomar/data/open_data_catalog.py`
- machine-readable source list with categories, variables and URLs

### Fetchers
- `src/nossomar/data/open_data_fetchers.py`
- direct download of small artifacts and metadata snapshots

### Builder
- `src/nossomar/data/database_builder.py`
- creates `data/open_database/manifest.json`

This is the minimum viable reproducible database layer.

## Fidelity Alignment Rule

The strategy explicitly forbids naive comparison between tiers that live in different physics spaces.

Examples:

- CFD velocity fields must be downgraded before comparing to depth-averaged or spectral products.
- Spectra must be reconstructed or summarized before comparing to phase-resolved time series.
- WEC response curves must be compared through shared frequency-domain descriptors or coupled simulation contexts.

Bridge utilities are implemented in `src/nossomar/physics/multifidelity.py`.

## Quality Gates

Every ingested or generated sample should pass some subset of the following:

- provenance known
- units known
- coordinates known
- variable names mapped into project schema
- physical sign conventions checked
- frequency or time axes monotonic
- metadata sufficient to reproduce the case

For WEC response data, additional checks include:

- damping non-negativity
- added-mass sanity bounds
- complex excitation consistency

## Current Status

Implemented now:

- open-source catalogue
- bootstrap downloader
- benchmark manifest
- analytic local database path

Planned next:

- regional subsetting for Portuguese waters
- richer EMODnet extraction adapters
- paired solver-generated L2/L4 cases
- normalized xarray-based storage for larger campaigns
