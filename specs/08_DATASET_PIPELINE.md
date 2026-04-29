# Specification 08: Multi-Fidelity Dataset Pipeline

## Purpose

Turn heterogeneous raw inputs into aligned training assets for NOSSO-MAR.

The dataset pipeline must support:

- open observations;
- spectral products;
- phase-resolved simulations;
- WEC benchmark response data;
- CFD-derived calibration cases.

## Pipeline Stages

### Stage 1 - Source registration

All external data sources should appear in a machine-readable catalogue.

Current implementation:
- `src/nossomar/data/open_data_catalog.py`

### Stage 2 - Lightweight acquisition

The first practical acquisition layer downloads artifacts that are small enough to materialize locally and snapshots metadata pages for larger portals.

Current implementation:
- `src/nossomar/data/open_data_fetchers.py`
- `src/nossomar/data/database_builder.py`

### Stage 3 - Local manifest generation

Every build writes a manifest with:

- build timestamp;
- source parameters;
- materialized artifacts;
- benchmark records;
- failures and missing steps.

Output location:
- `data/open_database/manifest.json`

### Stage 4 - Schema alignment

All later datasets should map into a common project schema:

- grid descriptors
- forcing descriptors
- device descriptors
- fidelity-tagged payloads

Current schema anchor:
- `src/nossomar/core/field_schema.py`

### Stage 5 - Derived products

The pipeline must derive comparison-ready states, not only store raw files.

Required examples:

- spectra from phase-resolved series
- bulk sea-state statistics from spectra
- downgraded phase fields from CFD snapshots
- frequency-response summaries for WEC cases

Current utilities:
- `src/nossomar/physics/multifidelity.py`

## Directory Layout

The repository now uses two complementary data paths.

### A. Analytic local baseline
- `data/phase1_wec_database.json`
- deterministic and lightweight

### B. Open database bootstrap
- `data/open_database/`
- catalogue, manifest, downloads and metadata snapshots

This split is intentional. The first path supports fast local testing; the second supports project-scale data growth.

## Sample Types

### Observation sample
- station id
- timestamp
- observed variables
- QC or source metadata

### Spectral sample
- frequency axis
- spectral density
- bulk wave statistics
- directional context if available

### Phase-resolved sample
- grid definition
- eta field
- velocity field
- boundary and forcing metadata

### WEC sample
- device metadata
- hydrodynamic response curves
- control and PTO context

### CFD-aligned sample
- original high-fidelity field metadata
- downgraded phase-resolved representation
- bridge diagnostics for comparison

## Split Strategy

Do not split randomly across incompatible fidelities.

Recommended split logic:

1. split by scenario family or geography first;
2. keep benchmark campaigns isolated for honest validation;
3. preserve rare high-fidelity cases for validation and calibration;
4. avoid leakage between the same physical case represented at different fidelities.

## Immediate Implementation Status

Already implemented:

- bootstrap open-data builder
- benchmark metadata manifest
- local analytic WEC dataset generation and splitting

Next required implementation:

- xarray-backed loaders for gridded multi-fidelity cases
- parsers for regional bathymetry and EMODnet subsets
- paired solver-to-operator datasets for F1B and coupled scenarios
