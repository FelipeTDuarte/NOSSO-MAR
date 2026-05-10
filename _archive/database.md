# NOSSO-MAR Open Database

## Purpose

This document defines the practical meaning of "database" in the current repository state.
It is a reproducible local data lake bootstrap, not yet the final full production corpus.

## Current Build Path

The database is built with:

```bash
python scripts/build_open_database.py
```

Configuration lives in `configs/open_data.yaml`.

The builder writes to `data/open_database/` and produces:

- `manifest.json`
- `catalog/open_data_catalog.json`
- `benchmarks/wec_benchmarks.json`
- `downloads/ndbc/...`
- `downloads/coops/...`
- `downloads/oisst/...`
- `metadata/*.html`

## Data Classes

### Direct downloads
- NDBC realtime buoy text feeds
- NOAA CO-OPS JSON products
- OISST daily NetCDF files

### Metadata snapshots
- GEBCO landing pages
- EMODnet Physics landing pages
- EMODnet Geology landing pages
- NEWA landing pages
- WEC benchmark landing pages

### Benchmark manifests
- RM3
- MBARI WEC
- AquaHarmonics
- LUPA

## Why this structure

Many of the most useful marine datasets are too large, too regionalized or too portal-driven to mirror fully on day one.
The current approach separates:

1. immediately downloadable artifacts;
2. high-value source metadata;
3. curated machine-readable benchmark descriptors.

That keeps the repository executable now and expandable later.

## Planned expansion

The next database increments should add:

- regional bathymetry subsets for target Portuguese domains;
- EMODnet and other European in situ extracts for wave, wind, currents and sea level;
- seabed substrate and granulometry subsets;
- WEC geometry, inertia and hydrodynamic coefficient packages where licensing and format allow;
- solver-generated L2 and L4 paired samples for multi-fidelity alignment.

## Important limitation

The presence of a source in the catalogue does not yet mean the full dataset has been ingested locally.
The manifest distinguishes between concrete local artifacts and catalogued future sources so the project can scale without losing provenance.
