"""Data access, dataset construction and open-data utilities."""

from .analytic_wec import (
    airy_wave_diagnostics,
    analytic_hydrodynamic_response,
    generate_analytic_wec_case,
)
from .capytaine_runner import CapytaineRunner
from .database_builder import build_open_database
from .open_data_catalog import OPEN_DATA_CATALOG, OpenDataSource, catalog_records
from .wec_dataset import (
    LHSConfig,
    WECDatabase,
    WECDataset,
    build_wec_database,
    latin_hypercube_sample,
    load_zarr_payload,
    split_database,
    write_dataset_zarr,
)

__all__ = [
    "CapytaineRunner",
    "LHSConfig",
    "OPEN_DATA_CATALOG",
    "OpenDataSource",
    "WECDatabase",
    "WECDataset",
    "airy_wave_diagnostics",
    "analytic_hydrodynamic_response",
    "build_open_database",
    "build_wec_database",
    "catalog_records",
    "generate_analytic_wec_case",
    "latin_hypercube_sample",
    "load_zarr_payload",
    "split_database",
    "write_dataset_zarr",
]
