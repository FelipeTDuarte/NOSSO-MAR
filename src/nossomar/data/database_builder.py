"""Bootstrap an open-data database for multi-fidelity NOSSO-MAR experiments."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
import json

from .open_data_catalog import OPEN_DATA_CATALOG, catalog_records, write_catalog
from .open_data_fetchers import (
    DownloadedArtifact,
    fetch_coops_product,
    fetch_ndbc_realtime,
    fetch_oisst_daily_file,
    fetch_remote_metadata_snapshot,
)


WEC_BENCHMARKS: list[dict[str, Any]] = [
    {
        "benchmark_id": "rm3",
        "name": "Reference Model 3",
        "device_class": "two-body point absorber",
        "source_urls": [
            "https://data.openei.org/submissions/8009",
            "https://wec-sim.github.io/WEC-Sim/main/user/tutorials.html",
        ],
        "known_characteristics": {
            "float_mass_tonnes": 727.01,
            "plate_mass_tonnes": 878.30,
            "float_center_of_gravity_z_m": -0.72,
            "plate_center_of_gravity_z_m": -21.29,
            "heave_travel_m": 4.0,
            "reaction_plate_bottom_depth_m": 35.0,
            "deployment_depth_range_m": [40.0, 100.0],
        },
        "available_assets": ["geometry", "mass_properties", "tutorial_hydrodata"],
        "status": "verified_open_source",
    },
    {
        "benchmark_id": "mbari_wec_2021",
        "name": "MBARI WEC 2021 deployment",
        "device_class": "small two-body point absorber",
        "source_urls": ["https://mhkdr.openei.org/submissions/381"],
        "known_characteristics": {
            "paired_environmental_data": ["wave", "wind", "temperature"],
            "paired_response_data": ["wec_hourly_production"],
        },
        "available_assets": ["mat_file", "spotter_csv", "journal_reference"],
        "status": "verified_open_source",
    },
    {
        "benchmark_id": "aquaharmonics_1_20",
        "name": "AquaHarmonics WEC 1:20 scale tank test",
        "device_class": "point absorber",
        "source_urls": ["https://data.openei.org/submissions/7966"],
        "known_characteristics": {"scale": "1:20", "campaign_year": 2018},
        "available_assets": ["lab_data", "control_context"],
        "status": "verified_open_source",
    },
    {
        "benchmark_id": "lupa_wecsim_moordyn",
        "name": "Laboratory Upgrade Point Absorber",
        "device_class": "point absorber",
        "source_urls": [
            "https://mhkdr.openei.org/submissions/572",
            "https://mhkdr.openei.org/submissions/577",
        ],
        "known_characteristics": {"includes_moorings": True, "open_source_model": True},
        "available_assets": ["wec_sim_model", "moordyn_model", "cad"],
        "status": "verified_open_source",
    },
]


def _artifact_to_dict(artifact: DownloadedArtifact) -> dict[str, Any]:
    return artifact.to_dict()


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def build_open_database(
    root_dir: str | Path,
    *,
    ndbc_station: str = "41009",
    coops_station: str = "9414290",
    coops_days: int = 2,
    oisst_date: date | None = None,
    include_metadata_snapshots: bool = True,
) -> dict[str, Any]:
    """Build a small but concrete open-data database manifest on disk."""

    root = Path(root_dir)
    downloads_dir = root / "downloads"
    catalog_dir = root / "catalog"
    metadata_dir = root / "metadata"
    benchmark_dir = root / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)

    write_catalog(catalog_dir / "open_data_catalog.json")
    _write_json(benchmark_dir / "wec_benchmarks.json", WEC_BENCHMARKS)

    artifacts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def attempt(label: str, func, *args, **kwargs) -> None:
        try:
            artifact = func(*args, **kwargs)
            artifacts.append(_artifact_to_dict(artifact))
        except Exception as exc:  # pragma: no cover - network failures are environment-specific
            errors.append({"step": label, "error": str(exc)})

    attempt("ndbc_realtime", fetch_ndbc_realtime, downloads_dir, ndbc_station)
    for product in ("water_level", "wind", "water_temperature"):
        attempt(
            f"coops_{product}",
            fetch_coops_product,
            downloads_dir,
            station_id=coops_station,
            product=product,
            days=coops_days,
        )
    attempt("oisst_daily", fetch_oisst_daily_file, downloads_dir, target_date=oisst_date)

    if include_metadata_snapshots:
        for entry in OPEN_DATA_CATALOG:
            if entry.access_mode == "direct_download" and entry.source_id not in {
                "rm3_geometry",
                "mbari_wec_2021",
                "aquaharmonics_tank_test",
                "lupa_wecsim_moordyn",
                "wecsim_rm3_tutorial",
                "newa_atlas",
                "emodnet_physics",
                "emodnet_geology",
                "gebco_grid",
            }:
                continue
            attempt(
                f"snapshot_{entry.source_id}",
                fetch_remote_metadata_snapshot,
                metadata_dir,
                entry.source_id,
                entry.access_url,
            )

    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "root_dir": str(root.resolve()),
        "catalog_entries": catalog_records(),
        "benchmarks": WEC_BENCHMARKS,
        "artifacts": artifacts,
        "errors": errors,
        "parameters": {
            "ndbc_station": ndbc_station,
            "coops_station": coops_station,
            "coops_days": coops_days,
            "oisst_date": None if oisst_date is None else oisst_date.isoformat(),
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest
