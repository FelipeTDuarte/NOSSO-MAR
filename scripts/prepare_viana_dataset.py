"""Prepare a Viana do Castelo dataset and open-database subset.

Creates:
- `configs/viana_dataset.yaml` (if missing)
- `configs/viana_50_layout.json` layout for 50 WECs ~1 km offshore
- `data/open_database/viana/manifest.json` via database_builder
- `data/datasets/viana_50.json` analytic WEC dataset with per-device locations

This is a lightweight, reproducible preparer that uses the local analytic
generator as a stand-in until real metocean data are ingested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nossomar.data.database_builder import build_open_database
from nossomar.data.wec_dataset import build_wec_database, write_dataset_json
from nossomar.data.analytic_wec import make_frequency_grid


def meters_to_deg(lat: float, lon: float, d_n: float = 0.0, d_e: float = 0.0) -> tuple[float, float]:
    """Convert meter offsets north/east to (lat, lon) degrees around (`lat`, `lon`)."""
    # Approx conversions
    lat_deg = d_n / 111_000.0
    lon_deg = d_e / (111_000.0 * max(1e-6, np.cos(np.deg2rad(lat))))
    return lat + lat_deg, lon + lon_deg


def generate_viana_layout(center_lat: float, center_lon: float, n_devices: int = 50) -> list[dict[str, float]]:
    """Generate a simple rectangular farm offset ~1 km offshore from center point.

    The layout is a grid of devices with small spacing; first the whole farm
    is translated ~1000 m offshore (west) from the supplied center coordinate.
    """
    # move ~1000 m west (negative east) to put farm ~1 km offshore
    offshore_east = -1000.0
    offshore_north = 0.0
    center_lat_off, center_lon_off = meters_to_deg(center_lat, center_lon, offshore_north, offshore_east)

    # grid dimensions (try to make rows x cols close to n_devices)
    cols = int(np.ceil(np.sqrt(n_devices)))
    rows = int(np.ceil(n_devices / cols))
    spacing_m = 200.0  # 200 m spacing

    coords: list[dict[str, float]] = []
    start_x = -0.5 * (cols - 1) * spacing_m
    start_y = -0.5 * (rows - 1) * spacing_m
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_devices:
                break
            x = start_x + c * spacing_m
            y = start_y + r * spacing_m
            lat = center_lat_off + (y / 111_000.0)
            lon = center_lon_off + (x / (111_000.0 * max(1e-6, np.cos(np.deg2rad(center_lat_off)))))
            coords.append({"lat": float(lat), "lon": float(lon)})
            idx += 1
    return coords


def ensure_viana_config(path: Path | str) -> None:
    p = Path(path)
    if p.exists():
        return
    cfg = {
        "name": "viana_do_castelo",
        "center": {"lat": 41.695, "lon": -8.833},
        "bbox_km": 5,
        "time_range": {"start": "2018-01-01", "end": "2023-12-31"},
        "open_database_root": "data/open_database/viana",
        "dataset_output": "data/datasets/viana_50.json",
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def main() -> int:
    repo_root = Path.cwd()
    cfg_path = repo_root / "configs" / "viana_dataset.yaml"
    ensure_viana_config(cfg_path)
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    center = cfg.get("center", {})
    center_lat = float(center.get("lat", 41.695))
    center_lon = float(center.get("lon", -8.833))

    layout = generate_viana_layout(center_lat=center_lat, center_lon=center_lon, n_devices=50)
    layout_path = repo_root / "configs" / "viana_50_layout.json"
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    print(f"Building open database at: {cfg['open_database_root']}")
    build_open_database(cfg["open_database_root"])  # populates a small manifest and downloads

    print("Generating analytic WEC dataset (50 devices)")
    freq = make_frequency_grid()
    splits = build_wec_database(n_samples=50, freq=freq, seed=1)

    # Flatten splits to assign locations deterministically to each case id
    all_records = splits["train"] + splits["val"] + splits["test"]
    for idx, rec in enumerate(all_records):
        loc = layout[idx % len(layout)]
        rec["location"] = loc

    out_path = repo_root / cfg["dataset_output"]
    write_dataset_json(out_path, {"all": all_records}, metadata={"source": "analytic_viana_phase1"})
    print(f"Wrote analytic viana dataset to: {out_path}")
    print(f"Layout written to: {layout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
