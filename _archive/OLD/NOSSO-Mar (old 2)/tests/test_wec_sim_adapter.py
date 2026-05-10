"""RED tests for WEC-Sim data adapter (P2-T2)."""
from __future__ import annotations

import json
import pytest
import numpy as np
from pathlib import Path
from nossomar.data.wec_sim_adapter import load_wecsim_dataset


SYNTHETIC_BENCH = {
    "source": "synthetic",
    "body_name": "test_cylinder",
    "omega": [0.2, 0.5, 1.0, 2.0, 3.0],
    "added_mass_heave": [1.2e5, 1.1e5, 9.0e4, 6.5e4, 4.0e4],
    "radiation_damping_heave": [1.0e3, 2.5e3, 4.0e3, 3.0e3, 1.5e3],
    "draft": 2.0,
    "depth": 50.0,
    "bpto": 1e4,
    "radius": 1.0,
}


@pytest.fixture
def bench_file(tmp_path):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(SYNTHETIC_BENCH), encoding="utf-8")
    return p


def test_load_returns_dataset(bench_file):
    import xarray as xr
    ds = load_wecsim_dataset(bench_file)
    assert isinstance(ds, xr.Dataset)


def test_required_vars_present(bench_file):
    ds = load_wecsim_dataset(bench_file)
    for var in ("added_mass", "damping", "radius", "draft", "depth", "bpto"):
        assert var in ds, f"Missing variable: {var}"


def test_dims_correct(bench_file):
    ds = load_wecsim_dataset(bench_file)
    assert ds["added_mass"].dims == ("case", "omega")
    assert ds["damping"].dims == ("case", "omega")


def test_no_nan(bench_file):
    ds = load_wecsim_dataset(bench_file)
    assert not ds["added_mass"].isnull().any()
    assert not ds["damping"].isnull().any()


def test_omega_ascending(bench_file):
    ds = load_wecsim_dataset(bench_file)
    omega = ds["omega"].values
    assert np.all(np.diff(omega) > 0), "omega must be strictly ascending"


def test_values_positive(bench_file):
    ds = load_wecsim_dataset(bench_file)
    assert (ds["added_mass"].values > 0).all(), "added_mass must be positive"
    assert (ds["damping"].values >= 0).all(), "damping must be non-negative"


def test_scalar_geometry_fields(bench_file):
    ds = load_wecsim_dataset(bench_file)
    for field in ("radius", "draft", "depth", "bpto"):
        assert ds[field].dims == ("case",), f"{field} must have dim (case,)"


def test_missing_key_raises(tmp_path):
    bad = {k: v for k, v in SYNTHETIC_BENCH.items() if k != "added_mass_heave"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises((KeyError, ValueError)):
        load_wecsim_dataset(p)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_wecsim_dataset(Path("nonexistent_file.json"))


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_wecsim_dataset(p)
