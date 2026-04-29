from __future__ import annotations

import json

from nossomar.data.open_data_catalog import catalog_records, write_catalog
from nossomar.data.database_builder import WEC_BENCHMARKS


def test_open_data_catalog_contains_core_categories(tmp_path) -> None:
    records = catalog_records()
    categories = {record["category"] for record in records}
    assert "bathymetry" in categories
    assert "temperature" in categories
    assert "wec_benchmarks" in categories

    path = write_catalog(tmp_path / "catalog.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == len(records)


def test_wec_benchmarks_are_seeded_with_open_sources() -> None:
    benchmark_ids = {record["benchmark_id"] for record in WEC_BENCHMARKS}
    assert {"rm3", "mbari_wec_2021", "aquaharmonics_1_20", "lupa_wecsim_moordyn"} <= benchmark_ids
