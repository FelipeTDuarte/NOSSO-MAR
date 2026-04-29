from nossomar.data.wec_dataset import generate_analytic_dataset, load_dataset


def test_dataset_roundtrip(tmp_path):
    ds = generate_analytic_dataset(n_cases=50, seed=42)
    path = tmp_path / "wec_test.zarr"
    ds.to_zarr(path)
    ds2 = load_dataset(path)
    assert "added_mass" in ds2
    assert ds2["added_mass"].dims == ("case", "omega")
    assert not ds2["added_mass"].isnull().any()
