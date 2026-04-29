from __future__ import annotations

import numpy as np

from nossomar.core.contracts import OceanState, WaveField, WECState
from nossomar.data.analytic_wec import build_analytic_wec_state, make_frequency_grid


def test_wec_state_json_round_trip(tmp_path) -> None:
    freq = make_frequency_grid(count=8)
    state = build_analytic_wec_state(radius=5.0, draft=4.0, freq=freq, depth=60.0, bpto=20_000.0)
    path = tmp_path / "wec_state.json"
    state.to_json(path)
    loaded = WECState.from_json(path)

    np.testing.assert_allclose(loaded.freq, state.freq)
    np.testing.assert_allclose(loaded.added_mass, state.added_mass)
    np.testing.assert_allclose(loaded.damping, state.damping)
    assert loaded.device_type == "cylinder"


def test_ocean_state_round_trip(tmp_path) -> None:
    freq = make_frequency_grid(count=6)
    state = build_analytic_wec_state(radius=6.0, draft=3.5, freq=freq, depth=45.0)
    wave_field = WaveField(
        x=np.array([0.0, 10.0]),
        y=np.array([0.0, 10.0]),
        time=np.array([0.0, 1.0, 2.0]),
        elevation=np.zeros((3, 2, 2)),
        metadata={"solver_reference": "local_baseline"},
    )
    ocean_state = OceanState(wec_states=[state], wave_field=wave_field, metadata={"case_id": "case-001"})
    path = tmp_path / "ocean_state.json"
    ocean_state.to_json(path)
    loaded = OceanState.from_json(path)

    assert loaded.metadata["case_id"] == "case-001"
    assert len(loaded.wec_states) == 1
    np.testing.assert_allclose(loaded.wave_field.elevation, wave_field.elevation)
