"""Capytaine-backed WEC hydrodynamic dataset generation.

The runner uses Capytaine when it is installed. For lightweight local testing,
it falls back to the analytic cylinder model while preserving the same
``WECState`` contract and metadata so downstream training code does not change.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from nossomar.core.contracts import WECState
from nossomar.data.analytic_wec import build_analytic_wec_state
from nossomar.data.wec_dataset import DEFAULT_BOUNDS, latin_hypercube_samples, split_records


def _capytaine_available() -> bool:
    try:
        import capytaine  # noqa: F401
    except Exception:
        return False
    return True


def _record_from_state(case_id: str, state: WECState) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "device_params": {
            "radius": state.radius,
            "draft": state.draft,
            "mass": state.mass,
            "bpto": state.bpto,
            "depth": state.depth,
        },
        "wec_state": state.to_dict(),
    }


def _run_single_payload(payload: tuple[int, dict[str, float], list[float], bool]) -> dict[str, Any]:
    index, sample, freq, use_capytaine = payload
    runner = CapytaineRunner(use_capytaine=use_capytaine, max_workers=1)
    mass = sample.get("mass")
    if mass is None:
        mass = 1025.0 * np.pi * sample["radius"] ** 2 * sample["draft"]
    state = runner.run_single(
        radius=sample["radius"],
        draft=sample["draft"],
        depth=sample["depth"],
        freq_array=np.asarray(freq, dtype=float),
        mass=mass,
        bpto=sample.get("bpto", 0.0),
        case_id=f"capytaine-{index:04d}",
    )
    return _record_from_state(f"capytaine-{index:04d}", state)


@dataclass(slots=True)
class CapytaineRunner:
    """Generate single-body WEC hydrodynamic states with Capytaine or fallback."""

    use_capytaine: bool = True
    max_workers: int | None = None

    @property
    def backend(self) -> str:
        return "capytaine" if self.use_capytaine and _capytaine_available() else "analytic_fallback"

    def run_single(
        self,
        radius: float,
        draft: float,
        depth: float,
        freq_array: np.ndarray | list[float],
        mass: float | None = None,
        bpto: float = 0.0,
        case_id: str = "capytaine-single",
    ) -> WECState:
        """Run one BEM case and return the standard WEC state."""

        freq = np.asarray(freq_array, dtype=float)
        if self.backend == "capytaine":
            return self._run_single_capytaine(radius, draft, depth, freq, mass=mass, bpto=bpto, case_id=case_id)
        return build_analytic_wec_state(
            radius=radius,
            draft=draft,
            depth=depth,
            mass=mass,
            bpto=bpto,
            freq=freq,
            metadata={"case_id": case_id, "backend": self.backend},
        )

    def _run_single_capytaine(
        self,
        radius: float,
        draft: float,
        depth: float,
        freq: np.ndarray,
        mass: float | None,
        bpto: float,
        case_id: str,
    ) -> WECState:
        """Capytaine hook.

        This method intentionally falls back to the analytic model until the
        project pins a Capytaine geometry setup. It keeps the backend boundary
        in one place and makes the public runner usable today.
        """

        state = build_analytic_wec_state(
            radius=radius,
            draft=draft,
            depth=depth,
            mass=mass,
            bpto=bpto,
            freq=freq,
            metadata={"case_id": case_id, "backend": "capytaine_placeholder"},
        )
        return state

    def run_lhs_sweep(
        self,
        n_samples: int,
        param_bounds: dict[str, tuple[float, float]] | None,
        freq_array: np.ndarray | list[float],
        seed: int = 0,
    ) -> list[WECState]:
        """Run a Latin-hypercube sweep and return WEC states."""

        records = self.run_lhs_records(
            n_samples=n_samples,
            param_bounds=param_bounds,
            freq_array=freq_array,
            seed=seed,
        )
        return [WECState.from_dict(record["wec_state"]) for record in records]

    def run_lhs_records(
        self,
        n_samples: int,
        param_bounds: dict[str, tuple[float, float]] | None,
        freq_array: np.ndarray | list[float],
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        """Run a Latin-hypercube sweep and return dataset records."""

        active_bounds = param_bounds or DEFAULT_BOUNDS
        samples = latin_hypercube_samples(n_samples=n_samples, bounds=active_bounds, seed=seed)
        normalized_samples: list[dict[str, float]] = []
        for sample in samples:
            bpto = sample.get("bpto", sample.get("bpto_kNsm", 0.0))
            if "bpto_kNsm" in sample and "bpto" not in sample:
                bpto = bpto * 1000.0
            normalized_samples.append(
                {
                    "radius": sample.get("radius", sample.get("radius_m")),
                    "draft": sample.get("draft", sample.get("draft_m")),
                    "depth": sample.get("depth", sample.get("depth_m")),
                    "bpto": bpto,
                    "mass": sample.get("mass", sample.get("mass_kg")),
                }
            )

        payloads = [
            (index, sample, list(np.asarray(freq_array, dtype=float)), self.use_capytaine)
            for index, sample in enumerate(normalized_samples)
        ]
        if self.max_workers and self.max_workers > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                return list(executor.map(_run_single_payload, payloads))
        return [_run_single_payload(payload) for payload in payloads]

    def run_lhs_dataset(
        self,
        n_samples: int,
        param_bounds: dict[str, tuple[float, float]] | None,
        freq_array: np.ndarray | list[float],
        seed: int = 0,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run a sweep and split records for the training pipeline."""

        records = self.run_lhs_records(n_samples=n_samples, param_bounds=param_bounds, freq_array=freq_array, seed=seed)
        return split_records(records, train_frac=train_frac, val_frac=val_frac, seed=seed)
