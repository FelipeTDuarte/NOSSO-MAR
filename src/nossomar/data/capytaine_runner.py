"""Capytaine-backed WEC hydrodynamic dataset generation."""

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


def _run_single_payload(
    payload: tuple[int, dict[str, float], list[float], bool, tuple[int, int, int], bool],
) -> dict[str, Any]:
    index, sample, freq, use_capytaine, mesh_resolution, check_wavelength = payload
    runner = CapytaineRunner(
        use_capytaine=use_capytaine,
        max_workers=1,
        mesh_resolution=mesh_resolution,
        check_wavelength=check_wavelength,
    )
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
    mesh_resolution: tuple[int, int, int] = (1, 8, 4)
    check_wavelength: bool = False

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
        """Run Capytaine radiation and diffraction problems for heave."""

        import capytaine as cpt

        body_mass = float(mass if mass is not None else 1025.0 * np.pi * radius**2 * draft)
        mesh = cpt.mesh_vertical_cylinder(
            length=float(draft),
            radius=float(radius),
            center=(0.0, 0.0, -0.5 * float(draft)),
            resolution=self.mesh_resolution,
            name=f"{case_id}_mesh",
        )
        body = cpt.FloatingBody(
            mesh=mesh,
            mass=body_mass,
            center_of_mass=(0.0, 0.0, -0.5 * float(draft)),
            name=case_id,
        )
        body.add_translation_dof(name="Heave")

        solver = cpt.BEMSolver()
        radiation_problems = [
            cpt.RadiationProblem(
                body=body,
                water_depth=float(depth),
                freq=float(freq_value),
                radiating_dof="Heave",
                rho=1025.0,
            )
            for freq_value in freq
        ]
        diffraction_problems = [
            cpt.DiffractionProblem(
                body=body,
                water_depth=float(depth),
                freq=float(freq_value),
                wave_direction=0.0,
                rho=1025.0,
            )
            for freq_value in freq
        ]
        results = solver.solve_all(
            radiation_problems + diffraction_problems,
            n_jobs=1,
            keep_details=False,
            _check_wavelength=self.check_wavelength,
        )
        radiation_results = sorted(
            [res for res in results if hasattr(res, "radiating_dof")],
            key=lambda res: float(res.freq),
        )
        diffraction_results = sorted(
            [res for res in results if not hasattr(res, "radiating_dof")],
            key=lambda res: float(res.freq),
        )
        if len(radiation_results) != len(freq) or len(diffraction_results) != len(freq):
            raise RuntimeError(
                "Capytaine returned an unexpected result set: "
                f"{len(radiation_results)} radiation and {len(diffraction_results)} diffraction results "
                f"for {len(freq)} frequencies."
            )

        added_mass_signed = np.asarray([res.added_mass["Heave"] for res in radiation_results], dtype=float)
        damping_signed = np.asarray([res.radiation_damping["Heave"] for res in radiation_results], dtype=float)
        excitation = np.asarray([res.forces["Heave"] for res in diffraction_results], dtype=complex)

        return WECState(
            freq=freq,
            added_mass=np.maximum(np.abs(added_mass_signed), 1.0e-9),
            damping=np.maximum(np.abs(damping_signed), 1.0e-9),
            excitation_real=excitation.real,
            excitation_imag=excitation.imag,
            device_type="capytaine_vertical_cylinder",
            radius=radius,
            draft=draft,
            mass=body_mass,
            bpto=bpto,
            depth=depth,
            metadata={
                "case_id": case_id,
                "backend": "capytaine",
                "dof": "Heave",
                "mesh_resolution": list(self.mesh_resolution),
                "raw_added_mass_signed": added_mass_signed.tolist(),
                "raw_radiation_damping_signed": damping_signed.tolist(),
                "sign_convention": "absolute values stored in WECState to satisfy non-negative A/B contract",
            },
        )

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
            (
                index,
                sample,
                list(np.asarray(freq_array, dtype=float)),
                self.use_capytaine,
                self.mesh_resolution,
                self.check_wavelength,
            )
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
