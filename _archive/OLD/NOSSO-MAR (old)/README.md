# NOSSO-MAR

**Neural Operator Scalable System for Ocean Modelling, Analysis, and Renewables**

A research-grade, modular AI framework for ocean and coastal physics simulation,
built on Physics-Informed Multi-Resolution Neural Operators (PIMR-NO) and
Multi-Agent Reinforcement Learning (MARL), targeting EU-scale ERC research.

---

## Architecture at a glance

| Layer | Components |
|---|---|
| **Neural Operators** | FNO2d/3d, F-FNO, Geo-FNO, WNO, GNO, DeepONet, RBF-NO, KAN-NO, AMR-NO |
| **Physics Modules** | Wave propagation, WEC FSI (BEM), Tides, Morphodynamics, Tracers |
| **Data Assimilation** | EnKF, 4D-Var (AD through NOs), buoy + satellite obs operators |
| **Digital Twin** | Sensor ingestion, state estimation, anomaly detection, REST API |
| **MARL** | MADDPG layout agent, MAPPO PTO agent, WaveFarmEnv |
| **HPC** | DDP (NCCL), SLURM launcher (LUMI/MN5/Leonardo), mixed precision |
| **Data pipeline** | Capytaine (BEM), OceanWave3D, LHS sampler, Zarr cloud store |

## Phase 1 — Core scientific contribution

Replaces phase-resolving wave solver (OceanWave3D) + BEM (Capytaine/HAMS)
with coupled neural operator surrogates inside a GA optimisation loop.

```
WavePropagationNO (WNO) ↔ WecFSIModule (DeepONet)  →  P_total  →  GA fitness
```

Expected speedup: **100×–1000×** per fitness evaluation vs classical solvers.

## Quickstart

```bash
conda env create -f environment.yml && conda activate nosso-mar
pre-commit install

# Generate BEM training data (requires capytaine)
python scripts/generate_bem_data.py --n_samples 10000 --n_workers 8

# Train Module 2 (BEM surrogate)
python scripts/train_deeponet_bem.py --config configs/training/deeponet_bem.yaml

# Train Module 1 (wave propagation)
python scripts/train_fno.py --config configs/training/fno_wave.yaml

# Benchmark inference speed
python scripts/benchmarks/operator_benchmark.py

# Train MARL farm optimisation
python scripts/train_marl_farm.py --config configs/marl/maddpg_farm.yaml

# Run digital twin
python scripts/run_digital_twin.py --config configs/digital_twin/site_atlantic.yaml
```

## HPC (LUMI, MareNostrum 5, Leonardo)

```bash
# Distributed training on 16 nodes
python -c "
from src.nosso_mar.hpc.slurm_launcher import SLURMLauncher
SLURMLauncher.from_config('configs/hpc/lumi.yaml').submit(
    'scripts/train_fno.py', 'configs/training/ffno_large.yaml')
"
```

See `docs/HPC_GUIDE.md` for full cluster setup.

## Documentation

- `docs/ARCHITECTURE.md` — full module map
- `docs/PHASE_1.md` — Phase 1 scientific design
- `docs/DATA_ASSIMILATION.md` — EnKF / 4D-Var
- `docs/MARL_GUIDE.md` — MADDPG / MAPPO agents
- `docs/HPC_GUIDE.md` — SLURM / DDP guide
- `docs/ROADMAP.md` — Phases 0–7

## Citation

```bibtex
@software{nosso_mar_2025,
  title  = {NOSSO-MAR: Neural Operator Scalable System for Ocean Modelling, Analysis, and Renewables},
  author = {Duarte, Felipe T.},
  year   = {2025},
  license = {MIT}
}
```
