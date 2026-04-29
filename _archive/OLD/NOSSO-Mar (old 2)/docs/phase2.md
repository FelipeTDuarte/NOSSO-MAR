# NOSSO-MAR Phase 2

## Overview

Phase 2 extends the Phase 1 operator-learning surrogate in three directions:

1. **Resolution-independent geometry encoding** via the RINO encoder, which accepts arbitrary point clouds instead of fixed parameter vectors.
2. **Real solver data ingestion** via the WEC-Sim adapter, which loads NREL reference case HDF5 output into the Phase 1 training schema.
3. **Pairwise hydrodynamic interaction** via the GAT interaction module, which computes wave-radiation coupling corrections between WECs in a farm.

The Phase 2 system is fully differentiable and trains end-to-end.

---

## Module Map

| Module | Path | Role |
|--------|------|------|
| RINO Encoder | `src/nossomar/operators/rino_encoder.py` | Point cloud → latent vector |
| WEC-Sim Adapter | `src/nossomar/data/wec_sim_adapter.py` | HDF5/JSON → xarray dataset |
| GAT Interaction | `src/nossomar/modules/multi_object_interaction.py` | Pairwise force correction |
| Phase 2 Training | `src/nossomar/training/train_phase2.py` | End-to-end training loop |
| Validation CLI | `scripts/validate_phase2.py` | Benchmark runner |

---

## RINO Encoder

The Resolution-Independent Neural Operator (RINO) encoder maps an arbitrary-size point cloud of geometry samples to a fixed-size latent vector using a PointNet-style shared MLP followed by global max-pooling.

**Input:** `points (B, N_pts, 3)`, `features (B, N_pts, F)` — spatial coordinates and per-point physical quantities (e.g. panel normals, areas).

**Output:** `latent (B, d_latent)` — resolution-invariant geometry representation.

The max-pool aggregation is permutation-invariant and resolution-invariant by construction: the same geometry sampled at N=64 or N=512 produces a consistent latent because the global maximum is taken across all points regardless of count.

---

## WEC-Sim Adapter

Loads hydrodynamic data from WEC-Sim output files into the NOSSO-MAR xarray schema.

**Supported formats:**
- `.json` — synthetic benchmarks and CI fixtures (no extra dependencies)
- `.h5` / `.hdf5` — real WEC-Sim output (requires `pip install h5py`)

**Output schema:**
```
Coordinates : omega  (N_omega,)        rad/s
Data vars   : added_mass   (case, omega)  kg
              damping      (case, omega)  kg/s
              radius       (case,)        m
              draft        (case,)        m
              depth        (case,)        m
              bpto         (case,)        N·s/m
Attributes  : source, body_name
```

**NREL reference cases** (Level 1 validation data):
- RM3 two-body point absorber: https://github.com/WEC-Sim/WEC-Sim_Applications
- OSWEC flap device: same repository, `OSWEC/` subdirectory

---

## GAT Interaction Module

Models pairwise hydrodynamic interaction between WECs using a two-layer Graph Attention Network.

**Architecture:**
- Fully-connected graph over N WECs (every WEC influences every other)
- Edge features: `[dx, dy, dist, dx/D_ref, dy/D_ref, 1/(1+dist)]`
- Two GAT layers with multi-head attention
- Output MLP: per-node 6-DOF force correction

**Key properties:**
- Permutation-equivariant by construction
- Single-WEC short-circuit returns zeros (avoids empty edge NaN)
- Distance decay encoded as prior via `1/(1+dist)` edge feature

---

## Training

```bash
# Quick training run with analytic data
python -c "
from nossomar.data.wec_dataset import generate_analytic_dataset
from nossomar.training.train_phase2 import train_phase2
ds = generate_analytic_dataset(n_cases=2000, seed=42)
history = train_phase2(ds, epochs=200, lr=3e-4)
print(f'Final loss: {history[-1]:.6f}')
"
```

The training loop uses a warm-start strategy: the RINO encoder is frozen for the first `warmup_epochs` (default 5) while the DeepONet adapts to the new latent input space. Both modules then fine-tune jointly with AdamW + cosine annealing.

---

## Validation

```bash
# Smoke test (analytic data, no checkpoints needed)
python scripts/validate_phase2.py --smoke

# Against WEC-Sim benchmark
python scripts/validate_phase2.py \
    --benchmark data/benchmarks/rm3_heave_synthetic.json

# Full comparison with saved checkpoints
python scripts/validate_phase2.py \
    --checkpoint-p1 checkpoints/phase1.pt \
    --checkpoint-p2 checkpoints/phase2.pt \
    --benchmark data/benchmarks/rm3_heave_synthetic.json
```

**Pass criterion:** Phase 2 relative RMSE on added mass A(ω) < 15% on the RM3 benchmark.

---

## Phase 3 Roadmap

Phase 2 establishes the full pipeline from arbitrary geometry to trained surrogate. Phase 3 will introduce:

- **Farm-scale optimisation** — layout optimisation using the GAT interaction module as a differentiable objective
- **Real OpenFOAM data** — CFD-validated training for extreme sea states
- **Uncertainty quantification** — ensemble or Bayesian extension for confidence intervals on power estimates
- **Multi-DOF generalisation** — extend from heave-only to 6-DOF response
