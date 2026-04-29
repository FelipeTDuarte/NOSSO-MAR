# NOSSO-MAR Subagent-Driven Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build NOSSO-MAR here in a staged, testable way with minimal dependence on numerical or physical-model training data.

**Architecture:** Start from strict IO contracts, then analytic low-cost data generators, then an xarray dataset pipeline, then a DeepONet surrogate for WEC hydrodynamics, then the training loop, public-benchmark validation, and only after that integrate the multi-object FSI loop. Every task is designed for fresh subagent execution with spec review and code-quality review after each task.

**Tech Stack:** Python, PyTorch, xarray, Zarr, pytest, scipy

---

## File Map

- `src/nossomar/core/contracts.py` — canonical dataclasses and validation helpers
- `src/nossomar/data/analytic_wec.py` — analytic low-cost hydrodynamics approximations
- `src/nossomar/data/wec_dataset.py` — LHS sampling and xarray dataset generation
- `src/nossomar/data/public_benchmarks.py` — loaders for public benchmark hydrodynamics
- `src/nossomar/operators/deeponet_wec.py` — DeepONet surrogate for A(ω), B(ω)
- `src/nossomar/modules/multi_object_fsi.py` — multi-body FSI adapter contract target
- `src/nossomar/coupling/coupled_pino.py` — coupled execution loop entrypoint
- `src/nossomar/training/train_wec.py` — train loop for phase-1 surrogate
- `tests/test_contracts.py` — contract tests
- `tests/test_analytic_wec.py` — analytic physics tests
- `tests/test_wec_dataset.py` — dataset roundtrip tests
- `tests/test_deeponet_wec.py` — network shape tests
- `tests/test_train_wec.py` — loss descent smoke test
- `tests/test_multi_object_fsi.py` — adapter integration smoke test
- `scripts/validate_phase1.py` — public benchmark validation runner
- `configs/phase1_wec.yaml` — phase-1 training config
- `.superpowers/subagent-workflow.md` — execution checklist for controller agent

---

### Task 1: Contracts and Shape Safety

**Files:**
- Create: `src/nossomar/core/contracts.py`
- Create: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import torch
from nossomar.core.contracts import WECState, WaveField, validate_wec_state


def test_wec_state_shape():
    s = WECState(pos=torch.zeros(3, 2), vel=torch.zeros(3, 6), force=torch.zeros(3, 6))
    assert s.pos.shape == (3, 2)


def test_wave_field_shape():
    wf = WaveField(
        eta=torch.zeros(2, 64, 64, 10),
        u=torch.zeros(2, 64, 64, 10),
        v=torch.zeros(2, 64, 64, 10),
    )
    assert wf.eta.shape == (2, 64, 64, 10)


def test_validate_rejects_nan():
    s = WECState(
        pos=torch.tensor([[float("nan"), 0.0]]),
        vel=torch.zeros(1, 6),
        force=torch.zeros(1, 6),
    )
    with pytest.raises(ValueError, match="NaN"):
        validate_wec_state(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contracts.py -v`
Expected: FAIL with import or symbol errors.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
import torch


@dataclass
class WECState:
    pos: torch.Tensor
    vel: torch.Tensor
    force: torch.Tensor


@dataclass
class WaveField:
    eta: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


def validate_wec_state(s: WECState):
    for name, t in (("pos", s.pos), ("vel", s.vel), ("force", s.force)):
        if torch.isnan(t).any():
            raise ValueError(f"NaN detected in WECState.{name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contracts.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/nossomar/core/contracts.py tests/test_contracts.py
git commit -m "feat: add phase1 IO contracts"
```

---

### Task 2: Analytic WEC Data Generator

**Files:**
- Create: `src/nossomar/data/analytic_wec.py`
- Create: `tests/test_analytic_wec.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from nossomar.data.analytic_wec import analytic_added_mass_cylinder, analytic_damping_cylinder


def test_added_mass_deep_water_limit():
    rho, r, draft = 1025.0, 1.0, 2.0
    omega = np.array([5.0, 10.0, 20.0])
    A = analytic_added_mass_cylinder(omega, r, draft, depth=50.0, rho=rho)
    A_inf = rho * np.pi * r**2 * draft
    assert abs(A[-1] - A_inf) / A_inf < 0.20


def test_damping_decays_at_high_frequency():
    omega = np.linspace(0.1, 15.0, 50)
    B = analytic_damping_cylinder(omega, 1.0, 2.0, depth=50.0)
    assert B[-1] < 0.05 * B.max()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analytic_wec.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np


def _wavenumber(omega, depth, g=9.81, tol=1e-8, maxiter=50):
    omega = np.atleast_1d(omega)
    k = omega**2 / g
    for _ in range(maxiter):
        k_new = omega**2 / (g * np.tanh(k * depth))
        if np.max(np.abs(k_new - k)) < tol:
            return k_new
        k = k_new
    return k


def analytic_added_mass_cylinder(omega, radius, draft, depth, rho=1025.0):
    k = _wavenumber(omega, depth)
    kr = k * radius
    A_inf = rho * np.pi * radius**2 * draft
    correction = 1.0 - 0.5 * np.exp(-kr * draft)
    return A_inf * correction


def analytic_damping_cylinder(omega, radius, draft, depth, rho=1025.0, g=9.81):
    omega = np.atleast_1d(omega)
    k = _wavenumber(omega, depth)
    cg = g / (2 * omega) * (1 + 2 * k * depth / np.sinh(2 * k * depth))
    a_rad = (np.pi * radius**2 * rho * omega) ** 2 / (rho * cg * k)
    return a_rad * np.exp(-2 * k * draft)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analytic_wec.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/nossomar/data/analytic_wec.py tests/test_analytic_wec.py
git commit -m "feat: add analytic cylinder hydrodynamics generator"
```

---

### Task 3: xarray Dataset Pipeline

**Files:**
- Create: `src/nossomar/data/wec_dataset.py`
- Create: `tests/test_wec_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
from nossomar.data.wec_dataset import generate_analytic_dataset, load_dataset


def test_dataset_roundtrip(tmp_path):
    ds = generate_analytic_dataset(n_cases=50, seed=42)
    path = tmp_path / "wec_test.zarr"
    ds.to_zarr(path)
    ds2 = load_dataset(path)
    assert "added_mass" in ds2
    assert ds2["added_mass"].dims == ("case", "omega")
    assert not ds2["added_mass"].isnull().any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wec_dataset.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np
import xarray as xr
from scipy.stats import qmc
from .analytic_wec import analytic_added_mass_cylinder, analytic_damping_cylinder


def generate_analytic_dataset(n_cases=1000, seed=42, omega=np.linspace(0.2, 4.0, 40)):
    rng = qmc.LatinHypercube(d=4, seed=seed)
    samples = qmc.scale(
        rng.random(n_cases),
        l_bounds=[0.5, 0.5, 10.0, 1e3],
        u_bounds=[5.0, 8.0, 80.0, 1e5],
    )
    radii, drafts, depths, bptos = samples.T
    A = np.stack([analytic_added_mass_cylinder(omega, r, d, h) for r, d, h in zip(radii, drafts, depths)])
    B = np.stack([analytic_damping_cylinder(omega, r, d, h) for r, d, h in zip(radii, drafts, depths)])
    return xr.Dataset(
        {
            "added_mass": (["case", "omega"], A),
            "damping": (["case", "omega"], B),
            "radius": (["case"], radii),
            "draft": (["case"], drafts),
            "depth": (["case"], depths),
            "bpto": (["case"], bptos),
        },
        coords={"omega": omega},
    )


def load_dataset(path):
    return xr.open_zarr(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wec_dataset.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/nossomar/data/wec_dataset.py tests/test_wec_dataset.py
git commit -m "feat: add xarray zarr dataset pipeline"
```

---

### Task 4: DeepONet Surrogate for Phase 1

**Files:**
- Create: `src/nossomar/operators/deeponet_wec.py`
- Create: `tests/test_deeponet_wec.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from nossomar.operators.deeponet_wec import WECDeepONet


def test_forward_shape():
    model = WECDeepONet(branch_dim=4, trunk_dim=1, hidden=64, n_modes=32)
    props = torch.randn(8, 4)
    omega = torch.linspace(0.2, 4.0, 40).unsqueeze(0).repeat(8, 1)
    A, B = model(props, omega)
    assert A.shape == (8, 40)
    assert B.shape == (8, 40)
    assert not torch.isnan(A).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deeponet_wec.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WECDeepONet(nn.Module):
    def __init__(self, branch_dim=4, trunk_dim=1, hidden=128, n_modes=64):
        super().__init__()
        self.branch_A = _MLP([branch_dim, hidden, hidden, n_modes])
        self.branch_B = _MLP([branch_dim, hidden, hidden, n_modes])
        self.trunk = _MLP([trunk_dim, hidden, hidden, n_modes])
        self.bias_A = nn.Parameter(torch.zeros(1))
        self.bias_B = nn.Parameter(torch.zeros(1))

    def forward(self, props, omega):
        b_A = self.branch_A(props)
        b_B = self.branch_B(props)
        t = self.trunk(omega.unsqueeze(-1))
        A = (b_A.unsqueeze(1) * t).sum(-1) + self.bias_A
        B = (b_B.unsqueeze(1) * t).sum(-1) + self.bias_B
        return A, torch.relu(B)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_deeponet_wec.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/nossomar/operators/deeponet_wec.py tests/test_deeponet_wec.py
git commit -m "feat: add phase1 DeepONet surrogate"
```

---

### Task 5: Training Loop Smoke Test

**Files:**
- Create: `src/nossomar/training/train_wec.py`
- Create: `tests/test_train_wec.py`
- Create: `configs/phase1_wec.yaml`

- [ ] **Step 1: Write the failing test**

```python
from nossomar.training.train_wec import train_wec_surrogate
from nossomar.data.wec_dataset import generate_analytic_dataset


def test_loss_decreases():
    ds = generate_analytic_dataset(n_cases=200, seed=0)
    history = train_wec_surrogate(ds, epochs=5, lr=1e-3, batch_size=32)
    assert history[-1] < history[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_wec.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..operators.deeponet_wec import WECDeepONet


def train_wec_surrogate(ds, epochs=100, lr=3e-4, batch_size=64, device="cpu"):
    props = torch.tensor(
        np.stack(
            [ds["radius"].values, ds["draft"].values, ds["depth"].values, ds["bpto"].values],
            axis=1,
        ),
        dtype=torch.float32,
    )
    omega = torch.tensor(ds["omega"].values, dtype=torch.float32)
    A_tgt = torch.tensor(ds["added_mass"].values, dtype=torch.float32)
    B_tgt = torch.tensor(ds["damping"].values, dtype=torch.float32)

    loader = DataLoader(TensorDataset(props, A_tgt, B_tgt), batch_size=batch_size, shuffle=True)
    model = WECDeepONet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []

    for _ in range(epochs):
        losses = []
        for p, a, b in loader:
            p, a, b = p.to(device), a.to(device), b.to(device)
            om = omega.unsqueeze(0).repeat(p.shape[0], 1).to(device)
            A_pred, B_pred = model(p, om)
            loss = ((A_pred - a) ** 2).mean() + ((B_pred - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_wec.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/nossomar/training/train_wec.py tests/test_train_wec.py configs/phase1_wec.yaml
git commit -m "feat: add phase1 train loop smoke test"
```

---

### Task 6: Public Benchmark Validation

**Files:**
- Create: `src/nossomar/data/public_benchmarks.py`
- Create: `scripts/validate_phase1.py`

- [ ] **Step 1: Write the validation loader**

Implement a loader for a public WEC benchmark file and normalize outputs into `omega`, `added_mass_heave`, `draft`, `depth`, `bpto`, `radius`.

- [ ] **Step 2: Add validation script**

Run the trained surrogate on the benchmark and print relative RMSE.

- [ ] **Step 3: Verify script execution**

Run: `python scripts/validate_phase1.py --help`
Expected: CLI usage text.

- [ ] **Step 4: Commit**

```bash
git add src/nossomar/data/public_benchmarks.py scripts/validate_phase1.py
git commit -m "feat: add public benchmark validation scaffold"
```

---

### Task 7: Multi-Object FSI Contract Adapter

**Files:**
- Create: `src/nossomar/modules/multi_object_fsi.py`
- Create: `src/nossomar/coupling/coupled_pino.py`
- Create: `tests/test_multi_object_fsi.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from nossomar.modules.multi_object_fsi import MultiObjectFSI
from nossomar.core.contracts import WECState, WaveField


def test_multi_wec_no_nan():
    fsi = MultiObjectFSI(n_max_objects=5, hidden=32)
    wf = WaveField(
        eta=torch.randn(2, 64, 64, 8),
        u=torch.randn(2, 64, 64, 8),
        v=torch.randn(2, 64, 64, 8),
    )
    state = WECState(
        pos=torch.rand(3, 2) * 100,
        vel=torch.zeros(3, 6),
        force=torch.zeros(3, 6),
    )
    out_state, force_field = fsi(wf, state)
    assert not torch.isnan(out_state.force).any()
    assert force_field.shape == (2, 3, 64, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multi_object_fsi.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

Create a contract-safe placeholder implementation that computes a zero force field with correct shape and echoes state.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multi_object_fsi.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/nossomar/modules/multi_object_fsi.py src/nossomar/coupling/coupled_pino.py tests/test_multi_object_fsi.py
git commit -m "feat: add multi-object FSI contract adapter"
```

---

## Controller Notes

- Use **fresh subagent per task**. [file:39]
- After each implementer subagent, run a **spec compliance review** first, then **code quality review**. [file:39][file:41]
- Do not mark a task done without fresh verification evidence from the exact command named in the task. [file:12]
- Keep each task self-contained, small, and TDD-first. [file:11][file:2]
