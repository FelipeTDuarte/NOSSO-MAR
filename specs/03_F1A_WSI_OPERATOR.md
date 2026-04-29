# Specification 03: F1A - Array WSI (Wave-Structure Interaction) Operator

## Purpose
Learn a differentiable operator that maps local wave conditions + device properties to hydrodynamic loads, body response, and power absorption for one or multiple WECs. This replaces BEM solvers (Capytaine, HAMS, NEMOH) at inference time.

## Scope
- **Phase(s)**: Phase 1A (Task 3–4)
- **Status**: Architecture draft → Implementation → Validation
- **Dependencies**: IO Contracts (Spec 01), Data Strategy (Spec 02)

---

## Physical Problem

**Inputs**:
- Incident wave spectrum: S(ω, heading) [m²/s]
- Local water depth: h [m]
- Device properties: geometry (radius, draft), mass, PTO damping
- Array layout: positions {(x_i, y_i)} relative to each other

**Outputs**:
- Frequency-dependent coefficients: A(ω), B(ω), Fex(ω, heading)
- Radiation impedance: Z_rad(ω) = B(ω) + iω·A(ω)
- Response amplitude operator: RAO(ω, heading) = Fex / (impedance)
- Absorbed power: P_abs = ∫ RAO² · S(ω) · dω
- Hydrodynamic interaction kernel: G_{ij}(ω) (between device i and j)

---

## Candidate Architectures

### 1. **DeepONet (Recommended for Phase 1A)**

**Why**: 
- Mesh-independent (can handle any geometry / frequency bin)
- Interpretable: branch network encodes device properties, trunk network queries frequencies
- Proven on PDE operator learning (Lu et al. 2021, 2022)
- Handles parameterized input geometry elegantly

**Architecture**:
```
Branch Net:
  Input: device params (radius, draft, mass, Bpto, depth) [5D]
  → Dense(64) → ReLU → Dense(128) → ReLU → Dense(256)
  → Embedding: b ∈ ℝ^256

Trunk Net:
  Input: query frequencies ω ∈ [0.1, 2.0] Hz [1D]
  → Fourier encoding: [sin(2πω), cos(2πω), sin(4πω), ...] [20D]
  → Dense(64) → ReLU → Dense(128) → ReLU → Dense(256)
  → Embedding: t ∈ ℝ^256

Operator:
  Output = (b ⊙ t) @ W_out + b_out
  where ⊙ is element-wise product, W_out ∈ ℝ^{256 × 4}
  
  Outputs 4 channels: [A(ω), B(ω), Re(Fex(ω)), Im(Fex(ω))]
```

**Advantages**:
- No need to recompute for new frequencies (continuous trunk)
- Generalizes to unseen parameter ranges (if training covers)
- Interpretable: branch can be inspected for learned physics features

---

### 2. **Graph Neural Operator (GNO) for Multi-Body Arrays**

**Why**: 
- Multi-body interactions are inherently graph structure
- Each WEC is a node; inter-device distance/interaction is edge feature
- Learns edge corrections (hydrodynamic interaction between pairs)

**Architecture** (Phase 1A2, multi-body):
```
Message Passing:
  For each WEC i:
    1. Self features: [radius_i, draft_i, mass_i, Bpto_i]
    2. Messages from neighbors j ∈ N(i):
       edge_feat = [distance_{ij}, relative_heading]
       msg_{ij} = MLP_edge(edge_feat)
    3. Aggregate: msg_i = Σ_j msg_{ij}
    4. Update: h_i^{t+1} = MLP_node([h_i^t, msg_i])
  
  After T steps of message passing:
    Output per node: A_i(ω), B_i(ω), Fex_i(ω) + correction term
```

---

## Training Data Requirements

**Dataset size**: 500–1000 cases
- **Single WEC** (F1A1): 500 cases from Capytaine
  - Parameters: radius [3–12] m, draft [2–10] m, depth [30–100] m, Bpto [0–100] kN·s/m
  - Frequencies: 0.1–2.0 Hz (100 bins)
  
- **2-body arrays** (F1A2 exploration): 100 cases
  - Same device parameters + separation distance [5–50] m

**Train/Val/Test split**: 70% / 15% / 15%

**Augmentation** (if needed):
- Frequency-domain scaling (invariance to frequency scaling)
- Parameter jittering (±5% noise on device properties)

---

## Loss Functions

### Supervised Loss
```
L_sup = MSE(A_pred, A_true) + MSE(B_pred, B_true) + MSE(Fex_pred, Fex_true)
      + (optional) MSE(power_pred, power_true)
```

### Physics-Informed Loss (PINO)
```
L_physics = L_causality + L_energy + L_convexity

L_causality:   operator must respect causality in frequency domain
L_energy:      B(ω) ≥ 0 (damping always dissipative)
L_convexity:   A(ω) ≥ A_∞ (added mass decreases with frequency)
```

### Total Loss
```
L_total = λ_sup · L_sup + λ_physics · L_physics
where λ_sup = 1.0, λ_physics = 0.1 (phase 1)
```

---

## Validation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| A(ω) RMSE | √[(A_pred - A_true)² / n] | < 5% of A_true |
| B(ω) RMSE | √[(B_pred - B_true)² / n] | < 5% of B_true |
| Power RMSE | √[(P_pred - P_true)² / n] | < 10% of P_true |
| Generalization | RMSE on held-out test set | Similar to val error |
| Physics violation | # cases where B(ω) < 0 | 0 |

---

## Implementation Roadmap

**Task 3 (Phase 1A1 - Single WEC DeepONet)**:
```
Step 1: Test RED — define DeepONet architecture + forward shape
Step 2: Implement src/nossomar/operators/deeponet_wec.py
Step 3: Train on 500 WEC cases with L_sup
Step 4: Validate: RMSE < 5% on test set
Step 5: Commit with paper summary
```

**Task 4 (Phase 1A2 - Multi-body GNO, optional)**:
```
Similar pattern, but with GNO architecture
Input: batch of device arrays, output: per-device predictions + interaction terms
```

---

## Computational Requirements

- **Training time**: 1–2 hours on GPU (V100/A100)
- **Memory**: ~4 GB GPU (single batch size 32)
- **Inference**: ~1 ms per query (real-time capable)

---

## Related Files

- **GitHub**: 
  - `src/nossomar/operators/deeponet_wec.py` (F1A1)
  - `src/nossomar/operators/gno_wec.py` (F1A2, optional)
  - `src/nossomar/modules/multi_object_fsi.py` (multi-body wrapper)
- **Tests**: `tests/test_deeponet_wec.py`, `tests/test_gno_wec.py`
- **Config**: `configs/f1a_deeponet.yaml`, `configs/f1a_gno.yaml`
- **Data**: `data/phase1_wec_database.zarr/`

---

## Benchmark: RM3 WEC

**Reference case**: National Renewable Energy Lab (NREL) RM3 (Reference Model 3)
- Cylindrical point absorber, radius 5.0 m, draft 3.5 m, mass 331 tons
- WEC-Sim hydrodynamic database available (NREL GitHub)

**Success criterion**: 
- Operator predictions within 10% of WEC-Sim for standard sea states
- Zero physics violations (B ≥ 0 always)

---

## Expected Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Frequency extrapolation (test frequencies outside training range) | Include boundary cases in training (0.05, 2.5 Hz) |
| Array interaction learned imperfectly | Use analytical superposition as baseline, learn residual |
| Few training cases for geometry corners | Use transfer learning or data augmentation |
| Physics loss destabilizes training | Start with L_sup only, add L_physics after convergence |

---

## Future Extensions (Phase 2+)

- **Shape-aware learning**: Support arbitrary WEC geometries (not just cylinders)
- **Viscous damping**: Include skin friction, flow separation damping
- **Nonlinear effects**: Amplitude-dependent coefficients A(ω, η)
- **Coupled feedback**: Modify incident wave spectrum based on array power extraction
