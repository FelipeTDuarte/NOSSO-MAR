# FAQ: NOSSO-MAR Project

## General Questions

### Q: What is NOSSO-MAR?
**A**: NOSSO-MAR is a **Physics-Informed Neural Operator** framework for fast simulation of ocean waves and wave-energy converter (WEC) systems. It learns surrogate models from physics-governed training data, enabling 100–1000× speedup vs. traditional numerical solvers while maintaining fidelity.

[→ See Architecture.md for details](ARCHITECTURE.md)

---

### Q: Why neural operators instead of traditional CFD?
**A**: 
- **Speed**: 1000s of sea states in seconds vs. hours on HPC
- **Gradients**: Differentiable → direct use in optimization, control, inversion
- **Ensemble**: Low cost enables Monte Carlo sampling, UQ
- **Multi-query**: Real-time forecasting, digital twin assimilation

Tradeoff: High-fidelity reference data needed for training. We minimize cost via analytical + public data hierarchy.

[→ See Data Strategy (Spec 02)](../specs/02_DATA_STRATEGY.md)

---

### Q: Is NOSSO-MAR replacing numerical solvers?
**A**: No. Numerical solvers (OceanWave3D, SWASH, Capytaine, WEC-Sim) remain the source of truth for:
- Training data generation
- Validation benchmarks
- Edge cases / extrapolation testing

Neural operators are the **runtime engine** for fast simulation. Solvers are offline tools.

[→ See Relationship to Traditional Solvers in Architecture.md](ARCHITECTURE.md#relationship-to-traditional-solvers)

---

## Architecture & Design

### Q: What are the three main modules (F1A, F1B, F1C)?
**A**:

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| **F1A** | Wave-Structure Interaction (WSI) | Wave conditions + device params | Hydrodynamic loads (A, B, Fex), power |
| **F1B** | Wave Field Propagation | Bathymetry + incident spectrum | Phase-resolved field η(x,y,t), velocities |
| **F1C** | Bidirectional Coupling | Both | Wave field that accounts for devices |

[→ See Architecture.md Layer 3](ARCHITECTURE.md#layer-3-core-neural-operators)

---

### Q: Which neural operator architecture should I use?
**A**:

| Architecture | Best For | Pros | Cons |
|--------------|----------|------|------|
| **DeepONet** | F1A (WSI) | Interpretable, mesh-independent | Requires branch/trunk design |
| **GINO** (Graph) | F1B (waves, irregular domains) | Handles coastlines naturally | More complex implementation |
| **FNO** (Fourier) | F1B (regular domains) | Efficient, multi-scale | Poor on irregular boundaries |
| **WNO** (Wavelet) | F1B (breaking zones) | Multi-scale wavelets | Slower than FNO |

**Phase 1 recommendation**: DeepONet for F1A, GINO for F1B (if irregular bathymetry) or FNO (if regular).

[→ See Spec 03 & 04](../specs/03_F1A_WSI_OPERATOR.md), [../specs/04_F1B_WAVE_FIELD_OPERATOR.md]

---

### Q: What's the "coupling order"?
**A**: Modules are applied in strict sequence to avoid physics ambiguity:
```
1. Wetting/Drying → prevent numerical singularities
2. Atmospheric Forcing → apply wind stress
3. Object Interaction → compute device forces
4. Wave Propagation (F1B) → core operator
5. Post-Processing → masks, checks
```

[→ See Coupling Order in Specification 06](../specs/06_SCIENTIFIC_MODULES.md#coupling-order-phase-1-standard)

---

## Data & Training

### Q: How much training data do I need?
**A**: Phase 1 targets:
- **F1A (WSI)**: 500 WEC cases (Capytaine BEM sweep, LHS sampling)
- **F1B (Wave)**: 500 shallow water cases (OceanWave3D or SWASH reference)

Generated via hierarchy:
1. Analytical (free, fast)
2. Public datasets (free, curated)
3. Capytaine BEM (1–2 hours, 500 cases)
4. Solvers (validation only, <10%)

[→ See Data Strategy (Spec 02)](../specs/02_DATA_STRATEGY.md)

---

### Q: Can I train on simulated data only (no real observations)?
**A**: Yes, **Phase 1 does this intentionally**. We generate training data from solvers (Capytaine, OceanWave3D, SWASH). Real observations are **not** used in Phase 1; they're reserved for Phase 2 (data assimilation) and Phase 3 (operational digital twin).

[→ See Data Hierarchy in Spec 02](../specs/02_DATA_STRATEGY.md#data-hierarchy)

---

### Q: How do I handle training data that isn't available yet?
**A**: Use the **data strategy hierarchy**:
1. **Can't get full-fidelity data?** Use lower-fidelity alternatives (analytical, public)
2. **Geometry outside training range?** Use transfer learning or extrapolation with uncertainty flags
3. **Domain too large for Capytaine?** Coarsen domain, train on subset, tile results

Never skip training. Start with analytical + public data; add custom solvers only if needed.

[→ See Data Strategy and Level 2 (Capytaine) in Spec 02](../specs/02_DATA_STRATEGY.md#level-2-capytaine-bem-cost-1–10-min-per-case)

---

## Implementation & Code

### Q: Do I need GPU for training?
**A**: Recommended but not required:
- **GPU** (V100/A100): 1–2 hours F1A, 4–8 hours F1B
- **CPU only**: 10–20× slower, but feasible for Phase 1 testing

[→ See Computational Requirements in each operator spec](../specs/03_F1A_WSI_OPERATOR.md#computational-requirements)

---

### Q: How do I validate that my operator is correct?
**A**: Multi-level validation:

1. **Unit tests**: Forward pass shapes, grad flow, reproducibility
2. **Physics tests**: B ≥ 0, energy balance, dispersion relation
3. **Benchmark tests**: Compare against reference solver (WEC-Sim, OceanWave3D)
4. **Generalization**: Hold-out test set (unseen parameter combinations)

[→ See Validation sections in each operator spec](../specs/03_F1A_WSI_OPERATOR.md#validation-metrics)

---

### Q: How do I save and load a trained operator?
**A**: 
```python
# Save
torch.save(model.state_dict(), "checkpoints/model.pt")

# Load
model = DeepONetWEC(cfg)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()
```

All operators are PyTorch modules. Use standard PyTorch checkpointing.

[→ See Training Loop (Spec 09) Pseudocode section](../specs/09_TRAINING_LOOP.md#pseudocode)

---

## Uncertainty & Robustness

### Q: What's included for uncertainty quantification?
**A**: Phase 1 includes:

1. **Conformal Prediction** (primary): 95% coverage guarantee, no distributional assumptions
2. **Ensemble** (optional): 5-member operators, mean ± std
3. **Extrapolation flags**: Warns when parameters outside training range
4. **Solver fallback**: Recommends numerical solver when uncertainty high

[→ See Uncertainty Architecture (Spec 07)](../specs/07_UNCERTAINTY_ARCHITECTURE.md)

---

### Q: How do I know if my prediction is trustworthy?
**A**: Check these flags:

```python
if flags.extrapolation_warning:
    print("⚠️ Parameters outside training domain")
if flags.confidence_low:
    print("⚠️ High uncertainty in this region")
if flags.solver_fallback:
    print("🔧 Consider using numerical solver")
```

Operationally, use uncertainty thresholds to trigger solver fallback for critical decisions.

[→ See Decision Logic for Operations in Spec 07](../specs/07_UNCERTAINTY_ARCHITECTURE.md#decision-logic-for-operations)

---

## Contributing & Collaboration

### Q: How do I contribute code?
**A**: Follow **Test-Driven Development (TDD)**:
1. Write test RED (fails, expected)
2. Implement code
3. Test PASS (all green)
4. Commit with message linking spec + task

[→ See CONTRIBUTING.md](CONTRIBUTING.md)

---

### Q: How do specs relate to GitHub code?
**A**: Each spec has a "Related Files" section with GitHub paths:

```
Spec 03 (F1A WSI Operator)
  → GitHub: src/nossomar/operators/deeponet_wec.py
  → Tests: tests/test_deeponet_wec.py
  → Config: configs/f1a_deeponet.yaml
```

Implementation = turning spec into code.

---

### Q: What if I want to change a spec?
**A**: 
1. Document why (physics, performance, scope)
2. Update spec file
3. Link GitHub PR/commit
4. Update any dependent specs
5. Mark roadmap as "revision pending"

Changes should be rare after Phase 1; use Phase 2 planning if major shifts needed.

---

## Common Problems

### Q: My RMSE is 15% instead of target 5%. What's wrong?
**A**: Check in order:
1. **Data quality**: Are training samples representative? (Check with `test_wec_dataset.py`)
2. **Architecture**: Is model capacity enough? (Try wider layers, more epochs)
3. **Loss function**: Is physics loss too aggressive? (Start with supervised only)
4. **Hyperparams**: Learning rate too high/low? (Try 1e-3 for Adam)
5. **Generalization gap**: Is train RMSE also 15%? (Then underfitting; add data or capacity)

[→ See Troubleshooting in Spec 09](../specs/09_TRAINING_LOOP.md#troubleshooting)

---

### Q: Physics loss makes training diverge. What do I do?
**A**:
1. Start training with **supervised loss only** (λ_physics = 0)
2. After 50% epochs, add physics loss with **λ_physics = 0.01**
3. Gradually increase to 0.1 in final epochs
4. Always clip gradients (max_norm=1.0)

This is **curriculum learning**; it prevents optimization traps.

[→ See Physics Loss in Spec 09](../specs/09_TRAINING_LOOP.md#physics-informed-loss-optional-phase-1b)

---

### Q: My operator predicts negative damping (B < 0). How do I fix it?
**A**: Add physics-informed loss:
```python
L_b_positive = torch.mean(torch.relu(-B_pred))  # Penalty if B < 0
```

Or use output transformation:
```python
B_output = torch.nn.functional.softplus(B_raw)  # Always ≥ 0
```

[→ See Physics Loss in Spec 09](../specs/09_TRAINING_LOOP.md#physics-informed-loss-optional-phase-1b)

---

## When to Use NOSSO-MAR vs. Traditional Solver

| Use NOSSO-MAR | Use Traditional Solver |
|---------------|------------------------|
| Real-time forecasting | Offline design verification |
| Ensemble uncertainty (MC, DA) | Single high-fidelity run |
| Multi-scenario screening | Deep physics investigation |
| Layout optimization loop | New physics regime |
| Edge device inference | Debugging extrapolation |

---

## Future Directions

### Q: Can NOSSO-MAR do data assimilation?
**A**: Not yet. Phase 2 targets:
- Ensemble Kalman Filter (EnKF) + neural operators
- Inverse problems (estimate bathymetry from observations)
- 4D-Var assimilation with observations

[→ See Future Directions in Architecture.md](ARCHITECTURE.md#future-directions-phase-2)

---

### Q: Can I couple NOSSO-MAR with a GA for layout optimization?
**A**: Yes, Phase 1D is planned for this:
- Population-based layout search
- Neural operator as fast simulator in fitness function
- Objectives: power, robustness, separation, carryover

Currently in planning; Phase 2 target.

---

## Still Have Questions?

- **Technical**: Check relevant spec files (01–09)
- **Architecture**: See ARCHITECTURE.md
- **Contributing**: See CONTRIBUTING.md
- **GitHub repo**: https://github.com/FelipeTDuarte/NOSSO-Mar
- **Contact**: Felipe Duarte ([github profile](https://github.com/FelipeTDuarte))

---

**Last updated**: April 2026
**Maintained by**: NOSSO-MAR team
**Feedback**: Issues or PRs welcome on GitHub
