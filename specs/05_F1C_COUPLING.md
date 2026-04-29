# Specification 05: F1C - Bidirectional Coupling (Wave ↔ WEC Interaction)

## Purpose
Integrate F1A (WSI operator) and F1B (wave field operator) into a single differentiable system where wave field influences device response and device response modifies the wave field. This is the first end-to-end system for coupled wave-WEC prediction.

## Scope
- **Phase(s)**: Phase 1C (Task 6, optional in Phase 1 but critical for Phase 2)
- **Status**: Architecture design → Prototype → Full implementation
- **Dependencies**: F1A (Spec 03), F1B (Spec 04), IO Contracts (Spec 01)

---

## Coupling Architecture

### Decoupled Baseline (Phase 1A-B Independent)
```
F1B (Wave Operator):
  η(x, y, t), u(x, y, t), v(x, y, t) ← incident wave only
  
F1A (WSI Operator):
  A(ω), B(ω), Fex(ω) ← local wave elevation at device location
  P_abs ← computed from RAO and spectrum
  
Physical limitation: No feedback — array doesn't affect incident field
```

### Iterative Coupling (Phase 1C Option 1)
```
Iteration 0:
  F1B: compute η_0(x, y, t) from incident boundary
  At device locations: η_dev(t) ← sample from η_0
  
Iteration k (k=1, 2, ...):
  F1A: use η_dev^{k-1} → compute forces, motion, power
  Radiation wave from device: η_rad^k ← from device response
  F1B: compute η_k(x, y, t) from (incident + radiation)
  At device: η_dev^k ← sample from η_k
  
Until convergence: |η_dev^k - η_dev^{k-1}| < tolerance
```

**Advantages**: Physically interpretable, stable
**Disadvantages**: Slow (multiple F1B calls per scenario)

### End-to-End Coupling (Phase 1C Option 2, Target)
```
Single neural operator trained jointly:

Input: [S(ω), heading, h(x,y), layout, device_params]
       
Learned coupling:
  Internal state evolves with explicit wave-device interaction terms
  F1A and F1B gradients flow together during backprop
  
Output: [η(x, y, t), u(x, y, t), v(x, y, t), power_per_device(t)]

Advantage: Single forward pass, faster, physics-aware gradients
Disadvantage: More complex training, harder to debug
```

---

## Data Requirements for Coupling

**Coupled simulation data** (from reference solvers):
- Run OceanWave3D + WEC-Sim in sequence or full coupled solver (if available)
- 50–100 cases with varying:
  - Spectrum type (Pierson-Moskowitz, JONSWAP, bimodal)
  - Hs: 1–3 m
  - Tp: 6–12 s
  - Array layout: 1 WEC, 2-body, 3-body
  - Depth: 30–100 m

**Data structure**:
```
For each coupled case:
  Input:  S(ω), heading, h(x,y), layout, device params
  Output: η(x, y, t) with device effects
          power_absorbed_per_device(t)
          force_per_device(t)
          
Validation metrics:
  - Total energy balance
  - Power extraction should reduce wave height behind array
  - Radiation damping should show in frequency response
```

---

## Loss Functions for Coupling

### Supervised Loss
```
L_sup = MSE(η_coupled_pred, η_coupled_ref) 
      + MSE(u_pred, u_ref)
      + α · MSE(power_pred, power_ref)
where α = 0.5
```

### Physics Loss: Energy Balance
```
L_energy = ||∇·(energy_flux_pred) + power_absorbed||²

Enforces: wave energy is balanced by power extraction
```

### Physics Loss: Causality
```
L_causality: no output before input forcing
            (enforced by architecture if using causal models)
```

### Total Loss
```
L_total = L_sup + λ_energy · L_energy
λ_energy = 0.01 (weak initially, strengthen after L_sup converges)
```

---

## Implementation Strategy

### Phase 1C Option A: Iterative Coupling (Prototype)
```python
# Pseudocode
class IterativeCoupledOperator:
    def __init__(self, f1b_operator, f1a_operator, max_iter=5):
        self.f1b = f1b_operator
        self.f1a = f1a_operator
        self.max_iter = max_iter
    
    def forward(self, incident_spectrum, device_layout, bathymetry):
        # Iteration 0: incident wave field only
        eta_incident = self.f1b(incident_spectrum, bathymetry)
        
        for k in range(self.max_iter):
            # Sample wave at device locations
            eta_at_dev = eta_incident[device_layout.x, device_layout.y]
            
            # Compute device response
            forces, power, motion = self.f1a(eta_at_dev, device_layout)
            
            # Compute radiation field (analytical or learnable)
            eta_radiation = self._compute_radiation(motion, device_layout)
            
            # Update total field
            eta_total = eta_incident + eta_radiation
            
            # Convergence check
            if torch.norm(eta_total - eta_incident) < tolerance:
                break
            
            eta_incident = eta_total
        
        return eta_total, forces, power
```

### Phase 1C Option B: End-to-End (Full Implementation)
```python
class CoupledPINOOperator(nn.Module):
    def __init__(self):
        self.encoder = ResNet(...)  # Encode input (spectrum, bathymetry, layout)
        self.coupling_layers = nn.ModuleList([...])  # Joint F1A-F1B blocks
        self.decoder_wave = MLP(...)  # Decode to η, u, v
        self.decoder_wec = MLP(...)   # Decode to power, forces
    
    def forward(self, spectrum, bathymetry, device_layout):
        # Joint evolution with coupling
        z = self.encoder(spectrum, bathymetry, device_layout)
        
        for layer in self.coupling_layers:
            z = layer(z)  # Implicit coupling through shared embedding
        
        eta = self.decoder_wave(z)
        power = self.decoder_wec(z)
        
        return eta, power
```

---

## Validation Plan

**Benchmark 1: Single WEC, Shallow Water**
- Domain: 2 km × 2 km, 50 m depth, flat bathymetry
- Device: RM3 equivalent (5 m radius cylinder)
- Compare against coupled OceanWave3D + WEC-Sim
- Success: Power prediction within 10%, wake effects visible

**Benchmark 2: 2-Body Array**
- Same domain, 2 WECs separated by 100 m
- Measure: hydrodynamic interaction (power reduction due to wake)
- Success: Wake effect captured (power drop 5–15% for 2nd WEC)

**Benchmark 3: Real-World Domain**
- Portuguese coast nearshore (~5 km × 5 km)
- Variable bathymetry (GEBCO)
- 3-body array
- Validate against SWAN + WEC-Sim ensemble

---

## Roadmap for Phase 1C

**Stage 1 (Phase 1, Optional)**: Iterative coupling prototype
```
- Implement IterativeCoupledOperator (500 lines of code)
- Train F1A and F1B separately
- Test on 1 WEC case: does power drop after 1 iteration?
- Publish as tech report
```

**Stage 2 (Phase 2): End-to-End Learning**
```
- Collect 100 fully coupled training cases
- Implement CoupledPINOOperator
- Train end-to-end with L_sup + L_energy
- Validate on array (2–5 bodies)
- Publish main paper
```

---

## Computational Overhead

| Approach | Forward Time | Memory | Training Time |
|----------|--------------|--------|---------------|
| Decoupled (F1A + F1B separate) | 100 ms | 8 GB | 12 h |
| Iterative coupling (5 iterations) | 500 ms | 8 GB | N/A (no training) |
| End-to-End joint | 100 ms | 12 GB | 24 h |

**Recommendation**: Phase 1 uses decoupled + iterative coupling for validation; Phase 2 implements end-to-end for speed and accuracy.

---

## Key Design Decisions

1. **Causality enforcement**: Use only past/current wave info to predict device motion (acausal feedback breaks physics)

2. **Radiation condition**: For Phase 1C, model radiation analytically or via simple regression; in Phase 2, learn from data

3. **Frequency-domain vs time-domain**:
   - Phase 1A produces frequency-domain coefficients (A, B, Fex)
   - Phase 1B produces time-domain field (η, u, v)
   - Coupling bridges via inverse Fourier: Fex(ω) ↔ force(t)

4. **Multi-scale**: Device (~10 m) vs domain (km) — may need multi-resolution grids

---

## Related Files

- **GitHub**:
  - `src/nossomar/coupling/coupled_pino.py` (iterative or end-to-end operator)
  - `src/nossomar/coupling/radiation_model.py` (radiation field computation)
  - `src/nossomar/training/train_coupled.py` (joint training loop)
- **Tests**: `tests/test_coupled_operator.py`
- **Config**: `configs/f1c_coupling.yaml`
- **Scripts**: `scripts/validate_coupling.py` (benchmark validation)

---

## Expected Outcomes

**Phase 1C (if completed)**:
- ✓ Iterative coupling validates core concept
- ✓ Coupling loss metrics show physics learning
- ✓ Wake effects visible in test cases

**Phase 2+**:
- End-to-end operator enables layout optimization with neural solver
- Full digital twin with data assimilation
