# Specification 07: Uncertainty Architecture

## Purpose
Design a modular uncertainty quantification (UQ) layer that produces confidence intervals, spread estimates, and decision logic for operational deployment. Keep Phase 1 modest; design for Phase 2/3 expansion.

## Scope
- **Phase(s)**: Phase 1 (basic implementation) → Phase 2 (operational)
- **Status**: Architecture design → Prototype
- **Dependencies**: IO Contracts (Spec 01), Neural Operators (Specs 03–04)

---

## Core Uncertainty Methods

### 1. **Ensemble Approach (Phase 1 Baseline)**

**Concept**: Train N independent neural operators with different random seeds / data subsets.

**Implementation**:
```python
class EnsembleOperator:
    def __init__(self, n_members=5):
        self.members = [DeepONetWEC() for _ in range(n_members)]
    
    def forward(self, x):
        predictions = [member(x) for member in self.members]
        mean = torch.stack(predictions).mean(dim=0)
        std = torch.stack(predictions).std(dim=0)
        return mean, std
```

**Advantages**:
- Simple to implement
- Captures aleatoric and model-form uncertainty
- Parallelizable

**Disadvantages**:
- N× computational cost at inference
- Requires N× training

**Phase 1 target**: 5-member ensemble for F1A operator

---

### 2. **Monte Carlo Dropout (Phase 1 Extension)**

**Concept**: Use dropout at test time to sample from posterior distribution.

**Implementation**:
```python
class MCDropoutOperator(nn.Module):
    def __init__(self, dropout_rate=0.2):
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 4)
        )
    
    def forward(self, x, n_samples=50):
        self.net.train()  # Keep dropout active at test time
        samples = [self.net(x) for _ in range(n_samples)]
        samples = torch.stack(samples)
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std
```

**Advantages**:
- Minimal overhead (just forward passes with dropout on)
- Bayesian interpretation
- Fast inference

**Disadvantages**:
- Requires retraining with specific dropout placement
- May underestimate uncertainty if dropout too aggressive

**Phase 1 target**: Optional; if time permits

---

### 3. **Conformal Calibration (Phase 1 Focus)**

**Concept**: Learn prediction intervals on held-out validation set, guarantee coverage without distributional assumptions.

**Theory**:
```
For a test point x:
  - Predict ŷ ± margin
  - margin computed from quantile of |ŷ - y_true| on validation set
  - Guarantees (1 - α) coverage (e.g., 95% of future test points in interval)
```

**Implementation**:
```python
class ConformalOperator:
    def __init__(self, operator, val_residuals, alpha=0.1):
        """
        Args:
            operator: trained neural operator
            val_residuals: |predictions - truth| on validation set (shape: N_val,)
            alpha: miscoverage rate (0.1 → 90% coverage)
        """
        self.operator = operator
        self.quantile = torch.quantile(val_residuals, 1 - alpha)
    
    def forward(self, x):
        mean = self.operator(x)
        lower = mean - self.quantile
        upper = mean + self.quantile
        return mean, lower, upper
```

**Advantages**:
- Mathematically rigorous coverage guarantee
- Distribution-free (no normality assumption)
- Easy to tune (α parameter)

**Disadvantages**:
- Requires held-out validation set
- Intervals may be conservative

**Phase 1 target**: Primary UQ method; apply to all operators

---

## Uncertainty Output Format

**Extended OceanState with uncertainty**:
```python
OceanStateWithUncertainty = {
    "prediction": {
        "eta": xr.DataArray,       # Mean prediction
        "u": xr.DataArray,
        "v": xr.DataArray,
        "power": xr.DataArray,
    },
    "uncertainty": {
        "eta_lower": xr.DataArray, # 95% lower confidence bound
        "eta_upper": xr.DataArray, # 95% upper confidence bound
        "eta_std": xr.DataArray,   # Standard deviation (ensemble)
        "power_lower": xr.DataArray,
        "power_upper": xr.DataArray,
        "confidence": 0.95,        # Coverage probability
    },
    "flags": {
        "extrapolation_warning": bool,  # Outside training parameter range?
        "confidence_low": bool,         # Is uncertainty high in region of interest?
        "solver_fallback": bool,        # Recommend numerical solver?
    }
}
```

---

## Decision Logic for Operations

**Automated fallback to numerical solver**:
```python
def should_use_solver_fallback(prediction, flags):
    """
    Decide whether to trust neural operator or fall back to numerical solver.
    """
    return (
        flags["extrapolation_warning"] or  # Beyond training domain
        flags["confidence_low"] or         # High uncertainty
        prediction.uncertainty["power_std"] > 0.3 * prediction.mean["power"]  # >30% uncertainty
    )
```

**Example operational workflow**:
```
1. Run neural operator with uncertainty
2. Check flags:
   - Extrapolation? → Mark for review
   - Low confidence? → Run solver in parallel, blend results
   - Unstable? → Use solver result
3. Output to digital twin with confidence metadata
```

---

## Calibration & Validation

### Calibration Data Requirements
```
Validation set: 100–200 held-out test cases not used in training
```

### Calibration Metrics
```
Coverage:
  - Compute |ŷ - y_true| for each validation sample
  - Check if (y_true ∈ [ŷ - margin, ŷ + margin]) for 95% of samples
  
Efficiency (avoid overly wide intervals):
  - Average interval width
  - Target: balance coverage vs. tight intervals
```

### Examples of Good vs Bad Calibration
```
Good:
  - 95% of test points fall within predicted 95% interval
  - Interval width varies with input complexity (wider for uncertain regions)
  
Bad:
  - 60% coverage (too narrow)
  - 99% coverage (too wide, uninformative)
  - Interval width constant regardless of input (not adaptive)
```

---

## Implementation Plan

### Phase 1 Minimal (Week 1–2)
```
1. Train base operators (F1A, F1B) without UQ
2. Compute residuals on validation set
3. Conformal quantile: Q_95 = quantile(residuals, 0.95)
4. Output: mean ± Q_95 interval
5. Simple flags: extrapolation check (param range check)
```

### Phase 1 Extended (Week 3, if time)
```
1. Train 5-member ensemble for F1A
2. MC Dropout on F1B (optional)
3. Compare ensemble vs conformal coverage
4. Publish comparison: which UQ method best?
```

### Phase 2 (Future)
```
1. Full Bayesian inference (variational)
2. Learned uncertainty (aleatoric vs epistemic)
3. Inverse problem inference (parameter estimation from observations)
4. Data assimilation (Kalman filter with neural operators)
```

---

## Related Files

- **Core UQ module**: `src/nossomar/uncertainty/conformal.py`
- **Ensemble**: `src/nossomar/uncertainty/ensemble.py`
- **MC Dropout**: `src/nossomar/uncertainty/mc_dropout.py`
- **Tests**: `tests/test_uncertainty.py`
- **Config**: `configs/uncertainty.yaml`

---

## Uncertainty Reporting Standards

**All papers/reports should include**:
```
1. Uncertainty method (ensemble / conformal / MC dropout)
2. Calibration metrics (coverage, interval width)
3. Computational cost (inference time, memory for UQ)
4. Failure modes documented (where uncertainty large, why?)
5. Recommendations for operational use (when to trust, when to fall back)
```

---

## Expected Outcomes

**Phase 1**:
- ✓ Conformal intervals with >90% coverage
- ✓ Interval width < 15% of signal amplitude
- ✓ Flags detect extrapolation (param bounds violated)

**Phase 2**:
- ✓ Ensemble comparison paper
- ✓ Operational digital twin with uncertainty-aware decisions
- ✓ Inverse inference (bathymetry from uncertain observations)

---

## Future Extensions (Phase 3+)

- **Learned aleatoric uncertainty**: Network outputs σ(x) naturally
- **Epistemic uncertainty via deep ensembles**: Track learning curves
- **Probabilistic coupling**: Covariance between F1A and F1B outputs
- **Inverse UQ**: Infer uncertain parameters (friction, Bpto) from observations
