# Design Evolution: Thread Archive

This document indexes the evolution of NOSSO-MAR design across multiple chat sessions, preserving design decisions and rationale for future reference.

---

## Thread Timeline

### Thread v1: Initial Feasibility (NOSSO-Mar_thread_v1.md)

**Focus**: Is neural operator learning viable for wave-WEC coupled systems?

**Key Discussion Points**:
- FNO vs. DeepONet vs. GNO architectures
- Mesh independence (critical for layout optimization)
- Multi-body interaction representation

**Outcomes**:
- ✓ DeepONet chosen for F1A (branch/trunk decomposition, interpretable)
- ✓ GNO identified for multi-WEC arrays (natural graph representation)
- ✓ FNO noted as alternative if domain regular (e.g., flat bathymetry)

**Reference Solver Discussion**:
- SWASH: good but spectral (not phase-resolved for Phase 1)
- OceanWave3D: preferred for phase-resolved waves, NLSW equations
- Capytaine: BEM is gold standard for hydrodynamics, Python API ideal for automation
- WEC-Sim: validation baseline for devices

**Key insight**: Phase 1 should use analytical + public data first; solvers only for validation

---

### Thread v202604120: Architecture Refinement

**Focus**: Modular scientific software design + physics constraints

**Key Discussion Points**:
- Wetting/drying module (separate from core operator, explicitly coupled)
- Atmospheric forcing (wind stress boundary condition)
- Object interaction (floating bodies as source/sink terms)
- Coupling order (disciplined sequence to avoid physics ambiguity)

**Outcomes**:
- ✓ Proposed coupling sequence: wetting → forcing → objects → propagation → postprocessing
- ✓ Scientific modules sit outside core but explicitly integrated
- ✓ Uncertainty from day 1 (conformal intervals, ensemble, MC dropout)

**Architecture Layer Formalized**:
```
Layer 1: Contracts & IO (freeze types early)
Layer 2: Data Pipeline (analytical → public → BEM → solvers)
Layer 3: Core Operators (F1A, F1B, F1C)
Layer 4: Scientific Modules (wetting, forcing, objects)
Layer 5: Uncertainty (conformal + operational flags)
Layer 6: Training Infrastructure (TDD workflow)
```

---

### Thread v20260419: Operational & Digital Twin Focus

**Focus**: Path to operational deployment, digital twin architecture

**Key Discussion Points**:
- Uncertainty is operational necessity, not luxury feature
- Conformal prediction: mathematically rigorous coverage (95% → 95%)
- Solver fallback logic (when to trust operator vs. run solver)
- Digital twin: data assimilation (Phase 2), inverse problems (Phase 3)

**Outcomes**:
- ✓ Uncertainty output format: predictions + lower/upper bounds + flags
- ✓ Solver fallback triggered if: extrapolation OR confidence_low OR high uncertainty
- ✓ Phase 2 target: Ensemble Kalman Filter + neural operators
- ✓ Phase 3 target: 4D-Var inversion with observations

**Critical Decision**: Phase 1 scope narrowed to single end-to-end cycle (F1A + F1B baseline), deferring F1C coupling & F1D optimization to Phase 2

---

### fulloceanPINO_thread.md: Comprehensive Reference Landscape

**Focus**: Neural operator ecosystem for ocean modeling

**Key Architectures Referenced**:
1. **FNO (Li et al. 2021)**: Fourier spectral operators, efficient, handles periodicity
2. **DeepONet (Lu et al. 2021)**: Mesh-independent, branch/trunk decomposition
3. **Latent DeepONet (Katiana)**: Compression layer for high-dimensional inputs
4. **Transfer Learning DeepONet (TL-DeepONet)**: Pre-training on low-fidelity, fine-tune on sparse high-fidelity
5. **PI-RINO (arXiv:2510.23810)**: Resolution-independent encoder + physics residuals
6. **WNO (Wavelet)**: Multi-scale phenomena, better for breaking zones
7. **GNO (Graph)**: Irregular geometries, natural for multi-body arrays
8. **FourCastNetv2**: Spherical harmonics for Earth-scale equivariance

**Key Insight (RINO)**: FNOs struggle with variable resolution input. **RINO solves via resolution-independent encoding**. This is critical for NOSSO-MAR Phase 2+ where bathymetry/observation resolution varies.

**Data Sources Discussed**:
- CMEMS: Global ocean model output (free)
- WEC-Sim: NREL benchmark cases (validated)
- FlowBench: Standardized PDE benchmarks
- HuggingFace: PDE inverse problems dataset

**Coupling Concepts**:
- Iterative coupling: Wave → WEC → radiation → iterate (stable, interpretable)
- End-to-end coupling: Joint operator (faster, harder to debug)
- Physics-informed losses: Enforce Navier-Stokes residuals (curriculum needed)

**Takeaway for NOSSO-MAR Phase 2**: RINO encoder should be explored for improved multi-resolution handling

---

## Design Principles Crystallized

### From Evolution (v1 → v202604120 → v20260419)

1. **Physics First**: Neural operators learn from governed data, not noise
   - Analytical solutions as ground truth
   - Public datasets reduce solver cost
   - Solvers = offline tools, not runtime

2. **Modular Architecture**: Scientific modules separate from core
   - Wetting/drying, forcing, objects outside operator
   - Explicit coupling order prevents ambiguity
   - Future modules plug in without rewriting

3. **Uncertainty Operational**: UQ is not afterthought
   - Conformal intervals from day 1
   - Flags for extrapolation & confidence
   - Solver fallback when uncertain

4. **Minimize Data Cost**: Hierarchy beats brute-force
   - Analytical → Public → Custom BEM → Full solvers
   - Transfer learning bridges domains
   - 90% training data free or quasi-free

5. **Reproducibility & Testing**: TDD from start
   - Every feature has test
   - Physics violations caught early
   - Generalization verified on hold-out test set

---

## Decision Log

### "Should Phase 1 include coupling (F1C)?"

**Decision**: Optional, defer to Phase 2 if time tight

**Rationale**:
- F1A + F1B standalone already enable fast simulation
- Coupling adds complexity (need radiation field model or full re-training)
- Iterative coupling works as protoproof (not fast but correct)
- Phase 2 high priority for end-to-end coupling for layout optimization

---

### "Should we train on real observations or solver data?"

**Decision**: Phase 1 uses solver data only

**Rationale**:
- Real data sparse and sparse (few buoys)
- Solver data abundant, consistent, controllable (LHS sampling)
- Phase 2 → data assimilation (blend solver predictions + observations)
- Phase 1 goal: prove neural operator learns physics, not operationalize yet

---

### "Which neural operator architecture?"

**Decision**: DeepONet for F1A, GINO for F1B

**Rationale**:
- **DeepONet F1A**: Interpretable branch/trunk, frequency-continuous queries, proven on hydrodynamics
- **GINO F1B**: Handles irregular bathymetry/coastlines naturally, message passing captures wave propagation, flexible domain size
- **Alternative FNO for F1B if domain regular**: Faster, multi-scale via FFT, but poor on boundaries

---

### "How to structure code?"

**Decision**: `src/nossomar/` with modular packages

**Rationale**:
- `core/`: contracts, types, base classes
- `data/`: data generation, validation, I/O
- `operators/`: F1A, F1B, neural operator implementations
- `modules/`: wetting, forcing, objects, scientific layers
- `training/`: training loops, losses, optimization
- `uncertainty/`: conformal, ensemble, UQ utilities
- `coupling/`: F1C iterative/end-to-end
- `scripts/`: entry points, CLI

---

## Future Design Conversations

### Phase 2 Topics Identified

1. **Inverse Problems**: Infer bathymetry / friction from observations
2. **Layout Optimization**: GA + neural solver (100× speedup target)
3. **Data Assimilation**: EnKF, 4D-Var with NOSSO-MAR
4. **Extended Physics**: Wetting/drying, sediment, breaking
5. **Multi-resolution**: RINO encoder for variable input resolution

---

## How to Use This Archive

- **Understand design rationale**: Read relevant thread section + reasoning
- **Avoid repeating debates**: Search thread for past decision, see why it was made
- **Propose improvements**: Link back to thread, explain why change is needed
- **Onboard new people**: Point to threads showing evolution of thinking

---

## Linking to Threads

**Workspace files**:
- `NOSSO-Mar_thread_v1.md` — Initial architecture feasibility
- `NOSSO-Mar_thread_v202604120.md` — Modular design + uncertainty
- `NOSSO-Mar_thread_v20260419.md` — Operational digital twin
- `fulloceanPINO_thread.md` — Comprehensive reference landscape

**GitHub**:
- Each spec notes "See THREADS_ARCHIVE.md" if design history relevant
- Commits link back to specs + threads for traceability

---

## Metadata

- **Archive created**: April 28, 2026
- **Threads covered**: v1, v202604120, v20260419, fulloceanPINO
- **Design phase**: End of Phase 0, ready for Phase 1 implementation
- **Next review**: After Phase 1 completion (5–6 weeks)
- **Maintainer**: Felipe Duarte (NOSSO-MAR lead)

---

**Key takeaway**: NOSSO-MAR design is not accidental. Each architectural choice has rationale rooted in ocean physics, neural operator theory, and operational constraints. Preserve this reasoning as we evolve.
