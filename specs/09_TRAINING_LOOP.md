# Specification 09: Training Loop and Review Loop

## Purpose

Define how NOSSO-MAR trains, evaluates and iteratively improves operator models across multiple fidelities.

This spec covers both:

- the numerical training loop;
- the engineering review loop that revises code, tests and datasets until no clear failure remains.

## Training Philosophy

NOSSO-MAR should not optimize one architecture in isolation.
It should compare operator families against the same task framing and physics criteria.

That is why the repository now includes:

- `src/nossomar/operators/factory.py`
- `src/nossomar/experiments/architecture_sweep.py`
- `scripts/smoke_operator_sweep.py`

## Core Loop

1. select task and fidelity target;
2. build the operator family;
3. prepare aligned inputs and targets;
4. compute supervised and physics losses;
5. update parameters;
6. validate on field, spectral and physics metrics;
7. checkpoint only when the model improves in meaningful ways.

## Loss Stack

### Supervised loss
- field or response mismatch

### Physics loss
- Navier-Stokes residuals where applicable
- shallow-water residuals for reduced propagation
- wave-action residuals for spectral alignment
- WEC equation-of-motion residuals for device response

### Cross-fidelity bridge loss
- spectral consistency from phase-resolved outputs
- CFD downgrade consistency
- response-summary consistency for WEC branches

### Constraint loss
- damping non-negativity
- bounded energy behavior
- boundary-condition compliance

Current residual utilities live in:
- `src/nossomar/physics/residuals_torch.py`

## Architecture Sweep Requirement

Before long training runs, each candidate architecture should pass a smoke sweep.

Current smoke sweep coverage:
- FNO2d
- WNO
- GNO
- DeepONet
- RINO2d

This is already executable through:

```bash
python scripts/smoke_operator_sweep.py
```

## Training Stages by Fidelity

### Stage A - Synthetic and local baseline
- use analytic generators
- harden interfaces
- validate loss implementation

### Stage B - Open-data conditioning
- add real forcing and benchmark context
- calibrate normalization and metadata handling

### Stage C - Reduced-physics task training
- train FNO/WNO/GNO on L1/L2 tasks
- train DeepONet or PI-DeepONet on L3 response tasks

### Stage D - High-fidelity bridge calibration
- align selected CFD and solver outputs with lower-fidelity representations
- use sparse but high-value L4 cases for regularization and final validation

## Review Loop

The project explicitly requires a repeat loop:

1. review code;
2. run tests;
3. fix failures;
4. re-run tests;
5. review again for obvious structural improvements;
6. stop only when no immediate high-confidence correction remains.

The local repository now supports that loop with:

- unit tests for baseline and new modules;
- executable scripts for architecture sweep and database build;
- deterministic analytic data generation.

## Minimum Acceptance Criteria

For any operator candidate:

- forward pass works on the target input shape;
- loss functions produce finite values;
- physics residuals do not immediately explode;
- validation metrics include at least one physics-aware measure;
- artifacts and configuration are reproducible.

## Current Status

Implemented now:

- baseline local training path for WEC response;
- architecture smoke sweep for multiple operator families;
- physics residual modules for future PINO-style training.

Not yet complete:

- full training harness for every operator family;
- regional production datasets;
- full multi-scenario calibration and validation campaign.

Those remain the next major execution phase rather than merely a design item.
