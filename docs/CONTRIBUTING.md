# Contributing to NOSSO-MAR

## Welcome!

We follow a **rigorous, test-driven workflow** with clear responsibility assignment via subagent task dispatch. This guide ensures consistency, reproducibility, and high-quality code.

---

## Workflow Overview

```
Spec (workspace) 
  ↓
Issue + Test RED (TDD)
  ↓
Implement (GitHub branch)
  ↓
Tests PASS (all green)
  ↓
PR Review → Merge
  ↓
Spec marked "Complete"
  ↓
Roadmap updated
```

---

## Step-by-Step Contribution Guide

### 1. Pick a Task

**Source**: [Phase 1 Roadmap](PHASE_1_ROADMAP.md)

Find a task marked:
- `→ IN PROGRESS` (claim it)
- `⏸️ BLOCKED` (understand blocker, comment)
- `⏳ NOT STARTED` (take it!)

Example task:
```
Task 2: Dataset Pipeline & Capytaine Integration → IN PROGRESS
Status: Expected completion Phase 1, Week 2
Artifacts: capytaine_runner.py, wec_dataset.py, data/phase1_wec_database.zarr/
Success Criteria: [checklist]
```

### 2. Read the Spec

Each task links to a specification file. **Must read before coding**.

Example: [Spec 08: Dataset Pipeline](../specs/08_DATASET_PIPELINE.md)

**What to extract**:
- Purpose & physical problem
- Inputs/outputs
- Architecture / design
- Validation metrics
- Code structure (in "Related Files" section)

### 3. Open an Issue (GitHub)

**Title**: `[Task N] Brief description`
**Example**: `[Task 2] Capytaine BEM sweep with LHS sampling`

**Body**:
```markdown
## Spec
- Spec 08: Dataset Pipeline
- Phase 1, Week 2

## Success Criteria
- [ ] 500 unique WEC cases generated (LHS)
- [ ] Capytaine runs in parallel (10 workers)
- [ ] Zarr dataset readable: >100 samples/s
- [ ] All tests pass

## Blockers
None currently

## Assignee
[Your name]

## Estimated Time
8 hours

## Notes
See [Phase 1 Roadmap](./docs/PHASE_1_ROADMAP.md#task-2-dataset-pipeline--capytaine-integration) for timeline.
```

### 4. Create a Branch

```bash
git checkout -b task-2-dataset-pipeline
```

**Branch naming**: `task-N-short-description` (kebab-case)

### 5. Write Tests First (RED)

**TDD rule**: Write test BEFORE implementation.

**Example** (`tests/test_wec_dataset.py`):
```python
import pytest
from nossomar.data import WECDataset

def test_wec_dataset_length():
    """Dataset should have correct number of samples."""
    db = WECDataset("data/phase1_wec_database.zarr/", split="train")
    assert len(db) == 350  # 70% of 500

def test_wec_dataset_batch_shape():
    """DataLoader batch should have expected shape."""
    from torch.utils.data import DataLoader
    db = WECDataset("data/phase1_wec_database.zarr/", split="train")
    loader = DataLoader(db, batch_size=32)
    batch = next(iter(loader))
    
    assert batch["A"].shape == (32, 100)  # 32 samples, 100 frequencies
    assert batch["B"].shape == (32, 100)
    assert batch["device_params"].shape == (32, 5)

def test_wec_dataset_zarr_readable():
    """Zarr dataset should be readable in parallel."""
    import zarr
    store = zarr.open_group("data/phase1_wec_database.zarr/train")
    
    # Check structure
    assert "A" in store
    assert "B" in store
    assert "device_params" in store
    
    # Check we can read chunks
    chunk = store["A"][0:32]  # Read first batch
    assert chunk.shape == (32, 100)
```

**Run test** (should FAIL):
```bash
pytest tests/test_wec_dataset.py::test_wec_dataset_length -v
# FAILED: FileNotFoundError: data/phase1_wec_database.zarr/
```

This is **expected RED**.

### 6. Implement Code

**File structure** (from spec's "Related Files"):
```
src/nossomar/data/
  ├── __init__.py
  ├── capytaine_runner.py      # LHS sweep + BEM calls
  ├── wec_dataset.py           # PyTorch Dataset API
  └── zarr_writer.py           # Serialization
```

**Example implementation** (`src/nossomar/data/wec_dataset.py`):
```python
import torch
import zarr
from torch.utils.data import Dataset

class WECDataset(Dataset):
    """PyTorch Dataset for WEC hydrodynamic database."""
    
    def __init__(self, zarr_path, split="train"):
        """
        Args:
            zarr_path: Path to Zarr store
            split: "train", "val", or "test"
        """
        self.store = zarr.open_group(f"{zarr_path}/{split}")
        self.n_samples = len(self.store["A"])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            "A": torch.from_numpy(self.store["A"][idx]).float(),
            "B": torch.from_numpy(self.store["B"][idx]).float(),
            "Fex_real": torch.from_numpy(self.store["Fex_real"][idx]).float(),
            "Fex_imag": torch.from_numpy(self.store["Fex_imag"][idx]).float(),
            "device_params": torch.from_numpy(self.store["device_params"][idx]).float(),
        }
```

### 7. Run Tests Again (GREEN)

```bash
pytest tests/test_wec_dataset.py -v
# PASSED: test_wec_dataset_length
# PASSED: test_wec_dataset_batch_shape
# PASSED: test_wec_dataset_zarr_readable
```

All tests should pass. If not, fix implementation.

### 8. Code Review Checklist

Before submitting PR, ensure:
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No physics violations (if applicable)
- [ ] Docstrings complete (numpy format)
- [ ] Type hints where possible
- [ ] Logging statements for key steps
- [ ] Config file if needed (YAML)
- [ ] Updated README / docs

### 9. Create Pull Request

**Title**: `[Task N] Brief description` (link issue #N)
**Example**: `[Task 2] Capytaine BEM sweep with LHS sampling (Closes #42)`

**Body**:
```markdown
## Description
Implements Capytaine-based hydrodynamic data generation for Phase 1.

## Related Spec
- Spec 08: Dataset Pipeline
- Section: "Stage 2: Data Sources, Level 2 (Capytaine BEM)"

## Changes
- `src/nossomar/data/capytaine_runner.py`: LHS sweep + parallel BEM
- `src/nossomar/data/wec_dataset.py`: PyTorch Dataset API
- `tests/test_wec_dataset.py`: 5 integration tests

## Validation
- ✓ All tests pass (5/5)
- ✓ Dataset size: 500 cases, 300 MB (compressed)
- ✓ I/O throughput: 150 samples/s (>100 target)
- ✓ No physics violations (A, B within bounds)

## Closes #42
```

### 10. Respond to Review

Maintainer reviews:
- Physics correctness
- Code quality
- Test coverage
- Documentation

**Address comments**:
```bash
git add .
git commit -m "Review: fix B damping bounds check"
git push
```

PR updates automatically.

### 11. Merge & Mark Complete

Once approved:
1. **Merge** PR (maintainer clicks button)
2. **Delete branch**: `git branch -d task-2-dataset-pipeline`
3. **Update roadmap**: Mark task ✓ COMPLETE in [Phase 1 Roadmap](PHASE_1_ROADMAP.md)
4. **Update spec**: Add "Last updated" note with commit hash

---

## Conventions

### Code Style
- **Python**: PEP 8 (use `black` formatter)
- **Type hints**: Annotate function signatures
- **Docstrings**: NumPy format (see example below)

**Example function**:
```python
def compute_wec_response(geometry_params: dict, env_params: dict) -> xr.Dataset:
    """
    Compute hydrodynamic response coefficients for a WEC.
    
    Parameters
    ----------
    geometry_params : dict
        Device parameters: {radius, draft, mass, bpto}
    env_params : dict
        Environment: {depth, rho, gravity}
    
    Returns
    -------
    wec_state : xr.Dataset
        Hydrodynamic coefficients A(ω), B(ω), Fex(ω)
    
    Examples
    --------
    >>> geom = {"radius": 5.0, "draft": 3.5, "mass": 331e3, "bpto": 1e5}
    >>> env = {"depth": 100, "rho": 1025, "gravity": 9.81}
    >>> wec = compute_wec_response(geom, env)
    >>> print(wec.A.shape)  # (100,) frequencies
    """
```

### Git Commit Messages

**Format**: `[Tag] Message (Fixes #N)`

**Tags**:
- `feat:` New feature / task
- `fix:` Bug fix
- `test:` Test addition / modification
- `docs:` Documentation only
- `refactor:` Code reorganization (no logic change)
- `chore:` Dependencies, config, cleanup

**Examples**:
```bash
git commit -m "feat: add Capytaine BEM sweep (Closes #42)"
git commit -m "test: add parametric validation for B(ω) >= 0"
git commit -m "docs: update README with usage example"
```

### Testing Requirements

**Minimum coverage per task**:
- ✓ Happy path (expected behavior)
- ✓ Edge cases (boundary values, empty input)
- ✓ Error handling (invalid inputs, file not found)
- ✓ Integration (across modules)

**Validation**:
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/nossomar --cov-report=html

# Specific test
pytest tests/test_wec_dataset.py::test_wec_dataset_zarr_readable -v
```

---

## Subagent Task Dispatch (Advanced)

For large tasks, use **subagent workflow** to parallelize implementation.

**Example**: Task 2 (Dataset Pipeline)

1. **Subagent 1**: Implement Capytaine wrapper + LHS sampling
2. **Subagent 2**: Implement Zarr serialization + PyTorch API
3. **Subagent 3**: Write all tests in parallel
4. **Lead**: Integrate & validate

**Workflow**:
```yaml
Task 2 - Dataset Pipeline:
  Subtask 2a (Capytaine):
    Subagent: "Developer 1"
    Files: capytaine_runner.py
    Tests: test_capytaine_runner.py
    Deadline: Wed EOD
  
  Subtask 2b (Zarr + API):
    Subagent: "Developer 2"
    Files: wec_dataset.py, zarr_writer.py
    Tests: test_wec_dataset.py
    Deadline: Wed EOD
  
  Subtask 2c (Integration):
    Subagent: "Lead"
    Merge, validate all, resolve conflicts
    Deadline: Thu AM
```

---

## Code Review Criteria

**As a reviewer, check**:

1. **Correctness**: Does code match spec? Are physics valid?
2. **Quality**: Is code readable, well-documented?
3. **Tests**: Do tests cover edge cases? Pass locally?
4. **Performance**: Are there bottlenecks? (Time complexity, memory)
5. **Maintainability**: Is code modular, reusable?

**Example review comment**:
```
Line 47: Added mass A should always be non-negative. 
Please add validation:
  assert (A >= 0).all(), "Added mass violation"
  
Also, consider adding to test suite to catch in future.
```

---

## Debugging Tips

### Test Fails Locally
```bash
# Run with verbose output
pytest tests/test_wec_dataset.py -vv

# Drop into debugger
pytest tests/test_wec_dataset.py --pdb
# Type 'c' to continue, 'l' to list, 'p var' to inspect

# Run single test
pytest tests/test_wec_dataset.py::test_specific_test -v
```

### Physics Violations
Check [Spec 02: Validation Rules](../specs/02_DATA_STRATEGY.md#validation-rules)
```python
# Example: damping must be ≥ 0
if (B < 0).any():
    print("Physics violation: negative damping detected")
    print(f"Min B = {B.min()}")
    raise ValueError("B must be >= 0")
```

### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size
python train_phase1.py --batch_size 16  # Instead of 32
```

---

## Troubleshooting Common Issues

| Problem | Solution |
|---------|----------|
| Tests fail: "File not found" | Check paths are absolute or relative to repo root |
| Physics loss diverges | Use curriculum learning (Spec 09, Loss Functions) |
| RMSE too high | Check data quality, model capacity, learning rate (Spec 09) |
| Merge conflicts | Resolve in Git: `git merge --abort`, discuss with team |
| Slow I/O | Use Zarr chunking, parallel DataLoader workers |

---

## When to Ask for Help

- **Spec unclear**: Comment on GitHub issue or ask maintainer
- **Physics question**: Check paper references in spec
- **Code stuck**: Share minimal reproducible example, ask in PR
- **Design decision**: Open discussion issue before implementing

---

## Resources

- **Python**: [PEP 8 Style](https://pep8.org/)
- **PyTorch**: [Official Tutorial](https://pytorch.org/tutorials/)
- **Testing**: [pytest docs](https://docs.pytest.org/)
- **Git**: [Pro Git Book](https://git-scm.com/book/en/v2)

---

## Code of Conduct

- ✓ Be respectful and constructive in code reviews
- ✓ Ask questions if spec is unclear
- ✓ Share knowledge (comment non-obvious code)
- ✓ Celebrate when tests pass! 🎉

---

## Questions?

- Check [FAQ.md](FAQ.md)
- See [Phase 1 Roadmap](PHASE_1_ROADMAP.md) for task assignments
- Open an issue on GitHub

**Last updated**: April 2026
