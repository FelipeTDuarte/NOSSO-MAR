# NOSSO-MAR Workspace Organization Guide

## File Structure Overview

```
Nosso-Mar/
├── README_WORKSPACE.md           ← START HERE: Workspace index & navigation
│
├── 00_MASTER_PLAN.ipynb         ← Single canonical project notebook
│   (consolidates both old notebooks)
│
├── THREADS_ARCHIVE.md           ← Indexed evolution of design discussions
│
├── specs/                        ← TECHNICAL SPECIFICATIONS (9 files)
│   ├── 01_IO_CONTRACTS.md                  [Data structures & IO formats]
│   ├── 02_DATA_STRATEGY.md                 [Training data hierarchy]
│   ├── 03_F1A_WSI_OPERATOR.md              [Wave-structure interaction]
│   ├── 04_F1B_WAVE_FIELD_OPERATOR.md       [Phase-resolved propagation]
│   ├── 05_F1C_COUPLING.md                  [Bidirectional coupling]
│   ├── 06_SCIENTIFIC_MODULES.md            [Wetting, forcing, objects]
│   ├── 07_UNCERTAINTY_ARCHITECTURE.md      [UQ strategy]
│   ├── 08_DATASET_PIPELINE.md              [Data generation workflow]
│   └── 09_TRAINING_LOOP.md                 [Training infrastructure]
│
├── docs/                         ← PROJECT DOCUMENTATION
│   ├── ARCHITECTURE.md                     [System design, layers, coupling]
│   ├── CONTRIBUTING.md                     [TDD workflow, subagent tasks]
│   ├── FAQ.md                              [Common questions]
│   ├── PHASE_1_BASELINE.md                 [Local executable baseline]
│   └── PHASE_1_ROADMAP.md                  [7-task checklist]
│
├── src/nossomar/                 ← LOCAL PHASE 1 BASELINE CODE
├── configs/                      ← Local dataset + training configs
├── scripts/                      ← Local entry points
├── tests/                        ← Local regression tests
├── data/                         ← Generated local baseline dataset
├── checkpoints/                  ← Saved local model + metrics
│
├── _archive/                     ← Old versions (reference only)
│   ├── Nosso-Mar_project_v1.ipynb_archive/
│   ├── Nosso-Mar_plano_definitivo_v1.ipynb_archive/
│   └── README_ARCHIVE.md
│
└── .sixth/                       ← Existing cache (leave as is)
```

---

## Quick Navigation by Role

### 🔬 If you're a researcher understanding the physics:
1. Start: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system overview
2. Deep dive: [specs/03_F1A_WSI_OPERATOR.md](specs/03_F1A_WSI_OPERATOR.md), [specs/04_F1B_WAVE_FIELD_OPERATOR.md](specs/04_F1B_WAVE_FIELD_OPERATOR.md)
3. Coupling: [specs/05_F1C_COUPLING.md](specs/05_F1C_COUPLING.md)

### 💻 If you're a developer implementing code:
1. Start: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — understand system layers
2. Follow specs in order (01 → 09) for implementation roadmap
3. Code structure: use the local baseline in `src/nossomar/`; check the [GitHub repo](https://github.com/FelipeTDuarte/NOSSO-Mar) for the broader implementation target
4. Workflow: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — TDD + subagent task dispatch

### 📋 If you're a project manager tracking progress:
1. Current status: [docs/PHASE_1_ROADMAP.md](docs/PHASE_1_ROADMAP.md) — 7-task checklist
2. Timeline: Each spec file has its own milestone/status
3. GitHub issues/PRs link back to spec details

### 🎓 If you're new to the project:
1. Read: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (5 min overview)
2. FAQ: [docs/FAQ.md](docs/FAQ.md) (common Q&A)
3. Then dive into specs that match your interest

---

## File Purposes (What to Read When)

| File | Purpose | Audience | Read Time |
|------|---------|----------|-----------|
| `00_MASTER_PLAN.ipynb` | Phases, tasks, high-level roadmap | Everyone | 30 min |
| `specs/01_IO_CONTRACTS.md` | Data structures, tensor conventions | Developers | 20 min |
| `specs/02_DATA_STRATEGY.md` | Training data sources & hierarchy | Data engineers | 20 min |
| `specs/03_F1A_WSI_OPERATOR.md` | WEC hydrodynamic operator design | Researchers + Devs | 25 min |
| `specs/04_F1B_WAVE_FIELD_OPERATOR.md` | Wave propagation operator design | Researchers + Devs | 25 min |
| `specs/05_F1C_COUPLING.md` | Wave-WEC feedback coupling | Advanced | 20 min |
| `specs/06_SCIENTIFIC_MODULES.md` | Wetting, forcing, object physics | Domain experts | 20 min |
| `specs/07_UNCERTAINTY_ARCHITECTURE.md` | UQ strategy, confidence intervals | ML engineers | 20 min |
| `specs/08_DATASET_PIPELINE.md` | Data generation workflow | Data engineers | 25 min |
| `specs/09_TRAINING_LOOP.md` | Training loop, losses, optimization | ML engineers | 30 min |
| `docs/ARCHITECTURE.md` | System design overview & layers | Everyone | 15 min |
| `docs/CONTRIBUTING.md` | How to contribute code | Developers | 15 min |
| `docs/PHASE_1_ROADMAP.md` | Current progress checklist | Managers | 10 min |
| `docs/FAQ.md` | Common questions answered | Everyone | 10 min |
| `THREADS_ARCHIVE.md` | Evolution of design decisions | Historians | 15 min |

---

## Relationship to GitHub Repository

**Workspace** (this folder) = Planning + Specifications + Local Baseline
**GitHub repo** = Wider Implementation + Integration Target

**Flow**:
```
Spec (workspace) → Local Baseline → GitHub Issue/PR → Full Implementation → Tests Pass → Merged
```

Each specification file has a "Related Files" section linking to GitHub code.

Example: [Spec 03: F1A WSI Operator](specs/03_F1A_WSI_OPERATOR.md#related-files)
```
GitHub:
  - src/nossomar/operators/deeponet_wec.py
  - tests/test_deeponet_wec.py
  - configs/f1a_deeponet.yaml
```

---

## How Specifications Work

### Structure of Each Spec
1. **Purpose**: Why this module exists
2. **Scope**: Phase, status, dependencies
3. **Technical Details**: Algorithms, data contracts, physics
4. **Implementation Roadmap**: Step-by-step tasks
5. **Validation**: How to test it
6. **Related Files**: Links to GitHub code + configs

### Example: "How do I implement F1A (WSI Operator)?"
1. Read [Spec 03: F1A WSI Operator](specs/03_F1A_WSI_OPERATOR.md)
2. Understand the DeepONet architecture (Section 1)
3. See validation metrics (Section 5)
4. Check GitHub files (Related Files section)
5. Follow "Implementation Roadmap" (Task 3–4)
6. Write tests (TDD workflow in [CONTRIBUTING.md](docs/CONTRIBUTING.md))

---

## Key Design Decisions (Why Things Are Organized This Way)

### 1. **One Master Notebook** (not two duplicates)
- Single source of truth for phases, tasks, roadmap
- Reduces inconsistency and confusion
- Old notebooks archived with version history

### 2. **9 Specification Files** (not one giant spec)
- Modular: each spec is standalone readable (~20 min per file)
- Versionable: each spec can be updated independently
- Linked: cross-references connect related specs

### 3. **Separate Specs + Docs**
- `specs/` = Technical specifications (what to build)
- `docs/` = Project documentation (how to work together)

### 4. **Architecture.md at Top Level**
- Connects all 9 specs into one coherent system picture
- Entry point for new people

### 5. **Workspace + GitHub Sync**
- Workspace = planning, specs, and a small runnable baseline
- GitHub = implementation, integrations, and larger-scale execution
- Both need to stay in sync (CONTRIBUTING.md describes this)

---

## Maintenance & Updates

### When to Update This Workspace:

1. **After Spec Implementation**:
   - Update [Phase 1 Roadmap](docs/PHASE_1_ROADMAP.md) checklist
   - Mark spec as "Implemented" (not just "Designed")

2. **When Specification Changes**:
   - Update the relevant spec file (e.g., change loss function in Spec 09)
   - Link GitHub PR/commit in spec file
   - Update any dependent specs

3. **New Findings / Learnings**:
   - Add to [FAQ.md](docs/FAQ.md)
   - Update spec if design changes
   - Create session memory note for future reference

### Who Maintains This?
- **Lead**: Project coordinator (Felipe)
- **Contributors**: Anyone can suggest updates via PR or issue
- **Process**: Update spec → Link to GitHub → Update roadmap

---

## Getting Started Checklist

If this is your first time:
- [ ] Read this file (you're doing it!)
- [ ] Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [ ] Browse [docs/FAQ.md](docs/FAQ.md)
- [ ] Open [00_MASTER_PLAN.ipynb](00_MASTER_PLAN.ipynb) in Jupyter
- [ ] Pick one spec related to your role (see table above)
- [ ] Run `python -m pytest -q` to verify the local baseline
- [ ] Check [GitHub repo](https://github.com/FelipeTDuarte/NOSSO-Mar) to see actual code
- [ ] Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) if contributing

---

## FAQ: "Where do I find X?"

| Question | Answer |
|----------|--------|
| "What are the phases?" | [00_MASTER_PLAN.ipynb](00_MASTER_PLAN.ipynb) or [Spec 01](specs/01_IO_CONTRACTS.md#scope) |
| "How do neural operators work?" | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#layer-3-core-neural-operators) + [fulloceanPINO_thread.md](../fulloceanPINO_thread.md) |
| "What's the training data strategy?" | [Spec 02](specs/02_DATA_STRATEGY.md) + [Spec 08](specs/08_DATASET_PIPELINE.md) |
| "How do I run the code?" | Local: `python scripts/generate_phase1_dataset.py` then `python scripts/train_phase1.py`; broader target: [GitHub README](https://github.com/FelipeTDuarte/NOSSO-Mar#usage) |
| "What's the current progress?" | [docs/PHASE_1_ROADMAP.md](docs/PHASE_1_ROADMAP.md) |
| "How do I contribute?" | [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) |
| "Is there uncertainty quantification?" | [Spec 07](specs/07_UNCERTAINTY_ARCHITECTURE.md) |
| "How are modules coupled?" | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#coupling-order--information-flow) + [Spec 06](specs/06_SCIENTIFIC_MODULES.md) |

---

## Document Versioning

- **Workspace files**: Updated as design evolves (tracked in workspace history)
- **GitHub code**: Versioned by git tags (e.g., `v0.1.0`, `phase-1-complete`)
- **Specs**: Linked to GitHub commits (each spec notes "Last updated: PR #123")

---

## Feedback & Contributions

Found an error or have a suggestion?
1. Update the file directly (if collaborator)
2. Or raise an issue on GitHub
3. Or add to [docs/FAQ.md](docs/FAQ.md) if it's a common question

---

## Next Steps

**If you're implementing**: Go to [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
**If you're planning**: Go to [docs/PHASE_1_ROADMAP.md](docs/PHASE_1_ROADMAP.md)
**If you want technical details**: Pick a spec from the table above
**If you're new**: Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first
