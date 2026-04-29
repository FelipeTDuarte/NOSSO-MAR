# Archived Files Reference

This folder contains superseded versions of project planning documents. They are kept for historical reference but are **no longer the canonical source**.

## What Was Archived

### 1. `Nosso-Mar project.ipynb` (OLD)
**Reason**: Consolidated into `00_MASTER_PLAN.ipynb` to eliminate duplication

**Content moved to**:
- Main plan phases → `00_MASTER_PLAN.ipynb`
- Technical specs → `specs/01_*.md` to `specs/09_*.md`
- Architecture → `docs/ARCHITECTURE.md`

**When you might reference**: Never. Use current files instead.

---

### 2. `Nosso-Mar plano definitivo.ipynb` (OLD)
**Reason**: Exact duplicate of project.ipynb; no longer needed

**Content moved to**:
- Same destinations as above

**When you might reference**: Never. Use current files instead.

---

### 3. `NOSSO-Mar_thread_v1.md`, `v202604120.md`, `v20260419.md` (ARCHIVED)
**Reason**: Preserved as design history in `THREADS_ARCHIVE.md`

**Content**: Design evolution, decision rationale, architectural thinking

**When you might reference**: 
- Understanding "why" a design choice was made
- Tracing evolution from feasibility → architecture → operational
- See [THREADS_ARCHIVE.md](../THREADS_ARCHIVE.md)

---

## How to Use This Folder

1. **If referencing history**: Check [THREADS_ARCHIVE.md](../THREADS_ARCHIVE.md) instead
2. **If implementing a feature**: Use current spec files in `specs/` + docs in `docs/`
3. **If confused about what to read**: See [README_WORKSPACE.md](../README_WORKSPACE.md) navigation table

---

## Why Archive Instead of Delete?

- **Historical record**: Decisions made with reasoning preserved
- **Traceability**: Git history shows evolution
- **Learning**: New team members understand design journey
- **Comparison**: Can see what changed and why

---

## File Structure of Archive

```
_archive/
├── README_ARCHIVE.md (this file)
├── OLD_notebooks/
│   └── [Original .ipynb files if kept]
└── OLD_threads/
    └── [Thread versions consolidated into THREADS_ARCHIVE.md]
```

---

## Single Source of Truth (NOW)

| Topic | Old Location | **New Location** |
|-------|-------------|------------------|
| Project phases & roadmap | 2x .ipynb files | `00_MASTER_PLAN.ipynb` |
| F1A WSI operator spec | project.ipynb cell | `specs/03_F1A_WSI_OPERATOR.md` |
| Data strategy | project.ipynb cell | `specs/02_DATA_STRATEGY.md` |
| System architecture | fulloceanPINO_thread + project.ipynb | `docs/ARCHITECTURE.md` |
| Design evolution | 4x thread files | `THREADS_ARCHIVE.md` |
| How to contribute | (nowhere!) | `docs/CONTRIBUTING.md` |
| Phase 1 progress | (nowhere!) | `docs/PHASE_1_ROADMAP.md` |
| FAQ | (scattered in threads) | `docs/FAQ.md` |

---

## Migration Checklist (completed)

- ✓ Consolidated 2x duplicate notebooks → 1x `00_MASTER_PLAN.ipynb`
- ✓ Extracted 9 technical specs from notebooks + threads → `specs/01_*.md` to `specs/09_*.md`
- ✓ Created system architecture doc → `docs/ARCHITECTURE.md`
- ✓ Indexed thread evolution → `THREADS_ARCHIVE.md`
- ✓ Added FAQ → `docs/FAQ.md`
- ✓ Added contributing guide → `docs/CONTRIBUTING.md`
- ✓ Added roadmap tracking → `docs/PHASE_1_ROADMAP.md`
- ✓ Created workspace navigation → `README_WORKSPACE.md`

---

## Contact

If you have questions about why something was archived:
- Check `THREADS_ARCHIVE.md` for design context
- Open issue on GitHub linking to relevant spec
- Contact Felipe Duarte (NOSSO-MAR lead)

---

**Last archived**: April 28, 2026
**Next review**: After Phase 1 completion
**Maintained by**: NOSSO-MAR team
