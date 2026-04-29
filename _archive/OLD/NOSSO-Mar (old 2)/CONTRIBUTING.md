# Contributing to NOSSO-MAR

## Workflow: Subagent-Driven Development

This project uses the [Superpowers](https://github.com/obra/superpowers) skills system.
Every contribution follows the same loop:

```
fresh implementer subagent
  → spec compliance review
  → code quality review
  → commit
```

### Prerequisites

- Claude Code with Superpowers plugin installed
- Python 3.11+
- `pip install -e .` from repo root

### Starting a task

1. Open Claude Code in this repo directory
2. Say: "dispatch Task N from docs/superpowers/plans/2026-04-19-nosso-mar-subagent-plan.md"
3. The subagent will follow TDD (RED → GREEN → REFACTOR) automatically

### Rules

- Never skip the RED step — always run the failing test first
- Never commit without running `pytest tests/ -v`
- Each task gets its own commit with `feat:` or `fix:` prefix
- Warnings are treated as errors — `pytest -W error` before final commit

### Data strategy

Phase 1 uses **analytic data only** (zero BEM solver cost).
Phase 2 adds Capytaine only if Phase-1 validation error > 15%.
Phase 3 adds OpenFOAM/OceanWave3D for paper validation cases only.

See `docs/superpowers/plans/2026-04-19-nosso-mar-subagent-plan.md` for full plan.
