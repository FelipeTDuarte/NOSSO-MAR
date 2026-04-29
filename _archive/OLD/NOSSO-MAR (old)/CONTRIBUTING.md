# Contributing to NOSSO-MAR

1. Fork → clone → `git checkout -b feature/<name>`
2. Make focused commits
3. `black . && isort . && pytest tests/ -x`
4. Open PR to `develop`
5. Wait for CI + review

Branch model: `main` (stable) / `develop` (integration) / `feature/*`
