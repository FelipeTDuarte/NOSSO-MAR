"""Tests for Phase 2 validation CLI (P2-T5)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "scripts/validate_phase2.py", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_help_exits_zero():
    result = _run("--help")
    assert result.returncode == 0, result.stderr


def test_help_shows_usage():
    result = _run("--help")
    assert "usage" in result.stdout.lower() or "validate" in result.stdout.lower()


def test_smoke_exits_zero():
    result = _run("--smoke")
    assert result.returncode == 0, (
        f"Smoke run failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_smoke_prints_table():
    result = _run("--smoke")
    assert "Phase 1" in result.stdout
    assert "Phase 2" in result.stdout


def test_smoke_prints_rmse():
    result = _run("--smoke")
    assert "RMSE" in result.stdout or "rmse" in result.stdout.lower()


def test_missing_benchmark_exits_nonzero(tmp_path):
    result = _run("--benchmark", str(tmp_path / "nonexistent.json"))
    assert result.returncode != 0
