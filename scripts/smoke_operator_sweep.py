"""Run a smoke-test sweep across the local operator families."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nossomar.experiments.architecture_sweep import run_smoke_sweep


def main() -> None:
    results = [item.to_dict() for item in run_smoke_sweep()]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
