"""Build a small open-data bootstrap database for NOSSO-MAR."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nossomar.data.database_builder import build_open_database


def main(config_path: str = "configs/open_data.yaml") -> None:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    oisst_date = config.get("oisst_date")
    manifest = build_open_database(
        config["root_dir"],
        ndbc_station=str(config.get("ndbc_station", "41009")),
        coops_station=str(config.get("coops_station", "9414290")),
        coops_days=int(config.get("coops_days", 2)),
        oisst_date=None if not oisst_date else date.fromisoformat(str(oisst_date)),
        include_metadata_snapshots=bool(config.get("include_metadata_snapshots", True)),
    )
    print(json.dumps(
        {
            "root_dir": manifest["root_dir"],
            "n_artifacts": len(manifest["artifacts"]),
            "n_errors": len(manifest["errors"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
