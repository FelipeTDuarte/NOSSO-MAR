import json, logging
from pathlib import Path


def setup_logger(name: str = "nosso_mar",
                 level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=level)
    return logging.getLogger(name)


def log_experiment(path: str, metadata: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metadata, indent=2))
