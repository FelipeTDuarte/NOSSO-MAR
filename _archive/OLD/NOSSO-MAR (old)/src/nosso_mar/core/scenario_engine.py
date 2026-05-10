"""
Scenario engine — loads YAML config, instantiates modules, runs coupling.
"""
from __future__ import annotations
from typing import Dict
import yaml
import logging

logger = logging.getLogger(__name__)


class ScenarioEngine:
    def __init__(self, config_file: str):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

    def run(self) -> Dict:
        name = self.config.get("scenario", {}).get("name", "unnamed")
        logger.info(f"Running scenario: {name}")

        from .model_factory import ModelFactory
        from .coupling_manager import CouplingManager

        module_names = self.config.get("modules", [])
        modules = [ModelFactory.build_module(m, self.config)
                   for m in module_names]

        coupling_cfg = self.config.get("coupling", {})
        if coupling_cfg.get("strategy") == "iterative" and len(modules) >= 2:
            cm  = CouplingManager(modules[0], modules[1],
                                  max_iter=coupling_cfg.get("max_iter", 10),
                                  tol=coupling_cfg.get("tol", 1e-4))
            out = cm.run_coupled(self.config)
        else:
            cm  = CouplingManager(None, None)
            out = cm.execute_sequential(modules, self.config)

        logger.info(f"Scenario '{name}' complete. "
                    f"Total power: {out.get('total_power', 'N/A')}")
        return out
