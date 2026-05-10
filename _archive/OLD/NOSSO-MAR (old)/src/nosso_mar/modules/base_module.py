"""Abstract base module for all NOSSO-MAR physics modules."""
from __future__ import annotations
import abc
from typing import Dict


class BaseModule(abc.ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abc.abstractmethod
    def run(self, inputs: Dict) -> Dict:
        """Execute module: inputs_dict -> outputs_dict."""
        ...

    def get_observation(self, agent_id: str) -> Dict:
        return {}

    def apply_agent_action(self, agent_id: str, action) -> None:
        pass

    def validate_inputs(self, inputs: Dict, required_keys: list):
        missing = [k for k in required_keys if k not in inputs]
        if missing:
            raise ValueError(f"{self.__class__.__name__}: missing inputs {missing}")
