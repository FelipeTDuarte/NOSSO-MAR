"""Abstract base class for all Neural Operators in NOSSO-MAR."""
from __future__ import annotations
import abc
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseOperator(nn.Module, abc.ABC):
    """
    All NOSSO-MAR neural operators inherit from this class.

    Convention:
        forward(u, query_points=None) -> v
        u            : input function values, shape (B, C_in, *spatial)
        query_points : optional target coordinates (for mesh-free / DeepONet)
        v            : output function values, shape (B, C_out, *spatial)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def forward(
        self,
        u: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"{self.__class__.__name__} | "
            f"params={self.count_parameters():,} | "
            f"cfg={self.cfg}"
        )
