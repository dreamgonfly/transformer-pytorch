from abc import ABC, abstractmethod

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class ModelInfo:
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: Optional[_LRScheduler]
    epoch: int
    global_step: int


class MonitorMode(Enum):
    MIN = "MIN"
    MAX = "MAX"


class Checkpointer(ABC):
    @abstractmethod
    def checkpoint(self, model_info: ModelInfo, metrics: dict) -> None:
        pass
