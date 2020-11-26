from typing import List, Dict, Any

import torch
from torch import nn

from .training_info import TrainingInfo


class Runner:
    def __init__(self):
        pass

    def training_step(
        self, model: nn.Module, batch: torch.Tensor, info: TrainingInfo
    ) -> Dict[str, Any]:
        pass

    def validation_step(self, model: nn.Module, batch: Any, info: TrainingInfo) -> Dict[str, Any]:
        pass

    def validation_epoch_end(self, val_results: Dict[str, List[Any]]) -> Dict[str, float]:
        pass
