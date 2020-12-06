import math
from typing import Dict, Any, List
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from transformer.losses import SequenceNLLLoss, SequenceLabelSmoothingLoss
from transformer.metrics import accuracy_metric_from_tensor
from transformer.training.runner import Runner
from transformer.training.training_info import TrainingInfo


class ModelRunner(Runner):
    def __init__(self, pad_token_index: int):
        super().__init__()

        self.label_smoothing_loss = SequenceLabelSmoothingLoss(
            smoothing_value=0.1, ignore_index=pad_token_index
        )
        self.nll_loss = SequenceNLLLoss(ignore_index=pad_token_index)
        self.pad_token_index = pad_token_index

    def training_step(
        self, model: nn.Module, batch: torch.Tensor, info: TrainingInfo
    ) -> Dict[str, Any]:
        inputs, targets = batch

        outputs = model(
            inputs["source_token_indices"],
            inputs["input_token_indices"],
            inputs["source_length"],
            inputs["input_length"],
        )
        _, predictions = outputs.max(dim=-1)

        loss, num_tokens = self.label_smoothing_loss(outputs, targets["target_token_indices"])
        num_correct, num_tokens = accuracy_metric_from_tensor(
            targets["target_token_indices"], predictions, self.pad_token_index
        )

        return {"loss": loss, "acc": num_correct / num_tokens}

    def validation_step(self, model: nn.Module, batch: Any, info: TrainingInfo) -> Dict[str, Any]:
        inputs, targets = batch

        outputs = model(
            inputs["source_token_indices"],
            inputs["input_token_indices"],
            inputs["source_length"],
            inputs["input_length"],
        )
        _, predictions = outputs.max(dim=-1)

        loss, num_tokens = self.nll_loss(outputs, targets["target_token_indices"])
        num_correct, num_tokens = accuracy_metric_from_tensor(
            targets["target_token_indices"], predictions, self.pad_token_index
        )

        return {"val_loss": loss, "num_correct": num_correct, "num_tokens": num_tokens}

    def validation_epoch_end(self, val_results: Dict[str, List[Any]]) -> Dict[str, float]:
        total_num_tokens = sum(val_results["num_tokens"])
        val_loss = sum(val_results["val_loss"]) / total_num_tokens
        perplexity = math.exp(val_loss)

        return {
            "val_loss": val_loss,
            "val_acc": sum(val_results["num_correct"]) / total_num_tokens,
            "val_ppl": perplexity,
        }
