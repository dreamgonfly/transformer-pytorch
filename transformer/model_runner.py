import math
from typing import Dict, Any, List
import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy

from transformer.training.runner import Runner
from transformer.training.training_info import TrainingInfo


class ModelRunner(Runner):
    def __init__(self):
        super().__init__()

    def training_step(
        self, model: nn.Module, batch: torch.Tensor, info: TrainingInfo
    ) -> Dict[str, Any]:
        inputs, targets = batch

        src = inputs["source_token_indices"]  # .transpose(0, 1)
        trg = targets["target_token_indices"]  # .transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        logits = model(src, trg)

        smoothing = True
        if smoothing:
            eps = 0.1
            n_class = logits.size(1)

            one_hot = torch.zeros_like(logits).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = torch.log_softmax(logits, dim=1)

            non_pad_mask = gold.ne(0)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            raise NotImplementedError

        logits = logits.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(0)
        n_correct = logits.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return {"loss": loss, "acc": n_correct / n_word}

    def validation_step(self, model: nn.Module, batch: Any, info: TrainingInfo) -> Dict[str, Any]:

        inputs, targets = batch

        src = inputs["source_token_indices"]  # .transpose(0, 1)
        trg = targets["target_token_indices"]  # .transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        logits = model(src, trg)

        loss = cross_entropy(logits, gold, ignore_index=0, reduction="sum")

        logits = logits.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(0)
        n_correct = logits.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return {"val_loss": loss, "n_correct": n_correct, "n_word": n_word}

    def validation_epoch_end(self, val_results: Dict[str, List[Any]]) -> Dict[str, float]:
        n_word_total = sum(val_results["n_word"])
        val_loss = sum(val_results["val_loss"]) / n_word_total
        ppl = math.exp(min(val_loss, 100))

        return {
            "val_loss": val_loss,
            "val_acc": sum(val_results["n_correct"]) / n_word_total,
            "val_ppl": ppl,
        }
