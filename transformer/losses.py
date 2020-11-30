import torch
from typing import Tuple

from torch import nn, Tensor


class SequenceNLLLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, int]:
        batch_size, sequence_length, vocab_size = input.size()
        input_flat = input.view(batch_size * sequence_length, vocab_size)
        target_flat = target.view(batch_size * sequence_length)
        loss = self.nll_loss(input_flat, target_flat)
        num_elements = target.ne(self.ignore_index).sum().item()
        return loss, num_elements


class SequenceLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing_value: float, ignore_index: int = -100):
        super().__init__()

        assert 0.0 < smoothing_value <= 1.0

        self.smoothing_value = smoothing_value
        self.confidence = 1 - smoothing_value
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, int]:
        batch_size, sequence_length, num_classes = input.size()
        num_other_classes = num_classes - 1  # -2 for pad and original value

        input_flat = input.view(batch_size * sequence_length, num_classes)
        target_flat = target.view(batch_size * sequence_length)

        one_hot = torch.zeros_like(input_flat).scatter(dim=1, index=target.view(-1, 1), value=1)
        one_hot = (
            one_hot * self.confidence + (1 - one_hot) * self.smoothing_value / num_other_classes
        )
        non_pad_mask = target_flat.ne(self.ignore_index)
        loss = -(one_hot * input_flat).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
        num_elements = target_flat.ne(self.ignore_index).sum().item()
        return loss, num_elements
