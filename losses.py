import torch
from torch import nn


class MaskedCrossEntropyLoss:

    def __init__(self, device):
        self.base_loss_function = nn.CrossEntropyLoss(reduce=False)
        self.device = torch.device(device)

    def __call__(self, outputs, targets, targets_lengths):
        batch_size, seq_len, vocabulary_size = outputs.size()
        outputs_flat = outputs.view(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.view(batch_size * seq_len)
        batch_loss = self.base_loss_function(outputs_flat, targets_flat)
        loss_mask = self.sequence_mask(targets_lengths).to(self.device)

        batch_loss_masked = batch_loss.masked_fill(loss_mask, 0)
        batch_loss_summed = batch_loss_masked.sum()

        count = targets_lengths.sum().item()

        return batch_loss_summed, count

    @staticmethod
    def sequence_mask(lengths, max_length=None):
        # lengths: (batch_size, )
        if not max_length:
            max_length = lengths.max()  # or predefined max_len
        batch_size = lengths.size(0)
        lengths_broadcastable = lengths.unsqueeze(1)
        mask = torch.arange(0, max_length).type_as(lengths).repeat(batch_size, 1) >= lengths_broadcastable
        # mask: (batch_size, seq_length)
        return mask.view(-1)