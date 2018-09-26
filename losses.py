import torch
from torch import nn
from torch.nn import KLDivLoss

PAD_INDEX = 0


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


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, vocabulary_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0

        super(LabelSmoothingLoss, self).__init__()

        self.padding_idx = padding_idx
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

        smoothing_value = label_smoothing / (vocabulary_size - 2)  # exclude pad and true label
        one_hot = torch.full((vocabulary_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets, tmp):
        """
        outputs (FloatTensor): batch_size x seq_len x n_classes
        targets (LongTensor): batch_size x seq_len
        """
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs = self.log_softmax(outputs)
        outputs = outputs.view(batch_size * seq_len, vocabulary_size)
        targets = targets.view(batch_size * seq_len)
        model_prob = self.one_hot.repeat(targets.size(0), 1)
        model_prob.scatter_(1, targets.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((targets == self.padding_idx).unsqueeze(1), 0)

        loss = self.criterion(outputs, model_prob)
        count = (targets != self.padding_idx).sum().item()

        return loss, count
