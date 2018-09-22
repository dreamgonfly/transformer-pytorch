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

    def __init__(self, label_smoothing, vocabulary_size):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= label_smoothing <= 1.0
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, vocabulary_size)
            one_hot.fill_(label_smoothing / (vocabulary_size - 2))  # exclude pad and true label
            one_hot[0][PAD_INDEX] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(vocabulary_size)
            weight[PAD_INDEX] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def forward(self, outputs, targets, targets_lengths):
        outputs_log_softmax = self.log_softmax(outputs)

        outputs_flat = outputs_log_softmax.view(-1, outputs_log_softmax.size(2))
        targets_flat = targets.view(-1)
        if self.confidence < 1:
            targets_detached = targets_flat.detach()
            mask = torch.nonzero(targets_detached == PAD_INDEX).squeeze(-1)
            tmp_ = self.one_hot.repeat(targets_flat.size(0), 1)
            tmp_.scatter_(1, targets_detached.unsqueeze(1), self.confidence)
            if mask.size(0) > 0:
                tmp_.index_fill_(0, mask, 0)
            targets_flat = tmp_
        loss = self.criterion(outputs_flat, targets_flat)

        count = (targets_detached != PAD_INDEX).sum().item()

        return loss, count