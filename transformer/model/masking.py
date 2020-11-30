import numpy as np

from torch import Tensor
import torch


def mask_from_lengths(lengths: Tensor) -> Tensor:
    """Make key_padding_mask

    key_padding_mask should be a ByteTensor where True values are positions that should be masked with float(‘-inf’)
    and False values will be unchanged.

    # length_mask: (batch_size, max_length)
    """
    max_length = lengths.max().item()
    positions = torch.as_tensor(np.arange(max_length), device=lengths.device)
    positions_expanded = positions.unsqueeze(0)
    lengths_expanded = lengths.unsqueeze(1)
    length_mask = torch.ge(positions_expanded, lengths_expanded)
    return length_mask


def mask_from_subsequent_positions(size: int):
    """Make attention_mask

    attention_mask should be filled with float(‘-inf’) for the masked positions and float(0.0) else.
    """
    subsequent_mask = np.triu(np.full(shape=(size, size), fill_value=1, dtype=np.float32), k=1)
    subsequent_mask = torch.as_tensor(subsequent_mask)
    return subsequent_mask.bool()
