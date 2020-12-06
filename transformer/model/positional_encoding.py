from typing import Optional

import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, num_positions: int):
        super(PositionalEncoding, self).__init__()
        if embedding_size % 2 != 0:
            raise ValueError(
                f"Expected even number for positional embedding size, but got {embedding_size} instead."
            )
        encoding_table = torch.zeros(num_positions, embedding_size)
        positions = torch.arange(0, num_positions, dtype=torch.float64).unsqueeze(1)
        dimensions = torch.arange(0, embedding_size, 2, dtype=torch.float64)
        div_term = torch.pow(10000.0, dimensions / embedding_size)
        encoding_table[:, 0::2] = torch.sin(positions / div_term)
        encoding_table[:, 1::2] = torch.cos(positions / div_term)
        encoding_table = encoding_table.unsqueeze(0).float()
        super(PositionalEncoding, self).__init__()
        self.register_buffer("encoding_table", encoding_table)

    def forward(self, x: Tensor, position: Optional[int] = None):
        if position is None:
            return x + self.encoding_table[:, x.size(1)]
        else:
            return x + self.encoding_table[:, position : position + x.size(1)]
