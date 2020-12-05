import math
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
        position_encodings = torch.zeros(num_positions, embedding_size, requires_grad=False)
        position = torch.arange(0, num_positions).unsqueeze(1)
        dimension = torch.arange(0, embedding_size, 2, dtype=torch.float)
        div_term = torch.exp(dimension * -(math.log(10000.0) / embedding_size))
        position_encodings[:, 0::2] = torch.sin(position.float() * div_term)
        position_encodings[:, 1::2] = torch.cos(position.float() * div_term)
        position_encodings = position_encodings.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("position_encodings", position_encodings)

    def forward(self, x: Tensor, position: Optional[int] = None):
        if position is None:
            return x + self.position_encodings[:, x.size(1)]
        else:
            return x + self.position_encodings[:, position : position + x.size(1)]
