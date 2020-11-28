from typing import Optional

import torch
from torch import Tensor
from torch import nn


class ScaledDotProductAttention(nn.Module):
    temperature: float

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=3)

    def forward(self, query_heads: Tensor, key_heads: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Args:
             query_heads: (batch_size, n_heads, query_len, d_head)
             key_heads: (batch_size, n_heads, key_len, d_head)
             mask:
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        scores = torch.matmul(query_heads, key_heads_transposed)
        # scores: (batch_size, n_heads, query_len, key_len)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
            scores = scores.masked_fill(mask, -1e9)

        tempered_scores = scores / self.temperature

        attention = self.softmax(tempered_scores)
        return attention
