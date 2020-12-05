from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
import numpy as np

from transformer.model.scaled_dot_product_attention import ScaledDotProductAttention
from transformer.model.state import AttentionMode, AttentionState


class MultiHeadAttention(nn.Module):
    d_head: int
    n_heads: int
    mode: AttentionMode

    def __init__(self, d_model: int, n_heads: int, dropout: float, mode: Optional[AttentionMode]):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.key_projection = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.value_projection = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.final_projection = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=np.sqrt(self.d_head))
        self.attention_dropout = nn.Dropout(dropout)

        self.mode = mode

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor],
        state: AttentionState,
        cache: bool,
    ) -> Tuple[Tensor, AttentionState]:
        """

        Args:
            query: (batch_size, query_length, d_model)
            key: (batch_size, key_length, d_model)
            value: (batch_size, value_length, d_model)
            mask: (query_length, key_length)
            cache: No cache if None
        """
        batch_size, query_len, d_model = query.size()
        n_heads = self.n_heads
        d_head = self.d_head

        query_projected = self.query_projection(query)
        if not cache or not state.cache:
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:
            if self.mode == AttentionMode.SELF:
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)
                key_projected = torch.cat([state.cache.key_projected, key_projected], dim=1)
                value_projected = torch.cat([state.cache.value_projected, value_projected], dim=1)
            elif self.mode == AttentionMode.MEMORY:
                key_projected = state.cache.key_projected
                value_projected = state.cache.value_projected
            else:
                raise NotImplementedError

        if cache:
            state.cache.key_projected = key_projected
            state.cache.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, n_heads, d_head).transpose(1, 2)
        # query_heads: (batch_size, n_heads, query_len, d_head)
        key_heads = key_projected.view(batch_size, key_len, n_heads, d_head).transpose(1, 2)
        # key_heads: (batch_size, n_heads, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, n_heads, d_head).transpose(1, 2)
        # value_heads: (batch_size, n_heads, value_len, d_head)

        if mask is not None:
            mask = mask.unsqueeze(1)
        attention = self.attention(query_heads, key_heads, mask)
        # attention: (batch_size, n_heads, query_len, key_len)
        attention_dropped = self.attention_dropout(attention)

        context_heads = torch.matmul(attention_dropped, value_heads)
        # context_heads: (batch_size, n_heads, query_len, d_head)
        context_sequence = context_heads.transpose(1, 2).contiguous()
        # (batch_size, query_len, n_heads, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)
        # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)

        if cache:
            state.attention = attention.detach().cpu()

        return final_output, state
