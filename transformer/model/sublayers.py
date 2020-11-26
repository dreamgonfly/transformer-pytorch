from typing import Optional, Tuple

from torch import Tensor
from torch import nn

from transformer.model.multi_head_attention import MultiHeadAttention
from transformer.model.position_wise_feed_forward import PositionWiseFeedForwardNetwork
from transformer.model.state import AttentionMode, AttentionState


class MultiHeadAttentionSublayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, mode: Optional[AttentionMode]):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, mode)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor],
        state: AttentionState,
        cache: bool,
    ) -> Tuple[Tensor, AttentionState]:
        x, state = self.attention(query, key, value, mask, state, cache)
        x = self.dropout(x)
        x += query
        x = self.layer_norm(x)
        return x, state


class PositionWiseFeedForwardSublayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.feed_forward = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.feed_forward(inputs)
        x = self.dropout(x)
        x += inputs
        x = self.layer_norm(x)
        return x
