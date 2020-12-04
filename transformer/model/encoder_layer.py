from typing import Optional, Tuple

from torch import Tensor
from torch import nn

from transformer.model.state import LayerState
from transformer.model.sublayers import MultiHeadAttentionSublayer, PositionWiseFeedForwardSublayer


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttentionSublayer(d_model, n_heads, dropout, mode=None)
        self.feed_forward = PositionWiseFeedForwardSublayer(d_model, d_ff, dropout)

    def forward(
        self, x: Tensor, mask: Optional[Tensor], state: LayerState
    ) -> Tuple[Tensor, LayerState]:
        x, self_attention_state = self.self_attention(
            x, x, x, mask, state.self_attention, cache=False
        )
        x = self.feed_forward(x)

        state.self_attention = self_attention_state
        return x, state
