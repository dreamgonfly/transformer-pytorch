from typing import Optional, Tuple

from torch import Tensor
from torch import nn

from transformer.model.state import AttentionMode, LayerState, AttentionState
from transformer.model.sublayers import MultiHeadAttentionSublayer, PositionWiseFeedForwardSublayer


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, n_heads: int, dropout: float, use_memory: bool,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionSublayer(
            d_model, n_heads, dropout, AttentionMode.SELF
        )

        if use_memory:
            self.memory_attention = MultiHeadAttentionSublayer(
                d_model, n_heads, dropout, AttentionMode.MEMORY
            )
        else:
            self.memory_attention = None

        self.feed_forward = PositionWiseFeedForwardSublayer(d_model, d_ff, dropout)

    def forward(
        self,
        inputs: Tensor,
        memories: Optional[Tensor],
        self_attention_mask: Optional[Tensor],
        memory_attention_mask: Optional[Tensor],
        state: LayerState,
        cache: bool,
    ) -> Tuple[Tensor, LayerState]:
        x, self_attention_state = self.self_attention(
            inputs, inputs, inputs, self_attention_mask, state.self_attention, cache
        )
        if self.memory_attention:
            x, memory_attention_state = self.memory_attention(
                x, memories, memories, memory_attention_mask, state.memory_attention, cache
            )
        else:
            memory_attention_state = AttentionState()
        x = self.feed_forward(x)

        state.self_attention = self_attention_state
        state.memory_attention = memory_attention_state
        return x, state
