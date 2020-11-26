from typing import Optional

from torch import Tensor
from torch import nn

from transformer.model.decoder_layer import TransformerDecoderLayer
from transformer.model.state import DecoderState


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        use_memory: bool,
    ):
        super(TransformerDecoder, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, d_ff, n_heads, dropout, use_memory)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: Tensor,
        memories: Optional[Tensor],
        self_attention_mask: Optional[Tensor],
        memory_attention_mask: Optional[Tensor],
        state: Optional[DecoderState],
        cache: bool,
    ):
        # inputs: (batch_size, input_length, d_model)
        # memory: (batch_size, memory_length, d_model)

        if state is None:
            state = DecoderState()

        x = self.layer_norm(inputs)
        for layer_index, layer in enumerate(self.layers):
            layer_state = state.select_layer(layer_index)
            x, layer_state = layer(
                x, memories, self_attention_mask, memory_attention_mask, layer_state, cache
            )

            # state.set_layer(layer_index, layer_state)
        return x, state
