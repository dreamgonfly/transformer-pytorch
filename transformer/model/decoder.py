from typing import Optional, Tuple

from torch import Tensor
from torch import nn

from transformer.model.decoder_layer import TransformerDecoderLayer
from transformer.model.masking import mask_from_lengths, mask_from_subsequent_positions
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

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
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
        input_lengths: Optional[Tensor],
        memory_lengths: Optional[Tensor],
        state: Optional[DecoderState],
        cache: bool,
    ) -> Tuple[Tensor, DecoderState]:
        # inputs: (batch_size, input_length, d_model)
        # memory: (batch_size, memory_length, d_model)

        if state is None:
            state = DecoderState()

        input_length_mask = mask_from_lengths(input_lengths).unsqueeze(1)
        input_subsequent_mask = mask_from_subsequent_positions(inputs.size(1)).to(inputs.device)
        self_attention_mask = input_length_mask | input_subsequent_mask

        memory_attention_mask = mask_from_lengths(memory_lengths).unsqueeze(1)

        x = self.layer_norm(inputs)
        for layer_index, layer in enumerate(self.layers):
            layer_state = state.select_layer(layer_index)
            x, layer_state = layer(
                x, memories, self_attention_mask, memory_attention_mask, layer_state, cache
            )

            state.set_layer(layer_index, layer_state)
        return x, state
