from typing import Optional, Tuple

from torch import Tensor
from torch import nn

from transformer.model.encoder_layer import TransformerEncoderLayer
from transformer.model.state import EncoderState


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, dropout):
        super(TransformerEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, sources: Tensor, mask: Optional[Tensor], state: Optional[EncoderState]
    ) -> Tuple[Tensor, EncoderState]:
        # x: (batch_size, source_length, d_model)

        if state is None:
            state = EncoderState()

        x = self.layer_norm(sources)
        for layer_index, layer in enumerate(self.layers):
            x, layer_state = layer(x, mask, state.select_layer(layer_index))

            # state.set_layer(layer_index, layer_state)
        return x, state
