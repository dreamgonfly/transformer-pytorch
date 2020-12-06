from typing import Optional, Tuple

from torch import Tensor
from torch import nn

from transformer.model.encoder_layer import TransformerEncoderLayer
from transformer.model.masking import mask_from_lengths
from transformer.model.state import EncoderState


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, d_model: int, d_ff: int, n_heads: int, dropout: float,
    ):
        super(TransformerEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, sources: Tensor, source_lengths: Optional[Tensor], state: Optional[EncoderState]
    ) -> Tuple[Tensor, EncoderState]:
        # x: (batch_size, source_length, d_model)

        length_mask = mask_from_lengths(source_lengths).unsqueeze(1)

        if state is None:
            state = EncoderState()

        x = self.layer_norm(sources)
        for layer_index, layer in enumerate(self.layers):
            x, layer_state = layer(x, length_mask, state.select_layer(layer_index))

            state.set_layer(layer_index, layer_state)

        return x, state
