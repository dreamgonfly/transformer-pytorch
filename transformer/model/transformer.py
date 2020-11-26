from torch import Tensor
from torch import nn

from transformer.model.decoder import TransformerDecoder
from transformer.model.encoder import TransformerEncoder
from transformer.model.masking import get_pad_mask, get_subsequent_mask
from transformer.model.positional_encoding import PositionalEncoding
from transformer.model.state import EncoderState


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        pad_token_index: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        num_positions: int,
        input_target_weight_sharing: bool,
        source_target_weight_sharing: bool,
    ):
        super().__init__()

        self.sources_embedding = nn.Embedding(
            source_vocab_size, d_model, padding_idx=pad_token_index
        )
        self.sources_position_enc = PositionalEncoding(d_model, n_position=num_positions)
        self.sources_dropout = nn.Dropout(p=dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, d_ff, n_heads, dropout)

        self.inputs_embedding = nn.Embedding(
            target_vocab_size, d_model, padding_idx=pad_token_index
        )
        self.inputs_position_enc = PositionalEncoding(d_model, n_position=num_positions)
        self.inputs_dropout = nn.Dropout(p=dropout)
        use_memory = True
        self.decoder = TransformerDecoder(num_layers, d_model, d_ff, n_heads, dropout, use_memory)

        self.generator = nn.Linear(d_model, target_vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.x_logit_scale = 1.0
        if input_target_weight_sharing:
            self.generator.weight = self.inputs_embedding.weight
            self.x_logit_scale = d_model ** -0.5

        if source_target_weight_sharing:
            self.sources_embedding.weight = self.inputs_embedding.weight

        self.pad_token_index = pad_token_index

    def forward(self, sources: Tensor, inputs: Tensor) -> Tensor:
        sources_mask = get_pad_mask(sources, self.pad_token_index) == 0
        inputs_mask = (
            get_pad_mask(inputs, self.pad_token_index) & get_subsequent_mask(inputs)
        ) == 0

        sources = self.sources_dropout(self.sources_position_enc(self.sources_embedding(sources)))
        memories, _ = self.encoder(sources, sources_mask, state=None)

        inputs = self.inputs_dropout(self.inputs_position_enc(self.inputs_embedding(inputs)))
        outputs, _ = self.decoder(
            inputs, memories, inputs_mask, sources_mask, state=None, cache=False
        )
        logits = self.generator(outputs) * self.x_logit_scale

        return logits.view(-1, logits.size(2))
