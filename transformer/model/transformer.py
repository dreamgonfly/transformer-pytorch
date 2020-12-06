from typing import Tuple, Optional

from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pad_packed_sequence,
    pack_padded_sequence,
)

from transformer.model.decoder import TransformerDecoder
from transformer.model.encoder import TransformerEncoder
from transformer.model.positional_encoding import PositionalEncoding
from transformer.model.state import EncoderState, DecoderState


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

        self.sources_embedding = nn.Sequential(
            nn.Embedding(source_vocab_size, d_model, padding_idx=pad_token_index),
            PositionalEncoding(d_model, num_positions),
            nn.Dropout(p=dropout),
        )
        self.encoder = TransformerEncoder(num_layers, d_model, d_ff, n_heads, dropout)

        self.inputs_token_embedding = nn.Embedding(
            target_vocab_size, d_model, padding_idx=pad_token_index
        )
        self.inputs_positional_encoding = PositionalEncoding(d_model, num_positions)
        self.inputs_dropout = nn.Dropout(p=dropout)
        use_memory = True
        self.decoder = TransformerDecoder(num_layers, d_model, d_ff, n_heads, dropout, use_memory)

        self.target_projection = nn.Linear(d_model, target_vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.logit_scale = 1.0
        if input_target_weight_sharing:
            self.target_projection.weight = self.inputs_token_embedding.weight
            self.logit_scale = d_model ** -0.5

        if source_target_weight_sharing:
            self.sources_embedding[0].weight = self.inputs_token_embedding.weight

        self.pad_token_index = pad_token_index

    def forward(self, sources: PackedSequence, inputs: PackedSequence) -> Tensor:
        memories, _ = self.encode(sources)
        log_probs, _ = self.decode(memories, inputs, state=None, cache=False)
        return log_probs

    def encode(self, sources: PackedSequence) -> Tuple[PackedSequence, EncoderState]:
        sources_sequence, source_lengths = pad_packed_sequence(
            sources, batch_first=True, padding_value=self.pad_token_index
        )
        sources = self.sources_embedding(sources_sequence)
        sources = pack_padded_sequence(
            sources, source_lengths, batch_first=True, enforce_sorted=False
        )
        memories, state = self.encoder(sources, state=None)
        return memories, state

    def decode(
        self,
        memories: PackedSequence,
        inputs: PackedSequence,
        state: Optional[DecoderState],
        cache: bool,
    ) -> Tuple[Tensor, DecoderState]:

        if state is None:
            state = DecoderState()

        inputs_sequence, input_lengths = pad_packed_sequence(
            inputs, batch_first=True, padding_value=self.pad_token_index
        )

        inputs = self.inputs_token_embedding(inputs_sequence)
        inputs = self.inputs_positional_encoding(inputs, state.position)
        inputs = self.inputs_dropout(inputs)

        inputs = pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, state = self.decoder(inputs, memories, state=state, cache=cache)
        logits = self.target_projection(outputs) * self.logit_scale
        log_probs = self.log_softmax(logits)

        state.position += 1

        return log_probs, state
