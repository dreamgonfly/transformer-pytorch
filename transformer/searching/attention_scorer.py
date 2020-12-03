from typing import Dict, Any
import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from transformer.model.state import DecoderState
from transformer.model.transformer import Transformer


class AttentionScorer:
    state: Dict[str, Any]
    device: str

    def __init__(self, model: Transformer, device: torch.device):
        self.model = model
        self.device = device
        self.state = {}

    def initialize(self, batch_size: int, memories: PackedSequence) -> None:
        # """Initilize AttentionScorer state.
        #
        # Args:
        #     batch_size: int.
        #     memory: (batch_size, memory_length, hidden_size).
        #     memory_lengths: (batch_size,).
        # """
        self.state = {
            "memories": memories,
            "decoder_state": None,
            "attention": [[] for _ in range(batch_size)],
        }

    def forward_step(self, sequences: Tensor) -> Tensor:
        """Predict probabilities of next step tokens.

        Args:
            sequences: (batch_size x beam_size, sequence_length).

        Returns:
            (batch_size x beam_size, vocab_size)
        """
        input_sequence = sequences[:, -1:].to(device=self.device)
        lengths = torch.full(size=(len(input_sequence),), fill_value=1, dtype=torch.long)
        outputs, state = self.model.decode(
            memories=self.state["memories"],
            inputs=pack_padded_sequence(
                input_sequence, lengths, batch_first=True, enforce_sorted=False
            ),
            state=self.state["decoder_state"],
            cache=True,
        )
        # attention: (batch_size, query_len=1, key_len)
        # step_attention = state.attention.squeeze(dim=1)
        # for index, previous_attentions in enumerate(self.state["attention"]):
        #     previous_attentions.append(step_attention[index])
        self.state["decoder_state"] = state
        return outputs.squeeze(1)

    def select_indices(self, selected_indices: Tensor) -> None:
        """Keep only states by indices.

        Args:
            selected_indices: (number of selected indices,).
        """
        memories_tensor, memory_lengths = pad_packed_sequence(
            self.state["memories"], batch_first=True
        )
        selected_memories = torch.index_select(memories_tensor, dim=0, index=selected_indices)
        selected_lengths = torch.index_select(memory_lengths, dim=0, index=selected_indices)
        self.state: Dict[str, Any] = {
            "memories": pack_padded_sequence(
                selected_memories, selected_lengths, batch_first=True, enforce_sorted=False
            ),
            "decoder_state": DecoderState.merge(
                [self.state["decoder_state"].select_sample(index) for index in selected_indices]
            ),
            # "attention": [self.state["attention"][index] for index in selected_indices],
        }
