from __future__ import annotations

from itertools import groupby
from typing import List, Tuple

import torch
from torch import Tensor
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

from transformer.searching.hypothesis import Hypothesis


@dataclass
class BatchHypotheses:
    sequences: Tensor  # (batch_size x beam_size, sequence_length)
    scores: Tensor  # (batch_size x beam_size)
    lengths: List[int]
    sample_indices: List[int]

    @classmethod
    def new(cls, batch_size: int, start_token_id: int):
        return cls(
            sequences=torch.full(size=(batch_size, 1), fill_value=start_token_id, dtype=torch.long),
            scores=torch.full(size=(batch_size,), fill_value=0.0),
            lengths=[1 for _ in range(batch_size)],
            sample_indices=[i for i in range(batch_size)],
        )

    @classmethod
    def batchify(cls, hypotheses: List[Hypothesis], pad_token_id: int):
        return cls(
            sequences=pad_sequence(
                [h.sequence for h in hypotheses], batch_first=True, padding_value=pad_token_id
            ),
            scores=torch.tensor([h.score for h in hypotheses]),
            lengths=[h.length for h in hypotheses],
            sample_indices=[h.sample_index for h in hypotheses],
        )

    def unbatchify(self) -> List[Hypothesis]:
        split_sequences = self.sequences.split(split_size=1)
        split_scores = self.scores.split(split_size=1)
        hypotheses = []
        for index in range(len(split_sequences)):
            hypothesis = Hypothesis(
                sequence=split_sequences[index].squeeze(dim=0)[: self.lengths[index]],
                score=float(split_scores[index]),
                length=self.lengths[index],
                sample_index=self.sample_indices[index],
            )
            hypotheses.append(hypothesis)

        # TODO: Check for memory leak
        # del self.sequences
        # del self.scores
        return hypotheses

    def update(
        self, previous_indices: Tensor, next_token_ids: Tensor, next_scores: Tensor,
    ) -> None:
        """Update sequences, scores, etc of batch hypotheses.

        Args:
            previous_indices: (batch_size x beam_size,)
            next_token_ids: (batch_size x beam_size,)
            next_scores: (batch_size x beam_size,)
        """
        previous_indices = previous_indices.detach().to("cpu")
        next_token_ids = next_token_ids.detach().to("cpu")
        next_scores = next_scores.detach().to("cpu")

        previous_sequences = self.sequences.index_select(dim=0, index=previous_indices)
        previous_scores = self.scores.index_select(dim=0, index=previous_indices)
        self.sequences = torch.cat([previous_sequences, next_token_ids.unsqueeze(dim=1)], dim=1)
        self.scores = previous_scores + next_scores
        self.lengths = [self.lengths[index] + 1 for index in previous_indices]
        self.sample_indices = [self.sample_indices[index] for index in previous_indices]

    def sample_ranges(self) -> List[Tuple[int, int]]:
        start_range = 0
        ranges = []
        for group_key, sample_group in groupby(self.sample_indices):
            end_range = start_range + len(list(sample_group))
            ranges.append((start_range, end_range))
            start_range = end_range

        return ranges

    def select_index(self, index: int) -> Hypothesis:
        return Hypothesis(
            sequence=self.sequences[index],
            score=float(self.scores[index]),
            length=self.lengths[index],
            sample_index=self.sample_indices[index],
        )

    def select_best(self) -> BatchHypotheses:
        best_indices = []
        for range_start, range_end in self.sample_ranges():
            sample_scores = self.scores[range_start:range_end]
            best_index = sample_scores.argmax()
            best_indices.append(range_start + best_index)
        best_indices = torch.tensor(best_indices)
        return BatchHypotheses(
            sequences=self.sequences.index_select(dim=0, index=best_indices),
            scores=self.scores.index_select(dim=0, index=best_indices),
            lengths=[self.lengths[index] for index in best_indices],
            sample_indices=[self.sample_indices[index] for index in best_indices],
        )
