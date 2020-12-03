from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from transformer.searching.attention_scorer import AttentionScorer
from transformer.searching.batch_hypotheses import BatchHypotheses
from transformer.searching.hypothesis import Hypothesis
from transformer.searching.search_parameters import BeamSearchParameters


class BeamSearcher:
    attention_scorer: AttentionScorer
    start_token_id: int
    end_token_id: int
    pad_token_id: int

    def __init__(
        self,
        attention_scorer: AttentionScorer,
        start_token_id: int,
        end_token_id: int,
        pad_token_id: int,
    ):
        self.attention_scorer = attention_scorer
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id

        self.maxpool = nn.MaxPool1d(kernel_size=10, stride=10, ceil_mode=True)

    def search(self, memories: PackedSequence, parameters: BeamSearchParameters) -> BatchHypotheses:
        # """Search decoded sequences with beam search.
        #
        # Args:
        #     memory: (batch_size, memory_length, hidden_size).
        #     memory_lengths: (batch_size,).
        #     parameters: Instance of BeamSearchParameters.
        #
        # Returns:
        #     BatchHypotheses
        # """

        batch_size = max(memories.batch_sizes)
        self._initialize_scorers(batch_size, memories)

        batch_hypotheses = BatchHypotheses.new(batch_size, self.start_token_id)
        running_hypotheses = batch_hypotheses.unbatchify()

        finished_hypotheses = self._run_hypotheses(running_hypotheses, parameters)

        return BatchHypotheses.batchify(finished_hypotheses, self.pad_token_id)

    def _initialize_scorers(self, batch_size: int, memories: PackedSequence):
        self.attention_scorer.initialize(batch_size, memories)

    def _run_hypotheses(
        self, running_hypotheses: List[Hypothesis], parameters: BeamSearchParameters
    ) -> List[Hypothesis]:
        finished_hypotheses = []
        for _ in range(parameters.max_length):
            next_hypotheses = self._advance(running_hypotheses, parameters)

            running_indices, ended_indices = self._detect_ending(next_hypotheses)
            running_hypotheses = [next_hypotheses[index] for index in running_indices]
            ended_hypotheses = [next_hypotheses[index] for index in ended_indices]

            ended_hypotheses = self._post_score_hypotheses(
                ended_indices, ended_hypotheses, parameters
            )

            self._update_scorer_indices(running_indices)
            finished_hypotheses.extend(ended_hypotheses)
            if not running_hypotheses:
                break

        finished_hypotheses.extend(self._finalize_hypotheses(running_hypotheses))
        return finished_hypotheses

    def _advance(
        self, hypotheses: List[Hypothesis], parameters: BeamSearchParameters
    ) -> List[Hypothesis]:

        batch_hypotheses = BatchHypotheses.batchify(hypotheses, self.pad_token_id)
        total_scores = self._score_next_step(batch_hypotheses.sequences)

        previous_hypothesis_indices, next_token_ids, next_scores = self._pick_next_beam(
            total_scores, batch_hypotheses.sample_ranges(), parameters.beam_size
        )

        self._update_scorers(previous_hypothesis_indices, next_token_ids, next_scores)

        batch_hypotheses.update(previous_hypothesis_indices, next_token_ids, next_scores)

        return batch_hypotheses.unbatchify()

    def _detect_ending(self, hypotheses: List[Hypothesis]) -> Tuple[List[int], List[int]]:
        remaining_hypotheses_indices = []
        ended_hypotheses_indices = []
        for index, hypothesis in enumerate(hypotheses):
            if hypothesis.sequence[-1] == self.end_token_id:
                ended_hypotheses_indices.append(index)
            else:
                remaining_hypotheses_indices.append(index)
        return remaining_hypotheses_indices, ended_hypotheses_indices

    def _post_score_hypotheses(
        self, indices: List[int], hypotheses: List[Hypothesis], parameters: BeamSearchParameters
    ) -> List[Hypothesis]:
        for index, hypothesis in zip(indices, hypotheses):
            score = hypothesis.score
            length = len(hypothesis.sequence) - 2  # exclude start and end of sentence tokens

            memory_length = self.attention_scorer.state["memory_lengths"][index].to("cpu")
            attention = torch.stack(self.attention_scorer.state["attention"])[:, :memory_length].to(
                "cpu"
            )

            post_score = self._post_score(
                score, length, attention, parameters.length_penalty, parameters.coverage_penalty
            )

            hypothesis.score = post_score
        return hypotheses

    def _update_scorer_indices(self, indices: List[int]) -> None:
        indices_tensor = torch.tensor(indices, device=self.attention_scorer.device)
        self.attention_scorer.select_indices(indices_tensor)

    def _finalize_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        for h in hypotheses:
            h.sequence = torch.cat([h.sequence, torch.tensor([self.end_token_id])])
        return hypotheses

    def _score_next_step(self, sequences: Tensor) -> Tensor:
        total_scores = self.attention_scorer.forward_step(sequences)
        return total_scores

    @staticmethod
    def _pick_next_beam(
        total_scores: Tensor, sample_ranges: List[Tuple[int, int]], beam_size: int
    ) -> Tuple[Tensor, Tensor, Tensor]:

        vocab_size = total_scores.size(1)

        samples_previous_hypothesis_indices = []
        samples_next_token_ids = []
        samples_next_scores = []

        for range_start, range_end in sample_ranges:
            sample_scores = total_scores[range_start:range_end]
            flattened_sample_scores = sample_scores.view(-1)
            top_beam = flattened_sample_scores.topk(beam_size)

            # Because of the flatten above, `top_beam.indices` is organized as:
            # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
            # where V is `vocab_size`` and K is `beam_size`
            sample_previous_hypothesis_indices = range_start + top_beam.indices // vocab_size
            sample_next_token_ids = top_beam.indices % vocab_size
            sample_next_scores = top_beam.values

            samples_previous_hypothesis_indices.append(sample_previous_hypothesis_indices)
            samples_next_token_ids.append(sample_next_token_ids)
            samples_next_scores.append(sample_next_scores)

        batch_previous_hypothesis_indices = torch.cat(samples_previous_hypothesis_indices)
        batch_next_token_ids = torch.cat(samples_next_token_ids)
        batch_next_scores = torch.cat(samples_next_scores)
        return batch_previous_hypothesis_indices, batch_next_token_ids, batch_next_scores

    def _update_scorers(
        self, previous_hypothesis_indices: Tensor, next_token_ids: Tensor, next_scores: Tensor,
    ):
        self.attention_scorer.select_indices(previous_hypothesis_indices)

    def _post_score(
        self,
        score: float,
        length: int,
        attention: Tensor,
        length_penalty: float,
        coverage_penalty: float,
    ):
        if length_penalty > 0:
            length_penalty_factor = self.length_penalty_score(length, length_penalty)
        else:
            length_penalty_factor = 1

        if coverage_penalty > 0:
            coverage_penalty_factor = self.coverage_penalty_score(attention, coverage_penalty)
        else:
            coverage_penalty_factor = 0

        score = score / length_penalty_factor + coverage_penalty_factor
        return score

    @staticmethod
    def length_penalty_score(length, length_penalty):
        return (5 + length) ** length_penalty / (5 + 1) ** length_penalty

    def coverage_penalty_score(self, attention, coverage_penalty):
        # attention: ["target_seq_len", "source_seq_len"]
        pooled_attention = self.maxpool(attention.unsqueeze(dim=0)).squeeze(dim=0)
        return coverage_penalty * torch.sum(
            torch.log(torch.clamp(pooled_attention.sum(dim=0), max=1.0))
        )
