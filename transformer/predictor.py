from pathlib import Path
from typing import Optional
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from transformer.dataset import TranslationDataset
from transformer.model.transformer import Transformer
from transformer.searching.attention_scorer import AttentionScorer
from transformer.searching.beam_searcher import BeamSearcher
from transformer.searching.search_parameters import BeamSearchParameters
from transformer.token_indexers.token_indexer import TokenIndexer
from transformer.token_indexers.vocabulary_indexer import (
    VocabularyIndexer,
    PAD_TOKEN_NAME,
    SENTNECE_START_TOKEN_NAME,
    SENTNECE_END_TOKEN_NAME,
)
from transformer.tokenizers.tokenizer_factory import create_tokenizer, TokenizerName, Language
from transformer.training.collate import variable_length_collate


class Translator:
    model: Transformer
    source_token_indexer: TokenIndexer
    target_token_indexer: TokenIndexer
    searcher: BeamSearcher
    device: torch.device

    def __init__(
        self,
        model: Transformer,
        source_token_indexer: TokenIndexer,
        target_token_indexer: TokenIndexer,
        searcher: BeamSearcher,
        device: torch.device,
    ):
        self.model = model.eval()
        self.source_token_indexer = source_token_indexer
        self.target_token_indexer = target_token_indexer
        self.searcher = searcher
        self.device = device

    def translate(self, source: str, parameters: BeamSearchParameters):
        source_indices = self.source_token_indexer.encode_sentence(source)
        inputs = {
            "source_token_indices": source_indices,
            "source_length": len(source_indices),
        }
        collate = variable_length_collate(
            variable_length_fields=TranslationDataset.variable_length_fields
        )
        inputs = collate([inputs])
        with torch.no_grad():
            sources = pack_padded_sequence(
                inputs["source_token_indices"],
                inputs["source_length"].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            sources.to(device=self.device)
            memories, _ = self.model.encode(sources)
            batch_hypotheses = self.searcher.search(memories, parameters)

        hypothesis = batch_hypotheses.select_best().unbatchify()[0]
        single_prediction = hypothesis.sequence.tolist()
        text_prediction = self.target_token_indexer.decode_indices_clean(single_prediction)
        return text_prediction

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        source_vocab_path: Optional[Path],
        target_vocab_path: Optional[Path],
        device: torch.device,
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        if source_vocab_path is None:
            source_vocab_path = Path(config["source_vocab_path"])
        if target_vocab_path is None:
            target_vocab_path = Path(config["target_vocab_path"])
        num_layers = config["num_layers"]

        source_tokenizer = create_tokenizer(TokenizerName("spacy"), Language("de"), lower=True)
        target_tokenizer = create_tokenizer(TokenizerName("spacy"), Language("en"), lower=True)

        source_token_indexer = VocabularyIndexer.load(source_vocab_path, source_tokenizer)
        target_token_indexer = VocabularyIndexer.load(target_vocab_path, target_tokenizer)

        pad_token_index = source_token_indexer.encode_token_name(PAD_TOKEN_NAME)

        model = Transformer(
            source_token_indexer.num_tokens(),
            target_token_indexer.num_tokens(),
            pad_token_index=pad_token_index,
            d_model=512,
            d_ff=2048,
            num_layers=num_layers,
            n_heads=8,
            dropout=0.1,
            num_positions=200,
            input_target_weight_sharing=True,
            source_target_weight_sharing=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device=device)

        searcher = BeamSearcher(
            attention_scorer=AttentionScorer(model, device),
            start_token_id=target_token_indexer.encode_token_name(SENTNECE_START_TOKEN_NAME),
            end_token_id=target_token_indexer.encode_token_name(SENTNECE_END_TOKEN_NAME),
            pad_token_id=target_token_indexer.encode_token_name(PAD_TOKEN_NAME),
        )

        return cls(
            model=model,
            source_token_indexer=source_token_indexer,
            target_token_indexer=target_token_indexer,
            searcher=searcher,
            device=device,
        )
