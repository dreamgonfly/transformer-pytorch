from collections import Counter
from pathlib import Path
from typing import Dict

from typer import Option

from transformer.data_models.translation_data import TranslationDataList
from transformer.token_indexers.vocabulary_indexer import (
    PAD_TOKEN_NAME,
    UNKNOWN_TOKEN_NAME,
    SENTNECE_START_TOKEN_NAME,
    SENTNECE_END_TOKEN_NAME,
    VocabularyIndexer,
)
from transformer.tokenizers.tokenizer_factory import TokenizerName, create_tokenizer, Language


def build_vocab(
    data_list_path: Path,
    vocab_path: Path,
    tokenizer_name: TokenizerName = TokenizerName.SPACY,
    source_language: Language = Option(...),
    target_language: Language = Option(...),
    min_freq: int = Option(1),
):
    data_list = TranslationDataList.load(data_list_path)
    source_tokenizer = create_tokenizer(tokenizer_name, source_language, lower=True)
    target_tokenizer = create_tokenizer(tokenizer_name, target_language, lower=True)

    source_counter = Counter()
    target_counter = Counter()

    for data_pair in data_list.pairs:
        # Normalize
        # text = text.strip('"').replace("\t", "").lower()
        if data_pair.source == "" or data_pair.target == "":
            raise ValueError
        source_tokens = source_tokenizer.tokenize(data_pair.source)
        target_tokens = target_tokenizer.tokenize(data_pair.target)
        if len(source_tokens) > 100 or len(target_tokens) > 100:
            continue
        source_counter.update(source_tokens)
        target_counter.update(target_tokens)

    token_names = [
        (PAD_TOKEN_NAME, "[PAD]"),
        (UNKNOWN_TOKEN_NAME, "[UNK]"),
        (SENTNECE_START_TOKEN_NAME, "[SOS]"),
        (SENTNECE_END_TOKEN_NAME, "[EOS]"),
    ]

    token_name_to_index: Dict[str, int] = {}
    index_to_token_name: Dict[int, str] = {}

    token_to_index: Dict[str, int] = {}
    index_to_token: Dict[int, str] = {}

    for token_name, token in token_names:
        next_index = len(token_to_index)
        token_to_index[token] = next_index
        index_to_token[next_index] = token

        token_name_to_index[token_name] = next_index
        index_to_token_name[next_index] = token_name

    for token, count in source_counter.most_common(None):
        if count < min_freq:
            continue
        next_index = len(token_to_index)
        token_to_index[token] = next_index
        index_to_token[next_index] = token
    for token, count in target_counter.most_common(None):
        if token in token_to_index:
            continue
        if count < min_freq:
            continue
        next_index = len(token_to_index)
        token_to_index[token] = next_index
        index_to_token[next_index] = token

    vocabulary = VocabularyIndexer(
        source_tokenizer, token_to_index, index_to_token, token_name_to_index, index_to_token_name
    )

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary.save(vocab_path)

    print(f"Vocabulary {vocabulary.num_tokens()} lines written to {vocab_path}")
