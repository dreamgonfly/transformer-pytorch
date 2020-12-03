from pathlib import Path
from typing import Dict, List

from transformer.token_indexers.token_indexer import TokenIndexer
from transformer.tokenizers.tokenizer import Tokenizer

PAD_TOKEN_NAME = "PAD"
UNKNOWN_TOKEN_NAME = "UNKNOWN"
SENTNECE_START_TOKEN_NAME = "SENTENCE_START"
SENTNECE_END_TOKEN_NAME = "SENTENCE_END"


class VocabularyIndexer(TokenIndexer):
    tokenizer: Tokenizer
    token_to_index: Dict[str, int]
    index_to_token: Dict[int, str]
    token_name_to_index: Dict[str, int]
    index_to_token_name: Dict[int, str]

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_to_index: Dict[str, int],
        index_to_token: Dict[int, str],
        token_name_to_index: Dict[str, int],
        index_to_token_name: Dict[int, str],
    ):
        self.tokenizer = tokenizer
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.token_name_to_index = token_name_to_index
        self.index_to_token_name = index_to_token_name

    def encode_sentence(self, sentence: str) -> List[int]:
        tokens = self.tokenizer.tokenize(sentence)
        token_indicies = [self.encode_token(token) for token in tokens]
        return token_indicies

    def decode_indices(self, indices: List[int]) -> str:
        tokens = [self.decode_index(token_index) for token_index in indices]
        sentence = self.tokenizer.detokenize(tokens)
        return sentence

    def decode_indices_clean(self, indices: List[int]) -> str:
        pad_token_index = self.encode_token_name(PAD_TOKEN_NAME)
        start_token_index = self.encode_token_name(SENTNECE_START_TOKEN_NAME)
        end_token_index = self.encode_token_name(SENTNECE_END_TOKEN_NAME)
        non_display_indices = [pad_token_index, start_token_index, end_token_index]
        clean_indices = [index for index in indices if index not in non_display_indices]
        clean_tokens = [self.decode_index(token_index) for token_index in clean_indices]
        sentence = self.tokenizer.detokenize(clean_tokens)
        return sentence

    def encode_token(self, token: str) -> int:
        return self.token_to_index.get(token, self.token_name_to_index[UNKNOWN_TOKEN_NAME])

    def decode_index(self, index: int) -> str:
        return self.index_to_token[index]

    def encode_token_name(self, token_name: str) -> int:
        return self.token_name_to_index[token_name]

    def save(self, path: Path) -> None:
        with path.open("w") as file:
            for token_index in sorted(self.index_to_token.keys()):
                token = self.decode_index(token_index)
                token_name = self.index_to_token_name.get(token_index, "")
                file.write(f"{token_index}\t{token}\t{token_name}\n")

    @classmethod
    def load(cls, path: Path, tokenizer: Tokenizer):
        token_to_index = {}
        index_to_token = {}
        token_name_to_index = {}
        index_to_token_name = {}

        with path.open() as file:
            for line in file:
                token_index, token, token_name = line.rstrip("\n").split("\t")
                token_to_index[token] = int(token_index)
                index_to_token[int(token_index)] = token
                if token_name:
                    token_name_to_index[token_name] = int(token_index)
                    index_to_token_name[int(token_index)] = token_name
        return cls(
            tokenizer, token_to_index, index_to_token, token_name_to_index, index_to_token_name
        )

    def num_tokens(self):
        return len(self.token_to_index)
