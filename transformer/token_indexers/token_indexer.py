from abc import ABC, abstractmethod
from typing import List


class TokenIndexer(ABC):
    @abstractmethod
    def encode_sentence(self, sentence: str) -> List[int]:
        pass

    @abstractmethod
    def decode_indices(self, indices: List[int]) -> str:
        pass

    @abstractmethod
    def num_tokens(self) -> int:
        pass

    @abstractmethod
    def encode_token_name(self, token_name: str) -> int:
        pass
