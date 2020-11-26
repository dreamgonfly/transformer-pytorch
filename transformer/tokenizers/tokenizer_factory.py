from enum import Enum

from transformer.tokenizers.spacy_tokenizer import SpacyTokenizer
from transformer.tokenizers.tokenizer import Tokenizer


class TokenizerName(str, Enum):
    SPACY = "spacy"


class Language(str, Enum):
    EN = "en"
    DE = "de"


def create_tokenizer(name: TokenizerName, language: Language) -> Tokenizer:
    if name == TokenizerName.SPACY:
        return SpacyTokenizer(language.value)
    else:
        raise NotImplementedError
