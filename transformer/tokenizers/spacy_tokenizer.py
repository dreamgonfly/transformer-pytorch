from typing import List

from transformer.tokenizers.tokenizer import Tokenizer
import spacy


class SpacyTokenizer(Tokenizer):
    def __init__(self, language: str):
        self.lang_model = spacy.load(language)

    def tokenize(self, text: str) -> List[str]:
        return [token.text for token in self.lang_model.tokenizer(text)]

    def detokenize(self, tokens: List[str]):
        return " ".join(tokens)