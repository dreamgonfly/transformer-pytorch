from typing import List

from transformer.tokenizers.tokenizer import Tokenizer
import spacy


class SpacyTokenizer(Tokenizer):
    def __init__(self, language: str, lower: bool):
        self.lang_model = spacy.load(language)
        self.lower = lower

    def tokenize(self, text: str) -> List[str]:
        if self.lower:
            return [token.text.lower() for token in self.lang_model.tokenizer(text)]
        else:
            return [token.text for token in self.lang_model.tokenizer(text)]

    def detokenize(self, tokens: List[str]):
        return " ".join(tokens)
