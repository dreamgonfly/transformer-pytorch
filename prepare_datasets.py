from datasets import TranslationDatasetOnTheFly, TokenizedTranslationDatasetOnTheFly, IndexedTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator
translation_dataset = TranslationDatasetOnTheFly('train')
print(translation_dataset[0])

tokenized_dataset = TokenizedTranslationDatasetOnTheFly('train')
print(tokenized_dataset[0])

source_generator = shared_tokens_generator(tokenized_dataset)
source_dictionary = IndexDictionaryOnTheFly(source_generator)
target_generator = shared_tokens_generator(tokenized_dataset)
target_dictionary = IndexDictionaryOnTheFly(target_generator)

indexed_dictinary = IndexedTranslationDatasetOnTheFly('train', source_dictionary, target_dictionary)
print(indexed_dictinary[0])