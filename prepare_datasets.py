from datasets import TranslationDatasetOnTheFly, TokenizedTranslationDatasetOnTheFly, IndexedTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator

from argparse import ArgumentParser

parser = ArgumentParser('Prepare datasets')
parser.add_argument('--train_source', type=str, default='data/example/raw/src-train.txt')
parser.add_argument('--train_target', type=str, default='data/example/raw/tgt-train.txt')
parser.add_argument('--val_source', type=str, default='data/example/raw/src-val.txt')
parser.add_argument('--val_target', type=str, default='data/example/raw/tgt-val.txt')
parser.add_argument('--save_data', type=str, default='data/example/processed')

args = parser.parse_args()

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