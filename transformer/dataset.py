from typing import List, Tuple

from torch.utils.data import Dataset

from transformer.data_models.translation_data import TranslationDataList
from transformer.token_indexers.token_indexer import TokenIndexer


class TranslationDataset(Dataset):
    variable_length_fields = {"source_token_indices", "input_token_indices", "target_token_indices"}
    data: List[Tuple[List[int], List[int]]]
    start_token_index: int
    end_token_index: int

    def __init__(
        self,
        data_list: TranslationDataList,
        source_token_indexer: TokenIndexer,
        target_token_indexer: TokenIndexer,
        max_length: int,
        start_token_index: int,
        end_token_index: int,
    ):
        data = []
        for pair in data_list.pairs:
            source_indices = source_token_indexer.encode_sentence(pair.source)
            target_indices = target_token_indexer.encode_sentence(pair.target)
            if len(source_indices) > max_length or len(target_indices) > max_length:
                continue
            data.append((source_indices, target_indices))

        self.data = data
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index

    def __getitem__(self, item):
        source_indices, target_indices = self.data[item]
        input_token_indices = [self.start_token_index] + target_indices
        target_token_indices = target_indices + [self.end_token_index]
        inputs = {
            "source_token_indices": source_indices,
            "source_length": len(source_indices),
            "input_token_indices": input_token_indices,
            "input_length": len(input_token_indices),
        }
        targets = {
            "target_token_indices": target_token_indices,
            "length": len(target_token_indices),
        }
        return inputs, targets

    def __len__(self):
        return len(self.data)
