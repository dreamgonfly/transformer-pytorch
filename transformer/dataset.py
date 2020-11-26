from typing import List, Tuple

from torch.utils.data import Dataset

from transformer.data_models.translation_data import TranslationDataList
from transformer.token_indexers.token_indexer import TokenIndexer


class TranslationDataset(Dataset):
    variable_length_fields = {"source_token_indices", "target_token_indices"}
    data: List[Tuple[List[int], List[int]]]

    def __init__(
        self,
        data_list: TranslationDataList,
        source_token_indexer: TokenIndexer,
        target_token_indexer: TokenIndexer,
        max_length: int,
    ):
        data = []
        for pair in data_list.pairs:
            source_indices = source_token_indexer.encode_sentence(
                pair.source.strip('"').replace("\t", "")
            )
            target_indices = target_token_indexer.encode_sentence(
                pair.target.strip('"').replace("\t", "")
            )
            if len(source_indices) > max_length:
                continue
            if len(target_indices) > max_length:
                continue
            data.append((source_indices, target_indices))

        self.data = data

    def __getitem__(self, item):
        source_indices, target_indices = self.data[item]
        source_indices = [2] + source_indices + [3]
        target_indices = [2] + target_indices + [3]
        inputs = {"source_token_indices": source_indices, "length": len(source_indices)}
        targets = {"target_token_indices": target_indices, "length": len(target_indices)}
        return inputs, targets

    def __len__(self):
        return len(self.data)
