from pathlib import Path

from torch.utils.data import DataLoader

from transformer.data_models.translation_data import TranslationDataList
from transformer.dataset import TranslationDataset
from transformer.token_indexers.token_indexer import TokenIndexer
from transformer.token_indexers.vocabulary_indexer import VocabularyIndexer
from transformer.tokenizers.tokenizer_factory import create_tokenizer, TokenizerName, Language
from transformer.training.collate import variable_length_collate
from transformer.training.feeder import Feeder


class DataFeeder(Feeder):
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    source_token_indexer: TokenIndexer
    target_token_indexer: TokenIndexer

    def __init__(
        self,
        train_path: Path,
        val_path: Path,
        source_vocab_path: Path,
        target_vocab_path: Path,
        batch_size: int,
        max_length: int,
    ):
        source_tokenizer = create_tokenizer(TokenizerName("spacy"), Language("de"), lower=True)
        target_tokenizer = create_tokenizer(TokenizerName("spacy"), Language("en"), lower=True)

        self.source_token_indexer = VocabularyIndexer.load(source_vocab_path, source_tokenizer)
        self.target_token_indexer = VocabularyIndexer.load(target_vocab_path, target_tokenizer)

        train_data_list = TranslationDataList.load(train_path)
        val_data_list = TranslationDataList.load(val_path)

        train_dataset = TranslationDataset(
            train_data_list,
            self.source_token_indexer,
            self.target_token_indexer,
            max_length,
            start_token_index=2,
            end_token_index=3,
        )
        val_dataset = TranslationDataset(
            val_data_list,
            self.source_token_indexer,
            self.target_token_indexer,
            max_length,
            start_token_index=2,
            end_token_index=3,
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=variable_length_collate(
                variable_length_fields=train_dataset.variable_length_fields
            ),
            pin_memory=True,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=variable_length_collate(
                variable_length_fields=val_dataset.variable_length_fields
            ),
            pin_memory=True,
        )
