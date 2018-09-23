from datasets import TokenizedTranslationDatasetOnTheFly, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator, source_tokens_generator, target_tokens_generator
from embeddings import PositionalEncoding
from models import TransformerEncoder, TransformerDecoder, Transformer
from losses import MaskedCrossEntropyLoss, LabelSmoothingLoss
from optimizers import NoamOptimizer
from trainer import EpochSeq2SeqTrainer, input_target_collate_fn
from utils.log import get_logger

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
import json
import random

parser = ArgumentParser(description='Train Transformer')
parser.add_argument('--config', type=str, default=None)

parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--dataset_limit', type=int, default=None)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--vocabulary_size', type=int, default=None)
parser.add_argument('--share_dictionary', type=bool, default=False)
parser.add_argument('--positional_encoding', type=bool, default=True)

parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--layers_count', type=int, default=1)
parser.add_argument('--heads_count', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout_prob', type=float, default=0.1)

parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Noam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_grads', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)


def run_trainer(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_name_format = (
        "d_model={d_model}-"
        "layers_count={layers_count}-"
        "heads_count={heads_count}-"
        "pe={positional_encoding}-"
        "optimizer={optimizer}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger = get_logger(run_name)
    logger.info(str(config))

    tokenized_dataset = TokenizedTranslationDatasetOnTheFly('train', limit=config['dataset_limit'])

    if config['share_dictionary']:
        source_generator = shared_tokens_generator(tokenized_dataset)
        source_dictionary = IndexDictionaryOnTheFly(source_generator, vocabulary_size=config['vocabulary_size'])
        target_generator = shared_tokens_generator(tokenized_dataset)
        target_dictionary = IndexDictionaryOnTheFly(target_generator, vocabulary_size=config['vocabulary_size'])
    else:
        source_generator = source_tokens_generator(tokenized_dataset)
        source_dictionary = IndexDictionaryOnTheFly(source_generator, vocabulary_size=config['vocabulary_size'])
        target_generator = target_tokens_generator(tokenized_dataset)
        target_dictionary = IndexDictionaryOnTheFly(target_generator, vocabulary_size=config['vocabulary_size'])

    if config['positional_encoding']:
        source_embedding = PositionalEncoding(
            num_embeddings=source_dictionary.vocabulary_size,
            embedding_dim=config['d_model'],
            dim=config['d_model'])  # why dim?
        target_embedding = PositionalEncoding(
            num_embeddings=target_dictionary.vocabulary_size,
            embedding_dim=config['d_model'],
            dim=config['d_model'])  # why dim?
    else:
        source_embedding = nn.Embedding(
            num_embeddings=source_dictionary.vocabulary_size,
            embedding_dim=config['d_model'])
        target_embedding = nn.Embedding(
            num_embeddings=target_dictionary.vocabulary_size,
            embedding_dim=config['d_model'])

    encoder = TransformerEncoder(
        layers_count=config['layers_count'],
        d_model=config['d_model'],
        heads_count=config['heads_count'],
        d_ff=config['d_ff'],
        dropout_prob=config['dropout_prob'],
        embedding=source_embedding)

    decoder = TransformerDecoder(
        layers_count=config['layers_count'],
        d_model=config['d_model'],
        heads_count=config['heads_count'],
        d_ff=config['d_ff'],
        dropout_prob=config['dropout_prob'],
        embedding=target_embedding)

    model = Transformer(encoder, decoder)

    train_dataset = IndexedInputTargetTranslationDatasetOnTheFly(
        'train',
        source_dictionary,
        target_dictionary,
        limit=config['dataset_limit'])

    val_dataset = IndexedInputTargetTranslationDatasetOnTheFly(
        'val',
        source_dictionary,
        target_dictionary,
        limit=config['dataset_limit'])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=input_target_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=input_target_collate_fn)

    loss_function = LabelSmoothingLoss(label_smoothing=config['label_smoothing'], vocabulary_size=target_dictionary.vocabulary_size)

    if config['optimizer'] == 'Noam':
        optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError()

    trainer = EpochSeq2SeqTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        config=config
    )

    trainer.run(config['epochs'])

    return trainer


if __name__ == '__main__':

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    run_trainer(config)