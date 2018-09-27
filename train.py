from models import build_model
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary
from losses import TokenCrossEntropyLoss, LabelSmoothingLoss
from metrics import AccuracyMetric
from optimizers import NoamOptimizer
from trainer import EpochSeq2SeqTrainer, input_target_collate_fn
from utils.log import get_logger

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
import json
import random

parser = ArgumentParser(description='Train Transformer')
parser.add_argument('--config', type=str, default=None)

parser.add_argument('--data_dir', type=str, default='data/example/processed')
parser.add_argument('--save_config', type=str, default=None)
parser.add_argument('--save_checkpoint', type=str, default=None)
parser.add_argument('--save_log', type=str, default=None)

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--dataset_limit', type=int, default=None)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--vocabulary_size', type=int, default=None)
parser.add_argument('--positional_encoding', action='store_true')

parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--layers_count', type=int, default=1)
parser.add_argument('--heads_count', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout_prob', type=float, default=0.1)

parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_grads', action='store_true')

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

    logger = get_logger(run_name, save_log=config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(config)

    logger.info('Constructing dictionaries...')
    source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
    target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])
    logger.info(f'Source dictionary vocabulary : {source_dictionary.vocabulary_size} tokens')
    logger.info(f'Target dictionary vocabulary : {target_dictionary.vocabulary_size} tokens')

    logger.info('Building model...')
    model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

    logger.info(model)
    logger.info('Encoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.encoder.parameters()])))
    logger.info('Decoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.decoder.parameters()])))
    logger.info('Total : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.parameters()])))

    logger.info('Loading datasets...')
    train_dataset = IndexedInputTargetTranslationDataset(
        data_dir=config['data_dir'],
        phase='train',
        vocabulary_size=config['vocabulary_size'],
        limit=config['dataset_limit'])

    val_dataset = IndexedInputTargetTranslationDataset(
        data_dir=config['data_dir'],
        phase='val',
        vocabulary_size=config['vocabulary_size'],
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

    if config['label_smoothing'] > 0.0:
        loss_function = LabelSmoothingLoss(label_smoothing=config['label_smoothing'],
                                           vocabulary_size=target_dictionary.vocabulary_size)
    else:
        loss_function = TokenCrossEntropyLoss()

    accuracy_function = AccuracyMetric()

    if config['optimizer'] == 'Noam':
        optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError()

    logger.info('Start training...')
    trainer = EpochSeq2SeqTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        metric_function=accuracy_function,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        save_config=config['save_config'],
        save_checkpoint=config['save_checkpoint'],
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
