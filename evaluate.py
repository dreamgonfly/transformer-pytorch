from evaluator import Evaluator
from predictors import Predictor
from models import build_model
from datasets import TranslationDataset
from datasets import IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionary

from argparse import ArgumentParser
import json
from datetime import datetime

parser = ArgumentParser(description='Predict translation')
parser.add_argument('--save_result', type=str, default=None)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)

args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

print('Constructing dictionaries...')
source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])

print('Building model...')
model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

predictor = Predictor(
    preprocess=IndexedInputTargetTranslationDatasetOnTheFly.preprocess(source_dictionary),
    postprocess=lambda x: ' '.join([token for token in target_dictionary.tokenify_indexes(x) if token != '<EndSent>']),
    model=model,
    checkpoint_filepath=args.checkpoint
)

timestamp = datetime.now()
if args.save_result is None:
    eval_filepath = 'logs/eval-{config}-time={timestamp}.csv'.format(
        config=args.config.replace('/', '-'),
        timestamp=timestamp.strftime("%Y_%m_%d_%H_%M_%S"))
else:
    eval_filepath = args.save_result

evaluator = Evaluator(
    predictor=predictor,
    save_filepath=eval_filepath
)

print('Evaluating...')
val_dataset = TranslationDataset(config['data_dir'], 'val')
bleu_score = evaluator.evaluate_dataset(val_dataset)
print('Evaluation time :', datetime.now() - timestamp)

print("BLEU score :", bleu_score)


