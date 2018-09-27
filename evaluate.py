from evaluator import Evaluator
from predictors import Predictor
from models import build_model
from datasets import TranslationDatasetOnTheFly
from datasets import TokenizedTranslationDatasetOnTheFly, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator, source_tokens_generator, target_tokens_generator

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
tokenized_dataset = TokenizedTranslationDatasetOnTheFly('train')

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

print('Building model...')
model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

predictor = Predictor(
    preprocess=IndexedInputTargetTranslationDatasetOnTheFly.preprocess(source_dictionary),
    postprocess=lambda x: ' '.join([token for token in target_dictionary.tokenify_indexes(x) if token != '<EndSent>']),
    model=model,
    checkpoint_filepath=args.checkpoint
)

timestamp = datetime.now()
eval_filename = 'logs/eval-{config}-time={timestamp}.csv'.format(
    config=args.config.replace('/', '-'),
    timestamp=timestamp.strftime("%Y_%m_%d_%H_%M_%S"))

evaluator = Evaluator(
    predictor=predictor,
    save_filepath=eval_filename
)

print('Evaluating...')
val_dataset = TranslationDatasetOnTheFly('val')
bleu_score = evaluator.evaluate_dataset(val_dataset)
print('Evaluation time :', datetime.now() - timestamp)

print("BLEU score :", bleu_score)


