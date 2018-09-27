from predictors import Predictor
from models import build_model
from datasets import IndexedInputTargetTranslationDataset
from dictionaries import IndexDictionary

from argparse import ArgumentParser
import json

parser = ArgumentParser(description='Predict translation')
parser.add_argument('--source', type=str)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--num_candidates', type=int, default=3)

args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

print('Constructing dictionaries...')
source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])

print('Building model...')
model = build_model(config, source_dictionary.vocabulary_size, target_dictionary.vocabulary_size)

predictor = Predictor(
    preprocess=IndexedInputTargetTranslationDataset.preprocess(source_dictionary),
    postprocess=lambda x: ' '.join([token for token in target_dictionary.tokenify_indexes(x) if token != '<EndSent>']),
    model=model,
    checkpoint_filepath=args.checkpoint
)

for index, candidate in enumerate(predictor.predict_one(args.source, num_candidates=args.num_candidates)):
    print(f'Candidate {index} : {candidate}')
