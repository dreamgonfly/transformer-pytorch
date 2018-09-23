from evaluator import Evaluator
from predictors import Predictor
from datasets import TranslationDatasetOnTheFly
from torch import nn
from predictors import Predictor
from models import Transformer, TransformerEncoder, TransformerDecoder
from datasets import TokenizedTranslationDatasetOnTheFly, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator

tokenized_dataset = TokenizedTranslationDatasetOnTheFly('train')

source_generator = shared_tokens_generator(tokenized_dataset)
source_dictionary = IndexDictionaryOnTheFly(source_generator)
target_generator = shared_tokens_generator(tokenized_dataset)
target_dictionary = IndexDictionaryOnTheFly(target_generator)

source_embedding = nn.Embedding(num_embeddings=source_dictionary.vocabulary_size, embedding_dim=128)
target_embedding = nn.Embedding(num_embeddings=target_dictionary.vocabulary_size, embedding_dim=128)
encoder = TransformerEncoder(layers_count=1, d_model=128, heads_count=2, d_ff=128, dropout_prob=0.1, embedding=source_embedding)
decoder = TransformerDecoder(layers_count=1, d_model=128, heads_count=2, d_ff=128, dropout_prob=0.1, embedding=target_embedding)
model = Transformer(encoder, decoder)

predictor = Predictor(
    preprocess=IndexedInputTargetTranslationDatasetOnTheFly.preprocess(source_dictionary),
    postprocess=lambda x: ' '.join(target_dictionary.tokenify_indexes(x)),
    model=model,
    checkpoint_filepath='checkpoints/hi/epoch=009-val_loss=1.77e+02-val_perplexity=9.26e+76.pth'
)

evaluator = Evaluator(
    predictor=predictor,
    save_filepath='tt.txt'
)

val_dataset = TranslationDatasetOnTheFly('val')
bleu_score = evaluator.evaluate_dataset(val_dataset)
print("BLEU score :", bleu_score)