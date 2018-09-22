from torch import nn
from predictors import Predictor
from models import Transformer, TransformerEncoder, TransformerDecoder
from datasets import TokenizedTranslationDatasetOnTheFly, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator

tokenized_dataset = TokenizedTranslationDatasetOnTheFly('train')

source_generator = shared_tokens_generator(tokenized_dataset)
source_dictionary = IndexDictionaryOnTheFly(source_generator)

embedding = nn.Embedding(num_embeddings=source_dictionary.vocabulary_size, embedding_dim=128)
encoder = TransformerEncoder(layers_count=1, d_model=128, heads_count=2, d_ff=128, dropout_prob=0.1, embedding=embedding)
decoder = TransformerDecoder(layers_count=1, d_model=128, heads_count=2, d_ff=128, dropout_prob=0.1, embedding=embedding)
model = Transformer(encoder, decoder)

predictor = Predictor(
    preprocess=IndexedInputTargetTranslationDatasetOnTheFly.preprocess(source_dictionary),
    postprocess=lambda x: x,
    model=model,
    checkpoint_filepath='checkpoints/hi/epoch=009-val_loss=1.77e+02-val_perplexity=9.26e+76.pth'
)

print(predictor.predict_one("Orlando Bloom and Miranda Kerr still love each other"))