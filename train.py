from datasets import TokenizedTranslationDatasetOnTheFly, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionaryOnTheFly, shared_tokens_generator
from models import TransformerEncoder, TransformerDecoder, Transformer
from losses import MaskedCrossEntropyLoss, LabelSmoothingLoss
from torch import nn
from torch.optim import Adam
import torch
from trainer import EpochSeq2SeqTrainer, input_target_collate_fn
from torch.utils.data import DataLoader
from embeddings import PositionalEncoding

tokenized_dataset = TokenizedTranslationDatasetOnTheFly('train')

source_generator = shared_tokens_generator(tokenized_dataset)
source_dictionary = IndexDictionaryOnTheFly(source_generator)
target_generator = shared_tokens_generator(tokenized_dataset)
target_dictionary = IndexDictionaryOnTheFly(target_generator)

indexed_datset = IndexedInputTargetTranslationDatasetOnTheFly('train', source_dictionary, target_dictionary)

embedding = PositionalEncoding(num_embeddings=source_dictionary.vocabulary_size, embedding_dim=128, dim=128) # nn.Embedding(num_embeddings=source_dictionary.vocabulary_size, embedding_dim=128)
encoder = TransformerEncoder(layers_count=1, d_model=128, heads_count=2, d_ff=128, dropout_prob=0.1, embedding=embedding)
decoder = TransformerDecoder(layers_count=1, d_model=128, heads_count=2, d_ff=128, dropout_prob=0.1, embedding=embedding)
model = Transformer(encoder, decoder)

example_source, example_input, example_target = indexed_datset[0]
example_sources = torch.tensor([example_source])
example_inputs = torch.tensor([example_input])
print(model(example_sources, example_inputs).size())

train_dataset =IndexedInputTargetTranslationDatasetOnTheFly('train', source_dictionary, target_dictionary, limit=100)
val_dataset =IndexedInputTargetTranslationDatasetOnTheFly('val', source_dictionary, target_dictionary, limit=100)

trainer = EpochSeq2SeqTrainer(
    model,
    train_dataloader=DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=input_target_collate_fn),
    val_dataloader=DataLoader(val_dataset, batch_size=10, collate_fn=input_target_collate_fn),
    loss_function=LabelSmoothingLoss(label_smoothing=0.1, vocabulary_size=target_dictionary.vocabulary_size),
    optimizer=Adam(model.parameters()),
    logger=None,
    run_name='hi',
    config={'device':'cpu', 'print_every':1, 'save_every':1, 'clip_grads':True}
)

trainer.run(10)