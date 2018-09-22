import torch
from torch import nn
import numpy as np
from collections import defaultdict

PAD_TOKEN_ID = 0


class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, inputs):
        # sources : (batch_size, sources_len)
        # inputs : (batch_size, targets_len - 1)
        batch_size, sources_len = sources.size()
        batch_size, inputs_len = inputs.size()

        self.sources_mask = self.pad_masking(sources, sources_len)
        self.memory_mask = self.pad_masking(sources, inputs_len)
        self.inputs_mask = self.subsequent_masking(inputs) | self.pad_masking(inputs, inputs_len)

        memory = self.encoder(sources, self.sources_mask)  # (batch_size, seq_len, d_model)
        outputs, state = self.decoder(inputs, memory, self.memory_mask, self.inputs_mask)  # (batch_size, seq_len, d_model)
        return outputs

    @staticmethod
    def pad_masking(x, target_len):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        padded_positions = x == PAD_TOKEN_ID  # (batch_size, seq_len)
        pad_mask = padded_positions.unsqueeze(1).expand(batch_size, target_len, seq_len)
        return pad_mask

    @staticmethod
    def subsequent_masking(x):
        # x: (batch_size, seq_len - 1)
        batch_size, seq_len = x.size()
        subsequent_mask = np.triu(np.ones(shape=(seq_len, seq_len)), k=1).astype('uint8')
        subsequent_mask = torch.tensor(subsequent_mask).to(x.device)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        return subsequent_mask


class TransformerEncoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )
        self.embedding = embedding

    def forward(self, sources, mask):
        """

        args:
           sources: embedded_sequence, (batch_size, seq_len, embed_size)
        """
        sources = self.embedding(sources)

        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(sources, mask)

        return sources


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, sources, sources_mask):
        # x: (batch_size, seq_len, d_model)

        sources = self.self_attention_layer(sources, sources, sources, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)

        return sources


class TransformerDecoder(nn.Module):

    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob, embedding):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads_count, d_ff, dropout_prob) for _ in range(layers_count)]
        )
        self.embedding = embedding
        self.generator = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)

    def forward(self, inputs, memory, memory_mask, inputs_mask=None, state=None):
        # inputs: (batch_size, seq_len - 1, d_model)
        # memory: (batch_size, seq_len, d_model)

        inputs = self.embedding(inputs)
        # if state is not None:
        #     inputs = torch.cat([state.previous_inputs, inputs], dim=1)
        #
        #     state.previous_inputs = inputs

        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            if state is None:
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask)
            else: # Use cache
                layer_cache = state.layer_caches[layer_index]
                # print('inputs_mask', inputs_mask)
                inputs = decoder_layer(inputs, memory, memory_mask, inputs_mask, layer_cache)

                state.update_state(
                    layer_index=layer_index,
                    layer_mode='self-attention',
                    key_projected=decoder_layer.self_attention_layer.sublayer.key_projected,
                    value_projected=decoder_layer.self_attention_layer.sublayer.value_projected,
                )
                state.update_state(
                    layer_index=layer_index,
                    layer_mode='memory-attention',
                    key_projected=decoder_layer.memory_attention_layer.sublayer.key_projected,
                    value_projected=decoder_layer.memory_attention_layer.sublayer.value_projected,
                )

        generated = self.generator(inputs)  # (batch_size, seq_len, vocab_size)
        return generated, state

    def init_decoder_state(self, **args):
        return DecoderState()


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob, mode='self-attention'), d_model)
        self.memory_attention_layer = Sublayer(MultiHeadAttention(heads_count, d_model, dropout_prob, mode='memory-attention'), d_model)
        self.pointwise_feedforward_layer = Sublayer(PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)

    def forward(self, inputs, memory, memory_mask, inputs_mask, layer_cache=None):
        # print('self attention')
        # print('inputs_mask', inputs_mask)
        inputs = self.self_attention_layer(inputs, inputs, inputs, inputs_mask, layer_cache)
        # print('memory attention')
        inputs = self.memory_attention_layer(inputs, memory, memory, memory_mask, layer_cache)
        inputs = self.pointwise_feedforward_layer(inputs)
        return inputs


class Sublayer(nn.Module):

    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()

        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class LayerNormalization(nn.Module):

    def __init__(self, features_count, epsilon=1e-6):
        super(LayerNormalization, self).__init__()

        self.gain = nn.Parameter(torch.ones(features_count))
        self.bias = nn.Parameter(torch.zeros(features_count))
        self.epsilon = epsilon

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class MultiHeadAttention(nn.Module):

    def __init__(self, heads_count, d_model, dropout_prob, mode='self-attention'):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads_count == 0
        assert mode in ('self-attention', 'memory-attention')

        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.mode = mode
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)

        self.attention = None
        # For cache
        self.key_projected = None
        self.value_projected = None

    def forward(self, query, key, value, mask=None, layer_cache=None):
        """

        Args:
            query: (batch_size, query_len, model_dim)
            key: (batch_size, key_len, model_dim)
            value: (batch_size, value_len, model_dim)
            mask: (batch_size, query_len, key_len)
            state: DecoderState
        """
        # print('attention mask', mask)
        batch_size, query_len, d_model = query.size()

        d_head = d_model // self.heads_count

        query_projected = self.query_projection(query)
        # print('query_projected', query_projected.shape)
        if layer_cache is None or layer_cache[self.mode] is None:  # Don't use cache
            key_projected = self.key_projection(key)
            value_projected = self.value_projection(value)
        else:  # Use cache
            if self.mode == 'self-attention':
                key_projected = self.key_projection(key)
                value_projected = self.value_projection(value)

                key_projected = torch.cat([key_projected, layer_cache[self.mode]['key_projected']], dim=1)
                value_projected = torch.cat([value_projected, layer_cache[self.mode]['value_projected']], dim=1)
            elif self.mode == 'memory-attention':
                key_projected = layer_cache[self.mode]['key_projected']
                value_projected = layer_cache[self.mode]['value_projected']

        # For cache
        self.key_projected = key_projected
        self.value_projected = value_projected

        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(batch_size, query_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, query_len, d_head)
        # print('query_heads', query_heads.shape)
        # print(batch_size, key_len, self.heads_count, d_head)
        # print(key_projected.shape)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, key_len, d_head)
        value_heads = value_projected.view(batch_size, value_len, self.heads_count, d_head).transpose(1, 2)  # (batch_size, heads_count, value_len, d_head)

        attention_weights = self.scaled_dot_product(query_heads, key_heads)  # (batch_size, heads_count, query_len, key_len)

        if mask is not None:
            # print('mode', self.mode)
            # print('mask', mask.shape)
            # print('attention_weights', attention_weights.shape)
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_expanded, -1e18)

        self.attention = self.softmax(attention_weights)  # Save attention to the object
        # print('attention_weights', attention_weights.shape)
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads)  # (batch_size, heads_count, query_len, d_head)
        # print('context_heads', context_heads.shape)
        context_sequence = context_heads.transpose(1, 2).contiguous()  # (batch_size, query_len, heads_count, d_head)
        context = context_sequence.view(batch_size, query_len, d_model)  # (batch_size, query_len, d_model)
        final_output = self.final_projection(context)
        # print('final_output', final_output.shape)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        """

        Args:
             query_heads: (batch_size, heads_count, query_len, d_head)
             key_heads: (batch_size, heads_count, key_len, d_head)
        """
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):

    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """

        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)


class DecoderState:

    def __init__(self):
        self.previous_inputs = torch.tensor([])
        self.layer_caches = defaultdict(lambda: {'self-attention': None, 'memory-attention': None})

    def update_state(self, layer_index, layer_mode, key_projected, value_projected):
        self.layer_caches[layer_index][layer_mode] = {
            'key_projected': key_projected,
            'value_projected': value_projected
        }

    # def repeat_beam_size_times(self, beam_size): # memory만 repeat하면 되는데 state에 memory는 넣지 않기로 했다.
    #     self.
    #     self.src = self.src.data.repeat(beam_size, 1)

    def beam_update(self, positions):
        for layer_index in self.layer_caches:
            for mode in ('self-attention', 'memory-attention'):
                if self.layer_caches[layer_index][mode] is not None:
                    for projection in self.layer_caches[layer_index][mode]:
                        cache = self.layer_caches[layer_index][mode][projection]
                        if cache is not None:
                            cache.data.copy_(cache.data.index_select(0, positions))
