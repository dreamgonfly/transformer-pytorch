from collections import Counter
from os.path import dirname, abspath, join, exists
from os import makedirs

BASE_DIR = dirname(abspath(__file__))

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'


class IndexDictionary:

    def __init__(self, iterable=None, mode='shared', vocabulary_size=None):

        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

        # On-the-fly mode
        if iterable is not None:

            self.vocab_tokens, self.token_counts = self._build_vocabulary(iterable, vocabulary_size)
            self.token_index_dict = {token: index for index, token in enumerate(self.vocab_tokens)}
            self.vocabulary_size = len(self.vocab_tokens)

        self.mode = mode

    def token_to_index(self, token):
        try:
            return self.token_index_dict[token]
        except KeyError:
            return self.token_index_dict[UNK_TOKEN]

    def index_to_token(self, index):
        if index >= self.vocabulary_size:
            return self.vocab_tokens[UNK_TOKEN]
        else:
            return self.vocab_tokens[index]

    def index_sentence(self, sentence):
        return [self.token_to_index(token) for token in sentence]

    def tokenify_indexes(self, token_indexes):
        return [self.index_to_token(token_index) for token_index in token_indexes]

    def _build_vocabulary(self, iterable, vocabulary_size):

        counter = Counter()
        for token in iterable:
            counter[token] += 1

        if vocabulary_size is not None:
            most_commons = counter.most_common(vocabulary_size - len(self.special_tokens))
            frequent_tokens = [token for token, count in most_commons]
            vocab_tokens = self.special_tokens + frequent_tokens
            token_counts = [0] * len(self.special_tokens) + [count for token, count in most_commons]
        else:
            all_tokens = [token for token, count in counter.items()]
            vocab_tokens = self.special_tokens + all_tokens
            token_counts = [0] * len(self.special_tokens) + [count for token, count in counter.items()]

        return vocab_tokens, token_counts

    def save(self, data_dir):

        vocabulary_filepath = join(data_dir, f'vocabulary-{self.mode}.txt')
        with open(vocabulary_filepath, 'w') as file:
            for vocab_index, (vocab_token, count) in enumerate(zip(self.vocab_tokens, self.token_counts)):
                file.write(str(vocab_index) + '\t' + vocab_token + '\t' + str(count) + '\n')

    @classmethod
    def load(cls, data_dir, mode='shared', vocabulary_size=None):
        vocabulary_filepath = join(data_dir, f'vocabulary-{mode}.txt')

        vocab_tokens = {}
        token_counts = []
        with open(vocabulary_filepath) as file:
            for line in file:
                vocab_index, vocab_token, count = line.strip().split('\t')
                vocab_index = int(vocab_index)
                vocab_tokens[vocab_index] = vocab_token
                token_counts.append(int(count))

        if vocabulary_size is not None:
            vocab_tokens = {k: v for k, v in vocab_tokens.items() if k < vocabulary_size}
            token_counts = token_counts[:vocabulary_size]

        instance = cls(mode=mode)
        instance.vocab_tokens = vocab_tokens
        instance.token_counts = token_counts
        instance.token_index_dict = {token: index for index, token in vocab_tokens.items()}
        instance.vocabulary_size = len(vocab_tokens)

        return instance

