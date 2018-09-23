from collections import Counter
from os.path import dirname, abspath, join, exists
from os import makedirs

BASE_DIR = dirname(abspath(__file__))

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'


def shared_tokens_generator(dataset):
    for source, target in dataset:
        for token in source:
            yield token
        for token in target:
            yield token


def source_tokens_generator(dataset):
    for source, target in dataset:
        for token in source:
            yield token


def target_tokens_generator(dataset):
    for source, target in dataset:
        for token in target:
            yield token


class IndexDictionaryOnTheFly:

    def __init__(self, iterable, vocabulary_size=None):

        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

        self.vocab_tokens = self._build_vocabulary(iterable, vocabulary_size)
        self.token_index_dict = {token: index for index, token in enumerate(self.vocab_tokens)}
        self.vocabulary_size = len(self.vocab_tokens)

    def token_to_index(self, token):
        try:
            return self.token_index_dict[token]
        except KeyError:
            return self.token_index_dict[UNK_TOKEN]

    def index_to_token(self, index):
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
        else:
            all_tokens = [token for token, count in counter.items()]
            vocab_tokens = self.special_tokens + all_tokens

        return vocab_tokens


class IndexDictionary:

    def __init__(self):

        vocabulary_dir = join(BASE_DIR, 'data', 'vocabulary')
        vocabbulary_filepath = join(vocabulary_dir, 'index_dictionary_vocabulary.txt')

        self.vocab_tokens = {}
        with open(vocabbulary_filepath) as file:
            for line in file:
                vocab_index, vocab_token, count = line.strip().split('\t')
                vocab_index = int(vocab_index)
                self.vocab_tokens[vocab_index] = vocab_token

        self.token_index_dict = {token: index for index, token in self.vocab_tokens.items()}

    def token_to_index(self, token):
        try:
            return self.token_index_dict[token]
        except KeyError:
            return self.token_index_dict[UNK_TOKEN]

    def index_to_token(self, index):
        return self.vocab_tokens[index]

    @staticmethod
    def prepare(iterable, vocabulary_size=None):

        special_tokens = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

        counter = Counter()
        for token in iterable:
            counter[token] += 1

        if vocabulary_size is not None:
            most_commons = counter.most_common(vocabulary_size - len(special_tokens))
            frequent_tokens = [token for token, count in most_commons]
            vocab_tokens = special_tokens + frequent_tokens
            token_counts = [0] * len(special_tokens) + [count for token, count in most_commons]
        else:
            all_tokens = [token for token, count in counter.items()]
            vocab_tokens = special_tokens + all_tokens
            token_counts = [0] * len(special_tokens) + [count for token, count in counter.items()]

        vocabulary_dir = join(BASE_DIR, 'data', 'vocabulary')
        if not exists(vocabulary_dir):
            makedirs(vocabulary_dir)

        vocabbulary_filepath = join(vocabulary_dir, 'index_dictionary_vocabulary.txt')
        with open(vocabbulary_filepath, 'w') as file:
            for vocab_index, (vocab_token, count) in enumerate(zip(vocab_tokens, token_counts)):
                file.write(str(vocab_index) + '\t' + vocab_token + '\t' + str(count) + '\n')
