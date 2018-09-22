from os.path import dirname, abspath, join, exists
from os import makedirs
from dictionaries import START_TOKEN, END_TOKEN

BASE_DIR = dirname(abspath(__file__))


class TranslationDatasetOnTheFly:

    def __init__(self, phase, limit=None):
        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"

        self.limit = limit

        if phase == 'train':
            source_filepath = join(BASE_DIR, 'data', 'raw', 'src-train.txt')
            target_filepath = join(BASE_DIR, 'data', 'raw', 'tgt-train.txt')
        elif phase == 'val':
            source_filepath = join(BASE_DIR, 'data', 'raw', 'src-val.txt')
            target_filepath = join(BASE_DIR, 'data', 'raw', 'tgt-val.txt')
        else:
            raise NotImplementedError()

        with open(source_filepath) as source_file:
            self.source_data = source_file.readlines()

        with open(target_filepath) as target_filepath:
            self.target_data = target_filepath.readlines()

    def __getitem__(self, item):
        if self.limit is not None and item >= self.limit:
            raise IndexError()

        source = self.source_data[item].strip()
        target = self.target_data[item].strip()
        return source, target

    def __len__(self):
        if self.limit is None:
            return len(self.source_data)
        else:
            return self.limit


class TokenizedTranslationDatasetOnTheFly:

    def __init__(self, phase, limit=None):

        self.raw_dataset = TranslationDatasetOnTheFly(phase, limit)

    def __getitem__(self, item):
        raw_source, raw_target = self.raw_dataset[item]
        tokenized_source = raw_source.split()
        tokenized_target = raw_target.split()
        return tokenized_source, tokenized_target

    def __len__(self):
        return len(self.raw_dataset)


class InputTargetTranslationDatasetOnTheFly:

    def __init__(self, phase, limit=None):
        self.tokenized_dataset = TokenizedTranslationDatasetOnTheFly(phase, limit)

    def __getitem__(self, item):
        tokenized_source, tokenized_target = self.tokenized_dataset[item]
        full_target = [START_TOKEN] + tokenized_target + [END_TOKEN]
        inputs = full_target[:-1]
        targets = full_target[1:]
        return tokenized_source, inputs, targets

    def __len__(self):
        return len(self.tokenized_dataset)


class IndexedInputTargetTranslationDatasetOnTheFly:

    def __init__(self, phase, source_dictionary, target_dictionary, limit=None):

        self.input_target_dataset = InputTargetTranslationDatasetOnTheFly(phase, limit)
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary

    def __getitem__(self, item):
        source, inputs, targets = self.input_target_dataset[item]
        indexed_source = self.source_dictionary.index_sentence(source)
        indexed_inputs = self.target_dictionary.index_sentence(inputs)
        indexed_targets = self.target_dictionary.index_sentence(targets)

        return indexed_source, indexed_inputs, indexed_targets

    def __len__(self):
        return len(self.input_target_dataset)
