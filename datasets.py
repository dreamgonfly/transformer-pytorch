from os.path import dirname, abspath, join, exists
from os import makedirs

BASE_DIR = dirname(abspath(__file__))


class TranslationDatasetOnTheFly:

    def __init__(self, phase):
        assert phase in ('train', 'val'), "Dataset phase must be either 'train' or 'val'"

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
        source = self.source_data[item].strip()
        target = self.target_data[item].strip()
        return source, target

    def __len__(self):
        return len(self.source_data)


class TokenizedTranslationDatasetOnTheFly:

    def __init__(self, phase):

        self.raw_dataset = TranslationDatasetOnTheFly(phase)

    def __getitem__(self, item):
        raw_source, raw_target = self.raw_dataset[item]
        tokenized_source = raw_source.split()
        tokenized_target = raw_target.split()
        return tokenized_source, tokenized_target

    def __len__(self):
        return len(self.raw_dataset)


class IndexedTranslationDatasetOnTheFly:

    def __init__(self, phase, source_dictionary, target_dictionary):

        self.tokenized_dataset = TokenizedTranslationDatasetOnTheFly(phase)
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary

    def __getitem__(self, item):
        tokenized_source, tokenized_target = self.tokenized_dataset[item]
        indexed_source = self.source_dictionary.index_sentence(tokenized_source)
        indexed_target = self.target_dictionary.index_sentence(tokenized_target)

        return indexed_source, indexed_target

    def __len__(self):
        return len(self.tokenized_dataset)


