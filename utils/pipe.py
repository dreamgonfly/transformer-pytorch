import torch

PAD_INDEX = 0


def input_target_collate_fn(batch):
    """merges a list of samples to form a mini-batch."""

    # indexed_sources = [sources for sources, inputs, targets in batch]
    # indexed_inputs = [inputs for sources, inputs, targets in batch]
    # indexed_targets = [targets for sources, inputs, targets in batch]

    sources_lengths = [len(sources) for sources, inputs, targets in batch]
    inputs_lengths = [len(inputs) for sources, inputs, targets in batch]
    targets_lengths = [len(targets) for sources, inputs, targets in batch]

    sources_max_length = max(sources_lengths)
    inputs_max_length = max(inputs_lengths)
    targets_max_length = max(targets_lengths)

    sources_padded = [sources + [PAD_INDEX] * (sources_max_length - len(sources)) for sources, inputs, targets in batch]
    inputs_padded = [inputs + [PAD_INDEX] * (inputs_max_length - len(inputs)) for sources, inputs, targets in batch]
    targets_padded = [targets + [PAD_INDEX] * (targets_max_length - len(targets)) for sources, inputs, targets in batch]

    sources_tensor = torch.tensor(sources_padded)
    inputs_tensor = torch.tensor(inputs_padded)
    targets_tensor = torch.tensor(targets_padded)

    # lengths = {
    #     'sources_lengths': torch.tensor(sources_lengths),
    #     'inputs_lengths': torch.tensor(inputs_lengths),
    #     'targets_lengths': torch.tensor(targets_lengths)
    # }

    return sources_tensor, inputs_tensor, targets_tensor


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