from typing import Tuple

from torch import Tensor


def accuracy_metric_from_tensor(
    targets: Tensor, predictions: Tensor, pad_token_id: int
) -> Tuple[int, int]:
    """Calculate accuracy assuming the predictions length and the targets length are the same."""
    non_pad_mask = targets.ne(pad_token_id)
    num_correct = targets[non_pad_mask].eq(predictions[non_pad_mask]).sum()
    return num_correct, non_pad_mask.sum()
