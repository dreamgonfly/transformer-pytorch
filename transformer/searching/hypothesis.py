from dataclasses import dataclass
from torch import Tensor


@dataclass
class Hypothesis:
    sequence: Tensor  # (sequence_length,)
    score: float
    length: int
    sample_index: int  # "Sample" means data sample. A batch consists of data samples.
