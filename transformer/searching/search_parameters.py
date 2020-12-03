from dataclasses import dataclass


@dataclass
class BeamSearchParameters:
    max_length: int
    beam_size: int
    length_penalty: float
    coverage_penalty: float
