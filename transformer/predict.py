from pathlib import Path
from typing import Optional

from transformer.predictor import Translator
import torch

from transformer.searching.search_parameters import BeamSearchParameters


def translate(
    text: str,
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    source_vocab_path: Optional[Path] = None,
    target_vocab_path: Optional[Path] = None,
    max_length: int = 80,
    beam_size: int = 1,
    length_penalty: float = 0.0,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    translator = Translator.from_checkpoint(
        checkpoint_path, source_vocab_path, target_vocab_path, device
    )
    translation = translator.translate(
        text,
        parameters=BeamSearchParameters(
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=length_penalty,
            coverage_penalty=0,
        ),
    )

    print(translation)
    if output_path:
        with output_path.open("w") as output_file:
            output_file.write(translation)
