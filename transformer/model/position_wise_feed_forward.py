from torch import Tensor
from torch import nn


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # TODO: Add dropout ?
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
             x: (batch_size, seq_len, d_model)
        """
        return self.feed_forward(x)
