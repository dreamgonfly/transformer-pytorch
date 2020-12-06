from typing import Optional

import torch
from torch import nn
from torch import Tensor
import numpy as np


# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_size: int, num_positions: int):
#         super(PositionalEncoding, self).__init__()
#         if embedding_size % 2 != 0:
#             raise ValueError(
#                 f"Expected even number for positional embedding size, but got {embedding_size} instead."
#             )
#         encoding_table = torch.zeros(num_positions, embedding_size)
#         positions = torch.arange(0, num_positions, dtype=torch.float64).unsqueeze(1)
#         dimensions = torch.arange(0, embedding_size, 2, dtype=torch.float64)
#         div_term = torch.pow(10000.0, dimensions / embedding_size)
#         encoding_table[:, 0::2] = torch.sin(positions / div_term)
#         encoding_table[:, 1::2] = torch.cos(positions / div_term)
#         encoding_table = encoding_table.unsqueeze(0).float()
#         super(PositionalEncoding, self).__init__()
#         self.register_buffer("encoding_table", encoding_table)
#
#     def forward(self, x: Tensor, position: Optional[int] = None):
#         if position is None:
#             return x + self.encoding_table[:, x.size(1)].clone().detach()
#         else:
#             return x + self.encoding_table[:, position : position + x.size(1)].clone().detach()
#


class PositionalEncoding2(nn.Module):
    def __init__(self, embedding_size: int, num_positions: int):
        super(PositionalEncoding2, self).__init__()
        print("Old positional encoding")

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(num_positions, embedding_size)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: Tensor, position: Optional[int] = None):
        if position is None:
            return x + self.pos_table[:, : x.size(1)].clone().detach()
        else:
            return x + self.pos_table[:, position : position + x.size(1)].clone().detach()


# (PositionalEncoding2(512, 200).position_encodings - PositionalEncoding(512, 200).pos_table)[(PositionalEncoding2(512, 200).position_encodings - PositionalEncoding(512, 200).pos_table).abs() > 0.0000001]
# (PositionalEncoding2(512, 200).position_encodings - PositionalEncoding(512, 200).pos_table)[(PositionalEncoding2(512, 200).position_encodings != PositionalEncoding(512, 200).pos_table)]
# (PositionalEncoding2(512, 200).position_encodings != PositionalEncoding(512, 200).pos_table).sum()
