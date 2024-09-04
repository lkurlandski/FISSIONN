"""

"""

import torch
from torch import Tensor
import torch.nn.functional as F


def normalized_deviation(input: Tensor, target: Tensor) -> Tensor:
    # https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    return torch.sum(torch.abs(input - target)) / torch.sum(torch.abs(input))


def normalized_root_mean_squared_error(input: Tensor, target: Tensor) -> Tensor:
    # https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    return torch.sqrt(F.mse_loss(input, target)) / torch.mean(torch.abs(input))
