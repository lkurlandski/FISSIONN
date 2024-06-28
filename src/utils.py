"""
Utilities.
"""

import math
from pprint import pformat

import torch
from torch import tensor, Tensor, nn


def one_hot_to_binary(x: Tensor) -> Tensor:
    """Converts a one-hot tensor into its binary representation.

    Args:
        x (Tensor): One-hot input tensor with shape (B, K).

    Returns:
        Tensor: Binary representation of tensor with shape (B, log2(K)).
    """
    B, K = x.shape
    num_bits = int(math.log2(K))

    if 2 ** num_bits != K:
        raise ValueError("Input tensor's second dimension must be a power of 2")

    # Get the indices of the non-zero elements
    indices = torch.argmax(x, dim=1)

    # Convert indices to binary representation
    binary = torch.zeros(B, num_bits, dtype=torch.float)
    for i in range(num_bits):
        binary[:, num_bits - 1 - i] = (indices % 2).float()
        indices = indices // 2

    return binary


def count_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


def tensor_memory_size(x: Tensor) -> int:
    return x.element_size() * x.nelement()


def test():

    x = tensor([
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [1., 0., 0., 0.],
    ])
    y = tensor([
        [1., 1.],
        [1., 0.],
        [0., 1.],
        [0., 0.],
    ])
    z = one_hot_to_binary(x)
    assert torch.equal(y, z), f"x={pformat(x.tolist())}\ny={pformat(y.tolist())}\nz={pformat(z.tolist())}"

    x = tensor([
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.],
    ])
    y = tensor([
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 0.],
        [1., 1., 1.],
    ])
    z = one_hot_to_binary(x)
    assert torch.equal(y, z), f"x={pformat(x.tolist())}\ny={pformat(y.tolist())}\nz={pformat(z.tolist())}"


if __name__ == "__main__":
    test()
