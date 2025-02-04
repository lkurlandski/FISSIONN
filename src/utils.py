"""
Utilities.
"""

import math
import random
from pprint import pformat
from typing import Optional

import numpy as np
import torch
from torch import tensor, Tensor, nn


def pad_sequence_with_mask(sequences, batch_first=False, padding_value=0.0):
    r"""
    Pads a list of variable length tensors with `padding_value` and returns
    both the padded tensor and a boolean mask indicating the padded positions.

    Args:
        sequences (list of torch.Tensor): List of tensors, each of shape (L, *),
            where L can be different for each tensor.
        batch_first (bool, optional): If True, the output will be in shape
            (batch_size, max_seq_length, *). Otherwise, it will be (max_seq_length, batch_size, *).
            (default: False)
        padding_value (float, optional): Value for padded elements (default: 0.0)

    Returns:
        tuple:
            - padded_tensor (torch.Tensor): Padded tensor of shape
              (batch_size, max_seq_length, *) if batch_first is True or
              (max_seq_length, batch_size, *) otherwise.
            - mask (torch.BoolTensor): Boolean mask of shape (batch_size, max_seq_length)
              if batch_first is True or (max_seq_length, batch_size) otherwise.
              The mask is True at positions where padding was applied.

    Example:
        >>> seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        >>> padded, mask = pad_sequence_with_mask(seqs, batch_first=True)
        >>> print(padded)
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> print(mask)
        tensor([[False, False, False],
                [False, False,  True]])
    """
    if not sequences:
        raise ValueError("Expected a non-empty list of sequences")

    # Determine maximum sequence length
    max_len = max(seq.size(0) for seq in sequences)

    # Determine the shape of the output tensor
    if batch_first:
        out_dims = (len(sequences), max_len) + sequences[0].size()[1:]
    else:
        out_dims = (max_len, len(sequences)) + sequences[0].size()[1:]

    # Create the padded tensor filled with padding_value
    padded_tensor = sequences[0].new_full(out_dims, padding_value)

    # Copy each sequence into the padded tensor
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if batch_first:
            padded_tensor[i, :length, ...] = seq
        else:
            padded_tensor[:length, i, ...] = seq

    # Create the boolean mask.
    # The mask only needs to cover the sequence length and batch dimensions.
    # It will be True for padded positions and False for valid positions.
    lengths = torch.tensor([seq.size(0) for seq in sequences], device=sequences[0].device)
    seq_range = torch.arange(max_len, device=sequences[0].device)

    if batch_first:
        # Shape: (batch_size, max_len)
        mask = seq_range.expand(len(sequences), max_len) >= lengths.unsqueeze(1)
    else:
        # Shape: (max_len, batch_size)
        mask = seq_range.expand(len(sequences), max_len).t() >= lengths.unsqueeze(1)

    return padded_tensor, mask


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


class ShapeError(ValueError):

    def __init__(self, actual_shape: tuple, expected_shape: Optional[tuple] = None):
        self.expected_shape = tuple(expected_shape)
        self.actual_shape = tuple(actual_shape) if actual_shape else None
        super().__init__(f"Recieved: {self.actual_shape}. Expected: {self.expected_shape}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
