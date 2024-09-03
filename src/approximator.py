"""
A network to approximate the noising process of IPDs through SSI chains.


Sources
-------
- "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention"
  (https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)
- "Language Translation with nn.Transformer and torchtext"
  (https://pytorch.org/tutorials/beginner/translation_transformer.html)


Notes
-----
- When decoding with RNNs, only the most recently generated token is fed back into the model because
  the hidden state carries information from preceding tokens. When decoding with Transformers, the
  entire generated sequence is fed back into the model.


Overflow Issue
--------------

Pytorch's TransformerEncoder seems to have some issues with numerical instability.
Training the tiny and small Transformer models results in NaN loss values. Using

```
torch.autograd.set_anomaly_detection(True)
```

reveals the root cause of the issue to be a RuntimeError"

'''
Function 'ScaledDotProductEfficientAttentionBackward0' returned nan values in its 0th output.
'''

This is a documented problem in pytorch, although it was supposedly fixed for 2.3.1
(https://github.com/pytorch/pytorch/issues/119320). The workaround is to the use the
SDPBackend.MATH backend instead of the SDPBackend.EFFICIENT_ATTENTION backend:

```
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.MATH):
    ...
```
"""

from __future__ import annotations
from collections.abc import Iterable
from itertools import chain, combinations
import math
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pformat
import random
from statistics import mean
import sys
from typing import Callable, Generator, Literal, Optional, Protocol, Self
import warnings

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor, BoolTensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data
from src.trainer import TrainerArgs, Trainer, TrainerArgumentParser, TeacherRatioScheduler, EarlyStopper
from src.utils import (
    count_parameters,
    seed_everything,
    ShapeError,
)
# pylint: enable=wrong-import-position


# TODO: improve upon this solution
BACKEND = SDPBackend.EFFICIENT_ATTENTION


PAD = -10000.0
BOS = -10001.0


def pad(shape: tuple[int]) -> Tensor:
    return torch.full(shape, PAD)


def bos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, BOS)


class ApproximatorDataset(Dataset):

    def __init__(self, ipd_pairs: Iterable[tuple[list[float], list[float]]]) -> None:
        self.ipd_pairs = [(torch.tensor(ipd_a), torch.tensor(ipd_b)) for ipd_a, ipd_b in ipd_pairs]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.ipd_pairs[idx]

    def __len__(self) -> int:
        return len(self.ipd_pairs)

    def __repr__(self) -> str:
        s = "ApproximatorDataset("
        s += f"num_pairs={len(self)}, "
        s += ")"
        return s

    def __str__(self) -> str:
        return repr(self)

    @staticmethod
    def build_pairs_from_single_hops(ipd_groups: list[list[list[float]]]) -> Generator[tuple[list[float], list[float]]]:
        """A pair consists of a IPD sequence and the IPD sequence following a single hop."""
        for group in ipd_groups:
            for i in range(len(group) - 1):
                yield group[i], group[i + 1]

    @staticmethod
    def build_pairs_from_hops(ipd_groups: list[list[list[float]]]) -> Generator[tuple[list[float], list[float]]]:
        """A pair consists of a IPD sequence and the IPD sequence following any number of hops."""
        for group in ipd_groups:
            for ipd_1, ipd_2 in combinations(group, 2):
                yield ipd_1, ipd_2

    @staticmethod
    def build_pairs_from_chains(ipd_groups: list[list[list[float]]]) -> Generator[tuple[list[float], list[float]]]:
        """A pair consists of a IPD sequence and the IPD sequence following a chain of hops."""
        for group in ipd_groups:
            yield group[0], group[-1]

    @staticmethod
    def get_synthetic_sample() -> np.ndarray:
        loc = abs(stats.laplace.rvs(loc=0, scale=2.19e-2))
        scale = abs(stats.laplace.rvs(loc=0, scale=1.03e-1))
        length = abs(stats.laplace.rvs(loc=0, scale=575))
        ipds = stats.laplace.rvs(loc=loc, scale=scale, size=int(length))
        ipds = np.absolute(ipds)
        return ipds

    @staticmethod
    def get_synthetic_samples(n_samples: int, n_processes: Optional[int] = None) -> list[np.ndarray]:
        n_processes = len(os.sched_getaffinity()) // 2 if n_processes is None else n_processes
        with mp.Pool(n_processes) as pool:
            return pool.map(ApproximatorDataset.get_synthetic_sample, range(n_samples))

    @staticmethod
    def get_synthetic_hop(ipds: np.ndarray, num_tries: int = 1) -> np.ndarray:
        if num_tries < 1:
            raise ValueError(f"{num_tries=}")

        org_scale = stats.laplace.fit(ipds)[1]
        for _ in range(num_tries):
            delta_scale = stats.laplace.rvs(loc=8.09e-3, scale=2.61e-2)
            new_scale = org_scale + delta_scale
            if new_scale > 0:
                break
        else:
            new_scale = org_scale

        org_length = len(ipds)
        for _ in range(num_tries):
            delta_length = stats.laplace.rvs(loc=-2.00, scale=298)
            new_length = math.ceil(org_length + delta_length)
            if new_length > 0:
                break
        else:
            new_length = org_length

        ipds = stats.laplace.rvs(loc=0.0, scale=new_scale, size=new_length)
        ipds = np.absolute(ipds)
        return ipds

    @staticmethod
    def get_synthetic_hops(ipds: list[np.ndarray], num_tries: int = 1, n_processes: Optional[int] = None) -> list[np.ndarray]:
        n_processes = len(os.sched_getaffinity()) // 2 if n_processes is None else n_processes
        with mp.Pool(n_processes) as pool:
            return pool.starmap(ApproximatorDataset.get_synthetic_hop, zip(ipds, [num_tries] * len(ipds)))


BUILDER_PAIR_MODES: dict[str, Callable[[list[list[list[float]]]], Generator[tuple[list[float], list[float]]]]] = {
    "single_hops": ApproximatorDataset.build_pairs_from_single_hops,
    "hops": ApproximatorDataset.build_pairs_from_hops,
    "chains": ApproximatorDataset.build_pairs_from_chains,
}


class ApproximatorCollateFn:

    def __init__(self, max_length: int = sys.maxsize) -> None:
        self.max_length = max_length

    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        b = bos((1,))
        x = []
        y = []
        for x_, y_ in batch:
            x_ = x_[0 : self.max_length - 1]
            y_ = y_[0 : self.max_length - 1]
            x.append(torch.cat([b, x_]))
            y.append(torch.cat([b, y_]))
        x = pad_sequence(x, batch_first=True, padding_value=PAD)
        y = pad_sequence(y, batch_first=True, padding_value=PAD)
        return x, y


class ApproximatorLossFn(nn.Module):
    """
    TODO:
     - Is there a better way to handle mismatching special tokens?
     - Is there a better way to handle mismatching sequence lengths?
    """

    def __init__(self, allow_different_lengths: bool = False, ratio: float = 1.0) -> None:
        super().__init__()
        self.allow_different_lengths = allow_different_lengths
        self.ratio = ratio
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if y_pred.dim() != 2 or y_true.dim() != 2 or y_pred.size(0) != y_true.size(0):
            raise ShapeError((y_pred.shape, y_true.shape), (("B", "T1"), ("B", "T2")))

        # If the lengths are different, pad/truncate y_pred to the length of y_true
        if y_pred.size(1) != y_true.size(1):
            if not self.allow_different_lengths:
                raise ShapeError((y_pred.shape, y_true.shape), (("B", "T"), ("B", "T")))
            y_pred = y_pred[:, :y_true.size(1)]
            padding = pad((y_pred.size(0), y_true.size(1) - y_pred.size(1))).to(y_pred.device)
            y_pred = torch.cat([y_pred, padding], dim=1)

        # Mask out the special tokens. Since MSE takes the mean of the squared differences,
        # masking out the padding tokens will neither help nor hurt the overall loss (I think).
        mask = (y_true != PAD) & (y_true != BOS) & (y_pred != PAD) & (y_pred != BOS)
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]
        loss_masked = self.loss_fn.forward(y_pred_masked, y_true_masked)

        return loss_masked


class Approximator(Protocol):

    def __init__(self) -> None:
        ...

    def embed(self, inputs: Tensor) -> Tensor:
        ...

    def encode(self, embeddings: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        ...

    def decode(self, encoder_outputs: Tensor, targets: Optional[Tensor], *args, **kwds) -> Tensor:
        ...

    def project(output: Tensor) -> Tensor:
        ...

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None, *args, **kwds) -> Tensor:
        ...

    def translate(self, inputs: Tensor) -> Tensor:
        ...

    @classmethod
    def from_pretrained(cls, file: os.PathLike, **kwds) -> Approximator:
        return torch.load(file, **kwds)


class Attention(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        w = self.Wa.forward(query)
        u = self.Ua.forward(keys)
        v = self.Va.forward(F.tanh(w + u))
        scores = v.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights


class RecurrentApproximator(nn.Module):

    # B - batch size
    # T - sequence length
    # H - hidden size
    # L - num layers
    # I - input size

    PUNY = {"hidden_size": 64, "num_layers": 1}
    TINY = {"hidden_size": 128, "num_layers": 2}
    SMALL = {"hidden_size": 256, "num_layers": 4}
    MEDIUM = {"hidden_size": 384, "num_layers": 6}
    LARGE = {"hidden_size": 512, "num_layers": 8}
    HUGE = {"hidden_size": 768, "num_layers": 12}
    CELL = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(
        self,
        max_length: int,
        hidden_size: int,
        num_layers: int,
        cell: Literal["rnn", "lstm", "gru"] = "rnn",
        **kwds,
    ) -> None:
        if kwds.get("bidirectional", False):
            raise NotImplementedError("Bidirectional RNNs are not supported.")

        super().__init__()
        self.max_length = max_length
        cell = RecurrentApproximator.CELL[cell]
        self.embedding = nn.Linear(1, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.attention = Attention(hidden_size)
        self.encoder: nn.RNN | nn.LSTM | nn.GRU = cell(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            **kwds,
        )
        self.decoder: nn.RNN | nn.LSTM | nn.GRU = cell(
            2 * hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            **kwds,
        )
        self.head = nn.Linear(hidden_size, 1)

    def embed(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 2:
            raise ShapeError((inputs.shape), ("B", "T"))
        x = inputs.unsqueeze(2)
        x = self.embedding.forward(x)
        x = self.dropout.forward(x)
        return x

    def encode(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        if embeddings.dim() != 3:
            raise ShapeError((embeddings.shape), ("B", "T", "H"))
        x = self.encoder.forward(embeddings)
        return x

    def decode(
        self,
        encoder_outputs: Tensor,
        encoder_hidden: Tensor,
        targets: Optional[Tensor] = None,
        teacher_force_ratio: float = 1.0,
        teacher_force_batch_mode: bool = True,
    ) -> Tensor:
        if encoder_outputs.dim() != 3:
            raise ShapeError((encoder_outputs.shape), ("B", "L", "H"))
        if encoder_hidden.dim() != 3:
            raise ShapeError((encoder_hidden.shape), ("L", "B", "H"))
        if targets is not None and targets.dim() != 2:
            raise ShapeError((targets.shape), ("B", "T - 1"))

        if targets is not None:
            if teacher_force_batch_mode:
                use_teacher_forcing = random.random() < teacher_force_ratio
            else:
                use_teacher_forcing = None
        else:
            use_teacher_forcing = False

        B = encoder_outputs.size(0)
        T_src = encoder_outputs.size(1)
        T_tgt = targets.size(1) if targets is not None else None
        T_max = self.max_length
        D = encoder_outputs.device

        decoder_hidden = encoder_hidden                                           # (L, B, H)
        predictions = torch.cat([bos((B, 1)), pad((B, T_max - 1))], dim=1).to(D)  # (B, T_max)
        decoder_input = bos((B, 1)).to(D)                                         # (B, 1)

        finished = torch.zeros((B,), dtype=torch.bool, device=D)
        for i in range(1, self.max_length):

            embeddings = self.embed(decoder_input)                        # (B, 1, H)
            final_hidden_state = decoder_hidden[-1:,:,:].transpose(0, 1)  # (B, 1, H)

            context, _ = self.attention.forward(final_hidden_state, encoder_outputs)                   # (B, 1, H), (B, 1, 2 * H)
            decoder_embeddings = torch.cat((embeddings, context), dim=2)                               # (B, 1, 2 * H)
            decoder_output, decoder_hidden = self.decoder.forward(decoder_embeddings, decoder_hidden)  # (B, 1, H), (L, B, H)

            prediction = self.project(decoder_output)  # (B, T)
            prediction = prediction[:, -1]             # (B,)
            predictions[:,i] = prediction              # (B, T)

            if (finished := finished | (prediction == PAD)).all():
                break

            if not teacher_force_batch_mode:
                use_teacher_forcing = targets is not None and random.random() < teacher_force_ratio

            if use_teacher_forcing:
                if i < T_tgt:
                    decoder_input = targets[:, i].unsqueeze(1)
                else:
                    decoder_input = pad((B, 1)).to(D)
            else:
                decoder_input = prediction.unsqueeze(1)

        return predictions[:, 1:], decoder_hidden

    def project(self, output: Tensor) -> Tensor:
        if output.dim() != 3:
            raise ShapeError((output.shape), ("B", "T", "H"))
        x = self.head.forward(output)
        x = x.squeeze(2)
        return x

    def forward(
        self,
        inputs: Tensor,
        targets: Optional[Tensor] = None,
        teacher_force_ratio: float = 1.0,
        teacher_force_batch_mode: bool = True,
    ) -> Tensor:
        embeddings = self.embed(inputs)                            # (B, T, H)
        encoder_outputs, encoder_hidden = self.encode(embeddings)  # (B, T, H), (L, B, H)
        predictions = self.decode(
            encoder_outputs,
            encoder_hidden,
            targets,
            teacher_force_ratio,
            teacher_force_batch_mode,
        )[0]
        return predictions

    def translate(self, inputs: Tensor) -> Tensor:
        is_training = self.training
        if is_training:
            self.eval()
        with torch.no_grad():
            targets = self.forward(inputs, None, 0.0)
        if is_training:
            self.train()
        return targets

    @classmethod
    def from_pretrained(cls, file: os.PathLike, **kwds) -> RecurrentApproximator:
        return torch.load(file, **kwds)


class PositionalEncoding(nn.Module):

    embedding: Tensor

    def __init__(self, emb_size: int, max_length: int) -> None:
        if emb_size % 2 != 0:
            raise ValueError(f"The embedding size {emb_size=} must be divisible by 2.")

        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        embedding = torch.zeros((max_length, emb_size))
        embedding[:, 0::2] = torch.sin(pos * den)
        embedding[:, 1::2] = torch.cos(pos * den)
        embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)

    def forward(self, t: Tensor):
        if t.dim() != 3:
            raise ShapeError((t.shape), ("B", "L", "H"))
        p = self.embedding[:, :t.size(1), :]
        t = t + p
        return t


class TransformerApproximator(nn.Module):

    # B - batch size
    # T - sequence length
    # H - hidden size
    # L - num layers
    # N - num heads

    PUNY = {"hidden_size": 64, "num_layers": 1, "nhead": 1, "intermediate_size": 256}
    TINY = {"hidden_size": 128, "num_layers": 2, "nhead": 2, "intermediate_size": 512}
    SMALL = {"hidden_size": 256, "num_layers": 4, "nhead": 4, "intermediate_size": 1024}
    MEDIUM = {"hidden_size": 384, "num_layers": 6, "nhead": 6, "intermediate_size": 1536}
    LARGE = {"hidden_size": 512, "num_layers": 8, "nhead": 8, "intermediate_size": 2048}
    HUGE = {"hidden_size": 768, "num_layers": 12, "nhead": 12, "intermediate_size": 3072}

    def __init__(
        self,
        max_length: int,
        hidden_size: int,
        num_layers: int,
        nhead: int,
        intermediate_size: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Linear(1, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_length)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, intermediate_size, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead, intermediate_size, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.head = nn.Linear(hidden_size, 1)

    def embed(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 2:
            raise ShapeError((inputs.shape), ("B", "T"))
        x = inputs.unsqueeze(2)
        x = self.embedding.forward(x)
        x = self.positional_encoding.forward(x)
        x = self.dropout.forward(x)
        return x

    def encode(self, embeddings: Tensor, mask: Optional[Tensor], padding_mask: Optional[Tensor]) -> Tensor:
        if embeddings.dim() != 3:
            raise ShapeError((embeddings.shape), ("B", "T", "H"))
        x = self.encoder.forward(embeddings, mask, padding_mask, is_causal=False)
        return x

    def decode(
        self,
        encoder_outputs: Tensor,
        targets: Optional[Tensor] = None,
        teacher_force_ratio: float = 1.0,
        teacher_force_batch_mode: bool = True,
    ) -> Tensor:

        if encoder_outputs.dim() != 3:
            raise ShapeError((encoder_outputs.shape), ("B", "T", "H"))
        if targets is not None and targets.dim() != 2:
            raise ShapeError((targets.shape), ("B", "T - 1"))

        B = encoder_outputs.size(0)
        T_src = encoder_outputs.size(1)
        T_tgt = targets.size(1) if targets is not None else None
        T_max = self.max_length
        D = encoder_outputs.device

        if targets is not None:
            if teacher_force_batch_mode:
                use_teacher_forcing = random.random() < teacher_force_ratio
            else:
                use_teacher_forcing = None
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=D, dtype=bool)
            tgt_padding_mask = (targets == PAD).to(D)  # (B, T)
            decoder_embeddings = self.embed(targets)   # (B, T, H)
            decoder_outputs = self.decoder.forward(decoder_embeddings, encoder_outputs, tgt_mask, None, tgt_padding_mask, tgt_is_causal=True)
            predictions = self.project(decoder_outputs)
            return predictions

        predictions = torch.cat([bos((B, 1)), pad((B, T_max - 1))], dim=1).to(D)  # (B, T_max)
        decoder_input = bos((B, 1)).to(D)                                         # (B, 1)
        finished = torch.zeros((B,), dtype=torch.bool, device=D)                  # (B,)

        for i in range(1, self.max_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(i, D, torch.bool)  # (i, i)
            tgt_padding_mask = (decoder_input == PAD).to(D)
            decoder_embeddings = self.embed(decoder_input)                               # (B, i, H)
            decoder_output = self.decoder.forward(decoder_embeddings, encoder_outputs, tgt_mask, None, tgt_padding_mask, tgt_is_causal=True)  # (B, i, H)

            prediction = self.project(decoder_output)  # (B, i)
            prediction = prediction[:, -1]             # (B,)
            predictions[:, i] = prediction

            if (finished := finished | (prediction == PAD)).all():
                break

            if not teacher_force_batch_mode:
                use_teacher_forcing = targets is not None and random.random() < teacher_force_ratio

            if use_teacher_forcing:
                decoder_input = targets[:, :i + 1]
            else:
                decoder_input = predictions[:, :i + 1]

        return predictions[:, 1:]

    def project(self, output: Tensor) -> Tensor:
        if output.dim() != 3:
            raise ShapeError((output.shape), ("B", "T", "H"))
        x = self.head.forward(output)
        x = x.squeeze(2)
        return x

    def forward(
        self,
        inputs: Tensor,
        targets: Optional[Tensor] = None,
        teacher_force_ratio: float = 1.0,
        teacher_force_batch_mode: bool = True,
    ) -> Tensor:

        src_mask = torch.zeros(
            (inputs.size(1), inputs.size(1)), dtype=torch.bool, device=inputs.device
        )
        src_padding_mask = (inputs == PAD).to(inputs.device)

        embeddings = self.embed(inputs)
        encoder_outputs = self.encode(embeddings, src_mask, src_padding_mask)
        predictions = self.decode(encoder_outputs, targets, teacher_force_ratio, teacher_force_batch_mode)
        return predictions

    def translate(self, inputs: Tensor) -> Tensor:
        is_training = self.training
        if is_training:
            self.eval()
        with torch.no_grad():
            targets = self.forward(inputs, None, 0.0)
        if is_training:
            self.train()
        return targets

    @classmethod
    def from_pretrained(cls, file: os.PathLike, **kwds) -> TransformerApproximator:
        return torch.load(file, **kwds)


class ApproximatorTrainer(Trainer):

    model: RecurrentApproximator | TransformerApproximator
    tr_dataset: ApproximatorDataset
    vl_dataset: ApproximatorDataset
    collate_fn: ApproximatorCollateFn
    loss_fn: ApproximatorLossFn

    def __call__(self) -> Self:
        with sdpa_kernel(BACKEND):
            return super().__call__()

    def create_scheduler(self) -> Optional[LRScheduler]:
        return ExponentialLR(self.optimizer, gamma=0.75)

    def create_stopper(self) -> None:
        return None

    def create_teacher_ratio_scheduler(self) -> TeacherRatioScheduler:
        return TeacherRatioScheduler(self.args.epochs, self.args.teacher_ratio_start, self.args.teacher_ratio_end)

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)
        y_pred = self.model.forward(x, y[:, :-1], teacher_force_ratio=self.teacher_ratio_scheduler.ratio)
        return (y_pred,)

    def forward_eval(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)
        y_pred_tch = self.model.forward(x, y[:, :-1], teacher_force_ratio=self.teacher_ratio_scheduler.ratio)
        y_pred_gen = self.model.forward(x, y[:, :-1], teacher_force_ratio=0.0)
        return (y_pred_tch, y_pred_gen)

    def compute_loss(self, batch: tuple[Tensor, Tensor], outputs: tuple[Tensor]) -> Tensor:
        y: Tensor = batch[1].to(self.args.device)
        y_pred: Tensor = outputs[0]
        loss: Tensor = self.loss_fn.forward(y_pred, y[:, 1:])
        return loss, {}

    def compute_metrics(self, batch: tuple[Tensor, Tensor], outputs: tuple[Tensor, Tensor]) -> dict[str, float]:
        y: Tensor = batch[1].to(self.args.device)
        y = y[:, 1:]
        y_pred_tch: Tensor = outputs[0]
        y_pred_gen: Tensor = outputs[1]
        loss_tch = self.loss_fn.forward(y_pred_tch, y)
        loss_gen = self.loss_fn.forward(y_pred_gen, y)
        return {"loss_tch": loss_tch.item(), "loss_gen": loss_gen.item()}


class OutputHelper:

    def __init__(
        self,
        root: Path,
        pair_mode: str,
        arch: str,
        arch_config: str,
        max_length: int,
        teacher_ratio_start: float,
        teacher_ratio_end: float,
    ) -> None:
        self.root = root
        self.pair_mode = pair_mode
        self.arch = arch
        self.arch_config = arch_config
        self.max_length = max_length
        self.teacher_ratio_start = teacher_ratio_start
        self.teacher_ratio_end = teacher_ratio_end

    @property
    def path(self) -> Path:
        args = [
            f"pair_mode--{self.pair_mode}",
            f"max_length--{self.max_length}",
            f"arch--{self.arch}",
            f"arch_config--{self.arch_config}",
            f"teacher_ratio_start--{self.teacher_ratio_start}",
            f"teacher_ratio_end--{self.teacher_ratio_end}",
        ]
        return Path(self.root).joinpath(*args) / "results"

    def mkdir(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)


def main() -> None:

    global BACKEND

    parser = TrainerArgumentParser()
    parser.add_argument("--max_length", type=int, default=64, help=".")
    parser.add_argument("--seed", type=int, default=0, help=".")
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "rnn", "lstm", "gru"], help=".")
    parser.add_argument("--arch_config", type=str, default="puny", choices=["puny", "tiny", "small", "medium", "large", "huge"], help=".")
    parser.add_argument("--pair_mode", type=str, default="single_hops", choices=["single_hops", "hops", "chains"], help=".")
    parser.add_argument("--tr_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--vl_num_samples", type=int, default=sys.maxsize, help=".")
    args = parser.parse_args()

    print(f"Command Line Arguments:\n{pformat(args.__dict__)}")
    print("-" * 80)

    oh = OutputHelper(
        args.outdir,
        args.pair_mode,
        args.arch,
        args.arch_config,
        args.max_length,
        args.teacher_ratio_start,
        args.teacher_ratio_end,
    )

    seed_everything(args.seed)

    ipd_groups = []
    for group in load_data():
        ipd_groups.append([ipds.tolist() for ipds in group])
    print(f"Collected {sum(len(group) for group in ipd_groups)} IPDs from {len(ipd_groups)} groups.")
    print("-" * 80)

    tr_ipd_groups, vl_ipd_groups = train_test_split(ipd_groups, test_size=0.10)
    build_pairs_fn = BUILDER_PAIR_MODES[args.pair_mode]
    tr_dataset = ApproximatorDataset(build_pairs_fn(tr_ipd_groups))
    tr_dataset = Subset(tr_dataset, range(min(args.tr_num_samples, len(tr_dataset))))
    vl_dataset = ApproximatorDataset(build_pairs_fn(vl_ipd_groups))
    vl_dataset = Subset(vl_dataset, range(min(args.vl_num_samples, len(vl_dataset))))

    print(f"Training Dataset: {tr_dataset}. Length: {len(tr_dataset)}")
    print(f"Validation Dataset: {vl_dataset}. Length: {len(vl_dataset)}")
    print("-" * 80)

    if args.arch == "transformer":
        config = {"max_length": args.max_length} | getattr(TransformerApproximator, args.arch_config.upper())
        model = TransformerApproximator(**config)
        if args.arch_config in ("tiny", "small"):
            ...
            # BACKEND = SDPBackend.MATH
    else:
        config = {"max_length": args.max_length} | getattr(RecurrentApproximator, args.arch_config.upper())
        model = RecurrentApproximator(**config)

    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    collate_fn = ApproximatorCollateFn(max_length=args.max_length)
    loss_fn = ApproximatorLossFn(allow_different_lengths=True)
    trainer_args = TrainerArgs(
        outdir=oh.path,
        device=args.device,
        epochs=args.epochs,
        tr_batch_size=args.tr_batch_size,
        vl_batch_size=args.vl_batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        disable_tqdm=args.disable_tqdm,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        find_executable_batch_size=args.find_executable_batch_size,
        teacher_ratio_start=args.teacher_ratio_start,
        teacher_ratio_end=args.teacher_ratio_end,
    )
    trainer = ApproximatorTrainer(
        trainer_args,
        model,
        tr_dataset,
        vl_dataset,
        collate_fn,
        loss_fn,
    )

    trainer()


if __name__ == "__main__":
    main()
