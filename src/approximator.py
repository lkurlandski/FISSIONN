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
from collections import defaultdict
from itertools import batched, chain, combinations
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from pprint import pformat
import random
from statistics import mean
import sys
from typing import Callable, Generator, Literal, Optional, Protocol, Self  # pylint: disable=no-name-in-module
import warnings

from geomloss import SamplesLoss
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, Tensor, BoolTensor, IntTensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler, SequentialLR, ConstantLR, LinearLR
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data
from src.metrics import regression_report
from src.trainer import TrainerArgs, Trainer, TrainerArgumentParser, TeacherRatioScheduler, EarlyStopper
from src.utils import (
    count_parameters,
    seed_everything,
    ShapeError,
)
# pylint: enable=wrong-import-position


# Strange issues with the SDP backend.
BACKEND = SDPBackend.EFFICIENT_ATTENTION


PAD = -10000.0
BOS = -10001.0
EOS = -10002.0


# Should we be returning predictions with a length equal to the predicted length?
# Or should we return predictions with a length of self.max_length and only trim them in self.translate?
# These variables control the behavior of the model with regard to these factors.
# For now, I'll prevent trimming in the training/evaluation loop because the model can still learn
# from timings in the ground truth flow that are past the predicted length.
TRIM_IN_TR = False
TRIM_IN_VL = False
TRIM_MINIMUM_LENGTH = 1


def pad(shape: tuple[int]) -> Tensor:
    return torch.full(shape, PAD)


def bos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, BOS)


def eos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, EOS)


def remove_padding(z: Tensor | list[Tensor]) -> list[Tensor]:
    if isinstance(z, Tensor) and z.dim() == 1:
        return z[z != PAD]
    return [remove_padding(t) for t in z]


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


BUILDER_PAIR_MODES: dict[str, Callable[[list[list[list[float]]]], Generator[tuple[list[float], list[float]]]]] = {
    "single_hops": ApproximatorDataset.build_pairs_from_single_hops,
    "hops": ApproximatorDataset.build_pairs_from_hops,
    "chains": ApproximatorDataset.build_pairs_from_chains,
}


class ApproximatorCollateFn:

    def __init__(self, max_length: int = sys.maxsize) -> None:
        self.max_length = max_length

    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        x = []
        y = []
        for x_, y_ in batch:
            x_ = x_[0 : self.max_length - 2]
            y_ = y_[0 : self.max_length - 2]
            x.append(self.add_special_tokens(x_))
            y.append(self.add_special_tokens(y_))
        x = pad_sequence(x, batch_first=True, padding_value=PAD)
        y = pad_sequence(y, batch_first=True, padding_value=PAD)
        return x, y

    @staticmethod
    def add_special_tokens(z: Tensor) -> Tensor:
        if z.dim() == 1:
            shape = (1,)
            dim = 0
        elif z.dim() == 2:
            shape = (z.size(0), 1)
            dim = 1
        else:
            raise ShapeError(z.shape, ("{B}", "T"))
        return torch.cat([bos(shape), z, eos(shape)], dim=dim)

    @staticmethod
    def verify_has_bos(z: Tensor):
        if not torch.all(z[:, 0] == BOS):
            raise ValueError(f"Sequences do not begin with {BOS=}")

    @staticmethod
    def verify_has_eos(z: Tensor):
        eos_positions = (z == EOS).nonzero(as_tuple=False)
        if eos_positions.shape[0] != z.size(0):
            raise ValueError(f"Sequences do not end with {EOS=}")


class ApproximatorLossFn(nn.Module):

    def __init__(
        self,
        timing_weight: float = 1.0,
        length_weight: float = 1.0,
        distrib_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.timing_weight = timing_weight
        self.length_weight = length_weight
        self.distrib_weight = distrib_weight

    def forward(self, y_pred: Tensor, y_true: Tensor, length_pred: Tensor, length_true: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if y_pred.dim() != 2 or y_true.dim() != 2 or y_pred.size(0) != y_true.size(0):
            raise ShapeError((y_pred.shape, y_true.shape), (("B", "T1"), ("B", "T2")))

        NUM = len(y_pred)

        # List of inhomogeneous predictions and ground truths without PAD tokens.
        y_pred = remove_padding(y_pred)
        y_true = remove_padding(y_true)

        # Length statistics about the predictions and ground truths.
        lengths = torch.tensor([[len(y_true[i]), len(y_pred[i])] for i in range(NUM)], dtype=torch.float32)
        minimum = torch.minimum(lengths[:,0], lengths[:,1]).to(torch.int64).tolist()

        # List of inhomogeneous predictions and ground truths without BOS and EOS tokens,
        # trimmed such that corresponding sequences have the same length.
        y_pred_trim = [y_pred[i][1:l-1] for i, l in enumerate(minimum)]
        y_true_trim = [y_true[i][1:l-1] for i, l in enumerate(minimum)]

        # Predictions and ground truths trimmed such that the corresponding sequences have the same length,
        # then padded into a homogeneous shape.
        y_pred_homo = pad_sequence(y_true_trim, batch_first=True, padding_value=PAD)
        y_true_homo = pad_sequence(y_pred_trim, batch_first=True, padding_value=PAD)

        # Compute the length loss, not considering the length added by BOS and EOS tokens.
        length_loss = F.mse_loss(length_pred, length_true)

        # Compute the timing loss over the shorter of the two sequences, excluding BOS and EOS tokens.
        timing_loss = F.mse_loss(torch.cat(y_pred_trim), torch.cat(y_true_trim))

        # Compute the distribution loss. We do this in batches to prevent CUDA OOM errors, which is
        # sufficiently fast with the default "tensorized" backend (no need for specialized engines).
        mask = torch.zeros_like(y_pred_homo)
        for i, length in enumerate(minimum):
            mask[i, :length] = 1.0 / length

        distrib_losses = []
        batch_size = 32
        for i in range(0, len(mask), batch_size):
            m = mask[i:i+batch_size]
            x = y_pred_homo[i:i+batch_size].unsqueeze(2)
            y = y_true_homo[i:i+batch_size].unsqueeze(2)
            l = SamplesLoss(loss="sinkhorn", backend="tensorized")(m, x, m, y)
            distrib_losses.append(l)
        distrib_loss = torch.cat(distrib_losses).mean()

        # Combine the timing and length losses.
        weighted_loss = self.timing_weight * timing_loss \
                      + self.length_weight * length_loss \
                      + self.distrib_weight * distrib_loss

        return weighted_loss, length_loss, timing_loss

    @staticmethod
    def prepare_inputs_and_targets(
        y_pred: Tensor,
        y_true: Tensor,
        allow_different_lengths: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Note that this will flatten 2-dimension inputs.
        """

        # If the lengths are different, pad/truncate y_pred to the length of y_true.
        if y_pred.size(1) != y_true.size(1):
            if not allow_different_lengths:
                raise ShapeError((y_pred.shape, y_true.shape), (("B", "T"), ("B", "T")))
            y_pred = y_pred[:, :y_true.size(1)]
            padding = pad((y_pred.size(0), y_true.size(1) - y_pred.size(1))).to(y_pred.device)
            y_pred = torch.cat([y_pred, padding], dim=1)

        # Mask out the special tokens. Since MSE takes the mean of the squared differences,
        # masking out the padding tokens will neither help nor hurt the metrics (I think.)
        mask = (y_true != PAD) & (y_true != BOS) & (y_true != EOS) & (y_pred != PAD) & (y_pred != BOS) & (y_pred != EOS)
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]

        return y_pred_masked, y_true_masked


class Approximator(Protocol):

    def __init__(self) -> None:
        ...

    def embed_src(self, inputs: Tensor) -> Tensor:
        ...

    def embed_tgt(self, targets: Tensor) -> Tensor:
        ...

    def encode(self, embeddings: Tensor) -> tuple[Tensor, Tensor] | Tensor:
        ...

    def predict_length(self, encoder_outputs: Tensor) -> Tensor:
        ...

    def decode(self, encoder_outputs: Tensor, targets: Optional[Tensor], *args, **kwds) -> Tensor:
        ...

    def project(self, output: Tensor) -> Tensor:
        ...

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None, *args, **kwds) -> tuple[Tensor, Tensor]:  # pylint: disable=keyword-arg-before-vararg
        ...

    def translate(self, inputs: Tensor) -> Tensor:
        ...

    @classmethod
    def from_pretrained(cls, file: os.PathLike, **kwds) -> Approximator:
        ...


def from_pretrained(file: str, **kwds) -> Approximator:
    if "transformer" in file:
        return TransformerApproximator.from_pretrained(file, **kwds)
    if any(s in file for s in ("rnn", "lstm", "gru")):
        return RecurrentApproximator.from_pretrained(file, **kwds)
    return torch.load(file, **kwds)


class LengthPredictionHead(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x).squeeze(1)

    @staticmethod
    def trim_predictions_(predictions: Tensor, lengths: Tensor, min_length: int = 0) -> Tensor:
        # Set the EOS token right after the predicted output lengths and PAD after that.
        # Also accounts for the possibility that EOS occurs before the predicted length.
        # TODO: vectorize using torch instead of for-loop.

        B = predictions.shape[0]

        for i in range(B):
            p = predictions[i]
            l = int(lengths[i].item())

            if len(eos_positions := torch.nonzero(p == EOS, as_tuple=False)):
                if not torch.all(p[eos_positions[0] + 1:] == PAD):
                    raise RuntimeError(f"Non-padding tokens found after EOS token: {p[eos_positions[0] + 1:].tolist()}")

            if l <= len(p):
                predictions[i, max(l, min_length) + 1 ] = EOS
                predictions[i, max(l, min_length) + 2:] = PAD

        return predictions


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
        self.cell = cell
        cell = RecurrentApproximator.CELL[self.cell]
        self.embedding_src = nn.Linear(1, hidden_size)
        self.embedding_tgt = nn.Linear(1, hidden_size)
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
        self.length_prediction_head = LengthPredictionHead(hidden_size, hidden_size)

    def embed_src(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 2:
            raise ShapeError((inputs.shape), ("B", "T"))
        x = inputs.unsqueeze(2)
        x = self.embedding_src.forward(x)
        x = self.dropout.forward(x)
        return x

    def embed_tgt(self, targets: Tensor) -> Tensor:
        if targets.dim() != 2:
            raise ShapeError((targets.shape), ("B", "T"))
        x = targets.unsqueeze(2)
        x = self.embedding_tgt.forward(x)
        x = self.dropout.forward(x)
        return x

    def encode(self, embeddings: Tensor) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        if embeddings.dim() != 3:
            raise ShapeError((embeddings.shape), ("B", "T", "H"))
        encoder_states = None
        if isinstance(self.encoder, nn.LSTM):
            encoder_outputs, (encoder_hidden, encoder_states) = self.encoder.forward(embeddings)
        else:
            encoder_outputs, encoder_hidden = self.encoder.forward(embeddings)
        return encoder_outputs, encoder_hidden, encoder_states

    def predict_length(self, encoder_outputs: Tensor) -> Tensor:
        if encoder_outputs.dim() != 3:
            raise ShapeError((encoder_outputs.shape), ("B", "L", "H"))
        x = self.length_prediction_head.forward(encoder_outputs.mean(1))
        return x

    def decode(
        self,
        encoder_outputs: Tensor,
        encoder_hidden: Tensor,
        encoder_states: Optional[Tensor],
        targets: Optional[Tensor] = None,
        teacher_force_ratio: float = 1.0,
        teacher_force_batch_mode: bool = True,
        max_length: Optional[int] = None,
    ) -> Tensor:
        if encoder_outputs.dim() != 3:
            raise ShapeError((encoder_outputs.shape), ("B", "L", "H"))
        if encoder_hidden.dim() != 3:
            raise ShapeError((encoder_hidden.shape), ("L", "B", "H"))
        if encoder_states is not None and encoder_states.dim() != 3:
            raise ShapeError((encoder_states.shape), ("L", "B", "H"))
        if targets is not None and targets.dim() != 2:
            raise ShapeError((targets.shape), ("B", "T - 1"))

        if targets is not None:
            if teacher_force_batch_mode:
                use_teacher_forcing = random.random() < teacher_force_ratio
            else:
                use_teacher_forcing = None
        else:
            use_teacher_forcing = False

        max_length = self.max_length if max_length is None else max_length

        B = encoder_outputs.size(0)
        T_src = encoder_outputs.size(1)  # pylint: disable=unused-variable
        T_tgt = targets.size(1) if targets is not None else None
        T_max = max_length
        D = encoder_outputs.device

        decoder_hidden = encoder_hidden                                           # (L, B, H)
        decoder_states = encoder_states                                           # (L, B, H)
        predictions = torch.cat([bos((B, 1)), pad((B, T_max - 1))], dim=1).to(D)  # (B, T_max)
        decoder_input = bos((B, 1)).to(D)                                         # (B, 1)

        finished = torch.zeros((B,), dtype=torch.bool, device=D)
        for i in range(1, max_length):

            embeddings = self.embed_tgt(decoder_input)                    # (B, 1, H)
            final_hidden_state = decoder_hidden[-1:,:,:].transpose(0, 1)  # (B, 1, H)

            context, _ = self.attention.forward(final_hidden_state, encoder_outputs)                   # (B, 1, H), (B, 1, 2 * H)
            decoder_embeddings = torch.cat((embeddings, context), dim=2)                               # (B, 1, 2 * H)
            if isinstance(self.decoder, nn.LSTM):
                decoder_output, (decoder_hidden, decoder_states) = self.decoder.forward(decoder_embeddings, (decoder_hidden, decoder_states))  # (B, 1, H), (L, B, H), (L, B, H)
            else:
                decoder_output, decoder_hidden = self.decoder.forward(decoder_embeddings, decoder_hidden)  # (B, 1, H), (L, B, H)

            prediction = self.project(decoder_output)  # (B, T)
            prediction = prediction[:, -1]             # (B,)
            predictions[:,i] = prediction              # (B, T)

            if (finished := finished | (prediction == EOS)).all():
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

        predictions[~finished, -1] = EOS
        return predictions, decoder_hidden

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
    ) -> tuple[Tensor, Tensor]:

        embeddings = self.embed_src(inputs)                                        # (B, T, H)
        encoder_outputs, encoder_hidden, encoder_states = self.encode(embeddings)  # (B, T, H), (L, B, H), (L, B, H)
        predictions = self.decode(
            encoder_outputs,
            encoder_hidden,
            encoder_states,
            targets,
            teacher_force_ratio,
            teacher_force_batch_mode,
        )[0]
        lengths = self.predict_length(encoder_outputs)

        return predictions, lengths

    def translate(self, inputs: Tensor, length_scaler: StandardScaler) -> Tensor:
        is_training = self.training
        if is_training:
            self.eval()

        with torch.no_grad():
            targets, lengths = self.forward(inputs, None, 0.0, False)
        lengths = lengths.unsqueeze(1).numpy(force=True)
        lengths = length_scaler.inverse_transform(lengths)
        lengths = torch.tensor(lengths, device=targets.device).squeeze(1)
        targets = LengthPredictionHead.trim_predictions_(targets, lengths, TRIM_MINIMUM_LENGTH)

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
        self.embedding_src = nn.Linear(1, hidden_size)
        self.embedding_tgt = nn.Linear(1, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_length)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead, intermediate_size, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead, intermediate_size, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.head = nn.Linear(hidden_size, 1)
        self.length_prediction_head = LengthPredictionHead(hidden_size, hidden_size)

    def embed_src(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 2:
            raise ShapeError((inputs.shape), ("B", "T"))
        ApproximatorCollateFn.verify_has_bos(inputs)
        ApproximatorCollateFn.verify_has_eos(inputs)
        x = inputs.unsqueeze(2)
        x = self.embedding_src.forward(x)
        x = self.positional_encoding.forward(x)
        x = self.dropout.forward(x)
        return x

    def embed_tgt(self, targets: Tensor) -> Tensor:
        if targets.dim() != 2:
            raise ShapeError((targets.shape), ("B", "T"))
        x = targets.unsqueeze(2)
        x = self.embedding_tgt.forward(x)
        x = self.positional_encoding.forward(x)
        x = self.dropout.forward(x)
        return x

    def encode(self, embeddings: Tensor, mask: Optional[Tensor], padding_mask: Optional[Tensor]) -> Tensor:
        if embeddings.dim() != 3:
            raise ShapeError((embeddings.shape), ("B", "T", "H"))
        x = self.encoder.forward(embeddings, mask, padding_mask, is_causal=False)
        return x

    def predict_length(self, encoder_outputs: Tensor) -> Tensor:
        if encoder_outputs.dim() != 3:
            raise ShapeError((encoder_outputs.shape), ("B", "L", "H"))
        x = self.length_prediction_head.forward(encoder_outputs.mean(1))
        return x

    def decode(
        self,
        encoder_outputs: Tensor,
        targets: Optional[Tensor] = None,
        teacher_force_ratio: float = 1.0,
        teacher_force_batch_mode: bool = True,
        max_length: Optional[int] = None,
    ) -> Tensor:

        if encoder_outputs.dim() != 3:
            raise ShapeError((encoder_outputs.shape), ("B", "T", "H"))
        if targets is not None and targets.dim() != 2:
            raise ShapeError((targets.shape), ("B", "T - 1"))

        max_length = self.max_length if max_length is None else max_length

        B = encoder_outputs.size(0)
        T_src = encoder_outputs.size(1)  # pylint: disable=unused-variable
        T_tgt = targets.size(1) if targets is not None else None
        T_max = max_length
        D = encoder_outputs.device

        if targets is not None:
            if teacher_force_batch_mode:
                use_teacher_forcing = random.random() < teacher_force_ratio
            else:
                use_teacher_forcing = None
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            targets = targets[:, :-1]
            T_tgt = targets.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=D, dtype=bool)
            tgt_padding_mask = (targets == PAD).to(D)     # (B, T)
            decoder_embeddings = self.embed_tgt(targets)  # (B, T, H)
            decoder_outputs = self.decoder.forward(decoder_embeddings, encoder_outputs, tgt_mask, None, tgt_padding_mask, tgt_is_causal=True)
            predictions = self.project(decoder_outputs)
            finished = (predictions == EOS).any(dim=1)
            predictions[~finished, -1] = EOS
            predictions = torch.cat([bos((B, 1)).to(predictions.device), predictions], dim=1)
            return predictions

        predictions = torch.cat([bos((B, 1)), pad((B, T_max - 1))], dim=1).to(D)  # (B, T_max)
        decoder_input = bos((B, 1)).to(D)                                         # (B, 1)
        finished = torch.zeros((B,), dtype=torch.bool, device=D)                  # (B,)

        for i in range(1, max_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(i, D, torch.bool)  # (i, i)
            tgt_padding_mask = (decoder_input == PAD).to(D)
            decoder_embeddings = self.embed_tgt(decoder_input)                           # (B, i, H)
            decoder_output = self.decoder.forward(decoder_embeddings, encoder_outputs, tgt_mask, None, tgt_padding_mask, tgt_is_causal=True)  # (B, i, H)

            prediction = self.project(decoder_output)  # (B, i)
            prediction = prediction[:, -1]             # (B,)
            predictions[:, i] = prediction

            if (finished := finished | (prediction == EOS)).all():
                break

            if not teacher_force_batch_mode:
                use_teacher_forcing = targets is not None and random.random() < teacher_force_ratio

            if use_teacher_forcing:
                decoder_input = targets[:, :i + 1]
            else:
                decoder_input = predictions[:, :i + 1]

        predictions[~finished, -1] = EOS
        return predictions

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
    ) -> tuple[Tensor, Tensor]:

        src_mask = torch.zeros(
            (inputs.size(1), inputs.size(1)), dtype=torch.bool, device=inputs.device
        )
        src_padding_mask = (inputs == PAD).to(inputs.device)
        embeddings = self.embed_src(inputs)
        encoder_outputs = self.encode(embeddings, src_mask, src_padding_mask)
        predictions = self.decode(
            encoder_outputs,
            targets,
            teacher_force_ratio,
            teacher_force_batch_mode,
        )
        lengths = self.predict_length(encoder_outputs)

        return predictions, lengths

    def translate(self, inputs: Tensor, length_scaler: StandardScaler) -> Tensor:
        is_training = self.training
        if is_training:
            self.eval()

        with torch.no_grad():
            targets, lengths = self.forward(inputs, None, 0.0, False)
        lengths = lengths.unsqueeze(1).numpy(force=True)
        lengths = length_scaler.inverse_transform(lengths)
        lengths = torch.tensor(lengths, device=targets.device).squeeze(1)
        targets = LengthPredictionHead.trim_predictions_(targets, lengths, TRIM_MINIMUM_LENGTH)

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

    def __init__(self, *args, length_scaler: StandardScaler, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.length_scaler = length_scaler

    def __call__(self) -> Self:
        with sdpa_kernel(BACKEND):
            return super().__call__()

    def create_scheduler(self) -> Optional[LRScheduler]:
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=10)
        decay_scheduler = ExponentialLR(self.optimizer, gamma=0.85)
        scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[10])
        return scheduler

    def create_stopper(self) -> None:
        return None

    def create_teacher_ratio_scheduler(self) -> TeacherRatioScheduler:
        return TeacherRatioScheduler(self.args.epochs, self.args.teacher_ratio_start, self.args.teacher_ratio_end)

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Optional[Tensor], Tensor]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)
        y_pred, length_pred = self.model.forward(x, y, teacher_force_ratio=self.teacher_ratio_scheduler.ratio)
        return (y_pred, None, length_pred)

    def forward_eval(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)
        y_pred_tch, _ = self.model.forward(x, y, teacher_force_ratio=self.teacher_ratio_scheduler.ratio)
        y_pred_gen, length_pred = self.model.forward(x, y, teacher_force_ratio=0.0)
        return (y_pred_tch, y_pred_gen, length_pred)

    def compute_loss(self, batch: tuple[Tensor, Tensor], outputs: tuple[Tensor, Optional[Tensor], Tensor]) -> tuple[Tensor, dict[str, float]]:
        y: Tensor = batch[1].to(self.args.device)
        y_pred: Tensor = outputs[0]
        length = torch.tensor(
            self.length_scaler.transform(np.expand_dims(np.array([len(y_) for y_ in y]), 1)),
            dtype=torch.float32, device=y.device,
        ).squeeze(1)
        length_pred: Tensor = outputs[2]
        loss, length_loss, timing_loss = self.loss_fn.forward(y_pred, y, length_pred, length)
        return loss, {"length_loss": length_loss.item(), "timing_loss": timing_loss.item()}

    def compute_metrics(  # pylint: disable=arguments-differ
        self,
        y_true: list[Tensor],
        y_pred_tch: list[Tensor],
        y_pred_gen: list[Tensor],
        length_pred: list[Tensor],
    ) -> dict[str, float]:
        if not len(y_true) == len(y_pred_tch) == len(y_pred_gen):
            raise ValueError(f"Lengths of {y_true=}, {y_pred_tch=}, and {y_pred_gen=} do not match.")

        y_true     = remove_padding(y_true)
        y_pred_tch = remove_padding(y_pred_tch)
        y_pred_gen = remove_padding(y_pred_gen)

        NUM = len(y_true)                            # Number of sequences
        MET = ("r2", "mae", "mse", "nrmse", "ndev")  # Metrics

        metrics = {}
        for (name, y_pred) in zip(("tch", "gen"), (y_pred_tch, y_pred_gen)):

            # Compute the length metrics, not considering the length added by BOS and EOS tokens.
            y_tr = torch.tensor([len(y) for y in y_true], dtype=torch.float32).numpy(force=True) - 2
            y_pr = self.length_scaler.inverse_transform(torch.tensor(length_pred).unsqueeze(1).numpy(force=True), 1).squeeze(1)
            m = regression_report(y_tr, y_pr)
            metrics.update({f"{name}_length_{k}": v for k, v in m.items() if k in MET})

            # Compute the timing metrics over the shorter of the two sequences, excluding BOS and EOS tokens.
            lengths = torch.tensor([[len(y_true[i]), len(y_pred[i])] for i in range(NUM)], dtype=torch.float32)
            minimum = torch.minimum(lengths[:,0], lengths[:,1]).to(torch.int64).tolist()
            y_tr = torch.cat([y_true[i][1:l-1] for i, l in enumerate(minimum)])
            y_pr = torch.cat([y_pred[i][1:l-1] for i, l in enumerate(minimum)])
            for tok in (BOS, EOS, PAD):
                for ten in (y_tr, y_pr):
                    if (ten == tok).any():
                        raise RuntimeError(f"{tok=} found in {ten.tolist()=}")
            m = regression_report(y_tr.numpy(force=True), y_pr.numpy(force=True))
            metrics.update({f"{name}_timing_{k}": v for k, v in m.items() if k in MET})

        return metrics

    def get_compute_metrics_inputs(self, batch: tuple, outputs: tuple) -> dict[str, list[Tensor]]:
        return {
            "y_true": [t for t in batch[1]],         # pylint: disable=unnecessary-comprehension
            "y_pred_tch": [t for t in outputs[0]],   # pylint: disable=unnecessary-comprehension
            "y_pred_gen": [t for t in outputs[1]],   # pylint: disable=unnecessary-comprehension
            "length_pred": [t for t in outputs[2]],  # pylint: disable=unnecessary-comprehension
        }


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
        timing_weight: float,
        length_weight: float,
        distrib_weight: float,
    ) -> None:
        self.root = root
        self.pair_mode = pair_mode
        self.arch = arch
        self.arch_config = arch_config
        self.max_length = max_length
        self.teacher_ratio_start = teacher_ratio_start
        self.teacher_ratio_end = teacher_ratio_end
        self.timing_weight = timing_weight
        self.length_weight = length_weight
        self.distrib_weight = distrib_weight

    @property
    def path(self) -> Path:
        args = [
            f"pair_mode--{self.pair_mode}",
            f"max_length--{self.max_length}",
            f"arch--{self.arch}",
            f"arch_config--{self.arch_config}",
            f"teacher_ratio_start--{self.teacher_ratio_start}",
            f"teacher_ratio_end--{self.teacher_ratio_end}",
            f"timing_weight--{self.timing_weight}",
            f"length_weight--{self.length_weight}",
            f"distrib_weight--{self.distrib_weight}",
        ]
        return Path(self.root).joinpath(*args) / "results"

    def mkdir(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)


def main() -> None:

    # global BACKEND

    parser = TrainerArgumentParser()
    parser.add_argument("--max_length", type=int, default=64, help=".")
    parser.add_argument("--seed", type=int, default=0, help=".")
    parser.add_argument("--arch", type=str, default="transformer", choices=["transformer", "rnn", "lstm", "gru"], help=".")
    parser.add_argument("--arch_config", type=str, default="puny", choices=["puny", "tiny", "small", "medium", "large", "huge"], help=".")
    parser.add_argument("--pair_mode", type=str, default="single_hops", choices=["single_hops", "hops", "chains"], help=".")
    parser.add_argument("--tr_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--vl_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--timing_weight", type=float, default=1.0, help=".")
    parser.add_argument("--length_weight", type=float, default=1.0, help=".")
    parser.add_argument("--distrib_weight", type=float, default=1.0, help=".")
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
        args.timing_weight,
        args.length_weight,
        args.distrib_weight,
    )
    print(f"{oh.path=}")

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

    lengths = [(len(x), len(y)) for x, y in tr_dataset.dataset.ipd_pairs]
    lengths = np.expand_dims(np.array(lengths).flatten(), 1)
    length_scaler = StandardScaler().fit(lengths, 1)
    with open("./cache/length_scaler.pkl", "wb") as fp:
        pickle.dump(length_scaler, fp)
    print(f"Length Scaler: {length_scaler}")

    if args.arch == "transformer":
        config = {"max_length": args.max_length} | getattr(TransformerApproximator, args.arch_config.upper())
        model = TransformerApproximator(**config)
        if args.arch_config in ("tiny", "small"):
            ...
            # BACKEND = SDPBackend.MATH
    else:
        config = {"max_length": args.max_length, "cell": args.arch} | getattr(RecurrentApproximator, args.arch_config.upper())
        model = RecurrentApproximator(**config)

    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    collate_fn = ApproximatorCollateFn(max_length=args.max_length)
    loss_fn = ApproximatorLossFn(
        timing_weight=args.timing_weight,
        length_weight=args.length_weight,
        distrib_weight=args.distrib_weight,
    )
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
        length_scaler=length_scaler,
    )

    trainer()


if __name__ == "__main__":
    main()
