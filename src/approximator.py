"""
A network to approximate the noising process of IPDs through SSI chains.


Sources
-------
- "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention"
  (https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py)
- "Language Translation with nn.Transformer and torchtext"
  (https://pytorch.org/tutorials/beginner/translation_transformer.html)


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
from typing import Callable, Generator, Literal, Optional, Self
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
from src.trainer import TrainerArgs, Seq2SeqTrainer, TrainerArgumentParser
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
EOS = -10002.0


def bos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, BOS)


def eos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, EOS)


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
        e = eos((1,))
        x = []
        y = []
        for x_, y_ in batch:
            x_ = x_[0 : self.max_length - 2]
            y_ = y_[0 : self.max_length - 2]
            x.append(torch.cat([b, x_, e]))
            y.append(torch.cat([b, y_, e]))
        x = pad_sequence(x, batch_first=True, padding_value=PAD)
        y = pad_sequence(y, batch_first=True, padding_value=PAD)
        return x, y


class ApproximatorLossFn(nn.Module):
    """
    TODO:
     - Figure out a better way to handle mismatching special tokens. The problem
     is that the loss between the real value and a special token is going to be
     disproportionately high. At the moment, we simply ignore the loss between
     values that correspond to the positions of special tokens in the target,
     but this is not ideal, because it does not consider special tokens that may
     appear in the input. We can't simply ignore positions where special tokens
     appear in the input AND the target because then the model could learn to
     predict excess special tokens since their loss is completely ignored.
     - Figure out a better way to handle mismatching sequence lengths.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # y_pred_specials = (y_true == PAD) | (y_true == BOS) | (y_true == EOS)
        # y_true_specials = (y_true == PAD) | (y_true == BOS) | (y_true == EOS)
        mask = (y_true != PAD) & (y_true != BOS) & (y_true != EOS)
        return self.loss_fn.forward(y_pred[mask], y_true[mask])


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
    # I - input size
    # H - hidden size
    # L - num layers

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
            raise ShapeError((inputs.shape), ("B", "L"))
        x = inputs.unsqueeze(2)
        x = self.embedding.forward(x)
        x = self.dropout.forward(x)
        return x

    def encode(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder.forward(embeddings)
        return x

    def decode(
        self,
        encoder_outputs: Tensor,
        encoder_hidden: Tensor,
        targets: Optional[Tensor] = None,
        ratio: float = 1.0,
    ) -> Tensor:
        decoder_hidden = encoder_hidden                                               # (L, B, H)
        decoder_input = bos((encoder_outputs.size(0), 1)).to(encoder_outputs.device)  # (B, 1)
    
        decoder_outputs, attentions = [], []
        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.decode_step(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
            )
    
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
    
            if targets is not None and random.random() < ratio:
                decoder_input = targets[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def decode_step(
        self,
        inputs: Tensor,
        hidden: Tensor,
        encoder_outputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = self.embed(inputs)          # (B, 1, H)
        query = hidden[-1:,:,:].transpose(0, 1)  # (B, 1, H)

        context, attn_weights = self.attention.forward(query, encoder_outputs)  # (B, 1, H), (B, 1, 2 * H)
        decoder_inputs = torch.cat((embeddings, context), dim=2)                # (B, 1, 2 * H)
        output, hidden = self.decoder.forward(decoder_inputs, hidden)           # (B, 1, H), (L, B, H)

        return output, hidden, attn_weights

    def project(self, output: Tensor) -> Tensor:
        if output.dim() != 3:
            raise ShapeError((x.shape), ("B", "L", "H"))
        x = self.head.forward(output)
        x = x.unsqueeze(2)
        return x

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None, ratio: float = 1.0) -> Tensor:

        embeddings = self.embed(inputs)                            # (B, L, H)
        encoder_outputs, encoder_hidden = self.encode(embeddings)  # (B, L, H), (L, B, H)
        decoder_outputs, decoder_hidden, attentions = self.decode(
            encoder_outputs,
            encoder_hidden,
            targets,
            ratio,
        )
        


class PositionalEncoding(nn.Module):

    embedding: Tensor

    def __init__(self, emb_size: int, max_length: int, dropout: float = 0.1) -> None:
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        embedding = torch.zeros((max_length, emb_size))
        embedding[:, 0::2] = torch.sin(pos * den)
        embedding[:, 1::2] = torch.cos(pos * den)
        embedding = embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("embedding", embedding)

    def forward(self, t: Tensor):
        if t.dim() != 3:
            raise ShapeError((t.shape), ("B", "L", "H"))
        p = self.embedding[:, :t.size(1), :]
        t = t + p
        t = self.dropout(t)
        return t


class TransformerApproximator(nn.Module):

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
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.projection_in = nn.Linear(1, hidden_size)
        self.positional_embedding = PositionalEncoding(hidden_size, max_length)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, nhead, intermediate_size, batch_first=True,
            ),
            num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_size, nhead, intermediate_size, batch_first=True,
            ),
            num_layers,
        )
        self.projection_out = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        mem: Optional[Tensor] = None,
    ) -> Tensor:
        if x is not None:
            if x.dim() != 2:
                raise ShapeError((x.shape), ("B", "L"))
        if y is not None:
            if y.dim() != 2:
                raise ShapeError((y.shape), ("B", "L"))
        if mem is not None:
            if mem.dim() != 3:
                raise ShapeError((mem.shape), ("B", "L", "H"))

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(x, y, x.device)

        if mem is None:
            src = self.embed(x)
            mem = self.encoder.forward(src, src_mask, src_padding_mask)

        tgt = self.embed(y)
        out = self.decoder.forward(tgt, mem, tgt_mask, None, tgt_padding_mask)
        y_pred = self.projection_out.forward(out).squeeze(2)

        return y_pred

    def embed(self, t: Tensor) -> Tensor:
        if t.dim() != 2:
            raise ShapeError((t.shape), ("B", "L"))
        t = t.unsqueeze(2)
        t = self.projection_in.forward(t)
        t = self.positional_embedding.forward(t)
        return t

    @classmethod
    def from_pretrained(cls, file: os.PathLike, **kwds) -> TransformerApproximator:
        return torch.load(file, **kwds)

    @staticmethod
    def create_mask(
        x: Tensor, y: Tensor, device: Optional[torch.cuda.device] = None,
    ) -> tuple[BoolTensor, BoolTensor, BoolTensor, BoolTensor]:
        # Positions with a True value are not allowed to participate in the attention
        l_x = x.size(1)
        l_y = y.size(1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(l_y, device=device, dtype=torch.bool)
        src_mask = torch.zeros((l_x, l_x), dtype=torch.bool, device=device)

        src_padding_mask = (x == PAD).to(device)
        tgt_padding_mask = (y == PAD).to(device)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def beam_search(self, x: Tensor, max_length: int, num_beams: int) -> Tensor:

        raise NotImplementedError("Beam Search algorithm requires a means to score the value of beams.")

        N = num_beams
        B = x.size(0)
        L = x.size(1)
        D = x.device

        # Initial embedding and encoding of the source sequence
        src = self.embed(x)                                          # (B, L, H)
        src_mask = torch.zeros((L, L), dtype=torch.bool, device=D)   # (L, L)
        src_padding_mask = (x == PAD).to(D)                          # (B, L)
        mem = self.encoder.forward(src, src_mask, src_padding_mask)  # (B, L, H)

        # Initialize the beam search
        y = bos((B * N, 1)).to(D)                                          # (B * N, 1)
        scores = torch.zeros((B * N, 1), device=D)                         # (B * N, 1)
        generated_eos = torch.zeros((B * N,), dtype=torch.bool, device=D)  # (B * N)

        # Expand memory to accommodate beam size
        mem = mem.repeat_interleave(N, dim=0)  # (B * N, L, H)

        for l in range(1, max_length - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(l, D, torch.bool)  # (l, l)
            tgt = self.embed(y)                                                          # (B * N, l, H)
            out = self.decoder.forward(tgt, mem, tgt_mask)                               # (B * N, l, H)
            y_pred = self.projection_out(out).squeeze(2)                                 # (B * N, l)

            # Compute scores and update beam
            # log_probs = torch.log_softmax(y_pred[:, -1], dim=-1)         # (B * N,)
            # scores = scores + log_probs                                  # (B * N, B * N)
            pred_next = y_pred[:, -1]                                      # (B * N,)
            abs_error = torch.abs(pred_next.unsqueeze(1) - y_pred[:, -1])  # (B * N, 1)
            scores = scores - abs_error                                    # (B * N, 1)
            scores = scores + abs_error                                    # (B * N, B * N)

            # Flatten scores for easy topk selection
            # scores = scores.view(B, N * log_probs.size(-1))            # (B, B * N * N)
            # topk_scores, topk_indices = torch.topk(scores, N, dim=-1)  # (B, N), (B, N)
            scores = scores.view(B, N * pred_next.size(-1))              # (B, N * N)
            topk_scores, topk_indices = torch.topk(scores, N, dim=-1)    # (B, N)

            # Determine the corresponding beams and vocabulary indices
            # beam_indices = topk_indices // log_probs.size(-1)          # (B, N)
            # vocab_indices = topk_indices % log_probs.size(-1)          # (B, N)
            beam_indices = topk_indices // pred_next.size(-1)            # (B, N)
            real_values = pred_next[topk_indices % pred_next.size(-1)]   # (B, N)

            # Update the beam search variables
            y = y.view(B, N, -1)                                                           # (B, N, l)
            y = torch.gather(y, 1, beam_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))  # (B, N, l)
            y = y.view(B * N, -1)                                                          # (B * N, l)
            # y = torch.cat([y, vocab_indices.view(-1, 1)], dim=-1)                        # (B * N, l + 1)
            y = torch.cat([y, real_values.view(-1, 1)], dim=-1)                            # (B * N, l + 1)

            scores = topk_scores.view(B * N, 1)                                # (B * N, 1)
            # generated_eos = generated_eos | (vocab_indices.view(-1) == EOS)  # (B * N)
            generated_eos = generated_eos | (real_values.view(-1) == EOS)      # (B * N)

            if generated_eos.all():
                break

        # Select the best beam for each sequence
        y = y.view(B, N, -1)                                                               # (B, N, L - 1)
        scores = scores.view(B, N)                                                         # (B, N)
        best_beam_indices = torch.argmax(scores, dim=-1)                                   # (B,)
        y = torch.cat([y[i, best_beam_indices[i]].unsqueeze(0) for i in range(B)], dim=0)  # (B, L - 1)

        # Add EOS to the sequences which have not generated EOS yet. Else add PAD.
        final = eos((B,)).to(D)                            # (B,)
        final[~generated_eos.view(B, N).any(dim=1)] = PAD  # (B,)
        y = torch.cat([y, final.unsqueeze(1)], dim=1)      # (B, L)

        return y

    def greedy_decode(self, x: Tensor, max_length: int, noise_level: Optional[float] = None) -> Tensor:

        _sampler = torch.distributions.Normal(0, noise_level) if noise_level else None

        def add_noise(x: Tensor) -> Tensor:
            if not noise_level:
                return x
            noise = _sampler.sample(x.size()).to(D)
            noisy_x = x + noise
            ignore = ((x == PAD) | (x == BOS) | (x == EOS) | (noisy_x < 0))
            noise[ignore] = 0.0
            return x + noise

        B = x.size(0)
        L = x.size(1)
        D = x.device

        src = self.embed(x)
        src_mask = torch.zeros((L, L), dtype=torch.bool, device=D)
        src_padding_mask = (x == PAD).to(D)
        mem = self.encoder.forward(src, src_mask, src_padding_mask)
        y = bos((B, 1)).to(D)

        # Tracks which sequences have generated EOS.
        generated_eos = torch.zeros((B,), dtype=torch.bool, device=D)

        for l in range(1, max_length - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(l, D, torch.bool)
            tgt = self.embed(y)
            out = self.decoder.forward(tgt, mem, tgt_mask)

            y_pred = self.projection_out.forward(out).squeeze(2)
            y_next = y_pred[:, -1]
            y_next = add_noise(y_next)
            y = torch.cat([y, y_next.unsqueeze(1)], dim=1)

            generated_eos = generated_eos | (y_next == EOS)
            if generated_eos.all():
                break

        # Add EOS to the sequences which have not generated EOS yet. Else add PAD.
        final = eos((B,)).to(D)
        final[~generated_eos] = PAD
        y = torch.cat([y, final.unsqueeze(1)], dim=1)

        return y

    def translate(
        self,
        x: Tensor,
        max_length: int,
        alg: Literal["greedy", "beam"],
        *,
        noise_level: Optional[float] = None,
        num_beams: Optional[int] = None,
    ) -> Tensor:
        if x.dim() != 2:
            raise ShapeError((x.shape), ("B", "L"))

        self.eval()
        with torch.no_grad():
            if alg == "greedy":
                y = self.greedy_decode(x, max_length, noise_level)
            elif alg == "beam":
                y = self.beam_search(x, max_length, num_beams)
            else:
                raise ValueError(f"Invalid Algorithm: {alg}")
        return y


class ApproximatorTrainer(Seq2SeqTrainer):

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

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)
        y_pred = self.model.forward(x, y[:, :-1])
        y_pred = y_pred[:, 1:] if y_pred.size(1) == y.size(1) else y_pred  # trim for recurrent models
        return (y_pred,)

    def compute_loss(self, batch: tuple[Tensor, Tensor], outputs: tuple[Tensor]) -> Tensor:
        y: Tensor = batch[1].to(self.args.device)
        y_pred: Tensor = outputs[0]
        loss: Tensor = self.loss_fn.forward(y_pred, y[:, 1:])
        return loss, {}

    def translate(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor]:
        x = batch[0].to(self.args.device)
        y = self.model.translate(x, 256, "greedy", noise_level=1.03e-2)
        return (y,)

    def compute_metrics_translate(self, batch: tuple[Tensor, Tensor], outputs: tuple[Tensor]) -> dict[str, float]:
        y_true = batch[1].to(self.args.device)
        y_pred = outputs[0].to(self.args.device)

        mask: Tensor = (y_true != PAD) & (y_true != BOS) & (y_true != EOS)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        mae = nn.L1Loss().forward(y_pred, y_true).item()
        mse = nn.MSELoss().forward(y_pred, y_true).item()

        return {"mse": mse, "mae": mae}

    def get_gn_dataloader(self) -> DataLoader:
        gn_dataset = Subset(self.vl_dataset, list(range(len(self.vl_dataset) // 4)))
        return self.get_dataloader(gn_dataset, self.args.vl_batch_size, False)


class OutputHelper:

    def __init__(self, root: Path, pair_mode: str, arch: str, arch_config: str, max_length: int) -> None:
        self.root = root
        self.pair_mode = pair_mode
        self.arch = arch
        self.arch_config = arch_config
        self.max_length = max_length

    @property
    def path(self) -> Path:
        args = [
            f"pair_mode--{self.pair_mode}",
            f"arch--{self.arch}",
            f"arch_config--{self.arch_config}",
            f"max_length--{self.max_length}",
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

    oh = OutputHelper(args.outdir, args.pair_mode, args.arch, args.arch_config, args.max_length)

    seed_everything(args.seed)

    ipd_groups = []
    for group in load_data():
        ipd_groups.append([ipds.tolist() for ipds in group])
    print(f"Collected {sum(len(group) for group in ipd_groups)} IPDs from {len(ipd_groups)} groups.")
    print("-" * 80)

    tr_ipd_groups, vl_ipd_groups = train_test_split(ipd_groups, test_size=0.15)
    build_pairs_fn = BUILDER_PAIR_MODES[args.pair_mode]
    tr_dataset = ApproximatorDataset(build_pairs_fn(tr_ipd_groups))
    tr_dataset = Subset(tr_dataset, range(min(args.tr_num_samples, len(tr_dataset))))
    vl_dataset = ApproximatorDataset(build_pairs_fn(vl_ipd_groups))
    vl_dataset = Subset(vl_dataset, range(min(args.vl_num_samples, len(vl_dataset))))

    print(f"Training Dataset: {tr_dataset}")
    print(f"Validation Dataset: {vl_dataset}")
    print("-" * 80)

    if args.arch == "transformer":
        config = {"max_length": args.max_length} | getattr(TransformerApproximator, args.arch_config.upper())
        model = TransformerApproximator(**config)
        if args.arch_config in ("tiny", "small"):
            BACKEND = SDPBackend.MATH
    else:
        config = {"max_length": args.max_length} | getattr(RecurrentApproximator, args.arch_config.upper())
        model = RecurrentApproximator(**config)

    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    collate_fn = ApproximatorCollateFn(max_length=args.max_length)
    loss_fn = ApproximatorLossFn()
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
