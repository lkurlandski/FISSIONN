"""
...
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
from pprint import pformat
import pickle
import sys
import time
from typing import Literal, Optional, Self  # pylint: disable=no-name-in-module

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, Tensor, BoolTensor, FloatTensor
from torch.distributions import Laplace, Uniform
from torch.nn import functional as F, CrossEntropyLoss, L1Loss
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler, SequentialLR, LinearLR, ConstantLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.approximator import (
    Approximator,
    RecurrentApproximator,
    TransformerApproximator,
    from_pretrained,
    PAD, BOS, EOS,
)
from src.caida import (
    stream_caida_data,
    stream_caida_data_demo,
    get_caida_ipds,
    CaidaSample,  # pylint: disable=unused-import  # Needed because of pickle
)
from src.finn import FinnLossFn, FinnEncoder, FinnDecoder
from src.trainer import TrainerArgs, Trainer, TrainerArgumentParser
from src.utils import (
    pad_sequence_with_mask,
    count_parameters,
    one_hot_to_binary,
    tensor_memory_size,
    seed_everything,
    ShapeError,
)
# pylint: enable=wrong-import-position


# fingerprint, ipd, delay, ipd_pad_mask
Sample = tuple[Tensor, FloatTensor, FloatTensor, BoolTensor]
Samples = tuple[Tensor, FloatTensor, FloatTensor, BoolTensor]


class FissionnDataset(Dataset, ABC):

    def __init__(
        self,
        ipds: list[np.ndarray],
        fingerprint_length: int,
        amplitude: int,
        flow_length: Optional[int] = None,
        pad_value: int = 0,
    ) -> None:
        self.ipds = [torch.from_numpy(ipd).to(torch.float32) for ipd in ipds]
        self.fingerprint_length = fingerprint_length
        self.aplitude = amplitude
        self.flow_length = flow_length
        self.pad_value = pad_value
        self.delay_sampler = Laplace(0, amplitude)

        self.__post_init__()

    def __getitem__(self, idx: int) -> Sample:
        ipd = self.get_ipd(idx)
        ipd_pad_mask = self.get_ipd_pad_mask(idx)
        fingerprint = self.get_fingerprint(idx)
        delay = self.get_delay(idx)
        return fingerprint, ipd, delay, ipd_pad_mask

    def __len__(self) -> int:
        return len(self.ipds)

    def __post_init__(self) -> None:
        pass

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  ipds={len(self.ipds)},{self.ipds[0].dtype}\n"
        s += f"  fingerprint_length={self.fingerprint_length},\n"
        s += f"  amplitude={self.aplitude},\n"
        s += f"  flow_length={self.flow_length},\n"
        s += f"  pad_value={self.pad_value},\n"
        s += f"  delay_sampler={self.delay_sampler},\n"
        s += f"  memory_size={round(self.memory_size / 1e9, 2)}G,\n"
        s += ")"
        return s

    def __str__(self) -> str:
        return repr(self)

    def _get_fingerprint(self, length: int) -> Tensor:
        f = torch.zeros((length,))
        r = torch.randint(0, length, (1,))
        f[r] = 1.0
        return f

    def _get_delay(self, length: int) -> Tensor:
        return self.delay_sampler.sample((length,)).abs()

    @property
    @abstractmethod
    def memory_size(self) -> int:
        ...

    def get_ipd(self, idx: int) -> FloatTensor:
        return self.ipds[idx]

    def get_ipd_pad_mask(self, idx: int) -> BoolTensor:
        return torch.full((len(self.ipds[idx]),), False)

    @abstractmethod
    def get_fingerprint(self, idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_delay(self, idx: int) -> FloatTensor:
        ...


class DynamicFissionnDataset(FissionnDataset):

    @property
    def memory_size(self) -> int:
        return sum(tensor_memory_size(x) for x in self.ipds)

    def get_fingerprint(self, idx: int) -> Tensor:  # pylint: disable=unused-argument
        return self._get_fingerprint(self.fingerprint_length)

    def get_delay(self, idx: int) -> Tensor:
        ipd = self.get_ipd(idx)
        return self._get_delay(len(ipd))


class StaticFissionnDataset(FissionnDataset):

    def __post_init__(self) -> None:
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Fingerprints...")
        self.fingerprints = [self._get_fingerprint(self.fingerprint_length) for _ in iterable]
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Delays...")
        self.delays = [self._get_delay(len(ipd)) for ipd in iterable]

    @property
    def memory_size(self) -> int:
        m_ipds = sum(tensor_memory_size(x) for x in self.ipds)
        m_fingerprints = sum(tensor_memory_size(x) for x in self.fingerprints)
        m_delays = sum(tensor_memory_size(x) for x in self.delays)
        return m_ipds + m_fingerprints + m_delays

    def get_fingerprint(self, idx: int) -> Tensor:
        return self.fingerprints[idx]

    def get_delay(self, idx: int) -> Tensor:
        return self.delays[idx]


class FissionnCollateFn:

    def __init__(
        self,
        fingerprint_length: int,
        flow_length: int,
        paddding: Optional[Literal["long", "max"]] = None,
        truncate: bool = False,
        pad_value: int = 0,
    ) -> None:
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.padding = paddding
        self.truncate = truncate
        self.pad_value = pad_value

    def __call__(self, batch: list[Sample]) -> Samples:
        fingerprints, ipds, delays = [], [], []
        for fingerprint, ipd, delay, _ in batch:
            fingerprints.append(fingerprint)
            ipds.append(ipd)
            delays.append(delay)

        fingerprints, _ = self.prepare_sequence(fingerprints, self.fingerprint_length)
        ipds, ipd_pad_mask = self.prepare_sequence(ipds, self.flow_length)
        delays, _ = self.prepare_sequence(delays, self.flow_length)

        return fingerprints, ipds, delays, ipd_pad_mask

    def prepare_sequence(self, sequence: list[Tensor], maximum: Optional[int] = None) -> tuple[Tensor, Tensor]:
        if self.truncate:
            sequence = [s[0: maximum] for s in sequence]
        if self.padding is None:
            return torch.stack(sequence), torch.full((len(sequence), len(sequence[0])), False)
        if self.padding == "long":
            return pad_sequence_with_mask(sequence, True, self.pad_value)
        if self.padding == "max":
            return pad_sequence_with_mask([torch.cat((sequence[0], torch.full((maximum - len(sequence[0]),), self.pad_value)))] + sequence[1:], True, self.pad_value)
        raise ValueError(f"Invalid padding option: {self.padding}")


class FissionnModel(nn.Module):

    def __init__(self, fingerprint_length: int, flow_length: int, approximator: Approximator, length_scaler: StandardScaler) -> None:
        """
        Args:
          fingerprint_length: int - The size of the fingerprint tensor.
          flow_length: int - The size of the flow tensor.
          approximator: Approximator - Pretrained flow approximator to use.
          length_scaler: StandardScaler - Scaler to use when translating with the approximator.
        """
        super().__init__()
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.encoder = FinnEncoder(fingerprint_length, flow_length)
        self.decoder = FinnDecoder(flow_length, fingerprint_length)
        self.approximator = approximator
        self.length_scaler = length_scaler

    def forward(self, fingerprint: Tensor, ipd: Tensor, ipd_pad_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
          fingerprint: Tensor - The fingerprint tensor.
          ipd: Tensor - The inter-packet delays tensor.
          ipd_pad_mask - A boolean mask indicating which indices are padding within the ipd tensor.
        Returns:
          Tensor - The predicted fingerprint delay.
          Tensor - The predicted fingerprint.
        """

        if fingerprint.dim() != 2 or fingerprint.shape[1] != self.fingerprint_length:
            raise ShapeError(fingerprint.shape, ("*", self.fingerprint_length))
        if ipd.dim() != 2 or ipd.shape[1] != self.flow_length:
            raise ShapeError(ipd.shape, ("*", self.flow_length))

        delay_pred = self.encoder.forward(fingerprint)
        marked_ipd = ipd + torch.cumsum(delay_pred, dim=1)
        marked_ipd = FissionnModel.convert_for_translation_(marked_ipd, ipd_pad_mask)
        noisy_marked_ipd = self.approximator.translate(marked_ipd, self.length_scaler)
        noisy_marked_ipd = FissionnModel.convert_from_translation_(noisy_marked_ipd, ipd_pad_mask)
        fingerprint_pred = self.decoder(noisy_marked_ipd)

        return delay_pred, fingerprint_pred

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.approximator.eval()
        return self

    @staticmethod
    def convert_for_translation_(X: FloatTensor, M: BoolTensor) -> Tensor:

        B = X.shape[0]
        T = X.shape[1]

        dtype  = X.dtype
        device = X.device

        X[M] = PAD

        Z = []
        for i in range(B):
            x = X[i]
            m = M[i]

            pad_positions = torch.nonzero(m, as_tuple=False)
            idx = pad_positions[0].item() if len(pad_positions) > 0 else T

            z = torch.empty((T + 2,), dtype=dtype, device=device)

            z[0]            = BOS
            z[idx + 1]      = EOS
            z[1:1 + idx]    = x[:idx]
            z[1 + idx + 1:] = x[idx:]

            Z.append(z)

        return torch.stack(Z, dim=0)

    @staticmethod
    def convert_from_translation_(Z: FloatTensor, M: BoolTensor) -> Tensor:

        X = Z[:,1:-1]
        X[M] = 0.0
        return X


class FissionnTrainer(Trainer):

    model: FissionnModel
    tr_dataset: FissionnDataset
    vl_dataset: FissionnDataset
    collate_fn: FissionnCollateFn
    loss_fn: FinnLossFn
    tr_metric_keys = ("tr_loss", "tr_enc_loss", "tr_dec_loss", "tr_time",)

    def create_scheduler(self) -> Optional[LRScheduler]:
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=10)
        constant_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=10)
        decay_scheduler = ExponentialLR(self.optimizer, gamma=0.85)
        scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, constant_scheduler, decay_scheduler], milestones=[10, 20])
        return scheduler

    def create_stopper(self) -> None:
        return None

    def forward(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        fingerprint: Tensor = batch[0].to(self.args.device)
        ipd: Tensor = batch[1].to(self.args.device)
        ipd_pad_mask: Tensor = batch[3].to(self.args.device)
        delay_pred, fingerprint_pred = self.model.forward(fingerprint, ipd, ipd_pad_mask)
        return delay_pred, fingerprint_pred

    def compute_loss(self, batch: tuple[Tensor, Tensor, Tensor], outputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, float]]:
        fingerprint: Tensor = batch[0].to(self.args.device)
        delay: Tensor = batch[2].to(self.args.device)
        delay_pred = outputs[0].to(self.args.device)
        fingerprint_pred = outputs[1].to(self.args.device)
        loss, enc_loss, dec_loss = self.loss_fn.forward(delay_pred, delay, fingerprint_pred, fingerprint)
        return loss, {"enc_loss": enc_loss.item(), "dec_loss": dec_loss.item()}

    def compute_metrics(self, fingerprint: list[Tensor], fingerprint_pred: list[Tensor]) -> dict[str, float]:  # pylint: disable=arguments-differ
        fingerprint = torch.stack(fingerprint, dim=0)
        fingerprint_pred = torch.stack(fingerprint_pred, dim=0)

        predictions = F.softmax(fingerprint_pred, dim=-1)

        # Compute the bit-error rate.
        one_hot_predictions = torch.zeros_like(predictions)
        one_hot_predictions[torch.arange(len(predictions)), torch.argmax(predictions, dim=1)] = 1
        binary_fingerprint = one_hot_to_binary(fingerprint).to(bool)
        binary_predictions = one_hot_to_binary(one_hot_predictions).to(bool)
        bit_difference: Tensor = binary_fingerprint ^ binary_predictions
        bit_errors = bit_difference.sum() / math.log2(fingerprint.size(1))
        bit_error_rate = bit_errors / fingerprint.size(0)

        # Compute the extraction rate.
        y_true = torch.argmax(fingerprint, dim=1).tolist()
        y_pred = torch.argmax(predictions, dim=1).tolist()
        accuracy = accuracy_score(y_true, y_pred)

        return {"bit_error_rate": bit_error_rate.item(), "extraction_rate": accuracy}

    def get_compute_metrics_inputs(self, batch: tuple, outputs: tuple) -> dict[str, list[Tensor]]:
        return {
            "fingerprint": [z for z in batch[0]],         # pylint: disable=unnecessary-comprehension
            "fingerprint_pred": [z for z in outputs[1]],  # pylint: disable=unnecessary-comprehension
        }


def main():

    print(f"STARTING {datetime.now().isoformat()}")
    print("-" * 80)

    parser = TrainerArgumentParser(formatter_class=type("F", (ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter), {}))
    parser.add_argument("--approximator_file", type=str, default=None, help=".")
    parser.add_argument("--fingerprint_length", type=int, default=512, help=".")
    parser.add_argument("--flow_length", type=int, default=150, help=".")
    parser.add_argument("--min_flow_length", type=int, default=150, help=".")
    parser.add_argument("--max_flow_length", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--amplitude", type=float, default=40 / 1e3, help=".")
    parser.add_argument("--encoder_loss_weight", type=float, default=1.0, help=".")
    parser.add_argument("--decoder_loss_weight", type=float, default=5.0, help=".")
    parser.add_argument("--tr_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--vl_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--seed", type=int, default=0, help=".")
    parser.add_argument("--dynamic", action="store_true", help=".")
    parser.add_argument("--use_same_dataset", action="store_true", help=".")
    parser.add_argument("--demo", action="store_true", help=".")
    args = parser.parse_args()

    print(f"Command Line Arguments:\n{pformat(args.__dict__)}")
    print("-" * 80)

    if args.outdir.exists() and args.outdir.name not in Trainer.OVERWRITE_OUTDIRS:
        raise FileExistsError(f"Output Directory Already Exists: {args.outdir}")

    seed_everything(args.seed)

    stream_caida = stream_caida_data_demo if args.demo else stream_caida_data
    if args.use_same_dataset:
        stream = stream_caida(year="passive-2016", source="equinix-chicago")
        ipds = get_caida_ipds(stream, args.min_flow_length, args.max_flow_length, args.tr_num_samples + args.vl_num_samples)
        tr_ipds, vl_ipds = train_test_split(ipds, test_size=args.vl_num_samples)
    else:
        tr_stream = stream_caida(year="passive-2016", source="equinix-chicago")
        vl_stream = stream_caida(year="passive-2018", source="equinix-nyc")
        tr_ipds = get_caida_ipds(tr_stream, args.min_flow_length, args.max_flow_length, args.tr_num_samples)
        vl_ipds = get_caida_ipds(vl_stream, args.min_flow_length, args.max_flow_length, args.vl_num_samples)

    print(f"IPDs{' (demo)' if args.demo else ''}:")
    print(f"Training Size: {len(tr_ipds)}. Mean Length: {np.mean([len(ipd) for ipd in tr_ipds])}")
    print(f"Validation Size: {len(vl_ipds)}. Mean Length: {np.mean([len(ipd) for ipd in vl_ipds])}")
    print("-" * 80)

    DatasetConstructor = DynamicFissionnDataset if args.dynamic else StaticFissionnDataset
    DatasetConstructor = partial(
        DatasetConstructor,
        fingerprint_length=args.fingerprint_length,
        amplitude=args.amplitude,
        flow_length=args.flow_length,
    )
    tr_dataset = DatasetConstructor(tr_ipds)
    vl_dataset = DatasetConstructor(vl_ipds)
    gc.collect()

    print(f"Training Dataset:\n{tr_dataset}")
    print(f"Validation Dataset:\n{vl_dataset}")
    print("-" * 80)

    if args.approximator_file:
        approximator = from_pretrained(args.approximator_file)
        approximator.max_length = args.flow_length + 2
    else:
        approximator = RecurrentApproximator(args.flow_length + 2, 64, 2, "rnn")
    print(f"Approximator: {approximator}")

    with open("./cache/length_scaler.pkl", "rb") as fp:
        length_scaler: StandardScaler = pickle.load(fp)
    print(f"Length Scaler: {length_scaler}")

    model = FissionnModel(args.fingerprint_length, args.flow_length, approximator, length_scaler)
    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    trainer_args = TrainerArgs(
        outdir=args.outdir,
        device=args.device,
        epochs=args.epochs,
        tr_batch_size=args.tr_batch_size,
        vl_batch_size=args.vl_batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        disable_tqdm=args.disable_tqdm,
        logging_steps=args.logging_steps,
    )
    collate_fn = FissionnCollateFn(args.fingerprint_length, args.flow_length, "max", truncate=True)
    loss_fn = FinnLossFn(encoder_weight=args.encoder_loss_weight, decoder_weight=args.decoder_loss_weight)
    trainer = FissionnTrainer(trainer_args, model, tr_dataset, vl_dataset, collate_fn, loss_fn)

    trainer()

    print(f"ENDING {datetime.now().isoformat()}")
    print("-" * 80)


if __name__ == "__main__":
    main()
