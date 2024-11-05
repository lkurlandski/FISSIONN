"""
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
import sys
import time
from typing import Literal, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor
from torch.distributions import Laplace, Uniform
from torch.nn import functional as F, CrossEntropyLoss, L1Loss
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.approximator import (
    TransformerApproximator,
    RecurrentApproximator,
    PositionalEncoding,
    Attention,
    PAD,
    BOS,
    pad,
    bos,
)
from src.data import load_data
from src.finn import FinnLossFn, FinnEncoder, FinnDecoder
from src.trainer import TrainerArgs, Trainer, TrainerArgumentParser
from src.utils import (
    count_parameters,
    one_hot_to_binary,
    seed_everything,
    ShapeError,
)
# pylint: enable=wrong-import-position


class FinnSSIDataset(Dataset):

    def __init__(
        self,
        ipds: list,
        fingerprint_length: int,
        amplitude: int,
        flow_length: Optional[int] = None,
    ) -> None:
        self.ipds = [torch.tensor(ipd).to(torch.float32) for ipd in ipds]
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.delay_sampler = Laplace(0, amplitude)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        ipd = self._get_ipd(idx)
        fingerprint = self._get_fingerprint(idx)
        delay = self._get_delay(idx)
        return fingerprint, ipd, delay

    def __len__(self) -> int:
        return len(self.ipds)

    def _get_ipd(self, idx: int) -> Tensor:
        return self.ipds[idx]

    def _get_fingerprint(self, idx: int) -> Tensor:
        f = torch.zeros((self.fingerprint_length,))
        r = torch.randint(0, self.fingerprint_length, (1,))
        f[r] = 1.0
        return f

    def _get_delay(self, idx: int) -> Tensor:
        return self.delay_sampler.sample((len(self._get_ipd(idx)),)).abs()


class FinnSSICollateFn:

    def __init__(
        self,
        fingerprint_length: int,
        flow_length: int,
        paddding: Optional[Literal["long", "max"]] = None,
        truncate: bool = False,
    ) -> None:
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.padding = paddding
        self.truncate = truncate

    def __call__(self, batch: list[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        fingerprints, ipds, delays = [], [], []
        for fingerprint, ipd, delay in batch:
            fingerprints.append(fingerprint)
            ipds.append(ipd)
            delays.append(delay)

        fingerprints = self.prepare_sequence(fingerprints, self.fingerprint_length)
        ipds = self.prepare_sequence(ipds, self.flow_length)
        delays = self.prepare_sequence(delays, self.flow_length)

        return fingerprints, ipds, delays

    def prepare_sequence(self, sequence: list[Tensor], maximum: Optional[int] = None) -> Tensor:
        if self.truncate:
            sequence = [s[0: maximum] for s in sequence]
        if self.padding is None:
            return torch.stack(sequence)
        if self.padding == "long":
            return pad_sequence(sequence, True, PAD)
        if self.padding == "max":
            return pad_sequence(sequence + [torch.zeros((maximum,))], True, PAD)[0:len(sequence)]
        raise ValueError(f"Invalid padding option: {self.padding}")


class FinnSSIModel(nn.Module):

    def __init__(
        self,
        encoder: FinnEncoder,
        decoder: FinnDecoder,
        approximator: TransformerApproximator | RecurrentApproximator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.approximator = approximator
        for param in self.approximator.parameters():
            param.requires_grad = False

    def forward(self, fingerprint: Tensor, ipd: Tensor) -> tuple[Tensor, Tensor]:
        # Feed the fingerprint and the ipd to the encoder to get the delays
        encoder_input = torch.cat((fingerprint, ipd), dim=1)  # requires_grad == False
        delays = self.encoder.forward(encoder_input)          # requires_grad == True
        noisy_ipd = self.get_noisy_ipd(ipd, delays)           # requires_grad == False
        marked_ipd = noisy_ipd + torch.cumsum(delays, dim=1)  # requires_grad == True
        fingerprint = self.decoder.forward(marked_ipd)        # requires_grad == True
        return delays, fingerprint

    def get_noisy_ipd(self, ipd: Tensor, delays: Tensor) -> Tensor:
        inputs = torch.cat([bos((ipd.size(0), 1)).to(ipd.device), ipd], dim=1)
        targets = self.approximator.translate(inputs)[:, :delays.size(1)]
        padding = pad((targets.size(0), max(0, targets.size(1) - delays.size(1)))).to(ipd.device)
        noisy_ipd = torch.cat([targets, padding], dim=1)
        return noisy_ipd


class FinnTrainer(Trainer):

    model: FinnSSIModel
    tr_dataset: FinnSSIDataset
    vl_dataset: FinnSSIDataset
    collate_fn: FinnSSICollateFn
    loss_fn: FinnLossFn
    tr_metric_keys = ("tr_loss", "tr_enc_loss", "tr_dec_loss", "tr_time",)

    # def create_scheduler(self) -> Optional[LRScheduler]:
    #     return ExponentialLR(self.optimizer, gamma=0.995)

    def create_stopper(self) -> None:
        return None

    def forward(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        fingerprint: Tensor = batch[0].to(self.args.device)
        ipd: Tensor = batch[1].to(self.args.device)

        delay_pred, fingerprint_pred = self.model.forward(fingerprint, ipd)

        return delay_pred, fingerprint_pred

    def compute_loss(self, batch: tuple[Tensor, Tensor, Tensor], outputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, float]]:
        fingerprint: Tensor = batch[0].to(self.args.device)
        delay: Tensor = batch[2].to(self.args.device)
        delay_pred = outputs[0].to(self.args.device)
        fingerprint_pred = outputs[1].to(self.args.device)

        loss, enc_loss, dec_loss = self.loss_fn.forward(delay_pred, delay, fingerprint_pred, fingerprint)

        return loss, {"enc_loss": enc_loss.item(), "dec_loss": dec_loss.item()}

    # TODO: compute_metrics should operate over the output of the entire evaluation set, not just a single batch.
    def compute_metrics(self, batch: tuple[Tensor, Tensor, Tensor], outputs: tuple[Tensor, Tensor]) -> dict[str, float]:
        fingerprint = batch[0].to(self.args.device)
        fingerprint_pred = outputs[1].to(self.args.device)

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


def main():

    print(f"STARTING {datetime.now().isoformat()}")
    print("-" * 80)

    parser = TrainerArgumentParser(formatter_class=type("F", (ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter), {}))
    parser.add_argument("--approximator_file", type=Path, required=True, help=".")
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
    args = parser.parse_args()

    print(f"Command Line Arguments:\n{pformat(args.__dict__)}")
    print("-" * 80)

    if args.outdir.exists() and args.outdir.name not in Trainer.OVERWRITE_OUTDIRS:
        raise FileExistsError(f"Output Directory Already Exists: {args.outdir}")

    seed_everything(args.seed)

    ipd_groups = []
    for group in tqdm(load_data(), desc="Loading Data..."):
        ipd_groups.append([ipds.tolist() for ipds in group])
    print(f"Collected {sum(len(group) for group in ipd_groups)} IPDs from {len(ipd_groups)} groups.")
    print("-" * 80)

    DatasetConstructor = partial(
        FinnSSIDataset,
        fingerprint_length=args.fingerprint_length,
        amplitude=args.amplitude,
        flow_length=args.flow_length,
    )

    tr_ipd_groups, vl_ipd_groups = train_test_split(ipd_groups, test_size=0.10)
    tr_ipds = [ipd for group in tr_ipd_groups for ipd in group]
    vl_ipds = [ipd for group in vl_ipd_groups for ipd in group]
    tr_dataset = DatasetConstructor(tr_ipds)
    vl_dataset = DatasetConstructor(vl_ipds)
    tr_dataset = Subset(tr_dataset, range(min(args.tr_num_samples, len(tr_dataset))))
    vl_dataset = Subset(vl_dataset, range(min(args.vl_num_samples, len(vl_dataset))))
    gc.collect()
    print(f"Training Dataset:\n{tr_dataset}")
    print(f"Validation Dataset:\n{vl_dataset}")
    print("-" * 80)

    encoder = FinnEncoder(args.fingerprint_length + args.flow_length, args.flow_length)
    decoder = FinnDecoder(args.flow_length, args.fingerprint_length)
    approximator = TransformerApproximator.from_pretrained(args.approximator_file)
    model = FinnSSIModel(encoder, decoder, approximator)
    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
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
    collate_fn = FinnSSICollateFn(args.fingerprint_length, args.flow_length, "max", truncate=True)
    loss_fn = FinnLossFn(encoder_weight=args.encoder_loss_weight, decoder_weight=args.decoder_loss_weight)
    trainer = FinnTrainer(trainer_args, model, tr_dataset, vl_dataset, collate_fn, loss_fn)

    trainer()

    print(f"ENDING {datetime.now().isoformat()}")
    print("-" * 80)


if __name__ == "__main__":
    main()
