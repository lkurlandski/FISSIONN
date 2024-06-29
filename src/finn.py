"""
Implementation of the network watermarking technique, "FINN".

CITATION
--------
@inproceedings{rezaei2021finn,
  title={FINN: Fingerprinting network flows using neural networks},
  author={Rezaei, Fatemeh and Houmansadr, Amir},
  booktitle={Proceedings of the 37th Annual Computer Security Applications Conference},
  year={2021}
}
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
from pprint import pformat, pprint
import random
import sys
from typing import Literal, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor
from torch.distributions import Laplace, Uniform, Normal
from torch.nn import functional as F, CrossEntropyLoss, L1Loss
from torch.optim import Optimizer, Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.caida import (
    stream_caida_data,
    stream_caida_data_demo,
    get_caida_ipds,
    CaidaSample,  # pylint: disable=unused-import  # Needed because of pickle
)
from src.utils import count_parameters, one_hot_to_binary, tensor_memory_size
# pylint: enable=wrong-import-position


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


class FINNDataset(Dataset, ABC):
    # TODO: the manner to produce noise is a tad ambigious.
    # This implementation uses a uniform distribution to sample the noise deviation.
    # This noise deviation is then used as the `scale` of the Laplace distribition to sample the noise.

    def __init__(
        self,
        ipds: list[np.ndarray],
        fingerprint_length: int,
        amplitude: int,
        noise_deviation_low: int,
        noise_deviation_high: int,
    ) -> None:
        self.ipds = [torch.from_numpy(ipd).to(torch.float32) for ipd in ipds]
        self.fingerprint_length = fingerprint_length
        self.aplitude = amplitude
        self.noise_deviation_low = noise_deviation_low
        self.noise_deviation_high = noise_deviation_high
        self.delay_sampler = Laplace(0, amplitude)
        self.noise_sampler_sampler = Uniform(noise_deviation_low, noise_deviation_high)
        self.__post_init__()

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ipd = self.get_ipd(idx)
        fingerprint = self.get_fingerprint(idx)
        delay = self.get_delay(idx)
        noise = self.get_noise(idx)
        return fingerprint, ipd, delay, noise

    def __len__(self) -> int:
        return len(self.ipds)

    def __post_init__(self) -> None:
        pass

    def _get_ipd(self, idx: int) -> Tensor:
        return self.ipds[idx]

    def _get_fingerprint(self, length: int) -> Tensor:
        f = torch.zeros((length,))
        r = torch.randint(0, length, (1,))
        f[r] = 1.0
        return f

    def _get_delay(self, length: int) -> Tensor:
        return self.delay_sampler.sample((length,)).abs()

    def _get_noise(self, length: int) -> Tensor:
        sampler = Laplace(0, self.noise_sampler_sampler.sample().abs().item())
        return sampler.sample((length,)).abs()

    @property
    @abstractmethod
    def memory_size(self) -> int:
        ...

    @abstractmethod
    def get_ipd(self, idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_fingerprint(self, idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_delay(self, idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_noise(self, idx: int) -> Tensor:
        ...


class DynamicFINNDataset(FINNDataset):

    def memory_size(self) -> int:
        return sum(tensor_memory_size(x) for x in self.ipds)

    def get_ipd(self, idx: int) -> Tensor:
        return self._get_ipd(idx)

    def get_fingerprint(self, idx: int) -> Tensor:  # pylint: disable=unused-argument
        return self._get_fingerprint(self.fingerprint_length)

    def get_delay(self, idx: int) -> Tensor:
        ipd = self.get_ipd(idx)
        return self._get_delay(len(ipd))

    def get_noise(self, idx: int) -> Tensor:
        ipd = self.get_ipd(idx)
        return self._get_noise(len(ipd))


class StaticFINNDataset(FINNDataset):

    def __post_init__(self) -> None:
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Fingerprints...")
        self.fingerprints = [self._get_fingerprint(self.fingerprint_length) for _ in iterable]
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Delays...")
        self.delays = [self._get_delay(len(ipd)) for ipd in iterable]
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Noises...")
        self.noises = [self._get_noise(len(ipd)) for ipd in iterable]

    @property
    def memory_size(self) -> int:
        m_ipds = sum(tensor_memory_size(x) for x in self.ipds)
        m_fingerprints = sum(tensor_memory_size(x) for x in self.fingerprints)
        m_delays = sum(tensor_memory_size(x) for x in self.delays)
        m_noises = sum(tensor_memory_size(x) for x in self.noises)
        return m_ipds + m_fingerprints + m_delays + m_noises

    def get_ipd(self, idx: int) -> Tensor:
        return self._get_ipd(idx)

    def get_fingerprint(self, idx: int) -> Tensor:
        return self.fingerprints[idx]

    def get_delay(self, idx: int) -> Tensor:
        return self.delays[idx]

    def get_noise(self, idx: int) -> Tensor:
        return self.noises[idx]


class FINNCollateFn:

    def __init__(
        self,
        fingerprint_length: int,
        flow_length: int,
        paddding: Optional[Literal["long", "max"]] = None,
        truncate: bool = False,
        pad_to: int = 1,
        pad_value: int = 0,
    ) -> None:
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.padding = paddding
        self.truncate = truncate
        self.pad_to = pad_to
        self.pad_value = pad_value

    def __call__(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        fingerprints, ipds, delays, noises = [], [], [], []
        for fingerprint, ipd, delay, noise in batch:
            fingerprints.append(fingerprint)
            ipds.append(ipd)
            delays.append(delay)
            noises.append(noise)

        fingerprints = self.prepare_sequence(fingerprints, self.fingerprint_length)
        ipds = self.prepare_sequence(ipds, self.flow_length)
        delays = self.prepare_sequence(delays, self.flow_length)
        noises = self.prepare_sequence(noises, self.flow_length)

        return fingerprints, ipds, delays, noises

    def prepare_sequence(self, sequence: list[Tensor], maximum: Optional[int] = None) -> Tensor:
        if self.truncate:
            sequence = [s[0: maximum] for s in sequence]
        if self.padding is None:
            return torch.stack(sequence)
        if self.padding == "long":
            return pad_sequence(sequence, True, self.pad_value)
        if self.padding == "max":
            return pad_sequence(sequence + [torch.zeros((maximum,))], True, self.pad_value)[0:len(sequence)]
        raise ValueError(f"Invalid padding option: {self.padding}")


class FINNLoss(nn.Module):

    def __init__(
        self,
        encoder_weight: float = 1.0,
        decoder_weight: float = 5.0,
    ) -> None:
        """
        Args:
            encoder_weight (float): weight to use for the encoder's loss
            decoder_weight (float): weight to use for the decoder's loss
        """
        super().__init__()
        self.encoder_weight = encoder_weight
        self.encoder_loss_fn = L1Loss()
        self.decoder_weight = decoder_weight
        self.decoder_loss_fn = CrossEntropyLoss()

    def forward(
        self,
        encoder_logits: Tensor,
        encoder_labels: Tensor,
        decoder_logits: Tensor,
        decoder_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            encoder_logits (Tensor): Predicted timing delays from the encoder.
            encoder_labels (Tensor): True timing delays corresponding to each packet in the flow.
            decoder_logits (Tensor): Unnormalized logits from the decoder predicting the fingerprint.
            decoder_labels (Tensor): One-hot or categorical labels indicating the true fingerprint.

        Returns:
            Tensor: Cumulative loss of the encoder and decoder, the encoder loss, and the decoder loss.
        """
        encoder_loss = self.get_encoder_loss(encoder_logits, encoder_labels)
        decoder_loss = self.get_decoder_loss(decoder_logits, decoder_labels)
        weighted_loss = encoder_loss + decoder_loss
        return weighted_loss, encoder_loss, decoder_loss

    def get_encoder_loss(self, encoder_logits: Tensor, encoder_labels: Tensor) -> Tensor:
        return self.encoder_weight * self.encoder_loss_fn(encoder_logits, encoder_labels)

    def get_decoder_loss(self, decoder_logits: Tensor, decoder_labels: Tensor) -> Tensor:
        return self.decoder_weight * self.decoder_loss_fn(decoder_logits, decoder_labels)  # argmax?


class FINNEncoder(nn.Module):
    # TODO: There is ambiguity about what the input to the encoder is.
    # Eequation 1 indicates that the input to the encoder is the fingerprint
    # concatenated with the network noise, but the first paragraph in
    # section 4.1 states "The encoder takes IPDs and fingerprints to generate fingerprinting delays"
    # which indicates that the encoder takes the fingerprint and the IPDs as input.
    # TODO: There is ambiguity about the sizes of individual layers in the paper,
    # specifically, the prose in section 4.3 and the values given in Table 2 contradict.

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Args:
          input_size: int - The size of the input tensor, e.g., 
            the fingerprint length and flow/noise length.
          output_size: int - The size of the output tensor, e.g., the flow length.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_1 = nn.Linear(input_size, 1024)
        self.layer_2 = nn.Linear(1024, 2048)
        self.layer_3 = nn.Linear(2048, 512)
        self.layer_4 = nn.Linear(512, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
          x: Tensor - The input tensor, e.g., the fingerprint.
        Returns:
          Tensor - The output tensor, e.g., the fingerprint delay ([alpha0, alpha1, ... alphaN])
        """
        if x.dim() != 2 or x.shape[1] != self.input_size:
            raise ShapeError(x.shape, ("*", self.input_size))

        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)
        x = self.layer_4(x)
        return x


class FINNDecoder(nn.Module):
    # TODO: Should the CNN layers be 2D CNNs? The paper seems to indicate this,
    # but such a choice is unusual given the nature of the decoder input.

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Args:
          input_size: int - The size of the input tensor, e.g., the flow length.
          output_size: int - The size of the output tensor, e.g., the fingerprint length.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv_1 = nn.Conv1d(1, 50, 10, stride=1)
        self.conv_2 = nn.Conv1d(50, 10, 10, stride=1)
        self.mlp_1 = nn.Linear(self.compute_mlp_input_size(), 1024)
        self.mlp_2 = nn.Linear(1024, 2048)
        self.mlp_3 = nn.Linear(2048, self.output_size)

    def compute_mlp_input_size(self) -> int:
        output_length_1 = (self.input_size - self.conv_1.kernel_size[0]) // self.conv_1.stride[0] + 1
        output_length_2 = (output_length_1 - self.conv_2.kernel_size[0]) // self.conv_2.stride[0] + 1
        return self.conv_2.out_channels * output_length_2

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
          x: Tensor - The input tensor, e.g., the noisy IPDs with the fingerprint.
        Returns:
          Tensor - The output tensor, e.g., the fingerprint prediction.
        """
        if x.dim() != 2 or x.shape[1] != self.input_size:
            raise ShapeError(x.shape, ("*", self.input_size))

        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.mlp_1(x)
        x = F.relu(x)
        x = self.mlp_2(x)
        x = F.relu(x)
        x = self.mlp_3(x)

        return x


class FINNModel(nn.Module):

    def __init__(self, fingerprint_length: int, flow_length: int) -> None:
        """
        Args:
          fingerprint_length: int - The size of the fingerprint tensor.
          flow_length: int - The size of the flow tensor.
        """
        super().__init__()
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.encoder = FINNEncoder(fingerprint_length + flow_length, flow_length)
        self.decoder = FINNDecoder(flow_length, fingerprint_length)

    def forward(self, fingerprint: Tensor, ipd: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
          fingerprint: Tensor - The fingerprint tensor.
          ipd: Tensor - The inter-packet delays tensor.
          noise: Tensor - The noise tensor.
        Returns:
          Tensor - The predicted fingerprint delay.
          Tensor - The predicted fingerprint.
        """
        if fingerprint.dim() != 2 or fingerprint.shape[1] != self.fingerprint_length:
            raise ShapeError(fingerprint.shape, ("*", self.fingerprint_length))
        if ipd.dim() != 2 or ipd.shape[1] != self.flow_length:
            raise ShapeError(ipd.shape, ("*", self.flow_length))
        if noise.dim() != 2 or noise.shape[1] != self.flow_length:
            raise ShapeError(noise.shape, ("*", self.flow_length))

        fingerprint_and_noise = torch.cat((fingerprint, noise), dim=1)
        delay_pred = self.encoder.forward(fingerprint_and_noise)
        noisy_marked_ipd = self.combine_3(ipd, delay_pred, noise)
        fingerprint_pred = self.decoder(noisy_marked_ipd)

        return delay_pred, fingerprint_pred

    def combine_1(self, ipd: Tensor, delay: Tensor, noise: Tensor) -> Tensor:
        return ipd + delay + noise

    def combine_2(self, ipd: Tensor, delay: Tensor, noise: Tensor) -> Tensor:
        return ipd + torch.cumsum(delay, dim=1) + noise

    def combine_3(self, ipd: Tensor, delay: Tensor, noise: Tensor) -> Tensor:
        return ipd + torch.cumsum(delay, dim=1) + torch.cumsum(noise, dim=1)


@dataclass
class FINNTrainerArgs:
    outfile: Path
    num_train_epochs: int = 1
    batch_size: int = 1
    dataloader_num_workers: int = 0
    device: torch.cuda.device = torch.device("cpu")
    disable_tqdm: bool = False
    logging_steps: int = -1


class FINNTrainer:

    def __init__(
        self,
        args: FINNTrainerArgs,
        model: FINNModel,
        tr_dataset: FINNDataset,
        vl_dataset: FINNDataset,
        collate_fn: FINNCollateFn,
        loss_fn: FINNLoss,
        optimizer: Optimizer,
    ) -> None:
        self.args = args
        self.model = model
        self.model.to(self.args.device)
        self.tr_dataset = tr_dataset
        self.vl_dataset = vl_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.log = []

    def __call__(self) -> None:
        if self.args.outfile.exists():
            raise FileExistsError(f"Logfile already exists: {self.args.outfile}")
        self.args.outfile.parent.mkdir(parents=True, exist_ok=True)

        tr_metrics = {"tr_weighted_loss": float("nan"), "tr_encoder_loss": float("nan"), "tr_decoder_loss": float("nan")}
        vl_metrics = self.evaluate()
        d = {"epoch": 0} | tr_metrics | vl_metrics
        self.log.append(d)
        print(self._fmt_dict(d))

        pbar = self._get_pbar(range(1, self.args.num_train_epochs + 1), desc="Epochs")
        for epoch in pbar:
            tr_metrics = self.train()
            vl_metrics = self.evaluate()
            d = {"epoch": epoch} | tr_metrics | vl_metrics
            self.log.append(d)
            print(self._fmt_dict(d))
            with open(self.args.outfile, "a") as fp:
                fp.write(json.dumps(d) + "\n")

    def _get_pbar(self, iterable: Iterable, **kwds) -> tqdm | Iterable:
        if self.args.disable_tqdm:
            return iterable
        return tqdm(iterable, **kwds)

    def _fmt_dict(self, d: dict[str, float]) -> dict[str, str]:
        return {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in d.items()}

    def train(self) -> dict[str, float]:
        self.model.train()
        cum_samples, cum_weighted_loss, cum_encoder_loss, cum_decoder_loss = 0, 0, 0, 0
        log_samples, log_weighted_loss, log_encoder_loss, log_decoder_loss = 0, 0, 0, 0
        dataloader = self.get_dataloader(self.tr_dataset, shuffle=True)
        pbar = self._get_pbar(dataloader, total=len(self.tr_dataset) // self.args.batch_size)
        for i, batch in enumerate(pbar):

            fingerprint: Tensor = batch[0].to(self.args.device)
            ipd: Tensor = batch[1].to(self.args.device)
            delay: Tensor = batch[2].to(self.args.device)
            noise: Tensor = batch[3].to(self.args.device)

            self.model.zero_grad()
            delay_pred, fingerprint_pred = self.model.forward(fingerprint, ipd, noise)
            weighted_loss, encoder_loss, decoder_loss = self.loss_fn.forward(delay_pred, delay, fingerprint_pred, fingerprint)
            weighted_loss.backward()
            self.optimizer.step()

            cum_samples += fingerprint.size(0)
            cum_weighted_loss += weighted_loss.item()
            cum_encoder_loss += encoder_loss.item()
            cum_decoder_loss += decoder_loss.item()

            log_samples += fingerprint.size(0)
            log_weighted_loss += weighted_loss.item()
            log_encoder_loss += encoder_loss.item()
            log_decoder_loss += decoder_loss.item()

            if self.args.logging_steps > 0 and i % self.args.logging_steps == 0:
                d = {
                    "log_weighted_loss": log_weighted_loss / log_samples,
                    "log_encoder_loss": log_encoder_loss / log_samples,
                    "log_decoder_loss": log_decoder_loss / log_samples,
                }
                d = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in d.items()}
                log_samples, log_weighted_loss, log_encoder_loss, log_decoder_loss = 0, 0, 0, 0
                print(d)

        d = {
            "tr_weighted_loss": cum_weighted_loss / cum_samples,
            "tr_encoder_loss": cum_encoder_loss / cum_samples,
            "tr_decoder_loss": cum_decoder_loss / cum_samples,
        }

        return d

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        cum_samples, cum_weighted_loss, cum_encoder_loss, cum_decoder_loss = 0, 0, 0, 0
        bit_errors, y_true, y_pred = 0, [], []
        dataloader = self.get_dataloader(self.vl_dataset, shuffle=False)
        pbar = self._get_pbar(dataloader, total=len(self.vl_dataset) // self.args.batch_size)
        for batch in pbar:

            fingerprint: Tensor = batch[0].to(self.args.device)
            ipd: Tensor = batch[1].to(self.args.device)
            delay: Tensor = batch[2].to(self.args.device)
            noise: Tensor = batch[3].to(self.args.device)

            delay_pred, fingerprint_pred = self.model.forward(fingerprint, ipd, noise)
            weighted_loss, encoder_loss, decoder_loss = self.loss_fn.forward(delay_pred, delay, fingerprint_pred, fingerprint)
            predictions = F.softmax(fingerprint_pred, dim=-1)

            one_hot_predictions = torch.zeros_like(predictions)
            one_hot_predictions[torch.arange(len(predictions)), torch.argmax(predictions, dim=1)] = 1
            binary_fingerprint = one_hot_to_binary(fingerprint).to(bool)
            binary_predictions = one_hot_to_binary(one_hot_predictions).to(bool)
            bit_difference = binary_fingerprint ^ binary_predictions
            bit_errors += bit_difference.sum().item()

            cum_samples += fingerprint.size(0)
            cum_weighted_loss += weighted_loss.item()
            cum_encoder_loss += encoder_loss.item()
            cum_decoder_loss += decoder_loss.item()

            y_true.extend(torch.argmax(fingerprint, dim=1).tolist())
            y_pred.extend(torch.argmax(predictions, dim=1).tolist())

        d = {
            "vl_weighted_loss": cum_weighted_loss / cum_samples,
            "vl_encoder_loss": cum_encoder_loss / cum_samples,
            "vl_decoder_loss": cum_decoder_loss / cum_samples,
            "bit_error_rate": bit_errors / cum_samples,
            "extraction_rate": accuracy_score(y_true, y_pred),
        }

        return d

    def get_dataloader(self, dataset: FINNDataset, shuffle: bool = False) -> None:
        return DataLoader(
            dataset,
            self.args.batch_size,
            shuffle,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )


def main():

    print(f"STARTING {datetime.now().isoformat()}")
    print("-" * 80)

    parser = ArgumentParser(formatter_class=type("F", (ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter), {}))
    parser.add_argument("--fingerprint_length", type=int, default=512, help=".")
    parser.add_argument("--flow_length", type=int, default=150, help=".")
    parser.add_argument("--min_flow_length", type=int, default=150, help=".")
    parser.add_argument("--max_flow_length", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--amplitude", type=float, default=40 / 1e3, help=".")
    parser.add_argument("--noise_deviation_low", type=float, default=2 / 1e3, help=".")
    parser.add_argument("--noise_deviation_high", type=float, default=10 / 1e3, help=".")
    parser.add_argument("--encoder_loss_weight", type=float, default=1.0, help=".")
    parser.add_argument("--decoder_loss_weight", type=float, default=5.0, help=".")
    parser.add_argument("--tr_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--vl_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--seed", type=int, default=0, help=".")
    parser.add_argument("--outfile", type=Path, default=Path("./output/tmp.jsonl"), help=".")
    parser.add_argument("--num_train_epochs", type=int, default=1, help=".")
    parser.add_argument("--batch_size", type=int, default=2, help=".")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help=".")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help=".")
    parser.add_argument("--device", type=torch.device, default="cpu", help=".")
    parser.add_argument("--disable_tqdm", action="store_true", help=".")
    parser.add_argument("--logging_steps", type=int, default=-1, help=".")
    parser.add_argument("--dynamic", action="store_true", help=".")
    parser.add_argument("--use_same_dataset", action="store_true", help=".")
    parser.add_argument("--demo", action="store_true", help=".")
    args = parser.parse_args()

    print(f"Command Line Arguments:\n{pformat(args.__dict__)}")
    print("-" * 80)

    if args.outfile.exists():
        raise FileExistsError(f"Logfile already exists: {args.outfile}")

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

    DatasetConstructor = DynamicFINNDataset if args.dynamic else StaticFINNDataset
    DatasetConstructor = partial(
        DatasetConstructor,
        fingerprint_length=args.fingerprint_length,
        amplitude=args.amplitude,
        noise_deviation_low=args.noise_deviation_low,
        noise_deviation_high=args.noise_deviation_high,
    )
    tr_dataset = DatasetConstructor(tr_ipds)
    vl_dataset = DatasetConstructor(vl_ipds)

    print(f"Training Dataset:\n{tr_dataset}\nMemory Size: {tr_dataset.memory_size}")
    print(f"Validation Dataset:\n{vl_dataset}\nMemory Size: {vl_dataset.memory_size}")
    print("-" * 80)

    model = FINNModel(args.fingerprint_length, args.flow_length)
    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    training_arguments = FINNTrainerArgs(
        outfile=args.outfile,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        device=args.device,
        disable_tqdm=args.disable_tqdm,
        logging_steps=args.logging_steps,
    )
    collate_fn = FINNCollateFn(args.fingerprint_length, args.flow_length, "max", truncate=True)
    loss_fn = FINNLoss(encoder_weight=args.encoder_loss_weight, decoder_weight=args.decoder_loss_weight)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    trainer = FINNTrainer(training_arguments, model, tr_dataset, vl_dataset, collate_fn, loss_fn, optimizer)

    trainer()

    print(f"ENDING {datetime.now().isoformat()}")
    print("-" * 80)


if __name__ == "__main__":
    main()
