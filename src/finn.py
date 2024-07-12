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
from src.trainer import TrainerArgs, Trainer, TrainerArgumentParser
from src.utils import (
    count_parameters,
    one_hot_to_binary,
    tensor_memory_size,
    seed_everything,
    ShapeError,
)
# pylint: enable=wrong-import-position


class FinnDataset(Dataset, ABC):
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
        as_tensors: bool = False,
        flow_length: Optional[int] = None,
        pad_value: int = 0,
        use_different_noises: bool = False,
    ) -> None:
        if as_tensors and not isinstance(flow_length, int):
            raise ValueError("If `as_tensors` is True, a `flow_length` must be provided.")

        self.ipds = [torch.from_numpy(ipd).to(torch.float32) for ipd in ipds]
        self.fingerprint_length = fingerprint_length
        self.aplitude = amplitude
        self.noise_deviation_low = noise_deviation_low
        self.noise_deviation_high = noise_deviation_high
        self.as_tensors = as_tensors
        self.flow_length = flow_length
        self.pad_value = pad_value
        self.delay_sampler = Laplace(0, amplitude)
        self.noise_sampler_sampler = Uniform(noise_deviation_low, noise_deviation_high)
        self.use_different_noises = use_different_noises

        if self.as_tensors:
            self.ipds = [ipd[0: self.flow_length] for ipd in self.ipds]
            self.ipds = pad_sequence(self.ipds, True, self.pad_value)

        self.__post_init__()

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        ipd = self.get_ipd(idx)
        fingerprint = self.get_fingerprint(idx)
        delay = self.get_delay(idx)
        noise_1 = self.get_noise_1(idx)
        noise_2 = self.get_noise_2(idx)
        return fingerprint, ipd, delay, noise_1, noise_2

    def __len__(self) -> int:
        return len(self.ipds)

    def __post_init__(self) -> None:
        pass

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  ipds={len(self.ipds)},{self.ipds[0].dtype}\n"
        s += f"  fingerprint_length={self.fingerprint_length},\n"
        s += f"  amplitude={self.aplitude},\n"
        s += f"  noise_deviation_low={self.noise_deviation_low},\n"
        s += f"  noise_deviation_high={self.noise_deviation_high},\n"
        s += f"  as_tensors={self.as_tensors},\n"
        s += f"  flow_length={self.flow_length},\n"
        s += f"  pad_value={self.pad_value},\n"
        s += f"  delay_sampler={self.delay_sampler},\n"
        s += f"  noise_sampler_sampler={self.noise_sampler_sampler},\n"
        s += f"  memory_size={round(self.memory_size / 1e9, 2)}G,\n"
        s += ")"
        return s

    def __str__(self) -> str:
        return repr(self)

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
    def get_noise_1(self, idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_noise_2(self, idx: int) -> Optional[Tensor]:
        ...


class DynamicFinnDataset(FinnDataset):

    @property
    def memory_size(self) -> int:
        return sum(tensor_memory_size(x) for x in self.ipds)

    def get_ipd(self, idx: int) -> Tensor:
        return self._get_ipd(idx)

    def get_fingerprint(self, idx: int) -> Tensor:  # pylint: disable=unused-argument
        return self._get_fingerprint(self.fingerprint_length)

    def get_delay(self, idx: int) -> Tensor:
        ipd = self.get_ipd(idx)
        return self._get_delay(len(ipd))

    def get_noise_1(self, idx: int) -> Tensor:
        ipd = self.get_ipd(idx)
        return self._get_noise(len(ipd))

    def get_noise_2(self, idx: int) -> Optional[Tensor]:
        if not self.use_different_noises:
            return None
        ipd = self.get_ipd(idx)
        return self._get_noise(len(ipd))


class StaticFinnDataset(FinnDataset):

    def __post_init__(self) -> None:
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Fingerprints...")
        self.fingerprints = [self._get_fingerprint(self.fingerprint_length) for _ in iterable]
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Delays...")
        self.delays = [self._get_delay(len(ipd)) for ipd in iterable]
        iterable = tqdm(self.ipds, total=len(self), desc="Generating Noises...")
        self.noises_1 = [self._get_noise(len(ipd)) for ipd in iterable]
        if self.use_different_noises:
            iterable = tqdm(self.ipds, total=len(self), desc="Generating Noises...")
            self.noises_2 = [self._get_noise(len(ipd)) for ipd in iterable]

        if self.as_tensors:
            # If True, the ipds should have already been padded to the same length, so we can stack
            # without concern that the tensors may have different lengths.
            self.fingerprints = torch.stack(self.fingerprints, dim=0)
            self.delays = torch.stack(self.delays, dim=0)
            self.noises_1 = torch.stack(self.noises_1, dim=0)
            if self.use_different_noises:
                self.noises_2 = torch.stack(self.noises_2, dim=0)

    @property
    def memory_size(self) -> int:
        m_ipds = sum(tensor_memory_size(x) for x in self.ipds)
        m_fingerprints = sum(tensor_memory_size(x) for x in self.fingerprints)
        m_delays = sum(tensor_memory_size(x) for x in self.delays)
        m_noises = sum(tensor_memory_size(x) for x in self.noises_1)
        if self.use_different_noises:
            m_noises += sum(tensor_memory_size(x) for x in self.noises_2)
        return m_ipds + m_fingerprints + m_delays + m_noises

    def get_ipd(self, idx: int) -> Tensor:
        return self._get_ipd(idx)

    def get_fingerprint(self, idx: int) -> Tensor:
        return self.fingerprints[idx]

    def get_delay(self, idx: int) -> Tensor:
        return self.delays[idx]

    def get_noise_1(self, idx: int) -> Tensor:
        return self.noises_1[idx]

    def get_noise_2(self, idx: int) -> Optional[Tensor]:
        if not self.use_different_noises:
            return None
        return self.noises_2[idx]


class FinnCollateFn:

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

    def __call__(self, batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        fingerprints, ipds, delays, noises_1, noises_2 = [], [], [], [], []
        for fingerprint, ipd, delay, noise_1, noise_2 in batch:
            fingerprints.append(fingerprint)
            ipds.append(ipd)
            delays.append(delay)
            noises_1.append(noise_1)
            noises_2.append(noise_2)

        fingerprints = self.prepare_sequence(fingerprints, self.fingerprint_length)
        ipds = self.prepare_sequence(ipds, self.flow_length)
        delays = self.prepare_sequence(delays, self.flow_length)
        noises_1 = self.prepare_sequence(noises_1, self.flow_length)
        noises_2 = self.prepare_sequence(noises_2, self.flow_length) if noises_2[0] is not None else None

        return fingerprints, ipds, delays, noises_1, noises_2

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


class FinnLossFn(nn.Module):

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


class FinnEncoder(nn.Module):
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


class FinnDecoder(nn.Module):
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


class FinnModel(nn.Module):

    def __init__(self, fingerprint_length: int, flow_length: int, no_encoding_noise: bool = False) -> None:
        """
        Args:
          fingerprint_length: int - The size of the fingerprint tensor.
          flow_length: int - The size of the flow tensor.
        """
        super().__init__()
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.no_encoding_noise = no_encoding_noise
        encoder_input_size = fingerprint_length if no_encoding_noise else fingerprint_length + flow_length
        self.encoder = FinnEncoder(encoder_input_size, flow_length)
        self.decoder = FinnDecoder(flow_length, fingerprint_length)

    def forward(self, fingerprint: Tensor, ipd: Tensor, noise_1: Tensor, noise_2: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        Args:
          fingerprint: Tensor - The fingerprint tensor.
          ipd: Tensor - The inter-packet delays tensor.
          noise_1: Tensor - The noise tensor to feed to the encoder.
          noise_2: Tensor - The noise tensor to add to the IPDs. If None, noise_1 is used.
        Returns:
          Tensor - The predicted fingerprint delay.
          Tensor - The predicted fingerprint.
        """
        noise_2 = noise_1.clone() if noise_2 is None else noise_2

        if fingerprint.dim() != 2 or fingerprint.shape[1] != self.fingerprint_length:
            raise ShapeError(fingerprint.shape, ("*", self.fingerprint_length))
        if ipd.dim() != 2 or ipd.shape[1] != self.flow_length:
            raise ShapeError(ipd.shape, ("*", self.flow_length))
        if noise_1.dim() != 2 or noise_1.shape[1] != self.flow_length:
            raise ShapeError(noise_1.shape, ("*", self.flow_length))
        if noise_2.dim() != 2 or noise_2.shape[1] != self.flow_length:
            raise ShapeError(noise_2.shape, ("*", self.flow_length))

        if self.no_encoding_noise:
            encoder_input = fingerprint
        else:
            encoder_input = torch.cat((fingerprint, noise_1), dim=1)

        delay_pred = self.encoder.forward(encoder_input)
        noisy_marked_ipd = self.combine_3(ipd, delay_pred, noise_2)
        fingerprint_pred = self.decoder(noisy_marked_ipd)

        return delay_pred, fingerprint_pred

    def combine_1(self, ipd: Tensor, delay: Tensor, noise: Tensor) -> Tensor:
        return ipd + delay + noise

    def combine_2(self, ipd: Tensor, delay: Tensor, noise: Tensor) -> Tensor:
        return ipd + torch.cumsum(delay, dim=1) + noise

    def combine_3(self, ipd: Tensor, delay: Tensor, noise: Tensor) -> Tensor:
        return ipd + torch.cumsum(delay, dim=1) + torch.cumsum(noise, dim=1)


class FinnTrainer(Trainer):

    model: FinnModel
    tr_dataset: FinnDataset
    vl_dataset: FinnDataset
    collate_fn: FinnCollateFn
    loss_fn: FinnLossFn
    tr_metric_keys = ("tr_loss", "tr_enc_loss", "tr_dec_loss", "tr_time",)

    def create_scheduler(self) -> None:
        return None

    def create_stopper(self) -> None:
        return None

    def forward(self, batch: tuple[Tensor, Tensor, Tensor, Optional[Tensor]]) -> tuple[Tensor, Tensor]:
        fingerprint: Tensor = batch[0].to(self.args.device)
        ipd: Tensor = batch[1].to(self.args.device)
        noise_1: Tensor = batch[3].to(self.args.device)
        noise_2: Optional[Tensor] = batch[4].to(self.args.device) if batch[4] is not None else None

        delay_pred, fingerprint_pred = self.model.forward(fingerprint, ipd, noise_1, noise_2)
        return delay_pred, fingerprint_pred

    def compute_loss(self, batch: tuple[Tensor, Tensor, Tensor, Optional[Tensor]], outputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, float]]:
        fingerprint: Tensor = batch[0].to(self.args.device)
        delay: Tensor = batch[2].to(self.args.device)
        delay_pred = outputs[0].to(self.args.device)
        fingerprint_pred = outputs[1].to(self.args.device)

        loss, enc_loss, dec_loss = self.loss_fn.forward(delay_pred, delay, fingerprint_pred, fingerprint)
        return loss, {"enc_loss": enc_loss.item(), "dec_loss": dec_loss.item()}

    def compute_metrics(self, batch: tuple[Tensor, Tensor, Tensor, Optional[Tensor]], outputs: tuple[Tensor, Tensor]) -> dict[str, float]:
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
    parser.add_argument("--dynamic", action="store_true", help=".")
    parser.add_argument("--use_same_dataset", action="store_true", help=".")
    parser.add_argument("--no_encoding_noise", action="store_true", help=".")
    parser.add_argument("--use_different_noises", action="store_true", help=".")
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

    DatasetConstructor = DynamicFinnDataset if args.dynamic else StaticFinnDataset
    DatasetConstructor = partial(
        DatasetConstructor,
        fingerprint_length=args.fingerprint_length,
        amplitude=args.amplitude,
        noise_deviation_low=args.noise_deviation_low,
        noise_deviation_high=args.noise_deviation_high,
        as_tensors=True,
        flow_length=args.flow_length,
        use_different_noises=args.use_different_noises,
    )
    tr_dataset = DatasetConstructor(tr_ipds)
    vl_dataset = DatasetConstructor(vl_ipds)
    gc.collect()

    print(f"Training Dataset:\n{tr_dataset}")
    print(f"Validation Dataset:\n{vl_dataset}")
    print("-" * 80)

    model = FinnModel(args.fingerprint_length, args.flow_length, args.no_encoding_noise)
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
    collate_fn = FinnCollateFn(args.fingerprint_length, args.flow_length, "max", truncate=True)
    loss_fn = FinnLossFn(encoder_weight=args.encoder_loss_weight, decoder_weight=args.decoder_loss_weight)
    trainer = FinnTrainer(trainer_args, model, tr_dataset, vl_dataset, collate_fn, loss_fn)

    trainer()

    print(f"ENDING {datetime.now().isoformat()}")
    print("-" * 80)


if __name__ == "__main__":
    main()
