"""

TODO:
 - The train and test split should be formed from distinct groups!
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections.abc import Iterable
from itertools import chain, combinations
import json
import os
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys
import time
from typing import Optional

from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor
from torch.optim import Optimizer, AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data
from src.finn import ShapeError
from src.utils import (
    count_parameters,
    seed_everything,
    ShapeError,
)
# pylint: enable=wrong-import-position


def bos(shape: tuple[int]) -> Tensor:
    return torch.zeros(shape, dtype=torch.float32)


class ApproximatorDataset(Dataset):

    def __init__(self, ipd_groups: list[list[list[int]]]) -> None:
        self.ipd_pairs = chain.from_iterable(combinations(ipd_group, 2) for ipd_group in ipd_groups)
        self.ipd_pairs: list[tuple[list[int], list[int]]] = list(self.ipd_pairs)
        self._basal_loss = None

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, y = self.ipd_pairs[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

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
    def compute_basal_loss(dataset: ApproximatorDataset) -> float:
        collate_fn = CollateFn(max_length=256)
        loss_fn = ApproximatorLossFn(pad_to_same_length=True)
        dataloader = DataLoader(dataset, batch_size=1024, collate_fn=collate_fn)
        pbar = tqdm(dataloader, desc="Computing Basal Loss...")
        loss = sum(loss_fn.forward(y, x).item() for x, y in pbar)
        return loss / len(dataset)


class CollateFn:

    def __init__(
        self,
        max_length: int = sys.maxsize,
        pad_value: int = 0,
    ) -> None:
        self.max_length = max_length
        self.pad_value = pad_value

    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        x, y = [], []
        for x_, y_ in batch:
            x.append(torch.cat([bos((1,)), x_])[0:self.max_length])
            y.append(torch.cat([bos((1,)), y_])[0:self.max_length])
        x = pad_sequence(x, batch_first=True, padding_value=self.pad_value)
        y = pad_sequence(y, batch_first=True, padding_value=self.pad_value)
        return x, y


class ApproximatorLossFn(nn.Module):

    def __init__(self, pad_to_same_length: bool = False, trim_to_same_length: bool = False) -> None:
        super().__init__()
        self.pad_to_same_length = pad_to_same_length
        self.trim_to_same_length = trim_to_same_length
        self.loss_fn = nn.MSELoss(reduction="sum")

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if y_pred.size(1) != y_true.size(1):
            if self.pad_to_same_length:
                length = max(y_pred.size(1), y_true.size(1))
                y_pred = nn.functional.pad(y_pred, (0, length - y_pred.size(1)), value=0)
                y_true = nn.functional.pad(y_true, (0, length - y_true.size(1)), value=0)
            if self.trim_to_same_length:
                length = min(y_pred.size(1), y_true.size(1))
                y_pred = y_pred[:, :length]
                y_true = y_true[:, :length]

        return self.loss_fn.forward(y_pred, y_true)


class RecurrentApproximator(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        input_size: Optional[int] = None,
        bidirectional: bool = False,
        cell: nn.RNN | nn.LSTM | nn.GRU = nn.RNN,
    ) -> None:
        super().__init__()
        input_size = input_size if input_size is not None else hidden_size
        self.projection_in = nn.Linear(1, input_size)
        self.encoder: nn.RNN | nn.LSTM | nn.GRU = cell(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
        )
        self.projection_mid = nn.Linear(2 * hidden_size if bidirectional else hidden_size, hidden_size)
        self.decoder: nn.RNN | nn.LSTM | nn.GRU = cell(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
        )
        self.projection_out = nn.Linear(2 * hidden_size if bidirectional else hidden_size, 1)

    def forward(self, x: Tensor, _: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Batch of IPDs before entering a stepping stone sequence. 
                Shape: (B, L). The IPDs should start with a BOS `token`, i.e., 0.0.

        Returns:
            Tensor: Batch of IPDs after entering a stepping stone sequence (simulated).
                Shape: (B, L).
        """
        if x.dim() != 2:
            raise ShapeError((x.shape), ("B", "L"))

        x = x.unsqueeze(2)
        src = self.projection_in.forward(x)
        mem = self.encoder.forward(src)[0]
        mem = self.projection_mid.forward(mem)
        out = self.decoder.forward(mem)[0]
        y_pred = self.projection_out.forward(out).squeeze(2)
        return y_pred


class TransformerApproximator(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        nhead: int,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.projection_in = nn.Linear(1, hidden_size)
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
        """
        Args:
            x (Tensor): Batch of IPDs before entering a stepping stone sequence. 
                Shape: (B, L). The IPDs should start with a BOS `token`, i.e., 0.0.
            y (Tensor): Batch of IPDs after entering a stepping stone sequence.
                Shape: (B, L). These IPDs should start with a BOS `token`, i.e., 0.0.
            mem (Tensor): Memory from the encoder. If not provided, the encoder will be called.
                Shape: (B, L, H).

        Returns:
            Tensor: Batch of IPDs after entering a stepping stone sequence (simulated).
                Shape: (B, L - 1). These IPDs do not start with the BOS `token`.
        """
        if x is not None:
            if x.dim() != 2:
                raise ShapeError((x.shape), ("B", "L"))
            x = x.unsqueeze(2)
        if y is not None:
            if y.dim() != 2:
                raise ShapeError((y.shape), ("B", "L"))
            y = y.unsqueeze(2)
        if mem is not None:
            if mem.dim() != 3:
                raise ShapeError((mem.shape), ("B", "L", "H"))

        if mem is None:
            src = self.projection_in.forward(x)
            mem = self.encoder.forward(src)

        tgt = self.projection_in.forward(y)
        out = self.decoder.forward(tgt[:, :-1, :], mem)
        out = torch.cat([bos((out.size(0), 1, out.size(2))).to(out.device), out], dim=1)

        y_pred = self.projection_out.forward(out).squeeze(2)
        return y_pred


class TrainerArgs:
    def __init__(self, device: torch.device, outdir: Path, num_train_epochs: int, batch_size: int, logging_steps: int, disable_tqdm: bool, dataloader_num_workers: int):
        self.device = device
        self.outdir = outdir
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.logging_steps = logging_steps
        self.disable_tqdm = disable_tqdm
        self.dataloader_num_workers = dataloader_num_workers


class Trainer:
    def __init__(
        self,
        args: TrainerArgs,
        model: TransformerApproximator | RecurrentApproximator,
        tr_dataset: Dataset,
        vl_dataset: Dataset,
        collate_fn: CollateFn,
        loss_fn: ApproximatorLossFn,
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
        if self.args.outdir.exists():
            raise FileExistsError(f"Output Directory Already Exists: {self.args.outdir}")
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        tr_metrics = {"tr_loss": float("nan")}
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
            with open(self.args.outdir / "results.jsonl", "a") as fp:
                fp.write(json.dumps(d) + "\n")

    def _get_pbar(self, iterable: Iterable, **kwds) -> tqdm | Iterable:
        if self.args.disable_tqdm:
            return iterable
        return tqdm(iterable, **kwds)

    def _fmt_dict(self, d: dict[str, float]) -> dict[str, str]:
        return {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in d.items()}

    def train(self) -> dict[str, float]:
        t_0 = time.time()
        self.model.train()
        cum_samples, cum_loss = 0, 0
        log_samples, log_loss = 0, 0
        dataloader = self.get_dataloader(self.tr_dataset, shuffle=True)
        pbar = self._get_pbar(dataloader, total=len(self.tr_dataset) // self.args.batch_size)
        for i, batch in enumerate(pbar):
            x: Tensor = batch[0].to(self.args.device)
            y: Tensor = batch[1].to(self.args.device)

            self.model.zero_grad()
            y_pred = self.model.forward(x, y)
            loss = self.loss_fn.forward(y_pred, y)
            loss.backward()
            self.optimizer.step()

            cum_samples += x.size(0)
            cum_loss += loss.item()

            log_samples += x.size(0)
            log_loss += loss.item()

            if self.args.logging_steps > 0 and i % self.args.logging_steps == 0:
                d = {
                    "log_loss": log_loss / log_samples,
                }
                d = {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in d.items()}
                log_samples, log_loss = 0, 0
                print(d)

        d = {
            "tr_loss": cum_loss / cum_samples,
            "tr_time": time.time() - t_0,
        }

        return d

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()
        self.model.eval()
        cum_samples, cum_loss = 0, 0
        dataloader = self.get_dataloader(self.vl_dataset, shuffle=False)
        pbar = self._get_pbar(dataloader, total=len(self.vl_dataset) // self.args.batch_size)
        with torch.no_grad():
            for batch in pbar:
                x: Tensor = batch[0].to(self.args.device)
                y: Tensor = batch[1].to(self.args.device)

                y_pred = self.model.forward(x, y)
                loss = self.loss_fn.forward(y_pred, y)

                cum_samples += x.size(0)
                cum_loss += loss.item()

        d = {
            "vl_loss": cum_loss / cum_samples,
            "vl_time": time.time() - t_0,
        }

        return d

    def get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            self.args.batch_size,
            shuffle,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--no_split_groups", action="store_true")
    parser.add_argument("--tr_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--vl_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--compute_basal_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help=".")
    parser.add_argument("--outdir", type=Path, default=Path("./output/tmp.jsonl"), help=".")
    args = parser.parse_args()

    print(f"Command Line Arguments:\n{pformat(args.__dict__)}")
    print("-" * 80)

    if args.outdir.exists():
        raise FileExistsError(f"Output Directory Already Exists: {args.outdir}")

    seed_everything(args.seed)

    ipd_groups = []
    for group in load_data():
        ipd_groups.append([ipds.tolist() for ipds in group])
    print(f"Collected {sum(len(group) for group in ipd_groups)} IPDs from {len(ipd_groups)} groups.")
    print("-" * 80)

    if args.no_split_groups:
        dataset = ApproximatorDataset(ipd_groups)
        tr_dataset, vl_dataset = random_split(dataset, [0.85, 0.15])
    else:
        tr_ipd_groups, vl_ipd_groups = train_test_split(ipd_groups, test_size=0.15)
        tr_dataset = ApproximatorDataset(tr_ipd_groups)
        vl_dataset = ApproximatorDataset(vl_ipd_groups)

    if args.tr_num_samples < len(tr_dataset):
        tr_dataset = Subset(tr_dataset, list(range(args.tr_num_samples)))
    if args.vl_num_samples < len(vl_dataset):
        vl_dataset = Subset(vl_dataset, list(range(args.vl_num_samples)))

    if args.compute_basal_loss:
        loss = ApproximatorDataset.compute_basal_loss(ApproximatorDataset(ipd_groups))
        print(f"Dataset Basal Loss: {loss}")

    print(f"Training Dataset: {tr_dataset}")
    print(f"Validation Dataset: {vl_dataset}")
    print("-" * 80)

    print()

    # model = TransformerApproximator(
    #     hidden_size=256,
    #     num_layers=6,
    #     nhead=4,
    #     intermediate_size=1024,
    # )

    model = RecurrentApproximator(
        hidden_size=256,
        num_layers=6,
        bidirectional=True,
        cell=nn.GRU,
    )

    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    collate_fn = CollateFn(max_length=256)
    loss_fn = ApproximatorLossFn()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    training_arguments = TrainerArgs(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        outdir=args.outdir,
        num_train_epochs=1000,
        batch_size=256,
        logging_steps=-1,
        disable_tqdm=False,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        training_arguments,
        model,
        tr_dataset,
        vl_dataset,
        collate_fn,
        loss_fn,
        optimizer,
    )

    trainer()


if __name__ == "__main__":
    main()
