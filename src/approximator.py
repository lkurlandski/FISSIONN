"""
A network to approximate the noising process of IPDs through SSI chains.
"""

from __future__ import annotations
from itertools import chain, combinations
import os
from pprint import pformat
import sys
from typing import Optional

from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_data
from src.trainer import TrainerArgs, Trainer, TrainerArgumentParser
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
        collate_fn = ApproximatorCollateFn(max_length=256)
        loss_fn = ApproximatorLossFn(pad_to_same_length=True)
        dataloader = DataLoader(dataset, batch_size=1024, collate_fn=collate_fn)
        pbar = tqdm(dataloader, desc="Computing Basal Loss...")
        loss = sum(loss_fn.forward(y, x).item() for x, y in pbar)
        return loss / len(dataset)


class ApproximatorCollateFn:

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
        self.loss_fn = nn.MSELoss()

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


class ApproximatorTrainer(Trainer):

    model: RecurrentApproximator | TransformerApproximator
    tr_dataset: ApproximatorDataset
    vl_dataset: ApproximatorDataset
    collate_fn: ApproximatorCollateFn
    loss_fn: ApproximatorLossFn

    def train_one_batch(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:

        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)

        self.model.zero_grad()
        y_pred = self.model.forward(x, y)
        loss: Tensor = self.loss_fn.forward(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return {"tr_loss": loss.item()}

    def evaluate_one_batch(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)

        y_pred = self.model.forward(x, y)
        loss = self.loss_fn.forward(y_pred, y)

        return {"vl_loss": loss.item()}


def main() -> None:

    parser = TrainerArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help=".")
    args = parser.parse_args()

    print(f"Command Line Arguments:\n{pformat(args.__dict__)}")
    print("-" * 80)

    if args.outdir.exists() and args.outdir.name not in Trainer.OVERWRITE_OUTDIRS:
        raise FileExistsError(f"Output Directory Already Exists: {args.outdir}")

    seed_everything(args.seed)

    ipd_groups = []
    for group in load_data():
        ipd_groups.append([ipds.tolist() for ipds in group])
    print(f"Collected {sum(len(group) for group in ipd_groups)} IPDs from {len(ipd_groups)} groups.")
    print("-" * 80)

    tr_ipd_groups, vl_ipd_groups = train_test_split(ipd_groups, test_size=0.15)
    tr_dataset = ApproximatorDataset(tr_ipd_groups)
    vl_dataset = ApproximatorDataset(vl_ipd_groups)

    # tr_dataset = Subset(tr_dataset, range(4096))
    # vl_dataset = Subset(vl_dataset, range(4096))

    # loss = ApproximatorDataset.compute_basal_loss(ApproximatorDataset(ipd_groups))
    # print(f"Dataset Basal Loss: {loss}")

    print(f"Training Dataset: {tr_dataset}")
    print(f"Validation Dataset: {vl_dataset}")
    print("-" * 80)

    model = TransformerApproximator(
        hidden_size=128,
        num_layers=3,
        nhead=2,
        intermediate_size=512,
    )

    # model = RecurrentApproximator(
    #     hidden_size=256,
    #     num_layers=6,
    #     bidirectional=True,
    #     cell=nn.GRU,
    # )

    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    collate_fn = ApproximatorCollateFn(max_length=256)
    loss_fn = ApproximatorLossFn()
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
