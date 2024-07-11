"""
A network to approximate the noising process of IPDs through SSI chains.
"""

from __future__ import annotations
from itertools import chain, combinations
import math
import os
from pprint import pformat
from statistics import mean
import sys
from typing import Literal, Optional

from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor, BoolTensor
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


PAD = -10000.0
BOS = -10001.0
EOS = -10002.0


def bos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, BOS)


def eos(shape: tuple[int]) -> Tensor:
    return torch.full(shape, EOS)


class ApproximatorDataset(Dataset):

    def __init__(self, ipd_groups: list[list[list[int]]]) -> None:
        self.ipd_pairs = chain.from_iterable(combinations(ipd_group, 2) for ipd_group in ipd_groups)
        self.ipd_pairs: list[tuple[list[int], list[int]]] = list(self.ipd_pairs)

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


def compute_basal_loss(
    ipd_groups: list[list[list[int]]],
    max_length: int,
    device: Optional[torch.cuda.device] = None,
    batch_size: int = 4096,
) -> float:
    """
    ~13.5M for max_length==256.
    """
    dataset = ApproximatorDataset(ipd_groups)
    collate_fn = ApproximatorCollateFn(max_length)
    loss_fn = ApproximatorLossFn()
    dataloader = DataLoader(dataset, batch_size, False, collate_fn=collate_fn, drop_last=True)
    pbar = tqdm(dataloader, desc="Computing Basal Loss...")
    losses = []
    for step, batch in enumerate(pbar):  # pylint: disable=unused-variable
        x = batch[0].to(device)
        y = batch[1].to(device)
        loss = loss_fn.forward(x[:, 1:], y[:, 1:]).item()
        losses.append(loss)
    return mean(losses)


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

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.loss_fn.forward(y_pred, y_true)


class ApproximatorDecoder:

    def __init__(self) -> None:
        ...


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
        if x.dim() != 2:
            raise ShapeError((x.shape), ("B", "L"))

        x = x.unsqueeze(2)
        src = self.projection_in.forward(x)
        mem = self.encoder.forward(src)[0]
        mem = self.projection_mid.forward(mem)
        out = self.decoder.forward(mem)[0]
        y_pred = self.projection_out.forward(out).squeeze(2)
        return y_pred


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

    def beam_search(self, x: Tensor, max_len: int, beam_size: int) -> Tensor:
        raise NotImplementedError("Beam Search Not Yet Implemented.")

    def greedy_decode(self, x: Tensor, max_length: int) -> Tensor:
        if self.training:
            raise RuntimeError("Call `eval()` before using model for inference.")
        if torch.is_grad_enabled():
            raise RuntimeError("Disable gradient computation before using model for inference.")
        if x.dim() != 2:
            raise ShapeError((x.shape), ("B", "L"))

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

            y_pred = self.projection_out(out).squeeze(2)
            y_next = y_pred[:, -1]
            y = torch.cat([y, y_next.unsqueeze(1)], dim=1)

            generated_eos = generated_eos | (y_next == EOS)
            if generated_eos.all():
                break

        # Add EOS to the sequences which have not generated EOS yet. Else add PAD.
        final = eos((B,)).to(D)
        final[~generated_eos] = PAD
        y = torch.cat([y, final.unsqueeze(1)], dim=1)

        return y

    def translate(self, x: Tensor, max_length: int, alg: Literal["greedy", "beam"]) -> Tensor:
        self.eval()
        with torch.no_grad():
            if alg == "greedy":
                y = self.greedy_decode(x, max_length)
            elif alg == "beam":
                y = self.beam_search(x, max_length)
            else:
                raise ValueError(f"Invalid Algorithm: {alg}")
        return y


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
        y_pred = self.model.forward(x, y[:, :-1])
        y_pred = y_pred[:, 1:] if y_pred.size(1) == y.size(1) else y_pred  # trim for recurrent models
        loss: Tensor = self.loss_fn.forward(y_pred, y[:, 1:])
        loss.backward()
        self.optimizer.step()

        return {"tr_loss": loss.item()}

    def evaluate_one_batch(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        x: Tensor = batch[0].to(self.args.device)
        y: Tensor = batch[1].to(self.args.device)

        y_pred = self.model.forward(x, y[:, :-1])
        y_pred = y_pred[:, 1:] if y_pred.size(1) == y.size(1) else y_pred  # trim for recurrent models
        loss = self.loss_fn.forward(y_pred, y[:, 1:])

        return {"vl_loss": loss.item()}


def main() -> None:

    MAX_LENGTH = 256

    parser = TrainerArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help=".")
    parser.add_argument("--tr_num_samples", type=int, default=sys.maxsize, help=".")
    parser.add_argument("--vl_num_samples", type=int, default=sys.maxsize, help=".")
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
    tr_dataset = Subset(tr_dataset, range(min(args.tr_num_samples, len(tr_dataset))))
    vl_dataset = ApproximatorDataset(vl_ipd_groups)
    vl_dataset = Subset(vl_dataset, range(min(args.vl_num_samples, len(vl_dataset))))

    # print(f"tr_basal_loss={compute_basal_loss(tr_ipd_groups, MAX_LENGTH, args.device, 4096)}")
    # print(f"vl_basal_loss={compute_basal_loss(vl_ipd_groups, MAX_LENGTH, args.device, 4096)}")

    print(f"Training Dataset: {tr_dataset}")
    print(f"Validation Dataset: {vl_dataset}")
    print("-" * 80)

    model = TransformerApproximator(
        max_length=MAX_LENGTH,
        hidden_size=256,
        num_layers=4,
        nhead=4,
        intermediate_size=1024,
    )

    # model = RecurrentApproximator(
    #     hidden_size=64,
    #     num_layers=2,
    #     bidirectional=True,
    #     cell=nn.GRU,
    # )

    print(f"Model:\n{model}")
    print(f"Total Parameters: {round(count_parameters(model) / 1e6, 2)}M")
    print(f"Encoder Parameters: {round(count_parameters(model.encoder) / 1e6, 2)}M")
    print(f"Decoder Parameters: {round(count_parameters(model.decoder) / 1e6, 2)}M")
    print("-" * 80)

    collate_fn = ApproximatorCollateFn(max_length=MAX_LENGTH)
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
