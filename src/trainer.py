"""
Basic training utilities.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from statistics import mean
import sys
import time
from typing import Callable, Optional, Self

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer, AdamW
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm


@dataclass
class TrainerArgs:
    outdir: Path = Path("./output/tmp")
    device: torch.cuda.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = 1
    tr_batch_size: int = 2
    vl_batch_size: int = 2
    learning_rate: float = 1e-3
    num_workers: int = 0
    disable_tqdm: bool = False
    logging_steps: int = -1
    metric: str = "vl_loss"
    lower_is_worse: bool = False


class TrainerArgumentParser(ArgumentParser):

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.add_argument("--outdir", type=Path, default=Path("./output/tmp"))
        self.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.add_argument("--epochs", type=int, default=1)
        self.add_argument("--tr_batch_size", type=int, default=2)
        self.add_argument("--vl_batch_size", type=int, default=2)
        self.add_argument("--learning_rate", type=float, default=1e-3)
        self.add_argument("--num_workers", type=int, default=0)
        self.add_argument("--disable_tqdm", action="store_true")
        self.add_argument("--logging_steps", type=int, default=-1)
        self.add_argument("--metric", type=str, default="vl_loss")
        self.add_argument("--lower_is_worse", action="store_true")


class Trainer(ABC):
    """A generic Trainer class for training models.
    
    Note that this class requires DataLoader.drop_last_batch to be True, as the logic for averaging
    metrics assumes that the batch size is constant.
    """

    OVERWRITE_OUTDIRS = ("tmp", "test")
    tr_metric_keys = ("tr_loss", "tr_time")
    vl_metric_keys = ("vl_loss", "vl_time")
    other_keys = ("epoch",)

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_dataset: Dataset | IterableDataset,
        vl_dataset: Dataset | IterableDataset,
        collate_fn: Callable,
        loss_fn: Module,
        optimizer_fn: Optional[Callable[[Iterable[Tensor]], Optimizer]] = None,
    ) -> None:
        self.args = args
        self.model = model
        self.tr_dataset = tr_dataset
        self.vl_dataset = vl_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        if optimizer_fn:
            self.optimizer = optimizer_fn(self.model.parameters())
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
        self.log = []
        self.best = (-1, sys.maxsize)

    def __call__(self) -> Self:
        if self.args.outdir.exists():
            if self.args.outdir.name in self.OVERWRITE_OUTDIRS:
                shutil.rmtree(self.args.outdir)
            else:
                raise FileExistsError(f"Output Directory Already Exists: {self.args.outdir}")
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.args.device)

        tr_metrics = {k: float("nan") for k in self.tr_metric_keys}
        vl_metrics = self.evaluate()
        d = {"epoch": 0} | tr_metrics | vl_metrics
        self.on_epoch_end(d, 0)

        pbar = self._get_pbar(range(1, self.args.epochs + 1), desc="Epochs")
        for epoch in pbar:
            tr_metrics = self.train()
            vl_metrics = self.evaluate()
            d = {"epoch": epoch} | tr_metrics | vl_metrics
            self.on_epoch_end(d, 0)

    def on_epoch_end(self, results: dict[str, float], epoch: int):
        self.log.append(results)
        print(self._fmt_dict(results))
        with open(self.args.outdir / "results.jsonl", "a") as fp:
            fp.write(json.dumps(results) + "\n")
        torch.save(self.model, self.args.outdir / f"model_{epoch}.pth")

        if self.args.lower_is_worse and results[self.args.metric] > self.best[1]:
            self.best = (epoch, results[self.args.metric])
        elif not self.args.lower_is_worse and results[self.args.metric] < self.best[1]:
            self.best = (epoch, results[self.args.metric])

        checkpoints = sorted(self.args.outdir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        for checkpoint in checkpoints:
            e = int(checkpoint.stem.split("_")[1])
            if e not in (self.best[0], epoch):
                checkpoint.unlink()

    def train(self) -> dict[str, float]:
        t_0 = time.time()

        # Collect the results of each metric for each batch
        results: defaultdict[str, list[float]] = defaultdict(list)

        self.model.train()
        dataloader = self.get_tr_dataloader()
        pbar = self._get_pbar(dataloader, total=len(self.tr_dataset) // self.args.tr_batch_size)
        for step, batch in enumerate(pbar):
            # Get the average of each metric for the batch.
            result = self.train_one_batch(batch)
            for k, v in result.items():
                r = mean(v) if isinstance(v, list) else v
                results[k].append(r)

            # Print the average of the most recent `logging_steps`
            if self.args.logging_steps > 0 and step % self.args.logging_steps == 0:
                d = {}
                for k, v in results.items():
                    v = v[-self.args.logging_steps:]
                    d[f"_{k}"] = mean(v)
                print(self._fmt_dict(d))

        # Get the average of each metric for the entire training set
        for k, v in results.items():
            results[k] = mean(v)
        results["tr_time"] = time.time() - t_0

        return dict(results)

    @abstractmethod
    def train_one_batch(self, batch: tuple) -> dict[str, float | list[float]]:
        """Update the model on a batch of examples.

        Args:
            batch (tuple): batch of inputs.

        Returns:
            dict[str, float | list[float]]: dictionary of metrics, which may be a single value,
                representing an average over the batch, or a list of values, representing the
                value of the metric for each sample (will be averaged by self.train).
        """

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()

        # Collect the results of each metric for each batch
        results: defaultdict[str, list[float]] = defaultdict(list)

        self.model.eval()
        dataloader = self.get_vl_dataloader()
        pbar = self._get_pbar(dataloader, total=len(self.vl_dataset) // self.args.vl_batch_size)
        with torch.no_grad():
            for step, batch in enumerate(pbar):  # pylint: disable=unused-variable
                # Get the average of each metric for the batch.
                result = self.evaluate_one_batch(batch)
                for k, v in result.items():
                    r = mean(v) if isinstance(v, list) else v
                    results[k].append(r)

        # Get the average of each metric for the entire validation set
        for k, v in results.items():
            results[k] = mean(v)
        results["vl_time"] = time.time() - t_0

        return dict(results)

    @abstractmethod
    def evaluate_one_batch(self, batch: tuple) -> dict[str, float | list[float]]:
        """Evaluate the model on a batch of examples.

        Args:
            batch (tuple): batch of inputs.

        Returns:
            dict[str, float | list[float]]: dictionary of metrics, which may be a single value,
                representing an average over the batch, or a list of values, representing the
                value of the metric for each sample (will be averaged by self.train).
        """

    def get_dataloader(self, dataset: Dataset | IterableDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def get_tr_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.tr_dataset, self.args.tr_batch_size, True)

    def get_vl_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.vl_dataset, self.args.vl_batch_size, False)

    def _get_pbar(self, iterable: Iterable, **kwds) -> tqdm | Iterable:
        if self.args.disable_tqdm:
            return iterable
        return tqdm(iterable, **kwds)

    def _fmt_dict(self, d: dict[str, float]) -> dict[str, str]:
        return {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in d.items()}
