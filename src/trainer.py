"""
Basic training utilities.

TODO:
 - Add support for better saving/loading of models.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from statistics import mean
import sys
import time
from typing import Callable, Optional, Self
import warnings

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
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
    lr_scheduler_patience: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1


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
        self.add_argument("--lr_scheduler_patience", type=int, default=None)
        self.add_argument("--early_stopping_patience", type=int, default=None)
        self.add_argument("--max_norm", type=float, default=1.0)
        self.add_argument("--gradient_accumulation_steps", type=int, default=1)


class EarlyStopper:

    def __init__(self, patience: int = 0, threshold: float = 0.0001, lower_is_worse: bool = False) -> None:
        self.patience = patience
        self.threshold = threshold
        self.lower_is_worse = lower_is_worse
        self.best = -sys.maxsize if lower_is_worse else sys.maxsize
        self.current = None
        self.count = 0

    def step(self, val: float) -> Self:
        self.current = val
        if self.lower_is_worse and (self.current > self.best + self.threshold):
            self.best = self.current
            self.count = 0
        elif not self.lower_is_worse and (self.current < self.best - self.threshold):
            self.best = self.current
            self.count = 0
        return self

    @property
    def stop(self) -> bool:
        if self.current == self.best:
            return False
        self.count += 1
        return self.count >= self.patience


class Trainer(ABC):
    """A generic Trainer class for training models.
    
    Note that this class requires DataLoader.drop_last_batch to be True, as the logic for averaging
    metrics assumes that the batch size is constant.
    """

    OVERWRITE_OUTDIRS = ("tmp", "test")
    tr_metric_keys = ("tr_loss", "tr_time")

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_dataset: Dataset | IterableDataset,
        vl_dataset: Dataset | IterableDataset,
        collate_fn: Callable,
        loss_fn: Module,
    ) -> None:
        self.args = args
        self.model: Module = model.to(args.device)
        self.tr_dataset = tr_dataset
        self.vl_dataset = vl_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.stopper = self.create_stopper()
        self.log = []
        self.best_epoch = -1
        self.best_metric = -sys.maxsize if args.lower_is_worse else sys.maxsize

    def create_optimizer(self) -> Optimizer:
        return AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def create_scheduler(self) -> Optional[LRScheduler]:
        # Patience varies between [0, 10] and will be one less than the stopper's patience.
        patience = self.args.lr_scheduler_patience
        if patience is None:
            patience = min(100000 // len(self.tr_dataset), 10)
        return ReduceLROnPlateau(self.optimizer, min_lr=1e-6, patience=patience)

    def create_stopper(self) -> Optional[EarlyStopper]:
        # Patience varies between [1, 11] and will be one greater than the scheduler's patience.
        patience = self.args.early_stopping_patience
        if patience is None:
            patience = min(100000 // len(self.tr_dataset) + 1, 11)
        return EarlyStopper(patience, threshold=1e-6, lower_is_worse=self.args.lower_is_worse)

    def __call__(self) -> Self:
        if self.args.outdir.exists():
            if self.args.outdir.name in self.OVERWRITE_OUTDIRS:
                shutil.rmtree(self.args.outdir)
            else:
                raise FileExistsError(f"Output Directory Already Exists: {self.args.outdir}")
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        tr_metrics = {k: float("nan") for k in self.tr_metric_keys}
        vl_metrics = self.evaluate()
        d = {"epoch": 0, "learning_rate": float("nan")} | tr_metrics | vl_metrics
        self.update_logs(d)
        self.update_best(d)
        self.update_save(d)

        pbar = self._get_pbar(range(1, self.args.epochs + 1), desc="Epochs")
        for epoch in pbar:
            tr_metrics = self.train()
            vl_metrics = self.evaluate()
            learning_rate = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.args.learning_rate
            d = {"epoch": epoch, "learning_rate": learning_rate} | tr_metrics | vl_metrics
            self.update_logs(d)
            self.update_best(d)
            self.update_save(d)
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(vl_metrics[self.args.metric], epoch=None)
                else:
                    self.scheduler.step(epoch=None)
            if self.stopper is not None:
                self.stopper.step(vl_metrics[self.args.metric])
                if self.stopper.stop:
                    break
            if any(math.isnan(d[m]) or math.isinf(d[m]) for m in ("tr_loss", "vl_loss")):
                raise ValueError(f"NaN/Inf Loss Detected!")

    def update_logs(self, results: dict[str, float]) -> None:
        self.log.append(results)
        print(self._fmt_dict(results))
        with open(self.args.outdir / "results.jsonl", "a") as fp:
            fp.write(json.dumps(results) + "\n")

    def update_best(self, results: dict[str, float]) -> None:
        if self.args.lower_is_worse and results[self.args.metric] > self.best_metric:
            self.best_epoch = results["epoch"]
            self.best_metric = results[self.args.metric]
        elif not self.args.lower_is_worse and results[self.args.metric] < self.best_metric:
            self.best_epoch = results["epoch"]
            self.best_metric = results[self.args.metric]

    def update_save(self, results: dict[str, float]) -> None:
        torch.save(self.model, self.args.outdir / f"model_{results['epoch']}.pth")
        checkpoints = sorted(self.args.outdir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        for checkpoint in checkpoints:
            e = int(checkpoint.stem.split("_")[1])
            if e not in (self.best_epoch, results["epoch"]):
                checkpoint.unlink()

    def train(self) -> dict[str, float]:
        t_0 = time.time()

        self.model.train()
        dataloader = self.get_tr_dataloader()
        num_steps = math.ceil(len(dataloader) / self.args.gradient_accumulation_steps)
        results: defaultdict[str, list[float]] = defaultdict(lambda: [0] * num_steps)
        step = 0

        pbar = self._get_pbar(dataloader, total=len(dataloader))
        for mini_step, batch in enumerate(pbar):

            # Compute normalized loss, skip noisy losses
            outputs = self.forward(batch)
            loss, losses = self.compute_loss(batch, outputs)
            loss = loss / self.args.gradient_accumulation_steps
            losses = {k: v / self.args.gradient_accumulation_steps for k, v in losses.items()}
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                warnings.warn(f"NaN/Inf Loss Detected! {mini_step=} loss={loss.item()}")
                continue  # TODO: could theoretically cause the weights to never step
            loss.backward()

            # Add to running metrics
            results["tr_loss"][step] += loss.item()
            for k, v in losses.items():
                results[f"tr_{k}"][step] += v

            # Update weights every `gradient_accumulation_steps` `mini_steps`
            condition_1 = (mini_step + 1) % self.args.gradient_accumulation_steps == 0
            condition_2 = (mini_step + 1) == len(dataloader)
            if condition_1 or condition_2:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.zero_grad()
                self.optimizer.step()
                step += 1

            # Perform logging every `logging_steps` `steps`
            condition_1 = self.args.logging_steps > 0
            condition_2 = (step + 1) % self.args.logging_steps == 0
            if condition_1 and condition_2:
                d = {"step": step}
                for k, v in results.items():
                    d[f"_{k}"] = mean(results[k][-self.args.logging_steps:])
                print(self._fmt_dict(d))

        # Average statistics over epoch
        for k, v in results.items():
            results[k] = mean(v)
        results["tr_time"] = time.time() - t_0

        return dict(results)

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()

        results: defaultdict[str, list[float]] = defaultdict(list)
        self.model.eval()
        dataloader = self.get_vl_dataloader()
        pbar = self._get_pbar(dataloader, total=len(self.vl_dataset) // self.args.vl_batch_size)
        with torch.no_grad():
            for step, batch in enumerate(pbar):  # pylint: disable=unused-variable

                outputs = self.forward(batch)
                loss, losses = self.compute_loss(batch, outputs)
                metrics = self.compute_metrics(batch, outputs)

                results["vl_loss"].append(loss.item())
                for k, v in losses.items():
                    results[f"vl_{k}"].append(v)
                for k, v in metrics.items():
                    results[f"vl_{k}"].append(v)

        for k, v in results.items():
            results[k] = mean(v)
        results["vl_time"] = time.time() - t_0

        return dict(results)

    @abstractmethod
    def forward(self, batch: tuple) -> tuple:
        """Send a batch of inputs forward through the model.

        Args:
            batch (tuple): batch of inputs.

        Returns:
            tuple: model output(s), e.g., logits.
        """

    @abstractmethod
    def compute_loss(self, batch: tuple, outputs: tuple) -> tuple[Tensor, dict[str, float]]:
        """Compute the loss over a batch of examples.

        Args:
            batch (tuple): batch of inputs.
            outputs (tuple): model output(s), e.g., logits, as return by self.forward.

        Returns:
            Tensor: loss over the batch.
            dict[str, float]: auxillary losses over the batch.
        """

    def compute_metrics(self, batch: tuple, outputs: tuple) -> dict[str, float]:
        """Compute the validation metrics over a batch of examples.

        Args:
            batch (tuple): batch of inputs.
            outputs (tuple): model output(s), e.g., logits, as return by self.forward.
            loss (Tensor): model loss over the batch.

        Returns:
            dict[str, float]: metrics for the batch.
        """
        return {}

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
