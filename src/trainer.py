"""
Basic training utilities.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
import gc
import inspect
import json
import math
from pathlib import Path
import shutil
from statistics import mean
import sys
import time
from typing import Any, Callable, Optional, Self  # pylint: disable=no-name-in-module
import warnings

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler, SequentialLR, ConstantLR, LinearLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, IterableDataset, Subset
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
    pin_memory: bool = True
    disable_tqdm: bool = False
    logging_steps: int = -1
    silent: bool = False
    metric: str = "vl_loss"
    lower_is_worse: bool = False
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    find_executable_batch_size: bool = False
    teacher_ratio_start: float = 1.0
    teacher_ratio_end: float = 1.0

    def __post_init__(self) -> None:
        if self.silent:
            self.disable_tqdm = True
            self.logging_steps = -1


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
        self.add_argument("--pin_memory", action="store_true")
        self.add_argument("--disable_tqdm", action="store_true")
        self.add_argument("--logging_steps", type=int, default=-1)
        self.add_argument("--metric", type=str, default="vl_loss")
        self.add_argument("--lower_is_worse", action="store_true")
        self.add_argument("--max_norm", type=float, default=1.0)
        self.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.add_argument("--find_executable_batch_size", action="store_true")
        self.add_argument("--teacher_ratio_start", type=float, default=1.0)
        self.add_argument("--teacher_ratio_end", type=float, default=1.0)


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


class TeacherRatioScheduler:

    def __init__(self, epochs: int, start: float = 1.0, end: float = 0.0) -> None:
        self.epochs = epochs
        self.start = start
        self.end = end
        self.ratios = torch.linspace(start, end, epochs).tolist()
        self.idx = 0

    def step(self):
        self.idx += 1

    @property
    def ratio(self) -> float:
        return self.ratios[self.idx]


def get_lr_scheduler(optimizer: Optimizer, epochs: int) -> LRScheduler:
    """
    Creates a pre-configured learning rate scheduler based on the number of epochs.
    """

    if epochs == 10:
        return SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.01, total_iters=2),
                ConstantLR(optimizer, factor=1.0, total_iters=3),
                ExponentialLR(optimizer, gamma=0.60),
            ],
            [2, 5],
        )

    if epochs == 20:
        return SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.01, total_iters=5),
                ConstantLR(optimizer, factor=1.0, total_iters=5),
                ExponentialLR(optimizer, gamma=0.60),
            ],
            [5, 10],
        )

    raise ValueError(f"No scheduler configured for the given number of epochs: {epochs}.")


def find_executable_batch_size(
    function: Optional[Callable] = None,
    starting_batch_size: int = -1,
    starting_gradient_accumulation_steps: int = 1,
) -> None:
    """Rerun a function with a smaller mini batch size if an OOM error is encountered.
    """

    if function is None:
        return partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            starting_gradient_accumulation_steps=starting_gradient_accumulation_steps,
        )


    batch_size = starting_batch_size
    gradient_accumulation_steps = starting_gradient_accumulation_steps


    def should_reduce_batch_size(exception: Exception) -> bool:
        statements = [
            "CUDA out of memory.",
            "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
            "DefaultCPUAllocator: can't allocate memory",
            "CUDA error: an illegal memory access was encountered",
            "Triton Error [CUDA]: an illegal memory access was encountered",
            "CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
        ]
        if isinstance(exception, RuntimeError) and len(exception.args) == 1:
            return any(err in exception.args[0] for err in statements)
        return False


    def decorator(*args, **kwargs):
        nonlocal batch_size, gradient_accumulation_steps

        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())

        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")

            try:
                return function(batch_size, gradient_accumulation_steps, *args, **kwargs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    gradient_accumulation_steps *= 2
                else:
                    raise

    return decorator


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
        self.scheduler = self.create_scheduler()                              # pylint: disable=assignment-from-none
        self.stopper = self.create_stopper()                                  # pylint: disable=assignment-from-none
        self.teacher_ratio_scheduler = self.create_teacher_ratio_scheduler()  # pylint: disable=assignment-from-none
        self.log = []
        self.best_epoch = -1
        self.best_metric = -sys.maxsize if args.lower_is_worse else sys.maxsize

    def create_optimizer(self) -> Optimizer:
        return AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def create_scheduler(self) -> Optional[LRScheduler]:
        return None

    def create_stopper(self) -> Optional[EarlyStopper]:
        return None

    def create_teacher_ratio_scheduler(self) -> Optional[TeacherRatioScheduler]:
        return None

    def __call__(self) -> Self:
        shutil.rmtree(self.args.outdir, ignore_errors=True)
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        @find_executable_batch_size(
            starting_batch_size=self.args.vl_batch_size,
            starting_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )
        def evaluate(batch_size: int, gradient_accumulation_steps: int) -> dict[str, float]:  # pylint: disable=unused-argument
            nonlocal self
            self.args.vl_batch_size = batch_size
            return self.evaluate()

        @find_executable_batch_size(
            starting_batch_size=self.args.vl_batch_size,
            starting_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )
        def train(batch_size: int, gradient_accumulation_steps: int) -> dict[str, float]:
            nonlocal self
            self.args.tr_batch_size = batch_size
            self.args.gradient_accumulation_steps = gradient_accumulation_steps
            return self.train()

        if not self.args.find_executable_batch_size:
            evaluate = self.evaluate
            train = self.train

        tr_metrics = {k: float("nan") for k in self.tr_metric_keys}
        vl_metrics = evaluate()
        teacher_ratio = None if self.teacher_ratio_scheduler is None else self.teacher_ratio_scheduler.ratio
        d = {"epoch": 0, "learning_rate": float("nan"), "teacher_ratio": float("nan")} | tr_metrics | vl_metrics
        self.update_logs(d)
        self.update_best(d)
        self.update_save(d)

        pbar = self._get_pbar(range(1, self.args.epochs + 1), desc="Epochs")
        for epoch in pbar:
            tr_metrics = train()
            vl_metrics = evaluate()
            learning_rate = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.args.learning_rate
            teacher_ratio = None if self.teacher_ratio_scheduler is None else self.teacher_ratio_scheduler.ratio
            d = {"epoch": epoch, "learning_rate": learning_rate, "teacher_ratio": teacher_ratio} | tr_metrics | vl_metrics
            self.update_logs(d)
            self.update_best(d)
            self.update_save(d)
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(metrics=vl_metrics[self.args.metric], epoch=None)
                else:
                    self.scheduler.step()
            if self.stopper is not None:
                self.stopper.step(vl_metrics[self.args.metric])
                if self.stopper.stop:
                    break
            if self.teacher_ratio_scheduler is not None:
                self.teacher_ratio_scheduler.step()
            if any(math.isnan(d[m]) or math.isinf(d[m]) for m in ("tr_loss", "vl_loss")):
                raise ValueError("NaN/Inf Loss Detected!")

        return self

    def evaluate_saved_models(self, epochs: Optional[list[int]] = None) -> None:

        @find_executable_batch_size(
            starting_batch_size=self.args.vl_batch_size,
            starting_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )
        def evaluate(batch_size: int, gradient_accumulation_steps: int) -> dict[str, float]:  # pylint: disable=unused-argument
            nonlocal self
            self.args.vl_batch_size = batch_size
            return self.evaluate()

        if not self.args.find_executable_batch_size:
            evaluate = self.evaluate

        outfile = self.args.outdir / "results_evaluation.jsonl"
        outfile.unlink(missing_ok=True)

        if epochs is None:
            epochs = [i for i, _ in enumerate(self.args.outdir.glob("model_*.pth"))]

        pbar = self._get_pbar(epochs, desc="Epochs")
        for epoch in pbar:
            checkpoint = self.args.outdir / f"model_{epoch}.pth"
            self.model = torch.load(checkpoint, map_location="cpu")
            self.model.to(self.args.device)
            vl_metrics = evaluate()
            results = {"epoch": epoch} | vl_metrics
            self.log.append(results)
            if not self.args.silent:
                print(self._fmt_dict(results))
            with open(outfile, "a") as fp:
                fp.write(json.dumps(results) + "\n")

    def update_logs(self, results: dict[str, float]) -> None:
        self.log.append(results)
        if not self.args.silent:
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
                ...

    def train(self) -> dict[str, float]:
        t_0 = time.time()

        self.model.train()
        dataloader = self.get_tr_dataloader()
        num_steps = math.ceil(len(dataloader) / self.args.gradient_accumulation_steps)
        results: defaultdict[str, list[float]] = defaultdict(lambda: [0] * num_steps)
        step = 0

        pbar = self._get_pbar(dataloader, total=len(dataloader), desc="Training...", leave=False)
        for mini_step, batch in enumerate(pbar):

            batch = self.batch_to_device(batch)

            # Compute normalized loss, skip noisy losses
            outputs = self.forward(batch)
            loss, losses = self.compute_loss(batch, outputs)
            loss = loss / self.args.gradient_accumulation_steps
            losses = {k: v / self.args.gradient_accumulation_steps for k, v in losses.items()}
            # NOTE: this could theoretically cause the weights to never step, which should probably be handled.
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                warnings.warn(f"NaN/Inf Loss Detected! {mini_step=} loss={loss.item()}")
                continue
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
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1

            # Perform logging every `logging_steps` `steps` (not minibatch steps!)
            condition_1 = self.args.logging_steps > 0
            condition_2 = step > 0
            condition_3 = step % self.args.logging_steps == 0
            if condition_1 and condition_2 and condition_3:
                d = {"step": step}
                for k, v in results.items():
                    start = step - self.args.logging_steps
                    stop = start + self.args.logging_steps
                    d[f"_{k}"] = mean(v[start:stop])
                print(self._fmt_dict(d))

        # Average statistics over epoch
        for k, v in results.items():
            results[k] = mean(v)
        results["tr_time"] = time.time() - t_0

        # If all we got was NaN's for Inf's, add to the dict and let the caller handle.
        results["tr_loss"] = results.get("tr_loss", float("nan"))

        return dict(results)

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()

        results: defaultdict[str, list[float]] = defaultdict(list)
        self.model.eval()
        dataloader = self.get_vl_dataloader()
        pbar = self._get_pbar(dataloader, total=len(self.vl_dataset) // self.args.vl_batch_size, desc="Validating...", leave=False)
        with torch.no_grad():
            inputs = defaultdict(list)
            for step, batch in enumerate(pbar):  # pylint: disable=unused-variable

                batch = self.batch_to_device(batch)

                outputs = self.forward_eval(batch)
                loss, losses = self.compute_loss(batch, outputs)
                for k, v in self.get_compute_metrics_inputs(batch, outputs).items():
                    inputs[k].extend(v)

                results["vl_loss"].append(loss.item())
                for k, v in losses.items():
                    results[f"vl_{k}"].append(v)

        for k, v in results.items():
            results[k] = mean(v)
        results["vl_time"] = time.time() - t_0
        results |= {f"vl_{k}": v for k, v in self.compute_metrics(**inputs).items()}

        return dict(results)

    @abstractmethod
    def forward(self, batch: tuple) -> tuple:
        """Send a batch of inputs forward through the model.

        Args:
            batch (tuple): batch of inputs.

        Returns:
            tuple: model output(s), e.g., logits.
        """

    def forward_eval(self, batch: tuple) -> tuple:
        return self.forward(batch)

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

    # TODO: implement in subclasses and adjust their methods accordingly.
    def batch_to_device(self, batch: tuple) -> tuple:
        """Send a batch of inputs to the associated device.

        Args:
            batch (tuple): batch of inputs.

        Returns:
            tuple: same batch of inputs on the device.
        """
        return tuple(x.to(self.args.device) for x in batch)

    def compute_metrics(self, **kwds: list[Any]) -> dict[str, float]:  # pylint: disable=unused-argument
        """Compute the validation metrics over a set of examples.

        Args:
            kwds (dict[str, List[Any]]): keyword arguments as returned by self.get_compute_metrics_inputs.

        Returns:
            dict[str, float]: metrics for the set.
        """
        return {}

    def get_compute_metrics_inputs(self, batch: tuple, outputs: tuple) -> dict[str, list]:  # pylint: disable=unused-argument
        """Extract the inputs for self.compute_metrics from a batch of inputs and outputs for computing metrics.

        Args:
            batch (tuple): batch of inputs.
            outputs (tuple): model output(s), e.g., logits, as return by self.forward.

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
            pin_memory=self.args.pin_memory,
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


class ListDataset(Dataset):

    def __init__(self, *lists: list[Tensor]) -> None:
        self.lists = lists

    def __len__(self) -> int:
        return len(self.lists[0])

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        return tuple(l[idx] for l in self.lists)


def collate_fn_with_basic_padding(batch: list[tuple[Tensor, Tensor]], padding_value: float | int = 0.0) -> tuple[Tensor, Tensor]:
    x = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=padding_value)
    y = torch.stack([b[1] for b in batch]).to(torch.float32)
    return x, y
