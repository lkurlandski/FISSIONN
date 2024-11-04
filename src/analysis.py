"""
Analyze the output of the FINNTrainer.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import os
from typing import Self

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm


def zscore_from_confidence(confidence_level: float) -> float:
    # Compute the z-score corresponding to the confidence level
    z_score = norm.ppf((1 + confidence_level) / 2)
    return z_score


def confidence_interval(x: np.ndarray, c: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basic confidence interval for a series of observations.

    Args:
        x (np.ndarray): data
        c (float, optional): confidence level

    Returns:
        np.ndarray: mean
        np.ndarray: interval
        np.ndarray: standard deviation

    Usage:
        >>> # One experiment; five observations
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> m, i, s = confidence_interval(x)
        >>> # Two experiments; five observations each
        >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        >>> m, i, s = confidence_interval(x)
    """
    if isinstance(x, list):
        x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise TypeError("Input must be a 1D or 2D array.")

    z = norm.ppf((1 + c) / 2)
    m = x.mean(axis=1)
    s = x.std(axis=1)
    i = z * s / np.sqrt(x.shape[0])
    return m, i, s


class Analyzer:

    def __init__(self, outdir: Path | os.PathLike) -> None:
        self.outdir = Path(outdir)
        self.log: list[dict[str, float]] = []
        self.data: dict[str, list[float] | np.ndarray] = {}

    def __call__(self) -> Self:

        with open(self.outdir / "results.jsonl", "r") as fp:
            for line in fp:
                self.log.append(json.loads(line))

        if not self.log:
            raise ValueError("No data found in log file.")

        self.data = {key: [] for key in self.log[0].keys()}
        for d in self.log:
            for k, v in d.items():
                self.data[k].append(v)

        self.data = {k: np.array(v) for k, v in self.data.items()}

        return self

    def plot(self, keys_for_main: tuple[str] = tuple()) -> Self:

        for k, v in self.data.items():
            if k == "epoch":
                continue
            fig, _ = self._plot(k)
            fig.savefig(self.outdir / f"{k}.png")
            plt.close(fig)

        if keys_for_main:
            fig, _ = self._plot(keys_for_main)
            fig.savefig(self.outdir / "main.png")
            plt.close(fig)

        fig, _ = self._plot([k for k in self.data.keys() if k != "epoch"])
        fig.savefig(self.outdir / "all.png")
        plt.close(fig)
        return self

    def _plot(self, keys: str | list[str]) -> tuple[Figure, Axes]:
        keys = [keys] if isinstance(keys, str) else keys

        fig, ax = plt.subplots()
        fig: Figure
        ax: Axes

        for k in keys:
            ax.plot(self.data["epoch"], self.data[k], label=k)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        # ax.set_ylim(0, 1)
        fig.legend()

        return fig, ax


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--keys_for_main", type=str, nargs="+", default=[])
    args = parser.parse_args()

    analyzer = Analyzer(args.outdir)
    analyzer = analyzer()
    analyzer.plot(keys_for_main=args.keys_for_main)


if __name__ == "__main__":
    main()
