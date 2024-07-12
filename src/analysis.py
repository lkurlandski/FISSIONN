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


class FINNTrainerAnalyzer:

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
        for k in self.log[1].keys():  # FIXME: remove
            if k not in self.data:
                self.data[k] = [np.NaN]
        for d in self.log:
            for k, v in d.items():
                self.data[k].append(v)
            
        self.data = {k: np.array(v) for k, v in self.data.items()}

        return self

    def plot(self) -> Self:

        for k, v in self.data.items():
            if k == "epoch":
                continue
            fig, _ = self._plot(k)
            fig.savefig(self.outdir / f"{k}.png")
            plt.close(fig)

        fig, _ = self._plot(["tr_loss", "vl_loss", "vl_extraction_rate", "vl_bit_error_rate"])
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
        ax.set_ylim(0, 1)
        fig.legend()

        return fig, ax


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    analyzer = FINNTrainerAnalyzer(args.outdir)
    analyzer()
    analyzer.plot()


if __name__ == "__main__":
    main()
