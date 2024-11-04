"""
"""

from argparse import ArgumentParser
from itertools import batched
import math
import multiprocessing as mp
from pprint import pprint
import random
import os
from statistics import quantiles
import sys
from typing import Literal, Optional

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.approximator import ApproximatorDataset, BUILDER_PAIR_MODES
from src.data import extract_data, Stream, Chain
from src.metrics import normalized_deviation, normalized_root_mean_squared_error
from src.utils import seed_everything


P_D = 0.51939008
P_C = 0.25000000
P_N = 0.23060992

LOC   = 0.0000000000000000E+00
SCALE = 1.0096913207851692E-06
BETA  = 2.5405908912984830E-01


def synthetic_hop_length(l_i: int, p_d: float, p_c: float, p_n: float) -> tuple[int, int]:
    if not math.isclose(p_d + p_c + p_n, 1):
        raise ValueError("Probabilities must sum to 1")

    l_f = 0
    n   = 0
    i   = 0

    while i < l_i:
        n += 1
        r = random.random()
        if r < p_d:
            l_f += 0
            i   += 1
        elif r < p_d + p_c:
            l_f += 1
            i   += 0
        else:
            l_f += 1
            i   += 1

    return l_f, n


def synthetic_hop_timings(
    dist: Literal["gennorm", "laplace", "norm"],
    num: int,
    loc: float,
    scale: float,
    beta: Optional[float] = None,
) -> np.ndarray:

    if dist == "laplace":
        beta = 1.0
    elif dist == "norm":
        scale /= math.log(2)
        beta = 2.0
    elif dist != "gennorm":
        raise ValueError("Invalid distribution")

    return stats.gennorm.rvs(beta, loc=loc, scale=scale, size=num)


def synthetic_hop(ipd: np.ndarray) -> np.ndarray:
    l_f = synthetic_hop_length(len(ipd), P_D, P_C, P_N)[0]
    return synthetic_hop_timings("gennorm", l_f, LOC, SCALE, BETA)


def synthetic_hops(ipds: list[np.ndarray] | np.ndarray, num_workers: int = 1) -> list[np.ndarray]:
    if num_workers is None or num_workers < 2:
        return [synthetic_hop(ipd) for ipd in ipds]

    with mp.Pool(num_workers) as pool:
        return pool.map(synthetic_hop, ipds)


def main():

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair_mode", type=str, default="single_hops")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    seed_everything(args.seed)

    ipd_groups = [[s.ipds for s in c] for c in extract_data()]
    build_pairs_fn = BUILDER_PAIR_MODES[args.pair_mode]
    dataset = ApproximatorDataset(build_pairs_fn(ipd_groups))
    total = len(dataset)

    x = []
    y = []
    for src, tgt in tqdm(dataset, total=total, desc="Prepping data..."):
        x.append(src.numpy())
        y.append(tgt.numpy())

    predictions = []
    for x_i in tqdm(batched(x, args.batch_size), total=total // args.batch_size, desc="Generating predictions..."):
        predictions.extend(synthetic_hops(x_i, args.num_workers))

    lengths = [(len(y_h), len(y_i)) for y_h, y_i in zip(predictions, y)]
    lengths = np.array(lengths)
    y_true = lengths[:, 0]
    y_pred = lengths[:, 1]
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    y_true = torch.from_numpy(y_true).to(torch.float32)
    y_pred = torch.from_numpy(y_pred).to(torch.float32)
    ndev  = normalized_deviation(y_pred, y_true).item()
    nrmse = normalized_root_mean_squared_error(y_pred, y_true).item()
    print(f"Length MSE: {mse}")
    print(f"Length MAE: {mae}")
    print(f"Length NDEV: {ndev}")
    print(f"Length NRMSE: {nrmse}")

    for i in range(total):
        l = min(lengths[i][0], lengths[i][1])
        y[i] = y[i][0:l]
        predictions[i] = predictions[i][0:l]
    y_true = np.concatenate(y)
    y_pred = np.concatenate(predictions)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    y_true = torch.from_numpy(y_true).to(torch.float32)
    y_pred = torch.from_numpy(y_pred).to(torch.float32)
    ndev  = normalized_deviation(y_pred, y_true).item()
    nrmse = normalized_root_mean_squared_error(y_pred, y_true).item()
    print(f"IPD MSE: {mse}")
    print(f"IPD MAE: {mae}")
    print(f"IPD NDEV: {ndev}")
    print(f"IPD NRMSE: {nrmse}")


if __name__ == "__main__":
    main()
