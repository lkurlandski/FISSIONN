"""
Compute baseline metrics for IPD transformation using stochasitc model.

RESULTS
-------
{
 'length_mae': 345.9457092285156,
 'length_mse': 1132191.875,
 'length_r2': 0.48062682151794434,
 'timing_mae': 0.00937829352915287,
 'timing_mse': 0.005738102365285158,
 'timing_nrmse': 8.561403274536133,
 'timing_r2': -0.014659881591796875,
}
"""

from argparse import ArgumentParser
from itertools import batched
import math
import multiprocessing as mp
from pprint import pformat, pprint
import random
import os
from statistics import quantiles
import sys
from typing import Literal, Optional

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.approximator import ApproximatorDataset, BUILDER_PAIR_MODES
from src.data import extract_data, Stream, Chain
from src.metrics import regression_report
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
    y_true = [torch.from_numpy(y_i).to(torch.float32) for y_i in y]
    y_pred = [torch.from_numpy(y_i).to(torch.float32) for y_i in predictions]

    # Below is a copy-paste form the Approximator.compute_metrics.

    NUM = len(y_true)                          # Number of sequences
    LEN = ("r2", "mae", "mse")                 # Length metrics
    IPD = ("r2", "mae", "mse", "nrmse", "nd")  # IPD metrics
    metrics = {}

    # Compute the length metrics, not considering the length added by BOS and EOS tokens.
    lengths = torch.tensor([[len(y_true[i]), len(y_pred[i])] for i in range(NUM)], dtype=torch.float32)
    y_tr = lengths[:, 0] - 2
    y_pr = lengths[:, 1] - 2
    m = regression_report(y_tr.numpy(force=True), y_pr.numpy(force=True))
    metrics.update({f"length_{k}": v for k, v in m.items() if k in LEN})

    # Compute the timing metrics over the shorter of the two sequences, excluding BOS and EOS tokens.
    minimum = torch.minimum(lengths[:,0], lengths[:,1]).to(torch.int64).tolist()
    y_tr = torch.cat([y_true[i][1:l-1] for i, l in enumerate(minimum)])
    y_pr = torch.cat([y_pred[i][1:l-1] for i, l in enumerate(minimum)])
    m = regression_report(y_tr.numpy(force=True), y_pr.numpy(force=True))
    metrics.update({f"timing_{k}": v for k, v in m.items() if k in IPD})

    pprint(metrics)


if __name__ == "__main__":
    main()
