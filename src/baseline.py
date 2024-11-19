"""
Compute baseline metrics for IPD transformation using stochasitc model.

RESULTS
-------
max_length: 64
{
 'length_mae': 22.262060165405273,
 'length_mse': 536.0961303710938,
 'length_ndev': 0.5603351593017578,
 'length_nrmse': 0.37352216243743896,
 'length_r2': -3107.142333984375,
 'timing_mae': 0.012232023291289806,
 'timing_mse': 0.0020123575814068317,
 'timing_ndev': 17.254379272460938,
 'timing_nrmse': 3.803612470626831,
 'timing_r2': -0.07701325416564941,
}
max_length: 128
{
 'length_mae': 46.03347396850586,
 'length_mse': 2295.72216796875,
 'length_ndev': 0.5856590270996094,
 'length_nrmse': 0.3859179615974426,
 'length_r2': -26.468759536743164,
 'timing_mae': 0.015545784495770931,
 'timing_mse': 0.0048371972516179085,
 'timing_ndev': 21.774375915527344,
 'timing_nrmse': 4.608433246612549,
 'timing_r2': -0.05042243003845215,
}
max_length: 256
{
 'length_mae': 85.72103881835938,
 'length_mse': 8759.0263671875,
 'length_ndev': 0.6410614848136902,
 'length_nrmse': 0.43246957659721375,
 'length_r2': -2.602083444595337,
 'timing_mae': 0.016090068966150284,
 'timing_mse': 0.007850836031138897,
 'timing_ndev': 22.558767318725586,
 'timing_nrmse': 5.678414821624756,
 'timing_r2': -0.03260445594787598,
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


def main(seed: int = 0, max_length: Optional[int] = None, pair_mode: str = "single_hops", num_workers: int = 1, batch_size: int = 1024) -> dict[str, float]:

    seed_everything(seed)

    ipd_groups = [[s.ipds for s in c] for c in extract_data()]
    build_pairs_fn = BUILDER_PAIR_MODES[pair_mode]
    dataset = ApproximatorDataset(build_pairs_fn(ipd_groups))
    total = len(dataset)

    x = []
    y = []
    for src, tgt in tqdm(dataset, total=total, desc="Prepping data..."):
        x.append(src.numpy()[0:max_length - 2])
        y.append(tgt.numpy()[0:max_length - 2])

    predictions = []
    for x_i in tqdm(batched(x, batch_size), total=total // batch_size, desc="Generating predictions..."):
        predictions.extend(synthetic_hops(x_i, num_workers))
    y_true = [torch.from_numpy(y_i).to(torch.float32) for y_i in y]
    y_pred = [torch.from_numpy(y_i).to(torch.float32) for y_i in predictions]

    # Below is nearly a copy-paste from the Approximator.compute_metrics.

    NUM = len(y_true)                            # Number of sequences
    MET = ("r2", "mae", "mse", "nrmse", "ndev")  # Metrics
    metrics = {}

    # Compute the length metrics, not considering the length added by BOS and EOS tokens.
    lengths = torch.tensor([[len(y_true[i]), len(y_pred[i])] for i in range(NUM)], dtype=torch.float32)
    y_tr = lengths[:, 0]
    y_pr = lengths[:, 1]
    m = regression_report(y_tr.numpy(force=True), y_pr.numpy(force=True))
    metrics.update({f"length_{k}": v for k, v in m.items() if k in MET})

    # Compute the timing metrics over the shorter of the two sequences, excluding BOS and EOS tokens.
    minimum = torch.minimum(lengths[:,0], lengths[:,1]).to(torch.int64).tolist()
    y_tr = torch.cat([y_true[i][:l] for i, l in enumerate(minimum)])
    y_pr = torch.cat([y_pred[i][:l] for i, l in enumerate(minimum)])
    m = regression_report(y_tr.numpy(force=True), y_pr.numpy(force=True))
    metrics.update({f"timing_{k}": v for k, v in m.items() if k in MET})

    return metrics


def cli():

    parser = ArgumentParser()
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair_mode", type=str, default="single_hops")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    metrics = main(args.seed, args.max_length, args.pair_mode, args.num_workers, args.batch_size)
    pprint(metrics)


if __name__ == "__main__":
    main()
