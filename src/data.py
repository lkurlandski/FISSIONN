"""
Loading data and the like.
"""

from collections import namedtuple
from dataclasses import dataclass
import pickle
from typing import Generator, Optional, TypeAlias  # pylint: disable=no-name-in-module

import numpy as np


FILE = "/home/lk3591/Documents/code/TrafficAnal/data/synthetic.pickle"


def process(x: tuple) -> np.ndarray:
    """
    Simple example function to use when processing 
    """
    timestamps, packet_sizes, directions = x  # pylint: disable=unused-variable
    iats = np.diff(timestamps)
    iats = np.concatenate(([0], iats))
    return iats


def load_data(file: str = "/home/lk3591/Documents/code/TrafficAnal/data/synthetic.pickle") -> list[list[np.ndarray]]:
    """
    Load the metadata for all samples collected in our SSID data, and process them using the process() function.

    Returns: a nested list of processed streams
        The outer list contains lists of correlated processed streams, while the inner lists contain individual instances 
        (with all instances within the list being streams produced by hosts within the same multi-hop tunnel)
    """
    with open(file, "rb") as fp:
        all_data = pickle.load(fp)

    IP_info = all_data['IPs']   # extra src. & dst. IP info available for each stream  # pylint: disable=unused-variable
    data = all_data['data']     # stream metadata organized by sample and hosts (per sample)

    # list of all sample idx
    sample_IDs = list(data.keys())

    # fill with lists of correlated samples
    all_streams = []

    attacker_ID = None
    target_ID   = None

    # each 'sample' contains a variable number of hosts (between 3 and 6 I believe)
    for s_idx in sample_IDs:
        sample = data[s_idx]
        host_IDs = list(sample.keys())

        # first and last hosts represent the attacker's machine and target endpoint of the chain respectively
        # these hosts should contain only one SSH stream in their sample
        attacker_ID = 1
        target_ID   = len(host_IDs)

        # the stepping stone hosts are everything in-between
        # these hosts should each contain two streams
        steppingstone_IDs = list(filter(lambda x: x not in [attacker_ID, target_ID], host_IDs))  # pylint: disable=unused-variable

        # loop through each host, process stream metadata into vectors, and add to list
        correlated_streams = []
        for h_idx in host_IDs:
            correlated_streams.extend([process(x) for x in sample[h_idx]])

        # add group of correlated streams for the sample into the data list
        all_streams.append(correlated_streams)

    return all_streams


def stream_synthetic_data() -> Generator[np.ndarray, None, None]:
    all_streams: list[list[np.ndarray]] = load_data()
    for group in all_streams:
        for stream in group:
            yield stream


StreamSlice = namedtuple("StreamSlice", ["timestamp", "packet_size", "direction", "ipd"])


@dataclass
class Stream:
    timestamps: np.ndarray
    packet_sizes: np.ndarray
    directions: np.ndarray
    ipds: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.ipds is None:
            self.ipds = np.concatenate(([0], np.diff(self.timestamps)))

    def __iter__(self):
        for t, p, d, i in zip(self.timestamps, self.packet_sizes, self.directions, self.ipds):
            yield StreamSlice(t, p, d, i)

    def __len__(self):
        lengths = [len(self.timestamps), len(self.packet_sizes), len(self.directions), len(self.ipds)]
        if len(set(lengths)) != 1:
            raise ValueError("All arrays must have the same length.")
        return lengths[0]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"Stream(\n"
            f"    timestamps={self.timestamps[0:8].round(4).tolist()}...,\n"
            f"    packet_sizes={self.packet_sizes[0:8].round(4).tolist()}...,\n"
            f"    directions={self.directions[0:8].round(4).tolist()}...,\n"
            f"    ipds={self.ipds[0:8].round(4).tolist()}...,\n"
            ")\n"
            f"{len(self)=}"
        )


Chain: TypeAlias = list[Stream]


def extract_data(reorder: bool = True) -> list[Chain]:
    """
    Extract and organize the data from the pickle file in a coherent manner.
    """
    with open(FILE, "rb") as fp:
        data: dict[str, dict[int, dict[int, list[np.ndarray]]]] = pickle.load(fp)["data"]

    chains = []
    for sample in data.values():
        correlated_streams = []
        for host in sample.values():
            streams = []
            for x in host:
                timestamps, packet_sizes, directions = x
                if reorder:
                    idx = np.argsort(timestamps)
                    timestamps = timestamps[idx]
                    packet_sizes = packet_sizes[idx]
                    directions = directions[idx]
                streams.append(Stream(timestamps, packet_sizes, directions))
            correlated_streams.extend(streams)
        chains.append(correlated_streams)

    return chains
