"""
Code to manage the CAIDA datasets.

First, download the CAIDA dataset using wget:
> wget --recursive --level=16 --no-parent --user="lk3591@g.rit.edu" --pasword=PASSWORD "https://data.caida.org/datasets/passive-2016/"
Next, decompress all the pcap files.
> find . -name "*.gz" -type f -print0 | xargs -0 gunzip ???
"""

from array import array
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
import gzip
import json
import os
from pathlib import Path
import pickle
from pprint import pformat, pprint
import multiprocessing as mp
import shutil
import sys
from tempfile import NamedTemporaryFile
import time
from typing import Generator, NamedTuple

import dpkt
from scapy.all import rdpcap
import numpy as np
import pyshark
from tqdm import tqdm


CAIDA_ROOT_PATH = Path("/home/lk3591/Documents/datasets/CAIDA/data.caida.org/datasets/")
VERBOSE = True


def decompress(f: Path) -> Path:
    temp_file = NamedTemporaryFile(delete=False)
    with gzip.open(f, "rb") as f_in:
        with open(temp_file.name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return Path(temp_file.name)


def extract_flows(f: Path) -> dict[tuple, np.ndarray]:
    flows = defaultdict(list)
    cap = pyshark.FileCapture(f, keep_packets=False, use_json=True)
    try:
        for _, packet in enumerate(cap):
            try:
                if "IP" in packet:
                    ip_src = packet.ip.src
                    ip_dst = packet.ip.dst
                    protocol = packet.ip.proto

                    if "TCP" in packet:
                        sport = packet.tcp.srcport
                        dport = packet.tcp.dstport
                    elif "UDP" in packet:
                        sport = packet.udp.srcport
                        dport = packet.udp.dstport
                    else:
                        continue

                    timestamp = float(packet.sniff_timestamp)
                    flow_key = (ip_src, ip_dst, sport, dport, protocol)
                    flows[flow_key].append(timestamp)
            except AttributeError:
                continue

    finally:
        cap.close()

    flows = {flow_key: np.array(timestamps) for flow_key, timestamps in flows.items()}
    return flows


def compute_ipds(flows: dict[tuple, np.ndarray]) -> dict[tuple, np.ndarray]:
    ipds = {}
    for flow_key, timestamps in flows.items():
        timestamps.sort()
        t1 = timestamps[:-1]
        t2 = timestamps[1:]
        ipds[flow_key] = t2 - t1
    return ipds


class CaidaSample(NamedTuple):
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: int
    ipds: array


def save_ipds(ipds: dict[tuple, np.ndarray], f: Path) -> None:
    data = []
    for flow_key, delays in ipds.items():
        sample = CaidaSample(*flow_key, array("f", delays.tolist()))
        data.append(sample)
    with open(f, "wb") as fp:
        pickle.dump(data, fp)


def process_file(input_file: Path, output_file: Path) -> None:
    try:
        t_0 = time.time()
        temp_file = decompress(input_file)
        flows = extract_flows(temp_file)
        ipds = compute_ipds(flows)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_ipds(ipds, output_file)
        os.unlink(temp_file)
        print(f"Worker {os.getpid()} process {len(flows)} flows in {round(time.time() - t_0)} seconds from {'/'.join(input_file.parts[-4:])}")
    except Exception as err:
        print(f"Worker {os.getpid()} failed to process {input_file} because of {type(err)}.\n{err=}")


def _json_to_pickle(f_json: Path, remove: bool) -> None:
        print(f"Worker {os.getpid()} processing {str(f_json)}")
        data = []
        with open(f_json, "r") as fp:
            for line in fp:
                d = json.loads(line.strip())
                d["ipds"] = array("f", d["ipds"])
                data.append(d)
        f_pickle = f_json.with_suffix(".pickle")
        data = [CaidaSample(*d.values()) for d in data]
        with open(f_pickle, "wb") as fp:
            pickle.dump(data, fp)
        if remove:
            f_json.unlink()


def json_to_pickle(
    output: Path = Path("./data"),
    year: str = "passive-2016",
    source: str = "equinix-chicago",
    remove: bool = False
) -> None:
    output_root = output / year / source
    files = sorted(list(output_root.glob("*.json")))
    with mp.Pool(8) as pool:
        pool.starmap(_json_to_pickle, [(f, remove) for f in files])


def stream_caida_data(
    output: Path = Path("./data"),
    year: str = "passive-2016",
    source: str = "equinix-chicago",
) -> Generator[CaidaSample, None, None]:
    output_root = output / year / source
    for f in output_root.rglob("*.pickle"):
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        for sample in data:
            yield sample


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--year", type=str, default="passive-2016")
    parser.add_argument("--source", type=str, default="equinix-chicago")
    parser.add_argument("--output", type=Path, default=Path("./data"))
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    input_root: Path = CAIDA_ROOT_PATH / args.year / args.source
    output_root = args.output / args.year / args.source
    input_files = list(input_root.rglob("*.pcap.gz"))
    output_files = [output_root / f"{f.stem}.pickle" for f in input_files]

    print(f"Found {len(input_files)} files to extract IPDs from. Saving to {str(output_root)}.")

    complete = []
    for i, output_file in enumerate(output_files):
        if output_file.exists():
            complete.append(i)
    input_files = [f for i, f in enumerate(input_files) if i not in complete]
    output_files = [f for i, f in enumerate(output_files) if i not in complete]

    print(f"Found {len(complete)} files already processed. Processing {len(input_files)} files...")

    if args.num_workers < 2:
        for input_file, output_file in tqdm(zip(input_files, output_files), total=len(input_files)):
            process_file(input_file, output_file)
    else:
        with mp.Pool(args.num_workers) as pool:
            pool.starmap(process_file, zip(input_files, output_files))


if __name__ == "__main__":
    main()
