"""
Creates bash files to run synthetic experiments.

Profiling:
  - Ran four experiments with dataloader_num_workers in (4, 8) and batch_size in (1024, 4096).
  - Counterintuitively, the fastest run was dataloader_num_workers==4 and batch_size==1024.
  - Smaller batch size had more gradient updates, so obviously, this had better metrics.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys


parser = ArgumentParser()
parser.add_argument("--iteration", type=int, required=True)
parser.add_argument("--device", type=int, default=0, help="-1, 0, 1, etc. for CPU, GPU, etc.")
parser.add_argument("--demo", action="store_true", help="Run demo experiments")
args = parser.parse_args()


def get_body(
    fingerprint_length: int,
    flow_length: int,
    amplitude: float,
    noise_deviation_low: float,
    noise_deviation_high: float,
    tr_num_samples: int,
    vl_num_samples: int,
    epochs: int,
    outdir: str,
    logfile: str,
) -> str:
    return f"""#!/bin/bash -l

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate TrafficAnal

    echo "Running {outdir}..."

    python -u src/finn.py \\
    --fingerprint_length={fingerprint_length} \\
    --flow_length={flow_length} \\
    --min_flow_length={flow_length} \\
    --max_flow_length={sys.maxsize} \\
    --amplitude={amplitude} \\
    --noise_deviation_low={noise_deviation_low} \\
    --noise_deviation_high={noise_deviation_high} \\
    --encoder_loss_weight=1.0 \\
    --decoder_loss_weight=5.0 \\
    --tr_num_samples={tr_num_samples} \\
    --vl_num_samples={vl_num_samples} \\
    --seed=0 \\
    --outdir={outdir} \\
    --epochs={epochs} \\
    --tr_batch_size=1024 \\
    --vl_batch_size=4096 \\
    --learning_rate=1e-4 \\
    --num_workers=4 \\
    {'--demo \\' if args.demo else 'REMOVE'}
    --device={'cuda:0' if args.device >= 0 else 'cpu'} > {logfile} 2>&1
    """.replace("    ", "").replace("REMOVE\n", "")


for f in Path("./run").glob("*.sh"):
    f.unlink()

runfiles = []


# Experiment 1: Impact of Fingerprint Length

for tr_num_samples in (200000, 500000):
    for fingerprint_length in (512, 1024, 2048, 4096, 8192, 16384):
        jobname = f"E1-{args.iteration}--{tr_num_samples}--{fingerprint_length}"
        logfile = f"./logs/{jobname}.log"
        outdir = f"./output/{jobname}"
        runfile = f"./run/{jobname}.sh"
        body = get_body(
            fingerprint_length=fingerprint_length,
            flow_length=100,
            amplitude=40e-3,
            noise_deviation_low=2e-3,
            noise_deviation_high=10e-3,
            tr_num_samples=tr_num_samples,
            vl_num_samples=50000,
            epochs=100,
            outdir=outdir,
            logfile=logfile,
        )
        with open(runfile, "w") as f:
            f.write(body + "\n")
        runfiles.append(runfile)


# Experiment 2: Impact of Noise Deviation

for noise_deviation_low, noise_deviation_high in ((2e-3, 10e-3), (10e-3, 20e-3), (20e-3, 30e-3)):
    for amplitude in (5e-3, 10e-3, 20e-3, 30e-3, 40e-3):
        jobname = f"E2-{args.iteration}--{noise_deviation_low}--{noise_deviation_high}--{amplitude}"
        logfile = f"./logs/{jobname}.log"
        outdir = f"./output/{jobname}"
        runfile = f"./run/{jobname}.sh"
        body = get_body(
            fingerprint_length=4096,
            flow_length=100,
            amplitude=amplitude,
            noise_deviation_low=noise_deviation_low,
            noise_deviation_high=noise_deviation_high,
            tr_num_samples=200000,
            vl_num_samples=50000,
            epochs=100,
            outdir=outdir,
            logfile=logfile,
        )
        with open(runfile, "w") as f:
            f.write(body + "\n")
        runfiles.append(runfile)


# Experiment 3: Impact of Flow Length

for tr_num_samples in (200000, 500000):
    for epochs in (100, 200):
        for flow_length in (50, 100, 150):
            jobname = f"E3-{args.iteration}--{tr_num_samples}--{epochs}--{flow_length}"
            logfile = f"./logs/{jobname}.log"
            outdir = f"./output/{jobname}"
            runfile = f"./run/{jobname}.sh"
            body = get_body(
                fingerprint_length=1024,
                flow_length=flow_length,
                amplitude=40e-3,
                noise_deviation_low=2e-3,
                noise_deviation_high=10e-3,
                tr_num_samples=tr_num_samples,
                vl_num_samples=50000,
                epochs=epochs,
                outdir=outdir,
                logfile=logfile,
            )
            with open(runfile, "w") as f:
                f.write(body + "\n")
            runfiles.append(runfile)


# Convienient run script.

with open("./run/run.sh", "w") as f:
    for runfile in runfiles:
        f.write(f"CUDA_VISIBLE_DEVICES={args.device} bash {runfile}\n")
        f.write(f"grep ENDING ./logs/{Path(runfile).stem}.log\n")
