"""
Creates bash files to run synthetic experiments.
"""

from argparse import ArgumentParser
import sys


parser = ArgumentParser()
parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES={device}")
parser.add_argument("--demo", action="store_true", help="Run demo experiments")
args = parser.parse_args()


def get_body(
    fingerprint_length: int,
    flow_length: int,
    amplitude: float,
    noise_deviation_low: float,
    noise_deviation_high: float,
    tr_num_samples: int,
    num_train_epochs: int,
    outfile: str,
    logfile: str,
) -> str:
    return f"""#!/bin/bash -l

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate TrafficAnal

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
    --vl_num_samples=50000 \\
    --seed=0 \\
    --outfile={outfile} \\
    --num_train_epochs={num_train_epochs} \\
    --batch_size=1024 \\
    --learning_rate=1e-4 \\
    --dataloader_num_workers=4 \\
    {'--demo \\' if args.demo else 'REMOVE'}
    --device=cuda:{args.device} > {logfile} 2>&1
    """.replace("    ", "").replace("REMOVE\n", "")


runfiles = []


# Experiment 1: Impact of Fingerprint Length

for tr_num_samples in (200000, 500000):
    for fingerprint_length in (512, 1024, 2048, 4096, 8192, 16384):
        jobname = f"E1--{tr_num_samples}--{fingerprint_length}"
        logfile = f"./logs/{jobname}.log"
        outfile = f"./output/{jobname}.jsonl"
        runfile = f"./run/{jobname}.sh"
        body = get_body(
            fingerprint_length=fingerprint_length,
            flow_length=100,
            amplitude=5e-3,
            noise_deviation_low=2e-3,
            noise_deviation_high=10e-3,
            tr_num_samples=tr_num_samples,
            num_train_epochs=100,
            outfile=outfile,
            logfile=logfile,
        )
        with open(runfile, "w") as f:
            f.write(body + "\n")
        runfiles.append(runfile)


# Experiment 2: Impact of Noise Deviation

for noise_deviation_low, noise_deviation_high in ((2e-3, 10e-3), (10e-3, 20e-3), (20e-3, 30e-3)):
    for amplitude in (5e-3, 10e-3, 20e-3, 30e-3, 40e-3):
        jobname = f"E2--{noise_deviation_low}--{noise_deviation_high}--{amplitude}"
        logfile = f"./logs/{jobname}.log"
        outfile = f"./output/{jobname}.jsonl"
        runfile = f"./run/{jobname}.sh"
        body = get_body(
            fingerprint_length=4096,
            flow_length=100,
            amplitude=amplitude,
            noise_deviation_low=noise_deviation_low,
            noise_deviation_high=noise_deviation_high,
            tr_num_samples=200000,
            num_train_epochs=100,
            outfile=outfile,
            logfile=logfile,
        )
        with open(runfile, "w") as f:
            f.write(body + "\n")
        runfiles.append(runfile)


# Experiment 3: Impact of Flow Length

for tr_num_samples in (200000, 500000):
    for num_train_epochs in (100, 200):
        for flow_length in (50, 100, 150):
            jobname = f"E3--{tr_num_samples}--{num_train_epochs}--{flow_length}"
            logfile = f"./logs/{jobname}.log"
            outfile = f"./output/{jobname}.jsonl"
            runfile = f"./run/{jobname}.sh"
            body = get_body(
                fingerprint_length=1024,
                flow_length=flow_length,
                amplitude=5e-3,
                noise_deviation_low=2,
                noise_deviation_high=10,
                tr_num_samples=tr_num_samples,
                num_train_epochs=num_train_epochs,
                outfile=outfile,
                logfile=logfile,
            )
            with open(runfile, "w") as f:
                f.write(body + "\n")
            runfiles.append(runfile)


with open("./run/run.sh", "w") as f:
    f.write("\n".join([f"CUDA_VISIBLE_DEVICES={args.device} bash {runfile}" for runfile in runfiles]) + "\n")
