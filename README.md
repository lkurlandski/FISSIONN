# README

Unofficial implementation of the network watermarking technique, "FINN". This code is not endorsed by the original authors and comes with aboslutely no guarentees or warranty.

CITATION
--------
@inproceedings{rezaei2021finn,
  title={FINN: Fingerprinting network flows using neural networks},
  author={Rezaei, Fatemeh and Houmansadr, Amir},
  booktitle={Proceedings of the 37th Annual Computer Security Applications Conference},
  year={2021}
}

## Set up Environment

Our implementation uses Python and Pytorch. Wireshark and pyshark are not required for the demo.

```bash
sudo apt install wireshark

conda create -n TrafficAnal python=3.12 pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 torchtext=0.18.0 pytorch-cuda=12.1 scikit-learn scipy pandas scikit-learn -c pytorch -c nvidia -c conda-forge

conda activate TrafficAnal

pip install pyshark
```

## Download CAIDA Dataset

Downloading the CAIDA dataset is not required for the demo.

To get access to the CAIDA dataset, follow the instructions from [CAIDA](https://www.caida.org/catalog/datasets/passive_dataset/). After creating a USERNAME and PASSWORD, you can download the data using wget.

```bash
wget --directory-prefix=/PATH/TO/CAIDA --user=USERNAME --password=PASSWORD --recursive --level=16 --no-parent --no-clobber "https://data.caida.org/datasets/passive-2016/"
wget --directory-prefix=/PATH/TO/CAIDA --user=USERNAME --password=PASSWORD --recursive --level=16 --no-parent --no-clobber "https://data.caida.org/datasets/passive-2018/"
```

## Process CAIDA Dataset

Processing the CAIDA dataset is not required for the demo.

After downloading the CAIDA datasets, we need to extract interpacket delay sequences from them. This is quite a lengthy process. On my server (40 cores), I found no benefit from using more than 8 python processes. It still took about a week to process all the data.

```bash
python src/caida.py --year="passive-2016" --source="equinix-chicago" --output="./data" --num_workers=8
python src/caida.py --year="passive-2018" --source="equinix-nyc" --output="./data" --num_workers=8
```

## Reproduce Experiments

To produce a set of bash files containing the configuration for each of the experiments:
```bash
python ./run/reproduce.py --device=0
```

Add the `--demo` flag to run the demo only.

You can then run these individually using
```bash
CUDA_VISIBLE_DEVICES=0 bash ./run/{jobname}.sh
```

Or all at once using
```bash
./run/run.sh
```
