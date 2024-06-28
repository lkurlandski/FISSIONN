# README

## TODO

- adjust caida.py to take a CAIDA root argument, then adjust the download scripts accordingly...

## Set up Environment

```bash
sudo apt install wireshark

conda create -n TrafficAnal \
python=3.12 \
pytorch=2.3.0 \
torchvision=0.18.0 \
torchaudio=2.3.0 \
torchtext=0.18.0 \
pytorch-cuda=12.1 \
scikit-learn \
scipy \
pandas \
scikit-learn \
-c pytorch -c nvidia -c conda-forge

conda activate TrafficAnal

pip install pyshark
```

## Download CAIDA Dataset

To get access to the CAIDA dataset, follow the instructions from [CAIDA](https://www.caida.org/catalog/datasets/passive_dataset/). After creating a USERNAME and PASSWORD, you can download the data using wget.

```bash
wget --recursive --level=16 --no-parent --user=USERNAME --pasword=PASSWORD "https://data/caida/org/datasets/passive-2016/"
wget --recursive --level=16 --no-parent --user=USERNAME --pasword=PASSWORD "https://data/caida/org/datasets/passive-2018/"
```

## Process CAIDA Dataset

After downloading the CAIDA datasets, we need to extract interpacket delay sequences from them. This is quite a lengthy process. On my server (40 cores), I found no benefit from using more than 8 python processes. It still took about a week to process all the data.

```bash
python src/caida.py --year="passive-2016" --source="equinix-chicago" --output="./data" --num_workers=8
python src/caida.py --year="passive-2018" --source="equinix-nyc" --output="./data" --num_workers=8
```

## Reproduce Experiments

```bash
CUDA_VISIBLE_DEVICES=0 bash run/reproduce.sh
```
