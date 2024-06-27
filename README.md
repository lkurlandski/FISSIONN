# README

## Set up Environment

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

pip install pyshark

# Download CAIDA Dataset

wget --recursive --level=16 --no-parent --user="lk3591@g.rit.edu" --pasword=PASWORD "https://data/caida/org/datasets/passive-2016/"
wget --recursive --level=16 --no-parent --user="lk3591@g.rit.edu" --pasword=PASWORD "https://data/caida/org/datasets/passive-2009/"

# Process CAIDA Dataset

python src/caida.py --year="passive-2016" --source="equinix-chicago" --output="./data" --num_workers=8
python src/caida.py --year="passive-2018" --source="equinix-nyc" --output="./data" --num_workers=8

# Reproduce Experiments


