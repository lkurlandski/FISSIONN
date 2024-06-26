# README

## Dependencies

conda create -n TrafficAnal python=3.12 pytorch torchvision torchaudio torchtext pytorch-cuda=12.1 scipy pandas scikit-learn -c pytorch -c nvidia -c conda-forge

sudo apt install wireshark

# Data

for year in {2009..2019}; do wget --recursive --level=16 --no-parent --user="lk3591@g.rit.edu" --pasword=PASWORD "https://data/caida/org/datasets/passive-${year}/"; done

