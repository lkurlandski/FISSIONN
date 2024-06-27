#!/bin/bash -l

#
#
#
#
#

source ~/anaconda3/etc/profile.d/conda.sh
conda activate TrafficAnal


CUDA_VISIBLE_DEVICES=$1, python -u src/finn.py \
--device="cuda:$1" \
--batch_size=256 \
--num_samples=10240 \
--dataloader_num_workers=4 \
--num_train_epochs=100 \
--logging_steps=1 \
--disable_tqdm \
--encoder_loss_weight=0.0 \
--learning_rate=1e-4
