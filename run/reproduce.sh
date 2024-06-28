#!/bin/bash -l

source ~/anaconda3/etc/profile.d/conda.sh
conda activate TrafficAnal

ENCODER_LOSS_WEIGHT=1.0
DECODER_LOSS_WEIGHT=5.0
MAXSIZE=9223372036854775807
VL_NUM_SAMPLES=5000
SEED=0
BATCH_SIZE=1024
LEARNING_RATE=1e-4
DATA_LOADER_NUM_WORKERS=4
DEVICE="cuda"

# Experiments from section 5.1

FLOW_LENGTH=100
AMPLITUDE=5e-3
NOISE_DEVIATION_LOW=2
NOISE_DEVIATION_HIGH=10
NUM_TRAIN_EPOCHS=100

for tr_num_samples in 200000 500000
do
	for fingerprint_length in 512 1024 2048 4096 8192 16384
	do
		file="fingerprinting_length/tr_num_samples--$tr_num_samples---fingerprint_length--$fingerprint_length"
		logfile="./logs/$file.log"
		outfile="./output/$file.jsonl"
		mkdir -p $(dirname "$stdout")
		mkdir -p $(dirname "$logfile")
		echo "Running experiment: $file"
		python -u src/finn.py \
		 --fingerprint_length=$fingerprint_length \
		--flow_length=$FLOW_LENGTH \
		--min_flow_length=$FLOW_LENGTH \
		--max_flow_length=$MAXSIZE \
		--amplitude=$AMPLITUDE \
		--noise_deviation_low=$NOISE_DEVIATION_LOW \
		--noise_deviation_high=$NOISE_DEVIATION_HIGH \
		--encoder_loss_weight=$ENCODER_LOSS_WEIGHT \
		--decoder_loss_weight=$DECODER_LOSS_WEIGHT \
		--tr_num_samples=$tr_num_samples \
		--vl_num_samples=$VL_NUM_SAMPLES \
		--seed=$SEED \
		--outfile=$outfile \
		--num_train_epochs=$NUM_TRAIN_EPOCHS \
		--batch_size=$BATCH_SIZE \
		--learning_rate=$LEARNING_RATE \
		--dataloader_num_workers=$DATA_LOADER_NUM_WORKERS \
		--device=$DEVICE > $logfile 2>&1
	done
done


# Experiments from section 5.2

FLOW_LENGTH=100
FINGERPRINT_LENGTH=4096
TR_NUM_SAMPLES=200000
NUM_TRAIN_EPOCHS=100

for sigma in "2 10" "10 20" "20 30"
do
	read -r noise_deviation_low noise_deviation_high <<< "$sigma"
	for amplitude in 5e-3 10e-3 20e-3 30e-3 40e-3
	do
		file="amplitude/noise_deviation_low--$noise_deviation_low---noise_deviation_high--$noise_deviation_high---amplitude--$amplitude"
		logfile="./logs/$file.log"
		outfile="./output/$file.jsonl"
		mkdir -p $(dirname "$stdout")
		mkdir -p $(dirname "$logfile")
		echo "Running experiment: $file"
		python -u src/finn.py \
		 --fingerprint_length=$FINGERPRINT_LENGTH \
		--flow_length=$FLOW_LENGTH \
		--min_flow_length=$FLOW_LENGTH \
		--max_flow_length=$MAXSIZE \
		--amplitude=$amplitude \
		--noise_deviation_low=$noise_deviation_low \
		--noise_deviation_high=$noise_deviation_high \
		--encoder_loss_weight=$ENCODER_LOSS_WEIGHT \
		--decoder_loss_weight=$DECODER_LOSS_WEIGHT \
		--tr_num_samples=$TR_NUM_SAMPLES \
		--vl_num_samples=$VL_NUM_SAMPLES \
		--seed=$SEED \
		--outfile=$outfile \
		--num_train_epochs=$NUM_TRAIN_EPOCHS \
		--batch_size=$BATCH_SIZE \
		--learning_rate=$LEARNING_RATE \
		--dataloader_num_workers=$DATA_LOADER_NUM_WORKERS \
		--device=$DEVICE > $logfile 2>&1
	done
done


# Experiments from section 5.3

FINGERPRINT_LENGTH=1024
AMPLITUDE=5e-3
NOISE_DEVIATION_LOW=2
NOISE_DEVIATION_HIGH=10

for tr_num_samples in 200000 500000
do
	for num_train_epochs in 100 200
	do
		for flow_length in 50 100 150
		do
			file="flow_length/tr_num_samples--$tr_num_samples---num_train_epochs--$num_train_epochs---flow_length--$flow_length"
			logfile="./logs/$file.log"
			outfile="./output/$file.jsonl"
			mkdir -p $(dirname "$stdout")
			mkdir -p $(dirname "$logfile")
			echo "Running experiment: $file"
			python -u src/finn.py \
			--fingerprint_length=$FINGERPRINT_LENGTH \
			--flow_length=$flow_length \
			--min_flow_length=$flow_length \
			--max_flow_length=$MAXSIZE \
			--amplitude=$AMPLITUDE \
			--noise_deviation_low=$NOISE_DEVIATION_LOW \
			--noise_deviation_high=$NOISE_DEVIATION_HIGH \
			--encoder_loss_weight=$ENCODER_LOSS_WEIGHT \
			--decoder_loss_weight=$DECODER_LOSS_WEIGHT \
			--tr_num_samples=$tr_num_samples \
			--vl_num_samples=$VL_NUM_SAMPLES \
			--seed=$SEED \
			--outfile=$outfile \
			--num_train_epochs=$num_train_epochs \
			--batch_size=$BATCH_SIZE \
			--learning_rate=$LEARNING_RATE \
			--dataloader_num_workers=$DATA_LOADER_NUM_WORKERS \
			--device=$DEVICE > $logfile 2>&1
		done
	done
done
