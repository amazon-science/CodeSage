#!/usr/bin/env bash

export model_name_or_path=${1:-"codesage-small"} # codesage-small, codesage-base, codesage-large

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -W ignore evaluation/clone_detection.py \
    --model_name_or_path "codesage/$model_name_or_path" \
    --max_seq_length 1024 \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --per_gpu_batch_size 4 \
    --seed 42 \
    --output_dir ./eval_results/clone_detection
