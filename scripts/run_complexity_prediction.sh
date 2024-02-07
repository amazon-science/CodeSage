#!/usr/bin/env bash

export model_name_or_path=${1:-"codesage-small"} # codesage-small, codesage-base, codesage-large

declare -A LR_MAP
LR_MAP["codesage-small"]=1e-5
LR_MAP["codesage-base"]=1e-5
LR_MAP["codesage-large"]=5e-6

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -W ignore evaluation/complexity_prediction.py \
    --model_name_or_path "codesage/$model_name_or_path" \
    --max_seq_length 1024 \
    --learning_rate ${LR_MAP[${model_name_or_path}]} \
    --num_epochs 5 \
    --per_gpu_batch_size 4 \
    --seed 42 \
    --output_dir ./eval_results/complexity_prediction
