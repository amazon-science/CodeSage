#!/usr/bin/env bash

data_dir=data/code2code
result_dir="eval_results"

export model_name_or_path=${1:-"codesage-small"} # codesage-small, codesage-base, codesage-large
export src_lang=${2:-"python"}
export tgt_lang=${3:-"python"}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -W ignore evaluation/code2code_search.py \
    --model_name_or_path "codesage/$model_name_or_path" \
    --data_dir $data_dir \
    --result_dir $result_dir \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --code_length 1024 \
    --eval_batch_size 128
