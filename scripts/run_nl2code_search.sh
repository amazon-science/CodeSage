#!/usr/bin/env bash

data_dir=data/nl2code
result_dir="eval_results"

export model_name_or_path=${1:-"codesage-small"} # codesage-small, codesage-base, codesage-large
export dataset=${2:-"cosqa"}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

function advTest() {
    python3 -W ignore evaluation/nl2code_search.py \
        --dataset_name advtest \
        --model_name_or_path "codesage/$model_name_or_path" \
        --result_dir $result_dir/advtest \
        --test_data_file $data_dir/AdvTest/test.jsonl \
        --codebase_file $data_dir/AdvTest/test.jsonl \
        --code_length 1024 \
        --nl_length 128 \
        --eval_batch_size 256 \
        --seed 123456
}

function cosqa() {
    python3 -W ignore evaluation/nl2code_search.py \
        --dataset_name cosqa \
        --model_name_or_path "codesage/$model_name_or_path" \
        --result_dir $result_dir/cosqa \
        --test_data_file $data_dir/cosqa/cosqa-retrieval-test-500.json \
        --codebase_file $data_dir/cosqa/code_idx_map.txt \
        --code_length 1024 \
        --nl_length 128 \
        --eval_batch_size 128 \
        --seed 123456
}

function csn() {
    for lang in python java ruby php javascript go; do
        python3 -W ignore evaluation/nl2code_search.py \
            --dataset_name csn \
            --language $lang \
            --model_name_or_path "codesage/$model_name_or_path" \
            --result_dir $result_dir/CSN \
            --test_data_file $data_dir/CSN/$lang/test.jsonl \
            --codebase_file $data_dir/CSN/$lang/codebase.jsonl \
            --code_length 1024 \
            --nl_length 128 \
            --eval_batch_size 256 \
            --seed 123456
        wait
    done
}

if [[ $dataset == "advTest" ]]; then
    advTest
elif [[ $dataset == "cosqa" ]]; then
    cosqa
elif [[ $dataset == "csn" ]]; then
    csn
fi
