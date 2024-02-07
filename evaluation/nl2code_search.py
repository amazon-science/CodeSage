# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from utils import NL2CodeDataset
from transformers import AutoTokenizer, AutoConfig, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer, file_name):
    query_dataset = NL2CodeDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4
    )

    code_dataset = NL2CodeDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(
        code_dataset,
        sampler=code_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4
    )

    logger.info("***** Running evaluation *****")
    logger.info("Num queries = %d", len(query_dataset))
    logger.info("Num codes = %d", len(code_dataset))
    logger.info("Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []

    for batch in tqdm(code_dataloader):
        code_inputs = batch[0].to(args.device)
        attention_mask = code_inputs.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model(input_ids=code_inputs, attention_mask=attention_mask, return_dict=True)
            code_vec = torch.nn.functional.normalize(outputs.pooler_output, p=2, dim=1)
            code_vecs.append(code_vec.cpu().numpy())

    for batch in tqdm(query_dataloader):
        nl_inputs = batch[1].to(args.device)
        attention_mask = nl_inputs.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model(input_ids=nl_inputs, attention_mask=attention_mask, return_dict=True)
            nl_vec = torch.nn.functional.normalize(outputs.pooler_output, p=2, dim=1)
            nl_vecs.append(nl_vec.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    print(f"nl_vecs_shape: {nl_vecs.shape} "
          f"\t code_vecs_shape: {code_vecs.shape} "
          f"\t score_matrix_shape: {scores.shape}")

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks)),
        "nl_embeddings": nl_vecs,
        "code_embeddings": code_vecs,
        "nl_urls": nl_urls,
        "code_urls": code_urls
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_name", default=None, type=str,
                        help="the name of the nl2code eval set")
    parser.add_argument("--language", default="python", type=str,
                        help="the name of the nl2code eval set")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--result_dir", required=True, type=str, help="path to store the evaluation results.")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args.seed)
    logger.info("device: %s, n_gpu: %s, seed: %s", device, args.n_gpu, args.seed)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Evaluation
    result = evaluate(args, model, tokenizer, args.test_data_file)
    logger.info("***** Eval results *****")
    logger.info("EVAL_MRR = %s", str(round(result["eval_mrr"] * 100, 2)))

    # save results
    model_dir_name = args.model_name_or_path.split("/")[-1]
    result_dir = f"{args.result_dir}/{model_dir_name}"
    from pathlib import Path
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(result_dir, f"{args.dataset_name}_{args.language}.npy"), result)


if __name__ == "__main__":
    main()
