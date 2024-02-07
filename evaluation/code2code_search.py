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
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from utils import Code2CodeDataset
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


def get_embeddings(args, model, tokenizer, lang):
    code_data_file = f"{args.data_dir}/{lang}_with_func.jsonl"
    code_dataset = Code2CodeDataset(tokenizer, args, code_data_file, lang)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num Code Examples = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Obtain embedding vectors
    model.eval()
    code_vecs = []
    code_labels = []
    for batch in tqdm(code_dataloader):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            code_vec = torch.nn.functional.normalize(outputs.pooler_output, p=2, dim=1)
            code_vecs.append(code_vec.cpu().numpy())
            code_labels.append(label.cpu().numpy())

    # Calculate cosine score for in-language search
    code_vecs = np.concatenate(code_vecs, 0)
    code_labels = list(np.concatenate(code_labels, 0))
    code_indexs = [code_dataset.examples[i].index for i in range(len(code_dataset))]

    result = {
        "embeddings": code_vecs,
        "labels": code_labels,
        "indexs": code_indexs
    }
    return result


def get_map_score(query_vecs, query_labels, query_indexs, candidate_vecs, candidate_labels, candidate_indexs):
    # Calculate MAP score
    scores = np.matmul(query_vecs, candidate_vecs.T)  # num_queries x num_candidates
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(query_labels[i])
        query_index = query_indexs[i]

        Avep = []
        for j, index in enumerate(list(sort_ids[i])):
            if query_index == candidate_indexs[index]:
                cont += 1
                continue
            if int(candidate_labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1 - cont))
        if len(Avep) != 0:
            MAP.append(sum(Avep) / len(Avep))

    map_score = float(np.mean(MAP))
    return map_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--result_dir", default=None, type=str)
    parser.add_argument("--src_lang", default="python", type=str, required=True)
    parser.add_argument("--tgt_lang", default="python", type=str, required=True)
    parser.add_argument("--code_length", default=256, type=int)
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(42)
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    ##########Evaluation###########
    embedding_path = os.path.join(args.result_dir, args.model_name_or_path, "embeddings")
    eval_path = os.path.join(args.result_dir, args.model_name_or_path, "search_results")
    Path(embedding_path).mkdir(parents=True, exist_ok=True)
    Path(eval_path).mkdir(parents=True, exist_ok=True)

    # load model
    print(f"Loading {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.src_lang == args.tgt_lang:
        embed_path = os.path.join(embedding_path, f"{args.src_lang}_embeddings.npy")
        if os.path.exists(embed_path):
            embd_dict = np.load(embed_path, allow_pickle=True).tolist()
        else:
            embd_dict = get_embeddings(args, model, tokenizer, args.src_lang)
            np.save(os.path.join(embedding_path, f"{args.src_lang}_embeddings.npy"), embd_dict)

        src_embd_dict = embd_dict
        tgt_embd_dict = embd_dict

    else:
        src_lang_embed_path = os.path.join(embedding_path, f"{args.src_lang}_embeddings.npy")
        if os.path.exists(src_lang_embed_path):
            src_embd_dict = np.load(src_lang_embed_path, allow_pickle=True).tolist()
        else:
            src_embd_dict = get_embeddings(args, model, tokenizer, args.src_lang)
            np.save(os.path.join(embedding_path, f"{args.src_lang}_embeddings.npy"), src_embd_dict)

        tgt_lang_embed_path = os.path.join(embedding_path, f"{args.tgt_lang}_embeddings.npy")
        if os.path.exists(tgt_lang_embed_path):
            tgt_embd_dict = np.load(tgt_lang_embed_path, allow_pickle=True).tolist()
        else:
            tgt_embd_dict = get_embeddings(args, model, tokenizer, args.tgt_lang)
            np.save(os.path.join(embedding_path, f"{args.tgt_lang}_embeddings.npy"), tgt_embd_dict)

    eval_map_score = get_map_score(
        query_vecs=src_embd_dict["embeddings"],
        query_labels=src_embd_dict["labels"],
        query_indexs=src_embd_dict["indexs"],
        candidate_vecs=tgt_embd_dict["embeddings"],
        candidate_labels=tgt_embd_dict["labels"],
        candidate_indexs=tgt_embd_dict["indexs"]
    )
    logger.info("  %s to %s, %s", args.src_lang, args.tgt_lang, str(round(eval_map_score * 100, 2)))


if __name__ == "__main__":
    main()
