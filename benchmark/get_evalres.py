import os
import sys
import jsonlines
import pandas as pd
import numpy as np


def eval_code2code(language, query_path, candidate_path):
    query_data = list(jsonlines.open(query_path))
    candidate_data = list(jsonlines.open(candidate_path))

    query_vecs = [item['dense_embedding'] for item in query_data]
    query_labels = [item['label'] for item in query_data]
    query_indexs = [item['index'] for item in candidate_data]

    candidate_vecs = [item['dense_embedding'] for item in candidate_data]
    candidate_labels = [item['label'] for item in candidate_data]
    candidate_indexs = [item['index'] for item in candidate_data]

    # Calculate cosine score
    query_vecs_array = np.array([np.array(xi) for xi in query_vecs])
    candidate_vecs_array = np.array([np.array(xi) for xi in candidate_vecs])
    scores = np.matmul(query_vecs_array, candidate_vecs_array.T)  # num_queries x num_candidates
    print(query_vecs_array.shape, candidate_vecs_array.shape, scores.shape)

    # Calculate MAP score
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    MAP = []
    results = {}
    for i in range(scores.shape[0]):
        cont = 0
        label = int(query_labels[i])
        query_index = query_indexs[i]
        results[query_index] = {}

        Avep = []
        for j, index in enumerate(list(sort_ids[i])):
            if query_index == candidate_indexs[index]:
                cont += 1
                continue
            if int(candidate_labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1 - cont))
        if len(Avep) != 0:
            MAP.append(sum(Avep) / len(Avep))
            results[query_index]["map"] = sum(Avep) / len(Avep)

    result = {
        "eval_map": float(np.mean(MAP))
    }
    print(language, result)
    return result



def eval_nl2code(dataname, query_path, candidate_path):
    query_data = list(jsonlines.open(query_path))
    candidate_data = list(jsonlines.open(candidate_path))

    query_vecs = [item['dense_embedding'] for item in query_data]
    query_labels = [item['url'] if "url" in item else item["retrieval_idx"] for item in query_data]

    candidate_vecs = [item['dense_embedding'] for item in candidate_data]
    candidate_labels = [item['url'] if "url" in item else item["retrieval_idx"] for item in candidate_data]


    # Calculate cosine score
    query_vecs_array = np.array([np.array(xi) for xi in query_vecs])
    candidate_vecs_array = np.array([np.array(xi) for xi in candidate_vecs])
    scores = np.matmul(query_vecs_array, candidate_vecs_array.T)  # num_queries x num_candidates
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    print(f"query_vecs: {query_vecs_array.shape} \t candidate_vecs: {candidate_vecs_array.shape} \t map_scores: {scores.shape}")

    ranks = []
    for url, sort_id in zip(query_labels, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if candidate_labels[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }
    print(dataname, result)

    return result






