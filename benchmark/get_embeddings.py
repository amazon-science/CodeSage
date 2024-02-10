import os
import time
import json
import jsonlines

import voyageai
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")
# embedding_model_id = "text-embedding-ada-002"
openai_model_id = "text-embedding-3-large"

SLEEP_SECOND = 0.8
MAX_SLEEP_SECOND = 5


def get_embedding_single(text, model_name):
    text = text.replace("\n", " ")  # This is required for the 001/003 model
    return openai.Embedding.create(input=[text], model=openai_model_id)['data'][0]['embedding']


def get_code2code_search_embeddings(model_name, language, datadir="./tmp/", resdir="./tmp/"):
    src_path=os.path.join(datadir, "code2code", f"{language}_with_func.jsonl")
    tgt_path=os.path.join(resdir, model_name, "code2code", f"code2code_embd_{language}.jsonl")

    raw_data = list(jsonlines.open(src_path))
    print(f"{language}: {len(raw_data)} examples")

    data_w_embd = []
    sleep_second = 2.5
    for idx, entry in enumerate(raw_data):
        entry["example_idx"] = idx
        retry = True
        retry_time = 0

        while retry and retry_time <= 3:
            try:
                entry['dense_embedding'] = \
                    get_embedding_single(entry["func"], model_name=model_name)
                data_w_embd.append(entry)
                time.sleep(SLEEP_SECOND)
                retry = False
                print(language, idx, retry, len(entry["dense_embedding"]))
            except:
                time.sleep(sleep_second)
                if sleep_second < MAX_SLEEP_SECOND:
                    sleep_second *= 2
                else:
                    sleep_second = MAX_SLEEP_SECOND
                retry_time += 1

    print(f"raw_data/embedding: {len(raw_data)}/{len(data_w_embd)}")
    with open(tgt_path, "w") as f:
        for item in data_w_embd:
            f.write(json.dumps(item))
            f.write("\n")
    f.close()


def get_nl2code_embeddings(model_name, dataname, language="python", datadir="./tmp/", resdir="./tmp/"):

    ## Load Data
    if dataname in ["AdvTest", "CSN"]:
        if dataname == "AdvTest":
            query_path = os.path.join(datadir, "nl2code", "AdvTest", "test.jsonl")
            candidate_path = os.path.join(datadir, "nl2code", "AdvTest", "test.jsonl")
            tgt_query_path = os.path.join(datadir, model_name, "nl2code", "AdvTest_query_embd.jsonl")
            tgt_candidate_path = os.path.join(datadir, model_name, "nl2code", "AdvTest_candidate_embd.jsonl")
            code_key = "function_tokens"
        else:
            query_path = os.path.join(datadir, "nl2code", "CSN", language, "test.jsonl")
            candidate_path = os.path.join(datadir, "nl2code", "CSN", language, "codebase.jsonl")
            tgt_query_path = os.path.join(datadir, model_name, "nl2code", f"CSN_query_embd_{language}.jsonl")
            tgt_candidate_path = os.path.join(datadir, model_name, "nl2code", f"CSN_candidate_embd_{language}.jsonl")
            code_key = "code_tokens"

        query_data = list(jsonlines.open(query_path))
        candidate_data = list(jsonlines.open(candidate_path))
    else:
        query_path = os.path.join(datadir, "nl2code", "cosqa", "cosqa-retrieval-test-500.json")
        query_data = []
        with open(query_path) as f:
            for js in json.load(f):
                query_data.append(js)

        candidate_path = os.path.join(datadir, "nl2code", "cosqa", "code_idx_map.txt")
        candidate_data = []
        with open(candidate_path) as f:
            js = json.load(f)
            for key in js:
                temp = {}
                temp['code_tokens'] = key.split()
                temp["retrieval_idx"] = js[key]
                temp['doc'] = ""
                temp['docstring_tokens'] = ""
                candidate_data.append(temp)

        tgt_query_path = os.path.join(resdir, model_name, "nl2code", "cosqa_query_embd.jsonl")
        tgt_candidate_path = os.path.join(resdir, model_name, "nl2code", "cosqa_candidate_embd.jsonl")
        code_key = "code_tokens"

    print(f"{dataname}/{language} with {len(query_data)} queries and {len(candidate_data)} candidates")

    ## Get Query Embedding
    sleep_second = 2.5
    query_embd = []
    for idx, entry in enumerate(query_data):
        entry["example_idx"] = idx
        docstring = ' '.join(entry['docstring_tokens']) if type(entry['docstring_tokens']) is list else ' '.join(entry['doc'].split())

        retry = True
        retry_time = 0
        while retry and retry_time <= 3:
            try:
                entry['dense_embedding'] = \
                    get_embedding_single(docstring, model_name=model_name)
                query_embd.append(entry)
                time.sleep(SLEEP_SECOND)
                retry = False
                print(language, idx, retry, len(entry["dense_embedding"]))
            except:
                time.sleep(sleep_second)
                sleep_second = sleep_second * 2 if sleep_second < MAX_SLEEP_SECOND else MAX_SLEEP_SECOND
                retry_time += 1
    print(f"{language}: raw_queries/embedding: {len(query_data)}/{len(query_embd)}")

    with open(tgt_query_path, "w") as f:
        for item in query_embd:
            f.write(json.dumps(item))
            f.write("\n")
    f.close()

    ## Get Code Embedding
    sleep_second = 2.5
    candidate_embd = []
    for idx, entry in enumerate(candidate_data):
        entry["example_idx"] = idx
        function = ' '.join(entry[code_key]) if type(entry[code_key]) is list else ' '.join(entry[code_key].split())

        retry = True
        retry_time = 0
        while retry and retry_time <= 3:
            try:
                entry['dense_embedding'] = \
                    get_embedding_single(function, model_name=model_name)
                candidate_embd.append(entry)
                time.sleep(SLEEP_SECOND)
                retry = False
                print(language, idx, retry, len(entry["dense_embedding"]))
            except:
                time.sleep(sleep_second)
                sleep_second = sleep_second * 2 if sleep_second < MAX_SLEEP_SECOND else MAX_SLEEP_SECOND
                retry_time += 1
    print(f"{language}: raw_candidates/embedding: {len(candidate_data)}/{len(candidate_embd)}")

    with open(tgt_candidate_path, "w") as f:
        for item in candidate_embd:
            f.write(json.dumps(item))
            f.write("\n")
    f.close()











