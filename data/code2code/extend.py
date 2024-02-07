# Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
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
import glob
import re
import csv
import json
import random
from tqdm import tqdm

"""
Download Project_CodeNet.tar.gz and Project_CodeNet_metadata.tar.gz from https://github.com/IBM/Project_CodeNet/tree/main#relevant-links.
Then uncompress both the tar.gz files.
Then run this script to extract the submissions.
"""

LANG_FNAME_MAP = {
    "C": "c",
    "C++": "cpp",
    "C#": "csharp",
    "Go": "go",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "PHP": "php"
}

FILE_EXTENSIONS = {
    "C": "c",
    "C++": "cpp",
    "C#": "cs",
    "Go": "go",
    "JavaScript": "js",
    "TypeScript": "ts",
    "PHP": "php"
}


def main(problem_ids):
    codenet_dir = "Project_CodeNet/data"
    metadata_dir = "Project_CodeNet/metadata"

    print("Total problem's solution to collect - ", len(problem_ids))
    for lang in FILE_EXTENSIONS.keys():
        json_data = []
        json_with_func = []
        for pid in tqdm(problem_ids):
            # project codenet cluster information format
            # https://github.com/IBM/Project_CodeNet/blob/main/tools/duplicates/Clusters.md#assumptions--definitions
            sub_id_to_cluster = {}
            total_cluster = 0
            cluster_file = f"Project_CodeNet/derived/duplicates/{lang}/clusters/{pid}-clusters"
            if os.path.exists(cluster_file):
                content_lines = open(cluster_file).read().split("\n")
                for line in content_lines:
                    if len(line.strip()) == 0:
                        total_cluster += 1
                    else:
                        sub_path = line.strip().split(":")[0]
                        sub_id = os.path.splitext(os.path.basename(sub_path))[0]
                        sub_id_to_cluster[sub_id] = total_cluster

            csv_filename = os.path.join(metadata_dir, f"{pid}.csv")
            selected_submissions = []
            with open(csv_filename) as csvfile:
                header = next(csvfile)
                reader = csv.reader(csvfile)
                used_cluster = set()
                for row in reader:
                    # "Project_CodeNet/derived/duplicates/C/clusters/p02839-clusters":
                    # we make sure two submissions do not belong to the same duplicate cluster
                    if row[5] == lang:
                        continue
                    submission_id = row[0]
                    assert submission_id.startswith("s")
                    if submission_id in sub_id_to_cluster:
                        if sub_id_to_cluster[submission_id] in used_cluster:
                            continue
                        used_cluster.add(sub_id_to_cluster[submission_id])
                    if row[7] == "Accepted":
                        selected_submissions.append(submission_id)

            # print(f"[{lang}]: {pid} - #clusters={total_cluster}, #submissions={len(selected_submissions)}")
            random.shuffle(selected_submissions)
            sub_for_this_pid = 0
            for submission_id in selected_submissions:
                # e.g., Project_CodeNet/data/p00001/JavaScript/s300682070.js
                file_to_read = f"{codenet_dir}/{pid}/{lang}/{submission_id}.{FILE_EXTENSIONS[lang]}"
                if os.path.exists(file_to_read):
                    code = open(file_to_read).read()
                    json_data.append({"index": submission_id, "label": int(pid[1:]), "func": ""})
                    json_with_func.append({"index": submission_id, "label": int(pid[1:]), "func": code})
                    sub_for_this_pid += 1
                    if sub_for_this_pid >= 10:
                        break

        with open(f"{LANG_FNAME_MAP[lang]}.jsonl", "w") as fw:
            fw.write("\n".join([json.dumps(s) for s in json_data]))
        with open(f"{LANG_FNAME_MAP[lang]}_with_func.jsonl", "w") as fw:
            fw.write("\n".join([json.dumps(s) for s in json_with_func]))


if __name__ == "__main__":
    problem_ids = set()
    label2id = {}
    with open("java.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            if ex['label'] not in label2id:
                label2id[ex['label']] = []
            label2id[ex['label']].append(ex['index'])
            problem_ids.add(f"p{str(ex['label']).zfill(5)}")

    # print(max([len(v) for k, v in label2id.items()]))
    main(problem_ids)
