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

import re
import json
import torch
import tokenize
import logging
from io import StringIO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


class Code2CodeInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
            self,
            input_ids,
            index,
            label
    ):
        self.input_ids = input_ids
        self.index = index
        self.label = label


def convert_code2code_examples_to_features(js, tokenizer, args, lang):
    """convert examples to token ids"""

    if "func" in js:
        code = " ".join(remove_comments_and_docstrings(js['func'], lang).split())
    else:
        code = " ".join(remove_comments_and_docstrings(js['code'], lang).split())

    token_feat = tokenizer(code,
                           max_length=args.code_length,
                           padding="max_length",
                           truncation=True)
    input_ids = token_feat["input_ids"]
    padding_length = args.code_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length

    return Code2CodeInputFeatures(input_ids, js["index"], int(js['label']))


class Code2CodeDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, lang):
        self.examples = []
        data = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                data.append(js)

        for js in data:
            self.examples.append(convert_code2code_examples_to_features(js, tokenizer, args, lang))

        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label))


class NL2CodeInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
            self,
            code_ids,
            nl_ids,
            url,
    ):
        self.code_ids = code_ids
        self.nl_ids = nl_ids
        self.url = url


def convert_nl2code_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""

    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    token_feat = tokenizer(code,
                           max_length=args.code_length,
                           padding="max_length",
                           truncation=True)
    code_ids = token_feat["input_ids"]
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_feat = tokenizer(nl,
                        max_length=args.nl_length,
                        padding="max_length",
                        truncation=True)
    nl_ids = nl_feat["input_ids"]
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return NL2CodeInputFeatures(
        code_ids,
        nl_ids,
        js['url'] if "url" in js else js["retrieval_idx"]
    )


class NL2CodeDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, lang="python"):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            self.examples.append(convert_nl2code_examples_to_features(js, tokenizer, args))

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))