## Data Download

#### 1. AdvTest dataset

```bash
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
cd ..
```

#### 2. CosQA dataset

```bash
mkdir cosqa && cd cosqa
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/code_idx_map.txt
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-dev-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-test-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-train-19604.json
cd ..
```

#### 3. CSN dataset

```bash
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh
cd ..

```