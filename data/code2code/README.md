## Zero-shot Code-to-Code Search

**Task**: Given a source code as the query, the task aims to retrieve codes with the same semantics from a collection of
candidates in zero-shot setting.

[Guo et al., 2022](https://aclanthology.org/2022.acl-long.499.pdf) collected 11,744/15,594/23,530 programs
from [CodeNet corpus](https://ml4code.github.io/publications/puri2021project/) for Ruby/Python/Java programming
languages. Each program solves one of 4,053 problems. The task is to take each program as the query and retrieve
programs that solves the same problem from each PL.

### Data

Get the data (jsonl files)
from https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/zero-shot-search/dataset.

### Statistics

#### Java

- Total examples - 23,530
- Total programming problems - 3,142

#### Python

- Total examples in Python - 15,594
- Total programming problems - 2,072

#### Ruby

- Total examples in Ruby - 11,744
- Total programming problems - 1,708

#### Overlapped problems

- Java, Python - 2,001
- Python, Ruby - 1,566
- Java, Ruby - 1,695
- Java, Python, Ruby - 1,560

### Dataset Extension

We extend this dataset to 3 other programming languages - C, C++, C#, Go, Javascript, Typescript, and PHP.

```
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_metadata.tar.gz
tar -xvzf Project_CodeNet.tar.gz
tar -xvzf Project_CodeNet_metadata.tar.gz
python extend.py
```

#### Total examples

- C - 11,260
- C++ - 30,197
- C# - 11,952
- Go - 9,720
- Javascript - 6,866
- Typescript - 3,385
- PHP - 6,782
