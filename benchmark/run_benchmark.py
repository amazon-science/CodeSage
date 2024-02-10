import os
import sys
from get_embeddings import get_nl2code_embeddings, get_code2code_search_embeddings
from get_evalres import eval_code2code, eval_nl2code

data_root = "parent-dir-of-nl2code-and-code2code-eval-data"

def benchmark_on_code2code_search(model_name, resdir):
    # get embeddings
    languages = ["typescript", "ruby", "python", "java", "javascript", "csharp", "c", "php", "go"]
    for language in languages:
        get_code2code_search_embeddings(model_name, language, data_root, resdir)

    # get evaluation results
    for language in languages:
        src_path = os.path.join(resdir, "code2code", f"code2code_embd_{language}.jsonl")
        eval_code2code(language, src_path, src_path)


def benchmark_on_nl2code_search(model_name, resdir):
    # get embeddings
    for dataname in ["cosqa", "AdvTest", "CSN"]:
        if dataname == "CSN":
            for language in ["python", "java", "javascript", "php", "go", "ruby"]:
                get_nl2code_embeddings(model_name, dataname, language=language, data_dir=data_root, res_dir=resdir)
        else:
            get_nl2code_embeddings(model_name, dataname, language="python", data_dir=data_root, res_dir=resdir)

    # get evaluation results
    for dataname in ["cosqa", "AdvTest", "CSN"]:
        if dataname in ["AdvTest", "cosqa"]:
            src_path = os.path.join(resdir, "nl2code", f"{dataname}_query_embd.jsonl")
            tgt_path = os.path.join(resdir, "nl2code", f"{dataname}_candidate_embd.jsonl")
            eval_nl2code(dataname, src_path, tgt_path)
            
        else: # CSN
            # for language in ["python", "java", "javascript", "php", "go", "ruby"]:
            for language in ["python", "java", "javascript", "php", "go", "ruby"]:
                src_path = os.path.join(resdir, "nl2code", f"{dataname}_query_embd_{language}.jsonl")
                tgt_path = os.path.join(resdir, "nl2code", f"{dataname}_candidate_embd_{language}.jsonl")

                eval_nl2code(dataname, src_path, tgt_path)



if __name__ == "__main__":

    model_name = "openai" # support cpt-code-001, ada-002, and text-embedding-3, minor changes required when switch between different versions
    resdir = os.path.join(data_root, model_name)

    benchmark_on_nl2code_search(model_name, resdir)
    benchmark_on_nl2code_search(model_name, resdir)





