import os
from typing import Literal

def get_exs_path(chebi_id, base_dir=None, split:Literal["train", "validation", "test"]="train"):
    if base_dir is None:
        base_dir = os.path.join("ilp", "chebi_v244")
    filename = f"exs_{split}.pl" if split in ["train", "validation"] else f"exs.pl"
    os.makedirs(os.path.join(base_dir, f"chebi_{chebi_id}"), exist_ok=True)
    return os.path.join(base_dir, f"chebi_{chebi_id}", filename)

def get_bk_path(chebi_id, base_dir=None, predicate_set="atoms", split:Literal["train", "validation", "test"]="train"):
    if base_dir is None:
        base_dir = os.path.join("ilp", "chebi_v244")
    if split == "train":
        bk_dir = os.path.join(base_dir, f"chebi_{chebi_id}", predicate_set)
        bk_file = f"bk.pl"
    else:
        bk_dir = os.path.join(base_dir, predicate_set)
        bk_file = f"bk_{split}.pl"
    os.makedirs(bk_dir, exist_ok=True)
    return os.path.join(bk_dir, bk_file)

def get_bias_path(chebi_id, base_dir=None, predicate_set="atoms", selection_mode:Literal["claude", "random", "top_k"]|None=None, max_vars=None, max_body=None, max_clauses=None):
    if base_dir is None:
        base_dir = os.path.join("ilp", "chebi_v244")
    bk_dir = os.path.join(base_dir, f"chebi_{chebi_id}", predicate_set)
    if selection_mode:
        bias_file = f"bias_{selection_mode}"
    else:
        bias_file = f"bias"
    if max_vars is not None:
        bias_file += f"_max_vars={max_vars}"
    if max_body is not None:
        bias_file += f"_max_body={max_body}"
    if max_clauses is not None:
        bias_file += f"_max_clauses={max_clauses}"
    bias_file += ".pl"
    os.makedirs(bk_dir, exist_ok=True)
    return os.path.join(bk_dir, bias_file)