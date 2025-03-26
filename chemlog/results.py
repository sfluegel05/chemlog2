import json
import os
import random

import tqdm
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

from chemlog.preprocessing.chebi_data import ChEBIData

LABEL = [24866, 25696, 25697, 27369, 60334, 60194, 60466, 90799, 155837, 16670, 25676, 46761, 47923, 48030, 48545,
         15841]


def compare_results_with_chebi(results_dir, data):
    results = pd.read_json(os.path.join(results_dir, "results.json"))
    print("Getting transitive closure")
    trans_hierarchy = data.get_trans_hierarchy()
    processed = data.processed
    eval = {}
    eval_3star = {}
    for label in LABEL:
        eval[label] = {"tps": 0, "fps": 0, "fns": 0}
        eval_3star[label] = {"tps": 0, "fps": 0, "fns": 0}
    for _, row in tqdm.tqdm(results.iterrows()):
        for label in LABEL:
            pos_label = row["chebi_id"] in trans_hierarchy.successors(label)
            pos_pred = label in row["chebi_classes"]
            if pos_label and pos_pred:
                eval[label]["tps"] += 1
                eval_3star[label]["tps"] += 1 * (processed.loc[row["chebi_id"], "subset"] == "3_STAR")
            elif pos_label and not pos_pred:
                eval[label]["fns"] += 1
                eval_3star[label]["fns"] += 1 * (processed.loc[row["chebi_id"], "subset"] == "3_STAR")
            elif not pos_label and pos_pred:
                eval[label]["fps"] += 1
                eval_3star[label]["fps"] += 1 * (processed.loc[row["chebi_id"], "subset"] == "3_STAR")

    eval["micro"] = {score: sum(r[score] for r in eval.values()) for score in ["tps", "fps", "fns"]}
    eval_3star["micro"] = {score: sum(r[score] for r in eval_3star.values()) for score in ["tps", "fps", "fns"]}
    with open(os.path.join(results_dir, "eval.json"), "w") as f:
        json.dump(eval, f)
    with open(os.path.join(results_dir, "eval_3star.json"), "w") as f:
        json.dump(eval_3star, f)
    print("Results (complete ChEBI):")
    md_print_eval(eval, data)
    print("Results (3-STAR):")
    md_print_eval(eval_3star, data)


def html_color_bool(value):
    return f'<span style="background:#{"087024" if value else "8a1308"};color:white">{value}</span>'


def compare_2_runs(results_dir1, results_dir2, data):
    results1 = pd.read_json(os.path.join(results_dir1, "results.json"))
    results2 = pd.read_json(os.path.join(results_dir2, "results.json"))
    trans_hierarchy = data.get_trans_hierarchy()
    processed = data.processed
    print(f"| ID | name | label | ChEBI | Run 1 | Run 2 |")
    print(f"| --- | --- | --- | --- | --- | --- |")
    for (_, row1), (_, row2) in zip(results1.iterrows(), results2.iterrows()):
        if row1["chebi_id"] != row2["chebi_id"]:
            raise Exception("Results are not aligned")
        if processed.loc[row1["chebi_id"], "subset"] != "3_STAR":
            continue
        for label in LABEL:
            pos_label = row1["chebi_id"] in trans_hierarchy.successors(label)
            pos_pred1 = label in row1["chebi_classes"]
            pos_pred2 = label in row2["chebi_classes"]
            if pos_pred1 != pos_pred2:
                print(f"| {row1['chebi_id']} | {processed.loc[int(row1['chebi_id']), 'name']} | {label} | "
                      f"{html_color_bool(pos_label)} | {html_color_bool(pos_pred1)} | {html_color_bool(pos_pred2)} |")


def md_print_eval(eval_dict: dict, data: ChEBIData):
    print("| Class | TPs | FPs | FNs | Precision | Recall | F1 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    eval_dict["micro"] = {score: sum(r[score] for r in eval_dict.values()) for score in ["tps", "fps", "fns"]}
    # labels for target classes are not stored in data.processed (because they don't have a mol),
    # retrieve from data.chebi instead (df for easier lookup)
    df_labels = pd.DataFrame.from_dict(data.chebi, orient="index")
    for cls in eval_dict:
        e = eval_dict[cls]
        label = f"{cls} ({df_labels.loc[int(cls), 'name']})" if cls != "micro" else cls
        print(f"| {label} | {e['tps']} | {e['fps']} | {e['fns']} "
              f"| {e['tps'] / (e['tps'] + e['fps']):.3f} | {e['tps'] / (e['tps'] + e['fns']):.3f} "
              f"| {e['tps'] / (e['tps'] + 0.5 * (e['fps'] + e['fns'])):.3f}")


def sample_results(results_dir, target_cls: int, data: ChEBIData, n_samples=10, sample_target="fns"):
    results = pd.read_json(os.path.join(results_dir, "results.json"))
    trans_hierarchy = data.get_trans_hierarchy()
    print("| id | name | ChEBI | ours |")
    print("| --- | --- | --- | --- |")

    while n_samples > 0:
        i = random.randint(0, len(results) - 1)
        row = results.loc[i]
        id = int(row["chebi_id"])
        if not data.processed.loc[id, "subset"] == "3_STAR":
            continue
        pos_label = id in trans_hierarchy.successors(target_cls)
        pos_pred = target_cls in row["chebi_classes"]
        if (pos_label and not pos_pred and sample_target == "fns") or (
                pos_pred and not pos_label and sample_target == "fps") or (pos_pred and pos_label and sample_target == "tps"):
            n_samples -= 1
            # FN
            print(f"| {id} | {data.processed.loc[id, 'name']} | {target_cls} | "
                  f"{', '.join(str(c) for c in row['chebi_classes'])} |")

            plot_mol(data.processed.loc[id, "mol"])

def plot_mol(mol):
    options = Chem.Draw.MolDrawOptions()
    options.addAtomIndices = True
    im = Chem.Draw.MolToImage(
        mol,
        size=(600, 600),
        options=options,
        highlightAtoms=[],
    )
    plt.axis("off")
    plt.imshow(im)
    plt.show()

if __name__ == "__main__":
    #results_dir1 = os.path.join("results", "250121_1612_CXY")
    results_dir2 = os.path.join("results", "250121_1622_COY")
    data = ChEBIData(239)
    # md_print_eval(json.load(open(os.path.join(results_dir, "eval.json"), "rb")), data)
    #compare_2_runs(results_dir1, results_dir2, data)
    sample_results(results_dir2, 24866, data, sample_target="tps")
