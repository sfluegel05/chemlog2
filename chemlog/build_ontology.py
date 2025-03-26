import os.path

import tqdm
import pandas as pd
import pyhornedowl
from pyhornedowl.model import SubClassOf
import subprocess

from chemlog.preprocessing.chebi_data import ChEBIData


def build_ontology_from_results(chebi_version, results_path):
    chebi_data = ChEBIData(chebi_version)
    # convert ChEBI to OWL
    chebi_path = chebi_data.chebi_path.replace(".obo", ".owl")
    if not os.path.exists(chebi_path):
        subprocess.run(
            ["robot", "convert", "-i", chebi_data.chebi_path, "-o", chebi_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    chebi_onto = pyhornedowl.open_ontology(chebi_path)
    results = pd.read_json(results_path)
    trans_hierarchy = chebi_data.get_trans_hierarchy()
    chebi_onto.add_prefix_mapping("", "http://purl.obolibrary.org/obo/")
    for _, row in tqdm.tqdm(results.iterrows()):
        chebi_id = row["chebi_id"]
        preds = row["chebi_classes"]
        for pred in preds:
            if not any(pred in trans_hierarchy.predecessors(p) for p in preds):
                chebi_onto.add_axiom(SubClassOf(chebi_onto.clazz(f":{chebi_id}"), chebi_onto.clazz(f":{pred}")))
    chebi_onto.save_to_file(chebi_path.replace(".owl", "_modified.owl"))

if __name__ == "__main__":
    build_ontology_from_results(239, "results/250203_1258/classify.json")