import logging
import os
import pickle
import time

import networkx as nx
from rdkit import Chem
import pandas as pd
from chebi_utils.downloader import download_chebi_obo, download_chebi_sdf
from chebi_utils.obo_extractor import build_chebi_graph
from chebi_utils.sdf_extractor import extract_molecules

class ChEBIData:

    def __init__(self, chebi_version: int):
        self.chebi_version = chebi_version

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, f"chebi_v{self.chebi_version}"), exist_ok=True)
        # chebi: dict with entries from chebi
        self.chebi = self.process_chebi()
        # processed: dataframe that combines chebi data with mols from sdf file
        self.processed = self.process_data()

    @property
    def base_dir(self):
        return "data"

    @property
    def chebi_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "chebi.obo")

    @property
    def chebi_dict_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "chebi_dict.pkl")

    @property
    def trans_hierarchy_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "trans_hierarchy.pkl")

    @property
    def sdf_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "chebi.sdf.gz")

    @property
    def processed_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "processed.pkl")

    def download_chebi(self) -> None:
        if not os.path.exists(self.chebi_path):
            logging.info(f"Downloading ChEBI v{self.chebi_version} obo file to {self.chebi_path}")
            download_chebi_obo(self.chebi_version, os.path.dirname(self.chebi_path),
                                os.path.basename(self.chebi_path))

    def process_chebi(self) -> dict:
        self.download_chebi()
        if not os.path.exists(self.chebi_dict_path):
            graph = build_chebi_graph(self.chebi_path, top_class=None)
            res = {
                int(node): {
                    "parents": [],
                    "name": attrs.get("name"),
                    "definition": attrs.get("definition"),
                    "smiles": attrs.get("smiles"),
                    "subset": attrs.get("subset"),
                }
                for node, attrs in graph.nodes(data=True)
            }
            for u, v, d in graph.edges(data=True):
                relation = d.get("relation")
                source, target = res[int(u)], int(v)
                if relation == "is_a":
                    source["parents"].append(target)
                else:
                    source.setdefault(relation, []).append(target)
            with open(self.chebi_dict_path, "wb") as f:
                pickle.dump(res, f)
            return res
        else:
            with open(self.chebi_dict_path, "rb") as f:
                return pickle.load(f)

    def download_sdf(self) -> None:
        if not os.path.exists(self.sdf_path):
            logging.info(f"Downloading ChEBI v{self.chebi_version} SDF data to {self.sdf_path}")
            download_chebi_sdf(self.chebi_version, os.path.dirname(self.sdf_path),
                                os.path.basename(self.sdf_path))

    def sdf_file_to_mol(self):
        self.download_sdf()
        molecules = extract_molecules(self.sdf_path)
        for _, row in molecules.iterrows():
            mol = row["mol"]
            # turn aromatic bond types into single/double
            try:
                Chem.Kekulize(mol)
            except Chem.KekulizeException as e:
                logging.debug(f"{Chem.MolToSmiles(mol)} - {e}")
            yield int(row["chebi_id"]), mol

    def process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.processed_path):
            res = {}
            for mol_id, mol in self.sdf_file_to_mol():
                if mol_id not in self.chebi.keys():
                    continue
                if "smiles" not in self.chebi[mol_id] or self.chebi[mol_id]["smiles"] is None:
                    # entries with mol but without smiles are usually [ ]n specifications
                    continue
                if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                    continue
                res[mol_id] = {"mol": mol, **self.chebi[mol_id]}
            df = pd.DataFrame.from_dict(res, orient="index")
            df.to_pickle(self.processed_path)
        else:
            df = pd.read_pickle(self.processed_path)
        return df

    def build_hierarchy_graph(self):
        logging.debug(f"Building hierarchy graph")
        start_time = time.perf_counter()
        g = nx.DiGraph()
        g.add_nodes_from(self.chebi.keys())
        for chebi_id, row in self.chebi.items():
            if "parents" in row:
                for parent in row["parents"]:
                    g.add_edge(parent, chebi_id)
        logging.debug(f"Built hierarchy graph in {time.perf_counter() - start_time:.2f} seconds")
        return g

    def get_trans_hierarchy(self):
        if not os.path.exists(self.trans_hierarchy_path):
            g = self.build_hierarchy_graph()
            with open(self.trans_hierarchy_path, "wb") as f:
                pickle.dump(nx.transitive_closure(g), f)
            return g
        with open(self.trans_hierarchy_path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    data = ChEBIData(chebi_version=239)
