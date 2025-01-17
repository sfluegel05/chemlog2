import gzip
import json
import logging
import os
import requests
import fastobo
from rdkit import Chem
import pandas as pd

class ChEBIData:

    def __init__(self, chebi_version: int):
        self.chebi_version = chebi_version

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, f"chebi_v{self.chebi_version}"), exist_ok=True)
        self.processed = self.process_data()

    @property
    def base_dir(self):
        return "data"

    @property
    def chebi_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "chebi.obo")

    @property
    def sdf_path(self):
        # sdf files are not versioned
        return os.path.join(self.base_dir, f"ChEBI_complete.sdf")

    @property
    def processed_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "processed.pkl")

    def download_chebi(self) -> None:
        if not os.path.exists(self.chebi_path):
            url = f"http://purl.obolibrary.org/obo/chebi/{self.chebi_version}/chebi.obo"
            logging.info(f"Downloading ChEBI from {url}")
            r = requests.get(url)
            if r.status_code != 200:
                logging.error(f"Failed to download ChEBI from {url}")
                raise Exception(f"Got response {r.status_code}: {r.content}")
            open(self.chebi_path, "wb").write(r.content)

    def process_chebi(self) -> dict:
        with open(self.chebi_path, encoding="utf-8") as chebi_raw:
            chebi = "\n".join(l for l in chebi_raw if not l.startswith("xref:"))
        res = {}
        for term in fastobo.loads(chebi):
            if term and ":" in str(term.id) and not any(
                    [clause.raw_tag() == "is_obsolete" and clause.raw_value() == "true" for clause in term]):
                term_id, term_properties = term_callback(term)
                res[term_id] = term_properties
        return res

    def download_sdf(self) -> None:
        if not os.path.exists(self.sdf_path):
            url = "https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/ChEBI_complete.sdf.gz"
            logging.info(f"Downloading ChEBI SDF data from {url}")
            r = requests.get(url)
            if r.status_code != 200:
                logging.error(f"Failed to download ChEBI SDF data from {url}")
                raise Exception(f"Got response {r.status_code}: {r.content}")
            open(self.sdf_path, "wb").write(gzip.decompress(r.content))

    def sdf_file_to_mol(self):
        supplier = Chem.SDMolSupplier(self.sdf_path, removeHs=False, strictParsing=False, sanitize=False)
        res = {}
        for mol in supplier:
            if mol is not None:
                # turn aromatic bond types into single/double
                try:
                    Chem.Kekulize(mol)
                except Chem.KekulizeException as e:
                    logging.debug(f"{Chem.MolToSmiles(mol)} - {e}")
                res[chebi_to_int(mol.GetProp("ChEBI ID"))] = mol
        return res

    def process_data(self) -> pd.DataFrame:
        if not os.path.exists(self.processed_path):
            self.download_chebi()
            chebi = self.process_chebi()
            self.download_sdf()
            mols = self.sdf_file_to_mol()
            res = {}
            for mol_id, mol in mols.items():
                if mol_id in chebi.keys():
                    res[mol_id] = {"mol": mol, **chebi[mol_id]}
            df = pd.DataFrame.from_dict(res, orient="index")
            df.to_pickle(self.processed_path)
        else:
            df = pd.read_pickle(self.processed_path)
        return df


def chebi_to_int(s):
    return int(s[s.index(":") + 1:]) if ":" in s else s


def term_callback(doc) -> (int, dict):
    relationships = {}
    parents = []
    name = None
    smiles = None
    definition = None
    subset = None
    for clause in doc:
        if isinstance(clause, fastobo.term.PropertyValueClause):
            t = clause.property_value
            if str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles":
                assert smiles is None
                smiles = t.value
        elif isinstance(clause, fastobo.term.RelationshipClause):
            # e.g. has functional parent
            if str(clause.typedef) in relationships:
                relationships[str(clause.typedef)].append(
                    chebi_to_int(str(clause.term))
                )
            else:
                relationships[str(clause.typedef)] = [chebi_to_int(str(clause.term))]
        elif isinstance(clause, fastobo.term.IsAClause):
            parents.append(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.DefClause):
            definition = clause.definition
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)
        elif isinstance(clause, fastobo.term.SubsetClause):
            subset = str(clause.subset)
    return chebi_to_int(str(doc.id)), {
        **relationships,
        "parents": parents,
        "name": name,
        "definition": definition,
        "smiles": smiles,
        "subset": subset,
    }


if __name__ == "__main__":
    data = ChEBIData(chebi_version=239)
