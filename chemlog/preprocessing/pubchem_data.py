import gzip
import logging
import os
import pickle

import requests
from rdkit import Chem, RDLogger


class PubChemData:

    def __init__(self):

        os.makedirs(self.base_dir, exist_ok=True)
        RDLogger.DisableLog('rdApp.*')
        # self.download_sdf()

    @property
    def base_dir(self):
        return os.path.join("data", "pubchem")

    def get_sdf_batch_name(self, batch_index):
        return f"Compound_{batch_index * 500000 + 1:09d}_{(batch_index + 1) * 500000:09d}"

    def sdf_path(self, batch_index):
        return os.path.join(self.base_dir, f"{self.get_sdf_batch_name(batch_index)}.sdf")

    def download_sdf(self, batch_index) -> None:
        # 345 batches are available
        if not os.path.exists(self.sdf_path(batch_index)):
            url = f"https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/{self.get_sdf_batch_name(batch_index)}.sdf.gz"
            logging.info(f"Downloading PubChem SDF data from {url}")
            r = requests.get(url)
            if r.status_code != 200:
                logging.error(f"Failed to download ChEBI SDF data from {url}")
                raise Exception(f"Got response {r.status_code}: {r.content}")
            open(self.sdf_path(batch_index), "wb").write(gzip.decompress(r.content))

    def sdf_file_to_mol(self, batch_index):
        supplier = Chem.SDMolSupplier(self.sdf_path(batch_index), removeHs=False, strictParsing=False, sanitize=False)
        for mol in supplier:
            if mol is not None:
                # turn aromatic bond types into single/double
                try:
                    Chem.Kekulize(mol)
                except Chem.KekulizeException as e:
                    logging.warning(f"Failed kekulisation of {int(mol.GetProp('PUBCHEM_COMPOUND_CID'))}: {e}")
                yield int(mol.GetProp("PUBCHEM_COMPOUND_CID")), mol

    def get_processed_batch(self, batch_index):
        if not os.path.exists(self.sdf_path(batch_index)):
            logging.info(f"Downloading batch {batch_index}")
            self.download_sdf(batch_index)
        processed_path = os.path.join(self.base_dir, f"processed_{batch_index:03d}.pkl")
        if not os.path.exists(processed_path):
            logging.info(f"Processing batch {batch_index}")
            res = {}
            for mol_id, mol in self.sdf_file_to_mol(batch_index):
                res[mol_id] = mol
            with open(os.path.join(self.base_dir, f"processed_{batch_index:03d}.pkl"), "wb") as f:
                pickle.dump(res, f)
        return pickle.load(open(processed_path, "rb"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = PubChemData()
    mol_batch = data.get_processed_batch(0)
    print(mol_batch)
