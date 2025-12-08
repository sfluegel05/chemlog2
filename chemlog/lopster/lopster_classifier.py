

from chemlog.base_classifier import Classifier


lopster_chebi_mapping = {
    "astatineMolEntity": "37138",
    "bromineMolEntity": "22928", 
    "chlorineMolEntity": "23117", 
    "fluorineMolEntity": "24062",
    "iodineMolEntity": "24860", 
    "halogenMolEntity": "24471", 
    "poloniumMolEntity": "36917", 
    "seleniumMolEntity": "26628",
    "sulfurMolEntity": "26835",
    "telluriumMolEntity": "33305",
    "chalcogenMolEntity": "33304",
    "antimonyMolEntity": "36919",
    "arsenicMolEntity": "22632",
    "bismuthMolEntity": "37196",
    "nitrogenMolEntity": "51143",
    "pnictogenMolEntity": "33302",
    "chromiumMolEntity": "23237",
    "molybdenumMolEntity": "25370",
    "seaborgiumMolEntity": "37224",
    "tungstenMolEntity": "33742",
    "chromiumGroupMolEntity": "33741",
    "nobleGasMolEntity": "33583",
    "carbonMolEntity": "50860",
    "oxygenMolEntity": "25805",
    "hydrogenMolEntity": "33608",
    "phosphorusMolEntity": "26082",
    "inorganic": "24835",
    "hydroCarbon": "24632",
    "haloHydroCarbon": "24472",
    "polyatomic": "36357",
    "monoatomic": "33238",
    "carboxylicAcid": "33575",
    "carboxylicEster": "33308",
    "amine": "32952",
    "aldehyde": "17478",
    "cyclic": "33595",
    "ketone": "17087",
    "organophosphorus": "25710",
    "alkane": "18310",
    "haloAlkane": "24469",
    "heteroOrganic": "33285"
}

class LopsterClassifier(Classifier):

    def __init__(self, cyclic_mode=False):
        # the original paper evaluates the cyclicity related rules separately
        # here, using cyclic_mode=True might result in performance issues
        self.cyclic_mode = cyclic_mode

    def classify(self, mol_list):
        res = []
        if not isinstance(mol_list, list):
            mol_list = [mol_list]
        for mol in mol_list:
            if not mol:
                res.append({})
                continue
            res.append({cls: self.get_single_classification(mol, lopster_predicate)
                        for lopster_predicate, cls in lopster_chebi_mapping.items() if self.cyclic_mode or cls != "33595"})
        return res
    
    def get_single_classification(self, mol, lopster_predicate):
        import chemlog.lopster.lopster_python as lopster
        # call function with lopster predicate name
        func = getattr(lopster, lopster_predicate)
        return func(mol, None)

if __name__ == "__main__":
    from rdkit import Chem
    mol = Chem.MolFromSmiles("NC(CC(=O)O)C(=O)O") # aspartic acid
    print(LopsterClassifier().classify(mol))