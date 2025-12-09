

import os
from tqdm import tqdm
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
        for mol in tqdm(mol_list, desc="Lopster classification"):
            if not mol:
                res.append({})
                continue
            ress = {}
            for lopster_predicate, cls in lopster_chebi_mapping.items():
                if not self.cyclic_mode and cls == "33595":
                    continue
                #print(f"Classifying {cls} using {lopster_predicate}")
                ress[cls] = self.get_single_classification(mol, lopster_predicate)
            res.append(ress)
        return res
    
    def get_single_classification(self, mol, lopster_predicate):
        import chemlog.lopster.lopster_python as lopster
        # call function with lopster predicate name
        func = getattr(lopster, lopster_predicate)
        return func(mol, None)


class LopsterClingoClassifier(LopsterClassifier):

    def __init__(self, cyclic_mode=False, batch_size=10):
        super().__init__(cyclic_mode)
        self.batch_size = batch_size
        self.rules_file_atom = os.path.join(os.path.dirname(__file__), 'lopster_rules_atom_level.pl')
        self.rules_file_molecule = os.path.join(os.path.dirname(__file__), 'lopster_rules_molecule_level.pl')
        
        self.lopster_rules = ""
        with open(self.rules_file_atom, 'r', encoding='utf-8') as f:
            for line in f:
                if cyclic_mode or (not "cyclic" in line and not "cleanReachable" in line):
                    self.lopster_rules += line
        
        with open(self.rules_file_molecule, 'r', encoding='utf-8') as f:
            for line in f:
                # filter out cyclicity related rules if not in cyclic mode
                if cyclic_mode or (not "cyclic" in line and not "cleanReachable" in line):
                    self.lopster_rules += line

    def build_logic_program(self, mol_list):
        from chemlog.lopster.lopster_code.dg_generator import DGGenerator
        from chemlog.lopster.lopster_code.dlv_program_manager import DLVProgramManager
        dg_generator = DGGenerator()
        description_graphs = dg_generator.assemble_dg_map(mol_list)

        dlv_manager = DLVProgramManager(description_graphs)
        program = dlv_manager.create_dlv_programs(mol_list)

        return program + "\n" + self.lopster_rules

    def classify(self, mol_list_all, verbose=False):
        res = []
        idx = 0
        for mol_list in mol_list_all[idx:idx + self.batch_size]:
            idx += self.batch_size
            if not isinstance(mol_list, list):
                mol_list = [mol_list]
            lp = self.build_logic_program(mol_list)
            if verbose:
                print("Generated logic program for Clingo -> demo_lopster_clingo.lp")
                with open("demo_lopster_clingo.lp", 'w+', encoding='utf-8') as f:
                    f.write(lp)
            import clingo
            if not verbose:
                ctl = clingo.Control(message_limit=0) # silent mode
            else:
                ctl = clingo.Control()
            ctl.add("base", [], lp)
            ctl.ground([("base", [])])
            model_list = []
            solution = ctl.solve(yield_=True)
            for m in solution:
                model_list = m.symbols(shown=True)
            if verbose:
                print("Clingo solution:", solution)
                print("Model list:", model_list)
            for mol_index in range(len(mol_list)):
                ress = {}
                for lopster_predicate, cls in lopster_chebi_mapping.items():
                    if not self.cyclic_mode and cls == "33595":
                        continue
                    predicate_found = False
                    for symbol in model_list:
                        if str(symbol) == f"{lopster_predicate}(c_mol_{mol_index})":
                            predicate_found = True
                            break
                    ress[cls] = predicate_found
                res.append(ress)
            
            if idx >= len(mol_list_all):
                break
        return res
        

if __name__ == "__main__":
    from rdkit import Chem
    mol = Chem.MolFromSmiles("NC(CC(=O)O)C(=O)O") # aspartic acid
    #mol = Chem.MolFromSmiles(r"[Cl-].[H][N+](C)(C)CC\C=C1\c2ccccc2CSc2ccccc12")
    #print(LopsterClassifier().classify(mol))
    print(LopsterClingoClassifier().classify([mol], verbose=False))