import os
from typing import Set, List, Dict
from .dlv_rules_generator import DLVRulesGenerator
from .description_graph import DescriptionGraph
from .dlv_rule import DLVRule
from rdkit import Chem

class DLVProgramManager:
    def __init__(self, description_graphs: Dict[str, DescriptionGraph]) -> str:
        self.description_graphs = description_graphs

    # create all program files
    def create_dlv_programs(self, mol_list: List[Chem.Mol]):
        all_rules = set()
        gen = DLVRulesGenerator()
        for mol_index, mol in enumerate(mol_list):
            dg = self.description_graphs.get(f"mol_{mol_index}")
            if dg is not None:
                all_rules.update(gen.translate_dg_into_rules(dg))
        return "\n".join([rule.dlv_syntax_rule() for rule in all_rules])

