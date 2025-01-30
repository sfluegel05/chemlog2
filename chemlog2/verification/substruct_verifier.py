import logging
from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
import os

from chemlog2.preprocessing.mol_to_fol import mol_to_fol_atoms, apply_variable_assignment
from chemlog2.verification.model_checking import ModelChecker, ModelCheckerOutcome


class SubstructVerifier:

    def __init__(self):
        with open(os.path.join("data", "fol_specifications", "substructs.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.substruct_defs = {f[0].formula.left.predicate.value:
                                    f[0].formula for f in tptp_parsed if len(f) > 0}


    def verify_substruct_class(self, mol: Chem.Mol, target_cls: str, atoms: list):
        universe, extensions = mol_to_fol_atoms(mol)
        model_checker = ModelChecker(
            universe, extensions
        )
        target_formula = self.substruct_defs[target_cls]
        if len(atoms) != len(target_formula.left.arguments):
            logging.warning(f"Expected {len(target_formula.left.arguments)} atoms for {target_cls}, got {len(atoms)}")
            result = ModelCheckerOutcome.NO_MODEL_INFERRED
        else:
            target_formula = apply_variable_assignment(target_formula.right, {var.symbol: atom for var, atom in zip(target_formula.left.arguments, atoms)})
            try:
                result = model_checker.find_model(target_formula)[0]
            except Exception as e:
                logging.error(f"Error while verifying {target_cls} with atoms {atoms}: {e}")
                result = ModelCheckerOutcome.ERROR

        return result, {"target": target_cls, "variable_assignments": atoms, "outcome": result.name}

    def classify_substruct_class(self, mol: Chem.Mol, target_cls: str):
        universe, extensions = mol_to_fol_atoms(mol)
        model_checker = ModelChecker(
            universe, extensions
        )
        target_formula = self.substruct_defs[target_cls]
        try:
                result = model_checker.find_model(target_formula)
        except Exception as e:
                logging.error(f"Error while classify {target_cls}: {e}")
                result = ModelCheckerOutcome.ERROR, []

        return result[0] in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED], result