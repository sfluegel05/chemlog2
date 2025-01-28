import logging

import numpy as np

from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic.logic_utils import get_vars_in_formula, substitute_var_in_formula
from gavel.logic import logic
import os

from chemlog2.preprocessing.mol_to_fol import mol_to_fol_atoms, apply_variable_assignment
from chemlog2.verification.model_checking import ModelChecker, ModelCheckerOutcome


class ProteinogenicsVerifier:

    def __init__(self):
        with open(os.path.join("data", "fol_specifications", "proteinogenics.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.proteinogenics_defs = {f[0].formula.left.predicate.value:
                                    f[0].formula for f in tptp_parsed if len(f) > 0}
        with open(os.path.join("data", "fol_specifications", "proteinogenics_helpers.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        self.helpers = {f[0].formula.left.predicate.value: f[0].formula for f in tptp_parsed if len(f) > 0}


    def verify_proteinogenics(self, mol: Chem.Mol, functional_groups, expected_proteinogenics: list):
        # expected_proteinogenics should be a list of tuples ("A", [0, 5, 7, 9, 13])
        # where the first element is the amino acid code and the second element is a list of atom indices
        universe, extensions = mol_to_fol_atoms(mol)
        # use atom-level extensions, enhanced with functional group information (broken down to single atoms)
        for fg_name, fg_atom in functional_groups.items():
            if fg_name not in extensions:
                extensions[fg_name] = np.zeros(universe, dtype=bool)
            extensions[fg_name][fg_atom] = True
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                            for pred, formula in {**self.proteinogenics_defs, **self.helpers}.items()}
        )
        proof_attempts = []
        all_successful = True
        for amino_acid_code, atoms in expected_proteinogenics:
                # only use a simple predicate expression -> the real work is in applying the predicate definitions
                target_formula = logic.PredicateExpression(f"aa_{amino_acid_code}", atoms)
                try:
                    result = model_checker.find_model(target_formula)[0]
                except Exception as e:
                    logging.error(f"Error while verifying {amino_acid_code} with atoms {atoms}: {e}")
                    result = ModelCheckerOutcome.ERROR
                if result not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                    all_successful = False
                proof_attempts.append(
                    {"target": f"aa_{amino_acid_code}", "variable_assignments": atoms, "outcome": result.name})
        # only return positive result if **all** proteinogenic amino acids have been found
        return ModelCheckerOutcome.MODEL_FOUND if all_successful else ModelCheckerOutcome.NO_MODEL, proof_attempts
