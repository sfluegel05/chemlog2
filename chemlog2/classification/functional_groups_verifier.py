import logging

import numpy as np

from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic.logic_utils import get_vars_in_formula, substitute_var_in_formula
from gavel.logic import logic
import os

from chemlog2.preprocessing.mol_to_fol import mol_to_fol_atoms, apply_variable_assignment
from chemlog2.verification.model_checking import ModelChecker, ModelCheckerOutcome


class FunctionalGroupsVerifier:

    def __init__(self):
        with open(os.path.join("data", "fol_specifications", "functional_groups.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.functional_group_defs = {f[0].formula.left.predicate.value:
                                    f[0].formula for f in tptp_parsed if len(f) > 0}
        with open(os.path.join("data", "fol_specifications", "functional_group_helpers.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.functional_group_helpers = {f[0].formula.left.predicate.value:
                                    f[0].formula for f in tptp_parsed if len(f) > 0}


    def verify_functional_groups(self, mol: Chem.Mol, expected_groups: dict):
        # expected_groups is expected to have keys that match the predicates from self.functional_group_defs
        # and values that match their arity
        universe, extensions = mol_to_fol_atoms(mol)
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                                                         for pred, formula in self.functional_group_helpers.items()})
        proof_attempts = []
        all_successful = True
        for group in expected_groups:
            for atoms in expected_groups[group]:
                target_formula = self.functional_group_defs[group]
                if len(atoms) != len(target_formula.left.arguments):
                    logging.warning(
                        f"Expected {len(target_formula.left.arguments)} atoms for {group}, got {len(atoms)}")
                    result = ModelCheckerOutcome.NO_MODEL_INFERRED
                else:
                    target_formula = apply_variable_assignment(target_formula.right, {var.symbol: atom for var, atom in
                                                                                      zip(target_formula.left.arguments,
                                                                                          atoms)})
                    result = model_checker.find_model(target_formula)[0]
                if result not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                    all_successful = False
                proof_attempts.append(
                    {"target": group, "variable_assignments": atoms, "outcome": result.name})
        # only return positive result if **all** functional groups have been found
        return ModelCheckerOutcome.MODEL_FOUND if all_successful else ModelCheckerOutcome.NO_MODEL, proof_attempts

    def classify_functional_groups(self, mol: Chem.Mol):
        universe, extensions = mol_to_fol_atoms(mol)
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                                                         for pred, formula in self.functional_group_helpers.items()})
        functional_groups = {}
        for group, target_formula in self.functional_group_defs.items():
            functional_groups[group] = []
            for atom_combination in permutations(range(mol.GetNumAtoms()), len(target_formula.left.arguments)):
                target_formula = apply_variable_assignment(target_formula.right, {var.symbol: atom for var, atom in
                                                                                  zip(target_formula.left.arguments,
                                                                                      atom_combination)})

                outcome = model_checker.find_model(target_formula)
                if outcome[0] in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                    functional_groups[group].append(outcome[1])

        return functional_groups


if __name__ == "__main__":
    from itertools import permutations

    print(list(permutations(range(5), 3)))
