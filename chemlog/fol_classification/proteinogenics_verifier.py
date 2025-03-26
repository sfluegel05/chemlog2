import logging

import numpy as np
from gavel.logic import logic, logic_utils

from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
import os

from chemlog.preprocessing.chebi_data import ChEBIData
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms, apply_variable_assignment
from chemlog.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome


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

    def setup_model_checker(self, mol, functional_groups):
        universe, extensions = mol_to_fol_atoms(mol)
        # use atom-level extensions, enhanced with functional group information (broken down to single atoms)
        for fg_name, fg_atom in functional_groups.items():
            if fg_name not in extensions:
                extensions[fg_name] = np.zeros(universe, dtype=bool)
            extensions[fg_name][fg_atom] = True
        # all_different adds pairwise inequality constraints for all quantified variables
        return ModelChecker(
            universe, extensions, all_different=True,
            predicate_definitions={pred: (formula.left.arguments, formula.right)
                                   for pred, formula in self.helpers.items()}
        )

    def verify_proteinogenics(self, mol: Chem.Mol, functional_groups, expected_proteinogenics: list):
        # expected_proteinogenics should be a list of tuples ("A", [0, 5, 7, 9, 13])
        # where the first element is the amino acid code and the second element is a list of atom indices
        model_checker = self.setup_model_checker(mol, functional_groups)
        proof_attempts = []
        all_successful = True
        for amino_acid_code, atoms in expected_proteinogenics:
                target_formula = self.proteinogenics_defs[f"aa_{amino_acid_code}"]
                if len(atoms) != len(target_formula.left.arguments):
                    logging.warning(f"Expected {len(target_formula.left.arguments)} atoms for {amino_acid_code}, got {len(atoms)}")
                    result = ModelCheckerOutcome.NO_MODEL_INFERRED
                else:
                    target_formula = apply_variable_assignment(target_formula.right, {var.symbol: atom for var, atom in zip(target_formula.left.arguments, atoms)})
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

    def classify_proteinogenics(self, mol: Chem.Mol, functional_groups):
        universe, extensions = mol_to_fol_atoms(mol)
        # use atom-level extensions, enhanced with functional group information (broken down to single atoms)
        model_checker = self.setup_model_checker(mol, functional_groups)
        proven_amino_acids = []
        variable_assignments = []
        for amino_acid_predicate, target_formula in self.proteinogenics_defs.items():
            target_formula = logic.QuantifiedFormula(logic.Quantifier.EXISTENTIAL,
                                                     target_formula.left.arguments,
                                                     target_formula.right)
            try:
                outcome = model_checker.find_model(target_formula)
            except Exception as e:
                logging.error(f"Error while classifying {amino_acid_predicate}: {e}")
                outcome = ModelCheckerOutcome.ERROR, []
            if outcome[0] in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                proven_amino_acids.append(amino_acid_predicate[3:])
                variable_assignments.append([ind for _, ind in outcome[1]])
                logging.debug(f"{amino_acid_predicate} has been found")
            else:
                logging.debug(f"{amino_acid_predicate} has not been found: {outcome}")

        return proven_amino_acids, variable_assignments

if __name__ == "__main__":
    import sys

    logging.basicConfig(

        format="[%(filename)s:%(lineno)s] %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    verifier = ProteinogenicsVerifier()
    data = ChEBIData(239)
    mol = data.processed.loc[73845, "mol"]
    from chemlog.results import plot_mol
    plot_mol(mol)
    fg = {"amino_residue_n": [4,9], "carboxy_residue_c": [0, 6, 10]}
    print(verifier.classify_proteinogenics(mol,fg))