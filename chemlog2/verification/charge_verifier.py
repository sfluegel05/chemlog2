import logging

import numpy as np

from chemlog2.classification.charge_classifier import ChargeCategories
from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic.logic_utils import get_vars_in_formula, substitute_var_in_formula
from gavel.logic import logic
import os

from chemlog2.preprocessing.mol_to_fol import mol_to_fol_atoms, mol_to_fol_fragments
from chemlog2.verification.model_checking import ModelChecker, ModelCheckerOutcome


class ChargeVerifier:

    def __init__(self):
        with open(os.path.join("data", "fol_specifications", "charges.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.charge_formulas = {ChargeCategories[f[0].formula.left.predicate.value.upper()]:
                                    f[0].formula for f in tptp_parsed if len(f) > 0}
        with open(os.path.join("data", "fol_specifications", "fragment_properties.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.fragment_property_formulas = {f[0].formula.left.predicate.value: f[0].formula for f in tptp_parsed if
                                           len(f) > 0}
        with open(os.path.join("data", "fol_specifications", "fragment_helpers.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.fragment_helper_formulas = {f[0].formula.left.predicate.value: f[0].formula for f in tptp_parsed if
                                           len(f) > 0}

    def verify_charge_category(self, mol: Chem.Mol, expected_category: ChargeCategories, variable_assignment: dict):
        universe, extensions = mol_to_fol_fragments(mol, self.fragment_property_formulas, self.fragment_helper_formulas)
        variable_assignment["Global"] = universe - 1
        model_checker_frags = ModelChecker(
            universe, extensions, predicate_definitions={formula.left.predicate.value
                                                         if isinstance(formula.left, logic.PredicateExpression)
                                                         else formula.left.symbol.value:
                                                             (formula.left.arguments, formula.right) if isinstance(
                                                                 formula.left, logic.PredicateExpression) else (
                                                             [], formula.right)
                                                         for formula in self.charge_formulas.values()})
        proof_attempts = []

        if expected_category in self.charge_formulas:
            target_formula = self.charge_formulas[expected_category].right
            target_formula = apply_variable_assignment(target_formula, variable_assignment)
            # prove salt as separate step
            # if expected_category in [ChargeCategories.CATION, ChargeCategories.ANION]:
            #    salt = model_checker_frags.find_model(self.charge_formulas[ChargeCategories.SALT].right)[0]
            #    proof_attempts.append({"target": ChargeCategories.SALT, "variable_assignments": {}, "outcome": salt})
            #    model_checker_frags.extensions["salt"] = np.zeros(model_checker_frags.universe, dtype=bool)
            #    model_checker_frags.extensions["salt"][-1] = salt in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]

            result = model_checker_frags.find_model(target_formula)[0]
            proof_attempts.append(
                {"target": expected_category.name, "variable_assignments": variable_assignment, "outcome": result.name})
            return result, proof_attempts

        # use neutral as default, a molecule is neutral if it is not in any other category
        elif expected_category == ChargeCategories.NEUTRAL:
            for target_category, target_formula in self.charge_formulas.items():
                target_formula = apply_variable_assignment(target_formula, variable_assignment)
                outcome = model_checker_frags.find_model(target_formula.right)[0]
                proof_attempts.append({
                    "target": target_category.name,
                    "variable_assignments": variable_assignment,
                    "outcome": outcome.name
                })
                if outcome not in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                    if outcome in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                        return ModelCheckerOutcome.NO_MODEL, proof_attempts
                    return ModelCheckerOutcome.UNKNOWN, proof_attempts
            return ModelCheckerOutcome.MODEL_FOUND, proof_attempts

        return ModelCheckerOutcome.UNKNOWN, []


def apply_variable_assignment(formula: logic.LogicElement, variable_assignment: dict):
    variables = get_vars_in_formula(formula)
    for variable_name, variable_value in variable_assignment.items():
        matching_variables = [v for v in variables if v.symbol.lower() == variable_name.lower()]
        if len(matching_variables) == 0:
            logging.warning(f"Variable {variable_name} not found in formula")
        if len(matching_variables) > 1:
            logging.warning(f"Multiple variables with name {variable_name} found in formula")
        formula = substitute_var_in_formula(formula, matching_variables[0], variable_value)
    return formula


if __name__ == "__main__":
    verifier = ChargeVerifier()
    print("Verifier initialized")
    # mol = Chem.MolFromSmiles("[Al+3].[Al+3].[O-]S([O-])(=O)=O.[O-]S([O-])(=O)=O.[O-]S([O-])(=O)=O")
    mol = Chem.MolFromSmiles("NCC([O-])=O")
    res = verifier.verify_charge_category(mol, ChargeCategories.ANION, {})
    print(res)
    print("Verification done")
