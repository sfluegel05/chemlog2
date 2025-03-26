import logging

import numpy as np

from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic.logic_utils import get_vars_in_formula, substitute_var_in_formula
from gavel.logic import logic
import os

from chemlog.preprocessing.mol_to_fol import mol_to_fol_building_blocks, apply_variable_assignment
from chemlog.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome


class PeptideSizeVerifier:

    def __init__(self):
        with open(os.path.join("data", "fol_specifications", "peptide_structure_helpers.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        self.structure_formulas = {f[0].formula.left.predicate.value:
                                    f[0].formula for f in tptp_parsed if len(f) > 0}


    def verify_n_plus_amino_acids(self, mol: Chem.Mol, expected_n: int, functional_group_extensions, variable_assignment: dict):
        # for functional_group_extensions, assume that they are true
        # note that this only verifies n+, not n (does not check if n+1 fails)
        universe, extensions, second_order_elements = mol_to_fol_building_blocks(mol, functional_group_extensions)
        logging.debug(f"Using the following second-order elements: {', '.join([str(i) + ' -> ' + str(v) for i, v in enumerate(second_order_elements)])}")
        # this model checker uses amide_bond, amino_residue and carboxy_residue from the extension and
        # amino_acid_residue from the definition (if it is not already in the extension)
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                                                         for pred, formula in self.structure_formulas.items()})
        # use second_order_elements to map variable_assignment from list of atoms to index in extension
        variable_assignment = {k: second_order_elements.index(v) for k, v in variable_assignment.items()}
        proof_attempts = []
        target_formula = build_peptide_structure_formula(expected_n)
        target_formula = apply_variable_assignment(target_formula, variable_assignment)
        result = model_checker.find_model(target_formula)[0]
        proof_attempts.append(
            {"target": expected_n, "variable_assignments": variable_assignment, "outcome": result.name})
        return result, proof_attempts

    def classify_n_amino_acids(self, mol: Chem.Mol, functional_groups):
        # for functional_group_extensions, assume that they are true
        universe, extensions, second_order_elements = mol_to_fol_building_blocks(mol, functional_groups)
        logging.debug(f"Using the following second-order elements: "
                      f"{', '.join([str(i) + ' -> ' + str(v) for i, v in enumerate(second_order_elements)])}")
        # this model checker uses amide_bond, amino_residue and carboxy_residue from the extension and
        # amino_acid_residue from the definition (if it is not already in the extension)
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                                                         for pred, formula in self.structure_formulas.items()})
        assignment = None
        for n in range(2, 11):
            target_formula = build_peptide_structure_formula(n)
            outcome = model_checker.find_model(target_formula)
            if outcome[0] in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                return n - 1, assignment
            elif outcome[0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                return -1, None
            # map second-order element back to atoms
            assignment = {v: second_order_elements[ind] for v, ind in outcome[1]}
        return 10, assignment


def build_peptide_structure_formula(n):
    variables = [logic.Variable(f"A{i}") for i in range(n)] + [logic.Variable(f"B{i}") for i in range(n-1)]
    clauses = []
    for i in range(n):
        clauses.append(logic.PredicateExpression("amino_acid_residue", [variables[i]]))
        for j in range(i + 1, n):
            clauses.append(logic.UnaryFormula(
                logic.UnaryConnective.NEGATION,
                logic.PredicateExpression("overlap", [variables[i], variables[j]])
            ))
    for i in range(n-1):
        clauses.append(logic.PredicateExpression("amide_bond", [variables[n + i]]))
        clauses.append(logic.PredicateExpression("overlap", [variables[i + 1], variables[n + i]]))
        disj = [logic.PredicateExpression("overlap", [variables[j], variables[n + i]]) for j in range(0, i + 1)]
        if len(disj) == 1:
            clauses.append(disj[0])
        else:
            clauses.append(logic.NaryFormula(logic.BinaryConnective.DISJUNCTION, disj))
    return logic.QuantifiedFormula(logic.Quantifier.EXISTENTIAL, variables,
                                   logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, clauses))
