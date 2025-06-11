import logging
from typing import Optional, Tuple

from gavel.logic import logic
from gavel.logic.logic_utils import convert_to_nnf, convert_to_cnf

from chemlog.fol_classification.model_checking import ModelCheckerOutcome


def nnf_to_pnf(formula):
    # assume formula in NNF without -> or <->
    # separate quantifiers from matrix
    quantifiers = []
    if isinstance(formula, logic.QuantifiedFormula):
        quantifiers.append((formula.quantifier, formula.variables))
        formula, qs = nnf_to_pnf(formula.formula)
        quantifiers += qs
    elif isinstance(formula, logic.UnaryFormula):
        pass
    elif isinstance(formula, logic.BinaryFormula):
        formula.left, qs_left = nnf_to_pnf(formula.left)
        formula.right, qs_right = nnf_to_pnf(formula.right)
        quantifiers += qs_left + qs_right
    elif isinstance(formula, logic.NaryFormula):
        f_qs = [nnf_to_pnf(f) for f in formula.formulae]
        formula.formulae = [f for f, _ in f_qs]
        quantifiers += [q for _, qs in f_qs for q in qs]
    else:
        pass

    return formula, quantifiers

def normalize_fol_formula(formula) -> (ModelCheckerOutcome, Optional[Tuple[str, int]]):
    """Converts formula to PNF, matrix CNF"""
    nnf_formula = convert_to_nnf(formula)
    pnf_matrix, quantifiers = nnf_to_pnf(nnf_formula)
    cnf_matrix = convert_to_cnf(pnf_matrix)

    if len(quantifiers) == 0:
        return cnf_matrix
    curr_quantifier, curr_variables = quantifiers[-1]
    if len(quantifiers) > 1:
        quantifiers.reverse()
        for q, vs in quantifiers[1:]:
            if q == curr_quantifier:
                curr_variables = vs + curr_variables
            else:
                cnf_matrix = logic.QuantifiedFormula(curr_quantifier, curr_variables, cnf_matrix)
                curr_quantifier, curr_variables = q, vs
    pnf = logic.QuantifiedFormula(curr_quantifier, curr_variables, cnf_matrix)
    logging.debug("Formula in PNF: " + str(pnf))
    return pnf