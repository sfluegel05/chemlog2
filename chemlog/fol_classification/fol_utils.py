import logging
from typing import Optional, Tuple

from gavel.logic import logic
from gavel.logic.logic_utils import convert_to_nnf, convert_to_cnf

from chemlog.fol_classification.model_checking import ModelCheckerOutcome


def nnf_to_existential_pnf(formula):
    # assume formula in NNF without -> or <->
    # separate existential quantifiers from matrix (keep universal quantifiers in matrix - but also normalise the universally quantified formulas)
    quantifiers = []
    if isinstance(formula, logic.QuantifiedFormula) and (formula.quantifier == logic.Quantifier.EXISTENTIAL):
        quantifiers.append((formula.quantifier, formula.variables))
        formula, qs = nnf_to_existential_pnf(formula.formula)
        quantifiers += qs
    elif isinstance(formula, logic.QuantifiedFormula) and (formula.quantifier == logic.Quantifier.UNIVERSAL):
        formula = logic.QuantifiedFormula(formula.quantifier, formula.variables, _unique_nnf_to_existential_cnf(formula.formula))
    elif isinstance(formula, logic.UnaryFormula):
        pass
    elif isinstance(formula, logic.BinaryFormula):
        formula.left, qs_left = nnf_to_existential_pnf(formula.left)
        formula.right, qs_right = nnf_to_existential_pnf(formula.right)
        quantifiers += qs_left + qs_right
    elif isinstance(formula, logic.NaryFormula):
        f_qs = [nnf_to_existential_pnf(f) for f in formula.formulae]
        formula.formulae = [f for f, _ in f_qs]
        quantifiers += [q for _, qs in f_qs for q in qs]
    else:
        pass

    return formula, quantifiers


def make_variables_unique(formula):
    v_counter = 0

    def _rename(f, renamings: dict):
        nonlocal v_counter
        if isinstance(f, logic.QuantifiedFormula):
            # rename variables to make them unique
            vars = list(f.variables)
            v_counter += len(vars)
            return logic.QuantifiedFormula(f.quantifier, [logic.Variable(f"x{v_counter - len(vars) + i}") for i, v in enumerate(vars)],
                                         _rename(f.formula, {**renamings, **{v: logic.Variable(f"x{v_counter - len(vars) + i}")
                                                             for i, v in enumerate(vars)}}))
        elif isinstance(f, logic.UnaryFormula):
            return logic.UnaryFormula(f.connective, _rename(f.formula, renamings))
        elif isinstance(f, logic.BinaryFormula):
            return logic.BinaryFormula(_rename(f.left, renamings), f.operator, _rename(f.right, renamings))
        elif isinstance(f, logic.NaryFormula):
            return logic.NaryFormula(f.operator, [_rename(sub_formula, renamings) for sub_formula in f.formulae])
        elif isinstance(f, logic.PredicateExpression):
            return logic.PredicateExpression(f.predicate, [_rename(arg, renamings) for arg in f.arguments])
        else:
            # replace if f is in renamings, otherwise return f
            return renamings.get(f, f)

    return _rename(formula, {})


def sort_clauses_by_complexity(cnf_matrix: logic.NaryFormula):
    """Assume CNF formula, sort clauses so that simple clauses (less members, no universal quantifiers) come first"""
    cnf_matrix.formulae = sorted(cnf_matrix.formulae, key=lambda clause: (sum(1 if not isinstance(clause, logic.QuantifiedFormula) else 100 for clause in clause.formulae)))
    return cnf_matrix

def _unique_nnf_to_existential_cnf(formula):
    """Assumes formula in NNF without -> or <->, returns formula with existential quantifiers at the front, matrix in CNF (but still with universal quantifiers)"""
    pnf_matrix, quantifiers = nnf_to_existential_pnf(formula)
    cnf_matrix_unsorted = convert_to_cnf(pnf_matrix)
    cnf_matrix = sort_clauses_by_complexity(cnf_matrix_unsorted)

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
    return pnf

def normalize_fol_formula(formula) -> (ModelCheckerOutcome, Optional[Tuple[str, int]]):
    """Converts formula to PNF, matrix CNF"""
    nnf_formula = convert_to_nnf(formula)
    nnf_formula_renamed = make_variables_unique(nnf_formula)
    res = _unique_nnf_to_existential_cnf(nnf_formula_renamed)
    logging.debug("Normalized formula to existential PNF/CNF: " + str(res))
    return res