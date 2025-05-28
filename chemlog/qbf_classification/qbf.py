import enum
import logging
from collections import deque
from copy import copy
from typing import List
from itertools import product

from gavel.logic.logic import Quantifier, BinaryFormula


class Quantifier(enum.Enum):
    # either A or E
    A = 0
    E = 1

    def __repr__(self):
        if self == Quantifier.A:
            return "\u2200"
        elif self == Quantifier.E:
            return "\u2203"
        else:
            raise NotImplementedError

class Connective(enum.Enum):
    NEG = 0
    AND = 1
    OR = 2
    IMPLIES = 3
    BIIMP = 4

    def __repr__(self):
        if self.value == 0:
            return "\u00AC"
        elif self.value == 1:
            return "\u2227"
        elif self.value == 2:
            return "\u2228"
        elif self.value == 3:
            return "\u2192"
        elif self.value == 4:
            return "\u21944"
        else:
            raise NotImplementedError


class NegFormula():

    def __init__(self, formula):
        self.formula = formula

    def __repr__(self):
        return f'{repr(Connective.NEG)}{self.formula}'


class BinaryFormula:

    def __init__(self, left, connective: Connective, right):
        self.left = left
        self.connective = connective
        self.right = right

    def __repr__(self):
        return f"({self.left} {repr(self.connective)} {self.right})"

class NaryFormula:

    def __init__(self, connective, formulas):
        self.connective = connective
        self.formulas = formulas

    def __repr__(self):
        return "(" + f" {repr(self.connective)} ".join([str(f) for f in self.formulas]) + ")"

class QuantifiedFormula:

    def __init__(self, quantifier: Quantifier, variables: List[str], formula):
        self.quantifier = quantifier
        self.variables = variables
        self.formula = formula

    def __repr__(self):
        return f"{repr(self.quantifier)}{'.'.join(str(v) for v in self.variables)}. {self.formula}"


def qbf_to_nnf(formula):
    # eliminates -> and <-> from a QBF formula, converts it to NNF
    if isinstance(formula, QuantifiedFormula):
        formula.formula = qbf_to_nnf(formula.formula)
    elif isinstance(formula, NegFormula):
        if isinstance(formula.formula, QuantifiedFormula):
            formula.formula.formula = qbf_to_nnf(NegFormula(formula.formula.formula))
            if formula.formula.quantifier == Quantifier.A:
                formula.formula.quantifier = Quantifier.E
            else:
                formula.formula.quantifier = Quantifier.A
            formula = formula.formula
        elif isinstance(formula.formula, NegFormula):
            formula = qbf_to_nnf(formula.formula.formula)
        elif isinstance(formula.formula, BinaryFormula):
            # ~(A -> B) iff (A & ~B)
            if formula.formula.connective == Connective.IMPLIES:
                formula = qbf_to_nnf(BinaryFormula(formula.formula.left, Connective.AND, NegFormula(formula.formula.right)))
            # ~(A <-> B) iff (A & ~B) | (~A & B)
            elif formula.formula.connective == Connective.BIIMP:
                formula = qbf_to_nnf(BinaryFormula(
                    BinaryFormula(formula.formula.left, Connective.AND, NegFormula(formula.formula.right)),
                    Connective.OR,
                    BinaryFormula(NegFormula(formula.formula.left), Connective.AND, formula.formula.right)))
            elif formula.formula.connective == Connective.OR:
                formula = BinaryFormula(qbf_to_nnf(NegFormula(formula.formula.left)), Connective.AND, qbf_to_nnf(NegFormula(formula.formula.right)))
            elif formula.formula.connective == Connective.AND:
                formula = BinaryFormula(qbf_to_nnf(NegFormula(formula.formula.left)), Connective.OR, qbf_to_nnf(NegFormula(formula.formula.right)))
            else:
                raise NotImplementedError(f"Encountered unknown connective: {formula.formula.connective}")
        elif isinstance(formula.formula, NaryFormula):
            if formula.formula.connective == Connective.OR:
                formula = NaryFormula(Connective.AND, [qbf_to_nnf(NegFormula(f)) for f in formula.formula.formulas])
            elif formula.formula.connective == Connective.AND:
                formula = NaryFormula(Connective.OR, [qbf_to_nnf(NegFormula(f)) for f in formula.formula.formulas])
            else:
                raise NotImplementedError(f"Encountered unknown connective: {formula.formula.connective}")
        elif isinstance(formula.formula, str):
            pass
        else:
            raise NotImplementedError(f"Encountered unknown formula type: {formula.formula} (type: {type(formula.formula)})")
    elif isinstance(formula, BinaryFormula):
        if formula.connective == Connective.IMPLIES:
            # A -> B iff ~A | B
            formula = qbf_to_nnf(BinaryFormula(NegFormula(formula.left), Connective.OR, formula.right))
        elif formula.connective == Connective.BIIMP:
            # A <-> B iff (A -> B) & (B -> A)
            formula = qbf_to_nnf(BinaryFormula(
                BinaryFormula(formula.left, Connective.IMPLIES, formula.right),
                Connective.AND,
                BinaryFormula(formula.right, Connective.IMPLIES, formula.left)))
        formula.left = qbf_to_nnf(formula.left)
        formula.right = qbf_to_nnf(formula.right)
    elif isinstance(formula, NaryFormula):
        formula.formulas = [qbf_to_nnf(f) for f in formula.formulas]
    elif isinstance(formula, str):
        pass
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")

    return formula

def rename_vars_in_formula(formula, renamings: dict):
    # assume formula in nnf, pnf, without -> or <->
    if isinstance(formula, NegFormula):
        formula.formula = rename_vars_in_formula(formula.formula, renamings)
    elif isinstance(formula, BinaryFormula):
        formula.left = rename_vars_in_formula(formula.left, renamings)
        formula.right = rename_vars_in_formula(formula.right, renamings)
    elif isinstance(formula, NaryFormula):
        formula.formulas = [rename_vars_in_formula(f, renamings) for f in formula.formulas]
    elif isinstance(formula, str):
        if formula in renamings:
            formula = renamings[formula]
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")
    return formula


def nnf_to_pnf(formula):
    # assume formula in NNF without -> or <->
    # separate quantifiers from matrix
    quantifiers = []
    if isinstance(formula, QuantifiedFormula):
        quantifiers.append((formula.quantifier, formula.variables))
        formula, qs = nnf_to_pnf(formula.formula)
        quantifiers += qs
    elif isinstance(formula, NegFormula):
        pass
    elif isinstance(formula, BinaryFormula):
        formula.left, qs_left = nnf_to_pnf(formula.left)
        formula.right, qs_right = nnf_to_pnf(formula.right)
        # if the same variable names are used, we have to rename them
        l_vars = [v for q, vs in qs_left for v in vs]
        r_vars = [v for q, vs in qs_right for v in vs]
        l_renames = {v: f"{v}_l" for v in l_vars if v in r_vars}
        r_renames = {v: f"{v}_r" for v in r_vars if v in l_vars}
        formula.left = rename_vars_in_formula(formula.left, l_renames)
        formula.right = rename_vars_in_formula(formula.right, r_renames)
        qs_left = [(q, [l_renames[v] if v in l_renames else v for v in vs]) for q, vs in qs_left]
        qs_right = [(q, [r_renames[v] if v in r_renames else v for v in vs]) for q, vs in qs_right]
        quantifiers += qs_left + qs_right
    elif isinstance(formula, NaryFormula):
        f_qs = [nnf_to_pnf(f) for f in formula.formulas]
        formula.formulas = [f for f, _ in f_qs]
        all_qs = [qs for _, qs in f_qs]
        all_vars = [[v for q, vs in qs for v in vs] for qs in all_qs]
        all_renames = [{v: f"{v}_{i}" for v in all_vars[i] if any(v in vs for j, vs in enumerate(all_vars) if j != i)} for i in range(len(all_vars))]
        formula.formulas = [rename_vars_in_formula(f, all_renames[i]) for i, f in enumerate(formula.formulas)]
        all_qs_renamed = [[(q, [all_renames[i][v] if v in all_renames[i] else v for v in vs]) for q, vs in all_qs[i]] for i in range(len(all_qs))]
        quantifiers += [qs for qs_list in all_qs_renamed for qs in qs_list]
    elif isinstance(formula, str):
        pass
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")

    return formula, quantifiers

def matrix_to_cnf_distributivity(formula) -> NaryFormula:
    # formula without quantifiers, in NNF, without -> or <->
    # uses distributivity laws (A & B) | (C & D) <=> (A | C) & (A | D) & (B | C) & (B | D)
    # be careful, exponential blowup possible
    if isinstance(formula, NegFormula) or isinstance(formula, str):
        formula = NaryFormula(Connective.AND, [NaryFormula(Connective.OR, [formula])])
    elif isinstance(formula, BinaryFormula):
        formula = matrix_to_cnf_distributivity(NaryFormula(formula.connective, [formula.left, formula.right]))
    elif isinstance(formula, NaryFormula):
        cnfs = [matrix_to_cnf_distributivity(f) for f in formula.formulas]
        if formula.connective == Connective.AND:
            # (C1 & C2 & ... & Cn) & (Cn+1 & ... & Cn+k) -> (C1 & C2 & ... & Cn & Cn+1 & ... & Cn+k) for CNF clauses C
            formula.formulas = [f for cnf in cnfs for f in cnf.formulas]
        elif formula.connective == Connective.OR:
            # (C1 & C2 & ... & Cn) | (Cn+1 & ... & Cn+k) -> (C1 | Cn+1) & (C1 | Cn+2) & ... & (Cn | Cn+k)
            fused_clauses = []
            for clauses in product(*[cnf.formulas for cnf in cnfs]):
                # remove duplicates, skip tautological clauses
                new_clause = [literal for clause in clauses for literal in clause.formulas]
                pos_literals = set([l for l in new_clause if not isinstance(l, NegFormula)])
                neg_literals = set([l.formula for l in new_clause if isinstance(l, NegFormula)])
                if len(pos_literals.intersection(neg_literals)) == 0:
                    new_clause = list(pos_literals) + [NegFormula(l) for l in neg_literals]
                    fused_clauses.append(new_clause)
            formula = NaryFormula(Connective.AND, [NaryFormula(Connective.OR, clause) for clause in fused_clauses])
    else:
        raise NotImplementedError(f"Encountered unknown formula: {formula} (type: {type(formula)}")

    return formula

def matrix_to_cnf_tseytin(formula, verbose=False) -> (NaryFormula, List[str]):
    if verbose:
        print(f"Starting Tseytin transformation for {formula}")
    # formula without quantifiers, in NNF, without -> or <->, with binary quantifiers replaced by Nary quantifiers
    # uses Tseytin transformation (https://personal.cis.strath.ac.uk/robert.atkey/cs208/converting-to-cnf.html)
    # result is satisfiable-equivalent to the original formula
    if isinstance(formula, NegFormula) or isinstance(formula, str):
        return NaryFormula(Connective.AND, [NaryFormula(Connective.OR, [formula])])
    assert isinstance(formula, NaryFormula)
    tseytin_equations = []
    queue = deque([(0, formula)])
    tseytin_var_counter = 1
    while len(queue) > 0:
        tvar, f = queue.popleft()
        if tvar != len(tseytin_equations):
            raise Exception("Tseytin transformation failed: Cannot connect variables to formulas")
        child_formulas = f.formulas
        equation = (f.connective, [])
        for child in child_formulas:
            # replace children with new variables (encoded as tseytin_{i}) if they are not literals
            if not (isinstance(child, NegFormula) or isinstance(child, str)):
                queue.append((tseytin_var_counter, child))
                equation[1].append(f"tseytin_{tseytin_var_counter}")
                tseytin_var_counter += 1
            else:
                equation[1].append(child)
        tseytin_equations.append(equation)
    if verbose:
        print(f"Tseytin equations:\n{"\n\t".join(str(eq) for eq in tseytin_equations)}")

    def neg(literal):
        if isinstance(literal, NegFormula):
            return literal.formula
        elif isinstance(literal, str):
            return NegFormula(literal)
        else:
            raise NotImplementedError(f"Encountered unknown literal type: {literal} (type: {type(literal)})")

    clauses = [NaryFormula(Connective.OR, ["tseytin_0"])]
    for i, eq in enumerate(tseytin_equations):
        if eq[0] == Connective.AND:
            # A & B => i turns into ~A | ~B | i
            clauses.append(NaryFormula(Connective.OR, [f"tseytin_{i}"] + [neg(l) for l in eq[1]]))
            # i => A & B turns into (A | ~i) & (B | ~i)
            for l in eq[1]:
                clauses.append(NaryFormula(Connective.OR, [l, neg(f"tseytin_{i}")]))
        elif eq[0] == Connective.OR:
            # eq[1] => i
            clauses.append(NaryFormula(Connective.OR, [neg(f"tseytin_{i}")] + [l for l in eq[1]]))
            # i => eq[1]
            for l in eq[1]:
                clauses.append(NaryFormula(Connective.OR, [neg(l), f"tseytin_{i}"]))

    return NaryFormula(Connective.AND, clauses), [f"tseytin_{i}" for i in range(tseytin_var_counter)]


def binary_to_nary(formula):
    # convert binary formulas to nary formulas
    # assume no quantifiers, NNF, no -> or <->
    if isinstance(formula, NegFormula):
        pass
    elif isinstance(formula, BinaryFormula):
        formula = binary_to_nary(NaryFormula(formula.connective, [formula.left, formula.right]))
    elif isinstance(formula, NaryFormula):
        # merge two levels of Nary formulas: (A & B) & (C & D) -> A & B & C & D
        children = [binary_to_nary(f) for f in formula.formulas]
        new_children = []
        for child in children:
            if isinstance(child, NaryFormula) and child.connective == formula.connective:
                new_children += child.formulas
            else:
                new_children.append(child)
        formula = NaryFormula(formula.connective, new_children)
    elif isinstance(formula, str):
        pass
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")
    return formula

def qbf_to_cnf(formula, use_tseytin=True, verbose=False):
    formula = copy(formula)
    if verbose:
        print(f"Converting QBF formula to CNF: {formula}")
    formula = qbf_to_nnf(formula)
    if verbose:
        print(f"Formula in NNF: {formula}")
    # formula is now in NNF without -> or <->
    matrix, quantifiers = nnf_to_pnf(formula)
    if verbose:
        print(f"Matrix: {matrix}")
        print(f"Quantifiers: {quantifiers}")
    matrix = binary_to_nary(matrix)
    if use_tseytin:
        matrix, tseytin_vars = matrix_to_cnf_tseytin(matrix, verbose=verbose)
        # this step is a modification of the Tseytin algorithm to adapt it to QBF
        # originally, the Tseytin variables are (implicitly) existentially quantified, here we need to put
        # the quantifier on the inside to keep the equisatisfiability of the formula
        quantifiers.append((Quantifier.E, tseytin_vars))
    else:
        matrix = matrix_to_cnf_distributivity(matrix)
    if verbose:
        print(f"Matrix in CNF ({'Tseytin' if use_tseytin else 'Distributivity'}): {matrix}")
    # add quantifiers to the matrix (from inside to outside), combine quantifiers A + A / E + E:
    # A x A y E z (m) -> A x A y (E z m) -> (A x,y E z m)
    if len(quantifiers) == 0:
        return matrix
    curr_quantifier, curr_variables = quantifiers[-1]
    if len(quantifiers) > 1:
        quantifiers.reverse()
        for q, vs in quantifiers[1:]:
            if q == curr_quantifier:
                curr_variables = vs + curr_variables
            else:
                matrix = QuantifiedFormula(curr_quantifier, curr_variables, matrix)
                curr_quantifier, curr_variables = q, vs
    matrix = QuantifiedFormula(curr_quantifier, curr_variables, matrix)
    return matrix


def cnf_to_qdimacs(formula, add_comments=True):
    # convert CNF formula to QDIMACS format
    variable_names = dict()
    comments = []
    lines = ["p cnf 0 0"]
    while isinstance(formula, QuantifiedFormula):
        variable_ints = []
        for v in formula.variables:
            if v not in variable_names:
                variable_names[v] = len(variable_names) + 1
            variable_ints.append(variable_names[v])
        lines.append(f"{'e' if formula.quantifier == Quantifier.E else 'a'} {' '.join([str(v) for v in variable_ints])} 0")
        if add_comments:
            comments.append(f"{'e' if formula.quantifier == Quantifier.E else 'a'} {' '.join([str(v) for v in variable_ints])} 0  " + " ".join(str(v) for v in formula.variables))
        formula = formula.formula
    assert(isinstance(formula, NaryFormula) and formula.connective == Connective.AND)
    for clause in formula.formulas:
        variable_ints = []
        assert(isinstance(clause, NaryFormula) and clause.connective == Connective.OR), f"Expected clause to be a disjunction, got {clause} (type: {type(clause)})"
        for literal in clause.formulas:
            negation = 1
            if isinstance(literal, NegFormula):
                negation = -1
                literal = literal.formula
            if literal not in variable_names:
                variable_names[literal] = len(variable_names) + 1
            variable_ints.append(negation * variable_names[literal])
        lines.append(" ".join([str(v) for v in variable_ints]) + " 0")
        if add_comments:
            comments.append(" ".join([str(v) for v in variable_ints]) + " 0  " + str(clause))

    logging.debug(f"Created QDIMACS with {len(variable_names)} variables and {len(formula.formulas)} clauses")
    lines[0] = f"p cnf {len(variable_names)} {len(formula.formulas)}"
    return "\n".join([f"c {c}" for c in comments] + lines)

def demo():
    formula = BinaryFormula("A", Connective.AND, NegFormula(NegFormula("B")))
    formulaq = QuantifiedFormula(Quantifier.E, ["x1", "x2"], NaryFormula(Connective.OR, [formula, QuantifiedFormula(Quantifier.E, ["x3"], "x3"), QuantifiedFormula(Quantifier.A, ["x3"], BinaryFormula("A", Connective.AND, NegFormula(NegFormula("x3"))))]))
    print(formulaq)

    dnf_formula = BinaryFormula(
        BinaryFormula(
            BinaryFormula("P", Connective.OR, "Q"),
            Connective.AND,
            BinaryFormula("R", Connective.OR, "S")
        ),
        Connective.OR,
        BinaryFormula(
            BinaryFormula("A", Connective.OR, "B"),
            Connective.AND,
            BinaryFormula("C", Connective.OR, "D")
        )
    )
    print(qbf_to_cnf(formulaq))
    print(cnf_to_qdimacs(qbf_to_cnf(formulaq, verbose=True)))

    print(dnf_formula)
    nary_dnf_formula = binary_to_nary(dnf_formula)
    print(nary_dnf_formula)
    print(qbf_to_cnf(dnf_formula))
    print(cnf_to_qdimacs(qbf_to_cnf(dnf_formula)))

    formulaq = QuantifiedFormula(Quantifier.E, ["x1", "x2"], NaryFormula(Connective.OR, [formula, QuantifiedFormula(Quantifier.E, ["x3"], "x3"), QuantifiedFormula(Quantifier.A, ["x3"], BinaryFormula("A", Connective.AND, NegFormula(NegFormula("x3"))))]))
    unsat_formula = BinaryFormula(BinaryFormula("A", Connective.AND, NegFormula("A")), Connective.AND, formulaq)
    print(unsat_formula)
    #print(qbf_to_cnf(unsat_formula, verbose=True))
    print(cnf_to_qdimacs(qbf_to_cnf(unsat_formula, verbose=False)))

if __name__ == '__main__':
    print(matrix_to_cnf_distributivity("x0"))