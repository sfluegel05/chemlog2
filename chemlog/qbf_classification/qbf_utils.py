import logging
from collections import deque
from copy import deepcopy
from itertools import product
from typing import List

from chemlog.qbf_classification import qbf
from chemlog.qbf_classification.qbf import NegFormula


def qbf_to_nnf(formula):
    # eliminates -> and <-> from a QBF formula, converts it to NNF
    if isinstance(formula, qbf.QuantifiedFormula):
        formula.formula = qbf_to_nnf(formula.formula)
    elif isinstance(formula, qbf.NegFormula):
        if isinstance(formula.formula, qbf.QuantifiedFormula):
            formula.formula.formula = qbf_to_nnf(qbf.NegFormula(formula.formula.formula))
            if formula.formula.quantifier == qbf.Quantifier.A:
                formula.formula.quantifier = qbf.Quantifier.E
            else:
                formula.formula.quantifier = qbf.Quantifier.A
            formula = formula.formula
        elif isinstance(formula.formula, qbf.NegFormula):
            formula = qbf_to_nnf(formula.formula.formula)
        elif isinstance(formula.formula, qbf.BinaryFormula):
            # ~(A -> B) iff (A & ~B)
            if formula.formula.connective == qbf.Connective.IMPLIES:
                formula = qbf_to_nnf(qbf.BinaryFormula(formula.formula.left, qbf.Connective.AND, qbf.NegFormula(formula.formula.right)))
            # ~(A <-> B) iff (A & ~B) | (~A & B)
            elif formula.formula.connective == qbf.Connective.BIIMP:
                formula = qbf_to_nnf(qbf.BinaryFormula(
                    qbf.BinaryFormula(formula.formula.left, qbf.Connective.AND, qbf.NegFormula(formula.formula.right)),
                    qbf.Connective.OR,
                    qbf.BinaryFormula(qbf.NegFormula(formula.formula.left), qbf.Connective.AND, formula.formula.right)))
            elif formula.formula.connective == qbf.Connective.OR:
                formula = qbf.BinaryFormula(qbf_to_nnf(qbf.NegFormula(formula.formula.left)), qbf.Connective.AND, qbf_to_nnf(qbf.NegFormula(formula.formula.right)))
            elif formula.formula.connective == qbf.Connective.AND:
                formula = qbf.BinaryFormula(qbf_to_nnf(qbf.NegFormula(formula.formula.left)), qbf.Connective.OR, qbf_to_nnf(qbf.NegFormula(formula.formula.right)))
            else:
                raise NotImplementedError(f"Encountered unknown connective: {formula.formula.connective}")
        elif isinstance(formula.formula, qbf.NaryFormula):
            if formula.formula.connective == qbf.Connective.OR:
                formula = qbf.NaryFormula(qbf.Connective.AND, [qbf_to_nnf(qbf.NegFormula(f)) for f in formula.formula.formulas])
            elif formula.formula.connective == qbf.Connective.AND:
                formula = qbf.NaryFormula(qbf.Connective.OR, [qbf_to_nnf(qbf.NegFormula(f)) for f in formula.formula.formulas])
            else:
                raise NotImplementedError(f"Encountered unknown connective: {formula.formula.connective}")
        elif isinstance(formula.formula, str):
            pass
        else:
            raise NotImplementedError(f"Encountered unknown formula type: {formula.formula} (type: {type(formula.formula)})")
    elif isinstance(formula, qbf.BinaryFormula):
        if formula.connective == qbf.Connective.IMPLIES:
            # A -> B iff ~A | B
            formula = qbf_to_nnf(qbf.BinaryFormula(qbf.NegFormula(formula.left), qbf.Connective.OR, formula.right))
        elif formula.connective == qbf.Connective.BIIMP:
            # A <-> B iff (A -> B) & (B -> A)
            formula = qbf_to_nnf(qbf.BinaryFormula(
                qbf.BinaryFormula(formula.left, qbf.Connective.IMPLIES, formula.right),
                qbf.Connective.AND,
                qbf.BinaryFormula(formula.right, qbf.Connective.IMPLIES, formula.left)))
        formula.left = qbf_to_nnf(formula.left)
        formula.right = qbf_to_nnf(formula.right)
    elif isinstance(formula, qbf.NaryFormula):
        formula.formulas = [qbf_to_nnf(f) for f in formula.formulas]
    elif isinstance(formula, str):
        pass
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")

    return formula

def rename_vars_in_formula(formula, renamings: dict):
    # assume formula in nnf, pnf, without -> or <->
    if isinstance(formula, qbf.NegFormula):
        formula.formula = rename_vars_in_formula(formula.formula, renamings)
    elif isinstance(formula, qbf.BinaryFormula):
        formula.left = rename_vars_in_formula(formula.left, renamings)
        formula.right = rename_vars_in_formula(formula.right, renamings)
    elif isinstance(formula, qbf.NaryFormula):
        formula.formulas = [rename_vars_in_formula(f, renamings) for f in formula.formulas]
    elif isinstance(formula, str):
        if formula in renamings:
            formula = renamings[formula]
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")
    return formula

def make_variables_unique(formula):
    v_counter = 0

    def _rename(f, renamings: dict):
        nonlocal v_counter
        if isinstance(f, qbf.QuantifiedFormula):
            # rename variables to make them unique
            v_counter += len(f.variables)
            return qbf.QuantifiedFormula(f.quantifier, [f"x{v_counter - len(f.variables) + i}" for i, v in enumerate(f.variables)],
                                         _rename(f.formula, {**renamings, **{v: f"x{v_counter - len(f.variables) + i}"
                                                             for i, v in enumerate(f.variables)}}))
        elif isinstance(f, qbf.NegFormula):
            return qbf.NegFormula(_rename(f.formula, renamings))
        elif isinstance(f, qbf.BinaryFormula):
            return qbf.BinaryFormula(_rename(f.left, renamings), f.connective, _rename(f.right, renamings))
        elif isinstance(f, qbf.NaryFormula):
            return qbf.NaryFormula(f.connective, [_rename(sub_formula, renamings) for sub_formula in f.formulas])
        elif isinstance(f, str):
            return renamings.get(f, f)
        else:
            raise NotImplementedError(f"Encountered unknown formula type: {f} (type: {type(f)})")

    return _rename(formula, {})

def nnf_to_pnf(formula):
    # assume formula in NNF without -> or <->
    # separate quantifiers from matrix
    quantifiers = []
    if isinstance(formula, qbf.QuantifiedFormula):
        quantifiers.append((formula.quantifier, formula.variables))
        formula, qs = nnf_to_pnf(formula.formula)
        quantifiers += qs
    elif isinstance(formula, qbf.NegFormula):
        pass
    elif isinstance(formula, qbf.BinaryFormula):
        formula.left, qs_left = nnf_to_pnf(formula.left)
        formula.right, qs_right = nnf_to_pnf(formula.right)
        quantifiers += qs_left + qs_right
    elif isinstance(formula, qbf.NaryFormula):
        f_qs = [nnf_to_pnf(f) for f in formula.formulas]
        formula.formulas = [f for f, _ in f_qs]
        quantifiers += [q for _, qs in f_qs for q in qs]
    elif isinstance(formula, str):
        pass
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")

    return formula, quantifiers

def matrix_to_cnf_distributivity(formula) -> qbf.NaryFormula:
    # formula without quantifiers, in NNF, without -> or <->
    # uses distributivity laws (A & B) | (C & D) <=> (A | C) & (A | D) & (B | C) & (B | D)
    # be careful, exponential blowup possible
    if isinstance(formula, qbf.NegFormula) or isinstance(formula, str):
        formula = qbf.NaryFormula(qbf.Connective.AND, [qbf.NaryFormula(qbf.Connective.OR, [formula])])
    elif isinstance(formula, qbf.BinaryFormula):
        formula = matrix_to_cnf_distributivity(qbf.NaryFormula(formula.connective, [formula.left, formula.right]))
    elif isinstance(formula, qbf.NaryFormula):
        cnfs = [matrix_to_cnf_distributivity(f) for f in formula.formulas]
        if formula.connective == qbf.Connective.AND:
            # (C1 & C2 & ... & Cn) & (Cn+1 & ... & Cn+k) -> (C1 & C2 & ... & Cn & Cn+1 & ... & Cn+k) for CNF clauses C
            formula.formulas = [f for cnf in cnfs for f in cnf.formulas]
        elif formula.connective == qbf.Connective.OR:
            # (C1 & C2 & ... & Cn) | (Cn+1 & ... & Cn+k) -> (C1 | Cn+1) & (C1 | Cn+2) & ... & (Cn | Cn+k)
            fused_clauses = []
            for clauses in product(*[cnf.formulas for cnf in cnfs]):
                # remove duplicates, skip tautological clauses
                new_clause = [literal for clause in clauses for literal in clause.formulas]
                pos_literals = set([l for l in new_clause if not isinstance(l, qbf.NegFormula)])
                neg_literals = set([l.formula for l in new_clause if isinstance(l, qbf.NegFormula)])
                if len(pos_literals.intersection(neg_literals)) == 0:
                    new_clause = list(pos_literals) + [qbf.NegFormula(l) for l in neg_literals]
                    fused_clauses.append(new_clause)
            formula = qbf.NaryFormula(qbf.Connective.AND, [qbf.NaryFormula(qbf.Connective.OR, clause) for clause in fused_clauses])
    else:
        raise NotImplementedError(f"Encountered unknown formula: {formula} (type: {type(formula)}")

    return formula

def matrix_to_cnf_tseytin(formula, verbose=False) -> (qbf.NaryFormula, List[str]):
    if verbose:
        print(f"Starting Tseytin transformation for {formula}")
    # formula without quantifiers, in NNF, without -> or <->, with binary quantifiers replaced by Nary quantifiers
    # uses Tseytin transformation (https://personal.cis.strath.ac.uk/robert.atkey/cs208/converting-to-cnf.html)
    # result is satisfiable-equivalent to the original formula
    if isinstance(formula, qbf.NegFormula) or isinstance(formula, str):
        return qbf.NaryFormula(qbf.Connective.AND, [qbf.NaryFormula(qbf.Connective.OR, [formula])])
    assert isinstance(formula, qbf.NaryFormula)
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
            if not (isinstance(child, qbf.NegFormula) or isinstance(child, str)):
                queue.append((tseytin_var_counter, child))
                equation[1].append(f"tseytin_{tseytin_var_counter}")
                tseytin_var_counter += 1
            else:
                equation[1].append(child)
        tseytin_equations.append(equation)
    #if verbose:
    #    print(f"Tseytin equations:\n{"\n\t".join(str(eq) for eq in tseytin_equations)}")

    def neg(literal):
        if isinstance(literal, qbf.NegFormula):
            return literal.formula
        elif isinstance(literal, str):
            return qbf.NegFormula(literal)
        else:
            raise NotImplementedError(f"Encountered unknown literal type: {literal} (type: {type(literal)})")

    clauses = [qbf.NaryFormula(qbf.Connective.OR, ["tseytin_0"])]
    for i, eq in enumerate(tseytin_equations):
        if eq[0] == qbf.Connective.AND:
            # A & B => i turns into ~A | ~B | i
            clauses.append(qbf.NaryFormula(qbf.Connective.OR, [f"tseytin_{i}"] + [neg(l) for l in eq[1]]))
            # i => A & B turns into (A | ~i) & (B | ~i)
            for l in eq[1]:
                clauses.append(qbf.NaryFormula(qbf.Connective.OR, [l, neg(f"tseytin_{i}")]))
        elif eq[0] == qbf.Connective.OR:
            # eq[1] => i
            clauses.append(qbf.NaryFormula(qbf.Connective.OR, [neg(f"tseytin_{i}")] + [l for l in eq[1]]))
            # i => eq[1]
            for l in eq[1]:
                clauses.append(qbf.NaryFormula(qbf.Connective.OR, [neg(l), f"tseytin_{i}"]))

    return qbf.NaryFormula(qbf.Connective.AND, clauses), [f"tseytin_{i}" for i in range(tseytin_var_counter)]


def binary_to_nary(formula):
    # convert binary formulas to nary formulas
    # assume no quantifiers, NNF, no -> or <->
    if isinstance(formula, qbf.NegFormula):
        pass
    elif isinstance(formula, qbf.BinaryFormula):
        formula = binary_to_nary(qbf.NaryFormula(formula.connective, [formula.left, formula.right]))
    elif isinstance(formula, qbf.NaryFormula):
        # merge two levels of Nary formulas: (A & B) & (C & D) -> A & B & C & D
        children = [binary_to_nary(f) for f in formula.formulas]
        new_children = []
        for child in children:
            if isinstance(child, qbf.NaryFormula) and child.connective == formula.connective:
                new_children += child.formulas
            else:
                new_children.append(child)
        formula = qbf.NaryFormula(formula.connective, new_children)
    elif isinstance(formula, str):
        pass
    else:
        raise NotImplementedError(f"Encountered unknown formula type: {formula} (type: {type(formula)})")
    return formula


def qbf_to_cnf(formula, use_tseytin=True, verbose=False):
    formula = deepcopy(formula)
    if verbose:
        print(f"Converting QBF formula to CNF: {formula}")
    formula = qbf_to_nnf(formula)
    if verbose:
        print(f"Formula in NNF: {formula}")
    # formula is now in NNF without -> or <->
    formula = make_variables_unique(formula)
    if verbose:
        print(f"Formula with unique variables: {formula}")
    matrix, quantifiers = nnf_to_pnf(formula)
    if verbose:
        print(f"Matrix: {matrix}")
        print(f"qbf.Quantifiers: {quantifiers}")
    matrix = binary_to_nary(matrix)
    #matrix = remove_tautological_clauses(matrix)
    if use_tseytin:
        matrix, tseytin_vars = matrix_to_cnf_tseytin(matrix, verbose=verbose)
        # this step is a modification of the Tseytin algorithm to adapt it to QBF
        # originally, the Tseytin variables are (implicitly) existentially quantified, here we need to put
        # the quantifier on the inside to keep the equisatisfiability of the formula
        quantifiers.append((qbf.Quantifier.E, tseytin_vars))
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
                matrix = qbf.QuantifiedFormula(curr_quantifier, curr_variables, matrix)
                curr_quantifier, curr_variables = q, vs
    matrix = qbf.QuantifiedFormula(curr_quantifier, curr_variables, matrix)
    return matrix


def cnf_to_qdimacs(formula, add_comments=False):
    # convert CNF formula to QDIMACS format
    variable_names = dict()
    comments = []
    lines = ["p cnf 0 0"]
    while isinstance(formula, qbf.QuantifiedFormula):
        variable_ints = []
        for v in formula.variables:
            if v not in variable_names:
                variable_names[v] = len(variable_names) + 1
            variable_ints.append(variable_names[v])
        lines.append(f"{'e' if formula.quantifier == qbf.Quantifier.E else 'a'} {' '.join([str(v) for v in variable_ints])} 0")
        if add_comments:
            comments.append(f"{'e' if formula.quantifier == qbf.Quantifier.E else 'a'} {' '.join([str(v) for v in variable_ints])} 0  " + " ".join(str(v) for v in formula.variables))
        formula = formula.formula
    assert(isinstance(formula, qbf.NaryFormula) and formula.connective == qbf.Connective.AND)
    for clause in formula.formulas:
        variable_ints = []
        if isinstance(clause, qbf.NaryFormula) and clause.connective == qbf.Connective.OR:
            for literal in clause.formulas:
                negation = 1
                if isinstance(literal, qbf.NegFormula):
                    negation = -1
                    literal = literal.formula
                if literal not in variable_names:
                    variable_names[literal] = len(variable_names) + 1
                variable_ints.append(negation * variable_names[literal])
        elif isinstance(clause, NegFormula):
            if clause.formula not in variable_names:
                variable_names[clause.formula] = len(variable_names) + 1
            variable_ints.append(-1 * variable_names[clause.formula])
        elif isinstance(clause, str):
            if clause not in variable_names:
                variable_names[clause] = len(variable_names) + 1
            variable_ints.append(variable_names[clause])
        else:
            raise NotImplementedError(f"Expected clause to be a disjunction, got {clause} (type: {type(clause)})")
        lines.append(" ".join([str(v) for v in variable_ints]) + " 0")
        if add_comments:
            comments.append(" ".join([str(v) for v in variable_ints]) + " 0  " + str(clause))

    logging.debug(f"Created QDIMACS with {len(variable_names)} variables and {len(formula.formulas)} clauses")
    lines[0] = f"p cnf {len(variable_names)} {len(formula.formulas)}"
    return "\n".join([f"c {c}" for c in comments] + lines)

def demo():
    formula = qbf.BinaryFormula("A", qbf.Connective.AND, qbf.NegFormula(qbf.NegFormula("B")))
    formulaq = qbf.QuantifiedFormula(qbf.Quantifier.E, ["x1", "x2"], qbf.NaryFormula(qbf.Connective.OR, [formula, qbf.QuantifiedFormula(qbf.Quantifier.E, ["x3"], "x3"), qbf.QuantifiedFormula(qbf.Quantifier.A, ["x3"], qbf.BinaryFormula("A", qbf.Connective.AND, qbf.NegFormula(qbf.NegFormula("x3"))))]))
    print(formulaq)

    dnf_formula = qbf.BinaryFormula(
        qbf.BinaryFormula(
            qbf.BinaryFormula("P", qbf.Connective.OR, "Q"),
            qbf.Connective.AND,
            qbf.BinaryFormula("R", qbf.Connective.OR, "S")
        ),
        qbf.Connective.OR,
        qbf.BinaryFormula(
            qbf.BinaryFormula("A", qbf.Connective.OR, "B"),
            qbf.Connective.AND,
            qbf.BinaryFormula("C", qbf.Connective.OR, "D")
        )
    )
    print(qbf_to_cnf(formulaq))
    print(cnf_to_qdimacs(qbf_to_cnf(formulaq, verbose=True)))

    print(dnf_formula)
    nary_dnf_formula = binary_to_nary(dnf_formula)
    print(nary_dnf_formula)
    print(qbf_to_cnf(dnf_formula))
    print(cnf_to_qdimacs(qbf_to_cnf(dnf_formula)))

    formulaq = qbf.QuantifiedFormula(qbf.Quantifier.E, ["x1", "x2"], qbf.NaryFormula(qbf.Connective.OR, [formula, qbf.QuantifiedFormula(qbf.Quantifier.E, ["x3"], "x3"), qbf.QuantifiedFormula(qbf.Quantifier.A, ["x3"], qbf.BinaryFormula("A", qbf.Connective.AND, qbf.NegFormula(qbf.NegFormula("x3"))))]))
    unsat_formula = qbf.BinaryFormula(qbf.BinaryFormula("A", qbf.Connective.AND, qbf.NegFormula("A")), qbf.Connective.AND, formulaq)
    print(unsat_formula)
    #print(qbf_to_cnf(unsat_formula, verbose=True))
    print(cnf_to_qdimacs(qbf_to_cnf(unsat_formula, verbose=False)))

if __name__ == '__main__':
    #print(matrix_to_cnf_distributivity("x0"))
    f = qbf.QuantifiedFormula(qbf.Quantifier.E, ["0_Y"], qbf.BinaryFormula(qbf.QuantifiedFormula(qbf.Quantifier.E, ["0_Y"], "0_Y"), qbf.Connective.AND, "0_Y"))
    print(f)
    print(make_variables_unique(qbf_to_nnf(f)))