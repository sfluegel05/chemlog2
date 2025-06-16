import logging

from chemlog.msol import msol
from chemlog.qbf_classification import qbf
from chemlog.qbf_classification.qbf import NegFormula


class QBFTranslator(msol.MSOLCompiler):
    """
    Compiler for MSOL model checking problems into QBF.
    """

    def __init__(self, n_atoms: int, predicate_definitions: dict = None):
        """Number of atoms (i.e., domain size) is necessary for translation. Predicate definitions are needed for
        all predicates that take second-order variables. If a predicate holds for a given second-order variable (=a set),
        its translation should hold for the set (with special translations for set-operations such as subseteq).
        E.g., given the definition P(X) <-> X subseteq C,
         \forall Y P(Y) will be translated as \forall y1, ..., y_n_atoms: (y_1 -> 1_C) & ... & (y_n_atoms -> n_atoms_C)."""
        super().__init__()
        self.n_atoms = n_atoms
        if predicate_definitions is None:
            predicate_definitions = dict()
        self.predicate_definitions = predicate_definitions

    def visit(self, obj, **kwargs):
        if isinstance(obj, str):
            return obj
        meth = getattr(self, f"visit_{obj.__visit_name__}", None)
        if meth is None:
            raise NotImplementedError(f"Translation for MSOL->QBF does not support {obj.__visit_name__} yet.")
        return meth(obj, **kwargs)

    def visit_quantifier(self, quantifier: msol.Quantifier):
        raise NotImplementedError(
            "Quantifiers are not directly translatable to QBF. Use a different approach for quantification.")

    def visit_binary_connective(self, connective: msol.BinaryConnective, var_indices=None):
        if connective == msol.BinaryConnective.CONJUNCTION:
            return qbf.Connective.AND
        elif connective == msol.BinaryConnective.DISJUNCTION:
            return qbf.Connective.OR
        elif connective == msol.BinaryConnective.IMPLICATION:
            return qbf.Connective.IMPLIES
        elif connective == msol.BinaryConnective.BIIMPLICATION:
            return qbf.Connective.BIIMP
        else:
            raise NotImplementedError(f"Binary connective {connective} not supported in QBF Translator.")

    def visit_unary_connective(self, predicate: msol.UnaryConnective):
        raise NotImplementedError(
            "Unary connectives are not directly translatable to QBF. Use a different approach for negation.")

    def visit_unary_formula(self, formula: msol.UnaryFormula, var_indices=None):
        return qbf.NegFormula(self.visit(formula.formula, var_indices=var_indices))

    def visit_quantified_formula(self, formula: msol.QuantifiedFormula, var_indices=None):
        if not var_indices:
            var_indices = dict()
        vs = list(formula.variables)
        if len(vs) == 0:
            return self.visit(formula.formula, var_indices=var_indices)
        variable = vs[0]
        if isinstance(variable, msol.Var1):
            # \forall x: P(x) --> 1_P & ... & n_P
            # \exists x: P(x) --> 1_P | ... | n_P
            return qbf.NaryFormula(qbf.Connective.AND if formula.quantifier.is_universal() else qbf.Connective.OR,
                                   [self.visit(msol.QuantifiedFormula(formula.quantifier, vs[1:], formula.formula), var_indices={variable.symbol: i, **var_indices}) for i in
                                    range(self.n_atoms)])
        elif isinstance(variable, msol.Var2):
            # \forall X: P(X) --> \forall x1, ..., xn: "extracted definition of P"
            return qbf.QuantifiedFormula(qbf.Quantifier.A if formula.quantifier.is_universal() else qbf.Quantifier.E,
                                         [self.visit(variable, v2_index=i) for i in range(self.n_atoms)],
                                         self.visit(msol.QuantifiedFormula(formula.quantifier, vs[1:], formula.formula), var_indices=var_indices))
        raise AssertionError(f"Variable {variable} has to be specified as first-order or second-order.")

    def visit_binary_formula(self, formula: msol.BinaryFormula, var_indices=None):
        return qbf.BinaryFormula(self.visit(formula.left, var_indices=var_indices), self.visit(formula.operator, var_indices=var_indices), self.visit(formula.right, var_indices=var_indices))

    def visit_predicate_expression(self, expression: msol.PredicateExpression, var_indices=None):
        if var_indices is None:
            var_indices = dict()
        # predicates with second-order variables -> use definitions
        if expression.predicate in self.predicate_definitions:
            # If the predicate is defined, we use the definition
            definition = self.predicate_definitions[expression.predicate](*expression.arguments)
            return self.visit(definition, var_indices=var_indices)
        # first-order variables -> if they are bound, we expect to find them in var_indices
        var_symbols = []
        for v in expression.arguments:
            if isinstance(v, msol.Var1):
                var_symbols.append(self.visit(v, var_indices=var_indices))
            elif isinstance(v, msol.Var2):
                raise ValueError(
                    f"Second-order variable {v.symbol} used in predicate {expression.predicate} but no definition is provided.")
            else:
                raise NotImplementedError(f"Predicate translation is only implemented for variable arguments.")
        return f"{'_'.join(var_symbols)}_{expression.predicate}"

    def visit_variable(self, variable: msol.Variable, var_indices=None, v2_index=None):
        # var_indices are propagated for first-order variables, v2_index is used for second-order variables
        if var_indices is None:
            var_indices = dict()
        if isinstance(variable, msol.Var1):
            if variable.symbol in var_indices:
                return str(var_indices[variable.symbol])
            else:
                raise ValueError(f"Variable {variable.symbol} is not bound in the current context.")

        elif isinstance(variable, msol.Var2):
            assert v2_index is not None, "v2_index must be provided for second-order variables."
            return f"{v2_index}_{variable.symbol}"
        else:
            raise NotImplementedError(f"Variable {variable} has to be specified as first- or second-order.")

    def visit_set_of(self, expression: msol.SetOf):
        raise NotImplementedError(
            "Set of variables is not directly translatable to QBF. Use a different approach for set handling.")

    def visit_set_set_formula(self, formula: msol.SetSetFormula, var_indices=None):
        if formula.operator == msol.SetSetOperator.SUBSET:
            return qbf.BinaryFormula(
                self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right), var_indices=var_indices),
                qbf.Connective.AND,
                qbf.NegFormula(self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right), var_indices=var_indices))
            )
        elif formula.operator == msol.SetSetOperator.SUBSET_EQ:
            if isinstance(formula.left, msol.SetSetFunctorExpression):
                # A union B \subseteq C = A \subseteq C & B \subseteq C
                if formula.left.operator == msol.SetSetFunctor.UNION:
                    return qbf.BinaryFormula(
                        self.visit(msol.SetSetFormula(formula.left.left, msol.SetSetOperator.SUBSET_EQ, formula.right), var_indices=var_indices),
                        qbf.Connective.AND,
                        self.visit(msol.SetSetFormula(formula.left.right, msol.SetSetOperator.SUBSET_EQ, formula.right), var_indices=var_indices)
                    )
            elif isinstance(formula.right, msol.SetSetFunctorExpression):
                # A \subseteq B union C = a_i -> (b_i | c_i) for all i
                if formula.right.operator == msol.SetSetFunctor.UNION:
                    return qbf.NaryFormula(qbf.Connective.AND, [
                        qbf.BinaryFormula(
                            self.visit(formula.left, v2_index=i),
                            qbf.Connective.IMPLIES,
                            qbf.BinaryFormula(self.visit(formula.right.left, v2_index=i), qbf.Connective.OR, self.visit(formula.right.right, v2_index=i))
                        )
                        for i in range(self.n_atoms)
                    ])
            elif isinstance(formula.left, msol.SetOf) and isinstance(formula.right, msol.Var2):
                # {i, j, k} \subseteq X = x_i & x_j & x_k
                return qbf.NaryFormula(qbf.Connective.AND,
                                       [self.visit(formula.right, v2_index=self.visit(i, var_indices=var_indices))
                                        for i in formula.left.variables])
            elif isinstance(formula.left, msol.Var2) and isinstance(formula.right, msol.SetOf):
                # X \subseteq {i, j, k} = ~x_l for all l not in {i, j, k}
                return qbf.NaryFormula(qbf.Connective.AND, [NegFormula(self.visit(formula.left, v2_index=i))
                                                            for i in range(self.n_atoms) if
                                                            str(i) not in [self.visit(v, var_indices=var_indices) for v in formula.right.variables]])
            elif isinstance(formula.left, msol.Var2) and isinstance(formula.right, msol.Var2):
                # X \subseteq Y = x_i -> y_i for all i
                return qbf.NaryFormula(qbf.Connective.AND, [
                    qbf.BinaryFormula(self.visit(formula.left, v2_index=i), qbf.Connective.IMPLIES,
                                      self.visit(formula.right, v2_index=i))
                    for i in range(self.n_atoms)
                ])
        elif formula.operator == msol.SetSetOperator.SET_NEQ:
            # A != B iff ~(A = B)
            return qbf.NegFormula(self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right), var_indices=var_indices))
        elif formula.operator == msol.SetSetOperator.SET_EQ:
            # A = B iff A \subseteq B & B \subseteq A
            return qbf.BinaryFormula(
                self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right), var_indices=var_indices),
                qbf.Connective.AND,
                self.visit(msol.SetSetFormula(formula.right, msol.SetSetOperator.SUBSET_EQ, formula.left), var_indices=var_indices)
            )
        raise NotImplementedError(f"Set-set formula {formula} not supported in QBF Translator.")

    def visit_in_set_formula(self, formula: msol.InSetFormula, var_indices=None):
        if isinstance(formula.right, msol.SetSetFunctorExpression):
            if formula.right.operator == msol.SetSetFunctor.UNION:
                # x in A union B = x in A | x in B
                return qbf.BinaryFormula(
                    self.visit(msol.InSetFormula(formula.left, formula.right.left), var_indices=var_indices),
                    qbf.Connective.OR,
                    self.visit(msol.InSetFormula(formula.left, formula.right.right), var_indices=var_indices))
        elif isinstance(formula.right, msol.Var2):
            # i in X = i_x
            return self.visit(formula.right, v2_index=self.visit(formula.left, var_indices=var_indices))
        raise NotImplementedError(
            f"In-set formula {formula} not supported in QBF Translator.")

    def visit_nary_formula(self, formula: msol.NaryFormula, var_indices=None):
        return qbf.NaryFormula(self.visit(formula.operator, var_indices=var_indices),
                               [self.visit(arg, var_indices=var_indices) for arg in formula.formulae])


if __name__ == '__main__':

    a = msol.Var2("A")
    b = msol.Var2("B")
    x = msol.Var2("X")
    u, v = msol.Var1("u"), msol.Var1("v")
    t, w = msol.Var1("t"), msol.Var1("w")
    msol_formula = msol.QuantifiedFormula(
                msol.Quantifier.EXISTENTIAL, [t, w],
                msol.PredicateExpression("has_bond_to", [t, w])
            )
    msol_formula2 = msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [x], msol.QuantifiedFormula(
        msol.Quantifier.UNIVERSAL, [a, b],
        msol.BinaryFormula(
            msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [u], msol.InSetFormula(u, a))
            & msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [v], msol.InSetFormula(v, b))
            & msol.SetSetFormula(msol.SetSetFunctorExpression(a, msol.SetSetFunctor.UNION, b),
                                 msol.SetSetOperator.SET_EQ, x),
            msol.BinaryConnective.IMPLICATION,
            msol.QuantifiedFormula(
                msol.Quantifier.EXISTENTIAL, [t, w],
                msol.InSetFormula(t, a) & msol.InSetFormula(w, b) & msol.PredicateExpression("has_bond_to", [t, w])
            )
        )
    ))
    msol_formula3 = msol.SetSetFormula(msol.Var2("X"), msol.SetSetOperator.SET_EQ, msol.Var2("Y"))

    translator = QBFTranslator(n_atoms=3)
    qbf_formula = translator.visit(msol_formula3, var_indices={"a": 0})
    print(qbf_formula)