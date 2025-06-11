from chemlog.msol import msol


class FOLTranslator(msol.MSOLCompiler):

    def __init__(self, so_predicate_variables=None, predicate_definitions=None):
        super().__init__()
        # this translator distinguishes between two types of second-order variables: Those that become predicates in
        # FOL and those that become FOL variables: a in C means that an atom a is a carbon atom (MSOL-style version
        # of C(a)), but \exists X: a in X means that a is part of a set X (in FOL, represented by a relation in(a, X))
        if so_predicate_variables is None:
            so_predicate_variables = []
        self.so_predicate_variables = so_predicate_variables

        if predicate_definitions is None:
            predicate_definitions = dict()
        self.predicate_definitions = predicate_definitions

    def visit_quantifier(self, quantifier: msol.Quantifier):
        return quantifier

    def visit_binary_connective(self, connective: msol.BinaryConnective):
        return connective

    def visit_unary_connective(self, connective: msol.UnaryConnective):
        return connective

    def visit_unary_formula(self, formula: msol.UnaryFormula):
        return msol.UnaryFormula(self.visit(formula.connective), self.visit(formula.formula))

    def visit_quantified_formula(self, formula: msol.QuantifiedFormula):
        return msol.QuantifiedFormula(self.visit(formula.quantifier), [self.visit(var) for var in formula.variables], self.visit(formula.formula))

    def visit_binary_formula(self, formula: msol.BinaryFormula):
        return msol.BinaryFormula(self.visit(formula.left), self.visit(formula.operator), self.visit(formula.right))

    def visit_predicate_expression(self, expression: msol.PredicateExpression):
        if expression.predicate in self.predicate_definitions:
            # If the predicate is defined, we use the definition
            definition = self.predicate_definitions[expression.predicate](*expression.arguments)
            return self.visit(definition)
        return msol.PredicateExpression(self.visit(expression.predicate), [self.visit(arg) for arg in expression.arguments])

    def visit_variable(self, variable: msol.Variable):
        return variable

    def visit_set_of(self, expression: msol.SetOf):
        raise NotImplementedError(
            "Set of variables is not directly translatable to FOL. Use a different approach for set handling.")

    def visit_set_set_formula(self, formula: msol.SetSetFormula):
        if formula.operator == msol.SetSetOperator.SUBSET:
            return msol.BinaryFormula(
                self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right)),
                msol.BinaryConnective.CONJUNCTION,
                ~self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right))
            )
        elif formula.operator == msol.SetSetOperator.SUBSET_EQ:
            if isinstance(formula.left, msol.SetSetFunctorExpression):
                # A union B \subseteq C = A \subseteq C & B \subseteq C
                if formula.left.operator == msol.SetSetFunctor.UNION:
                    return msol.BinaryFormula(
                        self.visit(msol.SetSetFormula(formula.left.left, msol.SetSetOperator.SUBSET_EQ, formula.right)),
                        msol.BinaryConnective.CONJUNCTION,
                        self.visit(msol.SetSetFormula(formula.left.right, msol.SetSetOperator.SUBSET_EQ, formula.right))
                    )
            elif isinstance(formula.left, msol.Var2) and isinstance(formula.right, msol.Var2):
                return msol.PredicateExpression("subset_eq", [self.visit(formula.left), self.visit(formula.right)])
        elif formula.operator == msol.SetSetOperator.SET_NEQ:
            # A != B iff ~(A = B)
            return msol.UnaryFormula(msol.UnaryConnective.NEGATION, self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right)))
        elif formula.operator == msol.SetSetOperator.SET_EQ:
            # A = B iff A \subseteq B & B \subseteq A
            return msol.BinaryFormula(
                self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right)),
                msol.BinaryConnective.CONJUNCTION,
                self.visit(msol.SetSetFormula(formula.right, msol.SetSetOperator.SUBSET_EQ, formula.left))
            )
        raise NotImplementedError(f"Set-set formula {formula} not supported in FOL Translator.")

    def visit_in_set_formula(self, formula: msol.InSetFormula):
        if isinstance(formula.right, msol.SetOf):
            # a in {b, c, d} iff a == b or a == c or a == d
            return msol.NaryFormula(
                msol.BinaryConnective.DISJUNCTION,
                [msol.BinaryFormula(self.visit(formula.left), msol.BinaryConnective.EQ, self.visit(arg))
                 for arg in formula.right.variables]
            )
        elif isinstance(formula.right, msol.Var2):
            if formula.right.symbol in self.so_predicate_variables:
                return msol.PredicateExpression(formula.right.symbol.lower(), [self.visit(formula.left)])
            return msol.PredicateExpression("in", [self.visit(formula.left), self.visit(formula.right)])
        raise NotImplementedError(
            f"In-set formula {formula} not supported in FOL Translator. "
        )

    def visit_nary_formula(self, formula: msol.NaryFormula):
        return msol.NaryFormula(self.visit(formula.operator),
                               [self.visit(arg) for arg in formula.formulae])
