from chemlog.msol import msol

class MONACompiler(msol.MSOLCompiler):
    """
    Compiler for MONA dialect.
    """

    def visit(self, obj, **kwargs):
        if isinstance(obj, str):
            return obj
        meth = getattr(self, f"visit_{obj.__visit_name__}", None)
        if meth is None:
            raise NotImplementedError(f"Compiler for MONA dialect does not support {obj.__visit_name__} yet.")
        return meth(obj, **kwargs)

    def visit_quantifier(self, quantifier: msol.Quantifier):
        if quantifier.is_universal():
            return "all"
        else:
            return "ex"

    def visit_binary_connective(self, connective: msol.BinaryConnective):
        if connective == msol.BinaryConnective.CONJUNCTION:
            return "&"
        elif connective == msol.BinaryConnective.DISJUNCTION:
            return "|"
        elif connective == msol.BinaryConnective.IMPLICATION:
            return "=>"
        elif connective == msol.BinaryConnective.BIIMPLICATION:
            return "<=>"
        else:
            raise NotImplementedError(f"Binary connective {connective} not supported in MONA Compiler.")

    def visit_unary_connective(self, predicate: msol.UnaryConnective):
        if predicate == msol.UnaryConnective.NEGATION:
            return "~"
        else:
            raise NotImplementedError(f"Unary connective {predicate} not supported in MONA Compiler.")

    def visit_unary_formula(self, formula: msol.UnaryFormula):
        return f"{self.visit(formula.connective)}{self.visit(formula.formula)}"

    def visit_quantified_formula(self, formula: msol.QuantifiedFormula):
        quantifier = self.visit(formula.quantifier)
        q_var_combinations = []
        for variable in formula.variables:
            if isinstance(variable, msol.Var1):
                q_var_combinations.append(f"{quantifier}1 {self.visit(variable)}: ")
            elif isinstance(variable, msol.Var2):
                q_var_combinations.append(f"{quantifier}2 {self.visit(variable)}: ")
            else:
                raise AssertionError(f"Variable {variable} has to be specified as first-order or second-order.")
        return "".join(q_var_combinations) + f"{self.visit(formula.formula)}"

    def visit_binary_formula(self, formula: msol.BinaryFormula):
            return f"({self.visit(formula.left)} {self.visit(formula.operator)} {self.visit(formula.right)})"

    def visit_predicate_expression(self, expression: msol.PredicateExpression):
        return f"{expression.predicate}({', '.join(self.visit(arg) for arg in expression.arguments)})"

    def visit_variable(self, variable: msol.Variable):
        return variable.symbol

    def visit_constant(self, constant: msol.Constant):
        return constant.symbol

    def visit_set_of(self, expression: msol.SetOf):
        return "{" + ", ".join(self.visit(v) for v in expression.variables) + "}"

    def visit_set_set_functor(self, functor: msol.SetSetFunctor):
        if functor == msol.SetSetFunctor.UNION:
            return "union"
        elif functor == msol.SetSetFunctor.INTERSECTION:
            return "inter"
        elif functor == msol.SetSetFunctor.DIFFERENCE:
            return "\\"
        else:
            raise NotImplementedError(f"Set-set functor {functor} not supported in MONA Compiler.")

    def visit_set_set_functor_expression(self, expression: msol.SetSetFunctorExpression):
        return f"({self.visit(expression.left)} {self.visit(expression.operator)} {self.visit(expression.right)})"

    def visit_set_set_operator(self, operator: msol.SetSetOperator):
        # MONA does not distinguish between real subset and subset with equality -> handled in set_set_formula
        if operator == msol.SetSetOperator.SUBSET:
            return "sub"
        elif operator == msol.SetSetOperator.SUBSET_EQ:
            return "sub"
        elif operator == msol.SetSetOperator.SET_EQ:
            return "="
        elif operator == msol.SetSetOperator.SET_NEQ:
            return "~="
        else:
            raise NotImplementedError(f"Set-set operator {operator} not supported in MONA Compiler.")

    def visit_set_set_formula(self, formula: msol.SetSetFormula):
        if formula.operator == msol.SetSetOperator.SUBSET:
            return self.visit(msol.BinaryFormula(
                msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right),
                msol.BinaryConnective.CONJUNCTION,
                msol.UnaryFormula(msol.UnaryConnective.NEGATION, msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right))
            ))
        return f"({self.visit(formula.left)} {self.visit(formula.operator)} {self.visit(formula.right)})"

    def visit_in_set_formula(self, formula: msol.InSetFormula):
        return f"{self.visit(formula.left)} in {self.visit(formula.right)}"

    def visit_nary_formula(self, formula: msol.NaryFormula):
        return f"({f' {self.visit(formula.operator)} '.join(self.visit(f) for f in formula.formulae)})"