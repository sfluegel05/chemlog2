import re

from gavel.dialects.tptp.compiler import TPTPCompiler
from gavel.logic import logic as fol

from chemlog.msol import msol



class TPTPMSOLCompiler(TPTPCompiler, msol.MSOLCompiler):
    """Compiler for MSOL formulas in TPTP format."""

    def visit_set_of(self, expression: msol.SetOf, set_set_formula_variable: msol.Var1 = None):
        if set_set_formula_variable is None:
            raise ValueError("A set_of expression may only occur inside a set_set_formula.")
        return self.visit(msol.NaryFormula(msol.BinaryConnective.DISJUNCTION, [msol.BinaryFormula(set_set_formula_variable, msol.BinaryConnective.EQ, v) for v in expression.variables]))

    def visit_set_set_functor_expression(self, expression: msol.SetSetFunctorExpression, set_set_formula_variable: msol.Var1 = None):
        if set_set_formula_variable is None:
            raise ValueError("A set_set_functor_expression may only occur inside a set_set_formula.")
        if expression.operator == msol.SetSetFunctor.UNION:
            return self.visit(expression.left, set_set_formula_variable= set_set_formula_variable) | self.visit(expression.right, set_set_formula_variable=set_set_formula_variable)
        elif expression.operator == msol.SetSetFunctor.INTERSECTION:
            return self.visit(expression.left, set_set_formula_variable=set_set_formula_variable) & self.visit(expression.right, set_set_formula_variable=set_set_formula_variable)
        elif expression.operator == msol.SetSetFunctor.DIFFERENCE:
            return self.visit(expression.left, set_set_formula_variable=set_set_formula_variable) & ~self.visit(expression.right, set_set_formula_variable=set_set_formula_variable)
        else:
            raise NotImplementedError(expression.operator)

    def visit_set_set_formula(self, formula: msol.SetSetFormula):
        if formula.operator == msol.SetSetOperator.SET_NEQ:
            return self.visit(~msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right))
        elif formula.operator == msol.SetSetOperator.SET_EQ:
            set_set_variable = msol.Var1("x_set_set")
            return self.visit(msol.QuantifiedFormula(
                msol.Quantifier.UNIVERSAL, [set_set_variable],
                msol.BinaryFormula(
                    self.visit(formula.left, set_set_variable = set_set_variable),
                    msol.BinaryConnective.BIIMPLICATION,
                    self.visit(formula.right, set_set_variable=set_set_variable)
                )))
        elif formula.operator == msol.SetSetOperator.SUBSET:
            return self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right) & ~msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right))
        elif formula.operator == msol.SetSetOperator.SUBSET_EQ:
            set_set_variable = msol.Var1("x_set_set")
            return self.visit(msol.QuantifiedFormula(
                msol.Quantifier.UNIVERSAL, [set_set_variable],
                msol.BinaryFormula(
                    self.visit(formula.left, set_set_variable=set_set_variable),
                    msol.BinaryConnective.IMPLICATION,
                    self.visit(formula.right, set_set_variable=set_set_variable)
                )))
        else:
            raise NotImplementedError(formula.operator)

    def visit_variable(self, variable: msol.Variable, set_set_formula_variable: msol.Var1 = None):
        if set_set_formula_variable is not None and isinstance(variable, msol.Var2):
            # second-order variable in a set-set formula needs to be connected to the rest of the formula
            return f"{self.visit(variable)} @ {self.visit(set_set_formula_variable)}"
        name = ""
        if variable.symbol:
            name = self.shorten_name(variable.symbol)
            name = name[:1].upper() + name[1:]
            name = re.sub("[^A-z_0-9]", "_", name)
            self.name_mapping[variable.symbol] = name
        return name

    def visit_in_set_formula(self, formula: msol.InSetFormula):
        if isinstance(formula.right, msol.Var2):
            return f"{self.visit(formula.right)} @ {self.visit(formula.left)}"
        raise NotImplementedError(formula)

    def visit_nary_formula(self, formula: msol.NaryFormula):
        formula.formulae = list(formula.formulae)
        if len(formula.formulae) == 0:
            return ""
        if len(formula.formulae) == 1:
            return self.visit(formula.formulae[0])
        if len(formula.formulae) == 2:
            return self.visit(msol.BinaryFormula(formula.formulae[0], formula.operator, formula.formulae[1]))
        return self.visit(msol.BinaryFormula(formula.formulae[0], formula.operator, msol.NaryFormula(formula.operator, formula.formulae[1:])))

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula):
        # swap standard variables with typed formulas
        # first-order variables get type i, second-order variables get type i > $o
        return "{}[{}]:{}".format(
            self.visit(formula.quantifier),
            ",".join(map(lambda v: self.visit(msol.TypedVariable(v.symbol, "i > $o" if isinstance(v, msol.Var2) else "i")), formula.variables)),
            self.parenthesise(formula.formula)
        )

    def visit_predicate_expression(self, expression: fol.PredicateExpression):
        name = ""
        if expression.predicate:
            name = self.shorten_name(expression.predicate)
            name = name[:1].lower() + name[1:]
            name = re.sub("[^A-z_0-9]", "_", name)
            self.name_mapping[expression.predicate] = name
        return f"{name} @ {'@'.join(map(self.visit, expression.arguments))}"



