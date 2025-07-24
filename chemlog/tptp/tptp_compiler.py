import inspect
import re

from gavel.dialects.tptp.compiler import TPTPCompiler
from gavel.logic import logic as fol
from gavel.logic.problem import AnnotatedFormula, FormulaRole

from chemlog.msol import msol



class TPTPMSOLCompiler(TPTPCompiler, msol.MSOLCompiler):
    """Compiler for MSOL formulas in TPTP format."""

    def __init__(self, given_predicate_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # list of names that are assumed to be known as predicates. In expressions such as a \in P, P will be treated
        # as a predicate if it is in this list, otherwise it will be treated as a second-order variable (this is only
        # relevant for lower-case / upper-case distinction)
        if given_predicate_names is None:
            given_predicate_names = []
        self.given_predicate_names = [n.lower() for n in given_predicate_names]

    def visit_msol_definition(self, definition: msol.MSOLDefinition):
        name = definition.name()
        name = name[0].lower() + name[1:]  # first letter lowercase
        # get signature of definition
        signature = inspect.signature(definition.__call__)
        parameters = signature.parameters
        parameter_types = list(map(lambda key: self.visit(msol.TypedVariable(self.visit(parameters[key].annotation(key)), "($i > $o)" if parameters[key].annotation == msol.Var2 else "$i")),
                parameters))
        type_formula = AnnotatedFormula("thf", f"{name}_type", FormulaRole.TYPE, f"{name}: ({' > '.join(p.split(":")[-1] for p in parameter_types)}{' > ' if len(parameter_types) > 0 else ' '}$o)")
        # definition
        msol_formula = definition(**{name: value.annotation(name) for name, value in parameters.items()})
        lambda_block = " ^ [" + ", ".join(parameter_types) + "]:" if len(parameter_types) > 0 else ""
        def_formula = AnnotatedFormula(
            "thf", f"{name}", FormulaRole.AXIOM,
            f"({name} = ({lambda_block} ({self.visit(msol_formula)}) ))"
                                       )
        return [self.visit(type_formula), self.visit(def_formula)]

    def visit_set_of(self, expression: msol.SetOf, set_set_formula_variable: msol.Var1 = None):
        if set_set_formula_variable is None:
            raise ValueError("A set_of expression may only occur inside a set_set_formula.")
        return self.visit(msol.NaryFormula(msol.BinaryConnective.DISJUNCTION, [msol.BinaryFormula(set_set_formula_variable, msol.BinaryConnective.EQ, v) for v in expression.variables]))

    def visit_set_set_functor_expression(self, expression: msol.SetSetFunctorExpression, set_set_formula_variable: msol.Var1 = None):
        if set_set_formula_variable is None:
            raise ValueError("A set_set_functor_expression may only occur inside a set_set_formula.")
        if expression.operator == msol.SetSetFunctor.UNION:
            return "(" + self.visit(expression.left, set_set_formula_variable= set_set_formula_variable) + "   |   " + self.visit(expression.right, set_set_formula_variable=set_set_formula_variable) + ")"
        elif expression.operator == msol.SetSetFunctor.INTERSECTION:
            return "(" + self.visit(expression.left, set_set_formula_variable=set_set_formula_variable) + "  &  " +  self.visit(expression.right, set_set_formula_variable=set_set_formula_variable) + ")"
        elif expression.operator == msol.SetSetFunctor.DIFFERENCE:
            return "(" + self.visit(expression.left, set_set_formula_variable=set_set_formula_variable) + "&" + "~" + self.visit(expression.right, set_set_formula_variable=set_set_formula_variable) + ")"
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
                    self.visit(formula.left, set_set_formula_variable = set_set_variable),
                    msol.BinaryConnective.BIIMPLICATION,
                    self.visit(formula.right, set_set_formula_variable=set_set_variable)
                )))
        elif formula.operator == msol.SetSetOperator.SUBSET:
            return self.visit(msol.SetSetFormula(formula.left, msol.SetSetOperator.SUBSET_EQ, formula.right) & ~msol.SetSetFormula(formula.left, msol.SetSetOperator.SET_EQ, formula.right))
        elif formula.operator == msol.SetSetOperator.SUBSET_EQ:
            set_set_variable = msol.Var1("x_set_set")
            return self.visit(msol.QuantifiedFormula(
                msol.Quantifier.UNIVERSAL, [set_set_variable],
                msol.BinaryFormula(
                    self.visit(formula.left, set_set_formula_variable=set_set_variable),
                    msol.BinaryConnective.IMPLICATION,
                    self.visit(formula.right, set_set_formula_variable=set_set_variable)
                )))
        else:
            raise NotImplementedError(formula.operator)

    def visit_variable(self, variable: msol.Variable, set_set_formula_variable: msol.Var1 = None):
        if set_set_formula_variable is not None and isinstance(variable, msol.Var2):
            # second-order variable in a set-set formula needs to be connected to the rest of the formula
            return f"({self.visit(variable)} @ {self.visit(set_set_formula_variable)})"
        name = ""
        if variable.symbol:
            name = self.shorten_name(variable.symbol)
            if name.lower() in self.given_predicate_names:
                name = name[:1].lower() + name[1:]  # predicate names start with a lowercase letter (treat this second-order variable like a unary predicate)
            else:
                name = name[:1].upper() + name[1:]
            name = re.sub("[^A-z_0-9]", "_", name)
            self.name_mapping[variable.symbol] = name
        return name

    def visit_in_set_formula(self, formula: msol.InSetFormula):
        if isinstance(formula.right, msol.Var2):
            return f"({self.visit(formula.right)} @ {self.visit(formula.left)})"
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
        return "{}[{}]:({})".format(
            self.visit(formula.quantifier),
            ",".join(map(lambda v: self.visit(msol.TypedVariable(self.visit(v), "$i > $o" if isinstance(v, msol.Var2) else "$i")), formula.variables)),
            self.parenthesise(formula.formula)
        )

    def visit_predicate_expression(self, expression: fol.PredicateExpression):
        name = ""
        if expression.predicate:
            name = self.shorten_name(expression.predicate)
            name = name[:1].lower() + name[1:]
            name = re.sub("[^A-z_0-9]", "_", name)
            self.name_mapping[expression.predicate] = name
        if len(expression.arguments) == 0:
            return name
        return f"({name} @ {'@'.join(map(self.visit, expression.arguments))})"



