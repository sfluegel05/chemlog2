import enum

from gavel.dialects.base.compiler import Compiler
from gavel.logic.logic import *



class SetTermExpression(TermExpression):
    """A term which is a second-order variable or a set of first-order variables"""
    pass

class Var1(Variable):

    def is_valid(self):
        return re.match(r"[a-z]\w*", self.symbol)


class Var2(Variable, SetTermExpression):
    pass

class SetOf(SetTermExpression):
    __visit_name__ = "set_of"

    def __init__(self, variables: Iterable[Var1]):
        self.variables = variables

    def __str__(self):
        return "{" + ", ".join(str(v) for v in self.variables) + "}"

    def symbols(self):
        return chain.from_iterable(v.symbols() for v in self.variables)

    def is_valid(self):
        return all(v.is_valid() for v in self.variables)


class SetSetFunctor(enum.Enum):
    __visit_name__ = "set_set_functor"
    UNION = 0
    INTERSECTION = 1
    DIFFERENCE = 2

    def __repr__(self):
        if self == SetSetFunctor.UNION:
            return "\u222A"
        elif self == SetSetFunctor.INTERSECTION:
            return "\u2229"
        elif self == SetSetFunctor.DIFFERENCE:
            return "\\"
        return None


class SetSetFunctorExpression(SetTermExpression):
    """A function which takes two sets and returns a set"""
    __visit_name__ = "set_set_functor_expression"

    def __init__(self, left: SetTermExpression, operator: SetSetFunctor, right: SetTermExpression):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return f"{self.left} {repr(self.operator.name)} {self.right}"

    def symbols(self):
        return chain(self.left.symbols(), self.right.symbols())

    def is_valid(self):
        return self.left.is_valid() and self.right.is_valid()

    def __eq__(self, other):
        if not isinstance(other, SetSetFunctorExpression):
            return False
        return (self.left == other.left and
                self.operator == other.operator and
                self.right == other.right)


class SetSetOperator(enum.Enum):
    __visit_name__ = "set_set_operator"
    SUBSET = 0
    SUBSET_EQ = 1
    SET_EQ = 2
    SET_NEQ = 3

    def __repr__(self):
        if self == SetSetOperator.SUBSET:
            return "\u2282"
        elif self == SetSetOperator.SUBSET_EQ:
            return "\u2286"

        elif self == SetSetOperator.SET_EQ:
            return "="
        elif self == SetSetOperator.SET_NEQ:
            return "\u2260"
        return None


class SetSetFormula(LogicExpression):
    __visit_name__ = "set_set_formula"
    """Relations between two sets"""
    def __init__(self, left: SetTermExpression, operator: SetSetOperator, right: SetTermExpression):
        self.left = left
        self.operator = operator
        self.right = right

    def __str__(self):
        return f"{self.left} {repr(self.operator.name)} {self.right}"

    def symbols(self):
        return chain(self.left.symbols(), self.right.symbols())

    def is_valid(self):
        return self.left.is_valid() and self.right.is_valid()

    def __eq__(self, other):
        if not isinstance(other, SetSetFormula):
            return False
        return (self.left == other.left and
                self.operator == other.operator and
                self.right == other.right)


class InSetFormula(LogicExpression):
    __visit_name__ = "in_set_formula"

    def __init__(self, left: Var1, right: SetTermExpression):
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} \u2208 {self.right}"

    def symbols(self):
        return chain(self.left.symbols(), self.right.symbols())

    def is_valid(self):
        return self.left.is_valid() and self.right.is_valid()

    def __eq__(self, other):
        if not isinstance(other, SetSetFormula):
            return False
        return (self.left == other.left and
                self.right == other.right)


class MSOLCompiler(Compiler):
    """Extends the base Compiler to handle MSOL-specific constructs."""

    def visit_set_of(self, expression: SetOf):
        raise NotImplementedError

    def visit_set_set_functor(self, functor: SetSetFunctor):
        raise NotImplementedError

    def visit_set_set_functor_expression(self, expression: SetSetFunctorExpression):
        raise NotImplementedError

    def visit_set_set_operator(self, operator: SetSetOperator):
        raise NotImplementedError

    def visit_set_set_formula(self, formula: SetSetFormula):
        raise NotImplementedError

    def visit_in_set_formula(self, formula: InSetFormula):
        raise NotImplementedError

    def visit_nary_formula(self, formula: NaryFormula):
        raise NotImplementedError