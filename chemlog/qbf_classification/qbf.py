import enum
from typing import List


class QBFExpression:

    def __repr__(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __lt__(self, other):
        if isinstance(other, QBFExpression):
            return self.__repr__() < other.__repr__()
        if isinstance(other, str):
            return self.__repr__() < other
        raise TypeError(f"Cannot compare {type(self)} with {type(other)}")

    def __gt__(self, other):
        if isinstance(other, QBFExpression):
            return self.__repr__() > other.__repr__()
        if isinstance(other, str):
            return self.__repr__() > other
        raise TypeError(f"Cannot compare {type(self)} with {type(other)}")

    def __hash__(self):
        return hash(self.__repr__())


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


class NegFormula(QBFExpression):

    def __init__(self, formula):
        self.formula = formula

    def __repr__(self):
        return f'{repr(Connective.NEG)}{self.formula}'


class BinaryFormula(QBFExpression):

    def __init__(self, left, connective: Connective, right):
        self.connective = connective
        if connective in [Connective.AND, Connective.OR]:
            self.left = min(left, right)
            self.right = max(left, right)
        else:
            self.left = left
            self.right = right

    def __repr__(self):
        return f"({self.left} {repr(self.connective)} {self.right})"

class NaryFormula(QBFExpression):

    def __init__(self, connective, formulas):
        self.connective = connective
        # remove duplicates and sort alphabetically
        self.formulas = list(set(formulas))
        self.formulas.sort()

    def __repr__(self):
        return "(" + f" {repr(self.connective)} ".join([str(f) for f in self.formulas]) + ")"

class QuantifiedFormula(QBFExpression):

    def __init__(self, quantifier: Quantifier, variables: List[str], formula):
        self.quantifier = quantifier
        self.variables = variables
        self.formula = formula

    def __repr__(self):
        return f"{repr(self.quantifier)}{'.'.join(str(v) for v in self.variables)}. {self.formula}"
