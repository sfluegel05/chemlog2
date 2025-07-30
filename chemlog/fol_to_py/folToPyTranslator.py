import inspect

from gavel.dialects.base.compiler import Compiler
from gavel.logic import logic as fol
from rdkit import Chem
from chemlog.msol.peptide_size import MSOLDefinition

PREDICATE_NAME_RDKIT_MAPPING = {
    "has_bond_to": lambda a, b: f"mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()) is not None",
    "bSINGLE": lambda a, b: f"mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.SINGLE",
    "bDOUBLE": lambda a, b: f"mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.DOUBLE",
    "bTRIPLE": lambda a, b: f"mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.TRIPLE",
    "bAROMATIC": lambda a, b: f"mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.AROMATIC"
}
for i in range(1, 119):
    element = Chem.GetPeriodicTable().GetElementSymbol(i)
    PREDICATE_NAME_RDKIT_MAPPING[element] = lambda a, element=element: f"{a}.GetSymbol() == '{element}'"
for i in range(5):
    PREDICATE_NAME_RDKIT_MAPPING[f"Has{i}Hs"] = lambda a, i=i: f"{a}.GetTotalNumHs() == {i}"
for charge in range(-3, 4):
    PREDICATE_NAME_RDKIT_MAPPING[f"Charge{charge}"] = lambda a, charge=charge: f"{a}.GetFormalCharge() == {charge}"

class PythonCompiler(Compiler):
    """
    A compiler that translates FOL into Python code. Note that this is specific to chemistry and only works under
    the assumption that the domain is the atoms of a molecule (the molecule has to be provided to the compiler).
    """

    def __init__(self, mol: Chem.Mol):
        self.mol = mol

    def visit_quantifier(self, quantifier: fol.Quantifier):
        if quantifier.is_universal():
            return "all"
        elif quantifier.is_existential():
            return "any"
        else:
            raise NotImplementedError(f"Quantifier {quantifier} not supported in Python Compiler.")

    def visit_binary_connective(self, connective: fol.BinaryConnective):
        if connective == fol.BinaryConnective.CONJUNCTION:
            return "and"
        elif connective == fol.BinaryConnective.DISJUNCTION:
            return "or"
        else:
            raise NotImplementedError(f"Binary connective {connective} not supported in Python Compiler.")

    def visit_unary_connective(self, predicate: fol.UnaryConnective):
        if predicate == fol.UnaryConnective.NEGATION:
            return "not"
        else:
            raise NotImplementedError(f"Unary connective {predicate} not supported in Python Compiler.")

    def visit_unary_formula(self, formula: fol.UnaryFormula):
        return f"{self.visit(formula.connective)}({self.visit(formula.formula)})"

    def visit_quantified_formula(self, formula: fol.QuantifiedFormula):
        # for loop
        return f"{self.visit(formula.quantifier)}({self.visit(formula.formula)} " + "".join(f"for {self.visit(v)} in mol.GetAtoms()" for v in formula.variables) + ")"

    def visit_binary_formula(self, formula: fol.BinaryFormula):
        if formula.operator == fol.BinaryConnective.IMPLICATION:
            return self.visit(~formula.left | formula.right)
        elif formula.operator == fol.BinaryConnective.BIIMPLICATION:
            return self.visit((~formula.left & ~formula.right) | (formula.left & formula.right))
        return f"({self.visit(formula.left)} {self.visit(formula.operator)} {self.visit(formula.right)})"

    def visit_predicate_expression(self, expression: fol.PredicateExpression):
        if expression.predicate in PREDICATE_NAME_RDKIT_MAPPING:
            return PREDICATE_NAME_RDKIT_MAPPING[expression.predicate](*map(self.visit, expression.arguments))
        else:
            # assume that the predicate is a python function defined (or translated) elsewhere
            # always pass the mol as first argument
            return f"{expression.predicate}(mol, {', '.join(self.visit(arg) for arg in expression.arguments)})"

    def visit_variable(self, variable: fol.Variable):
        return variable.symbol

    def visit_constant(self, constant: fol.Constant):
        return constant.symbol

    def visit_msol_definition(self, definition: MSOLDefinition):
        """
        Visit a MSOL definition and return the Python code for it.
        """
        name = definition.name()
        # get signature of definition
        signature = inspect.signature(definition.__call__)
        return f"def {name}({', '.join(['mol: Chem.Mol'] + [f'{param.name}: {param.annotation.__name__}' for param in signature.parameters.values()])}):\n" \
               f"    return {self.visit(definition(*[param.annotation(param.name) for param in signature.parameters.values()]))}"


# for testing, define alcohol

class Saturated(MSOLDefinition):

    def name(self):
        return "Saturated"

    def __call__(self, atom: fol.Variable):
        # an atom is saturated if it only has single bonds to other atoms
        x_var = fol.Variable("x")
        return fol.QuantifiedFormula(fol.Quantifier.UNIVERSAL, [x_var],
                                     ~fol.PredicateExpression("has_bond_to", [x_var, atom])
                                     | fol.PredicateExpression("bSINGLE", [x_var, atom]))
class Alcohol(MSOLDefinition):
    def name(self):
        return "Alcohol"

    def __call__(self):
        o_var = fol.Variable("O")
        c_var = fol.Variable("C")
        return fol.QuantifiedFormula(
            fol.Quantifier.EXISTENTIAL,
            [o_var, c_var],
            fol.PredicateExpression("O", [o_var]) & fol.PredicateExpression("C", [c_var])
            & fol.PredicateExpression("Charge0", [o_var]) & fol.PredicateExpression("Has1Hs", [o_var])
            & fol.PredicateExpression(Saturated().name(), [c_var])
            & fol.PredicateExpression("has_bond_to", [o_var, c_var])
        )

if __name__ == "__main__":
    # Example usage
    print(PREDICATE_NAME_RDKIT_MAPPING)
    mol = Chem.MolFromSmiles("CCO")  # Ethanol
    compiler = PythonCompiler(mol)
    alcohol_definition = Alcohol()
    print(compiler.visit_msol_definition(Saturated()))  # Should print the Python code for the Alcohol definition
    print(compiler.visit_msol_definition(alcohol_definition))  # Should print the Python code for the Alcohol definition