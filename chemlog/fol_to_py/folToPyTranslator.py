import inspect

from gavel.dialects.base.compiler import Compiler
from gavel.logic import logic as fol
from chemlog.msol.msol import Var2
from rdkit import Chem
from chemlog.msol.peptide_size import MSOLDefinition

SO_KEY = "so_elements"
PREDICATE_NAME_RDKIT_MAPPING = {
    "has_bond_to": lambda a, b: f"mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()) is not None",
    "bsingle": lambda a, b: f"(mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()) is not None and mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.SINGLE)",
    "bdouble": lambda a, b: f"(mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()) is not None and mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)",
    "btriple": lambda a, b: f"(mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()) is not None and mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.TRIPLE)",
    "baromatic": lambda a, b: f"(mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()) is not None and mol.GetBondBetweenAtoms({a}.GetIdx(), {b}.GetIdx()).GetBondType() == Chem.BondType.AROMATIC)",
    "is_in": lambda a, b: f"{a}.GetIdx() in {b}",
    "buildingblock": lambda a: f"{a} in {SO_KEY}",
    "chargen": lambda a: f"{a}.GetFormalCharge() < 0",
}
for i in range(1, 119):
    element = Chem.GetPeriodicTable().GetElementSymbol(i)
    PREDICATE_NAME_RDKIT_MAPPING[element.lower()] = lambda a, element=element: f"{a}.GetSymbol() == '{element}'"
for i in range(5):
    PREDICATE_NAME_RDKIT_MAPPING[f"has{i}hs"] = lambda a, i=i: f"{a}.GetTotalNumHs() == {i}"
for charge in range(-3, 4):
    charge_key = f"charge_m{-charge}" if charge < 0 else f"charge{charge}"
    PREDICATE_NAME_RDKIT_MAPPING[charge_key] = lambda a, charge=charge: f"{a}.GetFormalCharge() == {charge}"

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
        elif connective == fol.BinaryConnective.NEQ:
            return "!="
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
        return f"{self.visit(formula.quantifier)}({self.visit(formula.formula)} " + " ".join(f"for {self.visit(v)} in {SO_KEY}" if isinstance(v, Var2) else f"for {self.visit(v)} in mol.GetAtoms()" for v in formula.variables) + ")"

    def visit_binary_formula(self, formula: fol.BinaryFormula):
        if formula.operator == fol.BinaryConnective.IMPLICATION:
            return self.visit(~formula.left | formula.right)
        elif formula.operator == fol.BinaryConnective.BIIMPLICATION:
            return self.visit((~formula.left & ~formula.right) | (formula.left & formula.right))
        return f"({self.visit(formula.left)} {self.visit(formula.operator)} {self.visit(formula.right)})"

    def visit_nary_formula(self, formula: fol.NaryFormula):
        if len(list(formula.formulae)) == 0:
            return True
        if len(list(formula.formulae)) == 1:
            return self.visit(list(formula.formulae)[0])
        return f"({(' ' + self.visit(formula.operator) + ' ').join(self.visit(f) for f in formula.formulae)})"

    def visit_predicate_expression(self, expression: fol.PredicateExpression):
        # matching to mapping is not case-sensitive
        if expression.predicate.lower() in PREDICATE_NAME_RDKIT_MAPPING:
            return PREDICATE_NAME_RDKIT_MAPPING[expression.predicate.lower()](*map(self.visit, expression.arguments))
        else:
            # assume that the predicate is a python function defined (or translated) elsewhere
            # always pass the mol as first argument
            return f"{expression.predicate}(mol, {SO_KEY}, {', '.join(self.visit(arg) for arg in expression.arguments)})"

    def visit_variable(self, variable: fol.Variable):
        return variable.symbol

    def visit_constant(self, constant: fol.Constant):
        return constant.symbol

    def visit_msol_definition(self, definition: MSOLDefinition):
        """
        Visit a MSOL definition and return the Python code for it. (Note that the compiler can't actually handle
        MSOL formulas, this is only the definition structure)
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