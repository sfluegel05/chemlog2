from gavel.logic.problem import AnnotatedFormula
from gavel.dialects.prolog.parser import PrologParser
from gavel.logic import logic

from chemlog.fol_to_py.folToPyTranslator import SO_KEY, PythonCompiler

def lopster_to_fol(lopster_path: str, remove_x=False) -> list[AnnotatedFormula]:
    """
    Translates a Lopster program into a list of AnnotatedFormulas in FOL.

    Args:
        lopster_path (str): Path to the Lopster program file.

    Returns:
        list[AnnotatedFormula]: List of AnnotatedFormulas representing the Lopster program in FOL.

    """

    parser = PrologParser()
    with open(lopster_path, 'r') as file:
        lopster_program = file.read()

    predicate_name_mapping = {
        "single": "bsingle",
        "double": "bdouble",
        "triple": "btriple",
        "minus3": "charge_m3",
        "minus2": "charge_m2",
        "minus1": "charge_m1",
        "plus1": "charge1",
        "plus2": "charge2",
        "plus3": "charge3",
        "bond": "has_bond_to",
    }
    for lopster_name, fol_name in predicate_name_mapping.items():
        lopster_program = lopster_program.replace(lopster_name, fol_name)
    annotated_formulas = parser.parse(lopster_program)

    # remove molecule/1 and hasAtom/2 predicates from bodies
    # optional: remove X variable
    for af in annotated_formulas:
        body = af.formula.formula.left
        new_formulae = []
        if isinstance(body, logic.NaryFormula):
            for atom in body.formulae:
                if isinstance(atom, logic.PredicateExpression):
                    if atom.predicate == "molecule" or atom.predicate == "hasAtom":
                        continue
                    if remove_x:
                        atom.arguments = [s for s in atom.arguments if s != logic.Variable("X")]
                if isinstance(atom, logic.UnaryFormula) and isinstance(atom.formula, logic.PredicateExpression):
                    if atom.formula.predicate == "molecule" or atom.formula.predicate == "hasAtom":
                        continue
                    if remove_x:
                        atom.formula.arguments = [s for s in atom.formula.arguments if s != logic.Variable("X")]
                new_formulae.append(atom)
            af.formula.formula.left = logic.NaryFormula(
                operator=logic.BinaryConnective.CONJUNCTION,
                formulae=new_formulae
            )
        elif isinstance(body, logic.PredicateExpression):
            if body.predicate == "molecule" or body.predicate == "hasAtom":
                af.formula.formula.left = logic.NaryFormula(
                    operator=logic.BinaryConnective.CONJUNCTION,
                    formulae=[]
                )
            if remove_x:
                body.arguments = [s for s in body.arguments if s != logic.Variable("X")]
        elif isinstance(body, logic.UnaryFormula) and isinstance(body.formula, logic.PredicateExpression):
            if body.formula.predicate == "molecule" or body.formula.predicate == "hasAtom":
                af.formula.formula.left = logic.NaryFormula(
                    operator=logic.BinaryConnective.CONJUNCTION,
                    formulae=[]
                )
            if remove_x:
                body.formula.arguments = [s for s in body.formula.arguments if s != logic.Variable("X")]
        if remove_x:
            head = af.formula.formula.right
            if isinstance(head, logic.PredicateExpression):
                head.arguments = [s for s in head.arguments if s != logic.Variable("X")]
            af.formula.formula.right = head
            af.formula.variables = [v for v in af.formula.variables if v.symbol != "X"]

    return annotated_formulas

def lopster_to_python(lopster_path: str, verbose: bool = False, remove_x: bool = False) -> str:
    """
    Translates a Lopster program into Python code.

    Args:
        lopster_path (str): Path to the Lopster program file.

    Returns:
        str: Python code representing the Lopster program.

    """

    fol_formulas = lopster_to_fol(lopster_path, remove_x=remove_x)
    # todo: turn implication rules into definitions
    # collect implications and convert to definitions
    # A -> C, B -> C becomes (A or B) <-> C
    bodies = dict()
    for formula in fol_formulas:
        assert isinstance(formula.formula, logic.QuantifiedFormula) and formula.formula.quantifier == logic.Quantifier.UNIVERSAL
        impl = formula.formula.formula
        assert isinstance(impl, logic.BinaryFormula) and impl.operator == logic.BinaryConnective.IMPLICATION
        head = impl.right
        body = impl.left
        head_key = str(head)
        # add body-specific quantifiers
        vars = set(formula.formula.variables)
        head_vars = set(head.arguments) if isinstance(head, logic.PredicateExpression) else set()
        # moving quantifier into the body turns universal quantifiers into existential ones
        vars -= head_vars
        if vars:
            body = logic.QuantifiedFormula(logic.Quantifier.EXISTENTIAL, list(vars), body)
        if head_key not in bodies:
            bodies[head_key] = []
        bodies[head_key].append((head, body))
    if verbose:
        for head_key, body_list in bodies.items():
            print(f"Head: {head_key}")
            for head, body in body_list:
                print(f"  Body: {body}")
    definitions = dict()
    for head_key, body_list in bodies.items():
        if len(body_list) > 1:
            combined_body = logic.NaryFormula(
                operator=logic.BinaryConnective.DISJUNCTION,
                formulae=[body for head, body in body_list]
            )
        else:
            combined_body = body_list[0][1]
        head = body_list[0][0]
        assert isinstance(head, logic.PredicateExpression)
        definition = logic.BinaryFormula(
            left=head,
            operator=logic.BinaryConnective.BIIMPLICATION,
            right=combined_body
        )
        definitions[head.predicate] = definition
    # todo: translate definitions to Python code using PythonCompiler
    compiler = PythonCompiler(mol=None)
    python_codes = []
    for name, definition in definitions.items():
        definition = f"def {name}({', '.join(['mol: Chem.Mol', SO_KEY] + [str(a) for a in definition.left.arguments])}):\n" \
                    f"    # {definition}\n" \
                    f"    return {compiler.visit(definition.right)}\n"
        python_codes.append(definition)
    return "\n".join(python_codes)

def run_lopster_translation():
    """
    Runs the Lopster to Python translation and writes the output to a file.
    """
    import os 
    lopster_file = os.path.join(os.path.dirname(__file__), 'lopster_rules_atom_level.pl')
    lopster_file_molecules = os.path.join(os.path.dirname(__file__), 'lopster_rules_molecule_level.pl')
    with open(os.path.join(os.path.dirname(__file__), 'lopster_python_new.py'), 'w+', encoding='utf-8') as f:
        f.write("from rdkit import Chem\n")
        f.write("# Generated Python code from Lopster rules\n\n")
        f.write(lopster_to_python(lopster_file, remove_x=False))
        f.write("\n\n")
        f.write(lopster_to_python(lopster_file_molecules, remove_x=True))
