from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from rdkit import Chem

from chemlog.fol_classification.fol_utils import normalize_fol_formula
from chemlog.fol_classification.model_checking import (
    ModelChecker,
)
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms


class ModelCheckerTestWrapper:
    def __init__(self) -> None:
        self.parser = TPTPParser()
        self._background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ] = {}

    def parse_formula(
        self, formula_str: str
    ) -> tuple[list[logic.Variable], logic.QuantifiedFormula]:
        formula_wrapped = f"fof(temp, axiom, {formula_str})."
        tptp_parsed = self.parser.parse(formula_wrapped)[0].formula
        pred_variables = self._extract_predicate_variables(tptp_parsed.left)
        return pred_variables, normalize_fol_formula(tptp_parsed.right)

    def add_background_definitions(self, def_dict: dict[str, str]):
        for _, def_str in def_dict.items():
            formula_wrapped = f"fof(temp, axiom, {def_str})."
            tptp_parsed = self.parser.parse(formula_wrapped)[0].formula
            pred_name = str(tptp_parsed.left.predicate)
            vars = self._extract_predicate_variables(tptp_parsed.left)
            normalized_formula = normalize_fol_formula(tptp_parsed.right)
            self._background_definitions[pred_name] = (vars, normalized_formula)

    def check_formula_for_molecule(self, formula_str: str, molecule: Chem.Mol) -> bool:
        _, tptp_parsed = self.parse_formula(formula_str)
        universe, extensions = mol_to_fol_atoms(molecule)
        model_checker = ModelChecker(universe, extensions, self._background_definitions)
        outcome, _ = model_checker.find_model(tptp_parsed)
        return outcome

    def _extract_predicate_variables(
        self, formula_left_side: logic.PredicateExpression
    ) -> list[logic.Variable]:
        """Extract the variables from a predicate definition string.

        For a definition like `new_predicate(X1, X2) <=> ?[X3]: (...)`
        This extracts [X1, X2] from the predicate call on the left side of the biimplication.
        """
        # Extract variables from the predicate expression
        variables = []
        if isinstance(formula_left_side, logic.PredicateExpression):
            # The arguments should be Variable objects
            if hasattr(formula_left_side, "arguments") and formula_left_side.arguments:
                for arg in formula_left_side.arguments:
                    if isinstance(arg, logic.Variable):
                        variables.append(arg)

        return variables
