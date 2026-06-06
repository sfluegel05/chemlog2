import json
from pathlib import Path

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

    def load_background_definitions_from_json(
        self,
        file_path: str | Path,
        *,
        replace: bool = True,
    ) -> dict[str, tuple[list[logic.Variable], logic.QuantifiedFormula]]:
        """Load background definitions from a JSON file.

        Args:
            file_path: JSON file created by `save_background_definitions_to_json`.
            replace: If True, clear the current background definitions before loading.

        Returns:
            The loaded background definitions in the internal tuple format.
        """
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        print(f"Loading {len(payload)} background definitions from {file_path}")
        counter = 0
        loaded_definitions = {}
        for item in payload:
            predicate_name = item["predicate"]
            definition_str = item["definition"]
            try:
                pred_variables, fol_formula = self.parse_formula(definition_str)
            except Exception as e:
                counter += 1
                # print(f"Error parsing definition for {predicate_name}: {e}")
                continue
            loaded_definitions[predicate_name] = (pred_variables, fol_formula)

        print(f"Successfully loaded {len(loaded_definitions)} definitions")
        print(f"Failed to load {counter} definitions")
        if replace:
            self._background_definitions.clear()
        self._background_definitions.update(loaded_definitions)
        return loaded_definitions


if __name__ == "__main__":
    # Example usage
    checker = ModelCheckerTestWrapper()
    checker.load_background_definitions_from_json(
        "/home/staff/a/akhedekar/chebai-NL2FOL/nl_2_fol/inference/learner/validation_background_defs.json"
    )
