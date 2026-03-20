import pytest
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
from rdkit import Chem

from chemlog.fol_classification.fol_utils import normalize_fol_formula
from chemlog.fol_classification.model_checking import ModelChecker
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms


@pytest.fixture
def checker():
    return ModelCheckerTestWrapper()


def test_when_predicate_is_used_as_constant(checker: "ModelCheckerTestWrapper"):
    # In this `has_bond_to(C1, c))`of formula, `c` predicate but is being used as a constant.
    # This test checks if the model checker raise appropriate error.
    formula_str = (
        "glycolipid <=> ?[O1, C1, O2, C2]: (o(O1) & "
        "has_0_hs(O1) & c(C1) & bSINGLE(O1, C1) & o(O2) & has_0_hs(O2) & bSINGLE(C1, O2) "
        "& c(C2) & bSINGLE(O2, C2) & has_1_hs(C1) & has_bond_to(C1, c))"
    )
    molecule = Chem.MolFromSmiles(
        "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
    )

    with pytest.raises(
        Exception,
        match="Predicate 'c' is being used as a constant in the formula."
        "Please check the formula and ensure that predicates are not used as constants.",
    ):
        checker.check_formula_for_molecule(formula_str, molecule)


def test_predicate_arity_mismatch(checker: "ModelCheckerTestWrapper"):
    molecule = Chem.MolFromSmiles(
        "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
    )
    formula_str = "triterpenoidSaponin(X) <=> (terpeneGlycoside(X) & triterpenoid(X))"
    add_defs_dict = {
        "terpeneGlycoside": "terpeneGlycoside(X) <=> (terpenoid(X) & glycoside(X))",
        "glycoside": "glycoside(X) <=> (organicMolecularEntity(X) & hasSugarMoietyAttachedByGlycosidicBond(X))",
        "organicMolecularEntity": "organicMolecularEntity(X) <=> (molecule(X) & c(X))",
        "molecule": "molecule <=> net_charge_neutral",
        "terpenoid": "terpenoid <=> molecule",
        "hasSugarMoietyAttachedByGlycosidicBond": "hasSugarMoietyAttachedByGlycosidicBond(X) <=> ?[A1, A2, A3]: (c(A1) & o(A2) & o(A3) & inRing(A1) & inRing(A2) & bSINGLE(A1, A2) & bSINGLE(A1, A3) & hasRingOxygen(A2))",
        "triterpenoid": "triterpenoid <=> (terpenoid & ?[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]: (c(A1) & c(A2) & c(A3) & c(A4) & c(A5) & c(A6) & c(A7) & c(A8) & c(A9) & c(A10) & c(A11) & c(A12) & c(A13) & c(A14) & c(A15) & has_bond_to(A1, A2) & has_bond_to(A2, A3) & has_bond_to(A3, A4) & has_bond_to(A4, A5) & has_bond_to(A5, A6) & has_bond_to(A6, A7) & has_bond_to(A7, A8) & has_bond_to(A8, A9) & has_bond_to(A9, A10) & has_bond_to(A10, A11) & has_bond_to(A11, A12) & has_bond_to(A12, A13) & has_bond_to(A13, A14) & has_bond_to(A14, A15) & A1 != A2 & A1 != A3 & A1 != A4 & A1 != A5 & A1 != A6 & A1 != A7 & A1 != A8 & A1 != A9 & A1 != A10 & A1 != A11 & A1 != A12 & A1 != A13 & A1 != A14 & A1 != A15 & A2 != A3 & A2 != A4 & A2 != A5 & A2 != A6 & A2 != A7 & A2 != A8 & A2 != A9 & A2 != A10 & A2 != A11 & A2 != A12 & A2 != A13 & A2 != A14 & A2 != A15 & A3 != A4 & A3 != A5 & A3 != A6 & A3 != A7 & A3 != A8 & A3 != A9 & A3 != A10 & A3 != A11 & A3 != A12 & A3 != A13 & A3 != A14 & A3 != A15 & A4 != A5 & A4 != A6 & A4 != A7 & A4 != A8 & A4 != A9 & A4 != A10 & A4 != A11 & A4 != A12 & A4 != A13 & A4 != A14 & A4 != A15 & A5 != A6 & A5 != A7 & A5 != A8 & A5 != A9 & A5 != A10 & A5 != A11 & A5 != A12 & A5 != A13 & A5 != A14 & A5 != A15 & A6 != A7 & A6 != A8 & A6 != A9 & A6 != A10 & A6 != A11 & A6 != A12 & A6 != A13 & A6 != A14 & A6 != A15 & A7 != A8 & A7 != A9 & A7 != A10 & A7 != A11 & A7 != A12 & A7 != A13 & A7 != A14 & A7 != A15 & A8 != A9 & A8 != A10 & A8 != A11 & A8 != A12 & A8 != A13 & A8 != A14 & A8 != A15 & A9 != A10 & A9 != A11 & A9 != A12 & A9 != A13 & A9 != A14 & A9 != A15 & A10 != A11 & A10 != A12 & A10 != A13 & A10 != A14 & A10 != A15 & A11 != A12 & A11 != A13 & A11 != A14 & A11 != A15 & A12 != A13 & A12 != A14 & A12 != A15 & A13 != A14 & A13 != A15 & A14 != A15))",
    }
    checker.add_background_definition(add_defs_dict)
    with pytest.raises(
        Exception,
        match="Predicate 'c' is being used as a constant in the formula."
        "Please check the formula and ensure that predicates are not used as constants.",
    ):
        checker.check_formula_for_molecule(formula_str, molecule)


def test_predicate_arity_mismatch_2(checker: "ModelCheckerTestWrapper"):
    molecule = Chem.MolFromSmiles(
        "C=1[C@@]2([C@]3(CC[C@]4([C@]([C@@]3(C=CC2=CC(C1)=O)[H])(CCC4=O)[H])C)[H])C"
    )

    formula_str = "threeOxoSteroid <=> (oxoSteroid & ?[A1, A2]: (c(A1) & o(A2) & bDOUBLE(A1, A2) & steroidPosition3(A1)))"

    add_def_dict = {
        "oxoSteroid": "oxoSteroid <=> (steroid & hasCarbonylGroup)",
        "steroidPosition3": "steroidPosition3(X) <=> (c(X) & inRing(X) & has_0_hs(X) & bDOUBLE(X, Y) & o(Y) & ?[A1, A2]: (c(A1) & c(A2) & bSINGLE(X, A1) & bSINGLE(X, A2) & inRing(A1) & inRing(A2) & A1 != A2))",
        "molecule": "molecule <=> net_charge_neutral",
        "hasCarbonylGroup": "hasCarbonylGroup <=> ?[C1, O1]: (c(C1) & o(O1) & bDOUBLE(C1, O1))",
        "steroid": "steroid <=> (molecule & ?[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17]: (c(A1) & c(A2) & c(A3) & c(A4) & c(A5) & c(A6) & c(A7) & c(A8) & c(A9) & c(A10) & c(A11) & c(A12) & c(A13) & c(A14) & c(A15) & c(A16) & c(A17) & has_bond_to(A1, A2) & has_bond_to(A2, A3) & has_bond_to(A3, A4) & has_bond_to(A4, A5) & has_bond_to(A5, A10) & has_bond_to(A10, A1) & has_bond_to(A5, A6) & has_bond_to(A6, A7) & has_bond_to(A7, A8) & has_bond_to(A8, A9) & has_bond_to(A9, A10) & has_bond_to(A8, A14) & has_bond_to(A14, A15) & has_bond_to(A15, A16) & has_bond_to(A16, A17) & has_bond_to(A17, A13) & has_bond_to(A13, A14) & has_bond_to(A9, A11) & has_bond_to(A11, A12) & has_bond_to(A12, A13)))",
    }
    checker.add_background_definition(add_def_dict)
    with pytest.raises(
        Exception,
        match="Predicate 'c' is being used as a constant in the formula."
        "Please check the formula and ensure that predicates are not used as constants.",
    ):
        checker.check_formula_for_molecule(formula_str, molecule)


class ModelCheckerTestWrapper:
    def __init__(self) -> None:
        self.parser = TPTPParser()
        self._background_definitions: dict[
            str, tuple[list[logic.Variable], logic.QuantifiedFormula]
        ] = {}

    def parse_formula(
        self, formula_str
    ) -> tuple[list[logic.Variable], logic.QuantifiedFormula]:
        formula_wrapped = f"fof(temp, axiom, {formula_str})."
        tptp_parsed = self.parser.parse(formula_wrapped)[0].formula
        pred_variables = self._extract_predicate_variables(tptp_parsed.left)
        return pred_variables, normalize_fol_formula(tptp_parsed.right)

    def add_background_definition(self, def_dict: dict[str, str]):
        for pred_name, def_str in def_dict.items():
            vars, normalized_formula = self.parse_formula(def_str)
            self._background_definitions[pred_name] = (vars, normalized_formula)

    def check_formula_for_molecule(self, formula_str: str, molecule: Chem.Mol) -> bool:
        _, tptp_parsed = self.parse_formula(formula_str)
        universe, extensions = mol_to_fol_atoms(molecule)
        model_checker = ModelChecker(universe, extensions, self._background_definitions)
        return model_checker.find_model(tptp_parsed)

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
