import pytest
from rdkit import Chem

from chemlog.fol_classification.model_checking import (
    ModelCheckerInputError,
    ModelCheckerOutcome,
)
from tests.utils import ModelCheckerTestWrapper


@pytest.fixture
def checker():
    return ModelCheckerTestWrapper()


def test_when_predicate_is_used_as_constant(checker: ModelCheckerTestWrapper):
    molecule = Chem.MolFromSmiles(
        "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
    )

    # No exception raised for this formula
    formula_str = (
        "glycolipid <=> ?[O1, C1, O2, C2]: (o(O1) & "
        "has_0_hs(O1) & c(C1) & bSINGLE(O1, C1) & o(O2) & has_0_hs(O2) & bSINGLE(C1, O2) "
        "& c(C2) & bSINGLE(O2, C2) & has_1_hs(C1))"
    )
    checker.check_formula_for_molecule(formula_str, molecule)

    # In this `has_bond_to(C1, c))`of formula, `c` predicate but is being used as a constant.
    # This test checks if the model checker raise appropriate error.
    formula_str = (
        "glycolipid <=> ?[O1, C1, O2, C2]: (o(O1) & "
        "has_0_hs(O1) & c(C1) & bSINGLE(O1, C1) & o(O2) & has_0_hs(O2) & bSINGLE(C1, O2) "
        "& c(C2) & bSINGLE(O2, C2) & has_1_hs(C1) & has_bond_to(C1, c))"
    )

    with pytest.raises(
        ModelCheckerInputError,
        match=r"Predicate 'c' is being used as a constant in the formula\.\s*"
        r"Please check the formula and ensure that predicates are not used as constants\.",
    ):
        checker.check_formula_for_molecule(formula_str, molecule)


def test_raise_missing_predicate_exception(checker: ModelCheckerTestWrapper):
    ethanol = Chem.MolFromSmiles("CCO")
    # Predicate `oneCarbonCompound` is not defined in the background definitions,
    # and is being used in the formula. This should raise MissingPredicateException.
    formula_str = (
        "carbonMonoxide <=> ?[A1, A2]: (oneCarbonCompound & c(A1) & o(A2)"
        " & has_bond_to(A1,A2))"
    )

    with pytest.raises(
        ModelCheckerInputError,
        match="Predicate 'oneCarbonCompound' is not defined",
    ):
        checker.check_formula_for_molecule(formula_str, ethanol)

    formula_str = (
        "carbonMonoxide <=> ?[A1, A2]: (oneCarbonCompound & "
        "~twoPlusCarbonCompound & c(A1) & o(A2) & has_bond_to(A1,A2))"
    )

    with pytest.raises(
        ModelCheckerInputError,
        match="Predicates 'oneCarbonCompound' and 'twoPlusCarbonCompound' are not defined",
    ):
        checker.check_formula_for_molecule(formula_str, ethanol)


def test_predicate_arity_exception(checker: ModelCheckerTestWrapper):
    """Test that exceptions in does_mol_match_tptp_definition are properly raised."""
    checker.add_background_definitions(
        {
            "ptest": "ptest <=> has_bond(X, Y)",  # ptest has no variables
            "qtest(X)": "qtest(X) <=> has_bond(X, Y)",  # qtest has one variable
            "rtest(X, Y)": "rtest(X, Y) <=> has_bond(X, Y)",  # rtest has two variables
        }
    )

    # Here, the formula reference ptest predicate with a variable,
    # but the background definition of ptest has no variables,
    # which should cause an error during model checking
    formula_str = "test_pred(X) <=> (ptest(X) & qtest(X, Y) & rtest(X, Y, Z))"

    # Create a simple molecule
    mol = Chem.MolFromSmiles("C")

    with pytest.raises(
        ModelCheckerInputError,
        match=r"(?s).*Predicate `ptest` is defined with arity 0 but called with 1 arguments.*"
        r"Predicate `qtest` is defined with arity 1 but called with 2 arguments.*"
        r"Predicate `rtest` is defined with arity 2 but called with 3 arguments.*",
    ):
        checker.check_formula_for_molecule(formula_str, mol)

    formula_str = "test_pred(X) <=> (ptest & qtest & rtest & rtest(X))"

    with pytest.raises(
        ModelCheckerInputError,
        match=r"(?s).*Predicate `qtest` is defined with arity 1 but called with 0 arguments.*"
        r"Predicate `rtest` is defined with arity 2 but called with 0 arguments.*"
        r"Predicate `rtest` is defined with arity 2 but called with 1 arguments.*",
    ):
        checker.check_formula_for_molecule(formula_str, mol)


def test_predicate_arity_mismatch(checker: ModelCheckerTestWrapper):
    molecule = Chem.MolFromSmiles(
        "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
    )
    formula_str = "triterpenoidSaponin(X) <=> (terpeneGlycoside(X) & triterpenoid(X))"
    add_defs_dict = {
        "terpeneGlycoside": "terpeneGlycoside(X) <=> (terpenoid(X) & glycoside(X))",
        # "glycoside": "glycoside(X) <=> (organicMolecularEntity(X) & hasSugarMoietyAttachedByGlycosidicBond(X))",
        # "organicMolecularEntity": "organicMolecularEntity(X) <=> (molecule(X) & c(X))",
        # "molecule": "molecule <=> net_charge_neutral",
        # "terpenoid": "terpenoid <=> molecule",
        # "hasSugarMoietyAttachedByGlycosidicBond": "hasSugarMoietyAttachedByGlycosidicBond(X) <=> ?[A1, A2, A3]: (c(A1) & o(A2) & o(A3) & inRing(A1) & inRing(A2) & bSINGLE(A1, A2) & bSINGLE(A1, A3) & hasRingOxygen(A2))",
        # "triterpenoid": "triterpenoid <=> (terpenoid & ?[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]: (c(A1) & c(A2) & c(A3) & c(A4) & c(A5) & c(A6) & c(A7) & c(A8) & c(A9) & c(A10) & c(A11) & c(A12) & c(A13) & c(A14) & c(A15) & has_bond_to(A1, A2) & has_bond_to(A2, A3) & has_bond_to(A3, A4) & has_bond_to(A4, A5) & has_bond_to(A5, A6) & has_bond_to(A6, A7) & has_bond_to(A7, A8) & has_bond_to(A8, A9) & has_bond_to(A9, A10) & has_bond_to(A10, A11) & has_bond_to(A11, A12) & has_bond_to(A12, A13) & has_bond_to(A13, A14) & has_bond_to(A14, A15) & A1 != A2 & A1 != A3 & A1 != A4 & A1 != A5 & A1 != A6 & A1 != A7 & A1 != A8 & A1 != A9 & A1 != A10 & A1 != A11 & A1 != A12 & A1 != A13 & A1 != A14 & A1 != A15 & A2 != A3 & A2 != A4 & A2 != A5 & A2 != A6 & A2 != A7 & A2 != A8 & A2 != A9 & A2 != A10 & A2 != A11 & A2 != A12 & A2 != A13 & A2 != A14 & A2 != A15 & A3 != A4 & A3 != A5 & A3 != A6 & A3 != A7 & A3 != A8 & A3 != A9 & A3 != A10 & A3 != A11 & A3 != A12 & A3 != A13 & A3 != A14 & A3 != A15 & A4 != A5 & A4 != A6 & A4 != A7 & A4 != A8 & A4 != A9 & A4 != A10 & A4 != A11 & A4 != A12 & A4 != A13 & A4 != A14 & A4 != A15 & A5 != A6 & A5 != A7 & A5 != A8 & A5 != A9 & A5 != A10 & A5 != A11 & A5 != A12 & A5 != A13 & A5 != A14 & A5 != A15 & A6 != A7 & A6 != A8 & A6 != A9 & A6 != A10 & A6 != A11 & A6 != A12 & A6 != A13 & A6 != A14 & A6 != A15 & A7 != A8 & A7 != A9 & A7 != A10 & A7 != A11 & A7 != A12 & A7 != A13 & A7 != A14 & A7 != A15 & A8 != A9 & A8 != A10 & A8 != A11 & A8 != A12 & A8 != A13 & A8 != A14 & A8 != A15 & A9 != A10 & A9 != A11 & A9 != A12 & A9 != A13 & A9 != A14 & A9 != A15 & A10 != A11 & A10 != A12 & A10 != A13 & A10 != A14 & A10 != A15 & A11 != A12 & A11 != A13 & A11 != A14 & A11 != A15 & A12 != A13 & A12 != A14 & A12 != A15 & A13 != A14 & A13 != A15 & A14 != A15))",
    }
    checker.add_background_definitions(add_defs_dict)
    with pytest.raises(
        ModelCheckerInputError,
        match=r"Variable 'X' in predicate 'terpeneGlycoside' is not bound at evaluation time",
    ):
        checker.check_formula_for_molecule(formula_str, molecule)


def test_unknown_index_error(checker: ModelCheckerTestWrapper):
    molecule = Chem.MolFromSmiles(
        "C=1[C@@]2([C@]3(CC[C@]4([C@]([C@@]3(C=CC2=CC(C1)=O)[H])(CCC4=O)[H])C)[H])C"
    )

    formula_str = (
        "threeOxoSteroid <=> (oxoSteroid & ?[A1, A2]: (c(A1) & o(A2) & "
        "bDOUBLE(A1, A2) & steroidPosition3(A1)))"
    )

    add_def_dict = {
        "oxoSteroid": "oxoSteroid <=> (steroid & hasCarbonylGroup)",
        "steroidPosition3": "steroidPosition3(X) <=> (c(X) & inRing(X) & has_0_hs(X) & bDOUBLE(X, Y) & o(Y) & ?[A1, A2]: (c(A1) & c(A2) & bSINGLE(X, A1) & bSINGLE(X, A2) & inRing(A1) & inRing(A2) & A1 != A2))",
        "molecule": "molecule <=> net_charge_neutral",
        "hasCarbonylGroup": "hasCarbonylGroup <=> ?[C1, O1]: (c(C1) & o(O1) & bDOUBLE(C1, O1))",
        "steroid": "steroid <=> (molecule & ?[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17]: (c(A1) & c(A2) & c(A3) & c(A4) & c(A5) & c(A6) & c(A7) & c(A8) & c(A9) & c(A10) & c(A11) & c(A12) & c(A13) & c(A14) & c(A15) & c(A16) & c(A17) & has_bond_to(A1, A2) & has_bond_to(A2, A3) & has_bond_to(A3, A4) & has_bond_to(A4, A5) & has_bond_to(A5, A10) & has_bond_to(A10, A1) & has_bond_to(A5, A6) & has_bond_to(A6, A7) & has_bond_to(A7, A8) & has_bond_to(A8, A9) & has_bond_to(A9, A10) & has_bond_to(A8, A14) & has_bond_to(A14, A15) & has_bond_to(A15, A16) & has_bond_to(A16, A17) & has_bond_to(A17, A13) & has_bond_to(A13, A14) & has_bond_to(A9, A11) & has_bond_to(A11, A12) & has_bond_to(A12, A13)))",
    }
    checker.add_background_definitions(add_def_dict)
    with pytest.raises(
        ModelCheckerInputError,
        match=r"Variable 'Y' is used in the definition of predicate 'steroidPosition3' "
        r"but is not bound by predicate arguments or quantifiers",
    ):
        checker.check_formula_for_molecule(formula_str, molecule)


def test_model_check_success(checker: ModelCheckerTestWrapper):
    carbonMonoxide = Chem.MolFromSmiles("[C-]#[O+]")  # CHEBI:17245
    ethanol = Chem.MolFromSmiles("CCO")
    thionitrousAcid = Chem.MolFromSmiles("SN=O")  # CHEBI:6530

    formula_str = "carbonMonoxide <=> ?[A1, A2]: (oneCarbonCompound & c(A1) & o(A2) & has_bond_to(A1,A2))"
    add_def_dict = {
        "oneCarbonCompound": "oneCarbonCompound <=> ?[X]: (c(X) & ~twoPlusCarbonCompound)",
        "twoPlusCarbonCompound": "twoPlusCarbonCompound <=> ?[X, Y]: (c(X) & c(Y) & has_bond_to(X, Y) & X != Y)",
    }
    checker.add_background_definitions(add_def_dict)
    assert (
        checker.check_formula_for_molecule(formula_str, carbonMonoxide)
        == ModelCheckerOutcome.MODEL_FOUND
    )
    assert (
        checker.check_formula_for_molecule(formula_str, ethanol)
        == ModelCheckerOutcome.NO_MODEL
    )
    assert (
        checker.check_formula_for_molecule(formula_str, thionitrousAcid)
        == ModelCheckerOutcome.NO_MODEL
    )


def test_model_check_success_absent_predicate_in_spectrum(
    checker: ModelCheckerTestWrapper,
):
    ethanol = Chem.MolFromSmiles("CCO")
    thionitrous_acid = Chem.MolFromSmiles("SN=O")

    # Edge case: some molecules do not have sulfur atoms. The model checker should
    # treat missing unary extensions as False instead of raising.
    formula_str = "containsSulfur <=> ?[A1]: s(A1)"

    assert (
        checker.check_formula_for_molecule(formula_str, ethanol)
        == ModelCheckerOutcome.NO_MODEL
    )
    assert (
        checker.check_formula_for_molecule(formula_str, thionitrous_acid)
        == ModelCheckerOutcome.MODEL_FOUND
    )


def test_model_check_success_global_charge_shortcut(
    checker: ModelCheckerTestWrapper,
):
    ethanol = Chem.MolFromSmiles("CCO")
    ammonium = Chem.MolFromSmiles("[NH4+]")

    neutral_formula = "neutralMolecule <=> net_charge_neutral"
    cation_formula = "cationicMolecule <=> net_charge_positive"

    # Edge case: allow 0-arity use of global charge predicates.
    assert (
        checker.check_formula_for_molecule(neutral_formula, ethanol)
        == ModelCheckerOutcome.MODEL_FOUND
    )
    assert (
        checker.check_formula_for_molecule(neutral_formula, ammonium)
        == ModelCheckerOutcome.NO_MODEL
    )
    assert (
        checker.check_formula_for_molecule(cation_formula, ammonium)
        == ModelCheckerOutcome.MODEL_FOUND
    )
    assert (
        checker.check_formula_for_molecule(cation_formula, ethanol)
        == ModelCheckerOutcome.NO_MODEL
    )


def test_model_check_success_transitive_background_definitions(
    checker: ModelCheckerTestWrapper,
):
    ethanol = Chem.MolFromSmiles("CCO")
    methane = Chem.MolFromSmiles("C")

    formula_str = "target <=> oxygenCompound"
    add_def_dict = {
        "oxygenCompound": "oxygenCompound <=> hasOxygen",
        "hasOxygen": "hasOxygen <=> ?[X]: o(X)",
    }
    checker.add_background_definitions(add_def_dict)

    # Spectrum test: same chained definition should distinguish oxygen-containing
    # and oxygen-free molecules.
    assert (
        checker.check_formula_for_molecule(formula_str, ethanol)
        == ModelCheckerOutcome.MODEL_FOUND
    )
    assert (
        checker.check_formula_for_molecule(formula_str, methane)
        == ModelCheckerOutcome.NO_MODEL
    )


def test_model_checking_additional_examples(checker: ModelCheckerTestWrapper):
    # https://github.com/sfluegel05/chemlog-peptides/pull/12
    formula_str = "cation <=> net_charge_positive"

    mol = Chem.MolFromSmiles("C(=O)(C1=CC=C(C=C1F)OCCCCCC[NH+](CC=C)C)C=2C=CC(=CC2)Br")

    checker.check_formula_for_molecule(formula_str, mol)

    formula_str = (
        "glycolipid <=> (glycerolipid & ?[O1, C1, O2, C2]: (o(O1) & "
        "has_0_hs(O1) & c(C1) & bSINGLE(O1, C1) & o(O2) & has_0_hs(O2) & bSINGLE(C1, O2) "
        "& c(C2) & bSINGLE(O2, C2) & has_1_hs(C1)))"
    )

    add_def = {
        "glycerolipid": "glycerolipid <=> ?[C1, C2, C3, O1, O2, O3]: (c(C1) & c(C2) & c(C3) & o(O1) "
        "& o(O2) & o(O3) & bSINGLE(C1, C2) & bSINGLE(C2, C3) & bSINGLE(C1, O1) & "
        "bSINGLE(C2, O2) & bSINGLE(C3, O3))"
    }
    checker.add_background_definitions(add_def)

    mol = Chem.MolFromSmiles(
        "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
    )

    checker.check_formula_for_molecule(formula_str, mol)
