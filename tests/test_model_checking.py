import pytest
from gavel.dialects.tptp.parser import TPTPParser
from rdkit import Chem

from chemlog.fol_classification.fol_utils import normalize_fol_formula
from chemlog.fol_classification.model_checking import ModelChecker
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms


def test_when_predicate_is_used_as_constant():
    # In this `has_bond_to(C1, c))`of formula, `c` predicate but is being used as a constant.
    # This test checks if the model checker raise appropriate error.
    formula_str = (
        "glycolipid <=> ?[O1, C1, O2, C2]: (o(O1) & "
        "has_0_hs(O1) & c(C1) & bSINGLE(O1, C1) & o(O2) & has_0_hs(O2) & bSINGLE(C1, O2) "
        "& c(C2) & bSINGLE(O2, C2) & has_1_hs(C1) & has_bond_to(C1, c))"
    )

    formula_wrapped = f"fof(temp, axiom, {formula_str})."

    parser = TPTPParser()
    tptp_parsed = parser.parse(formula_wrapped)[0].formula.right
    tptp_parsed = normalize_fol_formula(tptp_parsed)

    molecule = Chem.MolFromSmiles(
        "C([C@@H]([C@@H](/C=C/CCCCCCCCCCCCC)O)NC(CCCCCCC/C=C\\CCCCCCCC)=O)O[C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO"
    )
    universe, extensions = mol_to_fol_atoms(molecule)
    model_checker = ModelChecker(universe, extensions, {})

    with pytest.raises(
        Exception,
        match="Predicate 'c' is being used as a constant in the formula."
        "Please check the formula and ensure that predicates are not used as constants.",
    ):
        model_checker.find_model(tptp_parsed)
