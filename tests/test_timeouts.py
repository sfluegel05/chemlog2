from rdkit import Chem

from chemlog.fol_classification.model_checking import (
    ModelCheckerOutcome,
)
from tests.utils import ModelCheckerTestWrapper


def _assert_for_no_timeout(formula_str, smiles_list):
    checker = ModelCheckerTestWrapper()
    # checker.load_background_definitions_from_json(
    #     "/home/staff/a/akhedekar/chebai-NL2FOL/nl_2_fol/inference/learner/validation_background_defs.json"
    # )
    checker.add_background_definitions(ADDITIONAL_DEFINITIONS)
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        assert (
            checker.check_formula_for_molecule(formula_str, molecule)
            != ModelCheckerOutcome.TIMEOUT
        ), f"Model checker timed out for SMILES: {smiles}"


def test_timeouts_for_diol():
    # ChEBI:23824
    # No exception raised for this formula
    formula_str = (
        "diol <=> (chemicalCompound & ?[A1, A2, C1, C2]: (o(A1) & has_1_hs(A1) & "
        "bSINGLE(A1, C1) & c(C1) & ~(?[O1]: (o(O1) & bDOUBLE(C1, O1))) & o(A2) & "
        "has_1_hs(A2) & bSINGLE(A2, C2) & c(C2) & ~(?[O2]: (o(O2) & bDOUBLE(C2, O2))) "
        "& A1 != A2))"
    )

    samples = [
        "Cc1cn([C@H]2C[C@H](O)[C@@H](COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(O)=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)O2)c(=O)[nH]c1=O",
    ]
    _assert_for_no_timeout(formula_str, samples)


def test_timeouts_for_oxy_fatty_acid():
    # CHEBI:59644 - oxoFattyAcid
    # No exception raised for this formula
    formula_str = (
        "oxoFattyAcid <=> (fattyAcid & (?[A1, A2]: (c(A1) & o(A2) & bDOUBLE(A1, A2) "
        "& has_at_least_1_hs(A1)) | ?[A1, A2, A3, A4]: (c(A1) & o(A2) & c(A3) & c(A4) "
        "& bDOUBLE(A1, A2) & has_bond_to(A1, A3) & has_bond_to(A1, A4) & ~(A3 = A4))))"
    )

    samples = [
        "O([C@H]1[C@H](O[C@@H]2O[C@H]([C@@H](O)[C@@H](O)[C@@H]2O)C)[C@@H](NC(=O)C)[C@@H](O[C@@H]1CO)OC[C@@H](O)[C@H](O)[C@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]3O)CO)[C@@H](NC(=O)C)CO)[C@@H]5O[C@@H]([C@H](O)[C@H](O[C@]6(O[C@H]([C@H](NC(=O)C)[C@@H](O)C6)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]5O)CO",
        "O=C/1N[C@@H](C(=O)O)[C@@H](C(=O)N[C@H](C(=O)N[C@@H](/C=C/C(=C/[C@@H]([C@@H](OC)CC2=CC=CC=C2)C)/C)[C@@H](C(N[C@H](CCC(N(\\C1=C/C)C)=O)C(=O)O)=O)C)CCCCN=C(N)N)C",
        "S(CCC(N)C(=O)NC(C(=O)NCC(=O)NC(C(=O)N1C(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C=O)CC2=CC=C(O)C=C2)CC(=O)O)C(CC)C)C(C)C)CCCCN)CCC1)CC(=O)O)CCCCN)C",
        "O=C1OCC(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)CC(O)CCCCCCC)CC(C)C)CCC(=O)O)CCC(=O)N)C(C)C)CC(C)C)CCC(=O)N)C(=O)NC(C(=O)NC(C(=O)NC(CCC(=O)N)C(NC(C(NC(C(NC(C(NC1C(CC)C)=O)CCC(=O)N)=O)CC(C)C)=O)CC(C)C)=O)CC(C)C)C(C)C",
    ]
    _assert_for_no_timeout(formula_str, samples)


ADDITIONAL_DEFINITIONS = {
    "chemicalCompound": "chemicalCompound <=> molecule",
    "molecule": "molecule <=> net_charge_neutral",
    "fattyAcid": (
        "fattyAcid <=> ?[X0, X1, X2, X3]: (organicMolecularEntity & "
        "c(X0) & o(X1) & o(X2) & has_1_hs(X2) & bDOUBLE(X0, X1) & bSINGLE(X0, X2)"
        " & (c(X3)) & (has_bond_to(X0, X3)))"
    ),
    "organicMolecularEntity": "organicMolecularEntity <=> (molecule & ?[X0]: (c(X0)))",
}
