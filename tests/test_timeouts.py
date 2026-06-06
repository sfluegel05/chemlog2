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
        result = checker.check_formula_for_molecule(formula_str, molecule)
        print(result)
        assert (
            checker.check_formula_for_molecule(formula_str, molecule)
            != ModelCheckerOutcome.TIMEOUT
        )


def test_timeouts_for_diol():
    # ChEBI:23824
    # No exception raised for this formula
    formula_str = (
        "diol <=> (chemicalCompound & ?[A1, A2, C1, C2]: (o(A1) & has_1_hs(A1) & "
        "bSINGLE(A1, C1) & c(C1) & ~(?[O1]: (o(O1) & bDOUBLE(C1, O1))) & o(A2) & "
        "has_1_hs(A2) & bSINGLE(A2, C2) & c(C2) & ~(?[O2]: (o(O2) & bDOUBLE(C2, O2))) "
        "& A1 != A2))"
    )

    positive_examples = [
        "O[C@@H]1[C@@H](O)[C@@H](CC2=CC=CC=C2)N(CC2=CC=CC(=C2)C(=O)NC2=NC=CS2)C(=O)N(CC2=CC(=CC=C2)C(=O)NC2=NC=CS2)[C@@H]1CC1=CC=CC=C1",
        "CC(C(O)=O)C1=CC=C[C@H](O)[C@@H]1O",
        "O[C@H]1C=CC=C(CCC(O)=O)[C@H]1O",
        "OC(CO)\\C=N\\OC",
        "CC(CN(CC(C)O)N=O)O",
        "[C@H]12N3C=4N=CN=C(C4N=C3[C@H]([C@]([C@H](C1)O)(O2)[H])O)N",
        "[C@H]12N3C=4N=C(NC(C4N=C3[C@H]([C@]([C@H](C1)O)(O2)[H])O)=O)N",
        "OCC(C)(O)C",
        "C(\\[C@H]1[C@@H](CC([C@H]1C/C=C\\CCCC(O)=O)=O)O)=C/[C@H](CCCCC)O",
        "CCN(CCO)CCO",
        "C1=C(CCCC(CO)O)C(NC(N1)=O)=O",
        "C1(C)(C)C(\\C=C\\C(=C\\C=C\\C(=C\\CO)\\C)\\C)=C(C)C(CC1)O",
        "C=1(/C=C/C(=C/C=C/C(=C/C=C/C=C(/C=C/C=C(/C=C/C=C(/CCCC(C)(O)C)\\C)\\C)\\C)/C)/C)C(C[C@@H](CC1C)O)(C)C",
        "[C@@H]1(CC(C(\\C=C\\C(=C\\C=C\\C(=C\\C=C\\C=C(\\C=C\\C=C(\\CCC=2C(C)(C[C@@H](CC2C)O)C)/C)/C)\\C)\\C)=C(C1)C)(C)C)O",
        "C[C@H]1[C@]2(O)[C@@H](C[C@]3([C@]4([C@](CC[C@]13CO2)([C@]5(CC[C@]6(CC[C@@](C[C@]6([C@@]5(CC4)C)[H])(C)C(=O)O)C)C)[H])C)[H])O",
        "C1[C@H](O)C([C@@]2(CC[C@@]3([C@](CC=C4[C@]3(CC[C@@]5([C@]4(C[C@](C[C@H]5O)(C(O)=O)C)[H])C)C)([C@]2(C1)C)[H])C)[H])(C)C",
        "C1[C@H](O)C([C@@]2(CC[C@@]3([C@](CC=C4[C@]3(CC[C@@]5([C@]4([C@H]([C@@H](C[C@@H]5O)C(O)=O)C)[H])C)C)([C@]2(C1)C)[H])C)[H])(C)C",
        "[C@]12([C@]([C@]3([C@](CC1)([C@@]4([C@](C[C@H](CC4)O)([C@H](C3)O)[H])C)[H])[H])(CC[C@@]2([C@@](CCCC(C)C)(C)[H])[H])[H])C",
        "[C@@]123[C@@]4([C@H](C[C@@]5([C@@]1(CCC6=C5COC6=O)C)[H])O4)[C@@H]([C@]7([C@H](CO)C)[C@H]([C@@H]2O3)O7)O",
    ]
    _assert_for_no_timeout(formula_str, positive_examples)


def test_timeouts_for_oxy_fatty_acid():
    # CHEBI:59644 - oxoFattyAcid
    # No exception raised for this formula
    formula_str = (
        "oxoFattyAcid <=> (fattyAcid & (?[A1, A2]: (c(A1) & o(A2) & bDOUBLE(A1, A2) "
        "& has_at_least_1_hs(A1)) | ?[A1, A2, A3, A4]: (c(A1) & o(A2) & c(A3) & c(A4) "
        "& bDOUBLE(A1, A2) & has_bond_to(A1, A3) & has_bond_to(A1, A4) & ~(A3 = A4))))"
    )

    positive_samples = [
        "C(\\[C@H](CCCC(O)=O)O)=C\\C=C\\C=C\\[C@@H](C/C=C\\C=C\\C(CC)=O)O",
        "C(O)(CCC(CCCC\\C=C/C=C\\C=C\\C=C\\CC)=O)=O",
        "C(\\CCC(O)=O)=C\\C[C@@H](C(/C=C/C=C/C=C\\C=C\\[C@H](C/C=C\\CC)O)=O)O",
        "C(CCCCCCCCCC(=O)[H])CCCCC(O)=O",
        "C(=CCCCCCCCC(O)=O)C=CC(CCCCC)=O",
        "C(C(/C=C/C=C/C=C/[C@H](CCCC(O)=O)O)=O)/C=C\\CCCCC",
        "CC\\C=C/C[C@H]1[C@@H](CCCCCCCC(O)=O)CCC1=O",
        "O[C@@H](CCCCC)/C=C/C=C\\C/C=C\\C=C\\C(=O)CCCC(O)=O",
        "O=C(CCCC(O)=O)/C=C/C=C\\C=C\\[C@H](C/C=C\\CCCCC)O",
        "O=C(CCCC(O)=O)/C=C/C=C\\CCCCCCCCC",
        "C(C(O)=O)C/C=C\\CC(/C=C/C=C\\C/C=C\\C/C=C\\C/C=C\\CC)=O",
        "C(C(O)=O)CCCCC(/C=C/C=C\\C/C=C\\C/C=C\\C/C=C\\CC)=O",
        "C(CCCC(CCCCCCO)=O)CCCCC(O)=O",
        "O=C(CCCCCCCC(O)=O)/C=C/C=C\\C/C=C\\CC",
        "C(CCCCCCCCCCC(=O)O)(=O)[H]",
        "CC(=O)CC(O)=O",
    ]
    _assert_for_no_timeout(formula_str, positive_samples)


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
