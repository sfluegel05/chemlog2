from rdkit import Chem

from chemlog.fol_classification.model_checking import (
    ModelCheckerOutcome,
)
from tests.utils import ModelCheckerTestWrapper


def _assert_for_no_timeout(formula_str, smiles_list):
    checker = ModelCheckerTestWrapper()
    checker.add_background_definitions(ADDITIONAL_DEFINITIONS)
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        assert (
            checker.check_formula_for_molecule(formula_str, molecule)
            != ModelCheckerOutcome.TIMEOUT
        ), f"Model checker timed out for SMILES: {smiles}"


def test_timeouts_for_diol():
    # ChEBI:23824
    formula_str = (
        "diol <=> (chemicalCompound & ?[A1, A2, C1, C2]: (o(A1) & has_1_hs(A1) & "
        "bSINGLE(A1, C1) & c(C1) & ~(?[O1]: (o(O1) & bDOUBLE(C1, O1))) & o(A2) & "
        "has_1_hs(A2) & bSINGLE(A2, C2) & c(C2) & ~(?[O2]: (o(O2) & bDOUBLE(C2, O2))) "
        "& A1 != A2))"
    )

    samples = [
        "O([C@H]1[C@H](O)[C@H](O[C@@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]2CO)O[C@H]3[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]3CO)O)[C@H]1O)CO[C@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O)[C@H]6O)CO[C@]7(O[C@H]([C@H](NC(=O)C)[C@@H](O)C7)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]5NC(=O)C)CO)[C@H](O)[C@@H]4O[C@@H]8O[C@@H]([C@@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O)[C@H]9O)CO[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]8NC(=O)C)CO)CO)[C@H]%11O[C@@H]([C@@H](O)[C@H](O)[C@@H]%11O[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O[C@]%14(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%14)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%13O)CO)[C@H](O)[C@H]%12NC(=O)C)CO)CO[C@@H]%15O[C@@H]([C@@H](O[C@@H]%16O[C@@H]([C@H](O)[C@H](O)[C@H]%16O)CO[C@]%17(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%17)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]%15NC(=O)C)CO",
        "Cc1cn([C@H]2C[C@H](O)[C@@H](COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(=O)O[C@H]3C[C@@H](O[C@@H]3COP(O)(O)=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)n3cc(C)c(=O)[nH]c3=O)O2)c(=O)[nH]c1=O",
        "O([C@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O)[C@@H]1O[C@@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@H](O)[C@H](O)[C@H]5O)CO[C@]6(O[C@H]([C@H](NC(=O)CO)[C@@H](O)C6)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]4NC(=O)C)CO)CO)[C@H]7[C@H](O)[C@H](O[C@@H](O[C@H]8[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]8CO)O[C@H]9[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]9CO)O)[C@H]7O)CO[C@H]%10O[C@@H]([C@@H](O)[C@H](O)[C@@H]%10O[C@@H]%11O[C@@H]([C@@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)CO)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H](O)[C@H]%11NC(=O)C)CO)CO",
        "O([C@H]1[C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]3O)CO)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1O[C@H]5[C@H](O)[C@H](O[C@@H](O[C@H]6[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]6CO)O[C@H]7[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]7CO[C@@H]8O[C@H]([C@@H](O)[C@@H](O)[C@@H]8O)C)O)[C@H]5O)CO[C@H]9O[C@@H]([C@@H](O)[C@H](O)[C@@H]9O[C@@H]%10O[C@@H]([C@@H](O[C@@H]%11O[C@@H]([C@H](O)[C@H](O)[C@H]%11O)CO)[C@H](O)[C@H]%10NC(=O)C)CO)CO[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O)[C@H]%13O)CO)[C@H](O)[C@H]%12NC(=O)C)CO)CO)[C@@H]%14O[C@@H]([C@@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O[C@]%16(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%16)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%15O)CO)[C@H](O)[C@H]%14NC(=O)C)CO",
        "[H][C@@]1(O[C@@](C[C@@H](O)[C@H]1O)(O[C@@H]1C[C@@](OC[C@H]2O[C@@H](OC[C@H]3O[C@H](OP(O)(O)=O)[C@H](NC(=O)C[C@H](O)CCCCCCCCCCC)[C@@H](OC(=O)C[C@H](O)CCCCCCCCCCC)[C@@H]3O)[C@H](NC(=O)C[C@@H](CCCCCCCCCCC)OC(=O)CCCCCCCCCCC)[C@@H](OC(=O)C[C@@H](CCCCCCCCCCC)OC(=O)CCCCCCCCCCCCC)[C@@H]2OP(O)(O)=O)(O[C@]([H])([C@H](O)CO)[C@@H]1O[C@H]1O[C@H]([C@@H](O)CO)[C@@H](OP(O)(O)=O)[C@H](O[C@H]2O[C@H]([C@@H](O)CO)[C@@H](O)[C@H](OC3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)[C@@H]2O)[C@@H]1O)C(O)=O)C(O)=O)[C@H](O)CO",
        "O([C@H]1[C@H](O)[C@H](NC(=O)C)[C@H](O[C@@H]2[C@@H](O[C@H]3[C@H](O)[C@H](O[C@@H](O[C@H]4[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]4CO)O[C@H]5[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]5CO[C@@H]6O[C@H]([C@@H](O)[C@@H](O)[C@@H]6O)C)O)[C@H]3O)CO[C@H]7O[C@@H]([C@@H](O)[C@H](O)[C@@H]7O[C@@H]8O[C@@H]([C@@H](O)[C@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]9O)CO)[C@H]8NC(=O)C)CO)CO[C@@H]%11O[C@@H]([C@@H](O)[C@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H]%11NC(=O)C)CO)O[C@@H]([C@@H](O)[C@@H]2O)CO)O[C@@H]1CO)[C@@H]%14O[C@@H]([C@H](O)[C@H](O[C@]%15(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%15)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%14NC(=O)C)CO",
        "O([C@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO[C@]4(O[C@H]([C@H](NC(=O)CO)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O)[C@@H]1O[C@@H]5O[C@@H]([C@@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O)[C@H]6O)CO[C@]7(O[C@H]([C@H](NC(=O)CO)[C@@H](O)C7)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]5NC(=O)C)CO)CO)[C@H]8[C@H](O)[C@H](O[C@@H](O[C@H]9[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]9CO)O[C@H]%10[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]%10CO)O)[C@H]8O)CO[C@H]%11O[C@@H]([C@@H](O)[C@H](O)[C@@H]%11O[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O)[C@H]%13O)CO)[C@H](O)[C@H]%12NC(=O)C)CO)CO[C@@H]%14O[C@@H]([C@@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O)[C@H]%15O)CO)[C@H](O)[C@H]%14NC(=O)C)CO",
        "O([C@H]1[C@H](O[C@@H]2O[C@@H]([C@@H](O)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H](O[C@H]3[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]3CO)O[C@H]4[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]4CO[C@@H]5O[C@H]([C@@H](O)[C@@H](O)[C@@H]5O)C)O)[C@H]1O)CO[C@H]6O[C@@H]([C@@H](O)[C@H](O)[C@@H]6O[C@@H]7O[C@@H]([C@@H](O[C@@H]8O[C@@H]([C@H](O)[C@H](O)[C@H]8O)CO)[C@H](O)[C@H]7NC(=O)C)CO)CO[C@@H]9O[C@@H]([C@@H](O[C@@H]%10O[C@@H]([C@H](O)[C@H](O)[C@H]%10O)CO)[C@@H](O)[C@H]9NC(=O)C)CO)[C@H]%11O[C@@H]([C@@H](O[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O)[C@H]%13O)CO)[C@H](O)[C@H]%12NC(=O)C)CO)[C@H](O)[C@@H]%11O[C@@H]%14O[C@@H]([C@@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O)[C@H]%15O)CO)[C@H](O)[C@H]%14NC(=O)C)CO)CO",
        "Nc1nc2n(cnc2c(=O)[nH]1)[C@H]1C[C@H](OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(S)(=O)OC[C@H]2O[C@H](C[C@@H]2OP(O)(S)=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)n2cnc3c2nc(N)[nH]c3=O)[C@@H](CO)O1",
        "O([C@H]1[C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1O[C@H]5[C@H](O)[C@H](O[C@@H](O[C@H]6[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]6CO)O)[C@H]5O)CO[C@H]7O[C@@H]([C@@H](O)[C@H](O)[C@@H]7O[C@@H]8O[C@@H]([C@@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O)[C@H]9O)CO[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]8NC(=O)C)CO)CO)CO)[C@@H]%11O[C@@H]([C@@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H](O)[C@@H]%11NC(=O)C)CO",
        "CCCCCCCCCCCCCCCC(=O)N[C@@H](CCC(=O)NCCCC[C@H](NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCC(N)=O)NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](C)NC(=O)[C@@H](N)Cc1cnc[nH]1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(O)=O)C(O)=O",
        "O([C@H]1[C@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1CO)O[C@H]5[C@H](O)[C@H](O[C@@H](O[C@H]6[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]6CO)O[C@@H]([C@H](O)[C@@H](NC(=O)C)CO)[C@H](O)CO)[C@H]5O)CO[C@H]7O[C@@H]([C@@H](O)[C@H](O)[C@@H]7O[C@@H]8O[C@@H]([C@@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O)[C@H]9O)CO[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]8NC(=O)C)CO)CO)[C@@H]%11O[C@@H]([C@@H](O)[C@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H]%11NC(=O)C)CO",
        "O([C@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O)[C@@H]1O[C@@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@H](O)[C@H](O[C@@H]6O[C@@H]([C@@H](O[C@@H]7O[C@@H]([C@H](O)[C@H](O)[C@@H]7O)CO)[C@H](O)[C@H]6NC(=O)C)CO)[C@H]5O)CO)[C@H](O)[C@H]4NC(=O)C)CO)CO)[C@H]8[C@H](O)[C@H](O[C@@H](O[C@H]9[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]9CO)O[C@H]%10[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]%10CO[C@@H]%11O[C@H]([C@@H](O)[C@@H](O)[C@@H]%11O)C)O)[C@H]8O)CO[C@H]%12O[C@@H]([C@@H](O)[C@H](O)[C@@H]%12O[C@@H]%13O[C@@H]([C@@H](O[C@@H]%14O[C@@H]([C@H](O)[C@H](O[C@@H]%15O[C@@H]([C@@H](O[C@@H]%16O[C@@H]([C@H](O)[C@H](O)[C@H]%16O)CO)[C@H](O)[C@H]%15NC(=O)C)CO)[C@@H]%14O)CO)[C@H](O)[C@H]%13NC(=O)C)CO)CO[C@@H]%17O[C@@H]([C@@H](O[C@@H]%18O[C@@H]([C@H](O)[C@H](O[C@@H]%19O[C@@H]([C@@H](O[C@@H]%20O[C@@H]([C@H](O)[C@H](O)[C@H]%20O)CO)[C@H](O)[C@H]%19NC(=O)C)CO)[C@H]%18O)CO)[C@H](O)[C@H]%17NC(=O)C)CO",
        "S(O[C@@H]1[C@@H](O)[C@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@H](O[C@H]3[C@@H](O)[C@H](O[C@@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O)[C@H]6NC(=O)C)CO)[C@H](O[C@]7(O[C@H]([C@H](NC(=O)C)[C@@H](O)C7)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]5O)CO)[C@H](O)[C@H]4NC(=O)C)CO)[C@H](O[C@@H]3O[C@@H]8[C@H](O)[C@@H](O[C@@H]([C@H]8O)CO[C@H]9O[C@@H]([C@@H](O)[C@H](O)[C@@H]9O[C@@H]%10O[C@@H]([C@@H](O[C@@H]%11O[C@@H]([C@H](O)[C@H](OS(O)(=O)=O)[C@H]%11O)CO)[C@H](O)[C@H]%10NC(=O)C)CO)CO)O[C@H]%12[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]%12CO)O[C@H]%13[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]%13CO[C@@H]%14O[C@H]([C@@H](O)[C@@H](O)[C@@H]%14O)C)O)CO)O[C@@H]2CO)O[C@@H]([C@@H]1O)CO)(O)(=O)=O",
        "O([C@H]1[C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O)[C@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]3O)CO)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1O[C@H]5[C@H](O)[C@H](O[C@@H](O[C@H]6[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]6CO)O[C@H]7[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]7CO[C@@H]8O[C@H]([C@@H](O)[C@@H](O)[C@@H]8O)C)O)[C@H]5O)CO[C@H]9O[C@@H]([C@@H](O)[C@H](O)[C@@H]9O[C@@H]%10O[C@@H]([C@@H](O)[C@H](O)[C@H]%10NC(=O)C)CO)CO[C@@H]%11O[C@@H]([C@@H](O)[C@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H]%11NC(=O)C)CO)CO)[C@@H]%14O[C@@H]([C@@H](O)[C@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O[C@]%16(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%16)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%15O)CO)[C@H]%14NC(=O)C)CO",
        "S(O[C@@H]1[C@@H](O)[C@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@H](O[C@H]3[C@H](O)[C@H](O[C@@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@H](O)[C@H](O[C@]6(O[C@H]([C@H](NC(=O)C)[C@@H](O)C6)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]5O)CO)[C@H](O)[C@H]4NC(=O)C)CO)[C@@H](O[C@H]7[C@H](O)[C@H](O[C@@H](O[C@H]8[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]8CO)O[C@H]9[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]9CO)O)[C@H]7O)CO[C@H]%10O[C@@H]([C@@H](O)[C@H](O)[C@@H]%10O[C@@H]%11O[C@@H]([C@@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H](O)[C@H]%11NC(=O)C)CO)CO[C@@H]%14O[C@@H]([C@@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O[C@]%16(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%16)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%15O)CO)[C@H](O)[C@H]%14NC(=O)C)CO)O[C@@H]3CO)O[C@@H]2CO)O[C@@H]([C@@H]1O)CO)(O)(=O)=O",
        "O([C@H]1[C@H](O)[C@H](O[C@@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]2CO)O[C@H]3[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]3CO)O)[C@H]1O)CO[C@H]4O[C@@H]([C@@H](O)[C@H](O)[C@@H]4O[C@@H]5O[C@@H]([C@@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O)[C@H]6O)CO)[C@H](O)[C@H]5NC(=O)C)CO)CO[C@@H]7O[C@@H]([C@@H](O)[C@H](O[C@@H]8O[C@@H]([C@H](O)[C@H](O[C@@H]9O[C@@H]([C@@H](O[C@@H]%10O[C@@H]([C@H](O)[C@H](O)[C@H]%10O)CO)[C@H](O)[C@H]9NC(=O)C)CO)[C@H]8O)CO)[C@H]7NC(=O)C)CO)[C@H]%11O[C@@H]([C@@H](O[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O)[C@H]%13O)CO)[C@H](O)[C@H]%12NC(=O)C)CO)[C@H](O)[C@@H]%11O[C@@H]%14O[C@@H]([C@@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O)[C@H]%15O)CO)[C@H](O)[C@H]%14NC(=O)C)CO)CO",
    ]
    _assert_for_no_timeout(formula_str, samples)


def test_timeouts_for_oxy_fatty_acid():
    # CHEBI:59644 - oxoFattyAcid
    formula_str = (
        "oxoFattyAcid <=> (fattyAcid & (?[A1, A2]: (c(A1) & o(A2) & bDOUBLE(A1, A2) "
        "& has_at_least_1_hs(A1)) | ?[A1, A2, A3, A4]: (c(A1) & o(A2) & c(A3) & c(A4) "
        "& bDOUBLE(A1, A2) & has_bond_to(A1, A3) & has_bond_to(A1, A4) & ~(A3 = A4))))"
    )

    samples = [
        "O=C1N[C@@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)O)CC(=O)N[C@@H](CCCN=C(N)N)C(N[C@H]([C@@H](C(N[C@H](CCC(NC1=CC)=O)C(=O)O)=O)C)/C=C/C(=C/[C@@H]([C@@H](OC(=O)C)CC2=CC=CC=C2)C)/C)=O)CC(C)C)C",
        "O1[C@@H](O[C@@H]2[C@@H](O[C@H]3[C@H](O)[C@H](O[C@@H](O[C@H]4[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]4CO)O[C@@H]([C@H](O)[C@@H](NC(=O)C)CO)[C@H](O)CO)[C@H]3O)CO[C@H]5O[C@@H]([C@@H](O)[C@H](O)[C@@H]5O[C@@H]6O[C@@H]([C@@H](O[C@@H]7O[C@@H]([C@H](O)[C@H](O)[C@H]7O)CO[C@]8(O[C@H]([C@H](NC(=O)C)[C@@H](O)C8)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]6NC(=O)C)CO)CO)O[C@@H]([C@@H](O)[C@@H]2O)CO)[C@H](NC(=O)C)[C@@H](O[C@@H]9O[C@@H]([C@H](O[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]9O)CO)[C@H](O)[C@H]1CO[C@]%11(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%11)[C@H](O)[C@H](O)CO)C(O)=O",
        "O=[N+]([O-])C1=C(O)C=2C3=C(O)C=CC(=C3)C(N(C(=O)CNC(=O)[C@H](NC(=O)[C@H](N(C(=O)CCCCCCCCCC(C)C)C)CO)C)C)C(=O)N[C@H](C(NC(CC(=C1)C2)C(=O)O)=O)C",
        "O([C@H]1[C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1OC[C@H]4O[C@@H](O[C@H]5[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]5CO)O[C@H]6[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]6CO)O)[C@@H](O)[C@@H](O[C@H]7O[C@@H]([C@@H](O)[C@H](O)[C@@H]7O[C@@H]8O[C@@H]([C@@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O)[C@H]9O)CO)[C@H](O)[C@H]8NC(=O)C)CO)CO)[C@@H]4O)CO)[C@@H]%10O[C@@H]([C@@H](O[C@@H]%11O[C@@H]([C@H](O)[C@H](O[C@]%12(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%12)[C@H](O)[C@H](O)CO)C(O)=O)[C@@H]%11O)CO)[C@H](O)[C@H]%10NC(=O)C)CO",
        "C1(CC2(OC(C1C(=O)O)CC(O[C@H]3[C@H]([C@H]([C@@H]([C@H](O3)C)O)N)O)C=CC=CC=CC=CCC(CCCC)OC(C=CC4C(O4)CC(C2)O)=O)O)O",
        "O([C@H]1[C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O[C@]4(O[C@H]([C@H](NC(=O)CO)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]3O)CO)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1O[C@@H]5[C@H](O)[C@@H](O[C@@H]([C@H]5O)CO[C@H]6O[C@@H]([C@@H](O)[C@H](O)[C@@H]6O)CO)O[C@H]7[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]7CO)O[C@H]8[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]8CO[C@@H]9O[C@H]([C@@H](O)[C@@H](O)[C@@H]9O)C)O)CO)[C@@H]%10O[C@@H]([C@@H](O[C@@H]%11O[C@@H]([C@H](O)[C@H](O[C@]%12(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%12)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%11O)CO)[C@H](O)[C@H]%10NC(=O)C)CO",
        "O([C@H]1[C@H](O)[C@H](O[C@@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]2CO)O[C@@H]([C@H](O)[C@@H](NC(=O)C)CO)[C@H](O)CO)[C@H]1O)CO[C@H]3O[C@@H]([C@@H](O)[C@H](O)[C@@H]3O[C@@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@H](O)[C@H](O)[C@H]5O)CO[C@]6(O[C@H]([C@H](NC(=O)C)[C@@H](O)C6)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]4NC(=O)C)CO)CO[C@@H]7O[C@@H]([C@@H](O[C@@H]8O[C@@H]([C@H](O)[C@H](O)[C@H]8O)CO[C@]9(O[C@H]([C@H](NC(=O)C)[C@@H](O)C9)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]7NC(=O)C)CO)[C@H]%10O[C@@H]([C@@H](O)[C@H](O)[C@@H]%10O[C@@H]%11O[C@@H]([C@@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O)[C@H]%12O)CO[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]%11NC(=O)C)CO)CO",
        "O([C@H]1[C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O)[C@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]3O)CO)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1O[C@H]5[C@H](O)[C@H](O[C@@H](O[C@H]6[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]6CO)O[C@H]7[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]7CO[C@@H]8O[C@H]([C@@H](O)[C@@H](O)[C@@H]8O)C)O)[C@H]5O)CO[C@H]9O[C@@H]([C@@H](O)[C@H](O)[C@@H]9O[C@@H]%10O[C@@H]([C@@H](O)[C@H](O)[C@H]%10NC(=O)C)CO)CO[C@@H]%11O[C@@H]([C@@H](O)[C@H](O[C@@H]%12O[C@@H]([C@H](O)[C@H](O[C@]%13(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%13)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%12O)CO)[C@H]%11NC(=O)C)CO)CO)[C@@H]%14O[C@@H]([C@@H](O)[C@H](O[C@@H]%15O[C@@H]([C@H](O)[C@H](O[C@]%16(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%16)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%15O)CO)[C@H]%14NC(=O)C)CO",
        "O=C1OC(C(NC(=O)C(NC(=O)CCCCC)CC(=O)O)C(=O)NC(C(=O)NC2CCC(N(C(C(N(C(C(NC1C(C)C)=O)CCC3=CC=C(O)C=C3)C)=O)CC4=CC=CC=C4)C2=O)O)CC5=CC=C(O)C=C5)C",
        "O=C(N1C(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)N/C(/C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)O)CCCCN)CCN)CCO)C(CC)C)C(O)C)=C\\C)C(C)C)CC(C)C)CCC(=O)N)C(C)C)C(C)C)CC(C)C)CO)C(C)C)CC(C)C)CO)CCC1)/C(/NC(=O)CC(O)CCCCC)=C/C",
        "O=C1OC([C@H](NC(=O)CCCCCCCCCCCCCC)C(=O)N[C@@H](C(=O)N[C@H](C(=O)NCC(=O)N[C@H](C(N[C@H](C(NCC(N[C@H](C(N[C@H]1C(C)C)=O)CC2=CC=C(O)C=C2)=O)=O)CC(=O)O)=O)CC(=O)N)CC(=O)O)CCC(=O)N)C",
        "O([C@@H]1[C@H](O)[C@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]2CO)O[C@H]3[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]3CO)O)O[C@@H]([C@H]1O)CO[C@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O)[C@H]6O)CO)[C@H](O)[C@H]5NC(=O)C)CO)[C@H](O)[C@@H]4O)CO)[C@H]7O[C@@H]([C@@H](O[C@@H]8O[C@@H]([C@@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O)[C@H]9O)CO[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]8NC(=O)C)CO)[C@H](O)[C@@H]7O)CO",
        "O([C@@H]1O[C@@H]([C@@H](O)[C@H](O[C@@H]2O[C@@H]([C@H](O)[C@H](O[C@]3(O[C@H]([C@H](NC(=O)C)[C@@H](O)C3)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]2O)CO)[C@H]1NC(=O)C)CO[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@@H]5[C@@H](O[C@H]6[C@H](O)[C@H](O[C@@H](O[C@H]7[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]7CO)O[C@H]8[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]8CO)O)[C@H]6O)CO[C@H]9O[C@@H]([C@@H](O)[C@H](O)[C@@H]9O[C@@H]%10O[C@@H]([C@@H](O[C@@H]%11O[C@@H]([C@H](O)[C@H](O[C@]%12(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%12)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%11O)CO)[C@H](O)[C@H]%10NC(=O)C)CO)CO)O[C@@H]([C@@H](O)[C@@H]5O)CO",
        "CCCCCCCCCCCCCCCC\\C=C/OC[C@H](COP(O)(=O)OC[C@H](N)C(O)=O)OC(=O)CC\\C=C/C\\C=C/C\\C=C/C\\C=C/C\\C=C/C\\C=C/CC",
        "O=C(O)C1=C(O)C(OC)=C(NC(=O)C2=C(O)C(OC)=C(NC(=O)C3=CC=C(NC(=O)[C@@H](NC(=O)C4=CC=C(NC(=O)/C(=C/C5=CC=C(O)C=C5)/C)C=C4)CC#N)C=C3)C=C2)C=C1",
        "O([C@H]1[C@@H](O[C@@H]2O[C@@H]([C@H](O)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H](O[C@H]3[C@@H](O)[C@H](O[C@@H](O)[C@@H]3NC(=O)C)CO[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@@H]1O)CO)[C@]5(O[C@H]([C@H](NC(=O)C)[C@@H](O)C5)[C@H](O)[C@H](O)CO)C(O)=O",
        "[H][C@]1(O[C@@](C[C@H](O)[C@H]1NC(C)=O)(O[C@H]1[C@@H](O)[C@@H](CO)O[C@@H](O[C@@H]2[C@@H](NC(C)=O)C(O)O[C@H](CO)[C@H]2O[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)[C@@H]1O)C(O)=O)[C@H](O)[C@H](O)CO",
        "O([C@@H]1[C@H](O[C@]2(O[C@H]([C@H](NC(=O)C)[C@@H](O)C2)[C@H](O)[C@H](O)CO)C(O)=O)[C@@H](O)[C@@H](O[C@@H]1CO)O[C@H]3[C@H](O)[C@@H](O)[C@@H](O[C@@H]3CO)O)[C@@H]4O[C@@H]([C@H](O)[C@H](O[C@@H]5O[C@@H]([C@H](O)[C@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O[C@@H]7O[C@H]([C@@H](O)[C@@H](O)[C@@H]7O)C)[C@H]6NC(=O)C)CO)[C@H]5O)CO)[C@H]4NC(=O)C)CO",
        "O([C@H]1[C@H](O)[C@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@H](O)[C@H](O)[C@H]3O)CO[C@]4(O[C@H]([C@H](NC(=O)C)[C@@H](O)C4)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]2NC(=O)C)CO)[C@H](O[C@@H]1CO)O[C@H]5[C@H](O)[C@H](O[C@@H](O[C@H]6[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]6CO)O[C@@H]([C@H](O)[C@@H](NC(=O)C)CO)[C@H](O)CO[C@@H]7O[C@H]([C@@H](O)[C@@H](O)[C@@H]7O)C)[C@H]5O)CO[C@H]8O[C@@H]([C@@H](O)[C@H](O)[C@@H]8O[C@@H]9O[C@@H]([C@@H](O[C@@H]%10O[C@@H]([C@H](O)[C@H](O)[C@H]%10O)CO)[C@H](O[C@@H]%11O[C@H]([C@@H](O)[C@@H](O)[C@@H]%11O)C)[C@H]9NC(=O)C)CO)CO[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O)[C@H]%13O)CO)[C@H](O[C@@H]%14O[C@H]([C@@H](O)[C@@H](O)[C@@H]%14O)C)[C@H]%12NC(=O)C)CO)[C@@H]%15O[C@@H]([C@@H](O[C@@H]%16O[C@@H]([C@H](O)[C@H](O)[C@H]%16O)CO)[C@H](O[C@@H]%17O[C@H]([C@@H](O)[C@@H](O)[C@@H]%17O)C)[C@H]%15NC(=O)C)CO",
    ]
    _assert_for_no_timeout(formula_str, samples)


def test_timeouts_for_hexose():
    # CHEBI:18133 - hexose
    formula_str = (
        "hexose <=> ?[X0, X1, X2, X3, X4, X5]: ((sugar) & (c(X0)) & (c(X1)) & (c(X2)) "
        "& (c(X5)) & ~(X0 = X3) & ~(X1 = X2) & ~(X1 = X3) & ~(X1 = X4) & ~(X2 = X4) & "
        "~(X3 = X5) & (c(X3)) & (c(X4)) & ~(X0 = X1) & ~(X0 = X2) & ~(X0 = X4) & "
        "~(X0 = X5) & ~(X1 = X5) & ~(X2 = X3) & ~(X2 = X5) & ~(X3 = X4) & ~(X4 = X5) & "
        "(![X6]: ((~(c(X6)) | (X6) = (X0) | (X6) = (X1) | (X6) = (X2) | (X6) = (X3) | "
        "(X6) = (X4) | (X6) = (X5)))))"
    )
    samples = [
        "O1C(C(O)C(O)C(O)C1OC(=O)C=2C(NC)=CC=CC2)C",
        "N[C@@H](CO)C(=O)N[C@@H](CC1=CC=C(C=C1)OC)C(=O)O",
        "O([C@@H]1O[C@@H]([C@@H](O[C@@H]2O[C@@H]([C@@H](O[C@@H]3O[C@@H]([C@@H](O[C@@H]4O[C@@H]([C@@H](O)[C@@H]([C@H]4O)O)CO)[C@@H]([C@H]3O)O)CO)[C@@H]([C@H]2O)O)CO)[C@@H]([C@H]1O)O)CO)[C@@H]5[C@H](O[C@@H](O[C@H]([C@@H](CO)O)[C@@H]([C@H](C([O-])=O)O)O)[C@@H]([C@H]5O)O)CO",
        "O=C(O[C@@H]1O[C@H]([C@@H](O)[C@H]([C@H]1O)O)C)C2=C(NC)C=CC=C2",
    ]
    _assert_for_no_timeout(formula_str, samples)


def test_timeouts_for_steroid():
    # CHEBI:35341 - steroid
    formula_str = (
        "steroid <=> ?[X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, "
        "X15, X16]: ((molecule) & (c(X0)) & (c(X1)) & (c(X2)) & (c(X3)) & (c(X4)) & "
        "(c(X5)) & (c(X6)) & (c(X9)) & (c(X10)) & (c(X11)) & (c(X12)) & (c(X13)) & "
        "(c(X14)) & (c(X15)) & (has_bond_to(X1, X2)) & (has_bond_to(X2, X3)) & "
        "(has_bond_to(X3, X4)) & (has_bond_to(X4, X9)) & (has_bond_to(X9, X0)) & "
        "(has_bond_to(X4, X5)) & (has_bond_to(X5, X6)) & (has_bond_to(X8, X9)) & "
        "(has_bond_to(X7, X13)) & (has_bond_to(X13, X14)) & (has_bond_to(X16, X12)) & "
        "(has_bond_to(X12, X13)) & (has_bond_to(X8, X10)) & (c(X7)) & (c(X8)) & (c(X16)) "
        "& (has_bond_to(X0, X1)) & (has_bond_to(X6, X7)) & (has_bond_to(X7, X8)) & "
        "(has_bond_to(X14, X15)) & (has_bond_to(X15, X16)) & (has_bond_to(X10, X11)) & "
        "(has_bond_to(X11, X12)))"
    )
    samples = [
        "H][C@@]12CC=C3[C@]4([H])CC[C@]([H])([C@H](C)\\C=C\\[C@H](C)[C@H](C)CO)[C@@]4(C)CC[C@]3([H])[C@@]1(C)CCC(=O)C2",
        "S(OC1C(O)C(O)C(OC1C2=C(O)C3=C(OC(=CC3=O)C4=CC=C(OC)C=C4)C=C2O)CO)(O)(=O)=O",
        "[H][C@@]1(CC[C@@]2([H])C3=C[C@@H](O)[C@@]4(O)C[C@@H](O)CC[C@]4(C)[C@@]3([H])CC[C@]12C)[C@H](C)\\C=C\\CC(C)C(C)C",
        "O=C(O[C@@H](CC(=O)N[C@H](CO)CC(C)C)CCCCCCC)C[C@H](OC(=O)C[C@H](O[C@@H]1O[C@H]([C@H](O)[C@H]([C@H]1O)O)C)CCCCCCC)CCCCCCC",
        "O=C1N2C=3C(O)=C(NC(=O)[C@@H](O)C)C4=CC=C(N=C4C3C=C[C@@]2(C)CC1)C",
        "C[C@@H]1O[C@@H](O[C@@H]2[C@@H](CO)O[C@@H](O[C@H]3[C@@H](O)[C@@H](CO)O[C@@H](O[C@@H]4[C@@H](CO)OC(O)[C@H](NC(C)=O)[C@H]4O[C@@H]4O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]4O)[C@@H]3O)[C@H](NC(C)=O)[C@H]2O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@@H]1O",
    ]
    _assert_for_no_timeout(formula_str, samples)


def test_timeouts_for_triterpenoid():
    # CHEBI:36615 - triterpenoid
    formula_str = (
        "triterpenoid <=> ?[X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, "
        "X14]: ((terpenoid) & (c(X0)) & (c(X1)) & (c(X2)) & (c(X3)) & (c(X4)) & (c(X5)) & "
        "(c(X6)) & (c(X7)) & (c(X8)) & (c(X9)) & (c(X10)) & (c(X11)) & (c(X12)) & (c(X13)) "
        "& (c(X14)) & (has_bond_to(X0, X1)) & (has_bond_to(X1, X2)) & (has_bond_to(X2, X3)) "
        "& (has_bond_to(X3, X4)) & (has_bond_to(X4, X5)) & (has_bond_to(X5, X6)) & "
        "(has_bond_to(X6, X7)) & (has_bond_to(X7, X8)) & (has_bond_to(X8, X9)) & "
        "(has_bond_to(X9, X10)) & (has_bond_to(X10, X11)) & (has_bond_to(X11, X12)) & "
        "(has_bond_to(X12, X13)) & (has_bond_to(X13, X14)) & ~(X0 = X1) & ~(X0 = X2) & ~(X0 = "
        "X5) & ~(X0 = X6) & ~(X0 = X7) & ~(X0 = X8) & ~(X0 = X9) & ~(X0 = X10) & ~(X0 = X11) & "
        "~(X0 = X12) & ~(X0 = X13) & ~(X0 = X14) & ~(X1 = X2) & ~(X1 = X3) & ~(X1 = X4) & "
        "~(X1 = X5) & ~(X1 = X6) & ~(X1 = X9) & ~(X1 = X10) & ~(X1 = X11) & ~(X1 = X12) & "
        "~(X1 = X13) & ~(X1 = X14) & ~(X2 = X3) & ~(X2 = X4) & ~(X2 = X5) & ~(X2 = X6) & "
        "~(X2 = X7) & ~(X2 = X8) & ~(X2 = X9) & ~(X2 = X10) & ~(X2 = X11) & ~(X2 = X14) & "
        "~(X3 = X4) & ~(X3 = X5) & ~(X3 = X6) & ~(X3 = X7) & ~(X3 = X8) & ~(X3 = X9) & "
        "~(X3 = X10) & ~(X3 = X11) & ~(X3 = X12) & ~(X3 = X13) & ~(X3 = X14) & ~(X4 = X5) "
        "& ~(X4 = X6) & ~(X4 = X7) & ~(X4 = X8) & ~(X4 = X9) & ~(X4 = X10) & ~(X4 = X11) & "
        "~(X4 = X12) & ~(X4 = X13) & ~(X4 = X14) & ~(X5 = X6) & ~(X5 = X7) & ~(X5 = X8) & "
        "~(X5 = X9) & ~(X5 = X10) & ~(X5 = X11) & ~(X5 = X12) & ~(X5 = X13) & ~(X5 = X14) "
        "& ~(X6 = X9) & ~(X6 = X10) & ~(X6 = X11) & ~(X6 = X12) & ~(X6 = X13) & ~(X6 = X14) "
        "& ~(X7 = X8) & ~(X7 = X9) & ~(X7 = X10) & ~(X7 = X11) & ~(X7 = X12) & ~(X7 = X13) & "
        "~(X7 = X14) & ~(X8 = X9) & ~(X8 = X10) & ~(X8 = X13) & ~(X8 = X14) & ~(X9 = X10) & "
        "~(X9 = X11) & ~(X9 = X12) & ~(X9 = X13) & ~(X9 = X14) & ~(X10 = X11) & ~(X10 = X12) "
        "& ~(X10 = X13) & ~(X10 = X14) & ~(X11 = X12) & ~(X11 = X13) & ~(X11 = X14) & ~(X12 = "
        "X13) & ~(X0 = X3) & ~(X0 = X4) & ~(X1 = X7) & ~(X1 = X8) & ~(X2 = X12) & ~(X2 = X13) "
        "& ~(X6 = X7) & ~(X6 = X8) & ~(X8 = X11) & ~(X8 = X12) & ~(X12 = X14) & ~(X13 = X14))"
    )
    samples = [
        "O([C@H]1[C@H](O)[C@H](O[C@@H](O[C@H]2[C@H](O)[C@@H](NC(=O)C)[C@@H](O[C@@H]2CO)O[C@H]3[C@H](O)[C@@H](NC(=O)C)C(O[C@@H]3CO)O)[C@H]1O)CO[C@H]4O[C@@H]([C@@H](O[C@@H]5O[C@@H]([C@@H](O[C@@H]6O[C@@H]([C@H](O)[C@H](O)[C@H]6O)CO[C@]7(O[C@H]([C@H](NC(=O)C)[C@@H](O)C7)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]5NC(=O)C)CO)[C@H](O)[C@@H]4O[C@@H]8O[C@@H]([C@@H](O[C@@H]9O[C@@H]([C@H](O)[C@H](O)[C@H]9O)CO[C@]%10(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%10)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]8NC(=O)C)CO)CO)[C@H]%11O[C@@H]([C@@H](O)[C@H](O)[C@@H]%11O[C@@H]%12O[C@@H]([C@@H](O[C@@H]%13O[C@@H]([C@H](O)[C@H](O[C@]%14(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%14)[C@H](O)[C@H](O)CO)C(O)=O)[C@H]%13O)CO)[C@H](O)[C@H]%12NC(=O)C)CO)CO[C@@H]%15O[C@@H]([C@@H](O[C@@H]%16O[C@@H]([C@H](O)[C@H](O)[C@H]%16O)CO[C@]%17(O[C@H]([C@H](NC(=O)C)[C@@H](O)C%17)[C@H](O)[C@H](O)CO)C(O)=O)[C@H](O)[C@H]%15NC(=O)C)CO",
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
    "terpenoid": "terpenoid <=> molecule",
}
