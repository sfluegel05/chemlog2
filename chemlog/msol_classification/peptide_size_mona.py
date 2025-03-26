import os

from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from chemlog.msol_classification.mona_model_checker import MonaModelChecker
from chemlog.preprocessing.mol_to_msol import mol_to_msol


class MonaPeptideSizeClassifier:

    def __init__(self):
        with open(os.path.join("data", "msol_specifications", "msol_formulas.mona"), "r") as f:
            self.predicate_definitions = "\n".join([l.strip() for l in f.readlines()])


    def classify_peptide_size_mona(self, mol):
        universe, mol_mona = mol_to_msol(mol)
        model_checker = MonaModelChecker(universe, mol_mona, self.predicate_definitions)
        proof_attempts = []
        for n in range(2, 11):
            target_formula = build_peptide_structure(n)
            outcome = model_checker.find_model(target_formula)
            proof_attempts.append(
                {"target": n, "variable_assignments": outcome[1], "outcome": outcome[0].name})
            if outcome[0] in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                return n - 1, proof_attempts
            elif outcome[0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                return 0, proof_attempts
        return 10, proof_attempts


def build_peptide_structure(n):
    aars = [f"A{i}" for i in range(n)]
    bonds = [f"B{i}" for i in range(n - 1)]
    res = f"var2 {','.join(aars)};\nvar2 {','.join(bonds)};\n"
    res += (
            " & ".join(
                [
                    f"AAR({aars[i]})"
                    + "".join(
                        [
                            f" & ~HasOverlap({aars[i]}, {aars[j]})"
                            for j in range(i + 1, n)
                        ]
                    )
                    for i in range(n)
                ]
            )
            + ";\n"
    )
    res += (
            " & ".join(
                [
                    f"AmideBond({bonds[i]}) & HasOverlap({bonds[i]}, {aars[i + 1]}) & ("
                    + " | ".join(
                        [f"HasOverlap({bonds[i]}, {aars[j]})" for j in range(i + 1)]
                    )
                    + ")"
                    for i in range(n - 1)
                ]
            )
            + ";\n"
    )
    return res