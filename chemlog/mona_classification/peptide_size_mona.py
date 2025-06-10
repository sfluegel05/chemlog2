import os
from inspect import signature

from chemlog.base_classifier import Classifier
from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from chemlog.msol import msol
from chemlog.mona_classification.mona_compiler import MONACompiler
from chemlog.mona_classification.mona_model_checker import MonaModelChecker
from chemlog.preprocessing.mol_to_msol import mol_to_msol



class MonaPeptideSizeClassifier(Classifier):

    def __init__(self):
        self._peptide_structures = dict()
        self.predicate_definitions = self.load_predicate_definitions()

    @staticmethod
    def load_predicate_definitions():
        with open(os.path.join("data", "msol_specifications", "msol_formulas.mona"), "r") as f:
            return "\n".join([l.strip() for l in f.readlines()])

    @staticmethod
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

    def get_peptide_structure(self, n):
        if n not in self._peptide_structures:
            self._peptide_structures[n] = self.build_peptide_structure(n)
        return self._peptide_structures[n]

    def classify(self, mol, *args, **kwargs):
        universe, mol_mona = mol_to_msol(mol)
        model_checker = MonaModelChecker(universe, mol_mona, self.predicate_definitions)
        proof_attempts = []
        for n in range(2, 11):
            target_formula = self.get_peptide_structure(n)
            outcome = model_checker.find_model(target_formula)
            proof_attempts.append(
                {"target": n, "variable_assignments": outcome[1], "outcome": outcome[0].name})
            if outcome[0] in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                return n - 1, {"proof_attempts": proof_attempts}
            elif outcome[0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                return 0, {"proof_attempts": proof_attempts}
        return 10, {"proof_attempts": proof_attempts}



class MonaPeptideSizeClassifierCompiled(MonaPeptideSizeClassifier):

    def __init__(self):
        self.compiler = MONACompiler()
        super().__init__()

    def load_predicate_definitions(self):
        # load from internal MSOL representation
        from chemlog.msol import peptide_size
        defs_compiled = []
        for definition in [peptide_size.HasOverlap(), peptide_size.IsConnected(), peptide_size.CarbonConnected(),
                           peptide_size.CarbonFragment(), peptide_size.AmideBondFO(), peptide_size.AmideBond(),
                           peptide_size.AminoGroupFO(), peptide_size.AminoGroup(), peptide_size.CarboxyResidueFO(),
                           peptide_size.CarboxyResidue(), peptide_size.BuildingBlock(), peptide_size.AAR()]:
            sig = signature(definition.__call__)
            variables = []
            for p_name, param in sig.parameters.items():
                variables.append(param.annotation(param.name))
            defs_compiled.append((definition.name(), variables, self.compiler.visit(definition(*variables))))

        mona_str = ""
        for name, vs, formula in defs_compiled:
            args = ', '.join(f"var{2 if isinstance(var, msol.Var2) else 1} {self.compiler.visit(var)}" for var in vs)
            mona_str += f"pred {name}({args}) = {formula};\n"

        return mona_str

    def build_peptide_structure(self, n):
        # build peptide structure using the internal MSOL representation
        from chemlog.msol import peptide_size
        peptide_size_def = peptide_size.Peptide(n)
        peptide_size_formula = peptide_size_def()
        variables = peptide_size_formula.variables
        so_variables = [v for v in variables if isinstance(v, msol.Var2)]
        fo_variables = [v for v in variables if isinstance(v, msol.Var1)]
        res = f"var2 {','.join(self.compiler.visit(v) for v in so_variables)};\n"
        res += f"var1 {','.join(self.compiler.visit(v) for v in fo_variables)};\n"
        res += self.compiler.visit(peptide_size_formula.formula) + ";\n"
        return res


if __name__ == "__main__":
    # Example usage
    classifier = MonaPeptideSizeClassifierCompiled()
    from rdkit import Chem
    mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)NCC(=O)O")
    print(classifier.classify(mol))
