from chemlog.base_classifier import Classifier, ChargeCategories
from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from chemlog.mona_classification.mona_compiler import MONACompiler
from chemlog.mona_classification.mona_model_checker import MonaModelChecker
from chemlog.msol import peptide_charge
from inspect import signature
from chemlog.msol import msol
from chemlog.preprocessing.mol_to_msol import mol_to_msol


class MonaChargeClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.compiler = MONACompiler()
        self.max_charge_level = 3
        self.predicate_definitions = self.load_predicate_definitions()

    def load_predicate_definitions(self):
        # load from internal MSOL representation
        defs_compiled = []
        for definition in [peptide_charge.HasChargeComponent(-i) for i in range(1, self.max_charge_level + 1)] \
                        + [peptide_charge.HasChargeComponent(i) for i in range(1, self.max_charge_level + 1)]\
                        + [peptide_charge.IsConnected(), peptide_charge.ConnectedComponent(),
                           peptide_charge.Salt(self.max_charge_level), peptide_charge.Zwitterion(),
                           peptide_charge.OrganicAnion(), peptide_charge.OrganicCation()]:

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

    def classify(self, mol, *args, **kwargs):
        universe, mol_mona = mol_to_msol(mol)
        model_checker = MonaModelChecker(universe, mol_mona, self.predicate_definitions)
        proof_attempts = []
        for target_predicate, target_category in [
            (peptide_charge.Salt(self.max_charge_level), ChargeCategories.SALT),
            (peptide_charge.Zwitterion(), ChargeCategories.ZWITTERION),
            (peptide_charge.OrganicAnion(), ChargeCategories.ANION),
            (peptide_charge.OrganicCation(), ChargeCategories.CATION)]:
            target_formula = self.compiler.visit(msol.PredicateExpression(target_predicate.name(), [])) + ";\n"
            outcome = model_checker.find_model(target_formula)
            proof_attempts.append(
                {"target": target_predicate.name(), "variable_assignments": outcome[1], "outcome": outcome[0].name})
            if outcome[0] in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                return target_category.name, {"proof_attempts": proof_attempts}
        # neutral is if all else fails
        return ChargeCategories.NEUTRAL.name, {"proof_attempts": proof_attempts}