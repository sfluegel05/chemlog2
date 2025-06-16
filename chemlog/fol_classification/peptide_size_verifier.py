import logging
from inspect import signature

from rdkit import Chem
from gavel.dialects.tptp.parser import TPTPParser
from gavel.logic import logic
import os

from chemlog.base_classifier import Classifier
from chemlog.fol_classification.fol_utils import normalize_fol_formula
from chemlog.fol_classification.msol_to_fol_translator import FOLTranslator
from chemlog.preprocessing.mol_to_fol import mol_to_fol_building_blocks, apply_variable_assignment, mol_to_fol_atoms_plus_building_blocks
from chemlog.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome
from chemlog.msol import peptide_size


class PeptideSizeVerifier(Classifier):

    def __init__(self):
        self.structure_formulas = self.get_structure_formulas()
        for f in self.structure_formulas.values():
            f.right = normalize_fol_formula(f.right)
        logging.debug(f"Loaded {len(self.structure_formulas)} peptide structure formulas:")
        logging.debug('\n'.join([f'{k}: {v}' for k, v in self.structure_formulas.items()]))

        self._peptide_formulas = dict()

    @staticmethod
    def get_fol_structure(mol: Chem.Mol, functional_groups=None):
        return mol_to_fol_building_blocks(mol, functional_groups=functional_groups)

    @staticmethod
    def get_structure_formulas():
        with open(os.path.join("data", "fol_specifications", "peptide_structure_helpers.tptp"), "r") as f:
            tptp_raw = f.readlines()
        tptp_parser = TPTPParser()
        tptp_parsed = [tptp_parser.parse(formula) for formula in tptp_raw]
        # take right-hand side of formulas
        return {f[0].formula.left.predicate.value: f[0].formula for f in tptp_parsed if len(f) > 0}

    @staticmethod
    def build_peptide_structure_formula(n: int):
            variables = [logic.Variable(f"A{i}") for i in range(n)] + [logic.Variable(f"B{i}") for i in range(n - 1)]
            clauses = []
            for i in range(n):
                clauses.append(logic.PredicateExpression("amino_acid_residue", [variables[i]]))
                for j in range(i + 1, n):
                    clauses.append(logic.UnaryFormula(
                        logic.UnaryConnective.NEGATION,
                        logic.PredicateExpression("overlap", [variables[i], variables[j]])
                    ))
            for i in range(n - 1):
                clauses.append(logic.PredicateExpression("amide_bond", [variables[n + i]]))
                clauses.append(logic.PredicateExpression("overlap", [variables[i + 1], variables[n + i]]))
                disj = [logic.PredicateExpression("overlap", [variables[j], variables[n + i]]) for j in range(0, i + 1)]
                if len(disj) == 1:
                    clauses.append(disj[0])
                else:
                    clauses.append(logic.NaryFormula(logic.BinaryConnective.DISJUNCTION, disj))
            return logic.QuantifiedFormula(logic.Quantifier.EXISTENTIAL, variables,
                                           logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, clauses))

    def verify_n_plus_amino_acids(self, mol: Chem.Mol, expected_n: int, functional_group_extensions, variable_assignment: dict):
        # for functional_group_extensions, assume that they are true
        # note that this only verifies n+, not n (does not check if n+1 fails)
        universe, extensions, second_order_elements = self.get_fol_structure(mol, functional_group_extensions)
        logging.debug(f"Using the following second-order elements: {', '.join([str(i) + ' -> ' + str(v) for i, v in enumerate(second_order_elements)])}")
        # this model checker uses amide_bond, amino_residue and carboxy_residue from the extension and
        # amino_acid_residue from the definition (if it is not already in the extension)
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                                                         for pred, formula in self.structure_formulas.items()})
        # use second_order_elements to map variable_assignment from list of atoms to index in extension
        variable_assignment = {k: second_order_elements.index(v) for k, v in variable_assignment.items()}
        proof_attempts = []
        target_formula = self.build_peptide_structure_formula(expected_n)
        target_formula = apply_variable_assignment(target_formula, variable_assignment)
        result = model_checker.find_model(target_formula)[0]
        proof_attempts.append(
            {"target": expected_n, "variable_assignments": variable_assignment, "outcome": result.name})
        return result, proof_attempts

    def get_peptide_formula(self, n_amino_acid_residues: int):
        if n_amino_acid_residues not in self._peptide_formulas:
            self._peptide_formulas[n_amino_acid_residues] = normalize_fol_formula(self.build_peptide_structure_formula(n_amino_acid_residues))
        return self._peptide_formulas[n_amino_acid_residues]

    def classify(self, mol: Chem.Mol, functional_groups=None, *args, **kwargs) -> (int, dict):
        # for functional_group_extensions, assume that they are true
        universe, extensions, second_order_elements = self.get_fol_structure(mol, functional_groups)
        logging.debug(f"Using the following second-order elements: "
                      f"{', '.join([str(i) + ' -> ' + str(v) for i, v in enumerate(second_order_elements)])}")
        # this model checker uses amide_bond, amino_residue and carboxy_residue from the extension and
        # amino_acid_residue from the definition (if it is not already in the extension)
        model_checker = ModelChecker(
            universe, extensions, predicate_definitions={pred: (formula.left.arguments, formula.right)
                                                         for pred, formula in self.structure_formulas.items()})
        assignment = None
        for n in range(2, 11):
            target_formula = self.get_peptide_formula(n)
            logging.debug(f"Target formula for n={n}: {target_formula}")

            outcome = model_checker.find_model(target_formula)
            if outcome[0] in [ModelCheckerOutcome.NO_MODEL, ModelCheckerOutcome.NO_MODEL_INFERRED]:
                return n - 1, assignment
            elif outcome[0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                return -1, None
            # map second-order element back to atoms
            assignment = {v: second_order_elements[ind - (mol.GetNumAtoms() + 1)] if ind > mol.GetNumAtoms() else ind for v, ind in outcome[1]}
        return 10, {"size_assignment": assignment}


class FOLPeptideSizeClassifierTranslated(PeptideSizeVerifier):

    @staticmethod
    def get_fol_structure(mol: Chem.Mol, functional_groups=None):
        return mol_to_fol_atoms_plus_building_blocks(mol)

    @staticmethod
    def get_structure_formulas():
        defs_compiled = dict()
        fol_translator = FOLTranslator(["C", "N", "O", "Has1Hs", "ChargeN"])
        for definition in [peptide_size.HasOverlap(), peptide_size.AmideBondFO(),
                           peptide_size.AminoGroupFO(), peptide_size.CarboxyResidueFO(),
                           peptide_size.AAR()]:
            sig = signature(definition.__call__)
            variables = []
            for p_name, param in sig.parameters.items():
                variables.append(param.annotation(param.name))
            deff = fol_translator.visit(definition(*variables))
            defs_compiled[definition.name()] = logic.BinaryFormula(logic.PredicateExpression(definition.name(), variables),
                                                                   logic.BinaryConnective.BIIMPLICATION, deff)
        return defs_compiled

    @staticmethod
    def build_peptide_structure_formula(n: int):
        peptide = peptide_size.Peptide(n)
        peptide_definitions = {peptide_size.HasOverlap().name(): peptide_size.HasOverlap(),
                               #peptide_size.AmideBondFO().name(): peptide_size.AmideBondFO(),
                               #peptide_size.AminoGroupFO().name(): peptide_size.AminoGroupFO(),
                               peptide_size.CarboxyResidueFO().name(): peptide_size.CarboxyResidueFO(),
                               peptide_size.AAR().name(): peptide_size.AAR()}
        translator = FOLTranslator(["C", "N", "O", "Has1Hs", "ChargeN"], predicate_definitions=peptide_definitions)
        return translator.visit(peptide())




if __name__ == "__main__":
    # Example usage
    verifier = FOLPeptideSizeClassifierTranslated()
    piperazine = "O=C1CNC(=O)CN1"  # CHEBI:16535
    mol = Chem.MolFromSmiles(piperazine)

    n, assignment = verifier.classify(mol)
    print(f"{n} amino acid residues: {assignment}")