import logging
from copy import deepcopy

from rdkit import Chem

from chemlog.base_classifier import Classifier
from chemlog.msol import peptide_charge
from chemlog.preprocessing.mol_to_qbf import mol_to_propositional
from chemlog.qbf_classification import qbf
from chemlog.qbf_classification.qbf_solver import qbf_solver_depqbf
from chemlog.qbf_classification.qbf_translator import QBFTranslator
from chemlog.qbf_classification.qbf_utils import qbf_to_cnf, cnf_to_qdimacs


class QBFChargeClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.translator_cls = QBFTranslator
        self.max_charge_level = 3
        self._target_formulas = {}
        self.target_predicates = {
            peptide_charge.Salt(self.max_charge_level).name(): peptide_charge.Salt(self.max_charge_level),
            peptide_charge.Zwitterion().name(): peptide_charge.Zwitterion(),
            peptide_charge.OrganicAnion().name(): peptide_charge.OrganicAnion(),
            peptide_charge.OrganicCation().name(): peptide_charge.OrganicCation(),
            peptide_charge.Neutral().name(): peptide_charge.Neutral(),
        }
        self.background_predicates = [peptide_charge.HasChargeComponent(-i) for i in range(1, self.max_charge_level + 1)] \
            + [peptide_charge.HasChargeComponent(i) for i in range(1, self.max_charge_level + 1)] \
            + [peptide_charge.IsConnected(), peptide_charge.ConnectedComponent(),
               peptide_charge.Salt(self.max_charge_level), peptide_charge.Zwitterion(),
               peptide_charge.OrganicAnion(), peptide_charge.OrganicCation()]
        self.background_predicates = {pred.name(): pred for pred in self.background_predicates}

    def build_target_formula(self, target_predicate: str, n_atoms: int):
        target_predicate = self.target_predicates[target_predicate]
        translator = QBFTranslator(n_atoms, self.background_predicates)
        return translator.visit(target_predicate())

    def get_target_formula(self, target_predicate: str, n_atoms: int):
        if target_predicate not in self._target_formulas:
            self._target_formulas[target_predicate] = {
                n_atoms: qbf_to_cnf(self.build_target_formula(target_predicate, n_atoms),
                                             use_tseytin=True, verbose=False)}
        elif n_atoms not in self._target_formulas[target_predicate]:
            self._target_formulas[target_predicate][n_atoms] = qbf_to_cnf(
                self.build_target_formula(target_predicate, n_atoms), use_tseytin=True, verbose=False)
        return deepcopy(self._target_formulas[target_predicate][n_atoms])

    def classify(self, mol, *args, **kwargs):
        positive_literals, negative_literals = mol_to_propositional(mol)
        n_atoms = mol.GetNumAtoms()

        proof_attempts = []
        for target_predicate in self.target_predicates:
            logging.debug(f"Running QBF for target {target_predicate} with {n_atoms} atoms")
            target_formula = self.get_target_formula(target_predicate, n_atoms)
            dimacs = [f"c {target_predicate} ({n_atoms} atoms)"]
            # get matrix
            matrix = target_formula
            while isinstance(matrix, qbf.QuantifiedFormula):
                matrix = matrix.formula
            assert isinstance(matrix, qbf.NaryFormula)
            matrix.formulas = matrix.formulas + [v for v in positive_literals] + [qbf.NegFormula(v) for v in
                                                                                  negative_literals]
            dimacs.append(
                cnf_to_qdimacs(target_formula, add_comments=False))

            outcome = self.solve_qdimacs(dimacs)
            proof_attempts.append({"target": target_predicate, "outcome": outcome})
            if outcome:
                return target_predicate, proof_attempts
            elif isinstance(outcome, str):
                return None, proof_attempts
        return "Neutral", proof_attempts

    def solve_qdimacs(self, qdimacs):
        return qbf_solver_depqbf(qdimacs)


if __name__ == "__main__":
    target = peptide_charge.HasChargeComponent(1)
    classifier = QBFChargeClassifier()
    mol = Chem.MolFromSmiles("O=C1CNC(=O)CN1")
    positive_literals, negative_literals = mol_to_propositional(mol)

    translator = QBFTranslator(mol.GetNumAtoms(), classifier.background_predicates)
    from chemlog.msol import msol
    target = msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [msol.Var2("X")],
                                    target(msol.Var2("X")) & ~msol.InSetFormula("0", msol.Var2("X")))
    print(translator.visit(target))

    dimacs = [f"c ... ({mol.GetNumAtoms()} atoms)"]
    # get matrix
    target = translator.visit(target)
    target = qbf_to_cnf(target, use_tseytin=True, verbose=False)
    matrix = target
    while isinstance(matrix, qbf.QuantifiedFormula):
        matrix = matrix.formula
    matrix.formulas = matrix.formulas + [v for v in positive_literals] + [qbf.NegFormula(v) for v in
                                                                          negative_literals]
    print(target)
    dimacs.append(
        cnf_to_qdimacs(target, add_comments=False))

    outcome = classifier.solve_qdimacs(dimacs)
    print(outcome)