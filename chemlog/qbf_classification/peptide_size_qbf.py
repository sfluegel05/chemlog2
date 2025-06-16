import logging
import os
import pickle
from copy import deepcopy

from rdkit import Chem

from chemlog.base_classifier import Classifier
from chemlog.preprocessing.mol_to_qbf import mol_to_propositional, get_atom_pvar, get_charge_pvar, get_h_count_pvar, \
    get_bond_pvar, get_bond_type_pvar
from chemlog.qbf_classification import qbf
from chemlog.qbf_classification.qbf_solver import qbf_solver_depqbf, qbf_solver_caqe
from chemlog.msol import peptide_size, msol
from chemlog.qbf_classification.qbf_translator import QBFTranslator
from chemlog.qbf_classification.qbf_utils import qbf_to_cnf, cnf_to_qdimacs


class QBFPeptideSizeClassifierCAQE(Classifier):

    def __init__(self, load_formulas=True, *args, **kwargs):
        if load_formulas:
            self._peptide_formulas = self.load_peptide_formulas()
        else:
            self._peptide_formulas = dict()
        if len(self._peptide_formulas) > 0:
            logging.debug(f"Using {len(self._peptide_formulas)} pre-calculated peptide formulas")

    @property
    def peptide_formula_path(self):
        return os.path.join("data", f"qbf_structures_{self.__class__.__name__}.pkl")

    def load_peptide_formulas(self):
        # load peptide structures from file or initialize empty dict
        if os.path.isfile(self.peptide_formula_path):
            logging.debug(f"Loading peptide formulas from {self.peptide_formula_path}")
            with open(self.peptide_formula_path, "rb") as f:
                return pickle.load(f)
        return dict()

    def on_finish(self):
        # save peptide structures to file
        with open(self.peptide_formula_path, "wb+") as f:
            pickle.dump(self._peptide_formulas, f)

    @staticmethod
    def build_peptide_structure(n_amino_acids, n_atoms):
        # get qbf formula for peptide structure
        amino_acids = [amino_acid_residue(n_atoms, [f"aar_{i}_{j}" for j in range(n_atoms)]) for i in
                       range(n_amino_acids)]
        # aars do not overlap - each atom j only appears once (at most)
        pairwise_inequality = [
            qbf.BinaryFormula(f"aar_{i}_{j}", qbf.Connective.IMPLIES,
                              qbf.NegFormula(qbf.NaryFormula(qbf.Connective.OR, [
                                  f"aar_{k}_{j}" for k in range(n_amino_acids) if k != i])))
            for i in range(n_amino_acids - 1) for j in range(n_atoms)]

        # atom j from aar i+1 and atom l from aar k<i+1 have to be part of an amide bond
        # for any j, j belongs to aar i and for any l, j and l belong to an amide bond and j belongs to any aar < i
        peptide_bond_overlaps = [
            qbf.NaryFormula(qbf.Connective.OR, [
                qbf.NaryFormula(qbf.Connective.AND, [
                    f"aar_{i}_{j}",
                    qbf.NaryFormula(qbf.Connective.OR, [
                        qbf.NaryFormula(qbf.Connective.AND, [
                            qbf.BinaryFormula(exists_amide_given_n_c(n_atoms, j, l), qbf.Connective.OR,
                                              exists_amide_given_n_c(n_atoms, l, j)),
                            qbf.NaryFormula(qbf.Connective.OR, [f"aar_{k}_{l}" for k in range(i + 1)])
                        ])
                        for l in range(n_atoms)
                    ])
                ])
                for j in range(n_atoms)
            ])
            for i in range(1, n_amino_acids)
        ]

        return qbf.QuantifiedFormula(
            qbf.Quantifier.E,
            [f"aar_{i}_{j}" for i in range(n_amino_acids) for j in range(n_atoms)],
            qbf.NaryFormula(qbf.Connective.AND, [
                *amino_acids,
                *pairwise_inequality,
                *peptide_bond_overlaps
            ])
        )

    def get_peptide_structure(self, n_aars, n_atoms):
        if n_aars not in self._peptide_formulas:
            self._peptide_formulas[n_aars] = {n_atoms: qbf_to_cnf(self.build_peptide_structure(n_aars, n_atoms), use_tseytin=True, verbose=False)}
        elif n_atoms not in self._peptide_formulas[n_aars]:
            self._peptide_formulas[n_aars][n_atoms] = qbf_to_cnf(self.build_peptide_structure(n_aars, n_atoms), use_tseytin=True, verbose=False)
        return deepcopy(self._peptide_formulas[n_aars][n_atoms])

    def solve_qdimacs(self, qdimacs):
        return qbf_solver_caqe(qdimacs)

    def classify(self, mol, *args, **kwargs):
        positive_literals, negative_literals = mol_to_propositional(mol)
        n_atoms = mol.GetNumAtoms()

        proof_attempts = []
        for n in range(2, 11):
            logging.debug(f"Running QBF for peptide size {n} with {n_atoms} atoms")
            target_formula = self.get_peptide_structure(n, n_atoms)
            dimacs = [f"c Peptide structure {n}+ ({n_atoms} atoms)"]
            # get matrix
            matrix = target_formula
            while isinstance(matrix, qbf.QuantifiedFormula):
                matrix = matrix.formula
            assert isinstance(matrix, qbf.NaryFormula)
            matrix.formulas = matrix.formulas + [v for v in positive_literals] + [qbf.NegFormula(v) for v in negative_literals]
            dimacs.append(
                cnf_to_qdimacs(target_formula, add_comments=False))

            outcome = self.solve_qdimacs(dimacs)
            proof_attempts.append(
                {"target": n, "outcome": outcome})
            if not outcome:
                return n - 1, proof_attempts
            elif isinstance(outcome, str):
                return 0, proof_attempts
        return 10, proof_attempts


class QBFPeptideSizeClassifierDepQBF(QBFPeptideSizeClassifierCAQE):

    def solve_qdimacs(self, qdimacs):
        return qbf_solver_depqbf(qdimacs)


class QBFPeptideSizeClassifierDepQBFTranslated(QBFPeptideSizeClassifierDepQBF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peptide_definitions = {
            peptide_size.HasOverlap().name(): peptide_size.HasOverlap(),
            peptide_size.IsConnected().name(): peptide_size.IsConnected(),
            peptide_size.CarbonConnected().name(): peptide_size.CarbonConnected(),
            peptide_size.CarbonFragment().name(): peptide_size.CarbonFragment(),
            peptide_size.AmideBond().name(): peptide_size.AmideBond(),
            peptide_size.AmideBondFO().name(): peptide_size.AmideBondFO(),
            peptide_size.AminoGroup().name(): peptide_size.AminoGroup(),
            peptide_size.AminoGroupFO().name(): peptide_size.AminoGroupFO(),
            peptide_size.CarboxyResidue().name(): peptide_size.CarboxyResidue(),
            peptide_size.CarboxyResidueFO().name(): peptide_size.CarboxyResidueFO(),
            peptide_size.BuildingBlock().name(): peptide_size.BuildingBlock(),
            peptide_size.AAR().name(): peptide_size.AAR()
        }

    def build_peptide_structure(self, n_amino_acids, n_atoms):
        peptide_msol = peptide_size.Peptide(n_amino_acids)
        translator = QBFTranslator(n_atoms, self.peptide_definitions)
        return translator.visit(peptide_msol())


def exists_amino(n_atoms, x_vars):
    # \exists A: Amino(A) \land A \subseteq X
    disjunction = []
    for n in range(n_atoms):
        f = qbf.NaryFormula(
            qbf.Connective.AND,
            [x_vars[n], get_atom_pvar(n, 7)] +
            [qbf.BinaryFormula(
                get_bond_pvar(n, x),
                qbf.Connective.IMPLIES,
                qbf.BinaryFormula(
                    qbf.BinaryFormula(
                        get_bond_type_pvar(n, x, Chem.BondType.SINGLE),
                        qbf.Connective.OR,
                        exists_amide_given_n_c(n_atoms, n, x)
                    ),
                    qbf.Connective.AND,
                    get_atom_pvar(x, 6)
                )
            ) for x in range(n_atoms)]
        )
        disjunction.append(f)
    return qbf.NaryFormula(qbf.Connective.OR, disjunction)


def exists_carboxy(n_atoms, x_vars):
    # \exists Y: Carboxy(Y) \land Y \subseteq X

    disjunction = []
    # carboxy residue consists of 3 atoms C(=O)O / C(=O)N where =O is d (double bond) and O/N/... is s (single bond)
    for c in range(n_atoms):
        for s in range(n_atoms):
            for d in range(n_atoms):
                f = qbf.NaryFormula(qbf.Connective.AND, [
                    # atoms are part of X
                    x_vars[c], x_vars[s], x_vars[d],
                    # carboxy formula
                    get_atom_pvar(c, 6),
                    get_atom_pvar(d, 8),
                    get_bond_type_pvar(c, s, Chem.BondType.SINGLE),
                    get_bond_type_pvar(c, d, Chem.BondType.DOUBLE),
                ])
                disjunction.append(f)
    return qbf.NaryFormula(qbf.Connective.OR, disjunction)


def exists_amide_given_n_c(n_atoms: int, n: int, c: int):
    # \exists o: Amide(A) \land A = {c, o, n}
    # this is used for checks within other definitions (e.g., amino groups)

    # A = {c, o, n} -> exists A becomes big disjunction
    disjunction = []
    for o in range(n_atoms):
        f = qbf.NaryFormula(qbf.Connective.AND, [
            # amide formula
            get_atom_pvar(c, 6),
            get_atom_pvar(o, 8),
            get_atom_pvar(n, 7),
            qbf.BinaryFormula(
                qbf.NaryFormula(qbf.Connective.AND, [
                    get_bond_type_pvar(c, o, Chem.BondType.SINGLE),
                    get_bond_type_pvar(c, n, Chem.BondType.DOUBLE),
                    qbf.BinaryFormula(get_h_count_pvar(o, 1), qbf.Connective.OR, get_charge_pvar(o, -1))
                ]),
                qbf.Connective.OR,
                qbf.NaryFormula(qbf.Connective.AND, [
                    get_bond_type_pvar(c, o, Chem.BondType.DOUBLE),
                    get_bond_type_pvar(c, n, Chem.BondType.SINGLE),
                ])
            )
        ])
        disjunction.append(f)
    return qbf.NaryFormula(qbf.Connective.OR, disjunction)


def exists_amide_subset(n_atoms, x_vars):
    # \exists A: Amide(A) \land A \subseteq X
    # x_vars are QBF variables representing the presence of each atom in X

    # A = {c, o, n} -> exists A becomes big disjunction
    disjunction = []
    for c in range(n_atoms):
        for o in range(n_atoms):
            for n in range(n_atoms):
                # X={c, o, n} & C(c) & O(o) & N(n) & [(SB(c,o) & DB(c, n) & (1h(o) | ChargeM1(o))) | (DB(c,o) & SB(c,n))]
                f = qbf.NaryFormula(qbf.Connective.AND, [
                    # atoms are part of X
                    x_vars[c], x_vars[o], x_vars[n],
                    # amide formula
                    get_atom_pvar(c, 6),
                    get_atom_pvar(o, 8),
                    get_atom_pvar(n, 7),
                    qbf.BinaryFormula(
                        qbf.NaryFormula(qbf.Connective.AND, [
                            get_bond_type_pvar(c, o, Chem.BondType.SINGLE),
                            get_bond_type_pvar(c, n, Chem.BondType.DOUBLE),
                            qbf.BinaryFormula(get_h_count_pvar(o, 1), qbf.Connective.OR, get_charge_pvar(o, -1))
                        ]),
                        qbf.Connective.OR,
                        qbf.NaryFormula(qbf.Connective.AND, [
                            get_bond_type_pvar(c, o, Chem.BondType.DOUBLE),
                            get_bond_type_pvar(c, n, Chem.BondType.SINGLE),
                        ])
                    )
                ])
                disjunction.append(f)
    return qbf.NaryFormula(qbf.Connective.OR, disjunction)


def carbon_connected(n_atoms: int, x_vars):
    # check if X is carbon-connected
    all_carbon = qbf.NaryFormula(qbf.Connective.AND, [
        qbf.BinaryFormula(x_vars[i], qbf.Connective.IMPLIES, get_atom_pvar(i, 6))
        for i in range(n_atoms)
    ])
    # any split of X into two non-empty sets A and B leads to a bond between A and B
    splits = qbf.QuantifiedFormula(
        qbf.Quantifier.A,
        [f"a{i}" for i in range(n_atoms)] + [f"b{i}" for i in range(n_atoms)],
        qbf.BinaryFormula(
            qbf.NaryFormula(qbf.Connective.AND, [
                # at least one atom in A / B
                qbf.NaryFormula(qbf.Connective.OR, [f"a{i}" for i in range(n_atoms)]),
                qbf.NaryFormula(qbf.Connective.OR, [f"b{i}" for i in range(n_atoms)]),
                # A != B
                qbf.NaryFormula(qbf.Connective.OR, [
                    qbf.NegFormula(qbf.BinaryFormula(f"a{i}", qbf.Connective.BIIMP, f"b{i}")) for i in range(n_atoms)]),
                # X = A \cup B
                qbf.NaryFormula(qbf.Connective.AND, [
                    qbf.BinaryFormula(x_vars[i], qbf.Connective.BIIMP,
                                      qbf.BinaryFormula(f"a{i}", qbf.Connective.OR, f"b{i}")) for i in range(n_atoms)])
            ]),
            qbf.Connective.IMPLIES,
            # at least one bond between A and B
            qbf.NaryFormula(qbf.Connective.OR, [
                qbf.NaryFormula(qbf.Connective.AND, [f"a{i}", f"b{j}", get_bond_pvar(i, j)])
                for i in range(n_atoms) for j in range(n_atoms)
            ])
        )
    )
    all_carbon.formulas.append(splits)
    return all_carbon


def carbon_component(n_atoms: int, x_vars):
    # check if X is a carbon component
    # X has to be carbon-connected and no Y can exist that is carbon-connected and a superset of X
    is_connected = carbon_connected(n_atoms, x_vars)
    y_vars = [f"z{i}" for i in range(n_atoms)]
    no_superset = qbf.NegFormula(
        qbf.QuantifiedFormula(
            qbf.Quantifier.E,
            y_vars,
            qbf.NaryFormula(
                qbf.Connective.AND,
                [carbon_connected(n_atoms, y_vars)] +
                [qbf.BinaryFormula(x_vars[i], qbf.Connective.IMPLIES, y_vars[i]) for i in range(n_atoms)] +
                [qbf.NegFormula(qbf.NaryFormula(
                    qbf.Connective.AND,
                    [qbf.BinaryFormula(y_vars[i], qbf.Connective.IMPLIES, x_vars[i]) for i in range(n_atoms)]
                ))]
            )
        ))
    return qbf.NaryFormula(qbf.Connective.AND, [is_connected, no_superset])


def building_block(n_atoms: int, x_vars):
    # a building block is a superset of a carbon component in which each atom is either part of the carbon component or
    # has a bond to an atom in the carbon component (except N atoms that are part of an amide bond)
    y_vars = [f"y{i}" for i in range(n_atoms)]
    carbon_comp = carbon_component(n_atoms, y_vars)
    y_subset_x = qbf.NaryFormula(qbf.Connective.AND, [
        qbf.BinaryFormula(y_vars[i], qbf.Connective.IMPLIES, x_vars[i]) for i in range(n_atoms)
    ])
    rules_for_x = qbf.NaryFormula(qbf.Connective.AND, [
        qbf.BinaryFormula(
            x_vars[i],
            qbf.Connective.IMPLIES,
            qbf.BinaryFormula(
                y_vars[i],
                qbf.Connective.OR,
                qbf.NaryFormula(qbf.Connective.OR, [
                    qbf.NaryFormula(
                        qbf.Connective.AND, [
                            y_vars[j],
                            get_bond_pvar(i, j),
                            qbf.BinaryFormula(
                                get_atom_pvar(i, 7),
                                qbf.Connective.IMPLIES,
                                qbf.NegFormula(exists_amide_given_n_c(n_atoms, i, j))
                            )
                        ]
                    )
                    for j in range(n_atoms)
                ])
            )
        )
        for i in range(n_atoms)
    ])
    return qbf.QuantifiedFormula(qbf.Quantifier.E, y_vars,
                                 qbf.NaryFormula(qbf.Connective.AND, [carbon_comp, y_subset_x, rules_for_x]))


def amino_acid_residue(n_atoms: int, x_vars):
    # building block that contains an amino and a carboxy group
    return qbf.NaryFormula(qbf.Connective.AND, [
        building_block(n_atoms, x_vars),
        exists_amino(n_atoms, x_vars),
        exists_carboxy(n_atoms, x_vars)
    ])


def amino_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x_vars = [f"x{i}" for i in range(mol.GetNumAtoms())]
    classifier = QBFPeptideSizeClassifierDepQBFTranslated()
    translator = QBFTranslator(mol.GetNumAtoms(), classifier.peptide_definitions)
    formula = translator.visit(msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [msol.Var2("X")],
                                                      peptide_size.AminoGroup()(msol.Var2("X"))))
    dimacs = ["c \\exists X: Amino(X)"]
    dimacs.append(f"c Target formula: {formula}")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
    dimacs.append(f"c Molecule SMILES: {smiles}")
    dimacs.append(f"c Molecule description: {mol_formula}")
    dimacs.append(cnf_to_qdimacs(qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
    with open(f"qdimacs_demo/amino_example_{smiles}.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))
    print(f"SAT?", qbf_solver_depqbf(dimacs))


def carboxy_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x_vars = [f"x{i}" for i in range(mol.GetNumAtoms())]
    classifier = QBFPeptideSizeClassifierDepQBFTranslated()
    translator = QBFTranslator(mol.GetNumAtoms(), classifier.peptide_definitions)
    formula = translator.visit(msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [msol.Var2("X")],
                                                      peptide_size.CarboxyResidue()(msol.Var2("X"))))
    dimacs = ["c \\exists X: Carboxy(X)"]
    dimacs.append(f"c Target formula: {formula}")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
    dimacs.append(f"c Molecule SMILES: {smiles}")
    dimacs.append(f"c Molecule description: {mol_formula}")
    dimacs.append(cnf_to_qdimacs(qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
    with open(f"qdimacs_demo/carboxy_example_{smiles}.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))
    print(f"SAT?", qbf_solver_depqbf(dimacs))

def amide_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x_vars = [f"x{i}" for i in range(mol.GetNumAtoms())]
    classifier = QBFPeptideSizeClassifierDepQBFTranslated()
    translator = QBFTranslator(mol.GetNumAtoms(), classifier.peptide_definitions)
    formula = translator.visit(msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [msol.Var2("X")],
                                                      peptide_size.AmideBond()(msol.Var2("X"))))
    dimacs = ["c \\exists X: AmideBond(X)"]
    dimacs.append(f"c Target formula: {formula}")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
    dimacs.append(f"c Molecule SMILES: {smiles}")
    dimacs.append(f"c Molecule description: {mol_formula}")
    dimacs.append(cnf_to_qdimacs(qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
    with open(f"qdimacs_demo/amide_example_{smiles}.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))
    print(f"SAT?", qbf_solver_depqbf(dimacs))

def building_block_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    classifier = QBFPeptideSizeClassifierDepQBFTranslated()
    translator = QBFTranslator(mol.GetNumAtoms(), classifier.peptide_definitions)
    y = msol.Var2("Y")
    x = msol.Var2("X")
    u = msol.Var1("uuu")
    v = msol.Var1("vvv")
    w = msol.Var1("www")
    a_o = msol.Var1("a_o")
    bb = (msol.PredicateExpression("has_bond_to", [u, w])
                    & msol.InSetFormula(u, msol.Var2("N")) &
                        ~msol.QuantifiedFormula(
                            msol.Quantifier.EXISTENTIAL, [a_o],
                            msol.PredicateExpression(peptide_size.AmideBondFO().name(), [u, a_o, w])
                        )
                    )


    for u_index in [1,2]:# range(mol.GetNumAtoms()): #[0, 3, 4, 5, 6, 7]:
        v_index = u_index + 1
        print(f"Setting u to {u_index}, w to 0")
        formula = translator.visit(bb, var_indices={"uuu": u_index, "www": 0})
        dimacs = ["c \\exists X: BuildingBlock(X)"]
        dimacs.append(f"c Target formula: {formula}")

        # add molecule to qbf
        mol_prop = mol_to_propositional(mol)
        print(f"Molecule properties: {mol_prop}")
        mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                      [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
        all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
        dimacs.append(f"c Molecule SMILES: {smiles}")
        dimacs.append(f"c Molecule description: {mol_formula}")
        dimacs.append(cnf_to_qdimacs(qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
        with open(f"qdimacs_demo/building_block_example_{smiles}.qdimacs",
                  "w", encoding="utf-8") as f:
            f.write("\n".join(dimacs))
        print(f"SAT?", qbf_solver_depqbf(dimacs))


def aar_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x_vars = [f"x{i}" for i in range(mol.GetNumAtoms())]
    classifier = QBFPeptideSizeClassifierDepQBF()
    #translator = QBFTranslator(mol.GetNumAtoms(), classifier.peptide_definitions)
    #formula = translator.visit(msol.QuantifiedFormula(msol.Quantifier.EXISTENTIAL, [msol.Var2("X")],
    #                                                  peptide_size.AAR()(msol.Var2("X"))))
    formula = amino_acid_residue(mol.GetNumAtoms(), x_vars)
    dimacs = ["c \\exists X: AAR(X)"]
    dimacs.append(f"c Target formula: {formula}")
    print(f"Target formula: {formula}")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.QuantifiedFormula(qbf.Quantifier.E, x_vars, qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula))
    dimacs.append(f"c Molecule SMILES: {smiles}")
    dimacs.append(f"c Molecule description: {mol_formula}")
    dimacs.append(cnf_to_qdimacs(qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
    with open(f"qdimacs_demo/aar_example_{smiles}.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))
    print(f"SAT?", qbf_solver_depqbf(dimacs))

def di_plus_peptide_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    classifier = QBFPeptideSizeClassifierDepQBFTranslated()
    formula = classifier.build_peptide_structure(3, mol.GetNumAtoms())
    dimacs = ["c \\exists X, Y: AAR(X) & AAR(Y) & X \\cap Y = \\emptyset"]
    #dimacs.append(f"c Target formula: {formula}")
    logging.debug(f"Target formula: {formula}")
    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
    dimacs.append(f"c Molecule SMILES: {smiles}")

    dimacs.append(f"c Molecule description: {mol_formula}")
    dimacs.append(cnf_to_qdimacs(qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
    with open(f"qdimacs_demo/di_plus_example_{smiles}.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))
    print(f"SAT?", qbf_solver_depqbf(dimacs))


if __name__ == "__main__":
    # no peptide
    smiles_no_peptide = "N1C(C(NCC1C2=CC=CC=C2)C)C"  # CHEBI:183966, not a peptide
    smiles_no_peptide2 = r"CCOC(=O)\C=C\c1ccccc1"  # CHEBI:4895
    # dipeptide
    piperazine = "O=C1CNC(=O)CN1"  # CHEBI:16535
    glycylglycine = "NCC(=O)NCC(=O)O"  # CHEBI:17201
    n_acetyl_methionyl_isoleucine = "CC[C@H](C)[C@H](NC(=O)[C@H](CCSC)NC(C)=O)C(=O)O"  # CHEBI:134478
    methyl_piperazine = "O=C1NCC(=O)NC1C" # CHEBI:144050
    # tripeptide
    glycyl_glycyl_glycine = "NCC(=O)NCC(=O)NCC(=O)O"  # CHEBI:63961
    sulfocysteinyl_glycine = "S(=O)(=O)(O)N[C@@H](CS)C(=O)NCC(=O)O"  # CHEBI:195396
    classifier = QBFPeptideSizeClassifierDepQBFTranslated(load_formulas=True)
    logging.basicConfig(level=logging.DEBUG)
    di_plus_peptide_example(piperazine)
    #print(classifier.classify(Chem.MolFromSmiles("O=C(N[C@H](C(=O)OC)CC(C)C)C(=O)N")))

