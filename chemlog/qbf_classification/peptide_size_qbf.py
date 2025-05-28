import logging
import os

from rdkit import Chem

from chemlog.preprocessing.mol_to_qbf import mol_to_propositional, get_atom_pvar, get_charge_pvar, get_h_count_pvar, \
    get_bond_pvar, get_bond_type_pvar
from chemlog.qbf_classification import qbf
from chemlog.qbf_classification.qbf_solver import qbf_solver


class QBFPeptideSizeClassifier:

    def classify_peptide_size_qbf(self, mol):
        positive_literals, negative_literals = mol_to_propositional(mol)
        n_atoms = mol.GetNumAtoms()

        proof_attempts = []
        for n in range(2, 11):
            logging.debug(f"Running QBF for peptide size {n} with {n_atoms} atoms")
            target_formula = build_peptide_structure(n, n_atoms)
            dimacs = [f"c Peptide structure {n}+"]
            dimacs.append(f"c Target formula: {target_formula}")

            formula = qbf.BinaryFormula(
                qbf.NaryFormula(
                    qbf.Connective.AND,
                    [v for v in positive_literals] + [qbf.NegFormula(v) for v in negative_literals]
                ),
                qbf.Connective.AND,
                target_formula
            )
            dimacs.append(qbf.cnf_to_qdimacs(qbf.qbf_to_cnf(formula, use_tseytin=True, verbose=False)))

            outcome = qbf_solver(dimacs)
            proof_attempts.append(
                {"target": n, "outcome": outcome})
            if not outcome:
                return n - 1, proof_attempts
            elif isinstance(outcome, str):
                return 0, proof_attempts
        return 10, proof_attempts


def build_peptide_structure(n_amino_acids, n_atoms):
    # get qbf formula for peptide structure
    amino_acids = [amino_acid_residue(n_atoms, [f"aar_{i}_{j}" for j in range(n_atoms)]) for i in range(n_amino_acids)]
    peptide_bonds = [exists_amide_subset(n_atoms, [f"pb_{i}_{j}" for j in range(n_atoms)]) for i in
                     range(n_amino_acids - 1)]
    # aars do not overlap - each atom j only appears once (at most)
    pairwise_inequality = [
        qbf.BinaryFormula(f"aar_{i}_{j}", qbf.Connective.IMPLIES, qbf.NegFormula(qbf.NaryFormula(qbf.Connective.OR, [
            f"aar_{k}_{j}" for k in range(n_amino_acids) if k != i])))
        for i in range(n_amino_acids - 1) for j in range(n_atoms)]
    # peptide bond i overlaps aar i+1
    bond_peptide_overlaps = [
        qbf.NaryFormula(qbf.Connective.OR, [
            qbf.BinaryFormula(f"pb_{i}_{j}", qbf.Connective.AND, f"aar_{i + 1}_{j}")
            for j in range(n_atoms)
        ])
        for i in range(n_amino_acids - 1)
    ]
    # peptide bond i overlaps aar <= i
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
        [f"aar_{i}_{j}" for i in range(n_amino_acids) for j in range(n_atoms)],  # +
        # [f"pb_{i}_{j}" for i in range(n_amino_acids - 1) for j in range(n_atoms)],
        qbf.NaryFormula(qbf.Connective.AND, [
            *amino_acids,
            # *peptide_bonds,
            *pairwise_inequality,
            # *bond_peptide_overlaps,
            *peptide_bond_overlaps
        ])
    )


def exists_amino(n_atoms, x_vars):
    # \exists A: Amino(A) \land A \subseteq X
    disjunction = []
    for n in range(n_atoms):
        f = qbf.NaryFormula(qbf.Connective.AND, [
            x_vars[n],
            get_atom_pvar(n, 7),
        ] + [
                                qbf.BinaryFormula(
                                    get_bond_pvar(n, x),
                                    qbf.Connective.IMPLIES,
                                    qbf.BinaryFormula(
                                        get_bond_type_pvar(n, x, Chem.BondType.SINGLE),
                                        qbf.Connective.OR,
                                        exists_amide_given_n_c(n_atoms, n, x)
                                    )
                                )
                                for x in range(n_atoms)])
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
    # mol = Chem.MolFromSmiles("O=C1CNC(=O)CN1") # CHEBI:16535
    amino_formula = exists_amino(mol.GetNumAtoms(), [f"x{i}" for i in range(mol.GetNumAtoms())])
    print("c \\exists A: Amino(A) & A \\subseteq X")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, amino_formula)
    print(f"c Molecule description: {mol_formula}")
    print(f"c Dimacs for molecule")
    print(qbf.cnf_to_qdimacs(qbf.qbf_to_cnf(all_formula)))


def carboxy_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    carboxy_formula = exists_carboxy(mol.GetNumAtoms(), [f"x{i}" for i in range(mol.GetNumAtoms())])
    dimacs = ["c \\exists C: Carboxy(C) & C \\subseteq X"]

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, carboxy_formula)
    dimacs.append(f"c Molecule description: {mol_formula}")
    print(dimacs)
    dimacs.append(qbf.cnf_to_qdimacs(qbf.qbf_to_cnf(all_formula)))
    with open(r"C:\Users\sifluegel\Downloads\depqbf-version-6.03\depqbf-version-6.03\peptides\carboxy_example.qdimacs",
              "w") as f:
        f.writelines(dimacs)


def aar_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x_vars = [f"x{i}" for i in range(mol.GetNumAtoms())]
    formula = qbf.QuantifiedFormula(qbf.Quantifier.E, x_vars, amino_acid_residue(mol.GetNumAtoms(), x_vars))
    dimacs = ["c \\exists X: AAR(X)"]
    dimacs.append(f"c Target formula: {formula}")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
    dimacs.append(f"c Molecule SMILES: {smiles}")
    dimacs.append(f"c Molecule description: {mol_formula}")
    print("\n".join(dimacs))
    dimacs.append(qbf.cnf_to_qdimacs(qbf.qbf_to_cnf(all_formula, use_tseytin=True, verbose=True), add_comments=False))
    print(dimacs)
    with open("qdimacs_demo/aar_example.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))


def di_plus_peptide_example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    formula = build_peptide_structure(2, mol.GetNumAtoms())
    dimacs = ["c \\exists X, Y: AAR(X) & AAR(Y) & X \\cap Y = \\emptyset"]
    dimacs.append(f"c Target formula: {formula}")

    # add molecule to qbf
    mol_prop = mol_to_propositional(mol)
    mol_formula = qbf.NaryFormula(qbf.Connective.AND,
                                  [v for v in mol_prop[0]] + [qbf.NegFormula(v) for v in mol_prop[1]])
    all_formula = qbf.BinaryFormula(mol_formula, qbf.Connective.AND, formula)
    dimacs.append(f"c Molecule SMILES: {smiles}")
    print("\n".join(dimacs))

    dimacs.append(f"c Molecule description: {mol_formula}")
    dimacs.append(qbf.cnf_to_qdimacs(qbf.qbf_to_cnf(all_formula, use_tseytin=True, verbose=False), add_comments=False))
    with open("qdimacs_demo/di_plus_example.qdimacs",
              "w", encoding="utf-8") as f:
        f.write("\n".join(dimacs))
    print(f"SAT?", qbf_solver(dimacs))


if __name__ == "__main__":
    # no peptide
    smiles_no_peptide = "N1C(C(NCC1C2=CC=CC=C2)C)C"  # CHEBI:183966, not a peptide
    smiles_no_peptide2 = r"CCOC(=O)\C=C\c1ccccc1"  # CHEBI:4895
    # dipeptide
    piperazine = "O=C1CNC(=O)CN1"  # CHEBI:16535
    glycylglycine = "NCC(=O)NCC(=O)O"  # CHEBI:17201
    n_acetyl_methionyl_isoleucine = "CC[C@H](C)[C@H](NC(=O)[C@H](CCSC)NC(C)=O)C(=O)O"  # CHEBI:134478
    # tripeptide
    glycyl_glycyl_glycine = "NCC(=O)NCC(=O)NCC(=O)O"  # CHEBI:63961
    classifier = QBFPeptideSizeClassifier()
    print(classifier.classify_peptide_size_qbf(Chem.MolFromSmiles(glycyl_glycyl_glycine)))
