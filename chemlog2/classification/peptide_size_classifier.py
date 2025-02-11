import logging

from networkx.classes import neighbors
from rdkit import Chem
import networkx as nx
from itertools import product


def get_n_amino_acid_residues(mol, max_amino_assignments: int = 10000) -> (int, dict):
    """
    Determine the number of amino acid residues that are connected via peptide bonds in a molecule.
    This method does not distinguish between 1 and 0. None is returned if some error has occurred.

    An amino acid residue is delimited by heteroatoms.

    max_amino_assignments: int - maximum number of amino acid assignments to consider (due to resource limitations,
    e.g. Pubchem 59097399 would require iterating 2^27 assignments)
    """

    amide_bonds, amide_bond_c_idxs, amide_bond_o_idxs, amide_bond_n_idxs = get_amide_bonds(mol)
    add_output = {"amide_bond": [(c, o, n) for c, o, n in zip(amide_bond_c_idxs, amide_bond_o_idxs, amide_bond_n_idxs)]}
    if len(amide_bonds) == 0:
        return 0, add_output
    carboxys = list(get_carboxy_derivatives(mol))
    carboxy_c_idxs = [c for c, _, _ in carboxys]
    add_output["carboxy_residue"] = carboxys
    amino_group_idxs = get_amino_groups(mol, amide_bond_c_idxs)
    add_output["amino_residue"] = amino_group_idxs

    # get carbon skeleton minus amide bonds
    chunks = get_chunks(mol, amide_bonds)
    add_output["chunks"] = chunks
    # for amino groups, it might be unclear to which chunk they belong -> try all options, e.g. for CHEBI:76162
    possible_amino_chunk_assignments = get_possible_amino_chunk_assignments(
        mol, amino_group_idxs, chunks, amide_bond_n_idxs, amide_bond_c_idxs, carboxy_c_idxs
    )

    # iterate over possible assignments of amino groups to chunks
    longest_aa_chain = []
    longest_aa_chain_with_atoms = []
    i = 0
    for amino_assignment in product(*possible_amino_chunk_assignments):
        i += 1
        if i > max_amino_assignments:
            logging.warning(f"Max amino assignments reached ({max_amino_assignments})")
            add_output["warning"] = f"Max amino assignments reached ({max_amino_assignments})"
            break
        # amino acid: carboxy residue and amino group in carbon-connected subgraph
        is_amino_acid = [i in amino_assignment for i in range(len(chunks))]
        if sum(is_amino_acid) < 2:
            continue
        # get amide bond connections between amino acids
        amino_acid_graph = nx.Graph()
        amino_acid_graph.add_nodes_from([i for i in range(len(chunks)) if is_amino_acid[i]])
        for amide_c_idx, amide_n_idx in zip(amide_bond_c_idxs, amide_bond_n_idxs):
            n_aa = amino_assignment[amino_group_idxs.index(amide_n_idx)]
            if n_aa >= 0 and is_amino_acid[n_aa]:
                for i, aa in [(idx, chunks[idx]) for idx in range(len(chunks)) if is_amino_acid[idx]]:
                    if amide_c_idx in aa:
                        amino_acid_graph.add_edge(i, n_aa)
                        break
        for aa_chain in nx.connected_components(amino_acid_graph):
            if len(aa_chain) > len(longest_aa_chain):
                longest_aa_chain = aa_chain
                longest_aa_chain_with_atoms = [
                    chunks[i] + [a for a, assign in zip(amino_group_idxs, amino_assignment) if assign == i] for i in
                    # use dfs_preorder to get a traversal of the peptide where each amino acid is connected to the
                    # previous one
                    nx.dfs_preorder_nodes(amino_acid_graph, source=list(aa_chain)[0])]
    add_output["longest_aa_chain"] = longest_aa_chain_with_atoms
    return len(longest_aa_chain), add_output


def get_possible_amino_chunk_assignments(mol, amino_group_idxs, chunks, amide_bond_n_idxs, amide_bond_c_idxs,
                                         carboxy_c_idxs):
    possible_amino_chunk_assignments = []
    for amino in amino_group_idxs:
        chunk_assignments = []
        no_amide_neighbors = [neighbor for neighbor in mol.GetAtomWithIdx(amino).GetNeighbors()
                              if not any(amide_n == amino and amide_c == neighbor.GetIdx()
                                         for amide_n, amide_c in zip(amide_bond_n_idxs, amide_bond_c_idxs))]
        for i, chunk in enumerate(chunks):
            # assign only to chunks that also have a carboxy derivative -> skip possible assignments that dont result
            # in amino acids
            if (any(no_amide_neighbor.GetIdx() in chunk for no_amide_neighbor in no_amide_neighbors)
                    and any(carboxy_c in chunk for carboxy_c in carboxy_c_idxs)):
                chunk_assignments.append(i)
        if len(chunk_assignments) == 0:
            chunk_assignments = [-1]
        possible_amino_chunk_assignments.append(chunk_assignments)
    return possible_amino_chunk_assignments


def get_chunks(mol, amide_bonds):
    carbon_graph = nx.Graph()
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6 and bond not in amide_bonds:
            carbon_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return [list(comp) for comp in nx.connected_components(carbon_graph)]


def get_amide_bonds(mol):
    bonds = []
    c_idxs = []
    n_idxs = []
    o_idxs = []
    single_os = [(atom.GetIdx(), o_atom.GetIdx()) for atom in mol.GetAtoms() for o_atom in atom.GetNeighbors()
                 if o_atom.GetAtomicNum() == 8
                 and mol.GetBondBetweenAtoms(atom.GetIdx(), o_atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                 and (o_atom.GetTotalNumHs(includeNeighbors=True) == 1 or o_atom.GetFormalCharge() == -1)]
    double_os = [(atom.GetIdx(), o_atom.GetIdx()) for atom in mol.GetAtoms() for o_atom in atom.GetNeighbors()
                 if o_atom.GetAtomicNum() == 8
                 and mol.GetBondBetweenAtoms(atom.GetIdx(), o_atom.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]

    for bond in mol.GetBonds():
        for n_atom, c_atom in [[bond.GetBeginAtom(), bond.GetEndAtom()], [bond.GetEndAtom(), bond.GetBeginAtom()]]:
            if n_atom.GetAtomicNum() == 7 and c_atom.GetAtomicNum() == 6:
                # nitrogen atom is not allowed to have bonds to heteroatoms
                # nitrogen is only allowed to have single bonds (except for imidic form of amide)
                if not all(neighbor.GetAtomicNum() in [1, 6]
                           and (mol.GetBondBetweenAtoms(neighbor.GetIdx(),
                                                        n_atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                                or (neighbor.GetIdx() in [c for c, o in single_os]
                                    and mol.GetBondBetweenAtoms(neighbor.GetIdx(),
                                                                n_atom.GetIdx()).GetBondType() == Chem.BondType.DOUBLE))
                           for neighbor in n_atom.GetNeighbors()):
                    break
                # amide vs imidic form: accept NC=O or N=CO
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n_atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                    look_for_bond = double_os
                elif mol.GetBondBetweenAtoms(c_atom.GetIdx(), n_atom.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                    look_for_bond = single_os
                else:
                    look_for_bond = []
                for c, o in look_for_bond:
                    if c_atom.GetIdx() == c:
                        bonds.append(bond)
                        c_idxs.append(c_atom.GetIdx())
                        n_idxs.append(n_atom.GetIdx())
                        o_idxs.append(o)
    return bonds, c_idxs, o_idxs, n_idxs


def get_carboxy_derivatives(mol):
    # C(=O)Y where Y != C,H
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 6):
            double_neighbor = None
            single_neighbor = None
            for neighbor in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                # neighbor.GetAtomicNum() == 8 -> strict C(=O)Y, neighbor.GetAtomicNum() not in [1,6] -> advanced C(=X)Y
                if neighbor.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE:
                    double_neighbor = neighbor
                if neighbor.GetAtomicNum() not in [1, 6] and bond.GetBondType() == Chem.BondType.SINGLE:
                    single_neighbor = neighbor
            if double_neighbor is not None and single_neighbor is not None:
                yield atom.GetIdx(), double_neighbor.GetIdx(), single_neighbor.GetIdx()


def get_amino_groups(mol, amide_bond_c_idxs):
    aminos = []
    for atom in mol.GetAtoms():
        # accept nitrogen, bonded to only hydrogen and carbon
        if (atom.GetAtomicNum() == 7
                and all(neighbor.GetAtomicNum() in [1, 6]
                        # accept only single bonds, except for imidic form of amide
                        and (mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                     neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                             or neighbor.GetIdx() in amide_bond_c_idxs) for neighbor in atom.GetNeighbors())):
            aminos.append(atom.GetIdx())
    return aminos
