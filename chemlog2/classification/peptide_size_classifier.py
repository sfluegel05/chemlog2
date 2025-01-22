import logging

from networkx.classes import neighbors
from rdkit import Chem
import networkx as nx
from itertools import product


def get_n_amino_acid_residues(mol) -> (int, dict):
    """
    Determine the number of amino acid residues that are connected via peptide bonds in a molecule.
    This method does not distinguish between 1 and 0. None is returned if some error has occurred.

    An amino acid residue is delimited by heteroatoms.
    """

    amide_bonds, amide_bond_c_idxs, amide_bond_n_idxs = get_amide_bonds(mol)
    add_output = {"amide_bonds": [(c, n) for c, n in zip(amide_bond_c_idxs, amide_bond_n_idxs)]}
    if len(amide_bonds) == 0:
        return 0, add_output
    carboxy_c_idxs = get_carboxy_derivatives(mol)
    add_output["carboxy_cs"] = carboxy_c_idxs
    amino_group_idxs = get_amino_groups(mol, amide_bond_c_idxs)
    add_output["aminos"] = amino_group_idxs

    # get carbon sceleton minus amide bonds
    carbon_graph = nx.Graph()
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6 and bond not in amide_bonds:
            carbon_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    chunks = [list(comp) for comp in nx.connected_components(carbon_graph)]
    add_output["chunks"] = chunks
    # for amino groups, it might be unclear to which chunk they belong -> try all options, e.g. for CHEBI:76162
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
    # iterate over possible assignments of amino groups to chunks
    longest_aa_chain = []
    longest_aa_chain_with_atoms = []
    for amino_assignment in product(*possible_amino_chunk_assignments):
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
                longest_aa_chain_with_atoms = [chunks[i] + [a for a, assign in zip(amino_group_idxs, amino_assignment) if assign == i]  for i in aa_chain]
    add_output["longest_aa_chain"] = longest_aa_chain_with_atoms
    return len(longest_aa_chain), add_output


def get_amide_bonds(mol):
    bonds = []
    c_idxs = []
    n_idxs = []
    single_os = [atom.GetIdx() for atom in mol.GetAtoms()
                   if any(o_atom.GetAtomicNum() == 8
                          and mol.GetBondBetweenAtoms(atom.GetIdx(), o_atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                          for o_atom in atom.GetNeighbors())]
    double_os = [atom.GetIdx() for atom in mol.GetAtoms()
                   if any(o_atom.GetAtomicNum() == 8
                          and mol.GetBondBetweenAtoms(atom.GetIdx(), o_atom.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                          for o_atom in atom.GetNeighbors())]
    for bond in mol.GetBonds():
        for n_atom, c_atom in [[bond.GetBeginAtom(), bond.GetEndAtom()], [bond.GetEndAtom(), bond.GetBeginAtom()]]:
            if n_atom.GetAtomicNum() == 7 and c_atom.GetAtomicNum() == 6:
                # nitrogen atom is not allowed to have bonds to heteroatoms
                # nitrogen is only allowed to have single bonds (except for imidic form of amide)
                if not all(neighbor.GetAtomicNum() in [1, 6]
                   and (mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                        or (neighbor.GetIdx() in single_os
                            and mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_atom.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) )
                   for neighbor in n_atom.GetNeighbors()):
                    break
                # amide vs imidic form: accept NC=O or N=CO
                if mol.GetBondBetweenAtoms(c_atom.GetIdx(), n_atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                    look_for_bond = double_os
                elif mol.GetBondBetweenAtoms(c_atom.GetIdx(), n_atom.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                    look_for_bond = single_os
                else:
                    look_for_bond = []
                if c_atom.GetIdx() in look_for_bond:
                    bonds.append(bond)
                    c_idxs.append(c_atom.GetIdx())
                    n_idxs.append(n_atom.GetIdx())
    return bonds, c_idxs, n_idxs

def get_carboxy_derivatives(mol):
    # use C(=X)Y structure
    carboxy_cs = []
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 6
            # neighbor.GetAtomicNum() == 8 -> strict C(=O)Y, neighbor.GetAtomicNum() not in [1,6] -> advanced C(=X)Y
            and any(mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                    and neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors())
            and any(mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                    and neighbor.GetAtomicNum() not in [1, 6] for neighbor in atom.GetNeighbors())
        ):
            carboxy_cs.append(atom.GetIdx())
    return carboxy_cs

def get_amino_groups(mol, amide_bond_c_idxs):
    aminos = []
    for atom in mol.GetAtoms():
        # accept nitrogen, bonded to only hydrogen and carbon
        if (atom.GetAtomicNum() == 7
            and all(neighbor.GetAtomicNum() in [1, 6]
                # accept only single bonds, except for imidic form of amide
                and (mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                     or neighbor.GetIdx() in amide_bond_c_idxs) for neighbor in atom.GetNeighbors())):
            aminos.append(atom.GetIdx())
    return aminos


