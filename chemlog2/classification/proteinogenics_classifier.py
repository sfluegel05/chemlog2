import logging

from rdkit import Chem

from chemlog2.classification.peptide_size_classifier import get_carboxy_derivatives, get_amino_groups
from chemlog2.results import plot_mol


def get_proteinogenic_amino_acids(mol: Chem.Mol, amino_ns, carboxy_cs):
    Chem.rdCIPLabeler.AssignCIPLabels(mol)
    side_chains = identify_side_chains_smarts(mol, amino_ns, carboxy_cs)
    # identify proteinogenic amino acids
    # re-use knowledge about accepted amino group and carboxylic acid derivatives
    results = []
    results_atoms = []
    for n in amino_ns:
        for alpha_c in mol.GetAtomWithIdx(n).GetNeighbors():
            if alpha_c.GetAtomicNum() == 6:
                for carboxy_c in alpha_c.GetNeighbors():
                    if carboxy_c.GetIdx() in carboxy_cs:
                        aa = [n, alpha_c.GetIdx(), carboxy_c.GetIdx()]
                        aa += [neighbor.GetIdx() for neighbor in carboxy_c.GetNeighbors()
                               if neighbor.GetIdx() not in aa and neighbor.GetIdx() not in amino_ns]
                        side_chain_start = [neighbor for neighbor in alpha_c.GetNeighbors()
                                            if neighbor.GetIdx() not in aa and neighbor.GetAtomicNum() == 6]
                        if len(side_chain_start) == 0:
                            results.append("G")
                            results_atoms.append(aa)
                        # distinguish R- and L-amino acids, only one side chain per amino acid is allowed
                        if len(side_chain_start) == 1 and alpha_c.HasProp("_CIPCode") and alpha_c.GetProp("_CIPCode") == "S":
                            start_at = side_chain_start[0].GetIdx()
                            amino_acid = match_side_chain_to_backbone(mol,
                                side_chains, [n, alpha_c.GetIdx(), carboxy_c.GetIdx(), start_at])
                            if amino_acid is not None:
                                results.append(amino_acid[0])
                                results_atoms.append(amino_acid[1])
    return results, results_atoms

def match_side_chain_to_backbone(mol, side_chains, backbone):
    # assume that backbone has 4 atoms, NC(C)R where the second C belongs to a carboxylic acid derivative
    for side_chain, matches in side_chains.items():
        if side_chain == "R_iso":
            side_chain = "R"
        matches_starting_at = [match for match in matches if backbone[-1] == match[0]]
        if len(matches_starting_at) == 1:
            # proline: close ring to amino group
            if side_chain == "P" and matches_starting_at[0][-1] != backbone[0]:
                continue
            # methionine: be aware of N-formyl-methionine
            if side_chain == "M":
                formyl = mol.GetSubstructMatches(Chem.MolFromSmarts("N[CH1X3]=[OX1]"))
                formyl = [f for f in formyl if f[0] == backbone[0]]
                if len(formyl) == 1:
                    return "fMet", backbone[:2] + list(matches_starting_at[0]) + list(formyl[0])[1:]
            logging.debug(f"{backbone} matches side chain {side_chain} with {matches_starting_at}")
            return side_chain, backbone[:2] + list(matches_starting_at[0])

        elif len(matches_starting_at) > 1:
            logging.warning(f"Found {len(matches_starting_at)} matches (expected max. 1) for "
                            f"side chain {side_chain} at {backbone}: {matches_starting_at}")


side_chains = {
    # order is important - these will match e.g. both valine and alanine for a valine residue
    # -> first, try valine SMARTS and don't get to alanine
    # amino and carboxylic acid groups are underspecified -> we already have those, including their derivatives
    # in peptides -> the lists include the indices in the matches where amino / carboxylic acid residues are expected
    "W": "[CH2X4][cX3]1[cX3H][nX3][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",  # tryptophan
    "Y": "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OX2,OX1-])[cX3H][cX3H]1",  # tyrosine
    "F": "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",  # phenylalanine
    "H": "[CH2X4][cX3]1[cX3H][nX3][cX3H][nX2]1",  # histidine
    "R": ("[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NX3+,NX2])N", [3, 6], []),  # arginine
    "R_iso": ("[CH2X4][CH2X4][CH2X4][NX2]=[CH0X3](N)N", [5, 6], []),  # arginine isomer
    "O": ("[CH2X4][CH2X4][CH2X4][CH2X4]N[CX3][CH1X4]1[NX2]=[CH1X3][CH2X4][CH1X4]([CH3X4])1", [4], [5]), # pyrrolysine
    "K": ("[CH2X4][CH2X4][CH2X4][CH2X4]N", [4], []),  # lysine
    "M": "[CH2X4][CH2X4][SX2][CH3X4]",  # methionine
    "E": ("[CH2X4][CH2X4][CX3]", [], [2]),  # glutamic acid
    "Q": ("[CH2X4][CH2X4][CX3](=[OX1])N", [4], []),  # glutamine
    "P": ("[CH2X4][CH2X4][CH2X4]N", [3], []),  # proline (without ring closed)
    "N": ("[CH2X4][CX3](=[OX1])N", [3], []),  # asparagine
    "D": ("[CH2X4][CX3]", [], [1]),  # aspartic acid
    "I": "[CHX4]([CH3X4])[CH2X4][CH3X4]", # isoleucine
    "L": "[CH2X4][CHX4]([CH3X4])[CH3X4]", # leucine
    "C": "[CH2X4][SX2,SX1-]", # cysteine
    "U": "[CH2X4][SeX2,SeX1-]",  # Selenocysteine (formula taken from cysteine, replace S with Se)
    "T": "[CHX4]([CH3X4])[OX2]", # threonine
    "V": "[CHX4]([CH3X4])[CH3X4]", # valine
    "S": "[CH2X4][OX2]", # serine
    "A": "[CH3X4]",  # alanine
}
proline = "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]"

def identify_side_chains_smarts(mol, amino_ns, carboxy_cs):
    side_chain_matches = {}
    # block matched atoms, arg isomers, amino / carboxys
    for side_chain, specification in side_chains.items():
        smarts = specification[0] if isinstance(specification, tuple) else specification
        smarts_matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        # check if amino / carboylic acid groups are present
        if isinstance(specification, tuple):
            smarts_matches = [match for match in smarts_matches
                              if all(match[amino_location] in amino_ns for amino_location in specification[1])]
            smarts_matches = [match for match in smarts_matches
                              if all(match[carboxy_location] in carboxy_cs for carboxy_location in specification[2])]
        side_chain_matches[side_chain] = smarts_matches
    return side_chain_matches


def identify_side_chain(mol, aa):
    # outdated, incomplete
    side_chain = [neighbor for neighbor in mol.GetAtomWithIdx(aa[1]).GetNeighbors() if neighbor.GetIdx() not in aa]
    if len(side_chain) == 0:
        return "G", aa
    elif len(side_chain) == 1 and side_chain[0].GetAtomicNum() == 6:
        aa.append(side_chain[0].GetIdx())
        print(mol.GetBondBetweenAtoms(aa[0], aa[1]).GetStereo())
        next_atoms = [neighbor for neighbor in mol.GetAtomWithIdx(aa[-1]).GetNeighbors() if neighbor.GetIdx() not in aa]
        if len(next_atoms) == 0:
            return "A", aa
        elif len(next_atoms) == 1:
            aa.append(next_atoms[0].GetIdx())
            next_atoms = [neighbor for neighbor in mol.GetAtomWithIdx(aa[-1]).GetNeighbors() if
                          neighbor.GetIdx() not in aa]
            if len(next_atoms) == 0:
                # serine, cysteine, selenocysteine
                if mol.GetAtomWithIdx(aa[-1]).GetAtomicNum() == 8:
                    return "S", aa
                elif mol.GetAtomWithIdx(aa[-1]).GetAtomicNum() == 16:
                    return "C", aa
                elif mol.GetAtomWithIdx(aa[-1]).GetAtomicNum() == 34:
                    return "U", aa
            elif mol.GetAtomWithIdx(aa[-1]).GetAtomicNum() == 6:
                # order is important: e.g. Phenylalanine would also match tyrosine
                for pattern, aa_code in [("c1cnc2ccccc21", "W"), ("c1ccc([#8H]")]:
                    for match in mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)):
                        if aa[-1] in match:
                            return aa_code, list(set.union(set(aa), set(match)))
        elif len(next_atoms) == 2:
            # valine, isoleucine, threonine
            aa.append(next_atoms[0].GetIdx())
            aa.append(next_atoms[1].GetIdx())
            if all(neighbor.GetIdx() in aa for a in next_atoms for neighbor in a.GetNeighbors()):
                if (sum(a.GetAtomicNum() == 8 for a in next_atoms) == 1
                        and sum(a.GetAtomicNum() == 6 for a in next_atoms) == 1):
                    return "T", aa
                if sum(a.GetAtomicNum() == 6 for a in next_atoms) == 2:
                    return "V", aa
            else:
                next_next_atoms = [neighbor for a in next_atoms for neighbor in a.GetNeighbors()
                                   if neighbor.GetIdx() not in aa]
                if (len(next_next_atoms) == 1 and next_next_atoms[0].GetAtomicNum() == 6
                        and all(neighbor.GetIdx() in aa for neighbor in next_next_atoms[0].GetNeighbors())):
                    return "I", aa
        return None


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("C(=O)([C@@H](N)CC1=CNC2=C1C=CC=C2)N[C@H](C(=O)N[C@H](C(=O)O)C(C)C)CC3=CNC4=C3C=CC=C4")
    print(get_proteinogenic_amino_acids(mol, get_amino_groups(mol, []), get_carboxy_derivatives(mol)))
    plot_mol(mol)