import logging

from rdkit import Chem

from chemlog2.classification.peptide_size_classifier import get_carboxy_derivatives, get_amino_groups
from chemlog2.results import plot_mol


def get_proteinogenic_amino_acids(mol: Chem.Mol, amino_ns, carboxys):
    Chem.rdCIPLabeler.AssignCIPLabels(mol)
    carboxy_cs = [c for c, _, _ in carboxys]
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
                        if len(side_chain_start) == 1:
                            start_at = side_chain_start[0].GetIdx()
                            amino_acid = match_side_chain_to_backbone(mol,
                                side_chains, [n, alpha_c.GetIdx(), carboxy_c.GetIdx(), start_at])
                            if amino_acid is not None:
                                # distinguish R- and L-amino acids, only one side chain per amino acid is allowed
                                # L-cysteine has a different CIP code
                                expected_cip_code = "R" if amino_acid[0] == "C" else "S"
                                if alpha_c.HasProp("_CIPCode") and alpha_c.GetProp("_CIPCode") == expected_cip_code:
                                    results.append(amino_acid[0])
                                    results_atoms.append(aa + amino_acid[1])
    return results, results_atoms

def match_side_chain_to_backbone(mol, side_chains, backbone):
    # assume that backbone has 4 atoms, NC(C)R where the second C belongs to a carboxylic acid derivative
    for side_chain, matches in side_chains.items():
        side_chain = side_chain.split("_")[0] # _iso are isomers of the same amino acid
        matches_starting_at = [match for match in matches if backbone[-1] == match[0] and backbone[1] not in match]
        if len(matches_starting_at) == 1:
            # proline: close ring to amino group
            if side_chain == "P" and matches_starting_at[0][-1] != backbone[0]:
                continue
            # methionine: be aware of N-formyl-methionine
            elif side_chain == "M":
                formyl = mol.GetSubstructMatches(Chem.MolFromSmarts("N[CH1X3]=[OX1]"))
                formyl = [f for f in formyl if f[0] == backbone[0]]
                if len(formyl) == 1:
                    return "fMet", list(matches_starting_at[0]) + list(formyl[0])[1:]
            logging.debug(f"{backbone} matches side chain {side_chain} with {matches_starting_at}")
            return side_chain, list(matches_starting_at[0])

        elif len(matches_starting_at) > 1:
            logging.warning(f"Found {len(matches_starting_at)} matches (expected max. 1) for "
                            f"side chain {side_chain} at {backbone}: {matches_starting_at}")


side_chains = {
    # order is important - these will match e.g. both valine and alanine for a valine residue
    # -> first, try valine SMARTS and don't get to alanine
    # amino and carboxylic acid groups are underspecified -> we already have those, including their derivatives
    # in peptides -> the lists include the indices in the matches where amino / carboxylic acid residues are expected
    # assume that input is kekulized, no aromaticity
    "W": "[CH2X4][CX3]1=[CX3H][NX3][CX3]2=[CX3]1[CX3]=[CX3H][CX3H]=[CX3]2",  # tryptophan
    "Y": "[CH2X4][CX3]1=[CX3H][CX3H]=[CX3]([OX2H])[CX3H]=[CX3H]1",  # tyrosine
    "F": "[CH2X4][CX3]1=[CX3H][CX3H]=[CX3H][CX3H]=[CX3H]1",  # phenylalanine
    "H": "[CH2X4][CX3]1=[CX3H][NX3][CX3H]=[NX2]1",  # histidine
    "H_iso": "[CH2X4][CX3]1=[CX3H][NX2]=[CX3H][NX3]1",  # histidine isomer
    "R": ("[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NX3+,NX2])N", [3, 6], []),  # arginine
    "R_iso": ("[CH2X4][CH2X4][CH2X4][NX2]=[CH0X3](N)N", [5, 6], []),  # arginine isomer
    "O": ("[CH2X4][CH2X4][CH2X4][CH2X4]N[CX3][CH1X4]1[NX2]=[CH1X3][CH2X4][CH1X4]([CH3X4])1", [4], [5]), # pyrrolysine
    "K": ("[CH2X4][CH2X4][CH2X4][CH2X4]N", [4], []),  # lysine
    "M": "[CH2X4][CH2X4][SX2][CH3X4]",  # methionine
    "Q": ("[CH2X4][CH2X4][CX3](=[OX1])[NH2X3]", [4], []),  # glutamine ! last n has to have two H bonds
    # (otherwise, it would be glutamic acid with an amide bond)
    "E": ("[CH2X4][CH2X4][CX3]", [], [2]),  # glutamic acid
    "P": ("[CH2X4][CH2X4][CH2X4]N", [3], []),  # proline (without ring closed)
    "N": ("[CH2X4][CX3](=[OX1])[NH2X3]", [3], []),  # asparagine ! last n has to have two H bonds
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


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("NCC(=O)N[C@@H](Cc1ccccc1)C(O)=O")
    print(get_proteinogenic_amino_acids(mol, get_amino_groups(mol, []), get_carboxy_derivatives(mol)))
    plot_mol(mol)