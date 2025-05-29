from rdkit import Chem

from chemlog.base_classifier import Classifier

EMERICELLAMIDE_SMARTS = ("[C@H,CH2]1[OX2H0][CX3H0](=[OX1H0])[C@HX4]([CX4H3])[NX3H][CX3H0](=[OX1H0])[C@HX4]([CX4H3])[NX3H]"
                         "[CX3H0](=[OX1H0])[C@HX4]([CX4H2][CX4H1]([CX4H3])[CX4H3])[NX3H][CX3H0](=[OX1H0])[C@@HX4]"
                         "([NX3H][CX3H0](=[OX1H0])[CX4H2][NX3H][CX3H0](=[OX1H0])[C@@H,CH2]1)[CX4H]([CX4H3])[CX4H3]")

DIKETOPIPERAZINE_SMARTS = "C1[NX3H][CX3H0](=[OX1H0])C[NX3H][CX3H0]1=[OX1H0]"


class AlgSubstructureClassifier(Classifier):

    def classify(self, mol: Chem.Mol, n_amino_acid_residues=None, *args, **kwargs):
        res, add = {}, {}
        if n_amino_acid_residues == 5:
            emericellamide = is_emericellamide(mol)
            res["emericellamide"] = emericellamide[0]
            if emericellamide[0] and emericellamide[1]:
                add["emericellamide_atoms"] = emericellamide[1]
        if n_amino_acid_residues == 2:
            diketopiperazine = is_diketopiperazine(mol)
            res["2,5-diketopiperazines"] = diketopiperazine[0]
            if diketopiperazine[0] and diketopiperazine[1]:
                add["2,5-diketopiperazines_atoms"] = diketopiperazine[1]

        return res, add

def apply_smarts(mol, smarts) -> (bool, list):
    has_match = mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))
    if has_match:
        return True, mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    else:
        return False, []

def is_emericellamide(mol) -> (bool, list):
    return apply_smarts(mol, EMERICELLAMIDE_SMARTS)

def is_diketopiperazine(mol) -> (bool, list):
    return apply_smarts(mol, DIKETOPIPERAZINE_SMARTS)
