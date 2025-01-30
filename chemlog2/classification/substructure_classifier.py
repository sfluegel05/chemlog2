from rdkit import Chem

EMERICELLAMIDE_SMARTS = ("[C@H,CH2]1[OX2H0][CX3H0](=[OX1H0])[C@HX4]([CX4H3])[NX3H][CX3H0](=[OX1H0])[C@HX4]([CX4H3])[NX3H]"
                         "[CX3H0](=[OX1H0])[C@HX4]([CX4H2][CX4H1]([CX4H3])[CX4H3])[NX3H][CX3H0](=[OX1H0])[C@@HX4]"
                         "([NX3H][CX3H0](=[OX1H0])[CX4H2][NX3H][CX3H0](=[OX1H0])[C@@H,CH2]1)[CX4H]([CX4H3])[CX4H3]")

DIKETOPIPERAZINE_SMARTS = "[CX4H]1[NX3H][CX3H0](=[OX1H0])[CX4H][NX3H][CX3H0]1=[OX1H0]"

def apply_smarts(mol, smarts):
    has_match = mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))
    if has_match:
        return True, mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    else:
        return False, []

def is_emericellamide(mol):
    return apply_smarts(mol, EMERICELLAMIDE_SMARTS)

def is_diketopiperazine(mol):
    return apply_smarts(mol, DIKETOPIPERAZINE_SMARTS)

if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CC(C)C[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC1=O")
    print(is_emericellamide(mol))
    print(is_diketopiperazine(mol))