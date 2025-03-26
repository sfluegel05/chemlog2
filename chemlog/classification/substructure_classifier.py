from rdkit import Chem

EMERICELLAMIDE_SMARTS = ("[C@H,CH2]1[OX2H0][CX3H0](=[OX1H0])[C@HX4]([CX4H3])[NX3H][CX3H0](=[OX1H0])[C@HX4]([CX4H3])[NX3H]"
                         "[CX3H0](=[OX1H0])[C@HX4]([CX4H2][CX4H1]([CX4H3])[CX4H3])[NX3H][CX3H0](=[OX1H0])[C@@HX4]"
                         "([NX3H][CX3H0](=[OX1H0])[CX4H2][NX3H][CX3H0](=[OX1H0])[C@@H,CH2]1)[CX4H]([CX4H3])[CX4H3]")

DIKETOPIPERAZINE_SMARTS = "C1[NX3H][CX3H0](=[OX1H0])C[NX3H][CX3H0]1=[OX1H0]"

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
    mol = Chem.MolFromSmiles("O=C1NCC(=O)NC1C")
    print(is_emericellamide(mol))
    print(is_diketopiperazine(mol))