from enum import auto
import enum
from rdkit import Chem

class ChargeCategories(enum.Enum):
    ANION = auto()
    CATION = auto()
    ZWITTERION = auto()
    SALT = auto()
    NEUTRAL = auto()
    UNKNOWN = auto()


def get_charge_category(mol):
    """
    Determine to which charge category a molecule belongs.

    - If there are at least two disconnected fragments (1 anion and 1 cation): salt
    Otherwise:
    - If net charge is negative / positive: anion / cation
    - If net charge is neutral, but there are two charges of opposite sign on
    non-adjacent atoms: zwitterion
    - Everything else: neutral

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule to determine the charge category of.

    Returns
    -------
    category : ChargeCategories
        The charge category of `mol`.
    """
    fragment_charges = []
    for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        if Chem.GetFormalCharge(fragment) < 0:
            fragment_charges.append(ChargeCategories.ANION)
        elif Chem.GetFormalCharge(fragment) > 0:
            fragment_charges.append(ChargeCategories.CATION)
        else:
            anions = [atom for atom in fragment.GetAtoms() if atom.GetFormalCharge() < 0 and all(neighbor.GetFormalCharge() == 0 for neighbor in atom.GetNeighbors())]
            cations = [atom for atom in fragment.GetAtoms() if atom.GetFormalCharge() > 0 and all(neighbor.GetFormalCharge() == 0 for neighbor in atom.GetNeighbors())]
            if len(anions) > 0 and len(cations) > 0:
                fragment_charges.append(ChargeCategories.ZWITTERION)

    if ChargeCategories.ANION in fragment_charges and ChargeCategories.CATION in fragment_charges:
        return ChargeCategories.SALT
    if ChargeCategories.ZWITTERION in fragment_charges and Chem.GetFormalCharge(mol) == 0:
        return ChargeCategories.ZWITTERION
    if ChargeCategories.ANION in fragment_charges:
        return ChargeCategories.ANION
    if ChargeCategories.CATION in fragment_charges:
        return ChargeCategories.CATION
    return ChargeCategories.NEUTRAL