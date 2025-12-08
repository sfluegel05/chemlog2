from rdkit import Chem
# Generated Python code from Lopster rules - this will be overwritten when calling translate_lopster.run_lopster_translation()

def has_bond_to(mol: Chem.Mol, so_elements, X, Y):
    # (has_bond_to(X, Y)) <=> ((bsingle(X, Y) | bdouble(X, Y) | btriple(X, Y)))
    return ((mol.GetBondBetweenAtoms(X.GetIdx(), Y.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y.GetIdx()).GetBondType() == Chem.BondType.SINGLE) or (mol.GetBondBetweenAtoms(X.GetIdx(), Y.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) or (mol.GetBondBetweenAtoms(X.GetIdx(), Y.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y.GetIdx()).GetBondType() == Chem.BondType.TRIPLE))

def horc(mol: Chem.Mol, so_elements, X):
    # (horc(X)) <=> ((h(X) | c(X)))
    return (X.GetSymbol() == 'H' or X.GetSymbol() == 'C')

def charged(mol: Chem.Mol, so_elements, X):
    # (charged(X)) <=> ((charge3(X) | charge2(X) | charge1(X) | charge_m3(X) | charge_m2(X) | charge_m1(X)))
    return (X.GetFormalCharge() == 3 or X.GetFormalCharge() == 2 or X.GetFormalCharge() == 1 or X.GetFormalCharge() == -3 or X.GetFormalCharge() == -2 or X.GetFormalCharge() == -1)

def midOxygen(mol: Chem.Mol, so_elements, Y):
    # (midOxygen(Y)) <=> (∃[X, Y1, Y2]: (o(Y) & has_bond_to(Y, Y1) & has_bond_to(Y1, Y) & has_bond_to(Y, Y2) & has_bond_to(Y2, Y) & (Y1) != (Y2)))
    return any((Y.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(Y.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y.GetIdx()) is not None and (Y1 != Y2)) for X in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def has_bond_toAtLeast1(mol: Chem.Mol, so_elements, X):
    # (has_bond_toAtLeast1(X)) <=> (∃[Y1]: (has_bond_to(X, Y1) & has_bond_to(Y1, X)))
    return any((mol.GetBondBetweenAtoms(X.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), X.GetIdx()) is not None) for Y1 in mol.GetAtoms())

def has_bond_toAtLeast2(mol: Chem.Mol, so_elements, X):
    # (has_bond_toAtLeast2(X)) <=> (∃[Y1, Y2]: (has_bond_to(X, Y1) & has_bond_to(Y1, X) & has_bond_to(X, Y2) & has_bond_to(Y2, X) & (Y1) != (Y2)))
    return any((mol.GetBondBetweenAtoms(X.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), X.GetIdx()) is not None and (Y1 != Y2)) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def has_bond_toAtLeast3(mol: Chem.Mol, so_elements, X):
    # (has_bond_toAtLeast3(X)) <=> (∃[Y3, Y1, Y2]: (has_bond_to(X, Y1) & has_bond_to(Y1, X) & has_bond_to(X, Y2) & has_bond_to(Y2, X) & has_bond_to(X, Y3) & has_bond_to(Y3, X) & (Y1) != (Y2) & (Y1) != (Y3) & (Y2) != (Y3)))
    return any((mol.GetBondBetweenAtoms(X.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), X.GetIdx()) is not None and (Y1 != Y2) and (Y1 != Y3) and (Y2 != Y3)) for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def has_bond_toAtLeast4(mol: Chem.Mol, so_elements, X):
    # (has_bond_toAtLeast4(X)) <=> (∃[Y1, Y3, Y4, Y2]: (has_bond_to(X, Y1) & has_bond_to(Y1, X) & has_bond_to(X, Y2) & has_bond_to(Y2, X) & has_bond_to(X, Y3) & has_bond_to(Y3, X) & has_bond_to(X, Y4) & has_bond_to(Y4, X) & (Y1) != (Y2) & (Y1) != (Y3) & (Y1) != (Y4) & (Y2) != (Y3) & (Y2) != (Y4) & (Y3) != (Y4)))
    return any((mol.GetBondBetweenAtoms(X.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), X.GetIdx()) is not None and (Y1 != Y2) and (Y1 != Y3) and (Y1 != Y4) and (Y2 != Y3) and (Y2 != Y4) and (Y3 != Y4)) for Y1 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y4 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def has_bond_toExactly2(mol: Chem.Mol, so_elements, X):
    # (has_bond_toExactly2(X)) <=> ((has_bond_toAtLeast2(X) & ~(has_bond_toAtLeast3(X))))
    return (has_bond_toAtLeast2(mol, so_elements, X) and not(has_bond_toAtLeast3(mol, so_elements, X)))

def has_bond_toExactly3(mol: Chem.Mol, so_elements, X):
    # (has_bond_toExactly3(X)) <=> ((has_bond_toAtLeast3(X) & ~(has_bond_toAtLeast4(X))))
    return (has_bond_toAtLeast3(mol, so_elements, X) and not(has_bond_toAtLeast4(mol, so_elements, X)))

def has_bond_to1to3(mol: Chem.Mol, so_elements, X):
    # (has_bond_to1to3(X)) <=> ((has_bond_toAtLeast1(X) & ~(has_bond_toAtLeast4(X))))
    return (has_bond_toAtLeast1(mol, so_elements, X) and not(has_bond_toAtLeast4(mol, so_elements, X)))

def acylCarbon(mol: Chem.Mol, so_elements, Y):
    # (acylCarbon(Y)) <=> (∃[X, Z]: (c(Y) & o(Z) & bdouble(Y, Z) & bdouble(Z, Y)))
    return any((Y.GetSymbol() == 'C' and Z.GetSymbol() == 'O' and (mol.GetBondBetweenAtoms(Y.GetIdx(), Z.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y.GetIdx(), Z.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Z.GetIdx(), Y.GetIdx()) is not None and mol.GetBondBetweenAtoms(Z.GetIdx(), Y.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)) for X in mol.GetAtoms() for Z in mol.GetAtoms())

def noble(mol: Chem.Mol, so_elements, X):
    # (noble(X)) <=> ((ar(X) | he(X) | kr(X) | ne(X) | rn(X) | xe(X)))
    return (X.GetSymbol() == 'Ar' or X.GetSymbol() == 'He' or X.GetSymbol() == 'Kr' or X.GetSymbol() == 'Ne' or X.GetSymbol() == 'Rn' or X.GetSymbol() == 'Xe')

def cleanReachable(mol: Chem.Mol, so_elements, X, Z, W):
    # (cleanReachable(X, Z, W)) <=> (∃[Y]: (cleanReachable(X, Y, Z) & has_bond_to(Z, W) & has_bond_to(W, Z) & (X) != (W) & (Y) != (W) & (W) != (Z)))
    return any((cleanReachable(mol, so_elements, X, Y, Z) and mol.GetBondBetweenAtoms(Z.GetIdx(), W.GetIdx()) is not None and mol.GetBondBetweenAtoms(W.GetIdx(), Z.GetIdx()) is not None and (X != W) and (Y != W) and (W != Z)) for Y in mol.GetAtoms())

def closedLoopAtLeast3(mol: Chem.Mol, so_elements, X):
    # (closedLoopAtLeast3(X)) <=> (∃[Z, Y]: (cleanReachable(X, Y, Z) & has_bond_to(Z, X) & has_bond_to(X, Z)))
    return any((cleanReachable(mol, so_elements, X, Y, Z) and mol.GetBondBetweenAtoms(Z.GetIdx(), X.GetIdx()) is not None and mol.GetBondBetweenAtoms(X.GetIdx(), Z.GetIdx()) is not None) for Z in mol.GetAtoms() for Y in mol.GetAtoms())

def halogen(mol: Chem.Mol, so_elements, X):
    # (halogen(X)) <=> ((f(X) | cl(X) | br(X) | i(X) | at(X)))
    return (X.GetSymbol() == 'F' or X.GetSymbol() == 'Cl' or X.GetSymbol() == 'Br' or X.GetSymbol() == 'I' or X.GetSymbol() == 'At')


def astatineMolEntity(mol: Chem.Mol, so_elements):
    # (astatineMolEntity()) <=> (∃[Y]: (at(Y)))
    return any(Y.GetSymbol() == 'At' for Y in mol.GetAtoms())

def bromineMolEntity(mol: Chem.Mol, so_elements):
    # (bromineMolEntity()) <=> (∃[Y]: (br(Y)))
    return any(Y.GetSymbol() == 'Br' for Y in mol.GetAtoms())

def chlorineMolEntity(mol: Chem.Mol, so_elements):
    # (chlorineMolEntity()) <=> (∃[Y]: (cl(Y)))
    return any(Y.GetSymbol() == 'Cl' for Y in mol.GetAtoms())

def fluorineMolEntity(mol: Chem.Mol, so_elements):
    # (fluorineMolEntity()) <=> (∃[Y]: (f(Y)))
    return any(Y.GetSymbol() == 'F' for Y in mol.GetAtoms())

def iodineMolEntity(mol: Chem.Mol, so_elements):
    # (iodineMolEntity()) <=> (∃[Y]: (i(Y)))
    return any(Y.GetSymbol() == 'I' for Y in mol.GetAtoms())

def halogenMolEntity(mol: Chem.Mol, so_elements):
    # (halogenMolEntity()) <=> ((bromineMolEntity() | chlorineMolEntity() | fluorineMolEntity() | astatineMolEntity() | iodineMolEntity()))
    return (bromineMolEntity(mol, so_elements, ) or chlorineMolEntity(mol, so_elements, ) or fluorineMolEntity(mol, so_elements, ) or astatineMolEntity(mol, so_elements, ) or iodineMolEntity(mol, so_elements, ))

def poloniumMolEntity(mol: Chem.Mol, so_elements):
    # (poloniumMolEntity()) <=> (∃[Y]: (po(Y)))
    return any(Y.GetSymbol() == 'Po' for Y in mol.GetAtoms())

def seleniumMolEntity(mol: Chem.Mol, so_elements):
    # (seleniumMolEntity()) <=> (∃[Y]: (se(Y)))
    return any(Y.GetSymbol() == 'Se' for Y in mol.GetAtoms())

def sulfurMolEntity(mol: Chem.Mol, so_elements):
    # (sulfurMolEntity()) <=> (∃[Y]: (s(Y)))
    return any(Y.GetSymbol() == 'S' for Y in mol.GetAtoms())

def telluriumMolEntity(mol: Chem.Mol, so_elements):
    # (telluriumMolEntity()) <=> (∃[Y]: (te(Y)))
    return any(Y.GetSymbol() == 'Te' for Y in mol.GetAtoms())

def chalcogenMolEntity(mol: Chem.Mol, so_elements):
    # (chalcogenMolEntity()) <=> ((oxygenMolEntity() | poloniumMolEntity() | seleniumMolEntity() | sulfurMolEntity() | telluriumMolEntity()))
    return (oxygenMolEntity(mol, so_elements, ) or poloniumMolEntity(mol, so_elements, ) or seleniumMolEntity(mol, so_elements, ) or sulfurMolEntity(mol, so_elements, ) or telluriumMolEntity(mol, so_elements, ))

def antimonyMolEntity(mol: Chem.Mol, so_elements):
    # (antimonyMolEntity()) <=> (∃[Y]: (sb(Y)))
    return any(Y.GetSymbol() == 'Sb' for Y in mol.GetAtoms())

def arsenicMolEntity(mol: Chem.Mol, so_elements):
    # (arsenicMolEntity()) <=> (∃[Y]: (as(Y)))
    return any(Y.GetSymbol() == 'As' for Y in mol.GetAtoms())

def bismuthMolEntity(mol: Chem.Mol, so_elements):
    # (bismuthMolEntity()) <=> (∃[Y]: (bi(Y)))
    return any(Y.GetSymbol() == 'Bi' for Y in mol.GetAtoms())

def nitrogenMolEntity(mol: Chem.Mol, so_elements):
    # (nitrogenMolEntity()) <=> (∃[Y]: (ni(Y)))
    return any(Y.GetSymbol() == 'Ni' for Y in mol.GetAtoms())

def pnictogenMolEntity(mol: Chem.Mol, so_elements):
    # (pnictogenMolEntity()) <=> ((antimonyMolEntity() | arsenicMolEntity() | bismuthMolEntity() | nitrogenMolEntity() | phosphorusMolEntity()))
    return (antimonyMolEntity(mol, so_elements, ) or arsenicMolEntity(mol, so_elements, ) or bismuthMolEntity(mol, so_elements, ) or nitrogenMolEntity(mol, so_elements, ) or phosphorusMolEntity(mol, so_elements, ))

def chromiumMolEntity(mol: Chem.Mol, so_elements):
    # (chromiumMolEntity()) <=> (∃[Y]: (cr(Y)))
    return any(Y.GetSymbol() == 'Cr' for Y in mol.GetAtoms())

def molybdenumMolEntity(mol: Chem.Mol, so_elements):
    # (molybdenumMolEntity()) <=> (∃[Y]: (mo(Y)))
    return any(Y.GetSymbol() == 'Mo' for Y in mol.GetAtoms())

def seaborgiumMolEntity(mol: Chem.Mol, so_elements):
    # (seaborgiumMolEntity()) <=> (∃[Y]: (sg(Y)))
    return any(Y.GetSymbol() == 'Sg' for Y in mol.GetAtoms())

def tungstenMolEntity(mol: Chem.Mol, so_elements):
    # (tungstenMolEntity()) <=> (∃[Y]: (w(Y)))
    return any(Y.GetSymbol() == 'W' for Y in mol.GetAtoms())

def chromiumGroupMolEntity(mol: Chem.Mol, so_elements):
    # (chromiumGroupMolEntity()) <=> ((chromiumMolEntity() | molybdenumMolEntity() | seaborgiumMolEntity() | tungstenMolEntity()))
    return (chromiumMolEntity(mol, so_elements, ) or molybdenumMolEntity(mol, so_elements, ) or seaborgiumMolEntity(mol, so_elements, ) or tungstenMolEntity(mol, so_elements, ))

def nobleGasMolEntity(mol: Chem.Mol, so_elements):
    # (nobleGasMolEntity()) <=> (∃[Y]: (noble(Y)))
    return any(noble(mol, so_elements, Y) for Y in mol.GetAtoms())

def carbonMolEntity(mol: Chem.Mol, so_elements):
    # (carbonMolEntity()) <=> (∃[Y]: (c(Y)))
    return any(Y.GetSymbol() == 'C' for Y in mol.GetAtoms())

def oxygenMolEntity(mol: Chem.Mol, so_elements):
    # (oxygenMolEntity()) <=> (∃[Y]: (o(Y)))
    return any(Y.GetSymbol() == 'O' for Y in mol.GetAtoms())

def hydrogenMolEntity(mol: Chem.Mol, so_elements):
    # (hydrogenMolEntity()) <=> (∃[Y]: (h(Y)))
    return any(Y.GetSymbol() == 'H' for Y in mol.GetAtoms())

def phosphorusMolEntity(mol: Chem.Mol, so_elements):
    # (phosphorusMolEntity()) <=> (∃[Y]: (p(Y)))
    return any(Y.GetSymbol() == 'P' for Y in mol.GetAtoms())

def inorganic(mol: Chem.Mol, so_elements):
    # (inorganic()) <=> ((~(carbonMolEntity())))
    return not(carbonMolEntity(mol, so_elements, ))

def notHydroCarbon(mol: Chem.Mol, so_elements):
    # (notHydroCarbon()) <=> (∃[Y]: (~(c(Y)) & ~(h(Y))))
    return any((not(Y.GetSymbol() == 'C') and not(Y.GetSymbol() == 'H')) for Y in mol.GetAtoms())

def hydroCarbon(mol: Chem.Mol, so_elements):
    # (hydroCarbon()) <=> ((carbonMolEntity() & ~(notHydroCarbon())))
    return (carbonMolEntity(mol, so_elements, ) and not(notHydroCarbon(mol, so_elements, )))

def notHaloHydroCarbon(mol: Chem.Mol, so_elements):
    # (notHaloHydroCarbon()) <=> (∃[Y]: (~(c(Y)) & ~(h(Y)) & ~(halogen(Y))))
    return any((not(Y.GetSymbol() == 'C') and not(Y.GetSymbol() == 'H') and not(halogen(mol, so_elements, Y))) for Y in mol.GetAtoms())

def haloHydroCarbon(mol: Chem.Mol, so_elements):
    # (haloHydroCarbon()) <=> ((carbonMolEntity() & halogenMolEntity() & ~(notHaloHydroCarbon())))
    return (carbonMolEntity(mol, so_elements, ) and halogenMolEntity(mol, so_elements, ) and not(notHaloHydroCarbon(mol, so_elements, )))

def atLeast2Carbons(mol: Chem.Mol, so_elements):
    # (atLeast2Carbons()) <=> (∃[Y1, Y2]: (c(Y1) & c(Y2) & (Y1) != (Y2)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (Y1 != Y2)) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def exactly1Carbon(mol: Chem.Mol, so_elements):
    # (exactly1Carbon()) <=> ((carbonMolEntity() & ~(atLeast2Carbons())))
    return (carbonMolEntity(mol, so_elements, ) and not(atLeast2Carbons(mol, so_elements, )))

def atLeast3Carbons(mol: Chem.Mol, so_elements):
    # (atLeast3Carbons()) <=> (∃[Y3, Y1, Y2]: (c(Y1) & c(Y2) & (Y1) != (Y2) & c(Y3) & (Y1) != (Y3) & (Y2) != (Y3)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (Y1 != Y2) and Y3.GetSymbol() == 'C' and (Y1 != Y3) and (Y2 != Y3)) for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def exactly2Carbons(mol: Chem.Mol, so_elements):
    # (exactly2Carbons()) <=> ((atLeast2Carbons() & ~(atLeast3Carbons())))
    return (atLeast2Carbons(mol, so_elements, ) and not(atLeast3Carbons(mol, so_elements, )))

def atLeast4Carbons(mol: Chem.Mol, so_elements):
    # (atLeast4Carbons()) <=> (∃[Y4, Y3, Y1, Y2]: (c(Y1) & c(Y2) & (Y1) != (Y2) & c(Y3) & (Y1) != (Y3) & (Y2) != (Y3) & c(Y4) & (Y1) != (Y4) & (Y2) != (Y4) & (Y3) != (Y4)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (Y1 != Y2) and Y3.GetSymbol() == 'C' and (Y1 != Y3) and (Y2 != Y3) and Y4.GetSymbol() == 'C' and (Y1 != Y4) and (Y2 != Y4) and (Y3 != Y4)) for Y4 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def exactly3Carbons(mol: Chem.Mol, so_elements):
    # (exactly3Carbons()) <=> ((atLeast3Carbons() & ~(atLeast4Carbons())))
    return (atLeast3Carbons(mol, so_elements, ) and not(atLeast4Carbons(mol, so_elements, )))

def atLeast5Carbons(mol: Chem.Mol, so_elements):
    # (atLeast5Carbons()) <=> (∃[Y1, Y3, Y5, Y4, Y2]: (c(Y1) & c(Y2) & (Y1) != (Y2) & c(Y3) & (Y1) != (Y3) & (Y2) != (Y3) & c(Y4) & (Y1) != (Y4) & (Y2) != (Y4) & (Y3) != (Y4) & c(Y5) & (Y1) != (Y5) & (Y2) != (Y5) & (Y3) != (Y5) & (Y4) != (Y5)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (Y1 != Y2) and Y3.GetSymbol() == 'C' and (Y1 != Y3) and (Y2 != Y3) and Y4.GetSymbol() == 'C' and (Y1 != Y4) and (Y2 != Y4) and (Y3 != Y4) and Y5.GetSymbol() == 'C' and (Y1 != Y5) and (Y2 != Y5) and (Y3 != Y5) and (Y4 != Y5)) for Y1 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y5 in mol.GetAtoms() for Y4 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def exactly4Carbons(mol: Chem.Mol, so_elements):
    # (exactly4Carbons()) <=> ((atLeast4Carbons() & ~(atLeast5Carbons())))
    return (atLeast4Carbons(mol, so_elements, ) and not(atLeast5Carbons(mol, so_elements, )))

def polyatomic(mol: Chem.Mol, so_elements):
    # (polyatomic()) <=> (∃[Y1, Y2]: ((Y1) != (Y2)))
    return any((Y1 != Y2) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def monoatomic(mol: Chem.Mol, so_elements):
    # (monoatomic()) <=> (∃[Y1]: (~(polyatomic())))
    return any(not(polyatomic(mol, so_elements, )) for Y1 in mol.GetAtoms())

def carboxylicAcid(mol: Chem.Mol, so_elements):
    # (carboxylicAcid()) <=> (∃[Y4, Y3, Y1, Y2]: (c(Y1) & o(Y2) & o(Y3) & horc(Y4) & bdouble(Y1, Y2) & bdouble(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1) & bsingle(Y1, Y4) & bsingle(Y4, Y1) & ~(midOxygen(Y3)) & ~(charged(Y3))))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'O' and Y3.GetSymbol() == 'O' and horc(mol, so_elements, Y4) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(midOxygen(mol, so_elements, Y3)) and not(charged(mol, so_elements, Y3))) for Y4 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def atLeast2CarboxyGroups(mol: Chem.Mol, so_elements):
    # (atLeast2CarboxyGroups()) <=> (∃[Y1, Y3, Y8, Y5, Y4, Y7, Y6, Y2]: (c(Y1) & o(Y2) & o(Y3) & horc(Y4) & bdouble(Y1, Y2) & bdouble(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1) & bsingle(Y1, Y4) & bsingle(Y4, Y1) & ~(midOxygen(Y3)) & ~(charged(Y3)) & c(Y5) & o(Y6) & o(Y7) & horc(Y8) & bdouble(Y5, Y6) & bdouble(Y6, Y5) & bsingle(Y5, Y7) & bsingle(Y7, Y5) & bsingle(Y5, Y8) & bsingle(Y8, Y5) & ~(midOxygen(Y7)) & ~(charged(Y7)) & (Y1) != (Y5)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'O' and Y3.GetSymbol() == 'O' and horc(mol, so_elements, Y4) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(midOxygen(mol, so_elements, Y3)) and not(charged(mol, so_elements, Y3)) and Y5.GetSymbol() == 'C' and Y6.GetSymbol() == 'O' and Y7.GetSymbol() == 'O' and horc(mol, so_elements, Y8) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y6.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y6.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y6.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y6.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y7.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y7.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y7.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y7.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y8.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y8.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y8.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y8.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(midOxygen(mol, so_elements, Y7)) and not(charged(mol, so_elements, Y7)) and (Y1 != Y5)) for Y1 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y8 in mol.GetAtoms() for Y5 in mol.GetAtoms() for Y4 in mol.GetAtoms() for Y7 in mol.GetAtoms() for Y6 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def exactly1CarboxyGroup(mol: Chem.Mol, so_elements):
    # (exactly1CarboxyGroup()) <=> ((carboxylicAcid() & ~(atLeast2CarboxyGroups())))
    return (carboxylicAcid(mol, so_elements, ) and not(atLeast2CarboxyGroups(mol, so_elements, )))

def atLeast3CarboxyGroups(mol: Chem.Mol, so_elements):
    # (atLeast3CarboxyGroups()) <=> (∃[Y1, Y9, Y3, Y8, Y10, Y5, Y11, Y4, Y7, Y12, Y6, Y2]: (c(Y1) & o(Y2) & o(Y3) & horc(Y4) & bdouble(Y1, Y2) & bdouble(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1) & bsingle(Y1, Y4) & bsingle(Y4, Y1) & ~(midOxygen(Y3)) & ~(charged(Y3)) & c(Y5) & o(Y6) & o(Y7) & horc(Y8) & bdouble(Y5, Y6) & bdouble(Y6, Y5) & bsingle(Y5, Y7) & bsingle(Y7, Y5) & bsingle(Y5, Y8) & bsingle(Y8, Y5) & ~(midOxygen(Y7)) & ~(charged(Y7)) & c(Y9) & o(Y10) & o(Y11) & horc(Y12) & bdouble(Y9, Y10) & bdouble(Y10, Y9) & bsingle(Y9, Y11) & bsingle(Y11, Y9) & bsingle(Y9, Y12) & bsingle(Y12, Y9) & ~(midOxygen(Y11)) & ~(charged(Y11)) & (Y1) != (Y5) & (Y9) != (Y5) & (Y9) != (Y1)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'O' and Y3.GetSymbol() == 'O' and horc(mol, so_elements, Y4) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(midOxygen(mol, so_elements, Y3)) and not(charged(mol, so_elements, Y3)) and Y5.GetSymbol() == 'C' and Y6.GetSymbol() == 'O' and Y7.GetSymbol() == 'O' and horc(mol, so_elements, Y8) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y6.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y6.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y6.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y6.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y7.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y7.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y7.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y7.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y8.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y8.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y8.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y8.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(midOxygen(mol, so_elements, Y7)) and not(charged(mol, so_elements, Y7)) and Y9.GetSymbol() == 'C' and Y10.GetSymbol() == 'O' and Y11.GetSymbol() == 'O' and horc(mol, so_elements, Y12) and (mol.GetBondBetweenAtoms(Y9.GetIdx(), Y10.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y9.GetIdx(), Y10.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y10.GetIdx(), Y9.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y10.GetIdx(), Y9.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y9.GetIdx(), Y11.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y9.GetIdx(), Y11.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y11.GetIdx(), Y9.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y11.GetIdx(), Y9.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y9.GetIdx(), Y12.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y9.GetIdx(), Y12.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y12.GetIdx(), Y9.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y12.GetIdx(), Y9.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(midOxygen(mol, so_elements, Y11)) and not(charged(mol, so_elements, Y11)) and (Y1 != Y5) and (Y9 != Y5) and (Y9 != Y1)) for Y1 in mol.GetAtoms() for Y9 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y8 in mol.GetAtoms() for Y10 in mol.GetAtoms() for Y5 in mol.GetAtoms() for Y11 in mol.GetAtoms() for Y4 in mol.GetAtoms() for Y7 in mol.GetAtoms() for Y12 in mol.GetAtoms() for Y6 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def exactly2CarboxyGroups(mol: Chem.Mol, so_elements):
    # (exactly2CarboxyGroups()) <=> ((atLeast2CarboxyGroups() & ~(atLeast3CarboxyGroups())))
    return (atLeast2CarboxyGroups(mol, so_elements, ) and not(atLeast3CarboxyGroups(mol, so_elements, )))

def carboxylicEster(mol: Chem.Mol, so_elements):
    # (carboxylicEster()) <=> (∃[Y1, Y3, Y5, Y4, Y2]: (c(Y1) & o(Y2) & o(Y3) & c(Y4) & horc(Y5) & bdouble(Y1, Y2) & bdouble(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1) & bsingle(Y1, Y5) & bsingle(Y5, Y1) & bsingle(Y3, Y4) & bsingle(Y4, Y3)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'O' and Y3.GetSymbol() == 'O' and Y4.GetSymbol() == 'C' and horc(mol, so_elements, Y5) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE)) for Y1 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y5 in mol.GetAtoms() for Y4 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def hasBenzeneRing(mol: Chem.Mol, so_elements):
    # (hasBenzeneRing()) <=> (∃[Y1, Y3, Y5, Y4, Y6, Y2]: (c(Y1) & c(Y2) & bsingle(Y1, Y2) & bsingle(Y2, Y1) & c(Y3) & bdouble(Y2, Y3) & bdouble(Y3, Y2) & c(Y4) & bsingle(Y3, Y4) & bsingle(Y4, Y3) & bdouble(Y4, Y5) & bdouble(Y5, Y4) & c(Y5) & c(Y6) & bsingle(Y5, Y6) & bsingle(Y6, Y5) & bdouble(Y6, Y1) & bdouble(Y1, Y6)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and Y3.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and Y4.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and Y5.GetSymbol() == 'C' and Y6.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y5.GetIdx(), Y6.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y5.GetIdx(), Y6.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y6.GetIdx(), Y5.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y6.GetIdx(), Y5.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y6.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y6.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y6.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y6.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)) for Y1 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y5 in mol.GetAtoms() for Y4 in mol.GetAtoms() for Y6 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def hasFourMemberedRing(mol: Chem.Mol, so_elements):
    # (hasFourMemberedRing()) <=> (∃[Y4, Y3, Y1, Y2]: ((Y1) != (Y2) & has_bond_to(Y1, Y2) & has_bond_to(Y2, Y1) & (Y1) != (Y3) & (Y2) != (Y3) & has_bond_to(Y2, Y3) & has_bond_to(Y3, Y2) & (Y1) != (Y4) & (Y2) != (Y4) & (Y3) != (Y4) & has_bond_to(Y3, Y4) & has_bond_to(Y4, Y3) & has_bond_to(Y4, Y1) & has_bond_to(Y1, Y4)))
    return any(((Y1 != Y2) and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and (Y1 != Y3) and (Y2 != Y3) and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y2.GetIdx()) is not None and (Y1 != Y4) and (Y2 != Y4) and (Y3 != Y4) and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()) is not None) for Y4 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def amine(mol: Chem.Mol, so_elements):
    # (amine()) <=> (∃[Y4, Y3, Y1, Y2]: (n(Y1) & has_bond_to1to3(Y1) & horc(Y2) & horc(Y3) & c(Y4) & bsingle(Y1, Y2) & bsingle(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1) & bsingle(Y1, Y4) & bsingle(Y4, Y1) & ~(acylCarbon(Y2)) & ~(acylCarbon(Y3)) & ~(acylCarbon(Y4)) & (Y2) != (Y3) & (Y2) != (Y4) & (Y3) != (Y4)))
    return any((Y1.GetSymbol() == 'N' and has_bond_to1to3(mol, so_elements, Y1) and horc(mol, so_elements, Y2) and horc(mol, so_elements, Y3) and Y4.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and not(acylCarbon(mol, so_elements, Y2)) and not(acylCarbon(mol, so_elements, Y3)) and not(acylCarbon(mol, so_elements, Y4)) and (Y2 != Y3) and (Y2 != Y4) and (Y3 != Y4)) for Y4 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def aldehyde(mol: Chem.Mol, so_elements):
    # (aldehyde()) <=> (∃[Y3, Y1, Y2]: (c(Y1) & has_bond_toExactly2(Y1) & o(Y2) & horc(Y3) & bdouble(Y1, Y2) & bdouble(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1)))
    return any((Y1.GetSymbol() == 'C' and has_bond_toExactly2(mol, so_elements, Y1) and Y2.GetSymbol() == 'O' and horc(mol, so_elements, Y3) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE)) for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def cyclic(mol: Chem.Mol, so_elements):
    # (cyclic()) <=> (∃[Y]: (closedLoopAtLeast3(Y)))
    return any(closedLoopAtLeast3(mol, so_elements, Y) for Y in mol.GetAtoms())

def ketone(mol: Chem.Mol, so_elements):
    # (ketone()) <=> (∃[Y4, Y3, Y1, Y2]: (c(Y1) & o(Y2) & c(Y3) & c(Y4) & bdouble(Y1, Y2) & bdouble(Y2, Y1) & bsingle(Y1, Y3) & bsingle(Y3, Y1) & bsingle(Y1, Y4) & bsingle(Y4, Y1) & (Y3) != (Y4)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'O' and Y3.GetSymbol() == 'C' and Y4.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y3.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y3.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y4.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y4.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (Y3 != Y4)) for Y4 in mol.GetAtoms() for Y3 in mol.GetAtoms() for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def unsaturated(mol: Chem.Mol, so_elements):
    # (unsaturated()) <=> ((∃[Y1, Y2]: (c(Y1) & c(Y2) & bdouble(Y1, Y2) & bdouble(Y2, Y1)) | ∃[Y1, Y2]: (c(Y1) & c(Y2) & btriple(Y1, Y2) & btriple(Y2, Y1))))
    return (any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.DOUBLE)) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms()) or any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'C' and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.TRIPLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.TRIPLE)) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms()))

def saturated(mol: Chem.Mol, so_elements):
    # (saturated()) <=> (∃[Y1]: (c(Y1) & ~(unsaturated())))
    return any((Y1.GetSymbol() == 'C' and not(unsaturated(mol, so_elements, ))) for Y1 in mol.GetAtoms())

def organophosphorus(mol: Chem.Mol, so_elements):
    # (organophosphorus()) <=> (∃[Y1, Y2]: (c(Y1) & p(Y2) & bsingle(Y1, Y2) & bsingle(Y2, Y1)))
    return any((Y1.GetSymbol() == 'C' and Y2.GetSymbol() == 'P' and (mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()).GetBondType() == Chem.BondType.SINGLE) and (mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()).GetBondType() == Chem.BondType.SINGLE)) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())

def alkane(mol: Chem.Mol, so_elements):
    # (alkane()) <=> ((saturated() & hydroCarbon() & ~(cyclic())))
    return (saturated(mol, so_elements, ) and hydroCarbon(mol, so_elements, ) and not(cyclic(mol, so_elements, )))

def haloAlkane(mol: Chem.Mol, so_elements):
    # (haloAlkane()) <=> ((saturated() & haloHydroCarbon() & ~(cyclic())))
    return (saturated(mol, so_elements, ) and haloHydroCarbon(mol, so_elements, ) and not(cyclic(mol, so_elements, )))

def heteroOrganic(mol: Chem.Mol, so_elements):
    # (heteroOrganic()) <=> (∃[Y1, Y2]: (c(Y1) & ~(c(Y2)) & ~(h(Y2)) & has_bond_to(Y1, Y2) & has_bond_to(Y2, Y1)))
    return any((Y1.GetSymbol() == 'C' and not(Y2.GetSymbol() == 'C') and not(Y2.GetSymbol() == 'H') and mol.GetBondBetweenAtoms(Y1.GetIdx(), Y2.GetIdx()) is not None and mol.GetBondBetweenAtoms(Y2.GetIdx(), Y1.GetIdx()) is not None) for Y1 in mol.GetAtoms() for Y2 in mol.GetAtoms())
