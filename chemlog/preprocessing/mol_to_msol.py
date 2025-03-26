from rdkit import Chem

def mol_to_msol(mol: Chem.Mol):
    lines = []

    # domain: atoms
    lines.append(f"var2 atom")
    lines.append(
        f"atom = {{{','.join([str(i) for i, _ in enumerate(mol.GetAtoms())])}}}"
    )
    # elements
    lines.append(f"PLACEHOLDER")  # -> this gets replaced by a variable declaration
    element_symbols = []
    for i in range(118):
        element = Chem.GetPeriodicTable().GetElementSymbol(i + 1)
        element = element[0].upper() + element[1:]
        element_symbols.append(element)
        lines.append(
            f"{element} = {{{','.join([str(idx) for idx, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() == i + 1])}}}"
        )
    lines[2] = f"var2 {','.join(element_symbols)}"
    # charges
    lines.append(f"var2 Charge0, ChargeP, ChargeN")
    lines.append(
        f"Charge0 = {{{','.join([str(i) for i, atom in enumerate(mol.GetAtoms()) if atom.GetFormalCharge() == 0])}}}"
    )
    lines.append(
        f"ChargeN = {{{','.join([str(i) for i, atom in enumerate(mol.GetAtoms()) if atom.GetFormalCharge() < 0])}}}"
    )
    lines.append(
        f"ChargeP = {{{','.join([str(i) for i, atom in enumerate(mol.GetAtoms()) if atom.GetFormalCharge() > 0])}}}"
    )
    # h counts
    hydrogen_counts = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
    h_preds = {i: f"Has{i}Hs" for i in range(5)}
    lines.append(f"var2 {','.join(h_preds.values())}")
    for h_count in h_preds.keys():
        lines.append(
            f"{h_preds[h_count]} = {{{','.join([str(i) for i, atom_count in enumerate(hydrogen_counts) if atom_count == h_count])}}}"
        )

    # bonds (these are not a set of atom pairs (variable), but a predicate)
    bond_options = []
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 0:
            bond_options.append(
                f"(a in {{{atom.GetIdx()}}} & b in {{{','.join([str(nbr.GetIdx()) for nbr in atom.GetNeighbors()])}}})"
            )
    lines.append(
        f"pred has_bond_to(var1 a, var1 b) = {'|'.join(bond_options) if len(bond_options) > 0 else 'false'}"
    )

    for bond_type in Chem.BondType.values.values():
        bond_options = []
        for bond in mol.GetBonds():
            if (
                    bond.GetBondType() == bond_type
                    or bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
                    and bond.GetBondType() == Chem.BondType.AROMATIC
            ):
                bond_options.append(
                    f"(a in {{{bond.GetBeginAtomIdx()}}} & b in {{{bond.GetEndAtomIdx()}}})"
                )
                bond_options.append(
                    f"(b in {{{bond.GetBeginAtomIdx()}}} & a in {{{bond.GetEndAtomIdx()}}})"
                )
        lines.append(
            f"pred b{str(bond_type).split('.')[-1]}(var1 a, var1 b) = {'|'.join(bond_options) if len(bond_options) > 0 else 'false'}"
        )

    # net charge
    net_charge = Chem.GetFormalCharge(mol)
    lines.append(f"var0 NetCharge0,NetChargeN,NetChargeP")
    lines.append(
        f"{'~' if net_charge != 0 else ''}NetCharge0 & {'~' if net_charge >= 0 else ''}NetChargeN "
        f"& {'~' if net_charge <= 0 else ''}NetChargeP"
    )

    return len(mol.GetAtoms()), ";\n".join(lines) + ";"