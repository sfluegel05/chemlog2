from rdkit import Chem

def get_atom_pvar(atom_index: int, element_number: int):
    element = Chem.GetPeriodicTable().GetElementSymbol(element_number)
    element = element[0].upper() + element[1:]
    return f"{atom_index}_{element}"

def get_charge_pvar(atom_index: int, charge: int):
    charge_symbol = f"chargeM{-charge}" if charge < 0 else f"charge{charge}"
    return f"{atom_index}_{charge_symbol}"

def get_h_count_pvar(atom_index: int, h_count: int):
    return f"{atom_index}_has{h_count}Hs"

def get_bond_pvar(left_index: int, right_index: int):
    return f"{left_index}_{right_index}_has_bond_to"

def get_bond_type_pvar(left_index: int, right_index: int, bond_type: Chem.BondType):
    return f"{left_index}_{right_index}_b{str(bond_type).split('.')[-1]}"


def mol_to_propositional(mol: Chem.Mol):
    # turn molecule into propositional literals
    relevant_elements = [1, 6, 7, 8]
    relevant_charges = [-1]
    relevant_h_counts = [1]
    relevant_bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]

    positive_propositional_vars, negative_propositional_vars = [], []
    # only relevant atoms (h, c, n, o, other heteroatom)
    for i in relevant_elements:
        element = Chem.GetPeriodicTable().GetElementSymbol(i)
        element = element[0].upper() + element[1:]
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == i:
                positive_propositional_vars.append(get_atom_pvar(atom.GetIdx(), i))
            else:
                negative_propositional_vars.append(get_atom_pvar(atom.GetIdx(), i))

    # charges
    for charge in relevant_charges:
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() == charge:
                positive_propositional_vars.append(get_charge_pvar(atom.GetIdx(), charge))
            else:
                negative_propositional_vars.append(get_charge_pvar(atom.GetIdx(), charge))
    # h counts
    for h_count in relevant_h_counts:
        for atom in mol.GetAtoms():
            if atom.GetTotalNumHs() == h_count:
                positive_propositional_vars.append(get_h_count_pvar(atom.GetIdx(), h_count))
            else:
                negative_propositional_vars.append(get_h_count_pvar(atom.GetIdx(), h_count))

    # bonds
    for l_atom in mol.GetAtoms():
        for r_atom in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(l_atom.GetIdx(), r_atom.GetIdx())
            if bond is not None:
                bond_type = bond.GetBondType()
                positive_propositional_vars.append(get_bond_pvar(l_atom.GetIdx(), r_atom.GetIdx()))
                for target_type in relevant_bond_types:
                    if target_type == bond_type:
                        positive_propositional_vars.append(get_bond_type_pvar(l_atom.GetIdx(), r_atom.GetIdx(), target_type))
                    else:
                        negative_propositional_vars.append(get_bond_type_pvar(l_atom.GetIdx(), r_atom.GetIdx(), target_type))
            else:
                negative_propositional_vars.append(get_bond_pvar(l_atom.GetIdx(), r_atom.GetIdx()))
                for target_type in relevant_bond_types:
                    negative_propositional_vars.append(get_bond_type_pvar(l_atom.GetIdx(), r_atom.GetIdx(), target_type))

    return positive_propositional_vars, negative_propositional_vars