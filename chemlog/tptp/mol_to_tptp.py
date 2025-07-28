from rdkit import Chem
from chemlog.msol import msol
from gavel.logic.problem import AnnotatedFormula, FormulaRole


def unary_predicate_matches_set_formula(predicate_symbol, atom_consts):
    # !x: (predicate_symbol(x) <=> (\/_{a_i \in atom_consts} x = a_i )
    return msol.QuantifiedFormula(
        msol.Quantifier.UNIVERSAL,
        [msol.Var1("x")],
        msol.BinaryFormula(
            msol.PredicateExpression(predicate_symbol, [msol.Var1("x")]),
            msol.BinaryConnective.BIIMPLICATION,
            msol.NaryFormula(msol.BinaryConnective.DISJUNCTION,
                             [msol.BinaryFormula(msol.Var1("x"), msol.BinaryConnective.EQ, c)
                              for c in atom_consts])

        ) if len(atom_consts) > 0 else msol.UnaryFormula(
            msol.UnaryConnective.NEGATION,
            msol.PredicateExpression(predicate_symbol, [msol.Var1("x")])
        ))

def binary_predicate_matches_set_formula(predicate_symbol, atom_consts):
    # !x,y: (predicate_symbol(x,y) <=> (\/_{(a_i, a_j) \in atom_consts} x = a_i & y = a_j)
    return msol.QuantifiedFormula(
        msol.Quantifier.UNIVERSAL,
        [msol.Var1("x")],
        msol.BinaryFormula(
            msol.PredicateExpression(predicate_symbol, [msol.Var1("x")]),
            msol.BinaryConnective.BIIMPLICATION,
            msol.NaryFormula(msol.BinaryConnective.DISJUNCTION,
                             [msol.BinaryFormula(msol.Var1("x"), msol.BinaryConnective.EQ, c)
                              for c in atom_consts])

        ) if len(atom_consts) > 0 else msol.UnaryFormula(
            msol.UnaryConnective.NEGATION,
            msol.PredicateExpression(predicate_symbol, [msol.Var1("x")])
        ))

def mol_to_tptp(mol: Chem.Mol):
    types = []
    axioms = []

    atom_consts = [f"atom_{i}" for i in range(len(mol.GetAtoms()))]

    for element in [6, 7, 8]:
        element_symbol = Chem.GetPeriodicTable().GetElementSymbol(element)
        types.append(f"{element_symbol}: $i > $o")
        matching_atoms = [atom_consts[i] for i in range(len(atom_consts)) if mol.GetAtomWithIdx(i).GetAtomicNum() == element]
        axioms.append(unary_predicate_matches_set_formula(element_symbol, matching_atoms))

    for charge in ["ChargeP", "ChargeN"]:
        types.append(f"{charge}: $i > $o")
    matching_atoms = [atom_consts[i] for i in range(len(atom_consts)) if mol.GetAtomWithIdx(i).GetFormalCharge() > 0]
    axioms.append(unary_predicate_matches_set_formula("ChargeP", matching_atoms))
    matching_atoms = [atom_consts[i] for i in range(len(atom_consts)) if mol.GetAtomWithIdx(i).GetFormalCharge() < 0]
    axioms.append(unary_predicate_matches_set_formula("ChargeN", matching_atoms))
    for charge in range(-3, 4):
        types.append(f"{charge}: $i > $o")
        matching_atoms = [atom_consts[i] for i in range(len(atom_consts)) if mol.GetAtomWithIdx(i).GetFormalCharge() == charge]
        axioms.append(unary_predicate_matches_set_formula(f"ChargeM{-charge}" if charge < 0 else f"Charge{charge}", matching_atoms))

    for h_count in range(1, 4):
        types.append(f"Has{h_count}Hs: $i > $o")
        matching_atoms = [atom_consts[i] for i in range(len(atom_consts)) if mol.GetAtomWithIdx(i).GetTotalNumHs() == h_count]
        axioms.append(unary_predicate_matches_set_formula(f"Has{h_count}Hs", matching_atoms))

    types.append("has_bond_to: $i > $i > $o")
    bonds = [(atom_consts[bond.GetBeginAtomIdx()], atom_consts[bond.GetEndAtomIdx()]) for bond in mol.GetBonds()]
    axioms.append(binary_predicate_matches_set_formula("has_bond_to", bonds + [(e, b) for b, e in bonds]))

    for bond_type in Chem.BondType.values.values():
        predicate_name = f"b{str(bond_type).split('.')[-1]}"
        types.append(f"{predicate_name}: $i > $i > $o")
        bonds = [(atom_consts[bond.GetBeginAtomIdx()], atom_consts[bond.GetEndAtomIdx()]) for bond in mol.GetBonds()
                 if bond.GetBondType() == bond_type or bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE] and bond.GetBondType() == Chem.BondType.AROMATIC]
        axioms.append(binary_predicate_matches_set_formula(predicate_name, bonds + [(e, b) for b, e in bonds]))

    for net_charge in ["NetCharge0", "NetChargeN", "NetChargeP"]:
        types.append(f"{net_charge}: $o")
    net_charge = Chem.GetFormalCharge(mol)
    axioms.append(~msol.PredicateExpression("NetCharge0", []) if net_charge != 0 else msol.PredicateExpression("NetCharge0", []))
    axioms.append(~msol.PredicateExpression("NetChargeN", []) if net_charge >= 0 else msol.PredicateExpression("NetChargeN", []))
    axioms.append(~msol.PredicateExpression("NetChargeP", []) if net_charge <= 0 else msol.PredicateExpression("NetChargeP", []))

    types = [AnnotatedFormula("thf", f"type_{i}", FormulaRole.TYPE, t) for i, t in enumerate(types)]
    axioms = [AnnotatedFormula("thf", f"axiom_{i}", FormulaRole.AXIOM, a) for i, a in enumerate(axioms)]
    return types + axioms