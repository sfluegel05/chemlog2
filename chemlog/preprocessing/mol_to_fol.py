import logging
from itertools import product

from rdkit import Chem
from gavel.logic import logic
from gavel.logic.logic_utils import substitute_var_in_formula, get_vars_in_formula
import numpy as np

from chemlog.preprocessing.chebi_data import ChEBIData
from chemlog.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome
from chemlog.classification.peptide_size_classifier import get_chunks, get_possible_amino_chunk_assignments


def mol_to_fol_atoms(mol: Chem.Mol):
    # assumes: no wildcards, no aromaticity (kekulized), no h atoms (unless they have been added explicitly)
    universe = mol.GetNumAtoms() + 1
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "atom": np.ones(universe, dtype=np.bool_),
    }
    extensions["atom"][-1] = False # last position is global, not an atom
    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception as e:
        logging.error(
            f"Failed to assign CIP labels to molecule, skipping chirality-related extensions: {e}"
        )

    # for each atom, add atom symbol, charge, (chirality), equality to only itself
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol().lower()
        if atom_symbol not in extensions:
            extensions[atom_symbol] = np.zeros(universe, dtype=np.bool_)
        extensions[atom_symbol][atom_idx] = True
        charge = atom.GetFormalCharge()
        if charge != 0:
            # get both general direction and specific charge
            for predicate_symbol_charge in [f"charge_{'n' if charge < 0 else 'p'}",
                                            f"charge{'_m' + str(-1 * charge) if charge < 0 else str(charge)}"]:
                if predicate_symbol_charge not in extensions:
                    extensions[predicate_symbol_charge] = np.zeros(universe, dtype=np.bool_)
                extensions[predicate_symbol_charge][atom_idx] = True
        else:
            predicate_symbol_charge = "charge0"
            if predicate_symbol_charge not in extensions:
                extensions[predicate_symbol_charge] = np.zeros(universe, dtype=np.bool_)
            extensions[predicate_symbol_charge][atom_idx] = True
        # add predicates for h atoms
        # exception: if molecule only consists of a single H atom, don't assume that a second H has to be added
        if universe != 1 or atom.GetAtomicNum() != 1:
            num_hs = atom.GetTotalNumHs(includeNeighbors=True)
            predicate_symbols = [f"has_{num_hs}_hs"] + [f"has_at_least_{n}_hs" for n in range(1, num_hs + 1)]
            for predicate_symbol in predicate_symbols:
                if predicate_symbol not in extensions:
                    extensions[predicate_symbol] = np.zeros(
                        universe, dtype=np.bool_
                    )
                extensions[predicate_symbol][atom_idx] = True

        if atom.HasProp("_CIPCode"):
            chiral_code = f'cip_code_{atom.GetProp("_CIPCode")}'
            if chiral_code not in extensions:
                extensions[chiral_code] = np.zeros(universe, dtype=np.bool_)
            extensions[chiral_code][atom_idx] = True

    # add has_bond_to and bond-type specific predicates for each bond (symmetric)
    for bond in mol.GetBonds():
        predicate_symbol = f"b{bond.GetBondType()}"
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()

        if predicate_symbol not in extensions:
            extensions[predicate_symbol] = np.zeros(
                (universe, universe), dtype=np.bool_
            )
        extensions[predicate_symbol][left][right] = True
        extensions[predicate_symbol][right][left] = True

        # has bond to
        predicate_symbol = "has_bond_to"
        if predicate_symbol not in extensions:
            extensions[predicate_symbol] = np.zeros(
                (universe, universe), dtype=np.bool_
            )
        extensions[predicate_symbol][left][right] = True
        extensions[predicate_symbol][right][left] = True

        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            stereo_pred = f"b{bond.GetStereo().name}"
            if stereo_pred not in extensions:
                extensions[stereo_pred] = np.zeros(
                    (universe, universe), dtype=np.bool_
                )
            extensions[stereo_pred][left][right] = True
            extensions[stereo_pred][right][left] = True

    # use last place in extension for global properties
    extensions["net_charge_positive"] = np.zeros(universe, dtype=np.bool_)
    extensions["net_charge_negative"] = np.zeros(universe, dtype=np.bool_)
    extensions["net_charge_neutral"] = np.zeros(universe, dtype=np.bool_)
    extensions["global"] = np.zeros(universe, dtype=np.bool_)

    extensions["net_charge_positive"][-1] = Chem.GetFormalCharge(mol) > 0
    extensions["net_charge_negative"][-1] = Chem.GetFormalCharge(mol) < 0
    extensions["net_charge_neutral"][-1] = Chem.GetFormalCharge(mol) == 0
    extensions["global"][-1] = True

    return universe, extensions


def mol_to_fol_fragments(mol: Chem.Mol, fragment_predicate_definitions: dict, fragment_helper_definitions: dict):
    # apply rdkit-magic to get fragments
    # use fol model checking for each fragment to determine the extensions of the predicates (according to
    # supplied definitions)
    # fragment_predicate_definitions: list of dictionaries with keys "predicate_symbol", "formula" - arity is always 1
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    universe = len(fragments) + 1
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "overlap": np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "fragment": np.ones(universe, dtype=np.bool_),
    }
    fragment_model_checkers = [
        ModelChecker(*mol_to_fol_atoms(fragment), predicate_definitions={
            formula.left.predicate.value: (formula.left.arguments, formula.right)
            for formula in fragment_helper_definitions.values()})
        for fragment in fragments]
    for predicate_symbol, formula in fragment_predicate_definitions.items():
        extensions[predicate_symbol] = np.zeros(universe, dtype=np.bool_)
        for i, model_checker in enumerate(fragment_model_checkers):
            # get atom-level FOL structure for each level, model-check properties, add properties to extensions
            # assumes that the predicate is unary and the variable is the Global variable
            target_formula = formula.right
            target_formula = substitute_var_in_formula(
                target_formula, logic.Variable("Global"), model_checker.universe - 1
            )
            extensions[predicate_symbol][i] = model_checker.find_model(
                target_formula
            )[0] in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]

    # use last place in extension for global properties
    extensions["net_charge_positive"] = np.zeros(universe, dtype=np.bool_)
    extensions["net_charge_negative"] = np.zeros(universe, dtype=np.bool_)
    extensions["net_charge_neutral"] = np.zeros(universe, dtype=np.bool_)
    extensions["global"] = np.zeros(universe, dtype=np.bool_)

    extensions["net_charge_positive"][-1] = Chem.GetFormalCharge(mol) > 0
    extensions["net_charge_negative"][-1] = Chem.GetFormalCharge(mol) < 0
    extensions["net_charge_neutral"][-1] = Chem.GetFormalCharge(mol) == 0
    extensions["global"][-1] = True

    return universe, extensions


def mol_to_fol_building_blocks(mol: Chem.Mol, functional_groups: dict):
    universe = 0
    # identify carbon-fragments and building blocks with python-magic
    amide_bond_bonds = [mol.GetBondBetweenAtoms(amide_bond[0], amide_bond[2]) for amide_bond in
                        functional_groups["amide_bond"]]
    chunks = get_chunks(mol, amide_bond_bonds)
    amino_chunk_assignments = get_possible_amino_chunk_assignments(
        mol, [amino[0] for amino in functional_groups["amino_residue"]], chunks,
        [amide_bond[2] for amide_bond in functional_groups["amide_bond"]],
        [amide_bond[0] for amide_bond in functional_groups["amide_bond"]],
        [carboxy[0] for carboxy in functional_groups["carboxy_residue"]],
    )
    building_blocks = []
    for assignment in product(*amino_chunk_assignments):
        building_blocks += [chunk + [amino[0] for j, amino in enumerate(functional_groups["amino_residue"])
                                     if assignment[j] == i]
                            for i, chunk in enumerate(chunks)]
    # remove duplicates
    building_blocks = [list(t) for t in {tuple(bb) for bb in building_blocks}]
    universe = sum(len(v) for v in functional_groups.values()) + len(building_blocks) + 1
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
    }
    second_order_elements = []
    for fg, fg_atoms in list(functional_groups.items()) + [("building_block", building_blocks)]:
        extensions[fg] = np.array([len(second_order_elements) <= i < len(second_order_elements + fg_atoms)
                                   for i in range(universe)], dtype=np.bool_)
        second_order_elements += fg_atoms
    second_order_elements.append([])  # global placeholder

    extensions["overlap"] = np.array(
        [[any(atom in second_order_elements[i] for atom in second_order_elements[j]) for i in range(universe)] for j in
         range(universe)]
    )

    return universe, extensions, second_order_elements


def apply_variable_assignment(formula: logic.LogicElement, variable_assignment: dict):
    variables = get_vars_in_formula(formula)
    for variable_name, variable_value in variable_assignment.items():
        matching_variables = [v for v in variables if v.symbol.lower() == variable_name.lower()]
        if len(matching_variables) == 0:
            logging.warning(f"Variable {variable_name} not found in formula")
        if len(matching_variables) > 1:
            logging.warning(f"Multiple variables with name {variable_name} found in formula")
        formula = substitute_var_in_formula(formula, matching_variables[0], variable_value)
    return formula


def mol_to_fol_formula(mol: Chem.Mol, allow_additional_bonds: bool = False, add_global_charge: bool = False):
    clauses = []
    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception as e:
        logging.error(
            f"Failed to assign CIP labels to molecule, skipping chirality-related clauses: {e}"
        )

    v = [logic.Variable(f"A{i}") for i in range(mol.GetNumAtoms() + 1)]

    for atom in mol.GetAtoms():
        variable = v[atom.GetIdx()]
        # skip H atoms
        if atom.GetAtomicNum() == 1:
            continue
        clauses.append(logic.PredicateExpression("atom", [variable]))
        if atom.GetAtomicNum() != 0:
            clauses.append(logic.PredicateExpression(atom.GetSymbol().lower(), [variable]))
            charge = atom.GetFormalCharge()
            clauses.append(logic.PredicateExpression(f"charge{'_m' + str(-1 * charge) if charge < 0 else str(charge)}", [variable]))
            if atom.HasProp("_CIPCode"):
                clauses.append(logic.PredicateExpression(f"cip_code_{atom.GetProp('_CIPCode')}", [variable]))
        num_hs = atom.GetTotalNumHs(includeNeighbors=True)
        if not allow_additional_bonds and atom.GetAtomicNum() != 0:
            # obsolete if Hs are filtered out: and (len(list(mol.GetAtoms())) != 1 or atom.GetAtomicNum() != 1):
            clauses.append(logic.PredicateExpression(f"has_{num_hs}_hs", [variable]))
        elif num_hs > 0:
            # wildcards are an exception and can always have more H atoms
            clauses.append(logic.PredicateExpression(f"has_min_{num_hs}_hs", [variable]))

    for bond in mol.GetBonds():
        left = v[bond.GetBeginAtomIdx()]
        right = v[bond.GetEndAtomIdx()]
        # skip H atoms
        if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
            continue
        clauses.append(logic.PredicateExpression(f"b{bond.GetBondType()}", [left, right]))
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            clauses.append(logic.PredicateExpression(f"b{bond.GetStereo().name}", [left, right]))

    if add_global_charge:
        if Chem.GetFormalCharge(mol) < 0:
            clauses.append(logic.PredicateExpression("net_charge_negative", [v[-1]]))
        elif Chem.GetFormalCharge(mol) > 0:
            clauses.append(logic.PredicateExpression("net_charge_positive", [v[-1]]))
        else:
            clauses.append(logic.PredicateExpression("net_charge_neutral", [v[-1]]))

    return logic.QuantifiedFormula(logic.Quantifier.EXISTENTIAL, v,
                                   logic.NaryFormula(logic.BinaryConnective.CONJUNCTION, clauses))


if __name__ == "__main__":
    data = ChEBIData(239)
    print(mol_to_fol_atoms(data.processed.loc[48604, "mol"]))
    #for _, row in  data.processed[[83813 in row["parents"] for _, row in data.processed.iterrows()]].iterrows():
    #    print(row["name"], mol_to_fol_formula(row["mol"], allow_additional_bonds=False))
