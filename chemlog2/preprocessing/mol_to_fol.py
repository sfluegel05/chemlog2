import logging

from rdkit import Chem
from gavel.logic import logic
from gavel.logic.logic_utils import substitute_var_in_formula
import numpy as np

from chemlog2.verification.model_checking import ModelChecker, ModelCheckerOutcome


def mol_to_fol_atoms(mol: Chem.Mol):
    # assumes: no wildcards, no aromaticity (kekulized), no h atoms
    universe = mol.GetNumAtoms() + 1
    extensions = {
        logic.BinaryConnective.EQ.name: np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "atom": np.ones(universe, dtype=np.bool_),
    }
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
            for predicate_symbol_charge in [f"charge_{'n' if charge < 0 else "p"}", f"charge{'_m' + str(-1 * charge) if charge < 0 else str(charge)}"]:
                if predicate_symbol_charge not in extensions:
                    extensions[predicate_symbol_charge] = np.zeros(universe, dtype=np.bool_)
                extensions[predicate_symbol_charge][atom_idx] = True
        # add predicates for h atoms
        # exception: if molecule only consists of a single H atom, don't assume that a second H has to be added
        if universe != 1 or atom.GetAtomicNum() != 1:
            num_hs = atom.GetTotalNumHs()
            predicate_symbol = f"has_{num_hs}_hs"
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
    for predicate_symbol, formula in fragment_predicate_definitions.items():
        extensions[predicate_symbol] = np.zeros(universe, dtype=np.bool_)
        for i, fragment in enumerate(fragments):
            # get atom-level FOL structure for each level, model-check properties, add properties to extensions
            fragment_universe, fragment_extensions = mol_to_fol_atoms(fragment)
            model_checker = ModelChecker(fragment_universe, fragment_extensions,
                                         predicate_definitions={formula.left.predicate.value: (
                                             formula.left.arguments, formula.right)
                                             for formula in fragment_helper_definitions.values()})
            target_formula = formula.right
            # assumes that the predicate is unary and the variable is the Global variable
            target_formula = substitute_var_in_formula(
                target_formula, logic.Variable("Global"), fragment_universe - 1
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
