import os
from rdkit import Chem
from rdkit.Chem import rdmolops
from chemlog.preprocessing.chebi_data import ChEBIData
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms


def build_ilp_problem(target_id, max_pos_samples=10, max_neg_samples=10, muggleton=False, save_dir=None):
    """
    Build an ILP problem for classifying molecules based on their membership in a ChEBI class.

    Args:
        target_id (str): The ChEBI ID of the target class (e.g., "24062" for alcohols).
        max_pos_samples (int): Maximum number of positive samples to include.
        max_neg_samples (int): Maximum number of negative samples to include.
    """
    if save_dir is None:
        save_dir = os.path.join("ilp", f"chebi_{target_id}")
    os.makedirs(save_dir, exist_ok=True)
    selected_rows = gather_samples_for_chebi_cls(target_id, save_dir, max_pos_samples, max_neg_samples)

    prolog_lines, body_predicates = build_background_muggleton(selected_rows) if muggleton else build_background_chemlog(selected_rows)

    with open(os.path.join(save_dir, "bk.pl"), "w+") as f:
        f.write("\n".join(prolog_lines) + "\n")
    # build bias file
    bias_lines = [
        f"%% CHEBI:{target_id}",
        f"%% intended rule:",
        f"%% chebi_{target_id}(V0) :- ???.",
        f"",
        f"head_pred(chebi_{target_id}, 1)."] + [
        f"body_pred({pred},{arity})." for pred, arity in body_predicates
    ] + [
        f"",
        #f"type(chebi_{target_id},(molecule,)).",
        #f"type(atom,(molecule,atom,atom_type,charge)).",    
        #f"type(bond,(molecule,atom,atom,bond_type)).",
        #f"",
        # directions dont work this way. If you know how they work, they can be helpful for learning
        #f"direction(chebi_{target_id},(in,)).",
        #f"direction(atom,(in,out,in,in)).",
        #f"direction(bond,(in,out,out,in)).",
    ]
    with open(os.path.join(save_dir, "bias.pl"), "w+") as f:
        f.write("\n".join(bias_lines) + "\n")

    print(f"ILP problem for ChEBI:{target_id} saved to {save_dir}")


def gather_samples_for_chebi_cls(target_id, save_dir, max_pos_samples=10, max_neg_samples=10):
    # take shortest SMILES with positive labels and random negative samples (from 3-STAR)
    chebi_data = ChEBIData(chebi_version=244)
    hierarchy_graph = chebi_data.get_trans_hierarchy()
    samples_df = chebi_data.processed[chebi_data.processed["subset"] == "3_STAR"]
    descendants = hierarchy_graph.successors(int(target_id))
    # not all descendants are molecules (i.e., have a SMILES annotation)
    mol_descendants = [d for d in descendants if d in samples_df.index]
    print(f"Found {len(mol_descendants)} molecular descendants of ChEBI:{target_id}")
    pos_mols, neg_mols = [], []
    pos_samples, neg_samples = [], []

    df_pos = samples_df.loc[mol_descendants]
    df_pos["smiles_length"] = df_pos["smiles"].apply(len)
    df_pos = df_pos.sort_values(by="smiles_length")
    for row in df_pos.itertuples():
        pos_mols.append(row)
        pos_samples.append(row.Index)
        if len(pos_samples) >= max_pos_samples:
            break

    df_neg = samples_df.loc[[ident for ident in samples_df.index if ident not in mol_descendants]]
    #df_neg["smiles_length"] = df_neg["smiles"].apply(len)
    #df_neg = df_neg.sort_values(by="smiles_length")
    for row in df_neg.sample(max_neg_samples).itertuples():
        neg_mols.append(row)
        neg_samples.append(row.Index)
        #if len(neg_samples) >= max_neg_samples:
        #    break
    
    with open(os.path.join(save_dir, "exs.pl"), "w+") as f:
        for sample in pos_samples:
            f.write(f"pos(chebi_{target_id}({sample})).\n")
        for sample in neg_samples:
            f.write(f"neg(chebi_{target_id}({sample})).\n")

    print(f"Collected {len(pos_samples)} positive and {len(neg_samples)} negative samples")

    return pos_mols + neg_mols

def get_atom_id(atom: int, molecule_id):
    return "a" + str(molecule_id) + "_" + str(atom + 1)  # Prolog indices start at 1


def build_background_chemlog(rows):
    comments = []
    lines_by_predicate = {"has_atom" : []}
    arities = {"has_atom" : 2}  # hardcode has_atom predicate
    for row in rows:
        comments.append(f"% CHEBI:{row.Index}, name: {row.name}, SMILES: {row.smiles}")
        # has atom predicates
        for atom in row.mol.GetAtoms():
            atom_id = get_atom_id(atom.GetIdx(), row.Index)
            lines_by_predicate["has_atom"].append(f"has_atom({row.Index},{atom_id}).")
        # predicates from FOL structure
        universe, extensions = mol_to_fol_atoms(row.mol)
        for predicate, sparse_extension in extensions.items():
            if predicate == "EQ" or predicate == "atom":
                continue  # skip equality predicate (implicit in Prolog)
            if predicate not in lines_by_predicate:
                lines_by_predicate[predicate] = []
            if predicate not in arities:
                arities[predicate] = len(sparse_extension.shape)
            if len(sparse_extension.shape) == 1:
                for idx in range(len(sparse_extension)):
                    if sparse_extension[idx]:
                        lines_by_predicate[predicate].append(f"{predicate}({get_atom_id(idx, row.Index)}).")
            elif len(sparse_extension.shape) == 2:
                for i in range(sparse_extension.shape[0]):
                    for j in range(sparse_extension.shape[1]):
                        if sparse_extension[i, j]:
                            lines_by_predicate[predicate].append(
                                f"{predicate}({get_atom_id(i, row.Index)},{get_atom_id(j, row.Index)})."
                            )
            else:
                raise ValueError(f"Unsupported sparse extension shape (>2D) for predicate {predicate}")
    
    

    return comments + [line for lines in lines_by_predicate.values() for line in lines], [(pred, arities[pred]) for pred in arities.keys()]


def build_background_muggleton(rows):
    all_atoms, all_bonds = [], []
    comments = []
    for row in rows:
        atoms, bonds = mol_to_prolog_muggleton(row.mol, molecule_id=row.Index)
        all_atoms.extend(atoms)
        all_bonds.extend(bonds)
        comments.append(f"% CHEBI:{row.Index}, name: {row.name}, SMILES: {row.smiles}")
    return comments + all_atoms + all_bonds, [("atom", 4), ("bond", 4)]

def mol_to_prolog_muggleton(mol, molecule_id="mol1"):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    prolog_atoms = []
    prolog_bonds = []

    for atom in atoms:
        atom_id = get_atom_id(atom.GetIdx(), molecule_id)
        atom_type = atom.GetSymbol().lower()
        atom_charge = atom.GetFormalCharge()
        prolog_atoms.append(
            f"atom({molecule_id},{atom_id},{atom_type},{atom_charge})."
        )

    for bond in bonds:
        start_atom_id = get_atom_id(bond.GetBeginAtom().GetIdx(), molecule_id)
        end_atom_id = get_atom_id(bond.GetEndAtom().GetIdx(), molecule_id)
        bond_type = str(bond.GetBondType()).lower()
        prolog_bonds.append(
            f"bond({molecule_id},{start_atom_id},{end_atom_id},{bond_type})."
        )
        prolog_bonds.append(
            f"bond({molecule_id},{end_atom_id},{start_atom_id},{bond_type})."
        )  # undirected 

    return prolog_atoms, prolog_bonds

if __name__ == "__main__":
    build_ilp_problem("24835", 100, 100)