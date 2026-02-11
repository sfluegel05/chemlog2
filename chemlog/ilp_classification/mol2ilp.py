import os
import subprocess
import sys
import json
import networkx as nx

import tqdm
from chemlog.ilp_classification.ilp_classifier import run_ilp_training_subprocess, run_ilp_validation_subprocess
from chemlog.preprocessing.chebi_data import ChEBIData
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms
import pandas as pd
import time


class ILPProblemBuilder:

    def __init__(self, chebi_version, chebi_split, problem_dir=None, muggleton=False):
        self.chebi_version = chebi_version
        if not problem_dir:
            problem_dir = os.path.join("ilp", f"chebi_v{chebi_version}")
        self.problem_dir = problem_dir
        os.makedirs(self.problem_dir, exist_ok=True)
        self.muggleton = muggleton

        self.chebi_data = ChEBIData(chebi_version=self.chebi_version)
        self.hierarchy_graph = self.chebi_data.get_trans_hierarchy()
        self.samples_df = self.chebi_data.processed[self.chebi_data.processed["subset"] == "3_STAR"]

        # load splits from csv file
        with open(chebi_split, "r") as f:
            lines = f.readlines()
        self.train_ids = set()
        self.validation_ids = set()
        self.test_ids = set()
        for line in lines[1:]:
            parts = line.strip().split(",")
            chebi_id = parts[0].strip()
            split = parts[1]
            if split == "train":
                self.train_ids.add(chebi_id)
            elif split == "validation":
                self.validation_ids.add(chebi_id)
            elif split == "test":
                self.test_ids.add(chebi_id)
            else:
                raise ValueError(f"Unknown split '{split}' for ChEBI ID {chebi_id}")
            
        
    def build_ilp_problem(self,target_id, max_pos_samples=100, max_neg_samples=100):
        """
        Build an ILP problem for classifying molecules based on their membership in a ChEBI class.

        Args:
            target_id (str): The ChEBI ID of the target class (e.g., "24062" for alcohols).
            max_pos_samples (int): Maximum number of positive samples to include.
            max_neg_samples (int): Maximum number of negative samples to include.
        """
        
        target_dir = os.path.join(self.problem_dir, f"chebi_{target_id}")
        os.makedirs(target_dir, exist_ok=True)

        selected_rows = self.gather_samples_for_chebi_cls(target_id, max_pos_samples, max_neg_samples)

        prolog_lines, body_predicates = build_background_muggleton(selected_rows) if self.muggleton else build_background_chemlog(selected_rows)

        with open(os.path.join(target_dir, "bk.pl"), "w+") as f:
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
        with open(os.path.join(target_dir, "bias.pl"), "w+") as f:
            f.write("\n".join(bias_lines) + "\n")

        print(f"ILP problem for ChEBI:{target_id} saved to {target_dir}")

        return

    def build_validation(self, target_ids, max_pos_samples=100, max_neg_samples=100):
        # validation bk knowledge
        validation_rows = self.samples_df[[str(id) in self.validation_ids for id in self.samples_df.index]]
        prolog_lines, body_predicates = build_background_muggleton(validation_rows) if self.muggleton else build_background_chemlog(validation_rows)

        with open(os.path.join(self.problem_dir, "bk_validation.pl"), "w+") as f:
            f.write("\n".join(prolog_lines) + "\n")

        validation_samples_df = self.samples_df[[str(id) in self.validation_ids for id in self.samples_df.index]]
        # take subgraph of chebi hierarchy containing all target classes and their ancestors
        nontrans_hierarchy = self.chebi_data.build_hierarchy_graph()
        subgraph_nodes = set()
        for target_id in list(target_ids) + list(self.validation_ids):
            subgraph_nodes.add(int(target_id))
            subgraph_nodes.update(nx.ancestors(nontrans_hierarchy, int(target_id)))
        subgraph = nontrans_hierarchy.subgraph(subgraph_nodes)
        print(f"Validation hierarchy subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        # nx make digraph undirected for distance calculations
        undirected_graph = subgraph.to_undirected()
        for target_id in tqdm.tqdm(target_ids, desc="Building validation data"):
            self.gather_validation_samples(target_id, validation_samples_df, undirected_graph, max_pos_samples, max_neg_samples)

    def gather_samples_for_chebi_cls(self, target_id, max_pos_samples=100, max_neg_samples=100) -> pd.DataFrame:
        # take shortest SMILES with positive labels and random negative samples (from 3-STAR)
        train_samples_df = self.samples_df[[str(id) in self.train_ids for id in self.samples_df.index]]
        descendants = list(self.hierarchy_graph.successors(int(target_id)))
        # not all descendants are molecules (i.e., have a SMILES annotation)
        print(f"Found {len(descendants)} molecular descendants of ChEBI:{target_id}")
        pos_samples, neg_samples = [], []

        df_pos = train_samples_df[[id in descendants for id in train_samples_df.index]]
        df_neg = train_samples_df[[id not in df_pos.index for id in train_samples_df.index]]

        df_pos["smiles_length"] = df_pos["smiles"].apply(len)
        df_pos = df_pos.sort_values(by="smiles_length")
        pos_samples = df_pos[:max_pos_samples]
        neg_samples = df_neg.sample(max_neg_samples)

        
        with open(os.path.join(self.problem_dir, f"chebi_{target_id}", "exs.pl"), "w+") as f:
            for sample in pos_samples.index:
                f.write(f"pos(chebi_{target_id}({sample})).\n")
            for sample in neg_samples.index:
                f.write(f"neg(chebi_{target_id}({sample})).\n")

        print(f"Training on {len(pos_samples)} positive and {len(neg_samples)} negative samples")

        return pd.concat([pos_samples, neg_samples])

    def gather_validation_samples(self, target_id, validation_samples_df, undirected_graph, max_pos_samples=100, max_neg_samples=100) -> tuple[int, int]:
        import networkx as nx
        descendants = list(self.hierarchy_graph.successors(int(target_id)))
        df_pos = validation_samples_df[[int(id) in descendants for id in validation_samples_df.index]]
        df_neg = validation_samples_df[[id not in df_pos.index for id in validation_samples_df.index]]
        df_pos = df_pos.sample(min(max_pos_samples, len(df_pos)))
        # instead of sampling, take samples that are closest in the chebi graph (minimum distance between classes)
        df_neg["dist_to_target"] = df_neg.index.to_series().apply(
            lambda x: min(nx.shortest_path_length(undirected_graph, int(label), int(target_id)) for label in self.hierarchy_graph.predecessors(int(x)) if int(label) in undirected_graph ))
        df_neg = df_neg.sort_values(by="dist_to_target")
        # sample for each distance until we have enough samples or run out of samples
        neg_samples = []
        for dist, group in df_neg.groupby("dist_to_target"):
            if len(neg_samples) >= max_neg_samples:
                break
            # shuffle group to get random samples from this distance
            group = group.sample(frac=1)
            neg_samples.extend(group.index.tolist())
        df_neg = df_neg.loc[neg_samples[:max_neg_samples]]

        with open(os.path.join(self.problem_dir, f"chebi_{target_id}", "exs_validation.pl"), "w+") as f:
            for mol_id in df_pos.index:
                f.write(f"pos(chebi_{target_id}({mol_id})).\n")
            for mol_id in df_neg.index:
                f.write(f"neg(chebi_{target_id}({mol_id})).\n")

        print(f"Validating ChEBI:{target_id} on {len(df_pos)} positive and {len(df_neg)} negative samples")

        return len(df_pos), len(df_neg)

def get_atom_id(atom: int, molecule_id):
    return "a" + str(molecule_id) + "_" + str(atom + 1)  # Prolog indices start at 1


def build_background_chemlog(rows):
    print(f"Building Chemlog-style background knowledge for {len(rows)} molecules")
    comments = []
    lines_by_predicate = {"has_atom" : []}
    arities = {"has_atom" : 2}  # hardcode has_atom predicate
    for row in rows.itertuples():
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


def build_validation_data(labels_list, chebi_version=244, chebi_splits_file=None, max_pos_samples=100, max_neg_samples=100):
    if not chebi_splits_file:
        chebi_splits_file = os.path.join("data", "splits_v244.csv")
    ilp_builder = ILPProblemBuilder(chebi_version=chebi_version, chebi_split=chebi_splits_file, muggleton=False)
    ilp_builder.build_validation(labels_list, max_pos_samples=max_pos_samples, max_neg_samples=max_neg_samples)

def learn_chebi_classes(classes_list, timeout=20, chebi_version=244, chebi_splits_file=None, max_pos_samples=100, max_neg_samples=100, **kwargs):
    if not chebi_splits_file:
        chebi_splits_file = os.path.join("data", "splits_v244.csv")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("ilp", "results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")  # create empty results file
    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write(f"chebi_version: {chebi_version}\n")
        f.write(f"chebi_splits_file: {chebi_splits_file}\n")
        f.write(f"timeout: {timeout}\n")
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")

    # Build settings parameters for Popper
    settings_parameters = {
        "noisy": True,
        "anytime_solver": "nuwls",
        "timeout": timeout,
    }
    settings_parameters.update(kwargs)
    
    ilp_builder = ILPProblemBuilder(chebi_version=chebi_version, chebi_split=chebi_splits_file, muggleton=False)
    
    for chebi_id in classes_list:
        start_time = time.perf_counter()
        try: 
            n_validation_pos, n_validation_neg = ilp_builder.build_ilp_problem(chebi_id, max_pos_samples=max_pos_samples, max_neg_samples=max_neg_samples)
            
            # Run training in subprocess (isolated Prolog session)
            problem_path = os.path.join(ilp_builder.problem_dir, f"chebi_{chebi_id}")
            train_result = run_ilp_training_subprocess(problem_path, settings_parameters, log_dir=results_dir)
            prog = train_result["prog"]  # actual prog object
            prog_str = train_result["prog_str"]  # string representation for display/storage
            score = train_result["score"]
            
            print(f"ChEBI:{chebi_id} - Score: {score}")
            print(f"    Learned program:\n{prog_str}")
            
            # Run validation in subprocess (isolated Prolog session)
            conf_matrix = run_ilp_validation_subprocess(
                chebi_id, prog, n_validation_pos, n_validation_neg,
                problem_dir=ilp_builder.problem_dir, 
                settings_parameters=settings_parameters,
                log_dir=results_dir
            )
        except Exception as e:
            print(f"Error processing ChEBI:{chebi_id} - {e}")
            prog_str = None
            score = None
            conf_matrix = None
        
        with open(os.path.join(results_dir, "results.json"), "a+") as f:
            result_entry = {
                "chebi_id": chebi_id,
                "train_score": {"TP": score[0], "FP": score[1], "TN": score[2], "FN": score[3]} if score else None,
                "time_taken": time.perf_counter() - start_time,
                "program": prog_str,
                "validation_score": conf_matrix,
            }
            f.write(json.dumps(result_entry) + "\n")


if __name__ == "__main__":
    # use command line arguments as kwargs for eval_chebi_classes
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate ILP classification on ChEBI classes using Popper.")
        parser.add_argument("--labels_file", type=str, default=None, help="Path to the labels file.")
        parser.add_argument("--chebi_version", type=int, default=244, help="ChEBI version to use.")
        parser.add_argument("--chebi_splits_file", type=str, default=None, help="Path to the ChEBI splits CSV file.")
        parser.add_argument("--timeout", type=int, default=20, help="Timeout for ILP solver in seconds.")
        # arbitrary additional arguments (optional)
        parser.add_argument("popper_kwargs", nargs="*", default=[], help="Arguments for the Popper solver.")
        args = parser.parse_args()
        with open(args.labels_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        build_validation_data(classes, chebi_version=args.chebi_version, chebi_splits_file=args.chebi_splits_file, max_pos_samples=100, max_neg_samples=100)
        learn_chebi_classes(classes, chebi_version=args.chebi_version, chebi_splits_file=args.chebi_splits_file, timeout=args.timeout, **{k: v for k, v in (arg.split("=") for arg in args.popper_kwargs)})
