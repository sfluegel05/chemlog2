import os
import subprocess
import sys
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Literal
import networkx as nx

import tqdm
from chemlog.ilp_classification.ilp_classifier import PopperWrapper, run_ilp_training_subprocess, run_ilp_validation_subprocess
from chemlog.preprocessing.chebi_data import ChEBIData
from chemlog.preprocessing.mol_to_fol import mol_to_fol_atoms
import pandas as pd
import time


@contextmanager
def tee_output(log_path):
    """Tee stdout/stderr to a log file while preserving console output."""
    log_file = open(log_path, "a", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
            self._buffer = ""

        def _emit(self, line, end=""):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for stream in self._streams:
                stream.write(f"[{timestamp}] {line}{end}")

        def write(self, data):
            self._buffer += data
            while True:
                newline_index = self._buffer.find("\n")
                if newline_index == -1:
                    break
                line = self._buffer[:newline_index].rstrip("\r")
                self._buffer = self._buffer[newline_index + 1:]
                self._emit(line, "\n")

        def flush(self):
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""
            for stream in self._streams:
                stream.flush()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


class ILPProblemBuilder:

    def __init__(self, chebi_version, chebi_split, problem_dir=None, muggleton=False, predicate_set: Literal["atoms", "chembl_fgs"] = "atoms", max_vars=6, max_body=6, max_clauses=2):
        self.chembl_fgs = predicate_set == "chembl_fgs"
        self.predicate_set = predicate_set
        self.chebi_version = chebi_version
        if not problem_dir:
            problem_dir = os.path.join("ilp", (f"chebi_v{chebi_version}"))
        self.problem_dir = problem_dir
        os.makedirs(self.problem_dir, exist_ok=True)
        self.muggleton = muggleton
        self.max_vars = max_vars
        self.max_body = max_body
        self.max_clauses = max_clauses

        self.chebi_data = ChEBIData(chebi_version=self.chebi_version)
        # we need 2 versions of the graph: one with directed transitive edges (e.g. to find all subclasses of x)
        # and one with undirected non-transitive edges (e.g. to find closest neighbors of x for sampling negatives)
        self.hierarchy_graph = self.chebi_data.get_trans_hierarchy()
        nontrans_hierarchy = self.chebi_data.build_hierarchy_graph()
        self.undirected_graph = nontrans_hierarchy.to_undirected()
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
            
    def build_train_bk(self):
        # big BK (for all training samples) -> not very efficient
        bk_dir = os.path.join(self.problem_dir, self.predicate_set)
        os.makedirs(bk_dir, exist_ok=True)
        train_rows = self.samples_df[[str(id) in self.train_ids for id in self.samples_df.index]]

        prolog_lines, body_predicates = build_background_muggleton(train_rows) if self.muggleton else build_background_chemlog(train_rows)
        if self.chembl_fgs:
            prolog_lines_fgs, body_predicates_fgs = build_background_chembl_fgs(self.chebi_data, train_rows)
            prolog_lines += prolog_lines_fgs
            body_predicates += body_predicates_fgs

        with open(os.path.join(bk_dir, "bk.pl"), "w+") as f:
            f.write("\n".join(prolog_lines) + "\n")
        # build bias file
        bias_lines = [
            f"%% (bias file without settings)",
            f"",
            f"%% max_vars(TODO).",
            f"%% max_body(TODO).",
            f"",
            f"head_pred(chebi_UNKNOWN, 1)."] + [
            f"body_pred({pred},{arity})." for pred, arity in body_predicates
        ]
        with open(os.path.join(bk_dir, "bias.pl"), "w+") as f:
            f.write("\n".join(bias_lines) + "\n")
        
        print(f"ILP train bk saved to {bk_dir}")
        return bk_dir
        
    def build_ilp_problem(self, target_id, rebuild_samples=False, max_pos_samples=100, max_neg_samples=100):
        """
        Build an ILP problem for classifying molecules based on their membership in a ChEBI class.

        Args:
            target_id (str): The ChEBI ID of the target class (e.g., "24062" for alcohols).
            rebuild_samples (bool): If False, reuse existing samples if they exist. If True, regenerate bk.pl and exs.pl even if they already exist.
            max_pos_samples (int): Maximum number of positive samples to include.
            max_neg_samples (int): Maximum number of negative samples to include.
        """
        target_dir = os.path.join(self.problem_dir, f"chebi_{target_id}")
        bk_dir = os.path.join(target_dir, self.predicate_set)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(bk_dir, exist_ok=True)

        bk_path = os.path.join(bk_dir, "bk.pl")
        exs_path = os.path.join(target_dir, "exs.pl")
        bias_path = os.path.join(bk_dir, f"bias_max_vars={self.max_vars}_max_body={self.max_body}.pl")

        selected_rows = None
        if rebuild_samples or not os.path.exists(exs_path):
            selected_rows = self.gather_samples_for_chebi_cls(target_id, max_pos_samples, max_neg_samples)
        
        if not (os.path.exists(bk_path) and os.path.exists(bias_path) and selected_rows is None):
            if selected_rows is None:
                with open(exs_path, "r") as f:
                    # for each line get id between inner parentheses (e.g. pos(chebi_123(456)). -> 456) and select corresponding rows from samples_df
                    selected_ids = [line.strip().split("(")[-1].split(")")[0] for line in f.readlines() if line.strip() and not line.startswith("%")]
                    selected_rows = self.samples_df[[str(id) in selected_ids for id in self.samples_df.index]]

            prolog_lines, body_predicates = build_background_muggleton(selected_rows) if self.muggleton else build_background_chemlog(selected_rows)
            if self.chembl_fgs:
                prolog_lines_fgs, body_predicates_fgs = build_background_chembl_fgs(self.chebi_data, selected_rows)
                prolog_lines += prolog_lines_fgs
                body_predicates += body_predicates_fgs

            with open(bk_path, "w+") as f:
                f.write("\n".join(prolog_lines) + "\n")
            # build bias file
            bias_lines = [
                f"%% CHEBI:{target_id} (bias file without settings)",
                f"",
                f"%% max_vars(TODO).",
                f"%% max_body(TODO).",
                f"%% max_clauses(TODO).",
                f"",
                f"head_pred(chebi_{target_id}, 1)."] + [
                f"body_pred({pred},{arity})." for pred, arity in body_predicates
            ]
            with open(os.path.join(bk_dir, "bias.pl"), "w+") as f:
                f.write("\n".join(bias_lines) + "\n")
            
            print(f"ILP problem for ChEBI:{target_id} saved to {target_dir}")
    

        # use bias.pl to generate settings-specific bias file
        with open(os.path.join(bk_dir, "bias.pl"), "r") as f:
            bias_content = f.read()
        bias_content = bias_content.replace("%% max_vars(TODO).", f"max_vars({self.max_vars}).")
        bias_content = bias_content.replace("%% max_body(TODO).", f"max_body({self.max_body}).")
        bias_content = bias_content.replace("%% max_clauses(TODO).", f"max_clauses({self.max_clauses}).") 
        with open(bias_path, "w+") as f:
            f.write(bias_content)



        return exs_path, bias_path

    def build_validation(self, target_ids, predicate_set: Literal["atoms", "chembl_fgs"], rebuild_samples=False, max_pos_samples=100, max_neg_samples=100):
        # validation bk knowledge
        validation_rows = self.samples_df[[str(id) in self.validation_ids for id in self.samples_df.index]]
        prolog_lines, body_predicates = build_background_muggleton(validation_rows) if self.muggleton else build_background_chemlog(validation_rows)
        if self.chembl_fgs:
            prolog_lines_fgs, body_predicates_fgs = build_background_chembl_fgs(self.chebi_data, validation_rows)
            prolog_lines += prolog_lines_fgs
            body_predicates += body_predicates_fgs

        os.makedirs(os.path.join(self.problem_dir, predicate_set), exist_ok=True)
        with open(os.path.join(self.problem_dir, predicate_set, "bk_validation.pl"), "w+") as f:
            f.write("\n".join(prolog_lines) + "\n")

        if rebuild_samples:
            validation_samples_df = self.samples_df[[str(id) in self.validation_ids for id in self.samples_df.index]]
            # nx make digraph undirected for distance calculations
            for target_id in tqdm.tqdm(target_ids, desc="Building validation data"):
                self.gather_validation_samples(target_id, validation_samples_df, self.undirected_graph, max_pos_samples, max_neg_samples)

    def get_closest_negatives(self, samples: pd.DataFrame, target_id, n_samples=100):
        # get closest samples in terms of distance in the chebi graph
        if n_samples >= len(samples):
            return samples
        import queue 
        q = queue.Queue()
        q.put(int(target_id))
        visited = set()
        selected = set()
        with open(os.path.join(self.problem_dir, "samples_idx.txt"), "w+") as f:
             f.write("\n".join(str(id) for id in samples.index))
        samples_index = list(str(id) for id in samples.index)
        while not q.empty() and len(selected) < n_samples:
            current = q.get()
            if str(current) in samples_index:
                selected.add(str(current))
            for neighbor in self.undirected_graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.put(neighbor)
        return self.samples_df.loc[[str(id) in selected for id in self.samples_df.index]]


    def gather_samples_for_chebi_cls(self, target_id, max_pos_samples=100, max_neg_samples=100) -> pd.DataFrame:
        # take shortest SMILES with positive labels and random negative samples (from 3-STAR)
        train_samples_df = self.samples_df[[str(id) in self.train_ids for id in self.samples_df.index]]
        descendants = list(self.hierarchy_graph.successors(int(target_id)))
        # not all descendants are molecules (i.e., have a SMILES annotation)
        print(f"Found {len(descendants)} molecular descendants of ChEBI:{target_id}")
        pos_samples, neg_samples = [], []

        df_pos = train_samples_df[[id in descendants for id in train_samples_df.index]]
        df_neg = train_samples_df[[id not in df_pos.index for id in train_samples_df.index]]

        #df_pos["smiles_length"] = df_pos["smiles"].apply(len)
        #df_pos = df_pos.sort_values(by="smiles_length")
        #pos_samples = df_pos[:max_pos_samples]
        pos_samples = df_pos.sample(min(max_pos_samples, len(df_pos)))

        #df_neg["dist_to_target"] = df_neg.index.to_series().apply(
        #    lambda x: min(nx.shortest_path_length(self.undirected_graph, int(label), int(target_id)) for label in self.hierarchy_graph.predecessors(int(x)) if int(label) in self.undirected_graph ))
        #df_neg = df_neg.sort_values(by="dist_to_target")
        # sample for each distance until we have enough samples or run out of samples
        neg_samples = []
        #for dist, group in df_neg.groupby("dist_to_target"):
        #    if len(neg_samples) >= max_neg_samples:
        #        break
        #    # shuffle group to get random samples from this distance
        #    group = group.sample(frac=1)
        #    neg_samples.extend(group.index.tolist())
        neg_samples = self.get_closest_negatives(df_neg, target_id, n_samples=max_neg_samples)
        
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

        os.makedirs(os.path.join(self.problem_dir, f"chebi_{target_id}"), exist_ok=True)
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
            # replace cip_code_s and cip_code_r with cip_code_S and cip_code_R
            if predicate.startswith("cip_code_"):
                predicate = "cip_code_" + predicate[-1].upper()
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


def build_background_chembl_fgs(chebi_data, rows):
    lines_by_predicate = dict()
    chembl_fgs = chebi_data.get_chembl_fgs()

    for row in rows.itertuples():
        for fg in chembl_fgs[row.Index]:
            if fg not in lines_by_predicate:
                lines_by_predicate[fg] = []
            lines_by_predicate[fg].append(f"{fg}({row.Index}).")
    total_lines = [line for lines in lines_by_predicate.values() for line in lines]
    return total_lines, [(pred, 1) for pred in lines_by_predicate.keys()]


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


def build_validation_data(labels_list, chebi_version=244, chebi_splits_file=None, rebuild_samples=False, predicate_set: Literal["atoms", "chembl_fgs"]="atoms", max_pos_samples=100, max_neg_samples=100):
    if not chebi_splits_file:
        chebi_splits_file = os.path.join("data", "splits_v244.csv")
    ilp_builder = ILPProblemBuilder(chebi_version=chebi_version, chebi_split=chebi_splits_file, muggleton=False, predicate_set=predicate_set)
    ilp_builder.build_validation(labels_list, max_pos_samples=max_pos_samples, max_neg_samples=max_neg_samples, predicate_set=predicate_set, rebuild_samples=rebuild_samples)

def learn_chebi_classes(classes_list, timeout=20, chebi_version=244, chebi_splits_file=None, rebuild_samples=False, predicate_set: Literal["atoms", "chembl_fgs"]="atoms", max_pos_samples=100, max_neg_samples=100, max_vars=6, max_body=6, max_clauses=2, **kwargs):
    if not chebi_splits_file:
        chebi_splits_file = os.path.join("data", "splits_v244.csv")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("ilp", "results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")  # create empty results file

    log_path = os.path.join(results_dir, "run.log")

    with tee_output(log_path):
        ilp_builder = ILPProblemBuilder(chebi_version=chebi_version, chebi_split=chebi_splits_file, muggleton=False, predicate_set=predicate_set, max_vars=max_vars, max_body=max_body, max_clauses=max_clauses)

        # Build settings parameters for Popper
        settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls",
            "timeout": timeout,
        }
        settings_parameters.update(kwargs)
        
        with open(os.path.join(results_dir, "config.yml"), "w+") as f:
            f.write(f"chebi_version: {chebi_version}\n")
            f.write(f"chebi_splits_file: {chebi_splits_file}\n")
            f.write(f"timeout: {timeout}\n")
            f.write(f"rebuild_samples: {rebuild_samples}\n")
            f.write(f"predicate_set: {predicate_set}\n")
            f.write(f"max_pos_samples: {max_pos_samples}\n")
            f.write(f"max_neg_samples: {max_neg_samples}\n")
            f.write(f"max_vars: {max_vars}\n")
            f.write(f"max_body: {max_body}\n")
            f.write(f"problem_dir: {ilp_builder.problem_dir}\n")
            f.write("popper_settings:\n")
            for key, value in settings_parameters.items():
                f.write(f"\t{key}: {value}\n")

        wrapper = PopperWrapper(settings_parameters)

        #if rebuild_samples:
            #bk_dir = ilp_builder.build_train_bk()
        #else:
            #bk_dir = os.path.join(ilp_builder.problem_dir, predicate_set)
        for chebi_id in classes_list:
            start_time = time.perf_counter()
            #try: 
            exs_path, bias_path = ilp_builder.build_ilp_problem(chebi_id, rebuild_samples=rebuild_samples, max_pos_samples=max_pos_samples, max_neg_samples=max_neg_samples)
            bk_dir = os.path.join(ilp_builder.problem_dir, f"chebi_{chebi_id}", predicate_set)
            
            # Run training in subprocess (isolated Prolog session)
            #train_result = run_ilp_training_subprocess(exs_path, bk_path, bias_path, settings_parameters, log_dir=results_dir)
            print(f"Training ChEBI:{chebi_id}")
            train_result = run_ilp_training_subprocess(exs_path, os.path.join(bk_dir, "bk.pl"), bias_path, settings_parameters, log_dir=results_dir)
            prog_str = train_result["prog_str"]  # string representation for display/storage
            score = train_result["score"]
            
            print(f"ChEBI:{chebi_id} - Score: {score}")
            print(f"    Learned program:\n{prog_str}")
            
            validate = False # todo restructure validation
            if validate:
                # Run validation in subprocess (isolated Prolog session)
                print(f"Validating ChEBI:{chebi_id}...")
                print(f"    Validation exs file: {os.path.join(ilp_builder.problem_dir, f'chebi_{chebi_id}', 'exs_validation.pl')}")
                print(f"    Validation bk file: {os.path.join(ilp_builder.problem_dir, predicate_set, 'bk_validation.pl')}")
                print(f"    Validation bias file: {bias_path}")
                conf_matrix = run_ilp_validation_subprocess(
                    chebi_id, prog_str,
                    exs_file=os.path.join(ilp_builder.problem_dir, f"chebi_{chebi_id}", "exs_validation.pl"),
                    bk_file=os.path.join(ilp_builder.problem_dir, predicate_set, "bk_validation.pl"),
                    bias_file=bias_path,
                    settings_parameters=settings_parameters,
                    log_dir=results_dir
                )
                #except Exception as e:
                #    print(f"Error processing ChEBI:{chebi_id} - {e}")
                #    prog_str = None
                #    score = None
                #    conf_matrix = None
            
            with open(os.path.join(results_dir, "results.json"), "a+") as f:
                result_entry = {
                    "chebi_id": chebi_id,
                    "train_score": {"TP": score[0], "FP": score[1], "TN": score[2], "FN": score[3]} if score else None,
                    "time_taken": time.perf_counter() - start_time,
                    "program": prog_str,
                    "validation_score": conf_matrix if validate else None,
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
        parser.add_argument("--max_vars", type=int, default=6, help="Maximum number of variables in learned rules.")
        parser.add_argument("--max_body", type=int, default=6, help="Maximum number of body literals in learned rules.")
        parser.add_argument("--max_clauses", type=int, default=2, help="Maximum number of clauses in the learned program.")
        parser.add_argument("--max_pos_samples", type=int, default=100, help="Maximum number of positive samples per class.")
        parser.add_argument("--max_neg_samples", type=int, default=100, help="Maximum number of negative samples per class.")
        parser.add_argument("--rebuild_samples", action="store_true", help="Whether to rebuild train bk.pl and exs.pl even if they already exist.")
        parser.add_argument("--build_validation", action="store_true", help="Whether to build validation data (bk_validation.pl and exs_validation.pl).")
        parser.add_argument("--predicate_set", type=str, default="atoms", choices=["atoms", "chembl_fgs"], help="Whether to include CHEMBL FG predicates in the background knowledge.")
        # arbitrary additional arguments (optional)
        parser.add_argument("popper_kwargs", nargs="*", default=[], help="Arguments for the Popper solver.")
        args = parser.parse_args()
        with open(args.labels_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        if args.build_validation:
            build_validation_data(classes, chebi_version=args.chebi_version, chebi_splits_file=args.chebi_splits_file, rebuild_samples=args.rebuild_samples, predicate_set=args.predicate_set, max_pos_samples=args.max_pos_samples, max_neg_samples=args.max_neg_samples)
        learn_chebi_classes(classes, chebi_version=args.chebi_version, chebi_splits_file=args.chebi_splits_file, rebuild_samples=args.rebuild_samples, predicate_set=args.predicate_set, timeout=args.timeout, max_pos_samples=args.max_pos_samples, max_neg_samples=args.max_neg_samples, max_vars=args.max_vars, max_body=args.max_body, max_clauses=args.max_clauses, **{k: v for k, v in (arg.split("=") for arg in args.popper_kwargs)})
