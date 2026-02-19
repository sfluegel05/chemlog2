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
from chemlog.ilp_classification.ilp_path_manager import get_bk_path, get_bias_path, get_exs_path


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


CHEBI_FG_RULES_PATH = os.path.join("data", "chebi_fg_rules_from_smiles.pl")


class ILPProblemBuilder:

    def __init__(self, chebi_version, chebi_split, problem_dir=None, muggleton=False, predicate_set: Literal["atoms", "chembl_fgs", "chebi_fgs", "chebi_fg_rules"] = "atoms", max_vars=6, max_body=6, max_clauses=2, **kwargs):
        # chembl_fgs: ChEMBL FGs supplied as samples
        # chebi_fgs: ChEBI FGs supplied as samples
        # chebi_fg_rules: ChEBI FGs supplied as Prolog rules (extracted from ChEBI SMILES) - currently broken
        # chebi_fgs_learned_rules #todo ChEBI FGs supplied as rules, learned with ILP from chebi_fgs
        self.chembl_fgs = predicate_set == "chembl_fgs"
        self.chebi_fg_rules = predicate_set == "chebi_fg_rules"
        self.chebi_fgs = predicate_set == "chebi_fgs"
        
        self.predicate_set = predicate_set
        self.chebi_version = chebi_version
        self._problem_dir = problem_dir
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
        self.samples_df = self.load_samples(kwargs["dataset_path"] if "dataset_path" in kwargs else None)

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
                    
    @property
    def problem_dir(self):
        return self._problem_dir if self._problem_dir else os.path.join("ilp", f"chebi_v{self.chebi_version}")
            
    def load_samples(self, dataset_path):
        return self.chebi_data.processed[self.chebi_data.processed["subset"] == "3_STAR"]
        
    def build_ilp_problem(self, target_id, rebuild_samples=False, max_pos_samples=100, max_neg_samples=100, selection_mode: Literal["claude", "random", "top_k"]|None=None):
        """
        Build an ILP problem for classifying molecules based on their membership in a ChEBI class.

        Args:
            target_id (str): The ChEBI ID of the target class (e.g., "24062" for alcohols).
            rebuild_samples (bool): If False, reuse existing samples if they exist. If True, regenerate bk.pl and exs.pl even if they already exist.
            max_pos_samples (int): Maximum number of positive samples to include.
            max_neg_samples (int): Maximum number of negative samples to include.
            selection_mode (Literal["claude", "random", "top_k"]|None):
            """
        
        bk_path = get_bk_path(target_id, base_dir=self.problem_dir, predicate_set=self.predicate_set)
        exs_path = get_exs_path(target_id, base_dir=self.problem_dir)
        bias_path = get_bias_path(target_id, base_dir=self.problem_dir, predicate_set=self.predicate_set, max_vars=self.max_vars, max_body=self.max_body, max_clauses=self.max_clauses, selection_mode=selection_mode)
        plain_bias_path = get_bias_path(target_id, base_dir=self.problem_dir, predicate_set=self.predicate_set, selection_mode=selection_mode)

        selected_rows = None
        if rebuild_samples or not os.path.exists(exs_path):
            selected_rows = self.gather_samples_for_chebi_cls(target_id, max_pos_samples, max_neg_samples)
        
        if not (os.path.exists(bk_path) and os.path.exists(plain_bias_path) and selected_rows is None):
            if selected_rows is None:
                with open(exs_path, "r") as f:
                    # for each line get id between inner parentheses (e.g. pos(chebi_123(456)). -> 456) and select corresponding rows from samples_df
                    selected_ids = [line.strip().split("(")[-1].split(")")[0] for line in f.readlines() if line.strip() and not line.startswith("%")]
                    selected_rows = self.samples_df[[str(id) in selected_ids for id in self.samples_df.index]]

            prolog_lines, body_predicates = build_background_muggleton(selected_rows) if self.muggleton else build_background_chemlog(selected_rows)
            if self.predicate_set in ["chembl_fgs", "chebi_fgs"]:
                prolog_lines_fgs, body_predicates_fgs = build_background_fg_data(self.chebi_data, selected_rows, source=self.predicate_set)
                prolog_lines += prolog_lines_fgs
                body_predicates += body_predicates_fgs
            if self.chebi_fg_rules:
                prolog_lines_fgs, body_predicates_fgs = build_background_chebi_fg_rules()
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
            # bias without settings (as template)
            with open(plain_bias_path, "w+") as f:
                f.write("\n".join(bias_lines) + "\n")
            
            print(f"ILP problem for ChEBI:{target_id} saved to exs: {exs_path}, bk: {bk_path}, bias: {plain_bias_path}")
    

        # use bias.pl to generate settings-specific bias file
        with open(plain_bias_path, "r") as f:
            bias_content = f.read()
        bias_content = bias_content.replace("%% max_vars(TODO).", f"max_vars({self.max_vars}).")
        bias_content = bias_content.replace("%% max_body(TODO).", f"max_body({self.max_body}).")
        bias_content = bias_content.replace("%% max_clauses(TODO).", f"max_clauses({self.max_clauses}).") 
                
        with open(bias_path, "w+") as f:
            f.write(bias_content)



        return exs_path, bias_path

    def build_validation(self, target_ids, predicate_set: Literal["atoms", "chembl_fgs", "chebi_fg_rules"], split: Literal["validation", "test"]="validation", rebuild_samples=False, max_pos_samples=100, max_neg_samples=100):
        # validation bk knowledge
        if split == "validation":
            validation_rows = self.samples_df[[str(id) in self.validation_ids for id in self.samples_df.index]]
        elif split == "test":
            validation_rows = self.samples_df[[str(id) in self.test_ids for id in self.samples_df.index]]
        prolog_lines, body_predicates = build_background_muggleton(validation_rows) if self.muggleton else build_background_chemlog(validation_rows)
        if self.predicate_set in ["chembl_fgs", "chebi_fgs"]:
            prolog_lines_fgs, body_predicates_fgs = build_background_fg_data(self.chebi_data, validation_rows, source=self.predicate_set)
            prolog_lines += prolog_lines_fgs
            body_predicates += body_predicates_fgs
        if self.chebi_fg_rules:
            prolog_lines_fgs, body_predicates_fgs = build_background_chebi_fg_rules()
            prolog_lines += prolog_lines_fgs
            body_predicates += body_predicates_fgs

        os.makedirs(os.path.join(self.problem_dir, predicate_set), exist_ok=True)
        bk_path = get_bk_path(None, base_dir=self.problem_dir, predicate_set=predicate_set, split=split)
        with open(bk_path, "w+") as f:
            f.write("\n".join(prolog_lines) + "\n")

        validation_samples_df = self.samples_df[[str(id) in self.validation_ids for id in self.samples_df.index]]
        for target_id in tqdm.tqdm(target_ids, desc=f"Building {split} data"):
            exs_path = get_exs_path(target_id, base_dir=self.problem_dir, split=split)
            if not os.path.exists(exs_path) or rebuild_samples:
                self.gather_validation_samples(target_id, validation_samples_df, max_pos_samples, max_neg_samples, split=split)

    def get_closest_negatives(self, samples: pd.DataFrame, target_id, n_samples=100):
        # get closest samples in terms of distance in the chebi graph
        if n_samples >= len(samples):
            return samples
        import queue 
        q = queue.Queue()
        q.put(int(target_id))
        visited = set() # visit closest labels
        selected = set() # select samples that are subclasses of closest labels until we have enough samples
        samples_index = list(str(id) for id in samples.index)
        while not q.empty() and len(selected) < n_samples:
            current = q.get()
            for neighbor in self.undirected_graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.put(neighbor)
                    for neighbor_sub in self.hierarchy_graph.successors(neighbor):
                        if str(neighbor_sub) in samples_index:
                            selected.add(str(neighbor_sub))
                        if len(selected) >= n_samples:
                            return self.samples_df.loc[[str(id) in selected for id in self.samples_df.index]]

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
        
        exs_path = get_exs_path(target_id, base_dir=self.problem_dir)
        with open(exs_path, "w+") as f:
            for sample in pos_samples.index:
                f.write(f"pos(chebi_{target_id}({sample})).\n")
            for sample in neg_samples.index:
                f.write(f"neg(chebi_{target_id}({sample})).\n")

        print(f"Training on {len(pos_samples)} positive and {len(neg_samples)} negative samples")

        return pd.concat([pos_samples, neg_samples])

    def gather_validation_samples(self, target_id, validation_samples_df, max_pos_samples=100, max_neg_samples=100, split: Literal["validation", "test"]="validation") -> tuple[int, int]:
        import networkx as nx
        descendants = list(self.hierarchy_graph.successors(int(target_id)))
        df_pos = validation_samples_df[[int(id) in descendants for id in validation_samples_df.index]]
        df_neg = validation_samples_df[[id not in df_pos.index for id in validation_samples_df.index]]
        df_pos = df_pos.sample(min(max_pos_samples, len(df_pos)))
        df_neg = self.get_closest_negatives(df_neg, target_id, n_samples=max_neg_samples)

        exs_path = get_exs_path(target_id, base_dir=self.problem_dir, split=split)
        with open(exs_path, "w+") as f:
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
            if predicate == "EQ" or predicate == "atom" or predicate == "*":
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


def build_background_chebi_fg_rules(rules_path=None):
    """Load ChEBI functional group rules from a Prolog file and return them as BK lines and body predicates.
    
    Each rule defines a chebi_XXXXX(M) predicate in terms of atom-level predicates.
    These are added as Prolog rules to the BK and as body_pred entries (arity 1) in the bias.
    """
    if rules_path is None:
        rules_path = CHEBI_FG_RULES_PATH
    
    prolog_lines = [f"% ChEBI FG rules from {os.path.basename(rules_path)}"]
    body_predicates = []
    seen_predicates = set()
    
    with open(rules_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            prolog_lines.append(line)
            # Extract predicate name from head: chebi_XXXXX(M) :- ...
            pred_name = line.split("(")[0].strip()
            if pred_name and pred_name not in seen_predicates:
                seen_predicates.add(pred_name)
                body_predicates.append((pred_name, 1))
    
    print(f"Loaded {len(body_predicates)} ChEBI FG rule predicates from {rules_path}")
    return prolog_lines, body_predicates


def build_background_fg_data(chebi_data, rows, source: Literal["chembl_fgs", "chebi_fgs"]):
    lines_by_predicate = dict()
    if source == "chembl_fgs":
        fg_data = chebi_data.get_chembl_fgs()
    elif source == "chebi_fgs":
        fg_data = chebi_data.get_chebi_fgs()
    else:
        raise ValueError(f"Unknown source {source}")

    for row in rows.itertuples():
        for fg in fg_data[row.Index]:
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
