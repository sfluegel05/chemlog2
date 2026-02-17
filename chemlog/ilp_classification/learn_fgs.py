
import os
import pandas as pd

from chemlog.ilp_classification.mol2ilp import ILPProblemBuilder, build_background_chemlog, build_background_chembl_fgs

class FGILPProblemBuilder(ILPProblemBuilder):

    @property
    def problem_dir(self):
        return self._problem_dir if self._problem_dir else os.path.join("ilp", "learn_fgs")
    
    def load_samples(self, dataset_path):
        if dataset_path is None:
            dataset_path = os.path.join("data", "chebi_fgs_dataset.pkl")
        return pd.read_pickle(dataset_path)

    def build_ilp_problem(self, target_id, rebuild_samples=False, max_pos_samples=200, max_neg_samples=200):
        """
        Build an ILP problem for classifying molecules based on their has_part relation to a functional group.

        Args:
            target_id (str): The ChEBI ID of the target functional group (e.g., "24062" for alcohols).
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

            prolog_lines, body_predicates = build_background_chemlog(selected_rows)
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
    

    def gather_samples_for_chebi_cls(self, target_id, max_pos_samples=100, max_neg_samples=100) -> pd.DataFrame:
        train_samples_df = self.samples_df[[str(id) in self.train_ids for id in self.samples_df.index]]

        df_pos = train_samples_df[train_samples_df[f"has_part_{target_id}"]]
        df_neg = train_samples_df[~train_samples_df[f"has_part_{target_id}"]]

        pos_samples = df_pos.sample(min(max_pos_samples, len(df_pos)))

        neg_samples = self.get_closest_negatives(df_neg, target_id, n_samples=max_neg_samples)
        
        with open(os.path.join(self.problem_dir, f"chebi_{target_id}", "exs.pl"), "w+") as f:
            for sample in pos_samples.index:
                f.write(f"pos(chebi_{target_id}({sample})).\n")
            for sample in neg_samples.index:
                f.write(f"neg(chebi_{target_id}({sample})).\n")

        print(f"Training on {len(pos_samples)} positive and {len(neg_samples)} negative samples")

        return pd.concat([pos_samples, neg_samples])
