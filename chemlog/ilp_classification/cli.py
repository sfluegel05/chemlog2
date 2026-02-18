import os
from typing import Literal
import json
import time

from chemlog.ilp_classification.mol2ilp import ILPProblemBuilder, tee_output
from chemlog.ilp_classification.learn_fgs import FGILPProblemBuilder
from chemlog.ilp_classification.ilp_classifier import run_ilp_training_subprocess, run_ilp_validation_subprocess
from chemlog.ilp_classification.ilp_path_manager import get_exs_path, get_bk_path, get_bias_path

def learn_chebi_classes(classes_list, ilp_builder: ILPProblemBuilder, results_dir, timeout=20, rebuild_samples=False, predicate_set: Literal["atoms", "chembl_fgs"]="atoms", max_pos_samples=100, max_neg_samples=100, selection_mode:Literal["claude", "random", "top_k"]|None=None, **kwargs):
    
        # Build settings parameters for Popper
        settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls",
            "timeout": timeout,
        }
        settings_parameters.update(kwargs)
        
        with open(os.path.join(results_dir, "config.yml"), "a+") as f:
            f.write(f"problem_dir: {ilp_builder.problem_dir}\n")
            f.write("popper_settings:\n")
            for key, value in settings_parameters.items():
                f.write(f"\t{key}: {value}\n")

        for chebi_id in classes_list:
            start_time = time.perf_counter()
            bk_path = get_bk_path(chebi_id, base_dir=ilp_builder.problem_dir, predicate_set=predicate_set)
            if selection_mode is not None:
                plain_bias_path = get_bias_path(chebi_id, base_dir=ilp_builder.problem_dir, predicate_set=predicate_set, selection_mode=selection_mode)
                if not os.path.exists(plain_bias_path):
                    raise ValueError(f"Settings-free bias file {plain_bias_path} does not exist. selection mode {selection_mode} requires pre-generated bias files with selected predicate names. Please generate bias files first or choose a different selection mode.")
            exs_path, bias_path = ilp_builder.build_ilp_problem(chebi_id, rebuild_samples=rebuild_samples, max_pos_samples=max_pos_samples, max_neg_samples=max_neg_samples, selection_mode=selection_mode)
            # Run training in subprocess (isolated Prolog session)
            print(f"Training ChEBI:{chebi_id}")
            train_result = run_ilp_training_subprocess(exs_path, bk_path, bias_path, settings_parameters, log_dir=results_dir)
            prog_str = train_result["prog_str"]  # string representation for display/storage
            score = train_result["score"]
            
            print(f"ChEBI:{chebi_id} - Score: {score}")
            print(f"    Learned program:\n{prog_str}")
            
            conf_matrix = None
            if prog_str is not None:
                # Run validation in subprocess (isolated Prolog session)
                print(f"Validating ChEBI:{chebi_id}...")
                try:
                    conf_matrix = run_ilp_validation_subprocess(
                        chebi_id, prog_str,
                        exs_file=get_exs_path(chebi_id, split="validation"),
                        bk_file=get_bk_path(chebi_id, predicate_set=predicate_set, split="validation"),
                        log_dir=results_dir
                    )
                except Exception as e:
                    print(f"Validation failed for ChEBI:{chebi_id} with error: {e}")
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
    parser.add_argument("--fg_mode", action="store_true", help="Whether to learn functional groups instead of ChEBI classes.")
    parser.add_argument("--chebi_version", type=int, default=244, help="ChEBI version to use.")
    parser.add_argument("--chebi_splits_file", type=str, default=None, help="Path to the ChEBI splits CSV file.")
    parser.add_argument("--timeout", type=int, default=20, help="Timeout for ILP solver in seconds.")
    parser.add_argument("--max_vars", type=int, default=6, help="Maximum number of variables in learned rules.")
    parser.add_argument("--max_body", type=int, default=6, help="Maximum number of body literals in learned rules.")
    parser.add_argument("--max_clauses", type=int, default=2, help="Maximum number of clauses in the learned program.")
    parser.add_argument("--max_pos_samples", type=int, default=200, help="Maximum number of positive samples per class.")
    parser.add_argument("--max_neg_samples", type=int, default=200, help="Maximum number of negative samples per class.")
    parser.add_argument("--rebuild_samples", action="store_true", help="Whether to rebuild train bk.pl and exs.pl even if they already exist.")
    parser.add_argument("--build_validation", action="store_true", help="Whether to build validation data (bk_validation.pl and exs_validation.pl).")
    parser.add_argument("--predicate_set", type=str, default="atoms", choices=["atoms", "chembl_fgs"], help="Whether to include CHEMBL FG predicates in the background knowledge.")
    parser.add_argument("--selection_mode", type=str, default=None, choices=["claude", "random", "top_k"], help="Mode for selecting body predicates to include in bias file. If not specified, no selection is done and all predicates are included in the bias file.")
    # arbitrary additional arguments (optional)
    parser.add_argument("popper_kwargs", nargs="*", default=[], help="Arguments for the Popper solver.")
    args = parser.parse_args()
    with open(args.labels_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    if not args.chebi_splits_file:
        chebi_splits_file = os.path.join("data", "splits_v244.csv")
    else:
        chebi_splits_file = args.chebi_splits_file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("ilp", "results", f"run_fgs_{timestamp}" if args.fg_mode else f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")  # create empty results file

    log_path = os.path.join(results_dir, "run.log")

    # write config file with settings used for this run
    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write(f"args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")


    with tee_output(log_path):
        if args.fg_mode:
            ilp_builder = FGILPProblemBuilder(chebi_version=args.chebi_version, chebi_split=chebi_splits_file, dataset_path=os.path.join("data", "chebi_fgs_dataset.pkl"), predicate_set=args.predicate_set, max_vars=args.max_vars, max_body=args.max_body, max_clauses=args.max_clauses)
        else:
            ilp_builder = ILPProblemBuilder(chebi_version=args.chebi_version, chebi_split=chebi_splits_file, muggleton=False, predicate_set=args.predicate_set, max_vars=args.max_vars, max_body=args.max_body, max_clauses=args.max_clauses)

    
        if args.build_validation:
            ilp_builder.build_validation(classes, max_pos_samples=args.max_pos_samples, max_neg_samples=args.max_neg_samples, predicate_set=args.predicate_set, rebuild_samples=args.rebuild_samples)
        learn_chebi_classes(classes, ilp_builder, results_dir, timeout=args.timeout, rebuild_samples=args.rebuild_samples, predicate_set=args.predicate_set, max_pos_samples=args.max_pos_samples, max_neg_samples=args.max_neg_samples, selection_mode=args.selection_mode, **{k: v for k, v in (arg.split("=") for arg in args.popper_kwargs)})
