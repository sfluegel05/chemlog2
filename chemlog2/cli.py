import json
import time

import click
import tqdm

from chemlog2.classification.charge_classifier import get_charge_category, ChargeCategories
from chemlog2.verification.functional_groups_verifier import FunctionalGroupsVerifier
from chemlog2.classification.peptide_size_classifier import get_n_amino_acid_residues
from chemlog2.classification.proteinogenics_classifier import get_proteinogenic_amino_acids
from chemlog2.classification.peptide_size_classifier import get_carboxy_derivatives, get_amide_bonds, get_amino_groups
from chemlog2.classification.substructure_classifier import is_emericellamide, is_diketopiperazine
from chemlog2.preprocessing.chebi_data import ChEBIData
from chemlog2.timestamped_logger import TimestampedLogger
import logging
import os
import ast

from chemlog2.verification.charge_verifier import ChargeVerifier
from chemlog2.verification.model_checking import ModelCheckerOutcome
from chemlog2.verification.peptide_size_verifier import PeptideSizeVerifier
from chemlog2.verification.proteinogenics_verifier import ProteinogenicsVerifier
from chemlog2.verification.substruct_verifier import SubstructVerifier


class LiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.group()
def cli():
    pass


def resolve_chebi_classes(classification):
    # todo: use the ontology to automatically add indirect superclasses
    n_amino_acid_residues = classification["n_amino_acid_residues"]
    charge_category = classification["charge_category"]
    res = []
    if charge_category == ChargeCategories.SALT.name:
        res.append(24866) # salt (there is no class peptide salt)
    elif charge_category == ChargeCategories.ANION.name:
        res.append(25696)
    elif charge_category == ChargeCategories.CATION.name:
        res.append(25697)
    elif charge_category == ChargeCategories.ZWITTERION.name:
        res.append(27369)
    if n_amino_acid_residues >= 2:
        if charge_category == ChargeCategories.ANION.name:
            # peptide anion
            res.append(60334)
        elif charge_category == ChargeCategories.CATION.name:
            # peptide cation
            res.append(60194)
        elif charge_category == ChargeCategories.ZWITTERION.name:
            res.append(60466)
            if n_amino_acid_residues == 2:
                # zwitterion, peptide zwitterion, dipeptide zwitterion
                res.append(90799)
            if n_amino_acid_residues == 3:
                res.append(155837)
        elif charge_category == ChargeCategories.NEUTRAL.name:
            res.append(16670)
            if n_amino_acid_residues == 2:
                res.append(46761)
            if n_amino_acid_residues == 3:
                res.append(47923)
            if n_amino_acid_residues == 4:
                res.append(48030)
            if n_amino_acid_residues == 5:
                res.append(48545)
            if n_amino_acid_residues >= 10:
                res.append(15841)
            else:
                # oligo
                res.append(25676)
    if "emericellamide" in classification and classification["emericellamide"]:
        res.append(64372)
    if "2,5-diketopiperazines" in classification and classification["2,5-diketopiperazines"]:
        res.append(65061)

    return res


@cli.command()
@click.option('--chebi-version', '-v', type=int, required=True, help='ChEBI version')
@click.option('--molecules', '-m', cls=LiteralOption, default="[]",
              help='List of ChEBI IDs to classify. Default: all ChEBI classes.')
@click.option('--return-chebi-classes', '-c', is_flag=True, help='Return ChEBI classes')
@click.option('--run-name', '-n', type=str, help='Results will be stored at results/%y%m%d_%H%M_{run_name}/')
@click.option('--debug-mode', '-d', is_flag=True, help='Logs at debug level')
@click.option('--additional-output', '-o', is_flag=True, help='Returns intermediate steps in output, '
                                                              'useful for explainability and verification')
def classify(chebi_version, molecules, return_chebi_classes, run_name, debug_mode, additional_output):
    json_logger = TimestampedLogger(None, run_name, debug_mode)
    json_logger.start_run("classify", {"chebi_version": chebi_version, "molecules": molecules,
                                       "return_chebi_classes": return_chebi_classes, "run_name": run_name,
                                       "debug_mode": debug_mode})
    data = ChEBIData(chebi_version).processed
    if len(molecules) > 0:
        data_filtered = data.loc[data.index.isin(molecules)]
    else:
        data_filtered = data

    # start with shortest SMILES
    data_filtered["smiles_length"] = [
        len(str(row["smiles"]) if row["smiles"] is not None else "")
        for _, row in data_filtered.iterrows()
    ]
    data_filtered.sort_values("smiles_length", inplace=True, ascending=True)

    results = []
    logging.info(f"Classifying {len(data_filtered)} molecules")
    for id, row in tqdm.tqdm(data_filtered.iterrows()):
        logging.debug(f"Classifying CHEBI:{id} ({row['name']})")
        start_time = time.perf_counter()
        charge_category = get_charge_category(row["mol"])
        logging.debug(f"Charge category is {charge_category}")
        n_amino_acid_residues, add_output = get_n_amino_acid_residues(row["mol"])
        logging.debug(f"Found {n_amino_acid_residues} amino acid residues")
        if n_amino_acid_residues > 0:
            proteinogenics, proteinogenics_locations, proteinogenics_locations_no_carboxy = get_proteinogenic_amino_acids(row["mol"],
                                                                                     add_output["amino_residue"],
                                                                                     add_output["carboxy_residue"])
        else:
            proteinogenics, proteinogenics_locations, proteinogenics_locations_no_carboxy = [], [], []
        results.append({
            'chebi_id': id,
            'charge_category': charge_category.name,
            'n_amino_acid_residues': n_amino_acid_residues,
            'proteinogenics': proteinogenics,
            'time': f"{time.perf_counter() - start_time:.4f}"
        })

        if n_amino_acid_residues == 5:
            emericellamide = is_emericellamide(row["mol"])
            results[-1]["emericellamide"] = emericellamide[0]
            if additional_output and emericellamide[0]:
                results[-1]["emericellamide_atoms"] = emericellamide[1]
        if n_amino_acid_residues == 2:
            diketopiperazine = is_diketopiperazine(row["mol"])
            results[-1]["2,5-diketopiperazines"] = diketopiperazine[0]
            if additional_output and diketopiperazine[0]:
                results[-1]["2,5-diketopiperazines_atoms"] = diketopiperazine[1]

        if return_chebi_classes:
            results[-1]['chebi_classes'] = resolve_chebi_classes(results[-1])
        if additional_output:
            results[-1] = {**results[-1], **add_output, "proteinogenics_locations": proteinogenics_locations,
                           "proteinogenics_locations_no_carboxy": proteinogenics_locations_no_carboxy}

    json_logger.save_items("classify", results)


@cli.command()
@click.option('--chebi-version', '-v', type=int, required=True, help='ChEBI version')
@click.option('--molecules', '-m', cls=LiteralOption, default="[]",
              help='List of ChEBI IDs to classify. Default: all ChEBI classes.')
@click.option('--return-chebi-classes', '-c', is_flag=True, help='Return ChEBI classes')
@click.option('--run-name', '-n', type=str, help='Results will be stored at results/%y%m%d_%H%M_{run_name}/')
@click.option('--debug-mode', '-d', is_flag=True, help='Logs at debug level')
@click.option('--additional-output', '-o', is_flag=True, help='Returns intermediate steps in output, '
                                                              'useful for explainability and verification')
def classify_fol(chebi_version, molecules, return_chebi_classes, run_name, debug_mode, additional_output):
    json_logger = TimestampedLogger(None, run_name, debug_mode)
    json_logger.start_run("classify_fol", {"chebi_version": chebi_version, "molecules": molecules,
                                       "return_chebi_classes": return_chebi_classes, "run_name": run_name,
                                       "debug_mode": debug_mode})
    data = ChEBIData(chebi_version).processed
    if len(molecules) > 0:
        data_filtered = data.loc[data.index.isin(molecules)]
    else:
        data_filtered = data

    # start with shortest SMILES
    data_filtered["smiles_length"] = [
        len(str(row["smiles"]) if row["smiles"] is not None else "")
        for _, row in data_filtered.iterrows()
    ]
    data_filtered.sort_values("smiles_length", inplace=True, ascending=True)

    charge_verifier = ChargeVerifier()
    functional_groups_verifier = FunctionalGroupsVerifier()
    peptide_size_verifier = PeptideSizeVerifier()
    proteinogenics_verifier = ProteinogenicsVerifier()
    substruct_verifier = SubstructVerifier()

    results = []
    logging.info(f"Classifying {len(data_filtered)} molecules")
    for id, row in tqdm.tqdm(data_filtered.iterrows()):
        logging.debug(f"Classifying CHEBI:{id} ({row['name']})")
        start_time = time.perf_counter()
        add_output = {}
        charge_category, charge_assignments = charge_verifier.classify_charge(row["mol"])
        if charge_assignments is not None:
            add_output["charge_assignments"] = charge_assignments
        logging.debug(f"Charge category is {charge_category}")
        functional_groups = functional_groups_verifier.classify_functional_groups(row["mol"])
        n_amino_acid_residues, size_assignments = peptide_size_verifier.classify_n_amino_acids(row["mol"], functional_groups)
        add_output["n_amino_acid_residues_atoms"] = size_assignments
        logging.debug(f"Found {n_amino_acid_residues} amino acid residues")
        if n_amino_acid_residues > 0:
            atom_level_functional_groups = {
                "amino_residue_n": [amino[0] for amino in functional_groups["amino_residue"]],
                "carboxy_residue_c": [carboxy[0] for carboxy in functional_groups["carboxy_residue"]]}

            proteinogenics, proteinogenics_locations = proteinogenics_verifier.classify_proteinogenics(
                row["mol"], atom_level_functional_groups)
        else:
            proteinogenics, proteinogenics_locations = [], []
        add_output["proteinogenics_locations"] = proteinogenics_locations

        results.append({
            'chebi_id': id,
            'charge_category': charge_category.name,
            'n_amino_acid_residues': n_amino_acid_residues,
            'functional_groups': functional_groups,
            'proteinogenics': proteinogenics,
        })

        if n_amino_acid_residues == 5:
            emericellamide = substruct_verifier.classify_substruct_class(row["mol"], "emericellamide")
            results[-1]["emericellamide"] = emericellamide[0]
            if emericellamide[0]:
                add_output["emericellamide_atoms"] = emericellamide[1]
        if n_amino_acid_residues == 2:
            diketopiperazine = is_diketopiperazine(row["mol"])
            results[-1]["2,5-diketopiperazines"] = diketopiperazine[0]
            if diketopiperazine[0]:
                add_output["2,5-diketopiperazines_atoms"] = diketopiperazine[1]

        results[-1]["time"] = f"{time.perf_counter() - start_time:.4f}"



        if return_chebi_classes:
            results[-1]['chebi_classes'] = resolve_chebi_classes(results[-1])
        if additional_output:
            results[-1] = {**results[-1], **add_output}

    json_logger.save_items("classify_fol", results)


@cli.command()
@click.option('--chebi-version', '-v', type=int, required=True, help='ChEBI version')
@click.option('--results-dir', '-r', type=str, required=True, help='Directory where results.json to analyse is located')
@click.option('--debug-mode', '-d', is_flag=True, help='Returns additional states')
@click.option('--molecules', '-m', cls=LiteralOption, default="[]",
              help='List of ChEBI IDs to verify. Default: all ChEBI classes.')
@click.option('--only-3star', '-3', is_flag=True, help='Only consider 3-star molecules')
def verify(chebi_version, results_dir, debug_mode, molecules, only_3star):
    json_logger = TimestampedLogger(results_dir, debug_mode=debug_mode)
    json_logger.start_run("verify" + ("_3star" if only_3star else ""),
                          {"chebi_version": chebi_version, "results_dir": results_dir, "debug_mode": debug_mode,
                           "molecules": molecules})
    data = ChEBIData(chebi_version)
    with open(os.path.join(results_dir, "classify.json"), "r") as f:
        results = json.load(f)
    charge_verifier = ChargeVerifier()
    functional_groups_verifier = FunctionalGroupsVerifier()
    peptide_size_verifier = PeptideSizeVerifier()
    proteinogenics_verifier = ProteinogenicsVerifier()
    substruct_verifier = SubstructVerifier()
    res = []

    results = [r for r in results if (len(molecules) == 0 or r["chebi_id"] in molecules) and (not only_3star or data.processed.loc[r["chebi_id"], "subset"] == "3_STAR")]

    save_results_at = len(results) / 4

    for i, result in tqdm.tqdm(enumerate(results), total=len(results), desc="Verifying"):
        outcome, expected = {}, {}
        start_time = time.perf_counter()
        expected["charge"] = ChargeCategories[result["charge_category"]].name
        mol = data.processed.loc[result["chebi_id"], "mol"]
        outcome["charge"] = charge_verifier.verify_charge_category(mol, ChargeCategories[result["charge_category"]], {})
        # functional groups
        expected["functional_groups"] = {}
        if "amide_bond" in result:
            expected["functional_groups"]["amide_bond"] = result["amide_bond"]
        else:
            _, amide_c, amide_o, amide_n = get_amide_bonds(mol)
            expected["functional_groups"]["amide_bond"] = [(c, o, n) for c, o, n in zip(amide_c, amide_o, amide_n)]
        if "amino_residue" in result:
            expected["functional_groups"]["amino_residue"] = [(n,) for n in result["amino_residue"]]
        else:
            expected["functional_groups"]["amino_residue"] = [(n,) for n in
                                                get_amino_groups(mol, [c for c, _, _ in expected["functional_groups"]["amide_bond"]])]
        if "carboxy_residue" in result:
            expected["functional_groups"]["carboxy_residue"] = result["carboxy_residue"]
        else:
            expected["functional_groups"]["carboxy_residue"] = list(get_carboxy_derivatives(mol))
        outcome["functional_groups"] = functional_groups_verifier.verify_functional_groups(mol, expected["functional_groups"])

        # n amino acids
        expected["n_amino_acid_residues"] = result["n_amino_acid_residues"]
        if expected["n_amino_acid_residues"] > 1:
            if "longest_aa_chain" in result:
                aars = result["longest_aa_chain"]
            else:
                add_output = get_n_amino_acid_residues(mol)[1]
                aars = add_output["longest_aa_chain"]
            outcome["size"] = peptide_size_verifier.verify_n_plus_amino_acids(
                mol, expected["n_amino_acid_residues"], expected["functional_groups"], {f"A{i}": aar for i, aar in enumerate(aars)}
            )
        else:
            # if there are no amino acids, then there is nothing to prove
            outcome["size"] = ModelCheckerOutcome.MODEL_FOUND_INFERRED, None

        # proteinogenics
        if "proteinogenics_locations_no_carboxy" in result:
            expected["proteinogenics"] = [(code, atoms) for code, atoms in zip(result["proteinogenics"],
                                                                            result["proteinogenics_locations_no_carboxy"])]
        else:
            proteinogenics, _, proteinogenics_locations_no_carboxy = get_proteinogenic_amino_acids(
                mol, [amino[0] for amino in expected["functional_groups"]["amino_residue"]], expected["functional_groups"]["carboxy_residue"]
            )
            expected["proteinogenics"] = [(code, atoms) for code, atoms in zip(proteinogenics, proteinogenics_locations_no_carboxy)]
        if len(expected["proteinogenics"]) > 0:
            # only take first atoms of functional groups
            atom_level_functional_groups = {"amino_residue_n": [amino[0] for amino in expected["functional_groups"]["amino_residue"]],
                                            "carboxy_residue_c": [carboxy[0] for carboxy in expected["functional_groups"]["carboxy_residue"]]}
            outcome["proteinogenics"] = proteinogenics_verifier.verify_proteinogenics(mol, atom_level_functional_groups,
                                                                                      expected["proteinogenics"])
        else:
            outcome["proteinogenics"] = ModelCheckerOutcome.MODEL_FOUND_INFERRED, None

        # substructures
        if "emericellamide" in result and result["emericellamide"]:
            if "emericellamide_atoms" in result:
                atoms = result["emericellamide_atoms"]
            else:
                atoms = is_emericellamide(mol)[1]
            expected["emericellamide"] = True
            outcome["emericellamide"] = substruct_verifier.verify_substruct_class(mol, "emericellamide", atoms)
        if "2,5-diketopiperazines" in result and result["2,5-diketopiperazines"]:
            if "2,5-diketopiperazines_atoms" in result:
                atoms = result["2,5-diketopiperazines_atoms"]
            else:
                atoms = is_diketopiperazine(mol)
            expected["2,5-diketopiperazines"] = True
            outcome["2,5-diketopiperazines"] = substruct_verifier.verify_substruct_class(mol, "diketopiperazines", atoms)

        res.append({
            "chebi_id": result["chebi_id"],
            "expected": expected,
            "outcome": {key: o[0].name for key, o in outcome.items()},
            "time": f"{time.perf_counter() - start_time:.4f}"
        })

        if debug_mode:
            res[-1]["proof_attempts"] = {key: o[1] for key, o in outcome.items() if o[1] is not None}
        if any(o[0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED] for o in
               outcome.values()):
            warning_str = ""
            if outcome["charge"][0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                warning_str += f"Expected charge: {expected["charge"]}, got: {outcome['charge'][0].name}, tried: {outcome['charge'][1]}\n"
            if outcome["functional_groups"][0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                warning_str += f"Expected groups: {expected["functional_groups"]}, got {outcome['functional_groups'][0].name}, tried: {outcome['functional_groups'][1]}\n"
            if outcome["size"][0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                warning_str += f"Expected {expected["n_amino_acid_residues"]} amino acids, got {outcome['size'][0].name}\n"
            if outcome["proteinogenics"][0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                warning_str += f"Expected proteinogenics: {expected["proteinogenics"]}, got {outcome['proteinogenics'][0].name}, tried {outcome['proteinogenics'][1]}\n"
            if "emericellamide" in expected and outcome["emericellamide"][0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                warning_str += f"Expected emericellamide, got {outcome['emericellamide'][0].name}\n"
            if "2,5-diketopiperazines" in expected and outcome["2,5-diketopiperazines"][0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
                warning_str += f"Expected 2,5-diketopiperazines, got {outcome['2,5-diketopiperazines'][0].name}\n"
            logging.warning(f"Verification failed for CHEBI:{result['chebi_id']} \n{warning_str}")

        if i >= save_results_at:
            save_results_at = i + (len(results) - i) // 4 + 10
            json_logger.save_items(f"verify_{json_logger.timestamp}", res)
    json_logger.save_items(f"verify_{json_logger.timestamp}", res)
