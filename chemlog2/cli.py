import json
import time

import click
import tqdm

from chemlog2.classification.charge_classifier import get_charge_category, ChargeCategories
from chemlog2.classification.peptide_size_classifier import get_n_amino_acid_residues
from chemlog2.classification.proteinogenics_classifier import get_proteinogenic_amino_acids
from chemlog2.preprocessing.chebi_data import ChEBIData
from chemlog2.verification.charge_verifier import ChargeVerifier
import logging
import os
import ast
import sys

from chemlog2.verification.charge_verifier import ChargeVerifier
from chemlog2.verification.model_checking import ModelCheckerOutcome


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
    if charge_category == ChargeCategories.SALT.name:
        return [24866] # salt (there is no class peptide salt)
    if n_amino_acid_residues < 2:
        # if not a peptide: only assign charge classes
        if charge_category == ChargeCategories.ANION.name:
            return [25696]
        if charge_category == ChargeCategories.CATION.name:
            return [25697]
        if charge_category == ChargeCategories.ZWITTERION.name:
            return [27369]
        return []
    # peptide ... classes
    if charge_category == ChargeCategories.ANION.name:
        # anion, peptide anion
        return [25696, 60334]
    if charge_category == ChargeCategories.CATION.name:
        return [25697, 60194]
    if charge_category == ChargeCategories.ZWITTERION.name:
        if n_amino_acid_residues == 2:
            # zwitterion, peptide zwitterion, dipeptide zwitterion
            return [27369, 60466, 90799]
        if n_amino_acid_residues == 3:
            return [27369, 60466, 155837]
        return [27369, 60466]
    if n_amino_acid_residues == 2:
        return [16670, 25676, 46761]
    if n_amino_acid_residues == 3:
        return [16670, 25676, 47923]
    if n_amino_acid_residues == 4:
        return [16670, 25676, 48030]
    if n_amino_acid_residues == 5:
        return [16670, 25676, 48545]
    if n_amino_acid_residues >= 10:
        return [16670, 15841]
    # only oligo
    return [16670, 25676]



@cli.command()
@click.option('--chebi-version', '-v', type=int, required=True, help='ChEBI version')
@click.option('--molecules', '-m', cls=LiteralOption, default="[]",
              help='List of ChEBI IDs to classify. Default: all ChEBI classes.')
@click.option('--return-chebi-classes', '-c', is_flag=True, help='Return ChEBI classes')
@click.option('--run-name', '-n', type=str, help='Results will be stored at results/%y%m%d_%H%M_{run_name}/')
@click.option('--debug-mode', '-d', is_flag=True, help='Returns additional states, logs at debug level')
def classify(chebi_version, molecules, return_chebi_classes, run_name, debug_mode):
    run_name = f'{time.strftime("%y%m%d_%H%M", time.localtime())}{"_" + run_name if run_name is not None else ""}'
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    logging.basicConfig(
        format="[%(filename)s:%(lineno)s] %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if debug_mode else logging.WARNING,
        handlers=[logging.FileHandler(os.path.join("results", run_name, "logs.log"), encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    with open(os.path.join("results", run_name, "results.json"), 'a') as f:
        f.write("[\n")
    # todo dump config

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
    skip_newline = True
    for id, row in tqdm.tqdm(data_filtered.iterrows()):
        logging.debug(f"Classifying CHEBI:{id} ({row['name']})")
        charge_category = get_charge_category(row["mol"])
        logging.debug(f"Charge category is {charge_category}")
        n_amino_acid_residues, add_output = get_n_amino_acid_residues(row["mol"])
        logging.debug(f"Found {n_amino_acid_residues} amino acid residues")
        if n_amino_acid_residues > 0:
            proteinogenics, proteinogenics_locations = get_proteinogenic_amino_acids(row["mol"], add_output["aminos"],
                                                                                 add_output["carboxy_cs"])
        else:
            proteinogenics, proteinogenics_locations = [], []
        results.append({
            'chebi_id': id,
            'charge_category': charge_category.name,
            'n_amino_acid_residues': n_amino_acid_residues,
            'proteinogenics': proteinogenics,
        })

        if return_chebi_classes:
            results[-1]['chebi_classes'] = resolve_chebi_classes(results[-1])
        if debug_mode:
            results[-1] = {**results[-1], **add_output, "proteinogenics_locations": proteinogenics_locations}

        with open(os.path.join("results", run_name, "results.json"), 'a') as f:
            if skip_newline:
                skip_newline = False
            else:
                f.write(",\n")
            json.dump(results[-1], f, indent=4)
    with open(os.path.join("results", run_name, "results.json"), 'a') as f:
            f.write("]")

@cli.command()
@click.option('--chebi-version', '-v', type=int, required=True, help='ChEBI version')
@click.option('--results-dir', '-r', type=str, required=True, help='Directory where results.json to analyse is located')
@click.option('--debug-mode', '-d', is_flag=True, help='Returns additional states')
@click.option('--molecules', '-m', cls=LiteralOption, default="[]",
              help='List of ChEBI IDs to classify. Default: all ChEBI classes.')
def verify(chebi_version, results_dir, debug_mode, molecules):
    timestamp = time.strftime("%y%m%d_%H%M", time.localtime())
    logging.basicConfig(
        format="[%(filename)s:%(lineno)s] %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if debug_mode else logging.WARNING,
        handlers=[logging.FileHandler(
            os.path.join(results_dir, f"logs_verify_{timestamp}.log"),
            encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    data = ChEBIData(chebi_version)
    with open(os.path.join(results_dir, "results.json"), "r") as f:
        results = json.load(f)
    verifier = ChargeVerifier()
    res = []

    for result in tqdm.tqdm(results):
        if len(molecules) > 0 and result["chebi_id"] not in molecules:
            continue
        expected_charge = ChargeCategories[result["charge_category"]]
        mol = data.processed.loc[result["chebi_id"], "mol"]
        verified = verifier.verify_charge_category(mol, expected_charge, {})
        res.append({
            "chebi_id": result["chebi_id"],
            "expected_charge": expected_charge.name,
            "outcome": verified[0].name
        })
        if debug_mode:
            res[-1]["proof_attempts"] = verified[1]
        if verified[0] not in [ModelCheckerOutcome.MODEL_FOUND, ModelCheckerOutcome.MODEL_FOUND_INFERRED]:
            logging.warning(f"Verification failed for CHEBI:{result['chebi_id']} (expected: {expected_charge.name})")
    with open(os.path.join(results_dir, f"verification_fol_{timestamp}.json"), 'w') as f:
        json.dump(res, f, indent=4)


