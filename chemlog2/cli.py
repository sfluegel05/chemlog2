import json
import time

import click
import tqdm

from chemlog2.classification.charge_classifier import get_charge_category, ChargeCategories
from chemlog2.classification.peptide_size_classifier import get_n_amino_acid_residues
from chemlog2.preprocessing.chebi_data import ChEBIData
import logging
import os
import ast
import sys

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
    n_amino_acid_residues = classification["n_amino_acid_residues"]
    charge_category = classification["charge_category"]
    if charge_category == ChargeCategories.SALT.name:
        return [24866] # salt
    if n_amino_acid_residues < 2:
        if charge_category == ChargeCategories.ANION.name:
            return ["organic anion"]
        if charge_category == ChargeCategories.CATION.name:
            return ["organic cation"]
        if charge_category == ChargeCategories.ZWITTERION.name:
            return ["zwitterion"]
        return []
    if charge_category == ChargeCategories.ANION.name:
        return ["peptide anion"]
    if charge_category == ChargeCategories.CATION.name:
        return ["peptide cation"]
    if charge_category == ChargeCategories.ZWITTERION.name:
        if n_amino_acid_residues == 2:
            return ["dizwitter "]
        if n_amino_acid_residues == 3:
            return ["trizwitter"]
        return ["peptide zwitterion"]
    if n_amino_acid_residues == 2:
        return ["di"]



@cli.command()
@click.option('--chebi-version', '-v', type=int, required=True, help='ChEBI version')
@click.option('--molecules', '-m', cls=LiteralOption, default="[]",
              help='List of ChEBI IDs to classify. Default: all ChEBI classes.')
@click.option('--return-chebi-classes', '-c', is_flag=True, help='Return ChEBI classes')
@click.option('--run-name', '-n', type=str, help='Results will be stored at results/%y%m%d_%H%M_{run_name}/')
def classify(chebi_version, molecules, return_chebi_classes, run_name):
    run_name = f'{time.strftime("%y%m%d_%H%M", time.localtime())}{"_" + run_name if run_name is not None else ""}'
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    logging.basicConfig(
        format="[%(filename)s:%(lineno)s] %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.WARNING,
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
        n_amino_acid_residues = get_n_amino_acid_residues(row["mol"])
        logging.debug(f"Found {n_amino_acid_residues} amino acid residues")
        results.append({
            'chebi_id': id,
            'charge_category': charge_category.name,
            'n_amino_acid_residues': n_amino_acid_residues
        })
        if return_chebi_classes:
            results[-1]['chebi_classes'] = resolve_chebi_classes(results[-1])
        with open(os.path.join("results", run_name, "results.json"), 'a') as f:
            if skip_newline:
                skip_newline = False
            else:
                f.write(",\n")
            json.dump(results[-1], f, indent=4)
    with open(os.path.join("results", run_name, "results.json"), 'a') as f:
            f.write("]")
