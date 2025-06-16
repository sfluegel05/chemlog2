ChemLog is a framework for rule-based ontology extension. 
This repository implements a classification of peptides on the ChEBI and PubChem datasets.


## How are peptides classified?

4 methods for classification are implemented: 
1. Using Monadic Second-Order Logic (MSOL) formulas with the MSOL model finder [MONA](https://www.brics.dk/mona/index.html)
2. Turning an MSOL model finding problem into a QBF satisfiability problem and solving that with [CAQE](https://github.com/ltentrup/caqe/tree/master) or [DepQBF](https://github.com/lonsing/depqbf), using the [Bloqqer](https://fmv.jku.at/bloqqer/) preprocessor.
3. Turning an MOSL model finding problem partially into First-Order Logic (FOL) and solving that with a custom FOL model checker (since not all MSOL axioms are translatable, the non-translatable parts are calculated algorithmically).
4. Using an algorithmic implementation

If you are just interested in the results, we recommend choosing the algorithmic implementation, as it is the fastest and can handle complex molecules.

The classification covers the following aspects:
1. Number of amino acids (up to 10, except for the algorithmic method, which covers arbitrary sizes)
2. Charge category (either salt, anion, cation, zwitterion or neutral)
3. Proteinogenic amino acids present
4. Emericellamides and 2,5-diketopiperazines

ChemLog will also return the ChEBI classes that match this classification. Currently supported are:

| ChEBI ID | name |
| --- | --- |
| 16670 | peptide |
| 60194 | peptide cation |
| 60334 | peptide anion |
| 60466 | peptide zwitterion |
| 25676 | oligopeptide |
| 46761 | dipeptide |
| 47923 | tripeptide |
| 48030 | tetrapeptide |
| 48545 | pentapeptide |
| 15841 | polypeptide |
| 90799 | dipeptide zwitterion |
| 155837 | tripeptide zwitterion |
| 64372 | emericellamide |
| 65061 | 2,5-diketopiperazines |
| 24866 | salt |
| 25696 | organic anion |
| 25697 | organic cation |
| 27369 | zwitterion |



All implementations are based on the same natural language definitions and have been developed jointly. Therefore, it is expected that all methods yield the same result. If you make a different experience, please open an issue.

If you face problems using ChemLog or have other questions, feel free to open an issue as well.

## Installation

Download the source code from this repository.

Install with
```
pip install .
```

If you want to use the MONA reasoner, you have to install it separately (the classifier expects the `mona` command to be available).

## Run the classification

ChemLog provides a command line interface for the classification. Results are in JSON format for each run, alongside a log and a config file. Currently, classification of ChEBI and PubChem data is supported. Download and preprocessing of the data are handled automatically. For instances, the following command classifies the 1,000 smallest peptides in ChEBI with the algorithmic method:
    
    python -m chemlog classify-chebi --chebi-version 239 --strategy algo --only-peptides --n-molecules 1000

For more details on the available command line options run

    python -m chemlog --help

