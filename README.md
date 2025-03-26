ChemLog is a framework for rule-based ontology extension. 
This repository implements a classification of peptides on the ChEBI and PubChem datasets.

3 methods for classification are implemented: 
1. Using Monadic Second-Order Logic (MSOL) formulas and the MSOL reasoner [MONA](https://www.brics.dk/mona/index.html)
2. Using First-Order Logic (FOL) formulas and a custom FOL model checker
3. Using an algorithmic implementation

The classification covers the following aspects:
1. Number of amino acids (in MSOL / FOL: up to 10)
2. Charge category (either salt, anion, cation, zwitterion or neutral)
3. Proteinogenic amino acids present

If the corresponding flag is set, ChemLog will also return the ChEBI classes that match this classification. Currently supported are:

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



All implementations are based on the same natural language definitions and have been developed jointly. Therefore, it is expected that all methods yield the same result. If you make a different experience, please open an issue. If you are just interested in the results, we recommend using the algorithmic implementation, as it is the fastest one.

If you face problems using ChemLog or have other questions, feel free to open an issue.

## Installation

Download the source code from this repository.

Install with
```
pip install .
```

If you want to use the MONA reasoner, you have to install it separately (the classifier expects the `mona` command to be available).

## Run the classification

ChemLog provides a command line interface for the classification. Results are in JSON format for each run, alongside a log and a config file.

**Command**: 
  
    python -m chemlog classify

  Apply the algorithmic implementation to ChEBI data.

Options:

    -v, --chebi-version INTEGER  ChEBI version  [required]
    -m, --molecules TEXT         List of ChEBI IDs to classify. Default: all
                                 ChEBI classes.
    -c, --return-chebi-classes   Return ChEBI classes
    -n, --run-name TEXT          Results will be stored at
                                 results/%y%m%d_%H%M_{run_name}/
    -d, --debug-mode             Logs at debug level
    -o, --additional-output      Returns intermediate steps in output, useful
                                 for explainability and verification
    -3, --only-3star             Only consider 3-star molecules
    --help                       Show this message and exit.

**Command**: 

    python -m chemlog classify-pubchem

  Apply the algorithmic implementation to PubChem data.

Options:

    -f, --from-batch INTEGER    Start at this PubChem batch (each batch consists of 500,000 ids)
    -t, --to-batch INTEGER      End at this PubChem batch (exclusive)
    -c, --return-chebi-classes  Return assigned ChEBI classes
    -m, --molecules TEXT        List of PubChem IDs to classify. Default: all
                                PubChem entries.
    --help                      Show this message and exit.

**Command**: 

    python -m chemlog classify-fol

  Apply the FOL implementation to PubChem data.

Options:

    -v, --chebi-version INTEGER  ChEBI version  [required]
    -m, --molecules TEXT         List of ChEBI IDs to classify. Default: all
                                 ChEBI classes.
    -c, --return-chebi-classes   Return ChEBI classes
    -n, --run-name TEXT          Results will be stored at
                                 results/%y%m%d_%H%M_{run_name}/
    -d, --debug-mode             Logs at debug level
    -o, --additional-output      Returns intermediate steps in output, useful
                                 for explainability and verification
    -3, --only-3star             Only consider 3-star molecules
    --help                       Show this message and exit.

**Command**: 

    python -m chemlog classify-msol

  Apply the MSOL implementation to PubChem data.

Options:

    -v, --chebi-version INTEGER  ChEBI version  [required]
    -m, --molecules TEXT         List of ChEBI IDs to classify. Default: all
                                 ChEBI classes.
    -n, --run-name TEXT          Results will be stored at
                                 results/%y%m%d_%H%M_{run_name}/
    -d, --debug-mode             Logs at debug level
    -p, --only-peptides          Only consider peptide molecules
    --help                       Show this message and exit.

**Command**: 

    python -m chemlog verify

  Given a results file, run the FOL classification for the same classes. This is typically used to check if the algorithmic and FOL classifications match for certain classes.

Options:
   
    -v, --chebi-version INTEGER  ChEBI version  [required]
    -r, --results-dir TEXT       Directory where results.json to analyse is
                                 located  [required]
    -d, --debug-mode             Returns additional states
    -m, --molecules TEXT         List of ChEBI IDs to verify. Default: all ChEBI
                                 classes.
    -3, --only-3star             Only consider 3-star molecules
    --help                       Show this message and exit.


