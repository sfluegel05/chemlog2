import logging
import subprocess
import time
from typing import List
import os

DEPQBF_PATH = "./depqbf" # solver #"/../../Downloads/depqbf-version-6.03/depqbf-version-6.03/depqbf"
BLOQQER_PATH = "./bloqqer" # preprocessor for qbf
CAQE_PATH = "./caqe" # alternative solver (can call bloqqer internally)

def qbf_solver_depqbf(input: List[str]):
    pid = os.getpid()
    with open(os.path.join("tmp", f"{pid}.qdimacs"), "w") as f:
        f.write("\n".join(input))
    bloqqer_start = time.perf_counter()
    logging.debug(f"Preprocessing with Bloqqer")
    res_bloqqer = subprocess.run(
        [BLOQQER_PATH, os.path.join("tmp", f"{pid}.qdimacs")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    with open(os.path.join("tmp", f"{pid}_bloqqed.qdimacs"), "w") as f:
        f.write(res_bloqqer.stdout)

    logging.debug(f"Bloqqer finished in {time.perf_counter() - bloqqer_start:.2f} seconds, starting DepQBF")
    depqbf_start = time.perf_counter()
    res = subprocess.run(
        [DEPQBF_PATH, os.path.join("tmp", f"{pid}_bloqqed.qdimacs")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    logging.debug(f"DepQBF finished with {res.stdout.strip()} in {time.perf_counter() - depqbf_start:.2f} seconds")
    if "UNSAT" in res.stdout:
        return False
    elif "SAT" in res.stdout:
        return True
    else:
        logging.warning(f"DepQBF failed with {res.stdout.strip()}")
        return res.stdout


def qbf_solver_caqe(input: List[str]):
    """
    Runs the CAQE solver (https://github.com/ltentrup/caqe) on the given input.
    :param input: The input to the solver.
    :return: The result of the solver.
    """
    pid = os.getpid()
    with open(os.path.join("tmp", f"{pid}.qdimacs"), "w") as f:
        f.write("\n".join(input))

    logging.debug(f"Running CAQE (with Bloqqer preprocessor)")
    res = subprocess.run(
        ["./caqe", "--preprocessor", "bloqqer", os.path.join("tmp", f"{pid}.qdimacs")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if "Unsatisfiable" in res.stdout:
        return False
    elif "Satisfiable" in res.stdout:
        return True
    else:
        logging.warning(f"CAQE failed: {res.stdout.strip()}")
        return res.stdout