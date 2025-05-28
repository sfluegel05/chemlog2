import logging
import subprocess
import time
from typing import List

DEPQBF_PATH = "./depqbf" # solver #"/../../Downloads/depqbf-version-6.03/depqbf-version-6.03/depqbf"
BLOQQER_PATH = "./bloqqer" # preprocessor for qbf

def qbf_solver(input: List[str]):
    with open("tmp.qdimacs", "w") as f:
        f.write("\n".join(input))
    bloqqer_start = time.perf_counter()
    logging.debug(f"Preprocessing with Bloqqer")
    res_bloqqer = subprocess.run(
        [BLOQQER_PATH, "tmp.qdimacs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    logging.debug(f"Bloqqer finished in {time.perf_counter() - bloqqer_start:.2f} seconds")

    with open("tmp_bloqqed.qdimacs", "w") as f:
        f.write(res_bloqqer.stdout)

    logging.debug(f"Running DepQBF on preprocessed file")
    depqbf_start = time.perf_counter()
    res = subprocess.run(
        [DEPQBF_PATH, "tmp_bloqqed.qdimacs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    logging.debug(f"DepQBF finished in {time.perf_counter() - depqbf_start:.2f} seconds")
    logging.debug(f"DepQBF output: {res.stdout.strip()}")
    if "UNSAT" in res.stdout:
        return False
    elif "SAT" in res.stdout:
        return True
    else:
        return res.stdout