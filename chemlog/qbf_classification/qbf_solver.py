
import subprocess
from typing import List

DEPQBF_PATH = "./depqbf" # solver #"/../../Downloads/depqbf-version-6.03/depqbf-version-6.03/depqbf"
BLOQQER_PATH = "./bloqqer" # preprocessor for qbf

def qbf_solver(input: List[str]):
    with open("tmp.qdimacs", "w") as f:
        f.write("\n".join(input))
    res_bloqqer = subprocess.run(
        [BLOQQER_PATH, "tmp.qdimacs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    with open("tmp_bloqqed.qdimacs", "w") as f:
        f.write(res_bloqqer.stdout)

    res = subprocess.run(
        [DEPQBF_PATH, "tmp_bloqqed.qdimacs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if "UNSAT" in res.stdout:
        return False
    elif "SAT" in res.stdout:
        return True
    else:
        return res.stdout