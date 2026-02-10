from popper.loop import learn_solution
from popper.util import Settings
import os
import subprocess
import sys
import json
import pickle
import base64

class PopperWrapper:

    def __init__(self):
        # override default settings for Popper
        self.settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls"
        }

    def solve(self, problem_dir):
        settings = Settings(kbpath=problem_dir, **self.settings_parameters)
        prog, score, stats = learn_solution(settings)
        return prog, score, stats
    

    
def run_ilp_training_subprocess(problem_dir, settings_parameters):
    """Run Popper ILP learning in a separate subprocess for isolated Prolog session."""
    script = f'''
import json
import pickle
import base64
from popper.loop import learn_solution
from popper.util import Settings, format_prog

settings = Settings(kbpath=r"{problem_dir}", **{repr(settings_parameters)})
prog, score, stats = learn_solution(settings)
prog_str = format_prog(prog) if prog else None

# Serialize prog object using pickle and base64 encode for JSON transport
prog_pickled = base64.b64encode(pickle.dumps(prog)).decode('ascii') if prog else None

result = {{"prog_pickled": prog_pickled, "prog_str": prog_str, "score": list(score) if score else None}}
print(json.dumps(result))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    if result.returncode != 0:
        raise RuntimeError(f"Popper training failed: {result.stderr}")
    output = json.loads(result.stdout)
    
    # Deserialize the prog object
    if output["prog_pickled"]:
        output["prog"] = pickle.loads(base64.b64decode(output["prog_pickled"]))
    else:
        output["prog"] = None
    
    return output


def run_ilp_validation_subprocess(chebi_id, prog, n_validation_pos, n_validation_neg, problem_dir, settings_parameters):
    """Run Popper validation in a separate subprocess for isolated Prolog session."""
    # Serialize prog object using pickle and base64 encode
    prog_pickled = base64.b64encode(pickle.dumps(prog)).decode('ascii') if prog else ""
    
    script = f'''
import json
import pickle
import base64
from popper.tester import Tester
from popper.util import Settings

# Deserialize prog object
prog_pickled = "{prog_pickled}"
if prog_pickled:
    prog = pickle.loads(base64.b64decode(prog_pickled))
else:
    prog = None

settings = Settings(kbpath=r"{problem_dir}/chebi_{chebi_id}", **{repr(settings_parameters)})
settings.datalog = False
settings.bk_file = r"{problem_dir}/bk_validation.pl"
settings.ex_file = r"{problem_dir}/chebi_{chebi_id}/exs_validation.pl"

if prog:
    ilp_tester = Tester(settings)
    pos_covered, neg_covered = ilp_tester.test_prog_all(prog)
    tp = pos_covered.count(1)
    fn = {n_validation_pos} - tp
    fp = neg_covered.count(1)
    tn = {n_validation_neg} - fp
else:
    tp, fn, fp, tn = 0, {n_validation_pos}, 0, {n_validation_neg}

result = {{"TP": tp, "FN": fn, "TN": tn, "FP": fp}}
print(json.dumps(result))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    if result.returncode != 0:
        raise RuntimeError(f"Popper validation failed: {result.stderr}")
    conf_matrix = json.loads(result.stdout)
    print(f"    Validation set: TP: {conf_matrix['TP']}, FN: {conf_matrix['FN']}, TN: {conf_matrix['TN']}, FP: {conf_matrix['FP']}")
    return conf_matrix

