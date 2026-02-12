from popper.loop import learn_solution
from popper.util import Settings
import os
import subprocess
import sys
import json
import pickle
import base64
from datetime import datetime


def log_stderr(log_dir, phase, stderr_code, stderr_content):
    """Write stderr content to log file with timestamp."""
    if not stderr_content.strip():
        return
    log_file = os.path.join(log_dir, "subprocess.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n[{timestamp}] === {phase} (Return code: {stderr_code}) ===\n")
        f.write(stderr_content)
        f.write("\n")

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
    

    
def run_ilp_training_subprocess(exs_file, bk_file, bias_file, settings_parameters, log_dir=None):
    """Run Popper ILP learning in a separate subprocess for isolated Prolog session."""
    script = f'''
import json
import pickle
import base64
from popper.loop import learn_solution
from popper.util import Settings, format_prog

def make_pickleable(prog):
    # If prog is a dict_values or similar, convert to list
    if hasattr(prog, 'items') or hasattr(prog, 'keys'):
        return dict(prog)
    if type(prog).__name__ == 'dict_values':
        return list(prog)
    return prog

settings = Settings(ex_file=r"{exs_file}", bk_file=r"{bk_file}", bias_file=r"{bias_file}", **{repr(settings_parameters)})
prog, score, stats = learn_solution(settings)


prog_str = format_prog(prog) if prog else None
# Serialize prog object using pickle and base64 encode for JSON transport
prog_pickleable = make_pickleable(prog) if prog else None
prog_pickled = base64.b64encode(pickle.dumps(prog_pickleable)).decode('ascii') if prog_pickleable else None

result = {{"prog_pickled": prog_pickled, "prog_str": prog_str, "score": list(score) if score else None}}
print(json.dumps(result))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        start_new_session=True,  # Start in a new session to isolate from parent process
        cwd=os.getcwd(),
    )
    if log_dir:
        log_stderr(log_dir, f"Training: {bias_file}", result.returncode, result.stderr)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    stdout_lines = result.stdout.strip().split('\n')
    output = json.loads(stdout_lines[-1])
    
    # Deserialize the prog object
    if output["prog_pickled"]:
        output["prog"] = pickle.loads(base64.b64decode(output["prog_pickled"]))
    else:
        output["prog"] = None
    
    return output


def run_ilp_validation_subprocess(chebi_id, prog, problem_dir, predicate_set, settings_parameters, log_dir=None):
    """Run Popper validation in a separate subprocess for isolated Prolog session."""
    # Serialize prog object using pickle and base64 encode
    prog_pickled = base64.b64encode(pickle.dumps(prog)).decode('ascii') if prog else ""

    exs_file = f"{problem_dir}/chebi_{chebi_id}/exs_validation.pl"
    with open(exs_file, "r") as f:
        exs_content = f.read()
        # count pos and neg examples
        n_validation_pos = exs_content.count("pos(")
        n_validation_neg = exs_content.count("neg(")
    
    script = f'''
import json
import pickle
import base64
from popper.tester import Tester
from popper.util import Settings

def make_pickleable(prog):
    if hasattr(prog, 'items') or hasattr(prog, 'keys'):
        return dict(prog)
    if type(prog).__name__ == 'dict_values':
        return list(prog)
    return prog

# Deserialize prog object
prog_pickled = "{prog_pickled}"
if prog_pickled:
    prog = pickle.loads(base64.b64decode(prog_pickled))
    prog = make_pickleable(prog)
else:
    prog = None

settings = Settings(bk_file=r"{problem_dir}/{predicate_set}/bk_validation.pl", ex_file=r"{problem_dir}/chebi_{chebi_id}/exs_validation.pl", **{repr(settings_parameters)})
settings.datalog = False

if prog:
    ilp_tester = Tester(settings)
    print(bk_file, exs_file)
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
    # Get timeout from settings_parameters (default 60 seconds for validation)
    timeout = settings_parameters.get("timeout", 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        if log_dir:
            log_stderr(log_dir, f"Validation: chebi_{chebi_id}", -1, f"Validation timed out after {timeout} seconds")
        print(f"    Validation timed out after {timeout} seconds")
        return {"TP": 0, "FN": n_validation_pos, "TN": n_validation_neg, "FP": 0, "timeout": True}
    
    if log_dir:
        log_stderr(log_dir, f"Validation: chebi_{chebi_id}", result.returncode, result.stderr)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    stdout_lines = result.stdout.strip().split('\n')
    conf_matrix = json.loads(stdout_lines[-1])
    print(f"    Validation set: TP: {conf_matrix['TP']}, FN: {conf_matrix['FN']}, TN: {conf_matrix['TN']}, FP: {conf_matrix['FP']}")
    return conf_matrix
