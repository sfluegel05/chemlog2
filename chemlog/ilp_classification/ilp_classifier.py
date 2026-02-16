from popper.loop import learn_solution, get_bk_cons, timeout, popper
from popper.tester import Tester, format_literal_janus, deduce_neg_example_recalls
from popper.util import Settings, format_prog, Literal
import os
import subprocess
import sys
import json
import pickle
import base64
from datetime import datetime
import time
from janus_swi import consult, query_once
from bitarray.util import ones

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


def reload_tester(tester, settings):
        exs_pl_path = settings.ex_file

        if not settings.pi_enabled:
            consult('prog', f':- dynamic {settings.head_literal.predicate}/{len(settings.head_literal.arguments)}.')

        for x in [exs_pl_path]:
            if os.name == 'nt': # if on Windows, SWI requires escaped directory separators
                x = x.replace('\\', '\\\\')
            consult(x)

        query_once('load_examples')

        neg_literal = Literal('neg_fact', tuple(range(len(settings.head_literal.arguments))))
        tester.neg_fact_str = format_literal_janus(neg_literal)
        tester.neg_literal_set = frozenset([neg_literal])

        q = 'findall(_Atom2, (neg_index(_K, _Atom1), term_string(_Atom1, _Atom2)), S)'
        res = query_once(q)['S']
        atoms = []
        for x in res:
            x = x[:-1].split('(')[1].split(',')
            atoms.append(x)

        if atoms:
            try:
                settings.recall = settings.recall | deduce_neg_example_recalls(settings, atoms)
            except Exception as e:
                print(e)

        tester.num_pos = query_once('findall(_K, pos_index(_K, _Atom), _S), length(_S, N)')['N']
        tester.num_neg = query_once('findall(_K, neg_index(_K, _Atom), _S), length(_S, N)')['N']

        print(f"Reloaded tester with {tester.num_pos} positive and {tester.num_neg} negative examples.")

        tester.pos_examples_ = ones(tester.num_pos)

        tester.cached_pos_covered = {}
        tester.cached_inconsistent = {}

        if tester.settings.recursion_enabled:
            query_once(f'assert(timeout({tester.settings.eval_timeout})), fail')


class PopperWrapper:

    def __init__(self, settings_parameters):
        # override default settings for Popper
        self.settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls",
        }
        self.settings_parameters.update(settings_parameters)
        self.settings = None
        self.tester = None

    def solve(self, chebi_id, exs_file, bk_file, bias_file):
        t1 = time.time()

        if self.settings is None:
            self.settings = Settings(ex_file=exs_file, bk_file=bk_file, bias_file=bias_file, **self.settings_parameters)
            self.settings.nonoise = not self.settings.noisy
            self.settings.datalog = False

            with self.settings.stats.duration('load data'):
                self.tester = Tester(self.settings)

        else:
            # override head_pred
            self.settings.head_pred = f"chebi_{chebi_id}"
            # Clear cache
            from janus_swi import query_once
            query_once("(retractall(pos(_)) ; abolish(pos/1), true)")
            query_once("(retractall(pos_index(_, _)) ; abolish(pos_index/2), true)")
            query_once("(retractall(neg(_)) ; abolish(neg/1), true)")
            query_once("(retractall(neg_index(_, _)) ; abolish(neg_index/2), true)")
            query_once("(retractall(neg_fact(_, _)) ; abolish(neg_fact/2), true)")
            num_pos = query_once('findall(_K, pos_index(_K, _Atom), _S), length(_S, N)')['N']
            num_neg = query_once('findall(_K, neg_index(_K, _Atom), _S), length(_S, N)')['N']
            print(f"Reset cache to {num_pos} positive and {num_neg} negative examples.") # should be 0 after retracting all examples

            # set ex_file 
            self.settings.ex_file = exs_file
            # reload tester with new examples
            reload_tester(self.tester, self.settings)


        # learn_solution
        self.settings.solution_found = False
        bkcons = get_bk_cons(self.settings, self.tester)
        self.settings.datalog = False
        time_so_far = time.time()-t1
        timeout(self.settings, popper, (self.settings, self.tester, bkcons), timeout_duration=int(self.settings.timeout-time_so_far),)
        prog_str = format_prog(self.settings.solution) if self.settings.solution else None
        return {
            "prog": self.settings.solution, 
            "prog_str": prog_str, 
            "score": list(self.settings.best_prog_score) if self.settings.best_prog_score else None
            }
    

    
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
        # save to file
        with open(os.path.join(log_dir,"learned_program.pkl"), "wb") as f:
            pickle.dump(output["prog"], f)
    else:
        output["prog"] = None
    
    return output


def run_ilp_validation_subprocess(chebi_id, prog, exs_file, bk_file, bias_file, settings_parameters, log_dir=None):
    """Run Popper validation in a separate subprocess for isolated Prolog session."""
    # Serialize prog object using pickle and base64 encode

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

with open(os.path.join(log_dir, "learned_program.pkl"), "rb") as f:
        prog = pickle.load(f)

print(f"Deserialized prog", prog)
print(bk_file, exs_file, bias_file)
settings = Settings(bk_file=r"{bk_file}", ex_file=r"{exs_file}", bias_file=r"{bias_file}", **{repr(settings_parameters)})
settings.datalog = False

if prog:
    ilp_tester = Tester(settings)
    print(bk_file, exs_file)
    pos_covered, neg_covered = ilp_tester.test_prog_all(prog)
    print(pos_covered, neg_covered)
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
    
    try:
        stdout_lines = result.stdout.strip().split('\n')
        conf_matrix = json.loads(stdout_lines[-1])
        print("stdout_lines:", stdout_lines)
    except json.decoder.JSONDecodeError:
        conf_matrix = {"TP": 0, "FN": n_validation_pos, "TN": n_validation_neg, "FP": 0}
        if log_dir:
            log_stderr(log_dir, f"Validation: chebi_{chebi_id}", result.returncode, f"Failed to parse JSON output. Raw stdout:\n{result.stdout}")
        print(f"    Failed to parse JSON output. See logs for details.")
        return conf_matrix
    if log_dir:
        log_stderr(log_dir, f"Validation: chebi_{chebi_id}", result.returncode, result.stderr)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    print(f"    Validation set: TP: {conf_matrix['TP']}, FN: {conf_matrix['FN']}, TN: {conf_matrix['TN']}, FP: {conf_matrix['FP']}")
    return conf_matrix


if __name__ == "__main__":
    # Example usage
    log_dir = os.path.join("ilp", "results", "run_20260216_095350")
    exs_file = "ilp/chebi_v244/chebi_23824/exs_validation.pl"
    bk_file = "ilp/chebi_v244/atoms/bk_validation.pl"
    bias_file = "ilp/chebi_v244/chebi_23824/atoms/bias_max_vars=6_max_body=6.pl"
    n_validation_pos = 100
    n_validation_neg = 100
    settings_parameters = {"timeout": 60}
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

    with open(os.path.join(log_dir, "learned_program.pkl"), "rb") as f:
        prog = pickle.load(f)

    print(f"Deserialized prog", prog)
    print(bk_file, exs_file, bias_file)
    settings = Settings(bk_file=f"{bk_file}", ex_file=f"{exs_file}", bias_file=f"{bias_file}", **settings_parameters)
    settings.datalog = False

    if prog:
        ilp_tester = Tester(settings)
        print(bk_file, exs_file)
        pos_covered, neg_covered = ilp_tester.test_prog_all(prog)
        print(pos_covered, neg_covered)
        tp = pos_covered.count(1)
        fn = n_validation_pos - tp
        fp = neg_covered.count(1)
        tn = n_validation_neg - fp
    else:
        tp, fn, fp, tn = 0, n_validation_pos, 0, n_validation_neg

    result = {{"TP": tp, "FN": fn, "TN": tn, "FP": fp}}
    print(json.dumps(result))