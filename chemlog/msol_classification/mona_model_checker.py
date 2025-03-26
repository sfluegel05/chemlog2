import logging
import subprocess

from chemlog.fol_classification.model_checking import ModelCheckerOutcome, AbstractModelChecker


class MonaModelChecker(AbstractModelChecker):

    def __init__(
        self,
        universe: int,
        predicate_extensions: str,
        predicate_definitions: str = None,
        **kwargs,
    ):
        if predicate_definitions is None:
            predicate_definitions = ""
        super().__init__(universe, predicate_extensions, predicate_definitions)

    def find_model(self, formula: str, timeout=0):
        with open("tmp.mona", "w") as f:
            f.write(self.extensions + self.definitions + formula)
        res = subprocess.run(
            ["mona", "-t", "-q", "tmp.mona"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        logging.debug(res.stdout)
        if f"A satisfying example of least length" not in res.stdout:
            if "unsatisfiable" in res.stdout:
                logging.info("Formula is unsatisfiable")
                return ModelCheckerOutcome.NO_MODEL, None
            else:
                logging.warning("An error occurred: " + res.stdout.strip("\n"))
                return ModelCheckerOutcome.ERROR, None
        lines = res.stdout.split("\n")
        total_time = [line for line in lines if "Total time" in line]
        if len(total_time) > 0:
            total_time = total_time[0][13:]
        else:
            total_time = "?"
        logging.info(f"Model found (internal time: {total_time})")
        model_size = res.stdout.split("A satisfying example of least length (")[
            1
        ].split(")")[0]
        if int(model_size) != self.universe:
            logging.warning(
                f"Unexpected model size: {model_size} (expected: {self.universe})"
            )
        allocations = []
        for line in lines:
            # only look at allocations made by the model checker
            if " = " in line:
                pred, atoms = line.split(" = ")
                if pred not in self.definitions and pred not in self.extensions:
                    allocations.append((pred, atoms))
        return ModelCheckerOutcome.MODEL_FOUND, allocations
