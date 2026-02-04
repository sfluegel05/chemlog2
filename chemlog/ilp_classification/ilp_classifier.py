from popper.loop import learn_solution
from popper.util import Settings

class PopperWrapper:

    def __init__(self):
        # override default settings for Popper
        self.settings_parameters = {
            "noisy": True
        }

    def solve(self, problem_dir):
        settings = Settings(kbpath=problem_dir, **self.settings_parameters)
        prog, score, stats = learn_solution(settings)
        return prog, score, stats