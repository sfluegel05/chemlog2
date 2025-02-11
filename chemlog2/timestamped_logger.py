import json
import logging
import os
import sys
import time
from typing import List, Optional

import yaml


class TimestampedLogger:

    def __init__(
            self,
            results_dir=None,
            run_name: Optional[str] = None,
            debug_mode: bool = False,
    ):
        self.timestamp = time.strftime("%y%m%d_%H%M", time.localtime())
        if results_dir is None:
            run_name = f'{self.timestamp}{"_" + run_name if run_name is not None else ""}'
            results_dir = os.path.join("results", run_name)
        self.results_dir = results_dir
        os.makedirs(os.path.join(results_dir), exist_ok=True)
        logging.basicConfig(
            format="[%(filename)s:%(lineno)s] %(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG if debug_mode else logging.INFO,
            handlers=[logging.FileHandler(os.path.join(results_dir, "logs.log"), encoding="utf-8"),
                      logging.StreamHandler(sys.stdout)],
        )
        self.skip_newline = False

    def start_run(self, command, config):
        with open(os.path.join(self.results_dir, f"{command}_config.yaml"), 'w') as f:
            yaml.dump(config, f)

    def save_items(self, command, items):
        with open(os.path.join(self.results_dir, f"{command}.json"), 'w') as f:
            json.dump(items, f, indent=2)
