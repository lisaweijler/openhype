import os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(6)

from dacite import from_dict
from openhype.utils.io import convert_dictconf_to_dict, load_config

from pathlib import Path
import tyro
from openhype.openhype_eval.evaluator_registry import (
    EVALUATOR_REGISTRY,
    EVALUATOR_CONFIG_REGISTRY,
)
from openhype.variable_registry import GlobalVars

import torch

torch.autograd.set_detect_anomaly(True)
MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


def main(config_path: str, evaluator: str, device: str = "cuda:0"):

    config_path = Path(config_path)
    config = load_config(config_path)

    for eval_config in config.eval_configs:
        eval_config = convert_dictconf_to_dict(eval_config)
        eval_config = from_dict(
            data_class=EVALUATOR_CONFIG_REGISTRY[evaluator], data=eval_config
        )

        evaluator_instance = EVALUATOR_REGISTRY[evaluator](eval_config, device=device)
        evaluator_instance()


if __name__ == "__main__":

    tyro.cli(main)
