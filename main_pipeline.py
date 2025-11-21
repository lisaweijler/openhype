import os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(6)
from pathlib import Path
from openhype.utils.io import load_config
from openhype.task_registry import TASK_REGISTRY
from openhype.variable_registry import GlobalVars
import tyro
import torch

torch.autograd.set_detect_anomaly(True)
MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


def main(config_path: str, output_dpath: str, return_global_vars: bool = False):
    config_path = Path(config_path)
    output_dpath = Path(output_dpath)
    global_vars = GlobalVars()
    configs = load_config(config_path)
    img_dir = configs.general.get("img_dir")
    for task in configs.execute:
        print(f"Working on task: '{task}'...")
        task_config = configs.get(task)
        stage = task_config.get("stage")

        task_output_dpath = output_dpath / f"{stage}_{task}"

        if not task_output_dpath.exists():
            task_output_dpath.mkdir(parents=True)

        TASK_REGISTRY[task](task_config, img_dir, global_vars, task_output_dpath)
        print(f"---finished task: '{task}'!")
    if return_global_vars:
        return global_vars


if __name__ == "__main__":
    tyro.cli(main)
