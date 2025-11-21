import os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
from pathlib import Path
import tyro
from openhype.utils.io import load_config, save_config
from main_pipeline import main as main_pipeline_main

import torch

torch.autograd.set_detect_anomaly(True)
MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


def main(
    exp_batch_name: str,
    config_template_path: str,
    eval_config_template_path: str,
    scene_id: str,
    output_dpath: str,
    n_repeats: int = 3,
):
    """
    Does all the logistics for batched experiments for one scene:
     - creates output folder per scene
     - creates config folder per batch experiment
     - creates n_repeats copies of the config template per scene with updated paths
     - starts the runs in a for loop
     - creates eval configs for each run and saves them in a eval config file
    There will be one train config per run (n_repeats train configs) but only one eval config file per scene per batch experiment. The eval config file contains n_repeats eval sub_configs, one for each trained model.
    Args:
        exp_batch_name (str): Name of the batch experiment.
        config_template_path (str): Path to the config template for training.
        eval_config_template_path (str): Path to the eval config template.
        scene_id (str): Scene id to run the experiment on.
        output_dpath (str): Output directory path.
        n_repeats (int, optional): Number of repeats for the experiment. Defaults to 3.

    """

    # 1. create output folder - concatenate outputfolder and batch experiment name and scene id
    # only if it doesnt already exist
    batch_exp_scene_output_dpath = Path(output_dpath) / scene_id
    if not batch_exp_scene_output_dpath.exists():
        batch_exp_scene_output_dpath.mkdir(parents=True)
    # 2. create config folder
    config_folder = batch_exp_scene_output_dpath / "configs" / exp_batch_name
    if not config_folder.exists():
        config_folder.mkdir(parents=True)
    # 3. create scene configs n_ times

    config_template = load_config(config_template_path)
    eval_config_template = load_config(eval_config_template_path)
    eval_config_copy = eval_config_template.copy()

    eval_config_copy.eval_configs = []
    eval_config_fpath = config_folder / f"eval_{scene_id}.yaml"
    for i in range(n_repeats):

        # create a copy of the config template
        config_copy = config_template.copy()

        # update the img dir in the config
        config_copy.general.img_dir = str(
            Path(config_template.general.img_dir) / scene_id / "images"
        )

        # change the output folder in the config
        config_copy.openhype_ae.experiment_name = (
            config_template.openhype_ae.experiment_name + f"_run{i+1}"
        )

        exp_config_fpath = config_folder / f"{scene_id}_run{i+1}.yaml"

        config_copy.openhype_nerf.experiment_name = (
            config_copy.openhype_nerf.experiment_name + f"_run{i+1}"
        )

        save_config(config_copy, exp_config_fpath)

        # start the run
        global_vars = main_pipeline_main(
            exp_config_fpath, batch_exp_scene_output_dpath, return_global_vars=True
        )

        config_copy.openhype_nerf.ae_ckpt_path = str(global_vars.ae_ckpt_path)
        config_copy.openhype_nerf.latent_eucl_feature_dir = str(
            global_vars.latent_eucl_feature_dir
        )

        save_config(config_copy, exp_config_fpath)

        for conf_idx, single_run_config in enumerate(
            eval_config_template.eval_configs
        ):  # could be several setups e.g. different amount of interp steps

            single_run_config_copy = single_run_config.copy()
            single_run_config_copy.img_dir = str(
                Path(single_run_config_copy.img_dir) / scene_id / "images"
            )
            # change the gt path in the config
            single_run_config_copy.gt_path = str(
                Path(single_run_config_copy.gt_path) / scene_id
            )
            # change the output path in the config
            single_run_config_copy.output_dir = str(
                batch_exp_scene_output_dpath / exp_batch_name / f"eval_output_run{i+1}"
            )

            single_run_config_copy.nerf_config_path = str(
                Path(global_vars.nerf_dir) / "openhype" / "run0" / "config.yml"
            )

            single_run_config_copy.ae_ckpt_path = str(global_vars.ae_ckpt_path)

            eval_config_copy.eval_configs.append(single_run_config_copy)
        # save the config
        save_config(eval_config_copy, eval_config_fpath)


if __name__ == "__main__":

    tyro.cli(main)
