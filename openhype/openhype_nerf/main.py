import os
from pathlib import Path
import subprocess
import torch

torch.autograd.set_detect_anomaly(True)

MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)
from openhype.task_registry import register_task
from openhype.variable_registry import GlobalVars


@register_task("openhype_nerf")
def main(config, img_dir, global_vars: GlobalVars, output_dpath):

    exp_name = config.experiment_name
    global_vars.nerf_dir = str(Path(output_dpath / exp_name))
    cache_dir = str(Path(output_dpath) / exp_name / "cache")
    if config.latent_eucl_feature_dir == "to_be_set":
        latent_eucl_feature_dir = global_vars.latent_eucl_feature_dir
    else:
        latent_eucl_feature_dir = config.latent_eucl_feature_dir
    cmd = [
        os.path.join(config.env_dir, "bin/python"),
        os.path.join(
            config.env_dir,
            "lib/python3.10/site-packages/nerfstudio/scripts/train.py",
        ),
        f"openhype",
        f"--vis=wandb",  # viewer+wandb
        f"--experiment-name={exp_name}",
        f"--viewer.num-rays-per-chunk=2048",
        f"--steps-per-eval-batch=500000",
        f"--steps-per-eval-image=500000",
        f"--steps-per-eval-all-images=500000",
        f"--max-num-iterations=30000",
        f"--pipeline.model.openhype-loss-weight=1.0",
        f"--pipeline.model.openhype-loss={config.loss}",
        f"--pipeline.datamanager.train-num-rays-per-batch=2048",
        f"--data={Path(img_dir).parent}",
        f"--output-dir={str(Path(output_dpath))}",
        f"--timestamp={config.time_stamp}",
        f"--pipeline.datamanager.cache_dir={cache_dir}",
        f"--pipeline.datamanager.mask_dir={Path(global_vars.mask_dir)}",
        f"--pipeline.datamanager.latent_eucl_feature_dir={latent_eucl_feature_dir}",
        f"--pipeline.datamanager.latent_eucl_feature_suffix={global_vars.latent_eucl_feature_suffix}",
        f"--pipeline.datamanager.latent_eucl_feat_mask_id_suffix={global_vars.latent_eucl_feat_mask_id_suffix}",
        f"--pipeline.model.lang_field_dim={config.lang_field_dim}",
        "nerfstudio-data",  # dataparser name, we have the nerfstudiodataparser
        f"--eval-mode={config.eval_mode}",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
