"""
OpenHype configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from openhype.openhype_nerf.openhypenerf.openhypenerf_datamanager import (
    OpenHypeDataManagerConfig,
)
from openhype.openhype_nerf.openhypenerf.openhypenerf_model import OpenHypeModelConfig
from openhype.openhype_nerf.openhypenerf.openhypenerf_pipeline import (
    OpenHypePipelineConfig,
)


openhype_method = MethodSpecification(
    config=TrainerConfig(
        method_name="openhype",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=60000,
        mixed_precision=True,
        pipeline=OpenHypePipelineConfig(
            datamanager=OpenHypeDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    eval_mode="filename"  # "split", train_split_fraction=1.0
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=OpenHypeModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_openhype_samples=24,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=30000
                ),
            },
            "openhype": {
                "optimizer": RAdamOptimizerConfig(
                    lr=1e-2, eps=1e-15, weight_decay=1e-9
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=4000  # 30000  # 4000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for OpenHype",
)
