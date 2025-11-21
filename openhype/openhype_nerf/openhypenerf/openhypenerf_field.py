from typing import Dict, List, Optional, Tuple
import sys
import numpy as np
import torch
from openhype.openhype_nerf.openhypenerf.openhypenerf_fieldheadnames import (
    OpenHypeFieldHeadNames,
)
from torch import Tensor
from jaxtyping import Float

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, FieldConfig

try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


class OpenHypeField(Field):
    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        num_hidden_clip_layers,
        spatial_distortion: SpatialDistortion = SceneContraction(),
        lang_field_dim: int = 768,  # 32, 512
    ):
        super().__init__()
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(
            grid_layers
        )
        self.spatial_distortion = spatial_distortion
        self.clip_encs = torch.nn.ModuleList(
            [
                OpenHypeField._get_encoding(
                    grid_resolutions[i][0],
                    grid_resolutions[i][1],
                    grid_layers[i],
                    indim=3,
                    hash_size=grid_sizes[i],
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])

        self.openhype_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=lang_field_dim,  # 768,  # 512, #768
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": num_hidden_clip_layers,
            },
        )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs(
        self, ray_samples: RaySamples, clip_scales=None
    ) -> Dict[OpenHypeFieldHeadNames, Float[Tensor, "bs dim"]]:
        outputs = {}

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        openhype_pass = self.openhype_net(x).view(*ray_samples.frustums.shape, -1)

        outputs[OpenHypeFieldHeadNames.OPENHYPE] = openhype_pass

        return outputs
