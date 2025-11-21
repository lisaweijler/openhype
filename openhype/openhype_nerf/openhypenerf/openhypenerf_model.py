from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal

import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter

from openhype.openhype_nerf.openhypenerf.openhypenerf_field import OpenHypeField
from openhype.openhype_nerf.openhypenerf.openhypenerf_fieldheadnames import (
    OpenHypeFieldHeadNames,
)
from openhype.openhype_nerf.openhypenerf.openhypenerf_renderers import (
    CLIPRenderer,
    MeanRenderer,
)
from openhype.utils import lorentz as L


@dataclass
class OpenHypeModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: OpenHypeModel)

    openhype_loss_weight: float = 1.0
    openhype_loss: Literal[
        "Huber",
        "Cosine",
        "MSE",
        "Hyperbolic_geodesic_regularized",
    ] = "Hyperbolic_geodesic_regularized"
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_openhype_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)
    num_hidden_clip_layers: int = 1
    lang_field_dim: int = 512


class OpenHypeModel(NerfactoModel):
    config: OpenHypeModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.openhype_field = OpenHypeField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            self.config.num_hidden_clip_layers,
            lang_field_dim=self.config.lang_field_dim,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(
            ray_samples
        )
        openhype_weights, best_ids = torch.topk(
            weights, self.config.num_openhype_samples, dim=-2, sorted=False
        )

        def gather_fn(tens):
            return torch.gather(
                tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1])
            )

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        openhype_samples: RaySamples = ray_samples._apply_fn_to_fields(
            gather_fn, dataclass_fn
        )

        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        openhype_field_outputs = self.openhype_field.get_outputs(openhype_samples)

        outputs["openhype"] = self.renderer_mean(
            embeds=openhype_field_outputs[OpenHypeFieldHeadNames.OPENHYPE],
            weights=openhype_weights.detach(),
        )

        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(
            ray_samples, compute_normals=self.config.predict_normals
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:

            with torch.autocast(outputs["openhype"].device.type, dtype=torch.float64):

                if self.config.openhype_loss == "Huber":
                    unreduced_openhype = (
                        self.config.openhype_loss_weight
                        * torch.nn.functional.huber_loss(
                            outputs["openhype"],
                            batch["openhype"],
                            delta=1.25,
                            reduction="none",
                        )
                    )
                elif self.config.openhype_loss == "Cosine":
                    unreduced_openhype = self.config.openhype_loss_weight * (
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            outputs["openhype"], batch["openhype"]
                        )
                    )

                elif self.config.openhype_loss == "MSE":
                    unreduced_openhype = (
                        self.config.openhype_loss_weight
                        * torch.nn.functional.mse_loss(
                            outputs["openhype"], batch["openhype"], reduction="none"
                        )
                    )  # n_samples x dim

                elif self.config.openhype_loss == "Hyperbolic_geodesic_regularized":
                    dist_gt = torch.norm(batch["openhype"], dim=-1, keepdim=True)
                    dist_pred = torch.norm(outputs["openhype"], dim=-1, keepdim=True)
                    reg_loss = torch.nn.functional.mse_loss(
                        dist_pred, dist_gt, reduction="none"
                    )  # n_samples x dim
                    geodesic_dist_loss = (
                        self.config.openhype_loss_weight
                        * L.rowwise_dist(
                            L.exp_map0(outputs["openhype"], curv=1.0),
                            L.exp_map0(batch["openhype"], curv=1.0),
                        )
                    )
                    unreduced_openhype = reg_loss + geodesic_dist_loss

                loss_dict["openhype_loss"] = unreduced_openhype.sum(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["openhype"] = list(self.openhype_field.parameters())
        return param_groups
