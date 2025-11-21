"""
Quick wrapper for Segment Anything Model - taken from garfield repo
"""

from dataclasses import dataclass, field
from typing import Literal, Union

try:  # ugly way to avoid dependency issues
    from semantic_sam import (
        prepare_image,
        build_semantic_sam,
        SemanticSamAutomaticMaskGenerator,
    )
except:
    print("Could not import semantic_sam. Make sure when using it it is installed.")
from torchvision.io import read_image
from pathlib import Path
import torchvision
import torchvision.transforms.functional as F
import torch


def resize_mask(mask, target_size):
    return F.resize(
        mask,
        target_size,
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )


@dataclass
class ImgSegModelConfig:
    """target class to instantiate"""

    model_type: Literal["semantic_sam"] = ""
    """
    Currently supports:
     - "semantic_sam": https://github.com/UX-Decoder/Semantic-SAM
    """

    sam_model_type: str = ""
    sam_model_ckpt: str = ""
    sam_kwargs: dict = field(default_factory=lambda: {})
    "Arguments for model, e.g. levels for semantic_sam"
    env_dir: str = ""  # path to conda env with semantic_sam installed


class ImgSegModel:
    """
    Wrapper for 2D image segmentation models (e.g. MaskFormer, SAM)
    Original paper uses SAM, but we can use any model that outputs masks.
    The code currently assumes that every image has at least one group/mask.
    """

    def __init__(self, config: ImgSegModelConfig, **kwargs):
        self.config = config
        self.device = kwargs["device"]
        self.model = None

    def __call__(self, img_path: Union[str, Path]):

        if self.config.model_type == "semantic_sam":
            if self.model is None:
                self.model = build_semantic_sam(
                    model_type=self.config.sam_model_type,  # "L",
                    ckpt=self.config.sam_model_ckpt,  # "pathtomodel/swinl_only_sam_many2many.pth",
                )
                self.mask_generator = SemanticSamAutomaticMaskGenerator(
                    self.model, **self.config.sam_kwargs  # level=[6, 5, 4, 3, 2, 1]
                )  # model_type: 'L' / 'T', depends on your checkpint

            img = read_image(img_path)
            img_size = (img.shape[1], img.shape[2])

            original_image, input_image = prepare_image(image_pth=img_path)
            masks = self.mask_generator.generate(input_image)
            masks_list = [
                m["segmentation"] for m in masks if m["segmentation"].sum() > 0
            ]

            # resize masks and sort
            masks_list_resized = [
                resize_mask(torch.from_numpy(m).unsqueeze(0), img_size)
                .squeeze(0)
                .numpy()
                for m in masks_list
            ]
            masks_list_resized = sorted(masks_list_resized, key=lambda x: x.sum())
            return masks_list_resized

        raise NotImplementedError(
            f"Model type {self.config.model_type} not implemented"
        )
