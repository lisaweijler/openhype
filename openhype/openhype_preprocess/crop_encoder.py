from __future__ import annotations
from dataclasses import dataclass
from einops import rearrange
from torchvision.ops import masks_to_boxes
from pathlib import Path
import torch
import numpy as np
import torchvision

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

try:
    import open_clip
except ImportError:
    assert (
        False
    ), "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class MaskCropEncoderConfig:
    clip_model_type: str = ""  # "ViT-B-16"
    clip_model_pretrained: str = ""  # "laion2b_s34b_b88k"


class MaskCropEncoder:
    """
    Wrapper for 2D Vision-Language mask features creation.
    First creat crops based on masks and then embed using e.g. CLIP.
    """

    def __init__(self, config: MaskCropEncoderConfig, **kwargs):
        self.config = config
        self.device = kwargs["device"]
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                config.clip_model_type,  # e.g., ViT-B-16
                pretrained=config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
                device=self.device,
                # precision="fp16"
            )
        )

        self.model.eval()
        if config.clip_model_type in ["ViT-SO400M-14-SigLIP", "ViT-L-14"]:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        self.preprocess_vlm = torchvision.transforms.Compose(  #  same as standard preproces_train, use this for now
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=mean,
                    std=std,
                ),
            ]
        )

    def _pad_img(self, img):
        """
        pad with zeros to make quadratic pic
        """
        _, h, w = img.shape
        l = max(w, h)
        pad = torch.zeros((3, l, l), dtype=torch.uint8)
        if h > w:
            pad[:, :, (h - w) // 2 : (h - w) // 2 + w] = img
        else:
            pad[:, (w - h) // 2 : (w - h) // 2 + h, :] = img
        return pad

    def _get_seg_img(self, mask, image, with_background=False):
        """
        get image crop based on mask.
        if with background get backround around  mask -> use image crop based on bounding box of mask.
        Otherwise set everything to zero = black, that is not in the mask.
        """
        image = image.clone()
        if not with_background:  # no background/context
            image[mask.expand(image.shape) == 0] = 0

        x1, y1, x2, y2 = (int(x.item()) for x in masks_to_boxes(mask)[0])
        seg_img = image[:, y1 : y2 + 1, x1 : x2 + 1]  # c x h x w - x= w , y= h

        return seg_img

    def _mask2box(self, mask: torch.Tensor):
        row = torch.nonzero(mask.sum(axis=0))[:, 0]
        if len(row) == 0:
            return None
        x1 = row.min().item()
        x2 = row.max().item()
        col = np.nonzero(mask.sum(axis=1))[:, 0]
        y1 = col.min().item()
        y2 = col.max().item()
        return x1, y1, x2 + 1, y2 + 1

    def _mask2box_multi_level(self, mask: torch.Tensor, level, expansion_ratio):
        x1, y1, x2, y2 = self._mask2box(mask)
        if level == 0:
            return x1, y1, x2, y2
        shape = mask.shape
        x_exp = int(abs(x2 - x1) * expansion_ratio) * level
        y_exp = int(abs(y2 - y1) * expansion_ratio) * level
        return (
            max(0, x1 - x_exp),
            max(0, y1 - y_exp),
            min(shape[1], x2 + x_exp),
            min(shape[0], y2 + y_exp),
        )

    def _get_multi_level_crops(
        self,
        mask,
        image,
        num_levels: int = 2,
        multi_level_expansion_ratio: float = 0.1,
        pad: bool = True,
    ):
        # MULTI LEVEL CROPS
        level_crops = []
        for level in range(num_levels):
            if level == 0:
                cropped_img = self._get_seg_img(
                    mask.unsqueeze(0), image, with_background=False
                )  # raw mask
            else:
                # get the bbox and corresponding crops
                x1, y1, x2, y2 = self._mask2box_multi_level(
                    mask, level, multi_level_expansion_ratio
                )

                cropped_img = image[:, y1:y2, x1:x2]

            if pad:
                cropped_img = self._pad_img(cropped_img)

            level_crops.append(cropped_img)

        crops = torch.stack(
            [self.preprocess_vlm(crop.float() / 255.0) for crop in level_crops],
            dim=0,
        )

        return crops

    def __call__(
        self,
        img: np.ndarray,
        masks: torch.Tensor,
        save_path_f: Path,
    ):

        crops_list = []
        for m in masks:
            crops = self._get_multi_level_crops(m, img, pad=True, num_levels=2)
            crops_list.append(crops)
        crops = torch.stack(crops_list, dim=0)  # n_masks x n_levels x c x cropdims
        crops = rearrange(crops, "m l c h w -> (m l) c h w")
        crops = crops.to(self.device)

        with torch.no_grad():
            clip_embed = self.model.encode_image(
                crops, normalize=False
            )  # n_masks x n_dims

        clip_embed = rearrange(clip_embed, "(m l) d -> m l d", l=2)
        clip_embed = clip_embed.mean(dim=1)
        np.save(save_path_f, clip_embed.detach().cpu().numpy())
