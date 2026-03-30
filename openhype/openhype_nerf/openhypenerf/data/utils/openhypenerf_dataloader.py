import gc
import json
import os
import typing

import numpy as np
import torch
import torch.nn.functional as F
from openhype.openhype_nerf.openhypenerf.data.utils.feature_dataloader import (
    FeatureDataloader,
)
from tqdm import tqdm
from pathlib import Path


class SamCropDataloader(FeatureDataloader):

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        mask_dir: Path,
        image_list: torch.Tensor,
        cache_path: str = None,
        cache_path_mask: str = None,
        cache_path_mask_id_per_pixel: str = None,
        latent_eucl_feature_dir: str = None,
        latent_eucl_feature_suffix: str = None,
        latent_eucl_feat_mask_id_suffix: str = None,
    ):
        assert "image_shape" in cfg
        self.mask_dir = Path(mask_dir)
        self.cache_path_mask = cache_path_mask
        self.cache_path_mask_id_per_pixel = cache_path_mask_id_per_pixel

        self.latent_eucl_feature_dir = latent_eucl_feature_dir
        self.latent_eucl_feature_suffix = latent_eucl_feature_suffix
        self.latent_eucl_feat_mask_id_suffix = latent_eucl_feat_mask_id_suffix

        super().__init__(cfg, device, image_list, cache_path)

    def _create_pixel_mask_array(self, masks: torch.Tensor):
        """
        Taken from garfield - masks should be sorted by size, smalles one is at idx 0
        Create per-pixel data structure for grouping supervision.
        pixel_mask_array[x, y] = [m1, m2, ...] means that pixel (x, y) belongs to masks m1, m2, ...
        where Area(m1) < Area(m2) < ... (sorted by area).
        """
        max_masks = masks.sum(dim=0).max().item()
        image_shape = masks.shape[1:]
        pixel_mask_array = torch.full(
            (max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int
        ).to(masks.device)

        for m, mask in enumerate(masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_mask_array[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_mask_array[i][right_index]) != 0:
                    pixel_mask_array[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_mask_array = pixel_mask_array.permute(1, 2, 0)

        return pixel_mask_array

    def create(self, image_path_list):
        mask_feat_list = []
        background_mask_list = []
        mask_id_pixel_list = []

        for image_id in tqdm(
            range(len(image_path_list)),
            desc="samcropfeats",
            total=len(image_path_list),
            leave=False,
        ):
            with torch.no_grad():

                # load image
                image_name = Path(image_path_list[image_id])

                # load masks
                mask_path_img = self.mask_dir / (image_name.stem + "_m.npy")

                masks_tensor = torch.tensor(
                    np.load(mask_path_img)
                )  # n_masks x h x w, sorted smallest masks has idx 0

                # supervising with smallest extrapolated mask feature per pixel
                mask_feats_path = Path(self.latent_eucl_feature_dir) / (
                    image_name.stem + "_extrapolated_" + self.latent_eucl_feature_suffix
                )

                mask_idx_path = Path(self.latent_eucl_feature_dir) / (
                    image_name.stem + self.latent_eucl_feat_mask_id_suffix
                )
                used_masks_feats = torch.from_numpy(np.load(mask_feats_path))

                mask_idx = torch.from_numpy(np.load(mask_idx_path))
                used_masks_id, sort_idx = mask_idx.sort()
                used_masks_feats = used_masks_feats[
                    sort_idx
                ]  # n_used_masks x feature dim, sorted by mask size

                used_masks = masks_tensor[used_masks_id]
                pixel_mask_map = self._create_pixel_mask_array(used_masks)
                smallest_mask_id_per_pixel = pixel_mask_map[:, :, 0]

                background_mask = smallest_mask_id_per_pixel == -1

            background_mask_list.append(background_mask)
            mask_feat_list.append(used_masks_feats)
            mask_id_pixel_list.append(smallest_mask_id_per_pixel)

        self.data = torch.nested.nested_tensor(
            mask_feat_list
        )  # nested tensor since we have different amount of masks per image
        self.mask_id_per_pixel = torch.stack(
            mask_id_pixel_list, dim=0
        )  # n_images x H x W x D
        self.background_mask = torch.stack(background_mask_list, dim=0)
        gc.collect()
        torch.cuda.empty_cache()
        del mask_feat_list
        del background_mask_list
        del mask_id_pixel_list

    def downsample_discrete_mask(self, mask_ids, scale=2):
        """
        mask_ids: torch.Tensor of shape [H, W] or [1, H, W] with integer labels
        Returns: downsampled integer mask of shape [H', W']
        """
        if mask_ids.dim() == 2:
            mask_ids = mask_ids.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif mask_ids.dim() == 3:
            mask_ids = mask_ids.unsqueeze(0)  # [1, 1, H, W]

        # Cast to float for interpolation (safe with nearest)
        mask_ids = mask_ids.float()

        downsampled = F.interpolate(mask_ids, scale_factor=1 / scale, mode="nearest")
        return downsampled.squeeze(0).squeeze(0).long()

    def adjust_resolution(self):
        """
        Adjust the resolution of the mask and mask id per pixel to match the image size.
        """
        orig_h, orig_w = self.background_mask.shape[1:3]
        h, w = self.cfg["image_shape"]
        if orig_h != h or orig_w != w:
            # downsample the mask and mask id per pixel
            scale = orig_h // h
            if scale == 0:
                raise ValueError(
                    f"Scale factor is 0, original size: {orig_h}, target size: {h}"
                )
            if scale == 1:
                return
            if scale > 1:
                self.background_mask = self.downsample_discrete_mask(
                    self.background_mask, scale=scale
                ).bool()
                self.mask_id_per_pixel = self.downsample_discrete_mask(
                    self.mask_id_per_pixel, scale=scale
                )

    def load(self):
        """
        Function overload from baseclass FeatureDataloader.
        """
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")

        loaded = torch.load(self.cache_path)

        # If the loaded object is a list of tensors, rebuild nested tensor
        if isinstance(loaded, list) and all(torch.is_tensor(x) for x in loaded):
            try:
                self.data = torch.nested.nested_tensor(loaded).to(self.device)
            except Exception:
                # fallback if nested_tensor fails
                self.data = loaded
        else:
            self.data = loaded.to(self.device)
        # self.data = torch.load(self.cache_path).to(self.device)
        self.background_mask = torch.from_numpy(np.load(self.cache_path_mask)).to(
            self.device
        )  # this we need for raysampler
        self.mask_id_per_pixel = torch.from_numpy(
            np.load(self.cache_path_mask_id_per_pixel)
        ).to(self.device)
        self.adjust_resolution()

    def save(self):
        """
        Function overload from baseclass FeatureDataloader.
        """

        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))

        # Convert NestedTensor → list-of-tensors before saving
        if torch.is_tensor(self.data) and self.data.is_nested:
            # torch.nested.to_padded_tensor exists, but we want raw list
            torch.save(list(self.data.unbind()), self.cache_path)
        else:
            torch.save(self.data, self.cache_path)

        # torch.save(self.data, self.cache_path)
        np.save(self.cache_path_mask, self.background_mask.numpy())
        np.save(self.cache_path_mask_id_per_pixel, self.mask_id_per_pixel.numpy())

        # after saving - downsample if necessary (e.g. if image is downsampled)
        self.adjust_resolution()

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        mask_ids = self.mask_id_per_pixel[
            img_points[:, 0].long(), img_points[:, 1].long(), img_points[:, 2].long()
        ]

        # create normal tensor from nested self.data tensor and pad with nans then get correspoing mask features
        # mask_ids should be in idx of only used_mask_ids
        feature = self.data.to_padded_tensor(torch.nan)[img_points[:, 0], mask_ids].to(
            self.device
        )

        return feature
