"""
Quick wrapper for Preprocessing Pipeline.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import subprocess

import torch
import numpy as np
from torchvision.io import read_image

from openhype.openhype_preprocess.img_seg_model import ImgSegModelConfig
from openhype.openhype_preprocess.crop_encoder import (
    MaskCropEncoder,
    MaskCropEncoderConfig,
)
from openhype.variable_registry import GlobalVars


@dataclass
class HyperPreProcessorConfig:
    img_seg_model_config: ImgSegModelConfig = ImgSegModelConfig()
    crop_enc_model_config: MaskCropEncoderConfig = MaskCropEncoderConfig()
    device: Union[torch.device, str] = "cuda:0"


class HyperPreProcessor:
    """
    Class that handles the preprocessing of 2D images.
        - extracting multi-level masks (all extracted are saved)
        - cleaning masks
        - creating mask hierarchies (the mask ids saved in this json are used for training)
        - getting crops
        - vision-language features per mask
    """

    def __init__(
        self,
        config: HyperPreProcessorConfig,
        img_folder: Union[str, Path],
        global_vars: GlobalVars,
        output_dpath: Union[
            str, Path
        ],  # folder where to store features and mask hierarchies etc.
    ) -> None:

        self.config = config
        self.device = self.config.device
        self.model = None
        self.img_folder = Path(img_folder)

        # self.img_seg_model = ImgSegModel(
        #     config.img_seg_model_config, device=self.device
        # ) # model that extracts masks, e.g. SemanticSAM - we call this as a separate process due to dependency issues
        if self.config.img_seg_model_config.model_type == "semantic_sam":
            print(
                "Using SemanticSAM for mask extraction. This will be called as a separate process due to dependency issues."
            )
        else:
            raise NotImplementedError(
                f"Image segmentation model {self.config.img_seg_model_config.model_type} not implemented. Please use 'semantic_sam'"
            )

        self.crop_enc_model = MaskCropEncoder(
            config.crop_enc_model_config, device=self.device
        )

        ## Create output folders and specify suffix for saving:
        ## - masks
        ## - mask hierarchies
        ## - VL-features per mask (clip_crop_features)
        output_dpath.mkdir(exist_ok=True)
        self.mask_output_folder = output_dpath / "masks"

        self.mask_hierarchy_output_folder = output_dpath / "mask_hierarchies"
        self.feature_output_folder = output_dpath / "clip_crop_features"
        self.feature_output_folder = self.feature_output_folder / (
            config.crop_enc_model_config.clip_model_type
            + "_"
            + config.crop_enc_model_config.clip_model_pretrained
        )
        self.mask_suffix = "_m.npy"
        self.mask_feature_suffix = "_f.npy"

        self.mask_hierarchy_suffix = "_h.json"
        self.mask_hierarchy_problems_suffix = "_h_problems.json"

        # set in global vars
        global_vars.mask_dir = self.mask_output_folder
        global_vars.mask_hierarchy_dir = self.mask_hierarchy_output_folder
        global_vars.mask_feature_dir = self.feature_output_folder
        global_vars.mask_suffix = self.mask_suffix
        global_vars.mask_feature_suffix = self.mask_feature_suffix
        global_vars.mask_hierarchy_suffix = self.mask_hierarchy_suffix

        # create folders
        self.feature_output_folder.mkdir(exist_ok=True, parents=True)
        self.mask_hierarchy_output_folder.mkdir(exist_ok=True)
        self.mask_output_folder.mkdir(exist_ok=True)

        ## Load images
        # create list of image paths
        self.data_list = os.listdir(str(self.img_folder))
        self.data_list.sort()
        self.data_list = self.data_list
        print(f"Found {len(self.data_list)} images to preprocess...")

    def get_mask_hierarchy(
        self, masks: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[Dict, List, List]:
        """
        Function that creates and stores hierarchies of masks
        in a parent, child manner.

        """

        hierarchy_dict = {}
        n_masks = len(masks)
        masks_area = torch.zeros(n_masks)
        for i in range(n_masks):
            masks_area[i] = masks[i].sum()

        mask_ids_sorted = torch.argsort(masks_area, dim=0)  # doesn't contain -1
        non_over_lap_p = []
        remove_list = []
        for current_idx, j in enumerate(mask_ids_sorted):
            j = j.item()

            if j in remove_list:  # skip this mask, already removed
                continue

            if str(j) not in hierarchy_dict:
                hierarchy_dict[str(j)] = {"parents": [], "children": []}

            for i in mask_ids_sorted[
                current_idx + 1 :
            ]:  # mask idx are sorted by area -> only need to check the ones afterwards
                i = i.item()

                # check if duplicates with curent mask j or any assigned parents masks
                # if almost same mask remove one.
                # for now we keep smaller one, we compare with mask at hand and with already assigned parents
                # otherwise it could be that it will be in general removed form the dict but is still assigned as parents somewhere
                to_check_for_duplicates = [p for p in hierarchy_dict[str(j)]["parents"]]
                to_check_for_duplicates.append(j)
                for p in to_check_for_duplicates:
                    overlap = (masks[i] * masks[p]).sum()

                    iou = overlap / ((masks[i] + masks[p]) > 0).sum()
                    if iou > 0.95:
                        print(f"------------------ {i} same mask as {p}: iou - {iou} ")
                        if i not in remove_list:  # and p not in remove_list:
                            print(f"---- adding {i} to remove list")
                            remove_list.append(i)
                        # stop and continue for loop
                        break
                if i in remove_list:  # skip this mask, already removed
                    continue

                # add to parent if mask i entails child mask j
                overlap = (masks[i] * masks[j]).sum()
                c_mask_size = masks[j].sum()
                overlap_ratio = overlap / c_mask_size
                if (
                    overlap_ratio > threshold
                ):  # only if actually a parent not due to edge pixel
                    hierarchy_dict[str(j)]["parents"].append(i)
                    # sanity check  - previous parent must also be included in this parent
                    # this will sometimes fail due to wobbly masks etc. - we ignore this but save those issues in a seperate json
                    current_parents = hierarchy_dict[str(j)]["parents"]
                    if len(current_parents) > 1:
                        overlap = (
                            masks[current_parents[-1]] * masks[current_parents[-2]]
                        ).sum()
                        c_mask_size = masks[current_parents[-2]].sum()
                        overlap_ratio_p = overlap / c_mask_size
                        if overlap_ratio_p <= threshold:
                            print(
                                f"------parents not overlapping: {current_parents[-2]} not in {current_parents[-1]}"
                            )
                            non_over_lap_p.append(
                                [current_parents[-2], current_parents[-1]]
                            )

                    # assign mask as child to parent mask
                    if str(i) not in hierarchy_dict:
                        hierarchy_dict[str(i)] = {"parents": [], "children": []}
                    hierarchy_dict[str(i)]["children"].append(
                        j
                    )  # already sorted ascending

        # clean hierarchies from this clusters
        for idx in remove_list:
            if str(idx) in hierarchy_dict:
                del hierarchy_dict[str(idx)]
            for k in hierarchy_dict.keys():
                if idx in hierarchy_dict[k]["parents"]:
                    hierarchy_dict[k]["parents"].remove(idx)
                if idx in hierarchy_dict[k]["children"]:
                    hierarchy_dict[k]["children"].remove(idx)

        return hierarchy_dict, remove_list, non_over_lap_p

    def extract_masks_semanticsam(self):
        sam_kwargs_str = json.dumps(
            self.config.img_seg_model_config.sam_kwargs
        )  # produce correct string representation for command line
        cmd = [
            os.path.join(self.config.img_seg_model_config.env_dir, "bin/python"),
            os.path.join(
                os.getcwd(), "openhype/openhype_preprocess/extract_masks_semanticsam.py"
            ),
            f"--img_folder={str(self.img_folder)}",
            f"--mask_output_folder={str(self.mask_output_folder)}",
            f"--mask_suffix={self.mask_suffix}",
            f"--sam_model_type={self.config.img_seg_model_config.sam_model_type}",
            f"--sam_model_ckpt={self.config.img_seg_model_config.sam_model_ckpt}",
            f"--sam_kwargs={sam_kwargs_str}",
            f"--device={str(self.config.device)}",
        ]

        subprocess.run(cmd)

    def preprocess(self):

        # since we have dependency issues with semanticsam and openhype lib versions,
        # we call the extraction as a separate process
        self.extract_masks_semanticsam()

        for img_idx, img_name in enumerate(
            tqdm(self.data_list, desc="Preprocess images")
        ):
            ## ----- 0.0) load image -----
            image_path = os.path.join(self.img_folder, img_name)
            img = read_image(image_path)

            ## ----- 0.1) get paths -----
            save_path_masks = self.mask_output_folder / (
                str(Path(img_name).stem) + self.mask_suffix
            )
            save_path_h = self.mask_hierarchy_output_folder / (
                str(Path(img_name).stem) + self.mask_hierarchy_suffix
            )
            save_path_h_problems = self.mask_hierarchy_output_folder / (
                str(Path(img_name).stem) + self.mask_hierarchy_problems_suffix
            )
            save_path_feats = self.feature_output_folder / (
                str(Path(img_name).stem) + self.mask_feature_suffix
            )

            if (
                save_path_feats.exists()
                and save_path_h.exists()
                and save_path_masks.exists()
            ):
                continue  # already preprocesssed

            ## ----- 1) get masks -----
            if save_path_masks.exists():
                masks = np.load(save_path_masks)
            else:
                raise NotImplementedError(
                    "Mask file not found. Make sure to run the extraction with SemanticSAM first."
                )
                # masks = self.img_seg_model(img_path=image_path)
                # masks = np.stack(masks)  # n_masks x H x W
                # np.save(save_path_masks, masks)

            masks_tensor = torch.tensor(masks, device=self.device)  # n_masks x H x W

            # erode all masks using 3x3 kernel
            eroded_masks = torch.conv2d(
                masks_tensor.unsqueeze(1).float(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3).to(self.device),
                padding=1,
            )

            masks_tensor = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

            ## ----- 2) clean and get mask hierarchies -----
            if not save_path_h.exists():
                prob_cases_mask_hierarchy = (
                    {}
                )  # problematic cases in hierarchies i.e. non overlapping parents or duplicates

                mask_hierarchy, remove_list, non_overlap_parents = (
                    self.get_mask_hierarchy(masks_tensor.cpu())
                )

                prob_cases_mask_hierarchy["remove"] = remove_list
                prob_cases_mask_hierarchy["parents_non_overlap"] = non_overlap_parents

                with open(str(save_path_h), "w") as outfile:
                    json.dump(mask_hierarchy, outfile)
                # with open(str(save_path_h_problems), "w") as outfile:
                #     json.dump(prob_cases_mask_hierarchy, outfile)

            ## ----- 3) create VL feature for mask crops -----
            if not save_path_feats.exists():
                self.crop_enc_model(
                    img,
                    masks_tensor,
                    save_path_feats,
                )
