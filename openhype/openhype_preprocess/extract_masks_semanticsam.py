import os
from pathlib import Path
import numpy as np
import tyro
from tqdm import tqdm
from typing import Any, Dict, List
import ast

from img_seg_model import ImgSegModel, ImgSegModelConfig


def main(
    img_folder: str,
    mask_output_folder: str,
    mask_suffix: str,
    sam_model_type: str,
    sam_model_ckpt: str,
    sam_kwargs: str,  # Dict[str, List[int]],
    device: str = "cuda:0",
):
    """Extract masks for all images in a folder using SemanticSAM.
    Args:
        img_folder (str): Path to folder with images.
        mask_output_folder (str): Path to folder to save masks.
        mask_suffix (str): Suffix to add to mask files.
        sam_model_type (str): Type of SAM model to use.
        sam_model_ckpt (str): Path to SAM model checkpoint.
        sam_kwargs (str): String representation of dictionary with SAM kwargs.
        device (str, optional): Device to use. Defaults to "cuda:0".
    """
    sam_kwargs = ast.literal_eval(sam_kwargs)  # convert string to dict
    img_folder = Path(img_folder)
    mask_output_folder = Path(mask_output_folder)
    if not mask_output_folder.exists():
        mask_output_folder.mkdir(parents=True)
    ## Load images
    # create list of image paths
    data_list = os.listdir(str(img_folder))
    data_list.sort()
    print(f"Found {len(data_list)} images to preprocess...")
    config = ImgSegModelConfig(
        model_type="semantic_sam",
        sam_model_type=sam_model_type,
        sam_model_ckpt=sam_model_ckpt,
        sam_kwargs=sam_kwargs,
    )
    img_seg_model = ImgSegModel(config, device=device)

    for img_idx, img_name in enumerate(
        tqdm(data_list, desc="Extracting masks with SemanticSAM")
    ):
        ## ----- 0.0) load image -----
        image_path = os.path.join(img_folder, img_name)
        # img = read_image(image_path)

        ## ----- 0.1) get paths -----
        save_path_masks = mask_output_folder / (str(Path(img_name).stem) + mask_suffix)

        ## ----- 1) get masks -----
        if save_path_masks.exists():
            continue
        else:
            masks = img_seg_model(img_path=image_path)
            masks = np.stack(masks)  # n_masks x H x W
            np.save(save_path_masks, masks)


if __name__ == "__main__":
    tyro.cli(main)
