from typing import Union
from pathlib import Path
from dataclasses import dataclass


@dataclass()  # does not include image dir since this is fixed from the beginning
class GlobalVars:
    """Stores variables that are created throughout the pipeline."""

    # set in preprocessor.py - could eventually set fixed things like suffix here
    mask_hierarchy_dir: Union[str, Path] = None
    mask_dir: Union[str, Path] = None  # set in preprocessor.py
    mask_feature_dir: Union[str, Path] = None
    mask_hierarchy_suffix: str = None
    mask_feature_suffix: str = None  # path is mask_feature_dir/image_name+suffix
    mask_suffix: str = None
    # set in main.py of train_hyperembedder - could eventually set fixed things like suffix here
    latent_eucl_feature_dir: Union[str, Path] = None

    latent_eucl_feature_suffix: str = "_f.npy"  # None
    latent_eucl_feat_mask_id_suffix: str = "_m_id.npy"  # None

    nerf_dir: Union[str, Path] = None  # set in main of openhyper_nerf

    ae_ckpt_path: Union[str, Path] = None  # only used for creating automatic eval files
