from typing import Union
from pathlib import Path
from einops import rearrange
import numpy as np
import torch


from openhype.openhype_ae.mask_feature_dataset import MaskFeatureDataset
from openhype.openhype_ae.hyperbolic_ae import HyperEmbedder
from openhype.utils import lorentz as L


MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


def embed_feats(
    model_config,
    img_dir: Union[Path, str],  # used in dataloader,
    output_dir: Union[Path, str],
    ckpt_dir: Union[Path, str],
    mask_feature_dir: Union[Path, str],
    mask_hierarchy_dir: Union[Path, str],
    mask_feature_suffix: str,
    mask_hierarchy_suffix: str,
    latent_eucl_feature_suffix: str,
    latent_eucl_feat_mask_id_suffix: str,
    save_extrapolated_latent_features: bool = True,
):

    # prepare paths
    device = torch.device(f"cuda:0")
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    ckpt_dir = Path(ckpt_dir)
    mask_feature_dir = Path(mask_feature_dir)
    mask_hierarchy_dir = Path(mask_hierarchy_dir)

    # load model
    checkpoint = torch.load(ckpt_dir / "model_best.pth", map_location="cpu")
    model = HyperEmbedder(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval

    # create dataset
    dataset = MaskFeatureDataset(
        mask_feature_dir,
        mask_hierarchy_dir,
        img_dir,
        mask_feature_suffix,
        mask_hierarchy_suffix,
        split="all",
    )

    # ----------- start embedding masks for each image
    for i in range(len(dataset)):
        img_data = dataset[i]
        print(f"Working on image: {dataset.data_list[i]}")  # e.g. frame_00001.jpg

        with torch.no_grad():
            mask_embeddings_img_hyper = model.encode_features(
                img_data["mask_feats"].to(device), project=True
            )

        save_path_feats = output_dir / (
            str(Path(dataset.data_list[i]).stem) + latent_eucl_feature_suffix
        )
        save_path_mask_idx = output_dir / (
            str(Path(dataset.data_list[i]).stem) + latent_eucl_feat_mask_id_suffix
        )
        # since it is projected and deprojected to hyperboloid, norm is clamped to max dist in hyperb. space (11.0904)
        mask_embeddings_img_eucl = L.log_map0(
            mask_embeddings_img_hyper, curv=model_config.curv_init
        )

        # embeddings are saved in euclidean space!!
        np.save(save_path_feats, mask_embeddings_img_eucl.detach().cpu().numpy())
        np.save(save_path_mask_idx, img_data["mask_id_used"].detach().cpu().numpy())

        if save_extrapolated_latent_features:
            extrapolated_features = L.get_interpolated_hyperbolic_features(
                L.exp_map0(mask_embeddings_img_eucl, curv=model_config.curv_init),
                steps=40,  # doesn't matter if different than eval steps since we only use the ones at the boundary!
                curv=model_config.curv_init,
            )  # (n_features * n_steps, latent_dim)
            extrapolated_features = rearrange(
                extrapolated_features, "(n s) d -> n s d", s=40
            )
            extrapolated_features_euclidean = L.log_map0(
                extrapolated_features[:, 0, :], curv=model_config.curv_init
            )  # 0 index furtheest away from root/origin, smalles mask!

            save_path_feats_extrapolated = output_dir / (
                str(Path(dataset.data_list[i]).stem)
                + "_extrapolated_"
                + latent_eucl_feature_suffix
            )

            # embeddings are saved in euclidean space!!
            np.save(
                save_path_feats_extrapolated,
                extrapolated_features_euclidean.detach().cpu().numpy(),
            )
