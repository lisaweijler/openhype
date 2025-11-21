import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Mapping


class MaskFeatCollater:
    """
    Collate function to batch samples from MaskFeatureDataset.
    Handles variable number of masks per image by creating block diagonal matrices where needed.
    """

    def __call__(self, batch: List[Dict]) -> Dict:
        elem = batch[0]

        if isinstance(elem, torch.Tensor):
            if len(elem.shape) == 2:
                max_dim = max([elem.shape[-1] for elem in batch])
                min_dim = min([elem.shape[-1] for elem in batch])
                if (
                    min_dim
                    != max_dim  # in case two samples have by accident the same amount of negs, then this fails.. so i added it in the dict comprehension
                ):  # different dimensions - make block diagonal matrix - n_masks x n_masks
                    return torch.block_diag(
                        *batch
                    )  # for keep_for_neg, - different for each image

            return torch.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):

            return_dict = {
                key: (
                    self([data[key] for data in batch])
                    if key != "keep_for_negs"
                    else torch.block_diag(*[data[key] for data in batch])
                )
                for key in elem
            }
            batch_ids = torch.arange(len(batch))
            return_dict["batch_ids"] = batch_ids.repeat_interleave(
                return_dict["n_masks_used"]
            )
            return return_dict

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class MaskFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset for training hierarchical hyperbolic embeddings of mask CLIP embeddings.
    Contains one featurevector of dim 512 for each mask of each image of one scene.
    """

    def __init__(
        self,
        mask_feature_dir: Path,
        mask_hierarchy_dir: Path,
        img_dir: Path,
        mask_feature_suffix: str,
        mask_hierarchy_suffix: str,
        split: str = "train",  # could also be "all"
    ):
        # get file suffix
        self.mask_feature_suffix = mask_feature_suffix
        self.mask_hierarchy_suffix = mask_hierarchy_suffix

        self.mask_feature_dir = mask_feature_dir
        self.mask_hierarchy_dir = mask_hierarchy_dir

        # create list of image paths
        self.data_list = os.listdir(img_dir)
        if split == "train":
            self.data_list = [img for img in self.data_list if "train" in img]

        self.data_list.sort()

        self.n_masks_sampled = None  # use all masks in image if None else n specified

    def __len__(self):
        return len(self.data_list)

    def getitem_by_name(self, image_name: str):
        if image_name not in self.data_list:
            raise ValueError(f"Image '{image_name}' not in Dataset file list.")
        idx = self.data_list.index(image_name)

        return self.__getitem__(idx)

    def __getitem__(self, idx) -> Dict:
        # load image
        image_name = Path(self.data_list[idx])

        # load features
        mask_feats_path = self.mask_feature_dir / (
            str(image_name.stem) + self.mask_feature_suffix
        )
        mask_feats = np.load(mask_feats_path)

        # load hierarchies
        hierarchy_dict_path = self.mask_hierarchy_dir / (
            str(image_name.stem) + self.mask_hierarchy_suffix
        )
        with open(str(hierarchy_dict_path)) as json_file:
            hierarchy_dict = json.load(json_file)

        used_masks_id = torch.tensor([int(k) for k in hierarchy_dict.keys()])
        used_mask_feats = torch.tensor(mask_feats[used_masks_id])
        parent_feat_list = []
        mask_id_used_parents = []
        has_parent = torch.ones(used_mask_feats.shape[0]).bool()
        has_child = torch.ones(used_mask_feats.shape[0]).bool()
        exclude_from_negs = torch.zeros(
            (used_masks_id.shape[0], used_masks_id.shape[0])
        ).bool()
        for k_i, k in enumerate(hierarchy_dict.keys()):
            if len(hierarchy_dict[k]["children"]) == 0:
                has_child[k_i] = False
            if len(hierarchy_dict[k]["parents"]) != 0:
                # only using one parent
                parent_feat_list.append(
                    torch.tensor(mask_feats[hierarchy_dict[k]["parents"][0]])
                )
                mask_id_used_parents.append(hierarchy_dict[k]["parents"][0])
            else:
                try:
                    has_parent[k_i] = False
                except:
                    print(
                        f"Something is not right with masks idx etc in file: {self.data_list[idx]} check dataloader and your preprocessing."
                    )

            # create mask to know what to exclude for negatives in contrastive learning
            # we exclude all parents and all children of the specific mask
            to_exclude = []
            to_exclude.extend(hierarchy_dict[k]["parents"])
            to_exclude.extend(hierarchy_dict[k]["children"])
            to_exclude_cleaned_id = [
                (used_masks_id == i).nonzero(as_tuple=True)[0] for i in to_exclude
            ]  # torch equivalent of list.index(i)
            exclude_from_negs[k_i, to_exclude_cleaned_id] = True
        parent_mask_feats = torch.stack(parent_feat_list, dim=0)
        mask_id_parents = torch.tensor(
            [
                (used_masks_id == m).nonzero(as_tuple=True)[0].item()
                for m in mask_id_used_parents
            ]
        )
        is_parent = torch.zeros_like(has_parent).bool()
        is_parent[mask_id_parents] = True

        return {
            "mask_feats": used_mask_feats.float(),  # (n_masks, feat_dim)
            "parent_feats": parent_mask_feats.float(),  # (n_masks_with_parent, feat_dim)
            "mask_id_parents": mask_id_parents,  # (n_masks_with_parent,)
            "is_parent": is_parent,  # (n_masks,)
            "img_idx": idx,  # scalar
            "mask_id_used": used_masks_id,  # (n_masks,)
            "n_masks_used": len(used_masks_id),  # scalar
            "has_parent": has_parent,  # (n_masks,)
            "has_child": has_child,  # (n_masks,)
            "keep_for_negs": (~exclude_from_negs),  # (n_masks, n_masks)
            # invert so it is easier for batching with block diag - 0 means no, automatically exclude other images/items of batch
        }
