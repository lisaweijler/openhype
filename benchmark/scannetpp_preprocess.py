"""
Convert the scannet dataset into nerfstudio format.
This script is adapted from RelationField's scannet preprocessing script.
"""

from pathlib import Path
import shutil
import numpy as np
import torch
import json
import os
import pymeshlab

import scannetpp

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE
from nerfstudio.utils.io import load_from_json


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def read_intrinsic(intrinsic_path, mode="rgb"):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()

    m_versionNumber = data[0].strip().split(" ")[-1]
    m_sensorName = data[1].strip().split(" ")[-2]

    if mode == "rgb":
        m_Width = int(data[2].strip().split(" ")[-1])
        m_Height = int(data[3].strip().split(" ")[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(" ")[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(float(data[4].strip().split(" ")[-1]))
        m_Height = int(float(data[5].strip().split(" ")[-1]))
        m_Shift = int(float(data[6].strip().split(" ")[-1]))
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(" ")[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))

    m_frames_size = int(float(data[11].strip().split(" ")[-1]))

    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=np.matrix(m_intrinsic),
        m_frames_size=m_frames_size,
    )


def process_scannet(data: Path, output_dir: Path, test_dir: Path):
    """Process scannet data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """
    meta = load_from_json(data / Path("dslr/nerfstudio/transforms_undistorted.json"))
    # convert mesh to triangle mesh (open3d can only read triangle meshes)
    mesh_path = data / Path("scans/mesh_aligned_0.05.ply")
    mesh_path = Path(mesh_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    ms.apply_filter("meshing_poly_to_tri")
    os.makedirs(output_dir, exist_ok=True)
    ms.save_current_mesh(str(output_dir / mesh_path.name), save_vertex_normal=True)

    verbose = True
    num_downscales = 0
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size = 250
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # get test imge names
    test_image_filenames = sorted([f.stem + ".JPG" for f in test_dir.glob("*.json")])

    image_filenames = []
    mask_filenames = []
    poses = []
    i_train = []
    i_test_openhype = []
    i_eval = []
    # sort the frames by fname
    frames = meta["frames"] + meta["test_frames"]
    test_frames = [f["file_path"] for f in meta["test_frames"]]
    frames.sort(key=lambda x: x["file_path"])

    for idx, frame in enumerate(frames):
        filepath = Path(frame["file_path"])
        fname = data / Path("dslr/undistorted_images") / filepath

        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))

        if frame["file_path"] in test_image_filenames:  # our testset
            i_test_openhype.append(idx)
        elif frame["file_path"] in test_frames:
            i_eval.append(idx)  # not used
        else:
            i_train.append(idx)

    assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames))

    if "orientation_override" in meta:
        orientation_method = meta["orientation_override"]
        CONSOLE.log(
            f"[yellow] Dataset is overriding orientation method to {orientation_method}"
        )
    else:
        orientation_method = "up"

    poses = torch.from_numpy(np.array(poses).astype(np.float32))

    indices = i_train

    image_filenames_train = [image_filenames[i] for i in indices]
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    poses_train = poses[idx_tensor]

    num_images = len(image_filenames_train)

    idx = np.arange(num_images)
    if max_dataset_size != -1 and num_images > max_dataset_size:
        idx = np.round(np.linspace(0, num_images - 1, max_dataset_size)).astype(int)

    image_filenames_train = [image_filenames_train[i] for i in idx]
    poses_train = poses_train[idx]

    image_filenames_test = [image_filenames[i] for i in i_test_openhype]
    # copy test frames to our labels folders
    for img_path in image_filenames_test:
        shutil.copy(img_path, test_dir / img_path.name)

    poses_test = poses[i_test_openhype]

    copied_image_paths_train = process_data_utils.copy_images_list(
        image_filenames_train,
        image_dir=image_dir,
        image_prefix="frame_train_",
        verbose=verbose,
        num_downscales=num_downscales,
    )
    copied_image_paths_test = process_data_utils.copy_images_list(
        image_filenames_test,
        image_dir=image_dir,
        image_prefix="frame_eval_",
        verbose=verbose,
        keep_image_dir=True,
        num_downscales=num_downscales,
    )

    copied_image_paths_test_train = copied_image_paths_test + copied_image_paths_train
    poses_test_train = torch.cat((poses_test, poses_train), dim=0)

    # if "camera_model" in meta:
    #     camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
    # else:
    #     camera_type = CameraType.PERSPECTIVE

    frames = []
    for i, im_path in enumerate(copied_image_paths_test_train):
        c2w = poses_test_train[i, :3, :4]
        # pad 0 0 0 1 to make it 4x4
        c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        frame = {
            "file_path": im_path.as_posix(),
            # "depth_file_path": copied_depth_paths[i].as_posix(),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    fx = float(meta["fl_x"])
    fy = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    height = int(meta["h"])
    width = int(meta["w"])
    out = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "k1": float(meta["k1"]) if "k1" in meta else 0.0,
        "k2": float(meta["k2"]) if "k2" in meta else 0.0,
        "k3": float(meta["k3"]) if "k3" in meta else 0.0,
        "k4": float(meta["k4"]) if "k4" in meta else 0.0,
        "p1": float(meta["p1"]) if "p1" in meta else 0.0,
        "p2": float(meta["p2"]) if "p2" in meta else 0.0,
        "ply_file_path": mesh_path.as_posix(),
        "camera_model": CAMERA_MODELS[
            CAMERA_MODEL_TO_TYPE[meta["camera_model"]].name.lower()
        ].name,
    }

    out["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    return len(frames)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the scannetpp dataset.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
        required=True,
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        help="Path to the unpacked OopenHype benchmark annotations directory.",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for scene in scannetpp.scenes:

        data = f"{args.data}/{scene}"
        output_dir = f"{args.ouput_dir}/{scene}"
        test_dir = f"{args.test_dir}/{scene}"
        process_scannet(Path(data), Path(output_dir), Path(test_dir))
