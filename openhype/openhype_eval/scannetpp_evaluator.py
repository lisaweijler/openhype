from collections import defaultdict
from dataclasses import asdict, dataclass
import glob
import json
import os
import torchvision
from tqdm import tqdm

from einops import rearrange
from pathlib import Path
from typing import Dict, Union
import numpy as np
import torch
import cv2
from PIL import Image

# from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE

torch.autograd.set_detect_anomaly(True)

from openhype.utils.io import (
    read_json,
    save_config,
    save_dict_as_json,
)
from openhype.utils.utils import (
    check_files_exist,
    smooth,
    colormap_saving,
    vis_mask_save,
    show_result,
)
from openhype.utils import lorentz as L
import openhype.utils.colormaps as colormaps

from openhype.openhype_eval.base_evaluator import (
    BaseEvaluater,
    BaseEvaluaterConfig,
)

from openhype.openhype_eval.evaluator_registry import (
    register_evaluator,
    register_evaluator_config,
)

MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


@register_evaluator_config("scannetpp_evaluator")
@dataclass
class ScannetppDataEvaluatorConfig(BaseEvaluaterConfig):
    img_dir: str = ""
    gt_path: str = ""
    mask_thresh: float = 0.4  # like langsplat
    verbose_visualization: bool = True


@register_evaluator("scannetpp_evaluator")
class ScannetppDataEvaluator(BaseEvaluater):
    def __init__(self, config: ScannetppDataEvaluatorConfig, device="cuda:0"):
        super().__init__(config, device)
        self.colormap_options = colormaps.ColormapOptions(
            colormap="turbo",
            normalize=True,
            colormap_min=-1.0,
            colormap_max=1.0,
        )

    def _render_eval_data(
        self,
        nerf_config_path: Union[str, Path],
        rendered_data_dir: Union[str, Path],
    ):
        from nerfstudio.utils.eval_utils import (
            eval_setup,
        )  # lazy import to not have circular import error

        dataparser_transforms = read_json(
            Path(nerf_config_path).parent / "dataparser_transforms.json"
        )

        global_transform = torch.tensor(dataparser_transforms["transform"])

        # get camera data
        c2ws = []
        frame_names_img = []
        flx = []
        fly = []
        cx = []
        cy = []
        camera_model = []
        w = []
        h = []
        gt_ann, _, _ = self.gt_data
        for gt_dict_frame in gt_ann.values():
            gt_dict = gt_dict_frame["camera_data"]
            frame_names_img.append(gt_dict["name"].split(".")[0])
            c2w = torch.tensor(gt_dict["transform_matrix"]).view(4, 4)  # [:3]
            flx.append(gt_dict["fl_x"])
            fly.append(gt_dict["fl_y"])
            cx.append(gt_dict["cx"])
            cy.append(gt_dict["cy"])
            camera_model.append(gt_dict["camera_model"])
            w.append(gt_dict["width"])
            h.append(gt_dict["height"])

            c2ws.append(c2w)

        camera_to_worlds = torch.stack(c2ws, dim=0)
        camera_to_worlds_oriented = global_transform @ camera_to_worlds
        camera_to_worlds_oriented[:, :3, 3] *= dataparser_transforms["scale"]

        # get cameras
        cameras = Cameras(
            fx=torch.tensor(flx),
            fy=torch.tensor(fly),
            cx=torch.tensor(cx),
            cy=torch.tensor(cy),
            camera_type=[
                CAMERA_MODEL_TO_TYPE[c] for c in camera_model
            ],  # opencv is same as perspective
            camera_to_worlds=camera_to_worlds_oriented,
            width=torch.tensor(w),
            height=torch.tensor(h),
        )

        # get trained nerf model
        _, pipeline, _, _ = eval_setup(
            Path(nerf_config_path),
            eval_num_rays_per_chunk=2048,
            test_mode="inference",
        )

        cameras = cameras.to(pipeline.device)

        rendered_data_dir.mkdir(parents=True, exist_ok=True)

        for camera_idx in tqdm(range(cameras.size)):

            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, obb_box=None, aabb_box=None
            )

            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle
                )

            rgb = outputs["rgb"]  # # h x w x 3
            openhype = outputs["openhype"]  # h x w x 768 or 512

            with open(
                rendered_data_dir / (frame_names_img[camera_idx] + ".JPG"),
                "wb",
            ) as f:
                torchvision.utils.save_image(rgb.detach().cpu().permute(2, 0, 1), f)

            with open(
                rendered_data_dir / (frame_names_img[camera_idx] + "_openhype.npy"),
                "wb",
            ) as f:
                np.save(f, openhype.detach().cpu().numpy())

    def get_rendered_eval_data(self):

        rendered_data_dir = (
            Path(self.config.nerf_config_path).parent / "rendered_eval_data"
        )

        rendered_data_dir.mkdir(parents=True, exist_ok=True)

        # get all eval images for scene
        frame_names_img = []
        gt_ann, _, _ = self.gt_data
        for eval_idx in gt_ann.keys():
            frame_names_img.append(gt_ann[eval_idx]["camera_data"]["name"])

        frame_names_vlf = [
            f.split(".")[0] + "_openhype.npy" for f in frame_names_img
        ]  # vision languag features
        # check if already rendered
        if not check_files_exist(
            frame_names_img + frame_names_vlf, rendered_data_dir
        ):  # all need to exist, ohterwise rerender
            self._render_eval_data(self.config.nerf_config_path, rendered_data_dir)

        rgb_renders = []
        openhype_renders = []
        for frame_name in frame_names_img:
            # rbg images
            image = Image.open(rendered_data_dir / frame_name)
            image_tensor = torchvision.transforms.ToTensor()(
                image
            )  # C, H, W and values between 0 and 1
            rgb_renders.append(
                image_tensor.permute(1, 2, 0)
            )  # H x W x C # to match dimension when it comes out of the nerf pipeline, rendered freshly
            # Vision language features
            with open(
                rendered_data_dir / (frame_name.split(".")[0] + "_openhype.npy"),
                "rb",
            ) as f:
                openhype_renders.append(torch.from_numpy(np.load(f)))

        rgb_tensor = torch.stack(rgb_renders, dim=0)
        openhype_tensor = torch.stack(openhype_renders, dim=0)

        eval_frames = [f.split(".")[0] for f in frame_names_img]
        return rgb_tensor, openhype_tensor, eval_frames

    def _get_gt_data(self) -> Dict:
        """Taken and adpated from LangSplat code.

        return:
            gt_ann: dict()
                keys: str(int(idx))
                values: dict()
                    keys: str(label)
                    values: dict() which contain 'bboxes' and 'mask'
        """
        gt_json_paths = sorted(
            glob.glob(os.path.join(str(self.config.gt_path), "*.json"))
        )
        img_paths = sorted(glob.glob(os.path.join(str(self.config.gt_path), "*.JPG")))
        gt_ann = {}
        for idx, js_path in enumerate(gt_json_paths):
            img_ann = defaultdict(dict)
            with open(js_path, "r") as f:
                gt_data = json.load(f)

            h, w = gt_data["info"]["height"], gt_data["info"]["width"]
            img_ann["camera_data"] = gt_data["info"]
            for prompt_data in gt_data["objects"]:
                label = prompt_data["category"]
                box = np.asarray(prompt_data["bbox"]).reshape(-1)  # x1y1x2y2
                if label not in img_ann:
                    img_ann[label]["path_to_mask"] = prompt_data["path_to_mask"]
                    img_ann[label]["is_part"] = prompt_data["is_part"]

                    mask = Image.open(
                        Path(self.config.gt_path) / prompt_data["path_to_mask"]
                    ).convert("L")

                    # Convert to NumPy array
                    mask = np.array(mask) // 255

                    img_ann[label]["mask"] = mask

                # get boxes
                if img_ann[label].get("bboxes", None) is not None:
                    img_ann[label]["bboxes"] = np.concatenate(
                        [img_ann[label]["bboxes"].reshape(-1, 4), box.reshape(-1, 4)],
                        axis=0,
                    )
                else:
                    img_ann[label]["bboxes"] = box

            gt_ann[f"{idx}"] = img_ann

        return gt_ann, (h, w), img_paths

    def _get_mask_segmentation_results(
        self,
        valid_map,
        image,
        prompts,
        aggregation_method,
        image_name: Path = None,
        img_ann: Dict = None,
        thresh: float = 0.4,
        do_plots: bool = True,
    ):
        """Taken and adapted from LangSplat code."""

        n_head, n_prompt, h, w = valid_map.shape  # n_head: only 1 here

        # positive prompts
        chosen_iou_list, chosen_lvl_list = [], []
        for k in range(n_prompt):
            iou_lvl = np.zeros(n_head)
            mask_lvl = np.zeros((n_head, h, w))

            for i in range(n_head):

                scale = 30
                kernel = np.ones((scale, scale)) / (scale**2)
                np_relev = valid_map[i][k].cpu().numpy()
                avg_filtered = cv2.filter2D(np_relev, -1, kernel)
                avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
                valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

                if do_plots:

                    output_path_relev = (
                        self.output_dir_experiment
                        / image_name
                        / "heatmap"
                        / aggregation_method
                        / f"{prompts[k]}_{i}"
                    )

                    output_path_relev.parent.mkdir(exist_ok=True, parents=True)
                    colormap_saving(
                        valid_map[i][k].unsqueeze(-1),
                        self.colormap_options,
                        output_path_relev,
                    )

                p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(
                    p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo")
                ).to(image.device)
                mask = (valid_map[i][k] < 0.5).squeeze().to(image.device)
                valid_composited[mask, :] = image[mask, :] * 0.3

                if do_plots:
                    output_path_compo = (
                        self.output_dir_experiment
                        / image_name
                        / "composited"
                        / aggregation_method
                        / f"{prompts[k]}_{i}"
                    )
                    output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                    colormap_saving(
                        valid_composited, self.colormap_options, output_path_compo
                    )

                # truncate the heatmap into mask
                output = valid_map[i][k]

                output = output - torch.min(output)
                output = output / (torch.max(output) + 1e-9)
                output = output * (1.0 - (-1.0)) + (-1.0)
                output = torch.clip(output, 0, 1)

                mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
                mask_pred = smooth(mask_pred)
                mask_lvl[i] = mask_pred
                mask_gt = img_ann[prompts[k]]["mask"].astype(np.uint8)

                # calculate iou
                intersection = np.sum(np.logical_and(mask_gt, mask_pred))
                union = np.sum(np.logical_or(mask_gt, mask_pred))
                iou = np.sum(intersection) / np.sum(union)
                iou_lvl[i] = iou

            score_lvl = torch.zeros((n_head,), device=valid_map.device)
            for i in range(n_head):
                score = valid_map[i, k].max()
                score_lvl[i] = score
            chosen_lvl = torch.argmax(score_lvl)

            chosen_iou_list.append(iou_lvl[chosen_lvl])
            chosen_lvl_list.append(chosen_lvl.cpu().numpy())

            # save for visualization
            save_path = (
                self.output_dir_experiment
                / image_name
                / aggregation_method
                / f"chosen_{prompts[k]}.png"
            )
            vis_mask_save(mask_lvl[chosen_lvl], save_path)

        return chosen_iou_list, chosen_lvl_list

    def _get_localization_results(
        self,
        valid_map,
        image,
        prompts,
        aggregation_method,
        image_name,
        img_ann,
        do_plots: bool = True,
    ):
        """Taken and adapted from LangSplat code."""

        valid_map = valid_map.repeat(2, 1, 1, 1)
        n_head, n_prompt, h, w = valid_map.shape

        # positive prompts
        correct = []

        for k in range(n_prompt):
            select_output = valid_map[:, k]

            scale = 30
            kernel = np.ones((scale, scale)) / (scale**2)
            np_relev = select_output.cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

            score_lvl = np.zeros((n_head,))
            coord_lvl = []
            for i in range(n_head):
                score = avg_filtered[..., i].max()
                coord = np.nonzero(avg_filtered[..., i] == score)
                score_lvl[i] = score
                coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])

            selec_head = np.argmax(score_lvl)
            coord_final = coord_lvl[selec_head]

            for box in img_ann[prompts[k]]["bboxes"].reshape(-1, 4):
                flag = 0
                x1, y1, x2, y2 = box
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                for cord_list in coord_final:
                    if (
                        cord_list[0] >= x_min
                        and cord_list[0] <= x_max
                        and cord_list[1] >= y_min
                        and cord_list[1] <= y_max
                    ):

                        flag = 1
                        break
                if flag != 0:
                    break
            correct.append(flag)

            avg_filtered = (
                torch.from_numpy(avg_filtered[..., selec_head])
                .unsqueeze(-1)
                .to(select_output.device)
            )
            torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
            p_i = torch.clip(torch_relev - 0.5, 0, 1)
            valid_composited = colormaps.apply_colormap(
                p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo")
            ).to(image.device)
            mask = (torch_relev < 0.5).squeeze().to(image.device)
            valid_composited[mask, :] = image[mask, :] * 0.3

            if do_plots:
                output_path_loca = (
                    self.output_dir_experiment
                    / image_name
                    / "localization"
                    / aggregation_method
                )
                output_path_loca.mkdir(exist_ok=True, parents=True)
                save_path = output_path_loca / f"{prompts[k]}.png"
                show_result(
                    valid_composited.cpu().numpy(),
                    coord_final,
                    img_ann[prompts[k]]["bboxes"],
                    save_path,
                )
        return correct

    def get_text_data(self, idx: int):
        gt_ann, _, _ = self.gt_data

        text_prompts = list(gt_ann[f"{idx}"].keys())
        text_prompts.remove("camera_data")
        text_embeddings = self.get_text_embeddings(text_prompts)
        return text_prompts, text_embeddings

    def get_results_single_frame(
        self,
        pixel_text_similarity_map,
        rgb_img,
        text_prompts,
        image_name,
        img_idx,
        aggregation_method,
        thresh,
        do_plots=True,
    ):
        gt_ann, _, _ = self.gt_data
        iou_list, lvl = self._get_mask_segmentation_results(
            pixel_text_similarity_map,  # 1 x n_prompts x H x W
            rgb_img,
            text_prompts,
            aggregation_method,
            image_name=image_name,
            img_ann=gt_ann[str(img_idx)],
            thresh=thresh,
            do_plots=do_plots,
        )
        # iou values for each prompt for this evaluation frame image stored in a list

        acc = self._get_localization_results(
            pixel_text_similarity_map,  # 1 x n_prompts x H x W
            rgb_img,
            text_prompts,
            aggregation_method,
            image_name,
            gt_ann[str(img_idx)],
            do_plots=do_plots,
        )
        is_object_part = [
            gt_ann[str(img_idx)][prompt]["is_part"] for prompt in text_prompts
        ]
        return {"iou": iou_list, "acc": acc, "is_part": is_object_part}

    def get_final_results(self, all_frame_results_dict, thresh):
        # save detailed results per frame
        save_dict_as_json(
            all_frame_results_dict,
            self.output_dir_experiment / "all_frames_results_dict.json",
        )

        metrics = {}
        metrics["iou"] = []
        metrics["acc"] = []
        metrics["is_part"] = []
        for frame_results in all_frame_results_dict.values():
            metrics["iou"].extend(frame_results["iou"])
            metrics["acc"].extend(frame_results["acc"])  # number of correct ones
            metrics["is_part"].extend(
                frame_results["is_part"]
            )  # number of correct ones

        iou_is_object = [
            iou
            for iou, is_part in zip(metrics["iou"], metrics["is_part"])
            if is_part == False
        ]
        acc_is_object = [
            acc
            for acc, is_part in zip(metrics["acc"], metrics["is_part"])
            if is_part == False
        ]
        iou_is_part = [
            iou for iou, is_part in zip(metrics["iou"], metrics["is_part"]) if is_part
        ]
        acc_is_part = [
            acc for acc, is_part in zip(metrics["acc"], metrics["is_part"]) if is_part
        ]

        overall_iou = sum(metrics["iou"]) / len(metrics["iou"])
        overall_acc = sum(metrics["acc"]) / len(metrics["acc"])

        part_iou = sum(iou_is_part) / len(iou_is_part)
        part_acc = sum(acc_is_part) / len(acc_is_part)

        object_iou = sum(iou_is_object) / len(iou_is_object)
        object_acc = sum(acc_is_object) / len(acc_is_object)

        print(f"trunc thresh: {str(thresh).replace('.', '_')}")
        print(f"iou chosen: {overall_iou:.8f}")
        print("Localization accuracy: " + f"{overall_acc:.8f}")
        print(f"iou chosen (part): {part_iou:.8f}")
        print("Localization accuracy (part): " + f"{part_acc:.8f}")
        print(f"iou chosen (object): {object_iou:.8f}")
        print("Localization accuracy (object): " + f"{object_acc:.8f}")

        with open(
            str(
                self.output_dir_experiment / (f"results_{self.config.aggregation}.txt")
            ),
            "w",
        ) as text_file:
            text_file.write(f"{Path(Path(self.config.img_dir).parent).stem}\n")
            text_file.write(f"trunc thresh: {str(thresh).replace('.', '_')}\n")
            text_file.write(f"iou chosen: {overall_iou:.8f}\n")
            text_file.write("Localization accuracy: " + f"{overall_acc:.8f}\n")
            text_file.write(f"iou chosen (part): {part_iou:.8f}\n")
            text_file.write("Localization accuracy (part): " + f"{part_acc:.8f}\n")
            text_file.write(f"iou chosen (object): {object_iou:.8f}\n")
            text_file.write("Localization accuracy (object): " + f"{object_acc:.8f}\n")

    def __call__(self, *args, **kwds):
        # save config - this call creates the output dir and saves the config there
        save_config(asdict(self.config), self.output_dir_experiment / "config.yaml")

        # load all jsons for eval frames
        rgb_renderings, openhype_renderings, eval_frames = self.get_rendered_eval_data()
        render_hight = openhype_renderings.shape[1]
        render_width = openhype_renderings.shape[2]
        print(render_hight, render_width)

        # here order of annotations and rendered openhype_tensor match,  since both are sorted .

        all_frames_results_dict = {}
        self.negative_text_embedings = self.get_text_embeddings(self.negatives)

        for idx, eval_frame in enumerate(tqdm(eval_frames, desc="Evaluate...")):
            print(f"Working on eval_frame: {eval_frame}")
            text_prompts, text_embeddings = self.get_text_data(idx)

            openhype_renderings_flattened = rearrange(
                openhype_renderings[idx], "h w d -> (h w) d"
            )

            # break it up due to memory, when decoding to clip space can be memory intensive,
            # clip_dim per
            parts = 20

            step = openhype_renderings_flattened.shape[0] // parts
            pixel_text_interpolated_sim_parts_list = []
            pixel_negatives_interpolated_sim_parts_list = []
            for i in range(parts + 1):
                if i == parts:
                    openhype_renderings_flattened_part = openhype_renderings_flattened[
                        i * step :
                    ]
                else:
                    openhype_renderings_flattened_part = openhype_renderings_flattened[
                        i * step : (i + 1) * step
                    ]
                if len(openhype_renderings_flattened_part) != 0:
                    # actual function that is split up for memory
                    pixel_text_interpolated_sim_part = (
                        self.get_pixel_text_interpolated_sim(
                            openhype_renderings_flattened_part,
                            text_embeddings,
                        )
                    )  # (H * W * interpolation steps) x n_prompts

                    # for negative prompts
                    pixel_negatives_interpolated_sim_part = (
                        self.get_pixel_text_interpolated_sim(
                            openhype_renderings_flattened_part,
                            self.negative_text_embedings,
                        )
                    )  # (H * W * interpolation steps) x n_prompts
                    pixel_negatives_interpolated_sim_parts_list.append(
                        pixel_negatives_interpolated_sim_part
                    )

                    del openhype_renderings_flattened_part
                    torch.cuda.empty_cache()
                    pixel_text_interpolated_sim_parts_list.append(
                        pixel_text_interpolated_sim_part
                    )

            pixel_text_interpolated_sim = torch.cat(
                pixel_text_interpolated_sim_parts_list
            )

            pixel_negatives_interpolated_sim = torch.cat(
                pixel_negatives_interpolated_sim_parts_list
            )

            # adapt sims with negative stepwise
            adapted_sims = []
            for i in range(pixel_text_interpolated_sim.shape[-1]):
                adapted_sims.append(
                    self.get_relevancy(
                        pixel_text_interpolated_sim,
                        pixel_negatives_interpolated_sim,
                        i,
                    )
                )
            pixel_text_interpolated_sim_with_neg_stepwise = torch.stack(
                adapted_sims, dim=-1
            )

            # adjust output dir for this threshold

            pixel_text_similarity_map = self.get_similarity_map(
                rearrange(
                    pixel_text_interpolated_sim_with_neg_stepwise,
                    "(h w s) p -> h w s p",
                    h=render_hight,
                    w=render_width,
                    s=self.config.interpolation_steps,
                ),
                self.config.aggregation,
                verbose_visualization=self.config.verbose_visualization,
                text_prompts=text_prompts,
                eval_frame_name=eval_frame,
            )

            all_frames_results_dict[eval_frame] = self.get_results_single_frame(
                pixel_text_similarity_map,
                rgb_renderings[idx],
                text_prompts,
                eval_frame,
                idx,
                self.config.aggregation,
                self.config.mask_thresh,
                do_plots=self.config.verbose_visualization,  # do_plots,
            )

        self.get_final_results(all_frames_results_dict, self.config.mask_thresh)
