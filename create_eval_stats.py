from collections import defaultdict
import json
from pathlib import Path
import tyro
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)
MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


def main(
    input_dpath: str,
    output_dpath: str,
    scene_id_list: list[str],
    exp_batch_name: str,  # exp_1 eg.g
    results_folder_prefix: str = "eval_output_run",
):
    """This function collects all eval results from multiple scenes and multiple runs per scene and aggregates them into final statistics per experiment type, scene, metric and part type.
    Args:
        input_dpath (str): Parent folder where all scene folders are saved. eg. openhyp_output/scannetpp
        output_dpath (str): Output directory path, where to save the results dict.
        scene_id_list (list[str]): List of scene ids to process.
        exp_batch_name (str): Experiment batch name. eg. exp_1, defines the subfolder in each scene folder where the results are stored.
        results_folder_prefix (str, optional): Prefix of the results folders to look for. Defaults to "eval_output_run".
    """

    # 1. create output folder
    output_dpath = Path(output_dpath) / "eval_aggregated_output" / exp_batch_name
    if not output_dpath.exists():
        output_dpath.mkdir(parents=True)

    result_dict = {}
    # 2. collect all result dictionaries of all scenes and runs:
    # get all run output folders and create a dictionary with the scene_id as key and the list of matching directories as value
    for scene_id in scene_id_list:
        result_dict[scene_id] = {}
        # find all
        search_dir = Path(input_dpath) / scene_id / exp_batch_name
        # find all directories that start with the prefix
        run_dir_list = [
            p
            for p in search_dir.iterdir()
            if p.is_dir() and p.name.startswith(results_folder_prefix)
        ]
        for run_dir in run_dir_list:
            result_dict[scene_id][run_dir.name] = {}
            # find  all all_frames_results_dict.json files in the run_dir
            result_json_fpaths = run_dir.rglob("all_frames_results_dict.json")
            for result_json_fpath in result_json_fpaths:
                # read the json file
                with result_json_fpath.open("r") as f:
                    result_dict[scene_id][run_dir.name][
                        str(result_json_fpath.relative_to(run_dir).parent).replace(
                            "/", "_"
                        )
                    ] = json.load(f)

    # #  save this dict inbetween
    # # this dict contains all results for all scenes and all runs, basically the individual results dicts collected
    # # scene_id -> run_name -> experiment_name -> frame_id -> metrics -> list of values
    # with (output_dpath / "all_runs_results_dict.json").open("w") as f:
    #     json.dump(result_dict, f, indent=4)  # `indent=4` makes it nicely formatted

    all_results_single_values = {}

    # we assume all have same aggregation methods
    for scene_id, run_dict in result_dict.items():
        for run, exp_dict in run_dict.items():
            for exp in exp_dict.keys():

                if exp not in all_results_single_values.keys():
                    all_results_single_values[exp] = {}
                if scene_id not in all_results_single_values[exp].keys():
                    all_results_single_values[exp][scene_id] = {}
                if "all_scenes" not in all_results_single_values[exp].keys():
                    all_results_single_values[exp]["all_scenes"] = {}

                if run not in all_results_single_values[exp][scene_id].keys():
                    all_results_single_values[exp][scene_id][run] = {}

                iou_list = []
                acc_list = []
                is_part_list = []
                # get the results for all frames
                for frame, frame_results in exp_dict[exp].items():
                    iou_list.extend(frame_results["iou"])
                    acc_list.extend(frame_results["acc"])
                    is_part_list.extend(frame_results["is_part"])

                all_results_single_values[exp][scene_id][run]["iou"] = iou_list
                all_results_single_values[exp][scene_id][run]["acc"] = acc_list
                all_results_single_values[exp][scene_id][run]["is_part"] = is_part_list
                all_results_single_values[exp][scene_id][run]["scene_counter"] = 1

                if run not in all_results_single_values[exp]["all_scenes"].keys():
                    all_results_single_values[exp]["all_scenes"][run] = {}
                    all_results_single_values[exp]["all_scenes"][run]["iou"] = []
                    all_results_single_values[exp]["all_scenes"][run]["acc"] = []
                    all_results_single_values[exp]["all_scenes"][run]["is_part"] = []
                    all_results_single_values[exp]["all_scenes"][run][
                        "scene_counter"
                    ] = 0

                all_results_single_values[exp]["all_scenes"][run]["iou"].extend(
                    iou_list
                )
                all_results_single_values[exp]["all_scenes"][run]["acc"].extend(
                    acc_list
                )
                all_results_single_values[exp]["all_scenes"][run]["is_part"].extend(
                    is_part_list
                )
                all_results_single_values[exp]["all_scenes"][run]["scene_counter"] += 1

    # #  save this dict inbetween
    # # this dict contains all results for all scenes and all runs, but only single value lists instead a list per frame
    # # experiment name -> scene_id -> run_name -> metric_name -> list of values
    # with (output_dpath / "all_runs_results_single_values.json").open("w") as f:
    #     json.dump(
    #         all_results_single_values, f, indent=4
    #     )  # `indent=4` makes it nicely formatted

    # 3.  aggregate over frames and scenes and create final stats per run
    all_runs_final_values = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for exp_type, scenes in all_results_single_values.items():
        for scene_id, runs in scenes.items():
            for run, metrics in runs.items():
                acc = np.array(metrics["acc"])
                iou = np.array(metrics["iou"])
                ispart = np.array(metrics["is_part"])  # 1=part, 0=object
                n_scenes = metrics["scene_counter"]

                # Masking
                mask_all = np.ones_like(ispart, dtype=bool)
                mask_obj = ispart == 0
                mask_part = ispart == 1

                # Prepare slices
                for metric_name, values in [("acc", acc), ("iou", iou)]:
                    for part_name, mask in [
                        ("all", mask_all),
                        ("objects", mask_obj),
                        ("parts", mask_part),
                    ]:
                        selected = values[mask]
                        if selected.size == 0:
                            print(
                                f"Warning: No values for {exp_type}, {scene_id}, {metric_name}, {part_name}"
                            )
                            continue  # skip if no values

                        all_runs_final_values[exp_type][scene_id][metric_name][
                            part_name
                        ][run] = float(np.mean(selected))

                        all_runs_final_values[exp_type][scene_id][metric_name][
                            "n_scenes"
                        ][run] = n_scenes

    # aggregate over runs to get final stats per experiment and scene
    all_runs_aggregated_final_values = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
    )

    for exp, scenes in all_runs_final_values.items():
        for scene, metrics in scenes.items():
            for metric, part_types in metrics.items():
                for part_type, runs in part_types.items():
                    values = list(runs.values())
                    run_count = len(values)
                    if not values:
                        continue
                    stats = {
                        "count": run_count,
                        "value": float(np.mean(values)),
                        "min_value": float(np.min(values)),
                        "max_value": float(np.max(values)),
                        "std_value": float(np.std(values)),
                    }
                    all_runs_aggregated_final_values[exp][scene][metric][
                        part_type
                    ] = stats

    # 3.5 save the final results
    with (output_dpath / "all_runs_aggregated_final_values.json").open("w") as f:
        json.dump(all_runs_aggregated_final_values, f, indent=4)


if __name__ == "__main__":

    tyro.cli(main)
