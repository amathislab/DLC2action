#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""SSL Experiments for the Sturman OFT dataset"""

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dlc2action.project import Project


@dataclass
class ProjectConfig:
    projects_path: str
    project_name: str
    data_type: str
    annotation_type: str
    data_path: str
    annotation_path: str


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed=seed)


def train_model(
    config: ProjectConfig,
    model: str,
    parameters_updated: dict,
    parameters: dict,
    train_indices: List[List[int]],
    seed: Optional[int] = None,
    epochs: int = 100,
    refine_epochs: int = 0,
    val_for_ssl: bool = False,
) -> List[str]:
    if seed is not None:
        set_seeds(seed)

    print(config.data_path)
    project = Project(
        name=config.project_name,
        data_type=config.data_type,
        annotation_type=config.annotation_type,
        data_path=config.data_path,
        annotation_path=config.annotation_path,
        projects_path=config.projects_path,
    )
    project.update_parameters(parameters_updated)
    episodes = []
    for train_idx in train_indices:

        if "pairwise_masked" in model:
            if not(train_idx == list(range(9,9+5)) or train_idx == list(range(12,12+5))):
                continue

        train_idx_str = ",".join([str(i) for i in train_idx])
        train_idx_pretty_str = "-".join([str(i) for i in train_idx])
        episode_name = f"abench_{model}_trainIdx_{train_idx_pretty_str}"
        episode_parameters = copy.deepcopy(parameters)
        episode_parameters["training"]["num_epochs"] = epochs

        if val_for_ssl:
            episode_parameters["training"][
                "partition_method"
            ] = f"leave-n-in:{train_idx_str}:val-for-ssl"
        else:
            episode_parameters["training"][
                "partition_method"
            ] = f"leave-n-in:{train_idx_str}:none"

        project.run_episode(
            episode_name,
            parameters_update=episode_parameters,
            force=True,
        )

        if refine_epochs == 0:
            episodes.append(episode_name)
        else:
            refined_episode_name = episode_name + f"_refine"
            episodes.append(refined_episode_name)
            episode_parameters["training"]["num_epochs"] = refine_epochs
            episode_parameters["training"]["lr"] = 1e-5
            project.run_episode(
                refined_episode_name,
                load_episode=episode_name,
                parameters_update=episode_parameters,
                force=True,
            )

    return episodes


def compile_results(
    project: Project,
    episodes: Dict[str, List[str]],
    behavior_indices: List[int],
    average: bool = True,
):
    results, stds = [], []
    for model, episodes in episodes.items():
        summ = project.get_summary(episodes, method="last")
        if average:
            results.append(
                [np.mean([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])]
            )
            std_arr = np.array(
                [summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0]
            )
            stds.append([np.sqrt(np.sum(std_arr**2)) / len(std_arr)])
        else:
            results.append(
                [summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0]
            )
            stds.append([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])

    return results, stds


def plot_results(
    model_name: str,
    models: List[str],
    behaviors: List[str],
    results: List[List],
    stds: List[List],
    average: bool = True,
):
    metric_label = "f1"
    colors = [
        "#99d096",
        "#ea678e",
        "#f9ba5b",
        "#639cd2",
        "#F1F285",
        "#B16CB9",
        "#ABE3CE",
        "#DD98A5",
        "#C44F53",
        "#BCC144",
        "#D6AF85",
    ]

    results_by_class = {i: [x[i] for x in results] for i in range(len(results[0]))}
    std_by_class = {i: [x[i] for x in stds] for i in range(len(results[0]))}

    mean_results = [np.mean(x) for x in results]
    indices = list(np.argsort(mean_results))[::-1]

    font = {"size": 20}

    matplotlib.rc("font", **font)

    x = np.arange(len(models))  # the label locations
    width = 0.5 if average else 0.2  # the width of the bars

    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots()
    N = len(results[0])
    dist_arr = [width * n / 2 for n in np.arange(-(N - 1), N, 2)]
    for i in range(len(results[0])):
        rects = ax.bar(
            x + dist_arr[i],
            np.array(results_by_class[i])[indices],
            width,
            yerr=np.array(std_by_class[i])[indices],
            label=behaviors[i],
        )
        if not average:
            for rect in rects:
                rect.set_color(colors[i])
        else:
            for j, rect in enumerate(rects):
                if np.array(models)[indices][j] != "published":
                    color = colors[0]
                else:
                    color = colors[1]
                rect.set_color(color)

    model_labels = []
    for model in models:
        model_labels.append(model)
    plt.xticks(ticks=x, labels=np.array(model_labels)[indices], rotation=70)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.ylabel(metric_label)
    plt.tight_layout()
    plt.savefig(f"{model_name}.jpg")


PARAMS_UPDATED = {
    "data": {
        "canvas_shape": [928, 576],
        "data_suffix": {"DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv"},
        "annotation_suffix": ".csv",
        "behaviors": ["Grooming", "Supported", "Unsupported"],
        "ignored_bodyparts": {"tl", "tr", "br", "bl", "centre"},
        "likelihood_threshold": 0.8,
        "len_segment": 256,
        "overlap": 200,
        "filter_background": False,
        "filter_annotated": False,
        "fps": 25,
        "clip_frames": 0,
        "normalize": True,
    },
    "general": {
        "model_name": "ms_tcn3",
        "exclusive": True,
        "only_load_annotated": True,
        "metric_functions": {"f1", "accuracy"},
        "dim": 2,
    },
    "metrics": {
        "recall": {"ignored_classes": {}, "average": "macro"},
        "precision": {"ignored_classes": {}, "average": "macro"},
        "f1": {"ignored_classes": {}, "average": "none"},
    },
    "training": {
        "num_epochs": 100,
        "device": "cuda:1",
        "lr": 1e-4,
        "batch_size": 256,
        "ssl_weights": {"pairwise": 0.01, "contrastive_regression": 1},
        "augment_train": 1,
        "skip_normalization_keys": ["speed_direction", "coord_diff"],
        "to_ram": False,
    },
    "losses": {
        "ms_tcn": {
            "weights": "dataset_inverse_weights",
            "gamma": 2.5,
            "alpha": 0.001,
        }
    },
    "augmentations": {
        "augmentations": {"add_noise", "mirror"},
        "mirror_dim": {0, 1},
        "noise_std": 0.001,
        "canvas_shape": [928, 576],
    },
    "features": {
        "egocentric": True,
        "distance_pairs": None,
        "keys": {
            "intra_distance",
            "angle_speeds",
            "areas",
            "coord_diff",
            "acc_joints",
            "center",
            "speed_direction",
            "speed_value",
            "zone_bools",
        },
        "angle_pairs": [
            ["tailbase", "tailcentre", "tailcentre", "tailtip"],
            ["hipr", "tailbase", "tailbase", "hipl"],
            ["tailbase", "bodycentre", "bodycentre", "neck"],
            ["bcr", "bodycentre", "bodycentre", "bcl"],
            ["bodycentre", "neck", "neck", "headcentre"],
            ["tailbase", "bodycentre", "neck", "headcentre"],
        ],
        "area_vertices": [
            ["tailbase", "hipr", "hipl"],
            ["hipr", "hipl", "bcl", "bcr"],
            ["bcr", "earr", "earl", "bcl"],
            ["earr", "nose", "earl"],
        ],
        "neighboring_frames": 0,
        "zone_vertices": {"arena": ["tl", "tr", "br", "bl"]},
        "zone_bools": [["arena", "nose"], ["arena", "headcentre"]],
    },
}

PARAMS_BASE_C2F_TCN = {
    "general": {
        "len_segment": 512,
        "overlap": 400,
        "model_name": "c2f_tcn",
        "ssl": [],
        "metric_functions": {"f1", "accuracy"},
    },
    "training": {
        "temporal_subsampling_size": 0.85,
        "ssl_on": False,
    },
    "model": {
        "num_f_maps": 128,
        "feature_dim": 256,  # int; if not null, intermediate features are generated with
        # this dim and then passed to a 2-layer MLP for classification (good for SSL)
    },
}

PARAMS_SSL_C2F_TCN = {
    "general": {
        "len_segment": 512,
        "overlap": 400,
        "model_name": "c2f_tcn",
        "ssl": ["contrastive"],
        "metric_functions": {"f1", "accuracy"},
    },
    "training": {
        "temporal_subsampling_size": 1.0,
        "ssl_on": True,
        "ssl_weights": {
            "contrastive": 0.01,
        },
    },
    "model": {
        "num_f_maps": 128,
        "feature_dim": 256,  # int; if not null, intermediate features are generated with
        # this dim and then passed to a 2-layer MLP for classification (good for SSL)
    },
    "ssl": {
        "contrastive": {
            "num_f_maps": "model_features",
        },
    },
}


def main(ssl_type="contrastive", train_base=True, ssl_weights=[0.01]):
    root = Path("/media1/data/andy")
    device = "cuda:1"

    model = "c2f_tcn"
    batch_size = 256
    lr = 1e-3

    PARAMS_UPDATED["training"]["device"] = device
    PARAMS_UPDATED["training"]["lr"] = lr
    PARAMS_UPDATED["training"]["batch_size"] = batch_size
    PARAMS_UPDATED["training"]["augment_train"] = 1
    # OTHERWISE OTHER VIDEOS FROM OFT ARE LOADED (UNANNOTATED)
    PARAMS_UPDATED["general"]["only_load_annotated"] = True

    PARAMS_SSL_C2F_TCN["ssl"][ssl_type] = {"num_f_maps": "model_features"}
    print(f"TRAINING MODELS {model}")
    config = ProjectConfig(
        projects_path=str(root / "DLC2Action"),
        project_name="oft_ssl",
        data_type="dlc_track",
        annotation_type="csv",
        data_path=str(root / "OFT/OFT/Output_DLC"),
        annotation_path=str(root / "OFT/OFT/Labels"),
    )

    train_indices_n1 = [[i] for i in range(5)]
    train_indices_n2 = [[i, i + 1] for i in [0, 2, 4, 6, 8]]
    train_indices_n5 = [list(range(i, i + 5)) for i in [0, 3, 6, 9, 12]]
    train_indices = train_indices_n1 + train_indices_n2 + train_indices_n5
    for idx in train_indices:
        print(idx)

    for ssl_weight in ssl_weights:

        if ssl_type == "pairwise_masked":
            if ssl_weight != 0.005:
                continue

        episodes = {}
        for ssl, model_name, parameters, epochs, refine_epochs, val_for_ssl in [
            (False, "c2ftcn_base", PARAMS_BASE_C2F_TCN, 60, 5, False),
            (True, f"c2ftcn_{ssl_type}", PARAMS_SSL_C2F_TCN, 60, 5, True),
        ]:
            if not train_base and not ssl:
                continue
            if not ssl:
                try:
                    episodes[model_name] = train_model(
                        config,
                        model=model_name,
                        parameters_updated=PARAMS_UPDATED,
                        parameters=parameters,
                        train_indices=train_indices,
                        seed=0,
                        epochs=epochs,
                        refine_epochs=refine_epochs,
                        val_for_ssl=val_for_ssl,
                    )
                except Exception as e:
                    print(f"Failed {ssl_type} with {e}")
            else:
                weight_str = str(ssl_weight).replace(".", "_")
                model_name_with_weight = f"{model_name}_w{weight_str}"
                print(f"training {model_name_with_weight}")
                try:
                    model_name_with_weight = f"{model_name}_w{weight_str}"
                    parameters["general"]["ssl"] = [ssl_type]
                    parameters["training"]["ssl_weights"][ssl_type] = ssl_weight
                    episodes[model_name_with_weight] = train_model(
                        config,
                        model=model_name_with_weight,
                        parameters_updated=PARAMS_UPDATED,
                        parameters=parameters,
                        train_indices=train_indices,
                        seed=0,
                        epochs=epochs,
                        refine_epochs=refine_epochs,
                        val_for_ssl=val_for_ssl,
                    )
                except Exception as e:
                    print(f"Failed {ssl_type} with {e} and weight {ssl_weight}")

    # Plotting results
    # project = Project(
    #     name=config.project_name,
    #     data_type=config.data_type,
    #     annotation_type=config.annotation_type,
    #     data_path=config.data_path,
    #     annotation_path=config.annotation_path,
    #     projects_path=config.projects_path,
    # )
    # average = True
    # models = [m for m in episodes.keys()]
    # all_behaviors = project.get_behavior_dictionary(episodes[models[0]][0]).values()
    # behaviors = [x for x in all_behaviors if x != "other"]
    # behavior_indices = list(range(1, len(behaviors) + 1))
    # results, stds = compile_results(project, episodes, behavior_indices, average=average)
    # plot_results(model, models, behaviors, results, stds, average=average)


if __name__ == "__main__":
    ssl_weights = [0.2,0.3,0.4,0.5]
    ssl_types = [
        "contrastive",
        # "contrastive_regression",
        # "contrastive_masked",
        "pairwise",
        # "pairwise_masked",
        # "masked_features",
        # "masked_joints",
        # "masked_frames",
        # "reverse",
        # "order",
        # "tcc",
    ]
    for ssl_type in ssl_types:
        try:
            main(ssl_type, train_base=False, ssl_weights=ssl_weights)
        except:
            print(f"Failed for {ssl_type}")
