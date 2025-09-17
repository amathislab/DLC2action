#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""SSL Benchmark: MS-TCN3 with Contrastive SSL"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dlc2action.project import Project


def hyperparameter_search(
    model_name: str,
    projects_path: str,
    project_name: str,
    data_path: str,
    annotation_path: str,
    train_idx: int,
    parameters_updated: dict,
    parameters_ssl: dict,
    search_space: dict,
    n_trials: int = 10,
    best_n: int = 1,
    seed: Optional[int] = None,
):
    if seed is not None:
        set_seeds(seed)

    parameters_updated["training"]["num_epochs"] = 30
    parameters_updated["training"]["partition_method"] = f"leave-one-in:{train_idx}:val-for-ssl"

    project = Project(
        name=project_name,
        data_type="dlc_track",
        annotation_type="csv",
        data_path=data_path,
        annotation_path=annotation_path,
        projects_path=projects_path,
    )
    project.update_parameters(parameters_updated)
    project.run_hyperparameter_search(
        search_name=f"{model_name}_ssl_search_idx{train_idx}",
        search_space=search_space,
        metric="f1",
        n_trials=n_trials,
        best_n=best_n,
        parameters_update=parameters_ssl,
        direction="maximize",
        prune=True,
    )


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
        "metric_functions": {'accuracy', 'f1'},
        "dim": 2,
    },
    "metrics": {
        "recall": {"ignored_classes": {}, "average": "macro"},
        "precision": {"ignored_classes": {}, "average": "macro"},
        "f1": {"ignored_classes": {0}, "average": "macro"},
    },
    "training": {
        "num_epochs": 100,
        "device": "cuda",
        "lr": 1e-4,
        "batch_size": 256,
        "ssl_weights": {"pairwise": 0.01, "contrastive_regression": 1},
        "augment_train": 1,
        "skip_normalization_keys": ["speed_direction", "coord_diff"],
        "to_ram": False
    },
    "losses": {
        "ms_tcn": {
            "weights": 'dataset_inverse_weights',
            "gamma": 2.5,
            "alpha": 0.001,
        }
    },
    "augmentations": {
        "augmentations": {"add_noise", "mirror"},
        "mirror_dim": {0, 1},
        "noise_std": 0.001,
        "canvas_shape": [928, 576]
    },
    "features": {
        "egocentric": True,
        "distance_pairs": None,
        "keys": {
            "intra_distance", "angle_speeds", "areas", "coord_diff",
            "acc_joints", "center", "speed_direction", "speed_value", "zone_bools"
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
        "zone_bools": [["arena", "nose"], ["arena", "headcentre"]]
    },
}

PARAMS_BASE_C2F_TCN = {
    "general": {
        "len_segment": 512,
        "overlap": 400,
        "model_name": "c2f_tcn",
        "ssl": [],
    },
    "training": {
        "temporal_subsampling_size": 0.85,
        "ssl_on": False,
    },
    "model": {
        "num_f_maps": 64,
        "feature_dim": 64  # int; if not null, intermediate features are generated with
        # this dim and then passed to a 2-layer MLP for classification (good for SSL)
    },
}

PARAMS_SSL_C2F_TCN = {
    "general": {
        "len_segment": 512,
        "overlap": 400,
        "model_name": "c2f_tcn",
        "ssl": ["contrastive"],
    },
    "training": {
        "temporal_subsampling_size": 1.0,
        "ssl_on": True,
        "ssl_weights": {
            "contrastive": 1e-1,
        },
    },
    "model": {
        "num_f_maps": 64,
        "feature_dim": 64  # int; if not null, intermediate features are generated with
        # this dim and then passed to a 2-layer MLP for classification (good for SSL)
    },
    "ssl": {
        "contrastive": {
            "num_f_maps": "model_features",
        },
    },
}

SEARCH_SPACE = {
    "ssl_weights/contrastive": ("categorical", [1e-3, 1e-2, 1e-1]),
    "training/lr": ("categorical", [1e-3, 1e-4, 1e-5]),
    "model/num_f_maps": ("categorical", [32, 64, 128, 256]),
    "model/feature_dim": ("categorical", [16, 32, 64, 128, 256]),
}


def main():
    root = Path("/home/niels/files")
    device = "cuda:0"

    model = "c2f_tcn"
    batch_size = 16
    lr = 5e-4

    PARAMS_UPDATED["training"]["device"] = device
    PARAMS_UPDATED["training"]["lr"] = lr
    PARAMS_UPDATED["training"]["batch_size"] = batch_size
    PARAMS_UPDATED["training"]["augment_train"] = 1

    # OTHERWISE OTHER VIDEOS FROM OFT ARE LOADED (UNANNOTATED)
    PARAMS_UPDATED["general"]["only_load_annotated"] = True

    (root / "datasets/projects").mkdir(exist_ok=True, parents=True)
    print(f"TRAINING MODELS {model}")
    config = ProjectConfig(
        projects_path=str(root / "dlc2a_projects"),
        project_name="oft",
        data_type="dlc_track",
        annotation_type="csv",
        data_path=str(root / "datasets/OFT/Output_DLC"),
        annotation_path=str(root / "datasets/OFT/Labels"),
    )

    hyperparameter_search(
        model_name=model,
        projects_path=config.projects_path,
        project_name=config.project_name,
        data_path=config.data_path,
        annotation_path=config.annotation_path,
        train_idx=0,
        parameters_updated=PARAMS_UPDATED,
        parameters_ssl=PARAMS_SSL_C2F_TCN,
        search_space=SEARCH_SPACE,
        n_trials=20,
        best_n=5,
        seed=0,
    )


if __name__ == "__main__":
    main()
