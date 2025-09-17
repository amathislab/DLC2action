#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dlc2action.project import Project


@dataclass
class ProjectConfig:
    projects_path: str
    project_name: str
    data_type: str
    annotation_type: str
    data_path: str
    annotation_path: str
    output_path: str


features_only = False
if features_only:
    config = ProjectConfig(
        projects_path="path/to/projects",
        project_name="oft_videomae",
        data_type="features",
        annotation_type="csv",
        data_path="path/to/data",
        annotation_path="path/to/OFT/Labels",
        output_path="path/to/output/results",
    )
else:
    config = ProjectConfig(
        projects_path="path/to/projects",
        project_name="oft_videomae_cropped",
        data_type="dlc_track",
        annotation_type="csv",
        data_path="path/to/data",
        annotation_path="/path/to/OFT/Labels",
        output_path="path/to/output/results",
    )

DEVICE = "cuda:0"

project = Project(
    name=config.project_name,
    data_type=config.data_type,
    annotation_type=config.annotation_type,
    data_path=config.data_path,
    annotation_path=config.annotation_path,
    projects_path=config.projects_path,
)

project.update_parameters(
    {
        "data": {
            "canvas_shape": [928, 576],
            "data_suffix": ".npy",
            "feature_suffix": (
                "_videomae.npy" if features_only else "_cropped_videomae.npy"
            ),
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
            "metric_functions": {"accuracy", "f1"},
            "dim": 2,
        },
        "metrics": {
            "recall": {"ignored_classes": {}, "average": "macro"},
            "precision": {"ignored_classes": {}, "average": "macro"},
            "f1": {"ignored_classes": {}, "average": "none"},
        },
        "training": {
            "num_epochs": 100,
            "device": DEVICE,
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
                # "focal": False
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
)

search_results = {
    "ms_tcn": {
        "general": {"model_name": "ms_tcn3"},
        "model": {
            "num_f_maps": 128,
            "num_layers_PG": 10,
            "num_layers_R": 5,
            "block_size_prediction": 5,
            "shared_weights": True,
        },
    },
    "c2f_tcn": {
        "general": {"model_name": "c2f_tcn", "len_segment": 512, "overlap": 400},
        "model": {"num_f_maps": 64},
    },
    "c2f_transformer": {
        "general": {
            "model_name": "c2f_transformer",
            "len_segment": 512,
            "overlap": 400,
        },
        "data": {"len_segment": 512, "overlap": 400},
        "model": {"num_f_maps": 64, "heads": 4},
    },
    "edtcn": {
        "general": {"model_name": "edtcn", "len_segment": 128, "overlap": 100},
        "losses": {"ms_tcn": {"alpha": 6e-4}},
        "training": {"temporal_subsampling_size": 0.82},
    },
    "transformer": {
        "general": {"model_name": "transformer", "len_segment": 128, "overlap": 100},
        "data": {"len_segment": 128, "overlap": 100},
        "model": {"num_f_maps": 256, "heads": 4, "N": 10, "num_pool": 1},
    },
    "motionbert": {
        "general": {"model_name": "motionbert", "len_segment": 128, "overlap": 100},
        "data": {"len_segment": 128, "overlap": 100},
        # "model": {"dim_feat": 256, "depth": 4, "num_heads": 2},
        "model": {
            "num_joints": 14,
            "dim_feat": 256,
            "dim_rep": 256,
            "depth": 4,
            "num_heads": 4,
        },
        "training": {"lr": 1e-5, "batch_size": 32},
        "features": {
            "keys": {
                "coords",
                "acc_joints",
                "coord_diff",
                "angle_speeds",
                "speed_value",
                "speed_direction",
            }
        },
    },
}


# Save results
if True:
    df = {"models": [], "f1_scores": [], "trials": [], "class": []}
    ext = "" if features_only else "_cropped_with_pose_lr1e-5"

    model_list = [
        "c2f_tcn",
        "c2f_transformer",
        "transformer",
        "edtcn",
        "ms_tcn",
    ]

    model_names = {
        "mlp": "MLP",
        "ms_tcn": "MS-TCN3",
        "c2f_tcn": "C2F-TCN",
        "c2f_transformer": "C2F-Transformer",
        "transformer": "MP-transformer",
        "edtcn": "EDTCN",
    }
    for exp in model_list:
        indices = list(range(0, 20))
        res, v = project.get_summary(
            [f"exp_{exp}_videomae_{i}{ext}" for i in indices if i != 10],
            return_values=True,
        )
        exp = model_names.get(exp)

        for i in range(4):
            df["models"] += [exp] * len(v[f"f1_{i}"])
            df["f1_scores"] += v[f"f1_{i}"]
            df["trials"] += list(np.arange(len(v[f"f1_{i}"])))
            df["class"] += [i] * len(v[f"f1_{i}"])
    df = pd.DataFrame(df)
    output_path = os.path.join(config.output_path, f"oft_videomae{ext}_kinematics.csv")
    df.to_csv(output_path)
