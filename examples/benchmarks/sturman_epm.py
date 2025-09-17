#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project

DATA_PATH = "path/to/EPM/Output_DLC"
ANNOTATION_PATH = "path/to/EPM/Labels"
PROJECTS_PATH = "path/to/projects"
DEVICE = "cuda"

project = Project(
    "epm",
    data_type="dlc_track",
    annotation_type="csv",
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
    projects_path=PROJECTS_PATH,
)

project.update_parameters(
    {
        "data": {
            "canvas_shape": [928, 576],
            "data_suffix": {"DeepCut_resnet50_epmMay17shuffle1_1030000.csv"},
            "annotation_suffix": ".csv",
            "ignored_bodyparts": {"tl", "tr", "br", "bl", "centre"},
            "likelihood_threshold": 0.8,
            "len_segment": 256,
            "overlap": 200,
            "filter_background": False,
            "filter_annotated": False,
            "fps": 25,
            "clip_frames": 0,
            "behaviors": ["Grooming", "Protected Stretch", "Head Dip", "Rearing"],
        },
        "general": {
            "model_name": "ms_tcn3",
            "exclusive": True,
            "only_load_annotated": True,
            "metric_functions": {"accuracy", "f1"},
            "dim": 2,
            "overlap": 400 / 512,
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
            "normalize": True,
            "device": "cuda:0",
            "val_frac": 0.2,
            "test_frac": 0,
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
        "general": {"len_segment": 128, "model_name": "ms_tcn3"},
        "losses": {"ms_tcn": {"alpha": 1e-4}},
        "training": {"temporal_subsampling_size": 0.82},
        "model": {
            "num_layers_PG": 9,
            "shared_weights": False,
            "num_layers_R": 7,
            "num_f_maps": 128,
            "skip_connections_refinement": False,
            "block_size_prediction": 0,
        },
    },
    "transformer": {
        "general": {"len_segment": 128, "model_name": "transformer"},
        "losses": {"ms_tcn": {"alpha": 0.008}},
        "training": {"temporal_subsampling_size": 0.78},
        "model": {"N": 8, "heads": 4, "num_pool": 2, "add_batchnorm": False},
    },
    "asformer": {
        "general": {"len_segment": 128, "model_name": "asformer"},
        "losses": {"ms_tcn": {"alpha": 0.001}},
        "training": {"temporal_subsampling_size": 0.92},
        "model": {
            "num_f_maps": 128,
            "num_layers": 7,
            "num_decoders": 3,
            "channel_masking_rate": 0.26,
        },
    },
    "c2f_tcn": {
        "general": {"len_segment": 1024, "model_name": "c2f_tcn"},
        "losses": {"ms_tcn": {"alpha": 5e-4}},
        "training": {"temporal_subsampling_size": 0.95},
        "model": {"num_f_maps": 128},
    },
    "edtcn": {
        "general": {"len_segment": 128, "model_name": "edtcn"},
        "losses": {"ms_tcn": {"alpha": 3e-5}},
        "training": {"temporal_subsampling_size": 0.98},
    },
    "c2f_transformer": {
        "general": {"len_segment": 512, "model_name": "c2f_transformer"},
        "losses": {"ms_tcn": {"alpha": 1e-3}},
        "training": {"temporal_subsampling_size": 0.85},
        "model": {"num_f_maps": 128, "heads": 1},
    },
    "motionbert": {
        "general": {"model_name": "motionbert", "len_segment": 128, "overlap": 100},
        "model": {"num_joints": 14, "dim_feat": 256, "dim_rep": 256, "depth": 4, "num_heads": 4},
        "training": {"lr": 1e-5, "batch_size": 32},
        "features": {"keys": {"coords", "acc_joints", "coord_diff", "angle_speeds",
                              "speed_value", "speed_direction"}},
    }
}

# # TRAINING
for exp in search_results.keys():
    for start in range(5):
        pars = search_results[exp]
        if "ms_tcn" in exp:
            num_epochs = 200
        else:
            num_epochs = 100
        pars = project._update(
            pars,
            {
                "training": {
                    "partition_method": f"time:start-from:0.{start}:strict",
                    "normalize": True,
                    "num_epochs": num_epochs,
                },
                "data": {"normalize": False},
            },
        )
        project.run_episode(
            f"exp_{exp}_{start}",
            parameters_update=pars,
            force=True,
        )
        pars = project._update(pars, {"training": {"num_epochs": 20, "lr": 1e-5}})
        project.run_episode(
            f"exp_{exp}_{start}_lr1e-5",
            load_episode=f"exp_{exp}_{start}",
            parameters_update=pars,
            force=True,
        )
