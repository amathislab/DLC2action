#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project

DATA_PATH = "path/to/data"
ANNOTATION_PATH = "path/to/OFT/Labels"
PROJECTS_PATH = "path/to/project"
DEVICE = "cuda:0"

project = Project(
    "oft_videomae_cropped",
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
            "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
            "feature_suffix": "_cropped_videomae.npy",
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
}


num_epochs = 200
for exp in list(search_results.keys()):
    if exp in ["ms_tcn", "edtcn"]:
        continue
    for leave_ind in range(20):

        pars = search_results[exp]
        pars = project._update(
            pars,
            {
                "training": {
                    "partition_method": f"leave-one-out:{leave_ind}",
                    "normalize": True,
                    "num_epochs": num_epochs,
                },
                "data": {"normalize": False},
            },
        )
        project.run_episode(
            f"exp_{exp}_{leave_ind}",
            parameters_update=pars,
            force=True,
        )
        pars = project._update(pars, {"training": {"num_epochs": 20, "lr": 1e-5}})
        project.run_episode(
            f"exp_{exp}_{leave_ind}_lr1e-5",
            load_episode=f"exp_{exp}_{leave_ind}",
            parameters_update=pars,
            force=True,
            remove_saved_features=True,
        )
