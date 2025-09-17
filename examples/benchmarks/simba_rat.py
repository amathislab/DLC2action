#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project

DATA_PATH = "path/to/data/simba_rat"
ANNOTATION_PATH = "path/to/data/simba_rat"
PROJECTS_PATH = "path/to/projects"
DEVICE = "cuda"

project = Project(
    "rat-simba",
    data_type="simba",
    annotation_type="simba",
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
    projects_path=PROJECTS_PATH,
)

project.update_parameters(
    {
        "data": {
            "canvas_shape": [1290, 730],
            "likelihood_threshold": 0.8,
            "len_segment": 256,
            "overlap": 200,
            "data_suffix": ".csv",
            "annotation_suffix": ".csv",
            "use_features": False, #Set to false for Motionbert
            "ignored_classes": ["pursuit"],
            "visibility_min_frac": 0,
        },
        "general": {
            "model_name": "ms_tcn3",
            "exclusive": False,
            "metric_functions": {"precision", "recall", "f1"},
            "overlap": 0,
        },
        "metrics": {
            "recall": {"ignored_classes": {}, "average": "macro"},
            "precision": {"ignored_classes": {}, "average": "macro"},
            "f1": {"ignored_classes": {}, "average": "none"},
            "pku-map": {"ignored_classes": None},
        },
        "training": {
            "num_epochs": 200,
            "device": DEVICE,
            "lr": 1e-4,
            "batch_size": 128,
            "ssl_weights": {"pairwise": 0.01, "contrastive_regression": 1, "tcc": 1},
            "to_ram": False,
            "augment_train": 1,
            "partition_method": "time:start-from:0.1:strict",
            "val_frac": 0.2,
            "normalize": True,
            "temporal_subsampling_size": 0.8,
            "skip_normalization_keys": ["speed_direction", "coord_diff", "center"],
        },
        "losses": {
            "ms_tcn": {
                "weights": "dataset_inverse_weights",
                "gamma": 2.5,
                "alpha": 5e-3,
            }
        },
        "augmentations": {
            "augmentations": {"add_noise", "mirror", "rotate", "shift", "switch"},
            "mirror_dim": {0, 1},
            "noise_std": 0.002,
            "canvas_shape": [1290, 730],
        },
        "features": {
            "keys": {
                "intra_distance",
                "angle_speeds",
                "coord_diff",
                "acc_joints",
                "center",
                "speed_direction",
                "inter_distance",
                "speed_value",
            },
        },
        "ssl": {
            "tcc": {
                "num_f_maps": "model_features",
                "projection_head_f_maps": None,
                "loss_type": "regression_mse_var",
                "variance_lambda": 0.001,
                "normalize_indices": True,
                "normalize_embeddings": False,
                "similarity_type": "l2",
                "num_cycles": 20,
                "cycle_length": 2,
                "temperature": 0.1,
                "label_smoothing": 0.1,
            }
        },
    }
)

search_results = {
    "ms_tcn": {
        "general": {"model_name": "ms_tcn3", "len_segment": 256, "overlap": 200},
        "model": {
            "num_f_maps": 128,
            "num_layers_PG": 13,
            "num_layers_R": 5,
            "shared_weights": True,
        },
    },
    "c2f_tcn": {
        "general": {"model_name": "c2f_tcn", "len_segment": 1024, "overlap": 800},
        "model": {"num_f_maps": 128},
    },
    "c2f_transformer": {
        "general": {
            "model_name": "c2f_transformer",
            "len_segment": 512,
            "overlap": 400,
        },
        "model": {"num_f_maps": 128, "heads": 1},
    },
    "asformer": {
        "general": {"model_name": "asformer", "len_segment": 128, "overlap": 100},
        "model": {
            "channel_masking_rate": 0.3,
            "num_decoders": 1,
            "num_f_maps": 128,
            "num_layers": 10,
        },
    },
    "edtcn": {
        "general": {"model_name": "edtcn", "len_segment": 128, "overlap": 100},
        "training": {"temporal_subsampling_size": 0.9},
    },
    "transformer": {
        "general": {"model_name": "transformer", "len_segment": 128, "overlap": 100},
        "losses": {"ms_tcn": {"alpha": 1e-3}},
        "training": {"temporal_subsampling_size": 0.9},
        "model": {"N": 11, "heads": 2, "num_pool": 1, "add_batchnorm": False},
    },
    "motionbert": {
        "general": {"model_name": "motionbert", "len_segment": 128, "overlap": 100},
        "data": {"len_segment": 128, "overlap": 100},
        "model": {"num_joints": 8, "dim_feat": 256, "dim_rep": 256, "depth": 4, "num_heads": 4},
        "training": {"lr": 1e-5, "batch_size": 64},
        "features": {"keys": {"coords"}},
    }
}

# TRAINING
if True:
    for exp in search_results.keys():
        for start in range(5):
            pars = project._update(
                search_results[exp],
                {
                    "training": {
                        "partition_method": f"time:start-from:0.{start * 2}:strict",
                        "num_epochs": 100 if "ms_tcn" not in exp else 250,
                    },
                    "data": {"recompute_annotation": False},
                },
            )
            project.run_episode(
                f"exp_{exp}_{start}",
                force=True,
                parameters_update=pars,
            )
