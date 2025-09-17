#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project

DATA_PATH = "path/to/calms21"
ANNOTATION_PATH = "path/to/calms21"
PROJECTS_PATH = "path/to/projects"
DEVICE = "cuda"

# initialize the project
project = Project(
    "calms21",
    data_type="calms21",
    annotation_type="calms21",
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
    projects_path=PROJECTS_PATH,
)

# update config files with project-specific information
project.update_parameters(
    parameters_update={
        "data": {
            "treba_files": True,
            "task_n": 1,
            "canvas_size": [1024, 570],
        },
        "training": {
            "num_epochs": 200,
            "device": DEVICE,
            "partition_method": "val-from-name:test:test-from-name:unlabeled",
            "batch_size": 32,
            "use_test": 0,
            "lr": 1e-4, # checked
            "to_ram": False,
            "ignore_tags": True,
            "validation_interval": 5,
            "ssl_weights": {
                "contrastive": 10,
                "contrastive_masked": 1,
                "pairwise": 1,
                "pairwise_masked": 1,
                "masked_features": 1
            },
            "normalize": True,
            "temporal_subsampling_size": 0.9,
            "skip_normalization_keys": ["speed_direction", "coord_diff"]
        },
        "general": {
            "metric_functions": {"f1"},
            "only_load_annotated": True,
            "exclusive": True,
            "len_segment": 128,
            "overlap": 100 / 128,
        },
        "metrics": {
            "precision": {
                "ignored_classes": {3},
                "average": "none"
            },
            "f1": {
                "ignored_classes": {3},
                "average": "macro",
            },
        },
        "losses": {
            "ms_tcn": {
                "focal": True,
                "alpha": 0.0001, # checked
                'gamma': 2.5, # checked
            }
        },
        "model": {
            "num_f_maps": 128, # 256 might be better (probably not)
            "num_layers_PG": 15, # checked
            "num_layers_R": 10, # checked
            "num_R": 3, # checked
            "shared_weights": False, # checked
            "direction_R": None, # checked
            "direction_PG": None, # bidirectional is better, but will leave at None for now
            "block_size_prediction": 3, # checked
            "block_size_refinement": 0, # checked
            "rare_dilations": False,
            "num_heads": 1,
            "num_joints": 14, #TODO Make an assert earlier on in the code for this
        },
        "ssl": {
            "masked_features": {
                "num_ssl_f_maps": 64, # search
                "frac_masked": 0.2, # search
            },
            "contrastive": {
                "ssl_features": 32, # search
                "tau": 10, # search
            },
            "pairwise_masked": {
                "num_masked": 20,
            },
            "contrastive_masked": {
                "ssl_features": 128,
            }
        },
        "augmentations": {
            'augmentations': {'rotate', 'add_noise', 'shift', 'mirror', 'switch'},
            'noise_std': 0.005,
            'mirror_dim': {0, 1},
            "canvas_shape": [1024, 570]
        },
        "features": {
            "keys": {"intra_distance", "angle_speeds", "coord_diff",
                     "acc_joints", "center", "speed_direction", "speed_value", "inter_distance"},
        },
    }
)

search_results = {
    "c2f_transformer": {
        "general": {"model_name": "c2f_transformer", "overlap": 400, "len_segment": 512},
        "model": {"heads": 2, "num_f_maps": 128},
        "data": {"overlap": 400, "len_segment": 512},
    },
    "c2f_tcn": {
        "general": {"model_name": "c2f_tcn", "overlap": 400, "len_segment": 512},
        "model": {"num_f_maps": 128},
        "data": {"overlap": 400, "len_segment": 512}
    },
    "ms_tcn": {
        "general": {"model_name": "ms_tcn3"},
        "model": {
            "num_f_maps": 128,
            "num_layers_PG": 20,
            "num_layers_R": 10,
        },
        "training": {"num_epochs": 300}
    },
    "edtcn": {
        "general": {"model_name": "edtcn"},
    },
    "transformer": {
        "general": {"model_name": "transformer", "len_segment": 128, "overlap": 100},
        "model": {"heads": 8, "num_f_maps": 512, "num_pool": 1, "N": 6},
        "data": {"len_segment": 128, "overlap": 100},
    },
    "asformer": {
        "general": {"model_name": "asformer", "len_segment": 128, "overlap": 100},
        "model": {"num_layers": 10, "num_f_maps": 64, "channel_masking_rate": 0.3},
        "data": {"len_segment": 128, "overlap": 100},
        "training": {"batch_size": 32, "lr": 5e-5, "num_epochs": 300}
    },
    "motionbert": {
        "general": {"model_name": "motionbert", "len_segment": 128, "overlap": 100},
        "model": {"num_joints": 15, "dim_feat": 256, "dim_rep": 256, "depth": 4, "num_heads": 4},
        "training": {"lr": 1e-5, "batch_size": 16},
        "features": {"keys": {"coords"}},
    }
}



# TRAINING
for exp in search_results.keys():
    pars = search_results[exp]
    project.run_episode(
        f"exp_{exp}",
        parameters_update=pars,
        force=True,
        n_runs=5,
    )
    if "training" in pars:
        pars["training"].update({"lr": 1e-5, "num_epochs": 30})
    else:
        pars["training"] = {"lr": 1e-5, "num_epochs": 30}
    project.run_episode(
        f'exp_{exp}_lr1e-5',
        load_episode=f"exp_{exp}",
        parameters_update=pars,
        force=True,
    )
