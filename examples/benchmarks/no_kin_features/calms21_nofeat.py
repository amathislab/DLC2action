#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

DATA_PATH = "path/to/calms21/data"
ANNOTATION_PATH = "path/to/calms21/annotations"
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
            "num_epochs": 100,
            "device": DEVICE,
            "partition_method": "val-from-name:test:test-from-name:unlabeled",
            "batch_size": 128,
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
            "len_segment": 256,
            "overlap": 200 / 256,
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
            "num_heads": 1
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
}


# # TRAINING
for exp in search_results.keys():
    pars = search_results[exp]
    pars = project._update(
        pars,
        {
            'features': {
                'keys': {"coords"},
            },
        }
    )
    project.run_episode(
        f"exp_{exp}_nf",
        parameters_update=pars,
        force=True,
        n_runs=5,
    )
    if "training" in pars:
        pars["training"].update({"lr": 1e-5, "num_epochs": 30})
    else:
        pars["training"] = {"lr": 1e-5, "num_epochs": 30}
    project.run_episode(
        f'exp_{exp}_nf_lr1e-5',
        load_episode=f"exp_{exp}_nf",
        parameters_update=pars,
        force=True,
    )


# FIGURES
metrics = ["f1"]
matplotlib.rcParams.update({'font.size': 20})
means = []
stds = []
labels = []
colors = ["#99d096", "#ea678e", "#f9ba5b", "#639cd2", "#F1F285", "#B16CB9", "#ABE3CE", "#DD98A5", "#C44F53", "#BCC144", "#D6AF85"]
label_dict = {
    "c2f_transformer": "C2F-transformer",
    "c2f_tcn": "C2F-TCN",
    "ms_tcn": "MS-TCN3",
    "edtcn": "EDTCN",
    "transformer": "MP-Transformer",
    "asformer": "ASFormer",
    "published": "Challenge top-1"
}

for name in search_results.keys():
    labels.append(label_dict[name])
    summ = project.get_summary([f'exp_{name}_nf_lr1e-5'], method="last")
    means.append([summ[metric]["mean"] for metric in metrics])
    stds.append([summ[metric]["std"] for metric in metrics])

means.append([0.864, 0, 0])
stds.append([0.011, 0, 0])
labels.append("Challenge top-1")

indices = np.argsort(means)[::-1]
means = np.array(means)[indices]
stds = np.array(stds)[indices]
labels = np.array(labels)[indices]

x = np.arange(len(labels))  # the label locations
width = 0.75  # the width of the bars
N = len(metrics)
dist_arr = [width * n / 2 for n in np.arange(- (N - 1), N, 2)]
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,
                         figsize=(10, 8))
for ax in [ax1, ax2]:
    for i in range(N):
        rects = ax.bar(
            x + dist_arr[i],
            np.array([x[i] for x in means]),
            width,
            yerr=np.array([x[i] for x in stds]),
            label=metrics[i],
        )
        for j, rect in enumerate(rects):
            color = colors[0]
            rect.set_color(color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom=False)
ax2.set_ylim(0,0.25)
ax1.set_ylim(0.65,0.9)

plt.xticks(ticks=x, rotation=70, labels=labels)
plt.ylabel("F1 score")

plt.tight_layout()

plt.savefig("examples/benchmarks/figures/calms21_nofeat.jpg")
