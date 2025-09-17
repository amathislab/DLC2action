#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

DATA_PATH = "path/to/simba_rat/data"
ANNOTATION_PATH = "path/to/simba_rat/annotations"
PROJECTS_PATH = "path/to/projects"
DEVICE = "cuda"

project = Project(
    "rat-simba",
    data_type="simba",
    annotation_type="simba",
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
    projects_path=PROJECTS_PATH
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
            "use_features": True,
            "ignored_classes": ["pursuit"],
            "visibility_min_frac": 0
        },
        "general": {
            "model_name": "ms_tcn3",
            "exclusive": False,
            "metric_functions": {'precision', 'recall', 'f1'},
            "overlap": 0
        },
        "metrics": {
            "recall": {"ignored_classes": {}, "average": "macro"},
            "precision": {"ignored_classes": {}, "average": "macro"},
            "f1": {"ignored_classes": {}, "average": "none"},
            "pku-map": {"ignored_classes": None}
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
            "skip_normalization_keys": ["speed_direction", "coord_diff", "center"]
        },
        "losses": {
            "ms_tcn": {
                "weights": 'dataset_inverse_weights',
                "gamma": 2.5,
                "alpha": 5e-3,
            }
        },
        "augmentations": {
            "augmentations": {"add_noise", "mirror", "rotate", "shift", "switch"},
            "mirror_dim": {0, 1},
            "noise_std": 0.002,
            "canvas_shape": [1290, 730]
        },
        "features": {
            "keys": {"intra_distance", "angle_speeds", "coord_diff",
                     "acc_joints", "center", "speed_direction", "inter_distance", "speed_value"},
        },
        "ssl": {
            "tcc": {
                "num_f_maps": 'model_features',
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
        }
    }
)

search_results = {
    "ms_tcn": {
        "general": {"model_name": "ms_tcn3", "len_segment": 256, "overlap": 200},
        "model": {"num_f_maps": 128, "num_layers_PG": 13, "num_layers_R": 5, "shared_weights": True},
    },
    "c2f_tcn": {
        "general": {"model_name": "c2f_tcn", "len_segment": 1024, "overlap": 800},
        "model": {"num_f_maps": 128},
    },
    "c2f_transformer": {
        "general": {"model_name": "c2f_transformer", "len_segment": 512, "overlap": 400},
        "model": {"num_f_maps": 128, "heads": 1},
    },
    "asformer": {
        "general": {"model_name": "asformer", "len_segment": 128, "overlap": 100},
        "model": {"channel_masking_rate": 0.3, "num_decoders": 1, "num_f_maps": 128, "num_layers": 10}
    },
    "edtcn": {
        "general": {"model_name": "edtcn", "len_segment": 128, "overlap": 100},
        "training": {"temporal_subsampling_size": 0.9},
    },
    "transformer": {
        "general": {"model_name": "transformer", "len_segment": 128, "overlap": 100},
        "losses": {"ms_tcn": {"alpha": 1e-3}},
        "training": {"temporal_subsampling_size": 0.9},
        "model": {"N": 11, "heads": 2, "num_pool": 1, "add_batchnorm": False}
    }
}

# # TRAINING
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
                "features": {"keys": {"coords"}}
            }
        )
        project.run_episode(
            f'exp_{exp}_nf_{start}',
            force=True,
            parameters_update=pars,
        )

# FIGURE
behaviors = project.get_behavior_dictionary("exp_ms_tcn_0")
models = list(search_results.keys())
behaviors = [x for x in list(project.get_behavior_dictionary(f"exp_ms_tcn_0").values()) if x != "other"]
behavior_indices = list(range(0, len(behaviors)))
metric_label = "f1"
average = "get"
paper_values = {
    "anogenital": [0.24898291293734748, 0.2818154850192821, 0.37121906507791025, 0.5146862483311081, 0.021765417170495766],
    "approach": [0.12448443944506937, 0.06789755807027993, 0.11913626209977661, 0.2519230769230769, 0.2459396751740139],
    "attack": [0.7308762943992874, 0.7517133581820367, 0.9003663003663005, 0.8806265924318203, 0.7404868617628966],
    "avoidance": [0.2749425287356322, 0.16725559481743227, 0.2889447236180904, 0.3224852071005917, 0.23440453686200374],
    "boxing": [0.604504722693146, 0.5959595959595959, 0.6136363636363636, 0.5724381625441696, 0.5183071802187351],
    "lateralthreat": [0.5102173207914368, 0.47407407407407404, 0.2816104461371055, 0.49203007518796993, 0.5147484094852516],
    "submission": [0.5731608730800324, 0.8805759457933371, 0.7664015904572563, 0.8064699205448355, 0.5875917662300274],
}

paper = {}
for key, value in paper_values.items():
    paper[key] = [np.mean(value), np.std(value)]

results = []
stds = []
for model in models:
    method = "last"
    summ = project.get_summary([f"exp_{model}_nf_{i}" for i in range(5)], method=method)
    if average == "compute":
        results.append([np.mean([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])])
        std_arr = np.array([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])
        stds.append([np.sqrt(np.sum(std_arr ** 2)) / len(std_arr)])
    elif average == "get":
        if "f1" in summ:
            results.append([summ["f1"]["mean"]])
            stds.append([summ["f1"]["std"]])
        else:
            results.append([np.mean([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])])
            std_arr = np.array([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])
            stds.append([np.sqrt(np.sum(std_arr ** 2)) / len(std_arr)])
    else:
        results.append([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])
        stds.append([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])

if paper is not None:
    if average != "none":
        results.append([np.mean([paper[behavior][0] for behavior in behaviors])])
        std_arr = np.array([paper[behavior][1] for behavior in behaviors])
        stds.append([np.sqrt(np.sum(std_arr ** 2)) / len(std_arr)])
    else:
        results.append([paper[behavior][0] for behavior in behaviors])
        stds.append([paper[behavior][1] for behavior in behaviors])
    models.append("published")

colors = ["#99d096", "#ea678e", "#f9ba5b", "#639cd2", "#F1F285", "#B16CB9", "#ABE3CE", "#DD98A5", "#C44F53", "#BCC144", "#D6AF85"]

results_by_class = {i: [x[i] for x in results] for i in range(len(results[0]))}
std_by_class = {i: [x[i] for x in stds] for i in range(len(results[0]))}

mean_results = [np.mean(x) for x in results]
indices = list(np.argsort(mean_results))[::-1]

font = {'size': 20}

matplotlib.rc('font', **font)

x = np.arange(len(models))  # the label locations
width = 0.5 if average != "none" else 0.2  # the width of the bars

fig = plt.figure(figsize=(10, 8))
ax = fig.subplots()
N = len(results[0])
dist_arr = [width * n / 2 for n in np.arange(- (N - 1), N, 2)]
for i in range(len(results[0])):
    rects = ax.bar(
        x + dist_arr[i],
        np.array(results_by_class[i])[indices],
        width,
        yerr=np.array(std_by_class[i])[indices],
        label=behaviors[i],
    )
    if average == "none":
        for rect in rects:
            rect.set_color(colors[i])
    else:
        for j, rect in enumerate(rects):
            color = colors[0]
            # if np.array(models)[indices][j] != "published":
            #     color = colors[0]
            # else:
            #     color = colors[1]
            rect.set_color(color)

model_labels = []
for model in models:
    if "ms_tcn++" in model:
        model_labels.append("MS-TCN++")
    elif "ms_tcn" in model:
        model_labels.append("MS-TCN3")
    elif "c2f_tcn" in model:
        model_labels.append("C2F-TCN")
    elif "c2f_tr" in model:
        model_labels.append("C2F-Transformer")
    elif "trans" in model:
        model_labels.append("Transformer")
    elif "asf" in model:
        model_labels.append("ASFormer")
    elif "edtcn" in model:
        model_labels.append("EDTCN")
    elif "published" in model:
        model_labels.append("RF + SMOTE")
    elif "average" in model:
        model_labels.append("Top-3 average")
plt.xticks(ticks=x, labels=np.array(model_labels)[indices], rotation=70)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.ylabel(metric_label)

plt.tight_layout()

plt.savefig("examples/benchmarks/figures/simba_rat_nofeat.jpg")
