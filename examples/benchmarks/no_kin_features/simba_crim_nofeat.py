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

DATA_PATH = "path/to/crim/data"
ANNOTATION_PATH = "path/to/crim/annotations"
PROJECTS_PATH = "path/to/projects"
DEVICE = "cuda"

project = Project(
    "crim-simba",
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
            "behaviors": ['approach', 'attack', 'copulation', 'chase', 'circle', 'drink', 'eat', 'clean', 'sniff', 'up', 'walk_away']
        },
        "general": {
            "model_name": "ms_tcn3",
            "exclusive": False,
            "metric_functions": {'precision', 'recall', 'f1'},
        },
        "metrics": {
            "recall": {"ignored_classes": {}, "average": "macro"},
            "precision": {"ignored_classes": {}, "average": "macro"},
            "f1": {"ignored_classes": {}, "average": "none"},
        },
        "training": {
            "num_epochs": 90,
            "device": DEVICE,
            "lr": 1e-4,
            "batch_size": 128,
            "ssl_weights": {"pairwise": 0.01, "contrastive_regression": 1},
            "to_ram": False,
            "augment_train": 1,
            "partition_method": "time:start-from:0.2:strict",
            "val_frac": 0.2,
            "normalize": True,
            "temporal_subsampling_size": 0.8,
            "skip_normalization_keys": ["speed_direction", "coord_diff"]
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
                     "acc_joints", "center", "speed_direction", "speed_value", "inter_distance"},
            # "interactive": True,
        },
    }
)

project.list_episodes()

search_results = {
    "c2f_tcn": {
        "general": {"model_name": "c2f_tcn", "len_segment": 512, "overlap": 400},
        "model": {"num_f_maps": 128},
        "losses": {"ms_tcn": {"alpha": 1e-4}},
        "training": {"temporal_subsampling_size": 0.75},
    },
    "c2f_transformer": {
        "general": {"model_name": "c2f_transformer", "len_segment": 512, "overlap": 400},
        "model": {"num_f_maps": 128, "heads": 8},
        "losses": {"ms_tcn": {"alpha": 1e-4}},
        "training": {"temporal_subsampling_size": 0.9},
    },
    "asformer": {
        "general": {"model_name": "asformer", "len_segment": 128, "overlap": 100},
        "model": {"channel_masking_rate": 0.3, "num_decoders": 1, "num_f_maps": 128, "num_layers": 10},
        "losses": {"ms_tcn": {"alpha": 5e-3}},
        "training": {"temporal_subsampling_size": 0.83},
    },
    "edtcn": {
        "general": {"model_name": "edtcn", "len_segment": 128, "overlap": 100},
        "training": {"temporal_subsampling_size": 0.9},
    },
    "transformer": {
        "losses": {"ms_tcn": {"alpha": 1e-4}},
        "general": {"model_name": "transformer", "len_segment": 128, "overlap": 100},
        "training": {"temporal_subsampling_size": 0.91},
        "model": {"N": 6, "heads": 8, "num_pool": 3, "add_batchnorm": False, "num_f_maps": 128}
    },
    "ms_tcn": {
        "general": {"model_name": "ms_tcn3", "len_segment": 128, "overlap": 100},
        "model": {"num_f_maps": 128, "num_layers_PG": 20, "num_layers_R": 10, "shared_weights": True},
        "losses": {"ms_tcn": {"alpha": 5e-3}},
        "training": {"temporal_subsampling_size": 0.85},
    },
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
models = list(search_results.keys())
behaviors = [x for x in list(project.get_behavior_dictionary(f"exp_c2f_tcn_0").values()) if x != "other"]
behavior_indices = list(range(len(behaviors)))
metric_label = "f1"
average = "compute"
paper_values = {
    "approach": [0.46920173096399964, 0.5297825401827924, 0.494807121661721, 0.5245818120064533, 0.5037958474343731],
    "attack": [0.640565961894043, 0.7137012736395214, 0.7090066674200475, 0.687557122339731, 0.6915510718789408],
    "chase": [0.36158192090395486, 0.36832061068702293, 0.4710351377018044, 0.30708661417322836, 0.3954154727793696],
    "circle": [0.2242462311557789, 0.29775158262388124, 0.21029572836801752, 0.28466707391569945, 0.2677165354330709],
    "clean": [0.6530464653769661, 0.6094719901908192, 0.589564639415088, 0.6171415733702774, 0.6626961198275478],
    "copulation": [0.7806457335462125, 0.8106424076266434, 0.7905122583217061, 0.8672448298865911, 0.7292307692307691],
    "sniff": [0.7899212056714717, 0.7457078695095742, 0.7476547058987278, 0.7328075920604158, 0.7033492822966508],
    "up": [0.6215733015494637, 0.6691931540342297, 0.7567029135884452, 0.7165059376743443, 0.6630155173844494],
    "walk_away": [0.44357976653696496, 0.44468645067610496, 0.428228054994712, 0.4405629304321788, 0.4657624875951042],
    "drink": [0.09863013698630138, 0.2113289760348584, 0.0, 0.05073431241655541],
    "eat": [0.12802056555269922, 0.10401561628555493, 0.16335877862595422, 0.11399686247167509],
}
paper = {}
for key, value in paper_values.items():
    paper[key] = [np.mean(value), np.std(value)]

results = []
stds = []
for model in models:
    summ = project.get_summary([f"exp_{model}_nf_{i}" for i in range(5)], method="last")
    if average == "compute":
        results.append([np.mean([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])])
        std_arr = np.array([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])
        stds.append([np.sqrt(np.sum(std_arr ** 2)) / len(std_arr)])
    elif average == "get":
        results.append([summ["f1"]["mean"]])
        stds.append([summ["f1"]["std"]])
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

colors = ["#99d096", "#ea678e", "#f9ba5b", "#639cd2", "#F1F285", "#B16CB9"]

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
        label=behaviors[i]
    )
    if average != "compute":
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
plt.xticks(ticks=x, labels=np.array(model_labels)[indices], rotation=70)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.ylabel(metric_label)

plt.tight_layout()

plt.savefig("examples/benchmarks/figures/simba_crim_nofeat.jpg")
