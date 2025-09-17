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

DATA_PATH = "path/to/epm/data"
ANNOTATION_PATH = "path/to/epm/annotations"
PROJECTS_PATH = "path/to/projects"
DEVICE = "cuda"

project = Project(
    "epm",
    data_type="dlc_track",
    annotation_type="csv",
    data_path=DATA_PATH,
    annotation_path=ANNOTATION_PATH,
    projects_path=PROJECTS_PATH
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
            "behaviors": ['Grooming', 'Protected Stretch', 'Head Dip', 'Rearing']
        },
        "general": {
            "model_name": "ms_tcn3",
            "exclusive": True,
            "only_load_annotated": True,
            "metric_functions": {'accuracy', 'f1'},
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
                "weights": 'dataset_inverse_weights',
                "gamma": 2.5,
                "alpha": 0.001,
                # "focal": False
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
            "keys": {"intra_distance", "angle_speeds", "areas", "coord_diff",
                     "acc_joints", "center", "speed_direction", "speed_value", "zone_bools"},
            "angle_pairs": [["tailbase", "tailcentre", "tailcentre", "tailtip"],
                         ["hipr", "tailbase", "tailbase", "hipl"],
                         ["tailbase", "bodycentre", "bodycentre", "neck"],
                         ["bcr", "bodycentre", "bodycentre", "bcl"],
                         ["bodycentre", "neck", "neck", "headcentre"],
                         ["tailbase", "bodycentre", "neck", "headcentre"]
                         ],
            "area_vertices": [["tailbase","hipr","hipl"], ["hipr","hipl","bcl","bcr"],
                           ["bcr","earr","earl","bcl"], ["earr","nose","earl"]],
            "neighboring_frames": 0,
            "zone_vertices": {"arena": ["tl", "tr", "br", "bl"]},
            "zone_bools": [["arena", "nose"], ["arena", "headcentre"]]
        },
    }
)

search_results = {
    "ms_tcn": {
        "general": {"len_segment": 128, "model_name": "ms_tcn3"},
        "losses": {"ms_tcn": {"alpha": 1e-4}},
        "training": {"temporal_subsampling_size": 0.82},
        "model": {"num_layers_PG": 9, "shared_weights": False, "num_layers_R": 7, "num_f_maps": 128,
                  "skip_connections_refinement": False, "block_size_prediction": 0,}
    },
    "transformer": {
        "general": {"len_segment": 128, "model_name": "transformer"},
        "losses": {"ms_tcn": {"alpha": 0.008}},
        "training": {"temporal_subsampling_size": 0.78},
        "model": {"N": 8, "heads": 4, "num_pool": 2, "add_batchnorm": False}
    },
    "asformer": {
        "general": {"len_segment": 128, "model_name": "asformer"},
        "losses": {"ms_tcn": {"alpha": 0.001}},
        "training": {"temporal_subsampling_size": 0.92},
        "model": {"num_f_maps": 128, "num_layers": 7, "num_decoders": 3, "channel_masking_rate": 0.26}
    },
    "c2f_tcn": {
        "general": {"len_segment": 1024, "model_name": "c2f_tcn"},
        "losses": {"ms_tcn": {"alpha": 5e-4}},
        "training": {"temporal_subsampling_size": 0.95},
        "model": {"num_f_maps": 128}
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
        "model": {"num_f_maps": 128, "heads": 1}
    },
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
                "training": {"partition_method": f"time:start-from:0.{start}:strict", "normalize": True, "num_epochs": num_epochs},
                "data": {"normalize": False},
                "features": {"keys": {"coords"}}
            }
        )
        project.run_episode(
            f"exp_{exp}_nf_{start}",
            parameters_update=pars,
            force=True,
        )
        pars = project._update(pars, {"training": {"num_epochs": 20, "lr": 1e-5}})
        project.run_episode(
            f"exp_{exp}_nf_{start}_lr1e-5",
            load_episode=f"exp_{exp}_nf_{start}",
            parameters_update=pars,
            force=True,
        )

# FIGURE
models = list(search_results.keys())

metric_label = "f1"
average = False
paper = None
behaviors = [x for x in list(project.get_behavior_dictionary(f"exp_edtcn_1").values()) if x != "other"]
behavior_indices = list(range(1, len(behaviors) + 1))

results = []
stds = []
for model in models:
    summ = project.get_summary([f"exp_{model}_nf_{i}_lr1e-5" for i in range(5)], method="last")
    if average:
        results.append([np.mean([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])])
        std_arr = np.array([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])
        stds.append([np.sqrt(np.sum(std_arr ** 2)) / len(std_arr)])
    else:
        results.append([summ[f"f1_{i}"]["mean"] for i in behavior_indices if i != 0])
        stds.append([summ[f"f1_{i}"]["std"] for i in behavior_indices if i != 0])

if paper is not None:
    if average:
        results.append([np.mean([paper[behavior][0] for behavior in behaviors])])
        std_arr = np.array([paper[behavior][1] for behavior in behaviors])
        stds.append([np.sqrt(np.sum(std_arr ** 2)) / len(std_arr)])
    else:
        results.append([paper[behavior][0] for behavior in behaviors])
        stds.append([paper[behavior][1] for behavior in behaviors])
    models.append("published")
#
#
colors = ["#99d096", "#ea678e", "#f9ba5b", "#639cd2", "#F1F285", "#B16CB9", "#ABE3CE", "#DD98A5", "#C44F53", "#BCC144", "#D6AF85"]

results_by_class = {i: [x[i] for x in results] for i in range(len(results[0]))}
std_by_class = {i: [x[i] for x in stds] for i in range(len(results[0]))}

mean_results = [np.mean(x) for x in results]
indices = list(np.argsort(mean_results))[::-1]

font = {'size': 20}

matplotlib.rc('font', **font)

x = np.arange(len(models))  # the label locations
width = 0.5 if average else 0.2  # the width of the bars

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
    if "ms_tcn++" in model:
        model_labels.append("MS-TCN++")
    elif "ms_tcn" in model:
        model_labels.append("MS-TCN3")
    elif "c2f_tcn" in model:
        model_labels.append("C2F-TCN")
    elif "c2f_transformer_linear" in model:
        model_labels.append("C2F-Transformer")
    elif "trans" in model:
        model_labels.append("Transformer")
    elif "asf" in model:
        model_labels.append("ASFormer")
    elif "edtcn" in model:
        model_labels.append("EDTCN")
    elif "published" in model:
        model_labels.append("Published")
    elif "average" in model:
        model_labels.append("Average")
plt.xticks(ticks=x, labels=np.array(model_labels)[indices], rotation=70)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.ylabel(metric_label)

plt.tight_layout()

plt.savefig("examples/benchmarks/figures/sturman_epm_nofeat.jpg")
