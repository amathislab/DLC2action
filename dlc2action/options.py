#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Here all option dictionaries are stored
"""

from torch.optim import Adam, SGD
from dlc2action.data.input_store import *
from dlc2action.data.annotation_store import *
from dlc2action.feature_extraction import *
from dlc2action.transformer import *
from dlc2action.loss import MS_TCN_Loss
from dlc2action.model.mlp import MLP
from dlc2action.model.c2f_tcn import C2F_TCN
from dlc2action.model.asformer import ASFormer
from dlc2action.model.transformer import Transformer
from dlc2action.model.edtcn import EDTCN
from dlc2action.ssl.contrastive import *
from dlc2action.ssl.masked import *
from dlc2action.ssl.segment_order import *
from dlc2action.ssl.tcc import TCCSSL
from dlc2action.metric.metrics import *


input_stores = {
    "dlc_tracklet": DLCTrackletStore,
    "dlc_track": DLCTrackStore,
    "pku-mmd": PKUMMDInputStore,
    "calms21": CalMS21InputStore,
    "np_3d": Numpy3DInputStore,
    "features": LoadedFeaturesInputStore,
    "simba": SIMBAInputStore,
}

annotation_stores = {
    "dlc": DLCAnnotationStore,
    "pku-mmd": PKUMMDAnnotationStore,
    "boris": BorisAnnotationStore,
    "none": EmptyAnnotationStore,
    "calms21": CalMS21AnnotationStore,
    "csv": CSVAnnotationStore,
    "simba": SIMBAAnnotationStore,
}

feature_extractors = {"kinematic": KinematicExtractor}

ssl_constructors = {
    "masked_features": MaskedFeaturesSSL_FC,
    "masked_joints": MaskedKinematicSSL_FC,
    "masked_frames": MaskedFramesSSL_FC,
    "contrastive": ContrastiveSSL,
    "pairwise": PairwiseSSL,
    "contrastive_masked": ContrastiveMaskedSSL,
    "pairwise_masked": PairwiseMaskedSSL,
    "reverse": ReverseSSL,
    "order": OrderSSL,
    "contrastive_regression": ContrastiveRegressionSSL,
    "tcc": TCCSSL,
}

transformers = {"kinematic": KinematicTransformer}

losses = {
    "ms_tcn": MS_TCN_Loss,
}
losses_multistage = [
    "ms_tcn",
]  # losses that expect predictions of shape (#stages, #batch, #classes, #frames)

metrics = {
    "accuracy": Accuracy,
    "precision": Precision,
    "f1": F1,
    "recall": Recall,
    "count": Count,
    # "mAP": mAP,
    "segmental_precision": SegmentalPrecision,
    "segmental_recall": SegmentalRecall,
    "segmental_f1": SegmentalF1,
    "edit_distance": EditDistance,
    "f_beta": Fbeta,
    "segmental_f_beta": SegmentalFbeta,
    "semisegmental_precision": SemiSegmentalPrecision,
    "semisegmental_recall": SemiSegmentalRecall,
    "semisegmental_f1": SemiSegmentalF1,
    "pr-auc": PR_AUC,
    "semisegmental_pr-auc": SemiSegmentalPR_AUC,
    "mAP": PKU_mAP,
}
metrics_minimize = [
    "edit_distance"
]  # metrics that decrease when prediction quality increases
metrics_no_direction = ["count"]  # metrics that do not indicate prediction quality

optimizers = {"Adam": Adam, "SGD": SGD}

models = {
    "asformer": ASFormer,
    "mlp": MLP,
    "c2f_tcn": C2F_TCN,
    "edtcn": EDTCN,
    "transformer": Transformer,
}

blanks = [
    "dataset_inverse_weights",
    "dataset_proportional_weights",
    "dataset_classes",
    "dataset_features",
    "dataset_len_segment",
    "dataset_bodyparts",
    "dataset_boundary_weight",
    "model_features",
]

extractor_to_transformer = {
    "kinematic": "kinematic",
    "heatmap": "heatmap",
}  # keys are feature extractor names, values are transformer names

partition_methods = {
    "random": [
        "random",
        "random:test-from-name",
        "random:test-from-name:{name}",
        "random:equalize:segments",
        "random:equalize:videos",
    ],
    "fixed": [
        "val-from-name:{val_name}:test-from-name:{test_name}",
        "time",
        "time:start-from:{frac}",
        "time:start-from:{frac}:strict",
        "time:strict",
        "file",
        "folders",
    ],
}

basic_parameters = {
    "data": [
        "data_suffix",
        "feature_suffix",
        "annotation_suffix",
        "canvas_shape",
        "ignored_bodyparts",
        "likelihood_threshold",
        "behaviors",
        "filter_annotated",
        "filter_background",
        "visibility_min_score",
        "visibility_min_frac",
    ],
    "augmentations": {
        "heatmap": ["augmentations", "rotation_degree_limits"],
        "kinematic": [
            "augmentations",
            "rotation_limits",
            "mirror_dim",
            "noise_std",
            "zoom_limits",
            "masking_probability",
        ],
    },
    "features": {
        "heatmap": ["keys", "channel_policy", "heatmap_width", "sigma"],
        "kinematic": [
            "keys",
            "averaging_window",
            "distance_pairs",
            "angle_pairs",
            "zone_vertices",
            "zone_bools",
            "zone_distances",
            "area_vertices",
        ],
    },
    "model": {
        "asformer": [
            "num_decoders",
            "num_layers",
            "r1",
            "r2",
            "num_f_maps",
            "channel_masking_rate",
        ],
        "c2f_tcn": ["num_f_maps", "feature_dim"],
        "edtcn": ["kernel_size", "mid_channels"],
        "mlp": ["f_maps_list", "dropout_rates"],
        "transformer": ["num_f_maps", "N", "heads", "num_pool"],
    },
    "general": [
        "model_name",
        "metric_functions",
        "ignored_clips",
        "len_segment",
        "overlap",
        "interactive",
    ],
    "losses": {
        "ms_tcn": ["focal", "gamma", "alpha"],
        "clip": ["focal", "gamma", "alpha", "fix_text"],
    },
    "metrics": {
        "f1": ["average", "ignored_classes", "threshold_value"],
        "precision": ["average", "ignored_classes", "threshold_value"],
        "recall": ["average", "ignored_classes", "threshold_value"],
        "f_beta": ["average", "ignored_classes", "threshold_value", "beta"],
        "count": ["classes"],
        "segmental_precision": [
            "average",
            "ignored_classes",
            "threshold_value",
            "iou_threshold",
        ],
        "segmental_recall": [
            "average",
            "ignored_classes",
            "threshold_value",
            "iou_threshold",
        ],
        "segmental_f1": [
            "average",
            "ignored_classes",
            "threshold_value",
            "iou_threshold",
        ],
        "segmental_f_beta": [
            "average",
            "ignored_classes",
            "threshold_value",
            "iou_threshold",
        ],
        "pr-auc": ["average", "ignored_classes", "threshold_step"],
        "mAP": ["average", "ignored_classes", "iou_threshold", "threshold_value"],
        "semisegmental_precision": ["average", "ignored_classes", "iou_threshold"],
        "semisegmental_recall": ["average", "ignored_classes", "iou_threshold"],
        "semisegmental_f1": ["average", "ignored_classes", "iou_threshold"],
    },
    "training": [
        "lr",
        "device",
        "num_epochs",
        "to_ram",
        "batch_size",
        "normalize",
        "temporal_subsampling_size",
        "parallel",
        "val_frac",
        "test_frac",
        "partition_method",
    ],
}

model_hyperparameters = {
    "asformer": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/num_decoders": ("int", 1, 4),
        "model/num_f_maps": ("categorical", [32, 64, 128]),
        "model/num_layers": ("int", 5, 10),
        "model/channel_masking_rate": ("float", 0.2, 0.4),
        "general/len_segment": ("categorical", [64, 128]),
    },
    "c2f_tcn": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/num_f_maps": ("int_log", 32, 128),
        "general/len_segment": ("categorical", [512, 1024]),
    },
    "edtcn": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "general/len_segment": ("categorical", [128, 256, 512]),
    },
    "transformer": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/N": ("int", 5, 12),
        "model/heads": ("categorical", [1, 2, 4, 8]),
        "model/num_pool": ("int", 0, 4),
        "model/add_batchnorm": ("categorical", [True, False]),
        "general/len_segment": ("categorical", [64, 128]),
    },
    "mlp": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/dropout_rates": ("float", 0.3, 0.6),
    },
}
