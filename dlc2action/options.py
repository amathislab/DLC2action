#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Here all option dictionaries are stored
"""

from dlc2action.data.annotation_store import (
    BorisStore,
    CalMS21Store,
    CSVActionSegStore,
    DLC2ActionStore,
    EmptyBehaviorStore,
    SIMBAStore,
)
from dlc2action.data.input_store import (
    CalMS21InputStore,
    DLCTrackletStore,
    DLCTrackStore,
    LoadedFeaturesInputStore,
    Numpy3DInputStore,
    SIMBAInputStore,
)
from dlc2action.feature_extraction import HeatmapExtractor, KinematicExtractor
from dlc2action.loss import MS_TCN_Loss
from dlc2action.metric.metrics import (
    F1,
    PR_AUC,
    Accuracy,
    Count,
    EditDistance,
    Fbeta,
    mAP,
    Precision,
    Recall,
    SegmentalF1,
    SegmentalFbeta,
    SegmentalPrecision,
    SegmentalRecall,
    SemiSegmentalF1,
    SemiSegmentalPR_AUC,
    SemiSegmentalPrecision,
    SemiSegmentalRecall,
)


# from dlc2action.model.c3d import C3D_A, C3D_A_MS
from dlc2action.model.asformer import ASFormer
from dlc2action.model.c2f_tcn import C2F_TCN
from dlc2action.model.c2f_transformer import C2F_Transformer
from dlc2action.model.edtcn import EDTCN
from dlc2action.model.mlp import MLP
from dlc2action.model.ms_tcn import MS_TCN3  # ,MS_TCN_P
from dlc2action.model.transformer import Transformer
from dlc2action.model.motionbert import MotionBERT

from dlc2action.ssl.contrastive import (
    ContrastiveMaskedSSL,
    ContrastiveRegressionSSL,
    ContrastiveSSL,
    PairwiseMaskedSSL,
    PairwiseSSL,
)
from dlc2action.ssl.masked import (
    MaskedFeaturesSSL_FC,
    MaskedFramesSSL_FC,
    MaskedKinematicSSL_FC,
    MaskedFeaturesSSL_TCN,
    MaskedFramesSSL_TCN,
    MaskedKinematicSSL_TCN,
)
from dlc2action.ssl.segment_order import OrderSSL, ReverseSSL
from dlc2action.ssl.tcc import TCCSSL
from dlc2action.transformer.heatmap import HeatmapTransformer
from dlc2action.transformer.kinematic import KinematicTransformer
from torch.optim import SGD, Adam

input_stores = {
    "dlc_tracklet": DLCTrackletStore,
    "dlc_track": DLCTrackStore,
    "calms21": CalMS21InputStore,
    "np_3d": Numpy3DInputStore,
    "features": LoadedFeaturesInputStore,
    "simba": SIMBAInputStore,
}

annotation_stores = {
    "dlc": DLC2ActionStore,
    "boris": BorisStore,
    "none": EmptyBehaviorStore,
    "calms21": CalMS21Store,
    "csv": CSVActionSegStore,
    "simba": SIMBAStore,
}

feature_extractors = {"kinematic": KinematicExtractor, "heatmap": HeatmapExtractor}

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

ssl_constructors_tcn = {
    "masked_features": MaskedFeaturesSSL_TCN,
    "masked_joints": MaskedKinematicSSL_TCN,
    "masked_frames": MaskedFramesSSL_TCN,
    "contrastive": ContrastiveSSL,
    "pairwise": PairwiseSSL,
    "contrastive_masked": ContrastiveMaskedSSL,
    "pairwise_masked": PairwiseMaskedSSL,
    "reverse": ReverseSSL,
    "order": OrderSSL,
    "contrastive_regression": ContrastiveRegressionSSL,
    "tcc": TCCSSL,
}

transformers = {"kinematic": KinematicTransformer, "heatmap": HeatmapTransformer}

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
    "mAP": mAP,
}
metrics_minimize = [
    "edit_distance"
]  # metrics that decrease when prediction quality increases
metrics_no_direction = ["count"]  # metrics that do not indicate prediction quality

optimizers = {"Adam": Adam, "SGD": SGD}

models = {
    "ms_tcn3": MS_TCN3,
    "asformer": ASFormer,
    "mlp": MLP,
    "c2f_tcn": C2F_TCN,
    "edtcn": EDTCN,
    "transformer": Transformer,
    "c2f_transformer": C2F_Transformer,
    "motionbert": MotionBERT,
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
        "c2f_transformer": ["num_f_maps", "feature_dim", "heads"],
        "edtcn": ["kernel_size", "mid_channels"],
        "mlp": ["f_maps_list", "dropout_rates"],
        "ms_tcn3": [
            "num_layers_PG",
            "num_layers_R",
            "num_R",
            "num_f_maps",
            "shared_weights",
        ],
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
        "general/len_segment": ("categorical", [256, 512, 1024, 2048]),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
    "c2f_tcn": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/num_f_maps": ("int_log", 32, 128),
        "general/len_segment": ("categorical", [512, 1024, 2048]),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
    "c2f_transformer": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/num_f_maps": ("categorical", [32, 64, 128]),
        "model/heads": ("categorical", [1, 2, 4, 8]),
        "general/len_segment": ("categorical", [512, 1024, 2048]),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
    "edtcn": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "general/len_segment": ("categorical", [256, 512, 1024, 2048]),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
    "ms_tcn3": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/num_layers_PG": ("int", 5, 20),
        "model/shared_weights": ("categorical", [True, False]),
        "model/num_layers_R": ("int", 5, 10),
        "model/num_f_maps": ("int_log", 32, 128),
        "general/len_segment": ("categorical", [256, 512, 1024, 2048]),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
    "transformer": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/N": ("int", 5, 12),
        "model/heads": ("categorical", [1, 2, 4, 8]),
        "model/num_pool": ("int", 0, 4),
        "model/add_batchnorm": ("categorical", [True, False]),
        "general/len_segment": ("categorical", [256, 512, 1024, 2048]),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
    "mlp": {
        "losses/ms_tcn/alpha": ("float_log", 1e-5, 1e-2),
        "losses/ms_tcn/focal": ("categorical", [True, False]),
        "training/temporal_subsampling_size": ("float", 0.75, 1),
        "model/dropout_rates": ("float", 0.3, 0.6),
        "losses/ms_tcn/weights": ("categorical", [None, "dataset_inverse_weights"]),
    },
}

dlc2action_colormaps = {
    "default": [
    "#BBBBBF",
    "#99d096",
    "#ea678e",
    "#f9ba5b",
    "#639cd2",
    "#F1F285",
    "#B16CB9",
    "#ABE3CE",
    "#DD98A5",
    "#C44F53",
    "#BCC144",
    "#D6AF85",
    "#BBBBBF",
]
}
