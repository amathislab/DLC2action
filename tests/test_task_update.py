#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.task.task_dispatcher import TaskDispatcher
from math import floor
import pytest

parameters = {
    "data": {
        "data_path": "/home/liza/data/cricket",
        "annotation_path": "/home/liza/data/cricket",
        "behaviors": ["Grooming", "Search", "Pursuit"],
        "annotation_suffix": {".csv"},
        "data_suffix": {
            "DLC_resnet50_preycapSep30shuffle1_20000_bx_filtered.h5",
        },
        "default_agent_name": "mouse",
    },
    "features": {
        "interactive": False,  # bool; if true, distances between two agents are included; if false, only the first agent features are computed
        "pickled_feature_suffix": None,  # str; the feature files should be stored in the data folder and named {video_id}{h5_feature_suffix}
        "keys": ["coords"],
    },
    "model": {
        "input_dims": "dataset_features",
        "num_f_maps": 16,  # int; number of feature maps
        "feature_dim": 16,
    },
    "general": {
        "ignored_clips": ["single"],
        "data_type": "dlc_track",
        "annotation_type": "boris",
        "model_name": "c2f_tcn",  # str; model name
        "num_classes": "dataset_classes",  # int; number of classes
        "exclusive": True,  # bool; if true, single-label classification is used; otherwise multi-label
        "len_segment": 64,
        "ssl": [
            "contrastive",
        ],  # list; ['contrastive', 'masked_features'] a list of SSL types to use
        "metric_functions": ["accuracy", "recall", "precision"],  # list; list of metric
        "loss_function": "ms_tcn",  # str; name of loss function
        "feature_extraction": "kinematic",  # str; the feature extraction method (only 'kinematic' at the moment)
        "save_dataset": False,  # bool; if true, pre-computed datasets are saved in a pickled file for faster loading
        "only_load_annotated": True,
    },
    "losses": {
        "ms_tcn": {
            "weights": "dataset_inverse_weights",  # list; list of weights for weighted cross-entropy
            "focal": True,  # bool; if True, focal loss will be used
            "gamma": 5,  # float; the gamma parameter of focal loss
            "alpha": 0.05,
        }
    },
    "metrics": {"recall": {"average": "none"}},
    "ssl": {
        "contrastive": {
            "len_segment": "dataset_len_segment",
            "num_f_maps": "model_features",
        },
    },
    "training": {
        "lr": 1e-3,  # float; learning rate
        "device": "cuda",  # str; device
        "augment_train": 1,  # [0, 1]; either 1 to use augmentations during training or 0 to not use
        "ssl_weights": {
            "contrastive": 10,
            "masked_features": 0.1,
        },  # dict; dictionary of SSL loss function weights
        "num_epochs": 500,  # int; number of epochs
        "to_ram": False,  # bool; transfer the dataset to RAM for training (preferred if the dataset fits in working memory)
        "batch_size": 32,  # int; batch size
        "model_save_epochs": 50,  # int; interval for saving training checkpoints (the last epoch is always saved)
        "test_frac": 0,
    },  # float; fraction of dataset to use as test
}
update = {
    "data": {
        "data_path": "/home/liza/data/cricket",
        "annotation_path": "/home/liza/data/cricket",
        "behaviors": ["Grooming", "Search", "Pursuit"],
        "annotation_suffix": {".csv"},
        "data_suffix": {
            "DLC_resnet50_preycapSep30shuffle1_20000_bx_filtered.h5",
        },
        "default_agent_name": "mouse",
    },
    "features": {
        "interactive": False,  # bool; if true, distances between two agents are included; if false, only the first agent features are computed
        "pickled_feature_suffix": None,  # str; the feature files should be stored in the data folder and named {video_id}{h5_feature_suffix}
        "keys": ["coords", "intra_distance"],
    },
    "model": {
        "num_f_maps": 16,  # int; number of feature maps
    },
    "general": {
        "len_segment": 128,
        "ignored_clips": ["single"],
        "data_type": "dlc_track",
        "annotation_type": "boris",
        "model_name": "c2f_tcn",  # str; model name
        "num_classes": "dataset_classes",  # int; number of classes
        "exclusive": False,  # bool; if true, single-label classification is used; otherwise multi-label
        "ssl": [
            "contrastive"
        ],  # list; ['contrastive', 'masked_features'] a list of SSL types to use
        "metric_functions": ["accuracy", "recall"],  # list; list of metric
        "loss_function": "ms_tcn",  # str; name of loss function
        "feature_extraction": "kinematic",  # str; the feature extraction method (only 'kinematic' at the moment)
        "save_dataset": False,  # bool; if true, pre-computed datasets are saved in a pickled file for faster loading
        "only_load_annotated": True,
    },
    "losses": {
        "ms_tcn": {
            "weights": "dataset_inverse_weights",  # list; list of weights for weighted cross-entropy
            "focal": False,  # bool; if True, focal loss will be used
            "gamma": 5,  # float; the gamma parameter of focal loss
            "alpha": 0.05,
        }
    },
    "metrics": {"recall": {"average": "micro"}},
    "ssl": {
        "contrastive": {
            "len_segment": "dataset_len_segment",
            "num_f_maps": "model_features",
        }
    },
    "training": {
        "lr": 1e-4,  # float; learning rate
        "device": "cpu",  # str; device
        "augment_train": 0,  # [0, 1]; either 1 to use augmentations during training or 0 to not use
        "ssl_weights": {
            "contrastive": 0.1
        },  # dict; dictionary of SSL loss function weights
        "num_epochs": 50,  # int; number of epochs
        "to_ram": True,  # bool; transfer the dataset to RAM for training (preferred if the dataset fits in working memory)
        "batch_size": 16,  # int; batch size
        "model_save_epochs": 5,  # int; interval for saving training checkpoints (the last epoch is always saved)
        "test_frac": 0.2,
    },  # float; fraction of dataset to use as test
}


def test_task_update():
    """
    Test `dlc2action.task.task_dispatcher.TaskDispatcher.update_task`

    Update a task with set parameters and check that all parameter groups get to the end destination.
    """

    task = TaskDispatcher(parameters)
    task.update_task(update)
    # check model parameters
    model = task.task.model
    # check dataset parameters
    assert task.task.train_dataloader.dataset[0]["input"]["coords"].shape[-1] == 128
    length = int(floor((128 - 5) / 2 + 1))
    length = floor((length - 5) / 2 + 1)
    features = length * (16 // 4)
    assert model.ssl[0].conv_1x1_out.in_channels == features
    # check feature extraction parameters
    sample = task.task.train_dataloader.dataset[0]["input"]
    assert len(sample.keys()) == 2
    # check exclusive
    assert task.task.loss.exclusive == False
    # check ssl
    assert len(model.ssl_type) == 1
    # check metrics
    assert len(task.task.metrics) == 2
    # check loss parameters
    assert not task.task.loss.focal
    # check metrics parameters
    assert task.task.metrics["recall"].average == "micro"
    # check lr
    assert task.task.lr == 1e-4
    # check device
    assert str(task.task.device) == "cpu"
    # check augment_train
    assert task.task.augment_train == 0
    # check ssl_weights
    assert task.task.ssl_weights[0] == 0.1
    # check num_epochs
    assert task.task.num_epochs == 50
    # check batch size
    assert task.task.train_dataloader.batch_size == 16
    # check model_save_epochs
    assert task.task.model_save_epochs == 5
    # check test_frac
    assert task.task.test_dataloader is not None and len(task.task.test_dataloader) != 0


# test_task_update()
