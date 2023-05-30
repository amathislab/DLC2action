#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.task.task_dispatcher import TaskDispatcher
import pytest
import os
import shutil

path = os.path.join(os.path.dirname(__file__), "data")

parameters = {
    "data": {
        "data_path": path,
        "annotation_path": path,
        "annotation_suffix": {"2.csv"},
        "data_suffix": {
            "2DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
        },
    },
    "features": {
        "keys": ["coords", "intra_distance"],
    },
    "model": {
        "input_dims": "dataset_features",
        "num_f_maps": 16,  # int; number of feature maps
        "feature_dim": 16,
    },
    "general": {
        "ignored_clips": ["single"],
        "data_type": "dlc_track",
        "annotation_type": "csv",
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
        "overlap": 0.8
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
        "partition_method": "random"
    },  # float; fraction of dataset to use as test
}


def test_task_creation():
    """
    Test `dlc2action.task.task_dispatcher.TaskDispatcher.initialize_task`

    Create a task with set parameters and check that all parameter groups get to the end destination.
    """

    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    task = TaskDispatcher(parameters)
    # check feature extraction parameters
    sample = task.task.train_dataloader.dataset[0]["input"]
    assert len(sample.keys()) == 2
    # check model parameters
    model = task.task.model
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
    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)


if __name__ == "__main__":
    test_task_creation()
