#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os

import pytest
from dlc2action.project import Project
import yaml

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]


@pytest.mark.parametrize(
    "augment_n,mode,exclusive,use_paths",
    [
        (0, "train", True, True),
        (10, "test", True, False),
        (1, "val", False, True),
        (1, "all", False, False),
    ],
)
def test_prediction(augment_n: int, mode: str, exclusive: bool, use_paths: bool):
    """
    Test `dlc2action.project.project.Project.run_prediction`

    Check that it runs without errors with different modes and augmentation numbers.
    """

    Project.remove_project("test_prediction")

    project = Project(
        "test_prediction",
        data_type="simba",
        annotation_type="simba",
        data_path=crim_data_path,
        annotation_path=crim_data_path,
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
                "behaviors": [
                    "approach",
                    "attack",
                    "copulation",
                    "chase",
                    "circle",
                    "drink",
                    "eat",
                    "clean",
                    "sniff",
                    "up",
                    "walk_away",
                ],
                "use_features" : False
            },
            "general": {
                "model_name": "ms_tcn3",
                "exclusive": exclusive,
                "metric_functions": {"f1"},
            },
            "training": {
                "num_epochs": 1,
                "partition_method": "time:strict",
                "val_frac": 0.3,
                "test_frac": 0.1,
                "skip_normalization_keys": ["speed_direction", "coord_diff"],
            },
        }
    )
    if use_paths:
        file_paths = set([os.path.join(crim_data_path, x) for x in os.listdir(crim_data_path)])
    else:
        file_paths = None
    project.run_episode("test")
    project.run_prediction(
        "prediction",
        episode_names=["test"],
        augment_n=augment_n,
        mode=mode,
        file_paths=file_paths,
    )
    if use_paths:
        assert project._predictions().get_saved_data_path(
            "prediction"
        ) == project._dataset_store_path("prediction")
    else:
        assert project._predictions().get_saved_data_path(
            "prediction"
        ) == project._dataset_store_path("test")
    Project.remove_project("test_prediction")


# augment_n = 10
# mode = 'test'
# exclusive = True
# use_paths = False
# test_prediction(augment_n=augment_n, mode=mode, exclusive=exclusive, use_paths=use_paths)
