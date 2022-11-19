#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import pytest
import os


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
        data_type="dlc_track",
        annotation_type="boris",
        data_path="/home/liza/data/cricket",
        annotation_path="/home/liza/data/cricket",
    )
    project.update_parameters(
        {
            "data": {
                "behaviors": [
                    "Search",
                    "Grooming",
                    "Pursuit",
                    "Inactive",
                    "Consumption",
                    "Capture",
                ],
                "data_suffix": {
                    "DLC_resnet50_preycapSep30shuffle1_20000_bx_filtered.h5",
                },
                "default_agent_name": "mouse+single",
                "canvas_shape": [2250, 1250],
                "interactive": True,
            },
            "general": {
                "exclusive": True,
                "ignored_clips": None,
                "len_segment": 512,
                "overlap": 100,
            },
            "features": {"interactive": True},
            "training": {"num_epochs": 1, "test_frac": 0.1},
        }
    )
    folder = "/home/liza/data/cricket"
    if use_paths:
        file_paths = set([os.path.join(folder, x) for x in os.listdir(folder)])
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


# augment_n = 0
# mode = 'train'
# exclusive = True
# use_paths = True
# test_prediction(augment_n=augment_n, mode=mode, exclusive=exclusive, use_paths=use_paths)
