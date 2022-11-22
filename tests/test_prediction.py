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
    path = os.path.join(os.path.dirname(__file__), "data")
    project = Project(
        "test_prediction",
        data_type="dlc_track",
        annotation_type="csv",
        data_path=path,
        annotation_path=path,
    )
    project.update_parameters(
        {
            "data": {
                "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv", # set; the data files should have the format of {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
                "canvas_shape": [1000, 500], # list; the size of the canvas where the pose was defined
                "annotation_suffix": ".csv", # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
                "fps": 25,
            },
            "general": {
                "exclusive": True, # bool; if true, single-label classification is used; otherwise multi-label
                "only_load_annotated": True,
                "metric_functions": {"f1"},
                "overlap": 0.8,
            }, 
            "training": {
                "partition_method": "time:strict", 
                "val_frac": 0.4, 
                "test_frac": 0.3,
                "normalize": False,
                "num_epochs": 1,
            }
        }
    )
    if use_paths:
        file_paths = set([os.path.join(path, x) for x in os.listdir(path)])
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
