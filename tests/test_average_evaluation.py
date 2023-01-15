#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import pytest
import os


def test_evaluation():
    """
    Test `dlc2action.project.project.Project.evaluate` and `dlc2action.project.project.Project.get_results_table`

    Check that everything runs successfully.
    """
    PROJECTS_PATH = os.path.join(os.path.expanduser('~'), "Projects")

    path = os.path.join(os.path.dirname(__file__), "data")
    Project.remove_project("test_average_evaluation")
    project = Project(
        "test_average_evaluation",
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
                "metric_functions": {"f1"}
            }, 
            "training": {
                "partition_method": "time:strict", 
                "val_frac": 0.5, 
                "normalize": False,
                "num_epochs": 1,
            }
        }
    )
    project.run_episode("first")
    project.run_episode("second")
    project.evaluate(
        ["first", "second"],
        [None, 1],
        mode="val",
        parameters_update={"general": {"metric_functions": {"precision"}}},
    )
    project.evaluate(
        ["first"],
        mode="train",
        parameters_update={"general": {"metric_functions": {"recall"}}},
    )
    table = project.get_results_table(["first"])
    assert "first f1" in table.columns
    assert "first precision" in table.columns
    assert "first recall" not in table.columns
    Project.remove_project("test_average_evaluation")


# test_evaluation()
