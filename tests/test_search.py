#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import pytest


@pytest.mark.skip
@pytest.mark.parametrize("exclusive", [True, False])
def test_search(exclusive: bool):
    """
    Test `dlc2action.project.project.Project.run_hyperparameter_search`

    Check that it chooses the best learning rate (1e-3 vs 1e-15).
    """

    Project.remove_project("test_search")
    project = Project(
        "test_search",
        data_type="dlc_track",
        annotation_type="boris",
        data_path="/home/liza/data/cricket",
        annotation_path="/home/liza/data/cricket",
    )
    project.update_parameters(
        {
            "data": {
                "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv", # set; the data files should have the format of {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
                "canvas_shape": [1000, 500], # list; the size of the canvas where the pose was defined
                "annotation_suffix": ".csv", # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
            },
            "general": {
                "exclusive": True, # bool; if true, single-label classification is used; otherwise multi-label
            }
        }
    )
    best_params = project.run_hyperparameter_search(
        "search",
        search_space={"training/lr": ("categorical", [1e-3, 1e-20])},
        metric="recall",
        n_trials=4,
    )
    Project.remove_project("test_search")
    assert best_params["training/lr"] == 1e-3


# test_search(False)
