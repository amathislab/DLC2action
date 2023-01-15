#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action import options
from dlc2action.project import Project
import pytest
from typing import List
import os

constructors = list(options.ssl_constructors.keys())


@pytest.mark.parametrize(
    "ssl_names", [[x] for x in constructors] + [constructors[:2], constructors[:3]]
)
def test_ssl(ssl_names: List):
    """
    Test SSL constructors

    Run one-epoch episodes with each SSL constructor + combinations of two and three.
    """

    Project.remove_project("test_ssl")
    path = os.path.join(os.path.dirname(__file__), "data")
    project = Project(
        "test_ssl",
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
                "partition_method": "random", 
                "val_frac": 0.5, 
                "normalize": False,
                "num_epochs": 1,
            }
        }
    )
    project.run_episode("test")
    Project.remove_project("test_ssl")


# names = [[x] for x in constructors] + [constructors[:2], constructors[:3]]
# test_ssl(["order"])
# for names in [[x] for x in constructors] + [constructors[:2], constructors[:3]]:
#     print(names)
#     test_ssl(names)
