#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import pytest
import os
from copy import copy


@pytest.mark.parametrize("add_load_experiment", [True, False])
@pytest.mark.parametrize("add_load_epoch", [True, False])
def test_project_fill(add_load_experiment: bool, add_load_epoch: bool):
    """
    Test `dlc2action.project.project.Project._fill`

    Check the filled parameters in several conditions.
    """

    Project.remove_project("test_project_fill")
    path = os.path.join(os.path.dirname(__file__), "data")
    project = Project(
        "test_project_fill",
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
                "num_epochs": 3,
                "model_save_epochs": 2,
            }
        }
    )
    project.run_episode("old")
    if add_load_experiment:
        load_experiment = "old"
    else:
        load_experiment = None
    if add_load_epoch:
        load_epoch = 1
    else:
        load_epoch = None

    pars = project._read_parameters()
    split_info = copy(pars["training"])
    split_info["only_load_annotated"] = pars["general"]["only_load_annotated"]
    split_info["len_segment"] = pars["general"]["len_segment"]
    split_info["overlap"] = pars["general"]["overlap"]
    split_path = project._default_split_file(split_info)
    if not add_load_experiment:
        checkpoint_path = -100
    elif add_load_epoch:
        checkpoint_path = os.path.join(
            project.project_path, "results", "model", "old", "epoch2.pt"
        )
    else:
        checkpoint_path = os.path.join(
            project.project_path, "results", "model", "old", "epoch3.pt"
        )
    saved_data_path = os.path.join(project.project_path, "saved_datasets", "old.pickle")
    feature_save_path = os.path.join(project.project_path, "saved_datasets", "old")
    partition_method = "file"

    parameters = project._fill(
        project._read_parameters(),
        episode_name="new",
        load_experiment=load_experiment,
        load_epoch=load_epoch,
    )
    Project.remove_project("test_project_fill")
    if parameters["training"].get("checkpoint_path") is None:
        parameters["training"]["checkpoint_path"] = -100
    assert parameters["training"]["split_path"] == split_path
    assert parameters["training"].get("checkpoint_path") == checkpoint_path
    assert parameters["data"]["saved_data_path"] == saved_data_path
    assert parameters["data"]["feature_save_path"] == feature_save_path
    assert parameters["training"]["partition_method"] == partition_method


# test_project_fill(True, True)
