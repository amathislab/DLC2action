#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
from pathlib import Path

import pytest
from dlc2action.project import Project
import yaml

device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]

def test_delete():
    """
    Test `dlc2action.project.project.Project.evaluate`

    Check that everything runs successfully.
    """

    Project.remove_project("test_delete_datasets")
    project_path = os.path.join(str(Path.home()), "DLC2Action", "test_delete_datasets")
    assert not os.path.exists(project_path)
    project = Project(
        "test_delete_datasets",
        data_type="simba",
        annotation_type="simba",
        data_path=crim_data_path,
        annotation_path=crim_data_path,
    )
    project.update_parameters(
        {
            "data": {
                "data_suffix": ".csv", # set; the data files should have the format of {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
                "canvas_shape": [1290, 730], # list; the size of the canvas where the pose was defined
                "annotation_suffix": ".csv", # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
                "use_features" : False
            },
            "general": {
                "exclusive": True, # bool; if true, single-label classification is used; otherwise multi-label
                "only_load_annotated": True,
                "metric_functions": {"f1"},
                "num_cpus": 10, # int; the number of CPUs to use in data processing (by default all are used)
            },
            "training": {
                "partition_method": "time:strict",
                "val_frac": 0.5,
                "normalize": False,
                "num_epochs": 1,
                "device": device
            },
            "features":
            {
                "keys": {"coords"}
            }
        }
    )
    project.run_episode("first")
    project.run_episode(
        "second",
        parameters_update={"general": {"overlap" : 90}},
        remove_saved_features=True,
    )
    assert len(os.listdir(os.path.join(project.project_path, "saved_datasets"))) == 2
    project.remove_saved_features()
    assert len(os.listdir(os.path.join(project.project_path, "saved_datasets"))) == 0
    Project.remove_project("test_delete_datasets")


if __name__ == "__main__":
    test_delete()
