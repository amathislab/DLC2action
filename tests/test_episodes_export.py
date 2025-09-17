#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import shutil

import pytest
from dlc2action.project import Project
import yaml

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]

def make_project(name):
    """Make a new project with the given name"""
    Project.remove_project(name)

    project = Project(
        name,
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
                "metric_functions": {"f1"}
            },
            "training": {
                "partition_method": "time:strict",
                "val_frac": 0.2,
                "normalize": False,
                "num_epochs": 1,
            }
        }
    )
    return project


def test_episodes_export():
    """
    Test export/import

    Export an episode to another project and load it to run another one
    """

    project = make_project("test_episodes_export_1")
    project.run_episode("test")
    if os.path.exists(os.path.join(".", "test_export")):
        shutil.rmtree(os.path.join(".", "test_export"))
    project.export_episodes(["test"], output_directory=".", name="test_export")

    project = make_project("test_episodes_export_2")
    project.import_episodes(
        os.path.join(".", "test_export"), name_map={"test": "old_test"}
    )
    project.run_episode("test", load_episode="old_test")
    assert len(project.list_episodes()) == 2

    Project.remove_project("test_episodes_export_1")
    Project.remove_project("test_episodes_export_2")
    shutil.rmtree(os.path.join(".", "test_export"))

