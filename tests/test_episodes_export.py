#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import shutil

from dlc2action.project import Project
import pytest
import os
import shutil


def make_project(name):
    Project.remove_project(name)
    path = os.path.join(os.path.dirname(__file__), "data")
    project = Project(
        name,
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


# test_episodes_export()
