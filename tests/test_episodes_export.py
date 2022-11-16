#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
import shutil

from dlc2action.project import Project
import pytest
import os
import shutil


def make_project(name):
    Project.remove_project(name)
    project = Project(
        name,
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
            "training": {"num_epochs": 1},
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
