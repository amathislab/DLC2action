#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action.project import Project
import pytest
import os


def test_delete():
    """
    Test `dlc2action.project.project.Project.evaluate`

    Check that everything runs successfully.
    """

    Project.remove_project("test_delete_datasets")
    project = Project(
        "test_delete_datasets",
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
    project.run_episode("first")
    project.run_episode(
        "second", parameters_update={"general": {"overlap": 50}}, remove_saved_features=True
    )
    assert len(os.listdir(os.path.join(project.project_path, "saved_datasets"))) == 2
    project.remove_saved_features()
    assert len(os.listdir(os.path.join(project.project_path, "saved_datasets"))) == 0
    Project.remove_project("test_delete_datasets")


# test_delete()
