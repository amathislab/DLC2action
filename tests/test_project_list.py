#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action.project import Project
import pytest


def test_project_list():
    """
    Test `dlc2action.project.meta.SavedRuns.list_episodes`

    Check the size of the output.
    """

    Project.remove_project("test_project_list")
    project = Project(
        "test_project_list",
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
    for i in range(3):
        project.remove_episode(f"test_{i}")
        project.run_episode(f"test_{i}")
    episodes = project.list_episodes(
        display_parameters=["training/lr", "general/len_segment", "meta/training_time"]
    )
    assert len(episodes) == 3
    assert len(episodes.columns) == 3

    Project.remove_project("test_project_list")


# test_project_list()
