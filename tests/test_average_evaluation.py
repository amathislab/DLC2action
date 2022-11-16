#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action.project import Project
import pytest


def test_evaluation():
    """
    Test `dlc2action.project.project.Project.evaluate`

    Check that everything runs successfully.
    """

    Project.remove_project("test_average_evaluation")
    project = Project(
        "test_average_evaluation",
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
                "metric_functions": {"f1"},
            },
            "features": {"interactive": True},
            "training": {"num_epochs": 1, "batch_size": 32, "device": "cuda:0"},
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
