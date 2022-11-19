#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import pytest


@pytest.mark.skip
@pytest.mark.parametrize("exclusive", [True, False])
def test_running(exclusive: bool):
    """
    Check `dlc2action.project.project.Project.run_episode` function

    Make sure that on average the accuracy metric improves sufficiently after 40 epochs
    """

    Project.remove_project("test_training")
    project = Project(
        "test_training",
        data_type="dlc_tracklet",
        annotation_type="dlc",
        data_path="/home/liza/data/marmoset_sample",
        annotation_path="/home/liza/data/marmoset_sample",
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
            "training": {"num_epochs": 40},
        }
    )
    for i in range(5):
        project.remove_episode(f"test_{i}")
        project.run_episode(f"test_{i}")
    episodes = project.list_episodes(display_parameters=["results/accuracy"])
    Project.remove_project("test_training")
    assert all(episodes.mean() > 0.16)


# test_running(True)
# test_running(False)
