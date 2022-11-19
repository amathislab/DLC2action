#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import pytest


@pytest.mark.skip
@pytest.mark.parametrize("exclusive", [True, False])
def test_search(exclusive: bool):
    """
    Test `dlc2action.project.project.Project.run_hyperparameter_search`

    Check that it chooses the best learning rate (1e-3 vs 1e-15).
    """

    Project.remove_project("test_search")
    project = Project(
        "test_search",
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
            "training": {"num_epochs": 30},
        }
    )
    best_params = project.run_hyperparameter_search(
        "search",
        search_space={"training/lr": ("categorical", [1e-3, 1e-20])},
        metric="recall",
        n_trials=4,
    )
    Project.remove_project("test_search")
    assert best_params["training/lr"] == 1e-3


# test_search(False)
