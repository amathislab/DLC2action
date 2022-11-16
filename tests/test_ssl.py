#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action import options
from dlc2action.project import Project
import pytest
from typing import List

constructors = list(options.ssl_constructors.keys())


@pytest.mark.parametrize(
    "ssl_names", [[x] for x in constructors] + [constructors[:2], constructors[:3]]
)
def test_ssl(ssl_names: List):
    """
    Test SSL constructors

    Run one-epoch episodes with each SSL constructor + combinations of two and three.
    """

    Project.remove_project("test_ssl")
    project = Project(
        "test_ssl",
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
    project.run_episode("test")
    Project.remove_project("test_ssl")


# names = [[x] for x in constructors] + [constructors[:2], constructors[:3]]
# test_ssl(["order"])
# for names in [[x] for x in constructors] + [constructors[:2], constructors[:3]]:
#     print(names)
#     test_ssl(names)
