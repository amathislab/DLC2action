#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import pytest
from dlc2action.project import Project
import yaml
import sys
import os

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]


skip_when_not_direct = pytest.mark.skipif(
    not any(f.endswith(os.path.basename(__file__)) for f in sys.argv),
    reason="Skipped when not run directly"
)


@skip_when_not_direct
@pytest.mark.parametrize("exclusive", [True, False])
def test_running(exclusive: bool):
    """
    Check `dlc2action.project.project.Project.run_episode` function

    Make sure that on average the accuracy metric improves sufficiently after 40 epochs
    """

    Project.remove_project("test_training")
    project = Project(
        "test_training",
        data_type="simba",
        annotation_type="simba",
        data_path=crim_data_path,
        annotation_path=crim_data_path,
    )

    project.update_parameters(
        {
            "data": {
                "data_suffix": ".csv",
                "canvas_shape": [1290, 730],
                "annotation_suffix": ".csv",
                "use_features": False
            },
            "general": {
                "exclusive": exclusive,
                "only_load_annotated": True,
                "metric_functions": {"accuracy"},
            },
            "training": {
                "partition_method": "time:strict",
                "val_frac": 0.3,
                "normalize": False,
                "num_epochs": 5,
            },
        }
    )
    for i in range(2):
        project.remove_episode(f"test_{i}")
        project.run_episode(f"test_{i}", n_seeds=2)
    episodes = project.list_episodes(display_parameters=["results/accuracy"])
    Project.remove_project("test_training")
    assert all(episodes.mean() > 0.16)
