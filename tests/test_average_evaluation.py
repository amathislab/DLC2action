#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
import yaml

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]


def test_evaluation():
    """
    Test `dlc2action.project.project.Project.evaluate` and `dlc2action.project.project.Project.get_results_table`

    Check that everything runs successfully.
    """

    Project.remove_project("test_average_evaluation")
    project = Project(
        "test_average_evaluation",
        data_type="simba",
        annotation_type="simba",
        data_path=crim_data_path,
        annotation_path=crim_data_path,
    )
    project.update_parameters(
        {
            "data": {
                "canvas_shape": [1290, 730],
                "likelihood_threshold": 0.8,
                "len_segment": 256,
                "overlap": 200,
                "data_suffix": ".csv",
                "annotation_suffix": ".csv",
                "behaviors": [
                    "approach",
                    "attack",
                    "copulation",
                    "chase",
                    "circle",
                    "drink",
                    "eat",
                    "clean",
                    "sniff",
                    "up",
                    "walk_away",
                ],
                "use_features" : False
            },
            "general": {
                "model_name": "ms_tcn3",
                "exclusive": True,
                "metric_functions": {"f1"},
            },
            "training": {
                "num_epochs": 1,
                "partition_method": "time:strict",
                "val_frac": 0.3,
                "skip_normalization_keys": ["speed_direction", "coord_diff"],
            },
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
