#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import pytest
from dlc2action.project import Project
import yaml
import os
import sys

skip_when_not_direct = pytest.mark.skipif(
    not any(f.endswith(os.path.basename(__file__)) for f in sys.argv),
    reason="Skipped when not run directly"
)

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]

@skip_when_not_direct
@pytest.mark.parametrize("exclusive", [True, False])
def test_search(exclusive):
    """
    Test `dlc2action.project.project.Project.run_hyperparameter_search`

    Check that it chooses the best learning rate (1e-3 vs 1e-15).
    """

    Project.remove_project("test_search")
    project = Project(
        "test_search",
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
                "copulation",
                "chase",
            ],
            "use_features": False
        },
        "general": {
            "model_name": "ms_tcn3",
            "exclusive": exclusive,
            "metric_functions": {"precision", "recall", "f1"},
        },
        "metrics": {
            "recall": {"ignored_classes": {}, "average": "macro"},
            "precision": {"ignored_classes": {}, "average": "macro"},
            "f1": {"ignored_classes": {}, "average": "macro"},
        },
        "training": {
            "num_epochs": 10,
            "partition_method": "time:strict",
            "val_frac": 0.5,
            "skip_normalization_keys": ["speed_direction", "coord_diff"],
        }
        },
    )
    best_params = project.run_hyperparameter_search(
        "search",
        search_space={"training/lr": ("categorical", [1e-3, 1e-20])},
        metric="accuracy",
        n_trials=4,
        make_plots=False,

    )
    Project.remove_project("test_search")
    assert best_params["training/lr"] == 1e-3


if __name__ == "__main__":
    test_search(True)
