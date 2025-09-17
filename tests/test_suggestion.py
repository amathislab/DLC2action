#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
from pathlib import Path

import pytest
from dlc2action.project import Project
import yaml

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]


@pytest.mark.parametrize(
    "mode,use_paths,use_suggestion",
    [
        ("train", True, True),
        ("test", False, True),
        ("val", True, False),
        ("all", False, False),
    ],
)
@pytest.mark.parametrize("use_excluded,use_error", [(True, False)])
def test_suggestion(
    mode: str,
    use_suggestion: bool,
    use_excluded: bool,
    use_error: bool,
    use_paths: bool,
):
    """
    Test `dlc2action.project.project.Project.run_suggestion`

    Check that it works (or raises the correct error) with a range of parameters.
    """

    Project.remove_project("test_suggestion")
    project_path = os.path.join(str(Path.home()), "DLC2Action", "test_suggestion")
    assert not os.path.exists(project_path)
    project = Project(
        "test_suggestion",
        data_type="simba",
        annotation_type="simba",
        data_path=crim_data_path,
        annotation_path=crim_data_path,
    )
    project.update_parameters(
        {
            "data": {
                "data_suffix": ".csv",  # set; the data files should have the format of {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
                "canvas_shape": [1290, 730],
                "annotation_suffix": ".csv",  # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
                "use_features": False
            },
            "general": {
                "model_name": "ms_tcn3",
                "exclusive": True,  # bool; if true, single-label classification is used; otherwise multi-label
                "only_load_annotated": True,
                "metric_functions": {"f1"},
            },
            "training": {
                "partition_method": "time:strict",
                "val_frac": 0.5,
                "test_frac": 0.2,
                "normalize": False,
                "num_epochs": 1,
            },
        }
    )
    if use_error:
        project.run_episode("error")
        error_episode = "error"
        error_class = "StartEnd"
    else:
        error_episode = None
        error_class = None
    if use_suggestion or use_excluded:
        project.run_episode("classes")
        suggestion_episode = "classes"
    else:
        suggestion_episode = None
    if use_suggestion:
        suggestion_classes = ["approach"]
    else:
        suggestion_classes = None
    if use_excluded:
        exclude_classes = ["other"]
    else:
        exclude_classes = None
    if use_paths:
        file_paths = [
            os.path.join(crim_data_path, x) for x in os.listdir(crim_data_path)
        ]
    else:
        file_paths = None

    if suggestion_episode is None and error_episode is None:
        with pytest.raises(ValueError):
            project.run_suggestion(
                "suggestion",
                error_episode=error_episode,
                suggestion_episodes=[suggestion_episode],
                mode=mode,
                suggestion_classes=suggestion_classes,
                exclude_classes=exclude_classes,
                error_class=error_class,
                file_paths=file_paths,
            )
    else:
        project.run_suggestion(
            "suggestion",
            error_episode=error_episode,
            suggestion_episodes=[suggestion_episode],
            mode=mode,
            suggestion_classes=suggestion_classes,
            exclude_classes=exclude_classes,
            error_class=error_class,
            file_paths=file_paths,
        )
        suggestions_path = os.path.join(
            project.project_path, "results", "suggestions", "suggestion"
        )
        n_files = 0
        if exclude_classes is not None or error_episode is not None:
            assert os.path.exists(project._al_points_path("suggestion"))
            n_files += 1
        if suggestion_classes is not None:
            n_files += 1
        assert len(os.listdir(suggestions_path)) >= n_files
    Project.remove_project("test_suggestion")


if __name__ == "__main__":
    test_suggestion("train", False, True, True, False)
