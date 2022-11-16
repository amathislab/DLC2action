#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action.project import Project
import pytest
import os
from copy import copy


@pytest.mark.parametrize("add_load_experiment", [True, False])
@pytest.mark.parametrize("add_load_epoch", [True, False])
def test_project_fill(add_load_experiment: bool, add_load_epoch: bool):
    """
    Test `dlc2action.project.project.Project._fill`

    Check the filled parameters in several conditions.
    """

    Project.remove_project("test_project_fill")
    project = Project(
        "test_project_fill",
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
            "training": {"num_epochs": 3, "model_save_epochs": 2},
        }
    )
    project.run_episode("old")
    if add_load_experiment:
        load_experiment = "old"
    else:
        load_experiment = None
    if add_load_epoch:
        load_epoch = 1
    else:
        load_epoch = None

    pars = project._read_parameters()
    split_info = copy(pars["training"])
    split_info["only_load_annotated"] = pars["general"]["only_load_annotated"]
    split_info["len_segment"] = pars["general"]["len_segment"]
    split_info["overlap"] = pars["general"]["overlap"]
    split_path = project._default_split_file(split_info)
    if not add_load_experiment:
        checkpoint_path = -100
    elif add_load_epoch:
        checkpoint_path = os.path.join(
            project.project_path, "results", "model", "old", "epoch2.pt"
        )
    else:
        checkpoint_path = os.path.join(
            project.project_path, "results", "model", "old", "epoch3.pt"
        )
    saved_data_path = os.path.join(project.project_path, "saved_datasets", "old.pickle")
    feature_save_path = os.path.join(project.project_path, "saved_datasets", "old")
    partition_method = "file"

    parameters = project._fill(
        project._read_parameters(),
        episode_name="new",
        load_experiment=load_experiment,
        load_epoch=load_epoch,
    )
    Project.remove_project("test_project_fill")
    if parameters["training"].get("checkpoint_path") is None:
        parameters["training"]["checkpoint_path"] = -100
    assert parameters["training"]["split_path"] == split_path
    assert parameters["training"].get("checkpoint_path") == checkpoint_path
    assert parameters["data"]["saved_data_path"] == saved_data_path
    assert parameters["data"]["feature_save_path"] == feature_save_path
    assert parameters["training"]["partition_method"] == partition_method


# test_project_fill(False, True)
