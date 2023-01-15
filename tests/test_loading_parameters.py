#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.project import Project
from collections import Mapping
import pytest
from dlc2action import options
import os


def test_loading_parameters():
    """
    Check parameter loading in `dlc2action.project.project.Project`

    Run an episode, load its parameters and make sure they are the same as used when creating it
    """

    Project.remove_project("test_loading_parameters")
    path = os.path.join(os.path.dirname(__file__), "data")
    project = Project(
        "test_loading_parameters",
        data_type="dlc_track",
        annotation_type="csv",
        data_path=path,
        annotation_path=path,
    )
    project.update_parameters(
        {
            "data": {
                "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv", # set; the data files should have the format of {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
                "canvas_shape": [1000, 500], # list; the size of the canvas where the pose was defined
                "annotation_suffix": ".csv", # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
                "fps": 25,
            },
            "general": {
                "exclusive": True, # bool; if true, single-label classification is used; otherwise multi-label
                "only_load_annotated": True,
                "metric_functions": {"f1"}
            }, 
            "training": {
                "partition_method": "time:strict", 
                "val_frac": 0.2, 
                "normalize": False,
                "num_epochs": 1,
            }
        }
    )
    project.run_episode("test", parameters_update={"model": {"num_f_maps": 64}})
    loaded_parameters = project._episodes().load_parameters("test")
    parameters = project._read_parameters()
    parameters = project._update(parameters, {"model": {"num_f_maps": 64}})
    parameters = project._fill(
        parameters, "test", only_load_model=True, continuing=True
    )
    for group, dic in loaded_parameters.items():
        for key, pars in dic.items():
            if isinstance(pars, Mapping) and key not in ["correction", "stats"]:
                for small_key, par in pars.items():
                    if (
                        parameters[group][key].get(small_key, None)
                        not in options.blanks
                    ):
                        assert (
                            par is None
                            and parameters[group][key].get(small_key, None) is None
                        ) or par == parameters[group][key].get(small_key, None)
            elif (
                key not in ["partition_method", "stats"]
                and parameters[group].get(key, None) not in options.blanks
            ):
                assert (
                    pars is None and parameters[group].get(key, None) is None
                ) or pars == parameters[group].get(key, None)
    Project.remove_project("test_loading_parameters")


# test_loading_parameters()
