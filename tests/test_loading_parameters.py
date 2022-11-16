#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
from dlc2action.project import Project
from collections import Mapping
import pytest
from dlc2action import options


def test_loading_parameters():
    """
    Check parameter loading in `dlc2action.project.project.Project`

    Run an episode, load its parameters and make sure they are the same as used when creating it
    """

    Project.remove_project("test_loading_parameters")
    project = Project(
        "test_loading_parameters",
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
