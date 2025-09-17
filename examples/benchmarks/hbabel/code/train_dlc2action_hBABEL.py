#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import pickle

import numpy as np
from dlc2action.project.project import Project
from dump_param import dump_param

DATA_PATH = f"path/to/converted_data"
ANNOTATION_PATH = f"path/to/converted_annotations"
DATA_TYPE = "dlc_track"
ANNOTATION_TYPE = "dlc"
PROJECTS_PATH = f"path/to/projects"


def run_main(
    param,
    test=False,
    hs=False,
    epochs=10,
    part_method="time:strict",
    split_path=None,
    gpu=0,
    experiment_suffix="",
    model_list=["transformer", "ms_tcn3", "mlp"],
    test_ratio=0.7,
    continue_from_epoch=None,
):

    print("Creating project")

    project_name = f"hBABEL_d2a"

    project = Project(
        project_name,
        data_path=DATA_PATH,
        annotation_path=ANNOTATION_PATH,
        projects_path=PROJECTS_PATH,
        data_type=DATA_TYPE,
        annotation_type=ANNOTATION_TYPE,
    )

    # Load parameters (see dump_param.py)
    print("Update parameters ... ")
    project.update_parameters(param)

    # Update partition method
    if part_method == "file":
        assert split_path is not None
        project.update_parameters(
            {
                "training": {"partition_method": part_method, "split_path": split_path},
            }
        )
    else:
        project.update_parameters(
            {
                "training": {"partition_method": part_method},
            }
        )

    # Update gpu device
    if part_method != "file":
        project.update_parameters(
            {"training": {"device": f"cuda:{gpu}", "val_frac": test_ratio}}
        )
    else:
        project.update_parameters({"training": {"device": f"cuda:{gpu}"}})

    # Test for data loading
    if test:
        print("Run load test episode")
        project.run_episode(
            "load_test",
            force=True,
            parameters_update={
                "training": {"num_epochs": 2},
            },
        )

    ext = "_" + part_method
    for model in model_list:
        if continue_from_epoch is not None:
            load_episode = f"{model}_best{ext}{experiment_suffix}"
            load_search = f"{model}_search{experiment_suffix}"
            episode_name = (
                f"{model}_best_cont_{continue_from_epoch}{ext}{experiment_suffix}"
            )
            hs = False
        # Run hyperparameter search
        if hs:
            project.run_default_hyperparameter_search(
                f"{model}_search{experiment_suffix}",
                model_name=model,
                metric="f1",
                num_epochs=3,
                n_trials=5,
                prune=True,
                best_n=3,
                force=True,
            )

        # Train models
        if hs:
            project.run_episode(
                f"{model}_best{ext}{experiment_suffix}",
                load_search=f"{model}_search{experiment_suffix}",  # loading the search
                force=True,  # when force=True, if an episode with this name already exists it will be overwritten -> use with caution!
                parameters_update={
                    "general": {"model_name": model},
                    "training": {"num_epochs": epochs},
                },
                n_seeds=10,  # we will repeat the experiment 3 times to get an estimation for how stable our results are
            )
        elif continue_from_epoch is not None:
            project.run_episode(
                episode_name,
                force=True,  # when force=True, if an episode with this name already exists it will be overwritten -> use with caution!
                parameters_update={
                    "general": {"model_name": model},
                    "training": {"num_epochs": epochs},
                },
                n_runs=10,  # we will repeat the experiment 3 times to get an estimation for how stable our results are
                load_episode=load_episode,
                load_search=load_search,
                load_epoch=continue_from_epoch,
            )
        else:
            project.run_episode(
                f"{model}_best{ext}{experiment_suffix}",
                force=True,  # when force=True, if an episode with this name already exists it will be overwritten -> use with caution!
                parameters_update={
                    "general": {"model_name": model},
                    "training": {"num_epochs": epochs},
                },
                n_seeds=10,  # we will repeat the experiment 3 times to get an estimation for how stable our results are
            )


# Update parameters, all parameters are stored in the dump_param.py function
test = True
hs = True
part_method = "file"
# split_path = "/media18/data/andy/BABEL_teach/hBABEL_4D2A_traintest/split_file.txt"
gpu = 0
model_list = [
    "transformer",
    "asformer",
    "ms_tcn3",
    "mlp",
    "c2f_tcn",
    "c2f_transformer",
    "edtcn",
]
continue_from_epoch = None  # int or None
epochs = 100


for hierarchy in ["frame"]:
    for top in [60, 30, 10, 90]:

        dump_param(hierarchy, top)
        with open("parameters.p", "rb") as f:
            param = pickle.load(f)
        split_path = os.path.join(
            ANNOTATION_PATH, f"split_file_{hierarchy}_top_{top}.txt"
        )
        run_main(
            param=param,
            test=test,
            hs=hs,
            epochs=epochs,
            gpu=gpu,
            experiment_suffix=f"_{hierarchy}_top_{top}",
            model_list=model_list,
            part_method=part_method,
            split_path=split_path,
            continue_from_epoch=continue_from_epoch,
        )
