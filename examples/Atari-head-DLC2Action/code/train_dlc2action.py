#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
from dlc2action.project.project import Project
from utils.data_load import Dataset
from utils.utils import *
import pickle
from dump_param import dump_param

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


DATA_PATH = "path/to/data"
DATA_TYPE = "dlc_track"
ANNOTATION_TYPE = "dlc"
PROJECTS_PATH = "path/to/projects"


def run_main(
    param,
    game=None,
    test=False,
    hs=False,
    epochs=10,
    random=False,
    use_i3d=False,
    gpu=1,
    model_list=["c2f_tcn", "transformer", "ms_tcn3", "mlp"],
    continue_training=False,
    experiment_name="",
    average=True,
):

    # DATA_PATH = DATA_PATH + "-upsampled" if not average else DATA_PATH
    (
        print("Creating project for all games")
        if game is None
        else print("Creating project for ", game)
    )
    i3d_ext = "" if not use_i3d else "_i3d"

    project_name = game + "_d2a_project_" + experiment_name + i3d_ext
    project_path = os.path.join(PROJECTS_PATH, "Atari-Head-D2A-projects")
    os.makedirs(project_path, exist_ok=True)

    if game is not None:
        data_path = os.path.join(DATA_PATH, game)
        annotation_path = os.path.join(DATA_PATH, game)
    else:
        data_path = DATA_PATH
        annotation_path = DATA_PATH

    project = Project(
        project_name,
        data_path=data_path,
        annotation_path=annotation_path,
        projects_path=project_path,
        data_type=DATA_TYPE,
        annotation_type=ANNOTATION_TYPE,
    )

    # Load parameters (see dump_param.py)
    print("Update parameters ... ")
    project.update_parameters(param)

    # Update partition method
    part_method = "time:strict"
    if random:
        part_method = "random"
    project.update_parameters(
        {
            "training": {"partition_method": part_method},
        }
    )

    # Add i3d features
    if use_i3d:
        project.update_parameters({"data": {"feature_suffix": "_i3d.npy"}})

    # Update gpu device
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
            n_runs=1,
        )

    # Run hyperparameter search
    if hs:
        for model in model_list:
            if not os.path.exists(
                os.path.join(
                    project_path,
                    project_name,
                    "results",
                    "searches",
                    f"{model}_search{i3d_ext}",
                )
            ):
                project.run_default_hyperparameter_search(
                    f"{model}_search_new{i3d_ext}",
                    model_name=model,
                    metric="f1",
                    num_epochs=10,
                    n_trials=5,
                    prune=True,
                    best_n=3,
                )
            else:
                print(f"{model}_search{i3d_ext} already exists")

    # Clean memory
    project.remove_datasets()

    ext = ""
    if not random:
        ext = "_timestrict"
    # Train models
    for model in model_list:

        if not continue_training:
            project.run_episode(
                f"{model}_best{ext}{i3d_ext}",
                load_search=f"{model}_search_new{i3d_ext}",  # loading the search
                force=True,
                parameters_update={
                    "general": {"model_name": model},
                    "training": {"num_epochs": epochs},
                },
                n_runs=5,
            )
        else:
            project.continue_episode(f"{model}_best{ext}{i3d_ext}", num_epochs=epochs)


# Update parameters, all parameters are stored in the dump_param.py function
dump_param()
dataset_template = Dataset()
test = True
hs = True
random = True
use_i3d = False
average = False
continue_training = False
gpu = 0
model_list = ["transformer", "ms_tcn3", "mlp", "c2f_tcn"]

with open("parameters.p", "rb") as f:
    param = pickle.load(f)

for game in dataset_template.game_names:
    run_main(
        param,
        game=game,
        test=test,
        hs=hs,
        epochs=120,
        use_i3d=use_i3d,
        gpu=gpu,
        model_list=model_list,
        random=random,
        continue_training=continue_training,
        average=average,
        experiment_name="random",
    )
