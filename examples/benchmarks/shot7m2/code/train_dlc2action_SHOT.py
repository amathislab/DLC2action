#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import pickle
from dlc2action.project.project import Project
from dump_param import dump_param

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# DATA_PATH = "path/to/converted_data"
# DATA_TYPE = "dlc_track"
# ANNOTATION_TYPE = "dlc"
# PROJECTS_PATH = "path/to/projects"

DATA_PATH = "/media1/data/andy/SHOT72/SHOT72_4D2A_traintest"
DATA_TYPE = "dlc_track"
ANNOTATION_TYPE = "dlc"
PROJECTS_PATH = "/media1/data/andy/SHOT72/SHOT72_D2A_projects"


def run_main(
    param,
    test=False,
    hs=False,
    epochs=10,
    part_method="time:strict",
    split_path=None,
    use_i3d=False,
    gpu=0,
    experiment_suffix="",
    model_list=["transformer", "ms_tcn3", "mlp"],
    test_ratio=None,
    continue_from_epoch=None,
    use_motion_bert=False,
):
    print("Creating project")

    i3d_ext = ""
    if use_i3d:
        i3d_ext = "_i3d"

    project_name = "SHOT7M2_d2a" + i3d_ext
    if use_motion_bert:
        project_name = project_name + "_motionbert2"
        model_list = ["motionbert"]
        print("Replaced model list with motionbert only")

    project = Project(
        project_name,
        data_path=DATA_PATH,
        annotation_path=DATA_PATH,
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
    else:
        assert test_ratio is not None
    # Add i3d features
    if use_i3d:
        project.update_parameters(
            {
                "data": {"feature_suffix": "_view3_0.npy"},
            }
        )

    # Test for data loading
    if test:
        print("Run load test episode")
        project.run_episode(
            "load_test",
            force=True,
            parameters_update={
                "training": {"num_epochs": 2, "device": f"cuda:{gpu}"},
            },
            n_seeds=1,
        )

    ext = "_" + part_method
    for model in model_list:
        load_search = None
        if continue_from_epoch is not None:
            load_episode = f"{model}_best{ext}{i3d_ext}{experiment_suffix}#0"
            load_search = f"{model}_search{i3d_ext}{experiment_suffix}"
            episode_name = f"{model}_best_cont_{continue_from_epoch}{ext}{i3d_ext}{experiment_suffix}"
            hs = False
        # Run hyperparameter search
        if hs:
            load_search = f"{model}_search{i3d_ext}{experiment_suffix}"
            project.run_default_hyperparameter_search(
                load_search,
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
                f"{model}_best{ext}{i3d_ext}{experiment_suffix}",
                load_search=load_search,
                force=True,  # when force=True, if an episode with this name already exists it will be overwritten -> use with caution!
                parameters_update={
                    "general": {"model_name": model},
                    "training": {
                        "num_epochs": epochs,
                        "partition_method": part_method,
                        "split_path": split_path,
                        "val_frac": test_ratio,
                        "device": f"cuda:{gpu}",
                    },
                },
                n_seeds=10,  # we will repeat the experiment 3 times to get an estimation for how stable our results are
            )
        elif continue_from_epoch is not None:
            load_search = None
            project.run_episode(
                episode_name,
                force=True,  # when force=True, if an episode with this name already exists it will be overwritten -> use with caution!
                parameters_update={
                    "general": {"model_name": model},
                    "model": {
                        "num_joints": 26,
                        "dim_feat": 256,
                        "dim_rep": 256,
                        "depth": 4,
                        "num_heads": 4,
                    },
                    "training": {
                        "num_epochs": epochs,
                        "partition_method": part_method,
                        "split_path": split_path,
                        "val_frac": test_ratio,
                        "device": f"cuda:{gpu}",
                    },
                },
                n_seeds=2,  # we will repeat the experiment 3 times to get an estimation for how stable our results are
                load_episode=load_episode,
                load_search=load_search,
                load_epoch=continue_from_epoch,
            )
        else:
            project.run_episode(
                f"{model}_best{ext}{i3d_ext}{experiment_suffix}",
                force=True,  # when force=True, if an episode with this name already exists it will be overwritten -> use with caution!
                parameters_update={
                    "general": {"model_name": model},
                    "training": {
                        "num_epochs": epochs,
                        "partition_method": part_method,
                        "split_path": split_path,
                        "val_frac": test_ratio,
                        "device": f"cuda:{gpu}",
                    },
                    "model": {
                        "num_joints": 26,
                        "dim_feat": 256,
                        "dim_rep": 256,
                        "depth": 4,
                        "num_heads": 4,
                    },
                    "features": {"keys": {"coords"}},
                },
                n_seeds=2,  # we will repeat the experiment 3 times to get an estimation for how stable our results are
            )


dump_param()
test = False
hs = False
part_method = "file"
# split_path = "path/to/split_file_75_25.txt"
split_path = "/media1/data/andy/SHOT72/split_files_eval/split_file_75_25.txt"
use_i3d = False
gpu = 1
# model_list = ["transformer","asformer"]
model_list = ["motionbert"]
# continue_from_epoch = 100  # int or None
continue_from_epoch = None  # int or None
epochs = 100
use_motion_bert = True

with open("parameters.p", "rb") as f:
    param = pickle.load(f)

run_main(
    param=param,
    test=test,
    hs=hs,
    epochs=epochs,
    use_i3d=use_i3d,
    gpu=gpu,
    experiment_suffix="_75_25",
    model_list=model_list,
    part_method=part_method,
    split_path=split_path,
    continue_from_epoch=continue_from_epoch,  # int or None
    use_motion_bert=use_motion_bert,
)
