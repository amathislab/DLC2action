#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import pickle
import numpy as np
from dlc2action.project.project import Project
from dump_param import dump_param
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

DATA_PATH = "path/to/converted_data"
DATA_TYPE = "dlc_track"
ANNOTATION_TYPE = "dlc"
PROJECTS_PATH = "path/to/projects"
SAVE_PATH = "./"

def run_main(
    param,
    part_method="time:strict",
    split_path=None,
    use_i3d=False,
    gpu=0,
    experiment_suffix="",
    model_list=["transformer", "ms_tcn3", "mlp"],
    test_ratio=0.7,
    continue_from_epoch = None,
):


    metrics = [
    "f1",
    "precision",
    ]

    print("Creating project")
    i3d_ext = ""
    if use_i3d:
        i3d_ext = "_i3d"

    project_name = f"hBABEL_d2a" + i3d_ext

    project = Project(
        project_name,
        data_path=None,
        annotation_path=None,
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

    # Add i3d features
    if use_i3d:
        project.update_parameters({"data": {"feature_suffix": "_view3_0.npy"}})

    # Update gpu device
    if part_method != "file":
        project.update_parameters(
            {"training": {"device": f"cuda:{gpu}", "val_frac": test_ratio}}
        )
    else:
        project.update_parameters({"training": {"device": f"cuda:{gpu}"}})

    ext = "_" + part_method
    for model in model_list:
        episode_name = f"{model}_best{ext}{i3d_ext}{experiment_suffix}"
        if continue_from_epoch is not None:
            episode_name = f"{model}_best_cont_{continue_from_epoch}{ext}{i3d_ext}{experiment_suffix}::0"

        results = project.evaluate(
            episode_names=[episode_name],
            data_path=DATA_PATH,
            augment_n=0,
            annotation_type=ANNOTATION_TYPE,
            parameters_update={"general": {"metric_functions": metrics}, "metrics": {"f1": {"average" : "micro"}, "precision": {"average" : "micro"}}},
        )
        with open(os.path.join(SAVE_PATH, "summary_hBABEL_eval.pickle"), "wb") as f:
            pickle.dump(results, f)


# Update parameters, all parameters are stored in the dump_param.py function
dump_param(hierarchy="frame", top=60)
part_method = "file"
split_path = "path/to/split_file.txt"
use_i3d = False
gpu = 0
model_list = ["ms_tcn3", "edtcn", "c2f_tcn", "c2f_transformer", "mlp"]
continue_from_epoch = 100  # int or None

with open("parameters.p", "rb") as f:
    param = pickle.load(f)

experiment_suffixes = ["_first_test"]
for experiment_suffix in experiment_suffixes:
    run_main(
        param=param,
        use_i3d=use_i3d,
        gpu=gpu,
        experiment_suffix=experiment_suffix,
        model_list=model_list,
        part_method=part_method,
        split_path=split_path,
        continue_from_epoch=continue_from_epoch,
    )
