#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import pickle

import pandas as pd
from dlc2action.project.project import Project
from dump_param import dump_param

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


DATA_PATH = "path/to/converted_data"
DATA_TYPE = "dlc_track"
ANNOTATION_TYPE = "dlc"
PROJECTS_PATH = "path/to/projects"
SAVE_PATH = "./results/"


def run_main(
    param,
    part_method="time:strict",
    split_path=None,
    gpu=0,
    experiment_suffix="",
    model_list=["transformer", "ms_tcn3", "mlp"],
    test_ratio=0.7,
    continue_from_epoch=None,
):

    metrics = [
        "f1",
        "segmental_f1",
        "precision",
        "accuracy",
    ]

    print("Creating project")

    project_name = "SHOT72_d2a"

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

    project.update_parameters(
        {
            "metrics": {
                "f1": {
                    "average": "micro",  # ['macro', 'micro', 'none']; averaging method for classes
                    "ignored_classes": "None",  # set; a set of class ids to ignore in calculation
                    "threshold_value": 0.5,  # float; the probability threshold for positive samples
                },
                "precision": {
                    "average": "micro",  # ['macro', 'micro', 'none']; averaging method for classes
                    "ignored_classes": "None",  # set; a set of class ids to ignore in calculation
                    "threshold_value": 0.5,  # float; the probability threshold for positive samples
                },
            },
        }
    )

    # Update partition method
    if part_method == "file":
        assert split_path is not None
        project.update_parameters(
            {
                "training": {
                    "partition_method": part_method,
                    "split_path": split_path,
                    "device": f"cuda:{gpu}",
                    "val_frac": test_ratio,
                },
                "model": {
                    "num_joints": 26,
                    "dim_feat": 256,
                    "dim_rep": 256,
                    "depth": 4,
                    "num_heads": 4,
                },
            }
        )
    else:
        project.update_parameters(
            {
                "training": {
                    "partition_method": part_method,
                    "device": f"cuda:{gpu}",
                    "val_frac": test_ratio,
                },
            }
        )

    ext = "_" + part_method
    df_save = {}
    os.makedirs(SAVE_PATH, exist_ok=True)
    for model in model_list:
        for i in range(2):
            episode_names = [f"{model}_best{ext}{experiment_suffix}"]
            if continue_from_epoch is not None:
                episode_names = [
                    f"{model}_best_cont_{continue_from_epoch}{ext}{experiment_suffix}::{i}"
                ]
            if model == "motionbert":
                episode_names = [f"{model}_best{ext}{experiment_suffix}#{i}"]
            results = project.evaluate(
                episode_names=episode_names,
                data_path=DATA_PATH,
                augment_n=0,
                annotation_type=ANNOTATION_TYPE,
                parameters_update={
                    "general": {"metric_functions": metrics},
                    "metrics": {
                        "f1": {"average": "micro"},
                        "precision": {"average": "micro"},
                    },
                    "training": {"val_frac": test_ratio},
                },
            )
            results = [r.tolist() for r in results["f1"]]
            df_save[episode_names[0]] = results
        df_save = pd.DataFrame(df_save)
        df_save.to_csv(os.path.join(SAVE_PATH, f"summary_SHOT7M2_{model}_eval.csv"))


# Update parameters, all parameters are stored in the dump_param.py function
dump_param()
part_method = "file"
split_path = "path/to/split_file_75_25.txt"
gpu = 0
# model_list = ["edtcn", "c2f_tcn", "c2f_transformer", "ms_tcn3", "mlp"]
model_list = ["motionbert"]
continue_from_epoch = None  # int or None

with open("parameters.p", "rb") as f:
    param = pickle.load(f)

run_main(
    param=param,
    gpu=gpu,
    experiment_suffix="_75_25",
    model_list=model_list,
    part_method=part_method,
    split_path=split_path,
    continue_from_epoch=continue_from_epoch,
)
