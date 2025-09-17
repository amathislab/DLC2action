#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os

import numpy as np
from dlc2action.project import Project
from utils.data_load import Dataset
from utils.utils import *

PROJECTS_PATH = "path/to/projects"
DATA_TYPE = "dlc_track"


def plot_curves(
    path_to_plots,
    game_names,
    episode_names,
    metrics=["f1", "precision", "loss", "recall"],
    training=True,
    confusion=True,
    use_i3d=False,
    random=False,
):
    """Plot training curves and confusion matrices from episode names to path_to_plots according to the chosen metrics"""

    for game in game_names:

        print("Create figures for ", game)

        i3d_ext = ""
        if use_i3d:
            i3d_ext = "_i3d"

        time_ext = ""
        if not random:
            time_ext = "_timestrict"

        project_name = game + "_d2a_project_random_multidim_50" + i3d_ext
        project_path = os.path.join(PROJECTS_PATH, "Atari-Head-D2A-projects")
        project = Project(project_name, data_type=DATA_TYPE, projects_path=project_path)

        # Save training curves
        if training:
            filename_base = os.path.join(
                path_to_plots,
                game,
                "training_curve" + i3d_ext,
                "training_curve" + time_ext + i3d_ext,
            )
            os.makedirs(os.path.dirname(filename_base), exist_ok=True)
            for metric in metrics:
                try:
                    for episode_name in episode_names:
                        print(episode_name)
                        project.plot_episodes(
                            [episode_name],
                            metrics=[metric],
                            modes=["val", "train"],
                            save_path="_".join(
                                [
                                    filename_base,
                                    episode_name,
                                    "multidim_50",
                                    f"{metric}.png",
                                ]
                            ),
                        )

                    project.plot_episodes(
                        episode_names,
                        metrics=[metric],
                        modes=["val", "train"],
                        save_path="_".join(
                            [
                                filename_base,
                                "all_models",
                                "multidim_50",
                                f"{metric}.png",
                            ]
                        ),
                    )
                except:
                    print("Failed for training curves in ", game, " ", metric)
        # Save confusion matrices
        if confusion:
            # try:
            model_list_trial = list(np.repeat(episode_names, 3))
            model_list_trial_ind = [0, 1, 2] * 3
            episode_name_trial = [
                elem + f"::{i}"
                for elem, i in zip(model_list_trial, model_list_trial_ind)
            ]

            filename_base = os.path.join(
                path_to_plots,
                game,
                "confusion_matrix" + i3d_ext,
                "confusion_matrix" + time_ext + i3d_ext,
            )
            os.makedirs(os.path.dirname(filename_base), exist_ok=True)
            for metric in metrics:
                if metric != "loss":
                    for episode_name in episode_name_trial:
                        confusion_matrix, classes = project.plot_confusion_matrix(
                            episode_name=episode_name,
                            type=metric,
                            mode="val",
                            save_path="_".join(
                                [
                                    filename_base,
                                    episode_name,
                                    "multidim_50",
                                    f"{metric}.png",
                                ]
                            ),
                        )
                        output_path = "_".join(
                            [filename_base, episode_name, f"{metric}_confusion.pickle"]
                        )
                        save_confusion(confusion_matrix, classes, output_path)
            # except:
            # print("Failed for confusion matrix for ", game)


dataset_template = Dataset()
random = True
game_names = dataset_template.game_names
model_list = ["mlp", "ms_tcn3", "c2f_tcn", "transformer"]
ext = ""
if not random:
    ext = "_timestrict"
metrics = ["f1", "precision", "loss"]
episode_names = [elem + "_best" + ext for elem in model_list]
path_to_plots = "../../plots"
plot_curves(
    path_to_plots,
    game_names,
    episode_names,
    metrics=metrics,
    training=True,
    confusion=True,
    use_i3d=False,
    random=random,
)
