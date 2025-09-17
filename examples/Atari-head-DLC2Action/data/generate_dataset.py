#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#

# Common librairies
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

# Personlalized libraries
from utils.data_load import Dataset
from utils.utils import *


def gazes2DLC(subject_dict, output_folder, average_case="average", num_timepoints=None):
    """Convert gazes to DLC track format by averaging the saccades in each frame"""

    if average_case == "body_parts":
        if num_timepoints is None:
            num_timepoints = 50

        frame_gaze = []
        for gaze_poses in subject_dict["gaze_positions"]:
            if len(gaze_poses) > 5:
                gaze_poses2D = np.reshape(gaze_poses, (int(len(gaze_poses) / 2), 2))
                new_gaze_poses2D = np.empty((num_timepoints, 2))
                x = np.arange(gaze_poses2D.shape[0])
                fx = interpolate.interp1d(x, gaze_poses2D[:, 0])
                fy = interpolate.interp1d(x, gaze_poses2D[:, 1])
                x_vals = np.linspace(0, gaze_poses2D.shape[0] - 1, num_timepoints)
                new_gaze_poses2D[:, 0] = fx(x_vals)
                new_gaze_poses2D[:, 1] = fy(x_vals)

                frame_gaze.append(new_gaze_poses2D)
            else:
                frame_gaze.append(np.array([[np.nan, np.nan]] * num_timepoints))
        frame_gaze = np.array(frame_gaze)
        frame_gaze = np.concatenate(
            (frame_gaze, np.ones((frame_gaze.shape[0], num_timepoints, 1))), axis=-1
        )
        frame_gaze = np.reshape(frame_gaze, (frame_gaze.shape[0], -1))

        columnindex = pd.MultiIndex.from_product(
            [
                ["andy"],
                ["individual0"],
                [str(i) for i in range(num_timepoints)],
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    else:
        if average_case == "average":
            frame_gaze = []
            for gaze_poses in subject_dict["gaze_positions"]:
                if len(gaze_poses) > 1:
                    gaze_poses2D = np.reshape(gaze_poses, (int(len(gaze_poses) / 2), 2))
                    frame_gaze.append(np.mean(gaze_poses2D, axis=0))
                else:
                    frame_gaze.append(np.array([np.nan] * 2))
            frame_gaze = np.array(frame_gaze)

        elif average_case == "upsampled":
            frame_gaze = np.empty((1, 2))
            gaze_frame_map = []
            for gaze_poses in tqdm(subject_dict["gaze_positions"]):
                if len(gaze_poses) > 1:
                    gaze_poses2D = np.reshape(gaze_poses, (int(len(gaze_poses) / 2), 2))
                    gaze_frame_map.append(gaze_poses2D.shape[0])
                    frame_gaze = np.concatenate([frame_gaze, gaze_poses2D], axis=0)
                else:
                    gaze_frame_map.append(1)
                    frame_gaze = np.concatenate(
                        [frame_gaze, np.expand_dims(np.array([np.nan] * 2), axis=0)],
                        axis=0,
                    )
            frame_gaze = frame_gaze[1:]
            write_frame_map(
                gaze_frame_map,
                os.path.join(
                    output_folder,
                    subject_dict["subject_name"] + "_gaze_frame_map.pickle",
                ),
            )

        frame_gaze = np.concatenate(
            (frame_gaze, np.ones((frame_gaze.shape[0], 1))), axis=1
        )
        frame_gaze = np.concatenate(
            (frame_gaze, frame_gaze), axis=1
        )  # Double the gaze to make 2 points (extact feature problem tries to get distances between keypoints)

        columnindex = pd.MultiIndex.from_product(
            [["andy"], ["individual0"], ["gaze1", "gaze2"], ["x", "y", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    df_list = pd.DataFrame(frame_gaze, columns=columnindex)
    output_filename = os.path.join(
        output_folder, subject_dict["subject_name"] + "_averaged_saccades.h5"
    )
    df_list.to_hdf(output_filename, key="tracks", format="table", mode="w")


def action2beh(subject_dict, output_folder, average_case="average"):
    """Convert game actions to DLC2Action compatible behavior files"""

    action_list = np.unique(subject_dict["action"])
    behaviors = [str(elem) for elem in action_list if not np.isnan(elem)]
    video_file = subject_dict["subject_name"] + ".mp4"
    infos = {
        "datetime": str(datetime.now()),
        "annotator": "andy",
        "video_file": video_file,
    }
    animals = ["individual0"]

    beh_data = []

    if average_case == "upsampled":
        frame_map = read_frame_map(
            os.path.join(
                output_folder, subject_dict["subject_name"] + "_gaze_frame_map.pickle"
            )
        )
        cum_frame_map = np.insert(np.cumsum(frame_map), 0, 0)[:-1]

    for action in action_list:
        if not np.isnan(action):
            action_ind = np.where(subject_dict["action"] == action)[0]
            st_st = get_start_stop(action_ind)
            if average_case == "upsampled":
                st_st = remap_st_st(cum_frame_map, st_st)
            beh_data.append(np.array(list(map(add_confusion, st_st))))

    beh_data = [beh_data]
    formated_data = (infos, behaviors, animals, beh_data)
    filename = os.path.join(
        output_folder, subject_dict["subject_name"] + "_action_behaviors.pickle"
    )
    pickle.dump(formated_data, open(os.path.join(output_folder, filename), "wb"))


if __name__ == "__main__":
    convert_poses = True
    convert_behaviors = True
    average_case = "body_parts"
    num_timepoints = 10

    dataset_general = Dataset()
    for game in dataset_general.game_names:
        dataset = Dataset(game)
        print("Convert data for ", game)
        if average_case == "average":
            output_folder_game = os.path.join(dataset_general.converted_data_path, game)
        elif average_case == "upsampled":
            output_folder_game = os.path.join(
                dataset_general.converted_data_path + "-upsampled", game
            )
        if average_case == "body_parts":
            output_folder_game = os.path.join(
                dataset_general.converted_data_path + f"-multidim-{num_timepoints}",
                game,
            )

        os.makedirs(output_folder_game, exist_ok=True)
        for subject in tqdm(dataset.subject_list):
            subject_dict = dataset.read_subject_info(subject)
            if convert_poses:
                gazes2DLC(
                    subject_dict,
                    output_folder_game,
                    average_case=average_case,
                    num_timepoints=num_timepoints,
                )
            if convert_behaviors:
                action2beh(subject_dict, output_folder_game, average_case=average_case)
