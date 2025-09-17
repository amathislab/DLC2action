#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from tqdm import tqdm
from utils.data_load import Dataset
from utils.utils import *


def plot_ethograms(dict_subject, num_plots=5, window_size=1000, cmap_label="Spectral"):
    """Plot ethogram from dataset"""
    for _ in range(num_plots):
        random_window = np.random.uniform(0, len(dict_subject["action"]))
        random_window = (random_window, random_window + window_size)
        action_list = np.unique(dict_subject["action"])
        action_list = action_list[~np.isnan(action_list)]
        fig, ax = plt.subplots(figsize=(30, 5))
        height = 0
        cmap = get_cmap(cmap_label)
        color_list = [cmap(elem / np.max(action_list)) for elem in action_list]
        for i, action in enumerate(action_list):
            action_ind = np.where(dict_subject["action"] == action)[0]
            st_st = get_start_stop(action_ind)

            res_indices_start = []
            res_indices_len = []

            for elem in st_st:
                res_indices_start.append(elem[0])
                res_indices_len.append(elem[1] - elem[0])

            ax.broken_barh(
                list(zip(res_indices_start, res_indices_len)),
                (height, 1),
                label=str(action),
                facecolors=color_list[i],
            )
        plt.legend()
        plt.xlim(random_window)
        plt.yticks([])
        plt.xlabel("Frame number")
        plt.show()


def plot_action_hist(dict_subject, cmap_label="Spectral", vis=True):
    """Plot histogram of actions for a given subject"""
    cmap = get_cmap(cmap_label)
    action_list = np.unique(dict_subject["action"])
    action_list = action_list[~np.isnan(action_list)]
    color_list = [cmap(elem / np.max(action_list)) for elem in action_list]
    results = []
    plt.figure(figsize=(20, 5))
    for i, action in enumerate(action_list):
        action_data = np.array(dict_subject["action"])[dict_subject["action"] == action]
        n = plt.hist(action_data, bins=1, color=color_list[i], label=str(action))
        results.append(n)

    if vis:
        plt.legend()
        plt.xticks(action_list)
        plt.ylabel("Number of frames")
        plt.xlabel("action index")
        plt.title(dict_subject["subject_name"])
        plt.show()
    else:
        plt.close()

    return results


def plot_random_gaze(save=None, games=None):

    img_size = (160, 210)
    dataset_template = Dataset()
    if save is not None:
        output_folder = os.path.join(save, "gaze_on_frame")
        os.makedirs(output_folder, exist_ok=True)

    if games is None:
        game_list = dataset.template.game_names
    else:
        game_list = games
    for game_name in game_list:
        dataset = Dataset(game_name)
        dict_subject = dataset.read_subject_info(dataset.subject_list[0])
        rand_pick = int(np.random.uniform(0, len(dict_subject["gaze_positions"])))
        frame_gaze = dict_subject["gaze_positions"][rand_pick]
        frame_id = dict_subject["frame_id"][rand_pick]
        img = dataset.fetch_frame(dict_subject["subject_name"], frame_id)
        if save is None:
            print("action: ", dict_subject["action"][rand_pick])
        if len(frame_gaze) > 0:
            f = plt.figure(figsize=(img_size[0] / 40, img_size[1] / 40))
            plt.imshow(img)
            if len(frame_gaze) > 2:
                frame_gaze_2d = np.reshape(frame_gaze, (int(len(frame_gaze) / 2), 2))
            colors = [
                (0.5, 0.3, elem / frame_gaze_2d.shape[0])
                for elem in np.arange(frame_gaze_2d.shape[0])
            ]
            plt.plot(frame_gaze_2d[:, 0], frame_gaze_2d[:, 1], c=[0.8, 0.8, 0.8])
            for i in range(frame_gaze_2d.shape[0]):
                plt.plot(frame_gaze_2d[i, 0], frame_gaze_2d[i, 1], "o", c=colors[i])
            plt.xlim((0, img_size[0]))
            plt.ylim((0, img_size[1]))
            plt.xticks([])
            plt.yticks([])
            plt.title("Gaze on frame: " + game_name)
            plt.gca().invert_yaxis()
            if save is not None:
                plt.savefig(os.path.join(output_folder, "Gaze_on_frame_" + game_name))
                plt.close()
            else:
                plt.show()


def plot_averaged_saccades(save=None, window=None, games=None, alpha=0.1):
    dataset_template = Dataset()
    img_size = (160, 210)

    if save is not None:
        output_folder = os.path.join(save, "average_saccades")
        os.makedirs(output_folder, exist_ok=True)

    if games is None:
        game_list = dataset.template.game_names
    else:
        game_list = games

    for game_name in game_list:
        dataset = Dataset(game_name)
        # Pick random subject
        pick_rand_subj = np.random.randint(0, len(dataset.subject_list))

        # Read converted poses
        path_to_poses = os.path.join(
            dataset_template.converted_data_path,
            game_name,
            dataset.subject_list[pick_rand_subj] + "_averaged_saccades.h5",
        )
        df = pd.read_hdf(path_to_poses, key="tracks")
        df_x = df["andy"]["individual0"]["gaze1"]["x"]
        df_y = df["andy"]["individual0"]["gaze1"]["y"]

        if window is None:
            frame_window = np.arange(len(df_x))
        else:
            frame_window = np.arange(window[0], window[1], 1)

        # Get random frame from the game
        dict_subject = dataset.read_subject_info(dataset.subject_list[pick_rand_subj])
        # rand_pick = int(np.random.uniform(frame_window[0], frame_window[-1]))
        rand_pick = int(frame_window[int(len(frame_window) / 2)])
        frame_id = dict_subject["frame_id"][rand_pick]
        img = dataset.fetch_frame(dict_subject["subject_name"], frame_id)
        plt.imshow(img)

        colors = [
            (0.5, 0.3, elem / len(frame_window), alpha)
            for elem in np.arange(len(frame_window))
        ]
        for k, i in tqdm(enumerate(frame_window), total=len(frame_window)):
            plt.plot(df_x[i], df_y[i], "o", c=colors[k])
        plt.xlim((0, img_size[0]))
        plt.ylim((0, img_size[1]))
        plt.title(
            "Averaged saccades for "
            + game_name
            + "_"
            + dataset.subject_list[pick_rand_subj]
        )
        plt.xticks([])
        plt.yticks([])
        plt.gca().invert_yaxis()

        if save is not None:
            plt.savefig(
                os.path.join(
                    output_folder,
                    "averaged_saccades"
                    + "_"
                    + game_name
                    + "_"
                    + dataset.subject_list[pick_rand_subj],
                )
            )
            plt.close()
        else:
            plt.show()


def plot_action_vs_gaze(
    dict_subject,
    dataset,
    game_name,
    window_size=1000,
    save=False,
    num_plots=1,
    cmap_label="Spectral",
):
    """Plot ethogram and gaze values in cartesian and polar from dataset"""

    for _ in range(num_plots):
        random_window = np.random.uniform(0, len(dict_subject["action"]))
        random_window = (random_window, random_window + window_size)
        action_list = np.unique(dict_subject["action"])
        action_list = action_list[~np.isnan(action_list)]
        fig, axs = plt.subplots(5, figsize=(40, 20))
        height = 0
        cmap = get_cmap(cmap_label)
        color_list = [cmap(elem / np.max(action_list)) for elem in action_list]
        for i, action in enumerate(action_list):
            action_ind = np.where(dict_subject["action"] == action)[0]
            st_st = get_start_stop(action_ind)

            res_indices_start = []
            res_indices_len = []

            for elem in st_st:
                res_indices_start.append(elem[0])
                res_indices_len.append(elem[1] - elem[0])

            axs[0].broken_barh(
                list(zip(res_indices_start, res_indices_len)),
                (height, 1),
                label=str(action),
                facecolors=color_list[i],
            )

        # Read converted poses
        path_to_poses = os.path.join(
            dataset.converted_data_path,
            game_name,
            dataset.subject_list[0] + "_averaged_saccades.h5",
        )
        df = pd.read_hdf(path_to_poses, key="tracks")
        df_x = df["andy"]["individual0"]["gaze"]["x"]
        df_y = df["andy"]["individual0"]["gaze"]["y"]

        # Convert to polar coordinates
        df_r, df_theta = polar(df_x.to_numpy(), df_y.to_numpy(), dataset.img_size)

        labels = ["x", "y", "r", "theta"]
        df_list = [df_x, df_y, df_r, df_theta]
        plot_colors = [[1, 0.2, i / len(labels)] for i in range(len(labels))]
        for i in range(len(axs)):
            axs[i].set_xlim(random_window)
            if i:
                axs[i].plot(df_list[i - 1], label=labels[i - 1], c=plot_colors[i - 1])
                axs[i].set_ylabel(labels[i - 1])
            else:
                axs[i].set_yticks([])
                axs[i].legend()
                axs[i].set_title(
                    "Comparison between actions and gazes for " + game_name
                )

        # plt.legend()
        plt.xlim(random_window)
        plt.xlabel("Frame number")

        if save is not None:
            plt.savefig(save)
            plt.close()
        else:
            plt.show()
